#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_http_client.h"
#include "esp_websocket_client.h"
#include "bsp_board.h"
#include "rgb_led_driver.h"
#include "wifi_connect.h"

static const char *TAG = "GEMINI_ASSISTANT";

// ==========================================================
// CONFIGURATION
// ==========================================================
#define SERVER_IP   "192.168.50.53"  // Laptop IP address on WiFi network
#define SERVER_PORT "8008"           // Relay server port
#define SESSION_ID  "673334d939a1270181963600"

#define SAMPLE_RATE 16000
#define CHANNELS    2                // ES7210 hardware input channels
#define CHUNK_SAMPLES 512            // 512 samples @ 16kHz = 32ms per chunk
#define CHUNK_MONO_BYTES (CHUNK_SAMPLES * sizeof(int16_t)) // 1024 bytes

#define VAD_WAKE_THRESHOLD_RMS  150.0f // Standby threshold to trigger session (calibrated for 24dB mic)
#define VAD_ACTIVE_THRESHOLD_RMS 90.0f  // Threshold to keep streaming speech
#define MIN_SPEECH_DURATION_MS  300   // Minimum speech required to trigger Gemini turn
#define SILENCE_TIMEOUT_MS      650   // 650ms silence ends utterance -> instant Gemini response
#define IDLE_STANDBY_TIMEOUT_S  25    // 25s inactivity returns to Standby

// ==========================================================
// CONVERSATIONAL STATE MACHINE
// ==========================================================
typedef enum {
    CONV_STATE_STANDBY = 0,     // Standby: Soft breathing blue. Listening for wake word.
    CONV_STATE_LISTENING,       // Active listening: Bright Cyan. Streaming live PCM to Gemini.
    CONV_STATE_THINKING,        // Thinking: Amber Yellow. Utterance ended, waiting for response.
    CONV_STATE_SPEAKING         // Speaking: Vibrant Green. Playing Gemini's audio response.
} conv_state_t;

static volatile conv_state_t s_conv_state = CONV_STATE_STANDBY;
static int64_t s_last_speech_time_ms = 0;
static volatile int64_t s_playback_ended_time_ms = 0;

static float compute_pcm_rms(const int16_t *pcm_samples, int num_samples) {
    if (num_samples <= 0) return 0.0f;
    double sum_sq = 0.0;
    for (int i = 0; i < num_samples; i++) {
        sum_sq += (double)pcm_samples[i] * (double)pcm_samples[i];
    }
    return (float)sqrt(sum_sq / num_samples);
}

typedef struct {
    esp_codec_dev_handle_t play_dev;
    int total_audio_read;
    bool is_playing;
} ws_playback_ctx_t;

static esp_websocket_client_handle_t s_persistent_ws_client = NULL;
static bool s_ws_connected = false;
static ws_playback_ctx_t s_playback_ctx;
static RingbufHandle_t s_audio_play_rb = NULL;
static volatile bool s_turn_complete = true;
static volatile size_t s_buffered_bytes = 0;
static volatile bool s_is_prebuffering = true;

#define PLAYBACK_PREBUFFER_BYTES  9600 // ~300ms of 16kHz 16-bit audio cushion (32,000 bytes/sec)

static void flush_playback_ringbuffer(void) {
    s_turn_complete = true;
    s_is_prebuffering = true;
    s_buffered_bytes = 0;
    s_playback_ended_time_ms = esp_timer_get_time() / 1000;
    if (s_audio_play_rb) {
        size_t dummy_size = 0;
        uint8_t *dummy = NULL;
        while ((dummy = (uint8_t *)xRingbufferReceive(s_audio_play_rb, &dummy_size, 0)) != NULL) {
            vRingbufferReturnItem(s_audio_play_rb, (void *)dummy);
        }
    }
}

// Set RGB LED based on current state
static void update_led_state(conv_state_t state) {
    switch (state) {
        case CONV_STATE_STANDBY:
            rgb_led_set_all(0, 30, 80);   // Soft Dim Blue (Standby / Listening for Wake Word)
            break;
        case CONV_STATE_LISTENING:
            rgb_led_set_all(0, 150, 255); // Bright Cyan (Active Conversation / User Speaking)
            break;
        case CONV_STATE_THINKING:
            rgb_led_set_all(220, 140, 0); // Amber Yellow (Processing / Generating Answer)
            break;
        case CONV_STATE_SPEAKING:
            rgb_led_set_all(0, 220, 30);  // Vibrant Green (Assistant Speaking)
            break;
    }
}

static void audio_playback_task(void *pvParameters) {
    uint8_t zero_silence[1024] = {0};
    int empty_stall_ms = 0;

    while (1) {
        // Pre-buffering phase to absorb Wi-Fi network jitter and eliminate micro-stutters
        if (s_is_prebuffering) {
            if (s_buffered_bytes >= PLAYBACK_PREBUFFER_BYTES || (s_turn_complete && s_buffered_bytes > 0)) {
                s_is_prebuffering = false;
            } else {
                vTaskDelay(pdMS_TO_TICKS(15));
                continue;
            }
        }

        size_t item_size = 0;
        uint8_t *item = (uint8_t *)xRingbufferReceive(s_audio_play_rb, &item_size, pdMS_TO_TICKS(100));

        if (item != NULL && item_size > 0) {
            // Align to 16-bit sample boundary
            item_size &= ~1;

            if (item_size > 0) {
                if (s_buffered_bytes >= item_size) {
                    s_buffered_bytes -= item_size;
                } else {
                    s_buffered_bytes = 0;
                }

                empty_stall_ms = 0;
                s_playback_ctx.is_playing = true;
                if (s_conv_state != CONV_STATE_SPEAKING) {
                    s_conv_state = CONV_STATE_SPEAKING;
                    update_led_state(CONV_STATE_SPEAKING);
                }

                if (s_playback_ctx.play_dev) {
                    esp_codec_dev_write(s_playback_ctx.play_dev, (void *)item, item_size);
                }

                s_playback_ctx.total_audio_read += item_size;
            }
            vRingbufferReturnItem(s_audio_play_rb, (void *)item);
        } else {
            // Ringbuffer is momentarily empty
            if (s_playback_ctx.is_playing) {
                empty_stall_ms += 100;

                if (s_turn_complete || empty_stall_ms >= 800) {
                    if (s_playback_ctx.play_dev) {
                        esp_codec_dev_write(s_playback_ctx.play_dev, (void *)zero_silence, sizeof(zero_silence));
                    }
                    s_playback_ctx.is_playing = false;
                    s_turn_complete = true;
                    s_is_prebuffering = true;
                    s_buffered_bytes = 0;
                    empty_stall_ms = 0;
                    s_playback_ended_time_ms = esp_timer_get_time() / 1000;

                    // Switch back to active listening
                    if (s_conv_state == CONV_STATE_SPEAKING) {
                        s_conv_state = CONV_STATE_LISTENING;
                        s_last_speech_time_ms = esp_timer_get_time() / 1000;
                        update_led_state(CONV_STATE_LISTENING);
                        ESP_LOGI(TAG, "🗣️ Assistant finished speaking. Ready for user's next question!");
                    }
                }
            } else {
                s_is_prebuffering = true;
                s_buffered_bytes = 0;
            }
        }
    }
}

static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;

    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            s_ws_connected = true;
            ESP_LOGI(TAG, "⚡ Persistent WebSocket Connected to Relay Server!");
            break;
        case WEBSOCKET_EVENT_DISCONNECTED:
            s_ws_connected = false;
            ESP_LOGI(TAG, "⚡ Persistent WebSocket Disconnected");
            break;
        case WEBSOCKET_EVENT_DATA:
            if (data->op_code == 0x01) { // Text JSON frame
                ESP_LOGI(TAG, "📩 WS Metadata: %.*s", data->data_len, data->data_ptr);
                if (strstr(data->data_ptr, "turn_complete") != NULL) {
                    s_turn_complete = true;
                }
            } else if (data->op_code == 0x02 || data->op_code == 0x00) { // Binary PCM audio frame
                if (data->data_len > 0 && s_audio_play_rb) {
                    s_turn_complete = false;
                    BaseType_t res = xRingbufferSend(s_audio_play_rb, data->data_ptr, data->data_len, pdMS_TO_TICKS(2500));
                    if (res == pdTRUE) {
                        s_buffered_bytes += data->data_len;
                    } else {
                        ESP_LOGW(TAG, "Audio play ring buffer full, dropped chunk (%d bytes)", data->data_len);
                    }
                }
            }
            break;
    }
}

static esp_err_t ensure_websocket_connected(ws_playback_ctx_t *ctx) {
    if (s_persistent_ws_client != NULL && s_ws_connected && esp_websocket_client_is_connected(s_persistent_ws_client)) {
        return ESP_OK;
    }

    char ws_url[160];
    snprintf(ws_url, sizeof(ws_url), "ws://%s:%s/ws/live/%s", SERVER_IP, SERVER_PORT, SESSION_ID);
    ESP_LOGI(TAG, "Establishing persistent Live WebSocket connection to: %s", ws_url);

    esp_websocket_client_config_t ws_cfg = {
        .uri = ws_url,
        .buffer_size = 8192,
        .reconnect_timeout_ms = 2000,
        .network_timeout_ms = 15000,
        .pingpong_timeout_sec = 0,
    };

    if (s_persistent_ws_client != NULL) {
        s_ws_connected = false;
        esp_websocket_client_stop(s_persistent_ws_client);
        esp_websocket_client_destroy(s_persistent_ws_client);
        s_persistent_ws_client = NULL;
    }

    s_persistent_ws_client = esp_websocket_client_init(&ws_cfg);
    if (!s_persistent_ws_client) {
        ESP_LOGE(TAG, "Failed to allocate websocket client!");
        return ESP_FAIL;
    }
    esp_websocket_register_events(s_persistent_ws_client, WEBSOCKET_EVENT_ANY, websocket_event_handler, (void *)ctx);

    esp_err_t err = esp_websocket_client_start(s_persistent_ws_client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start persistent WebSocket: %s", esp_err_to_name(err));
        return err;
    }

    int wait_count = 0;
    while (!s_ws_connected && wait_count++ < 40) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    if (!s_ws_connected) {
        ESP_LOGE(TAG, "Persistent WebSocket connection handshake timed out!");
        return ESP_FAIL;
    }

    return ESP_OK;
}

// Continuous Microphone Processing & Real-time Live Streaming Task
static void continuous_mic_stream_task(void *pvParameters)
{
    int chunk_samples = CHUNK_SAMPLES;
    int chunk_raw_bytes = chunk_samples * CHANNELS * sizeof(int32_t); // 4096 bytes
    int32_t *raw_chunk = (int32_t *)malloc(chunk_raw_bytes);
    if (!raw_chunk) {
        ESP_LOGE(TAG, "Failed to allocate raw chunk buffer!");
        vTaskDelete(NULL);
        return;
    }

    // Flush initial ADC queue
    for (int f = 0; f < 5; f++) {
        esp_get_feed_data(true, (int16_t *)raw_chunk, chunk_raw_bytes);
    }

    int16_t chunk_mono[CHUNK_SAMPLES];
    // Circular pre-roll buffer (2 chunks = 64ms cushion)
    int16_t preroll[CHUNK_SAMPLES * 2];
    int preroll_count = 0;

    int speech_accum_ms = 0;
    int silence_accum_ms = 0;
    int chunk_duration_ms = (CHUNK_SAMPLES * 1000) / SAMPLE_RATE; // ~32ms

    s_last_speech_time_ms = esp_timer_get_time() / 1000;
    update_led_state(CONV_STATE_STANDBY);

    ESP_LOGI(TAG, "🎙️ Continuous Mic Streaming Engine Started. Listening for Wake Word...");

    while (1) {
        esp_err_t ret = esp_get_feed_data(true, (int16_t *)raw_chunk, chunk_raw_bytes);
        if (ret != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        // Downmix ES7210 2-channel 32-bit to mono 16-bit
        for (int i = 0; i < chunk_samples; i++) {
            int16_t ch0 = (int16_t)(raw_chunk[CHANNELS * i + 0] >> 16);
            int16_t ch1 = (int16_t)(raw_chunk[CHANNELS * i + 1] >> 16);
            int16_t max_s = (abs(ch1) > abs(ch0)) ? ch1 : ch0;
            chunk_mono[i] = max_s;
        }

        float rms = compute_pcm_rms(chunk_mono, chunk_samples);
        int64_t now_ms = esp_timer_get_time() / 1000;

        // 1. If Assistant is currently playing response (or within 500ms cooldown), mute mic to eliminate acoustic feedback
        int64_t time_since_playback_ms = now_ms - s_playback_ended_time_ms;
        if (s_conv_state == CONV_STATE_SPEAKING || time_since_playback_ms < 500) {
            speech_accum_ms = 0;
            silence_accum_ms = 0;
            preroll_count = 0;
            continue;
        }

        // 2. State: STANDBY (Waiting for Wake Word / Speech)
        if (s_conv_state == CONV_STATE_STANDBY) {
            if (rms >= VAD_WAKE_THRESHOLD_RMS) {
                ESP_LOGI(TAG, "🗣️ Wake Word / Speech Detected! (RMS: %.1f). Activating Live Session...", rms);
                s_conv_state = CONV_STATE_LISTENING;
                update_led_state(CONV_STATE_LISTENING);
                s_last_speech_time_ms = now_ms;

                ensure_websocket_connected(&s_playback_ctx);
                if (s_ws_connected && s_persistent_ws_client && esp_websocket_client_is_connected(s_persistent_ws_client)) {
                    // Send pre-roll buffer to catch initial consonant
                    if (preroll_count > 0) {
                        esp_websocket_client_send_bin(s_persistent_ws_client, (const char *)preroll, preroll_count * sizeof(int16_t), pdMS_TO_TICKS(200));
                    }
                    // Send current chunk
                    esp_websocket_client_send_bin(s_persistent_ws_client, (const char *)chunk_mono, CHUNK_MONO_BYTES, pdMS_TO_TICKS(200));
                }

                speech_accum_ms = chunk_duration_ms * 2;
                silence_accum_ms = 0;
                preroll_count = 0;
            } else {
                // Update pre-roll buffer
                memcpy(preroll, preroll + chunk_samples, chunk_samples * sizeof(int16_t));
                memcpy(preroll + chunk_samples, chunk_mono, chunk_samples * sizeof(int16_t));
                preroll_count = chunk_samples * 2;
            }
            continue;
        }

        // 3. State: ACTIVE LISTENING (Streaming Live Audio Chunks to Gemini)
        if (s_conv_state == CONV_STATE_LISTENING) {
            if (rms >= VAD_ACTIVE_THRESHOLD_RMS) {
                // User is actively speaking: send frame immediately!
                s_last_speech_time_ms = now_ms;
                speech_accum_ms += chunk_duration_ms;
                silence_accum_ms = 0;

                if (s_ws_connected && s_persistent_ws_client && esp_websocket_client_is_connected(s_persistent_ws_client)) {
                    int send_ret = esp_websocket_client_send_bin(s_persistent_ws_client, (const char *)chunk_mono, CHUNK_MONO_BYTES, pdMS_TO_TICKS(200));
                    if (send_ret < 0) {
                        ESP_LOGW(TAG, "Failed to stream PCM frame to server!");
                    }
                }
            } else {
                // RMS below speech threshold
                if (speech_accum_ms > 0) {
                    // User was speaking, now paused
                    silence_accum_ms += chunk_duration_ms;

                    // Stream quiet transition frame
                    if (s_ws_connected && s_persistent_ws_client && esp_websocket_client_is_connected(s_persistent_ws_client)) {
                        esp_websocket_client_send_bin(s_persistent_ws_client, (const char *)chunk_mono, CHUNK_MONO_BYTES, pdMS_TO_TICKS(200));
                    }

                    // Check if utterance is complete (at least 350ms of speech and 600ms of silence)
                    if (speech_accum_ms >= MIN_SPEECH_DURATION_MS && silence_accum_ms >= SILENCE_TIMEOUT_MS) {
                        ESP_LOGI(TAG, "⏹️ Utterance complete (Speech: %d ms, Silence: %d ms). Triggering Gemini response...", 
                                 speech_accum_ms, silence_accum_ms);
                        
                        const char *eot_msg = "{\"event\":\"audio_end\"}";
                        if (s_ws_connected && s_persistent_ws_client && esp_websocket_client_is_connected(s_persistent_ws_client)) {
                            esp_websocket_client_send_text(s_persistent_ws_client, eot_msg, strlen(eot_msg), pdMS_TO_TICKS(500));
                        }

                        s_conv_state = CONV_STATE_THINKING;
                        update_led_state(CONV_STATE_THINKING);

                        speech_accum_ms = 0;
                        silence_accum_ms = 0;
                    }
                } else {
                    // Idle listening: check for session inactivity timeout (25 seconds)
                    if ((now_ms - s_last_speech_time_ms) > (IDLE_STANDBY_TIMEOUT_S * 1000)) {
                        ESP_LOGI(TAG, "⏳ Session idle timeout (%d seconds). Returning to Standby mode.", IDLE_STANDBY_TIMEOUT_S);
                        s_conv_state = CONV_STATE_STANDBY;
                        update_led_state(CONV_STATE_STANDBY);
                    }
                }
            }
            continue;
        }

        // 4. State: THINKING (Waiting for Gemini Live response audio)
        if (s_conv_state == CONV_STATE_THINKING) {
            // Drop mic frames while waiting for response audio
            speech_accum_ms = 0;
            silence_accum_ms = 0;
            continue;
        }
    }

    free(raw_chunk);
}

#define GPIO_BTN_BOOT     GPIO_NUM_0   // BOOT Button

static void gpio_button_task(void *pvParameters)
{
    // Enable pullups on button GPIO pins
    uint64_t pin_mask = (1ULL << GPIO_NUM_0)  | (1ULL << GPIO_NUM_1)  | (1ULL << GPIO_NUM_2)  |
                        (1ULL << GPIO_NUM_3)  | (1ULL << GPIO_NUM_4)  | (1ULL << GPIO_NUM_5)  |
                        (1ULL << GPIO_NUM_6)  | (1ULL << GPIO_NUM_7)  | (1ULL << GPIO_NUM_38) |
                        (1ULL << GPIO_NUM_39) | (1ULL << GPIO_NUM_40) | (1ULL << GPIO_NUM_41) |
                        (1ULL << GPIO_NUM_42);

    gpio_config_t io_conf = {
        .pin_bit_mask = pin_mask,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);

    bool prev_boot_state = 1;
    uint16_t prev_tca_val = 0xFFFF;

    while (1) {
        // Check BOOT Button (GPIO 0)
        bool boot_pressed = (gpio_get_level(GPIO_BTN_BOOT) == 0);
        if (boot_pressed && prev_boot_state == 1) {
            // Toggle Standby / Active Conversation
            if (s_conv_state == CONV_STATE_STANDBY) {
                ESP_LOGI(TAG, "🔘 BOOT Button Pressed -> Conversation ACTIVATED!");
                s_conv_state = CONV_STATE_LISTENING;
                s_last_speech_time_ms = esp_timer_get_time() / 1000;
                update_led_state(CONV_STATE_LISTENING);
            } else {
                ESP_LOGI(TAG, "🔘 BOOT Button Pressed -> Conversation STOPPED (Standby)");
                s_conv_state = CONV_STATE_STANDBY;
                flush_playback_ringbuffer();
                update_led_state(CONV_STATE_STANDBY);
            }
            vTaskDelay(pdMS_TO_TICKS(400));
        }
        prev_boot_state = !boot_pressed;

        // Read Waveshare TCA9555 I2C Expander
        uint16_t tca_val = bsp_board_read_tca9555_inputs();
        if (tca_val != prev_tca_val && tca_val != 0xFFFF) {
            uint8_t port1 = (uint8_t)(tca_val >> 8);
            uint8_t prev_port1 = (uint8_t)(prev_tca_val >> 8);

            // 1. Button 1 (P1_1): Volume UP (+15%)
            if ((port1 & 0x02) == 0 && (prev_port1 & 0x02) != 0) {
                int cur_vol = bsp_board_get_volume();
                int new_vol = cur_vol + 15;
                if (new_vol > 100) new_vol = 100;
                bsp_board_set_volume(new_vol);
                ESP_LOGI(TAG, "🔊 Volume UP: %d%%", new_vol);
                
                int num_leds = (new_vol * 7 + 50) / 100;
                if (num_leds < 1 && new_vol > 0) num_leds = 1;
                rgb_led_set_vu_meter(num_leds);
                vTaskDelay(pdMS_TO_TICKS(600));
                update_led_state(s_conv_state);
            }

            // 2. Button 2 (P1_2): Volume DOWN (-15%)
            if ((port1 & 0x04) == 0 && (prev_port1 & 0x04) != 0) {
                int cur_vol = bsp_board_get_volume();
                int new_vol = cur_vol - 15;
                if (new_vol < 0) new_vol = 0;
                bsp_board_set_volume(new_vol);
                ESP_LOGI(TAG, "🔉 Volume DOWN: %d%%", new_vol);

                int num_leds = (new_vol * 7 + 50) / 100;
                rgb_led_set_vu_meter(num_leds);
                vTaskDelay(pdMS_TO_TICKS(600));
                update_led_state(s_conv_state);
            }

            // 3. Button 3 (P1_3): Toggle Active / Standby (Manual Stop / Start)
            if ((port1 & 0x08) == 0 && (prev_port1 & 0x08) != 0) {
                if (s_conv_state == CONV_STATE_STANDBY) {
                    ESP_LOGI(TAG, "🔘 Waveshare Button 3 Pressed -> Conversation ACTIVATED!");
                    s_conv_state = CONV_STATE_LISTENING;
                    s_last_speech_time_ms = esp_timer_get_time() / 1000;
                    update_led_state(CONV_STATE_LISTENING);
                } else {
                    ESP_LOGI(TAG, "🔘 Waveshare Button 3 Pressed -> Conversation STOPPED (Standby)");
                    s_conv_state = CONV_STATE_STANDBY;
                    flush_playback_ringbuffer();
                    update_led_state(CONV_STATE_STANDBY);
                }
            }

            prev_tca_val = tca_val;
        }

        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, " ESP32-S3 Continuous Full-Duplex Voice Assistant");
    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, "🎛️ CONTROLS & WAKE WORDS:");
    ESP_LOGI(TAG, "   • Wake Words : 'Hello Assistant', 'Hey Assistant'");
    ESP_LOGI(TAG, "   • Stop Words : 'Stop', 'Goodbye', 'Bye'");
    ESP_LOGI(TAG, "   • BOOT Button: Manual Start / Stop Toggle");
    ESP_LOGI(TAG, "   • VOL+/-     : TCA9555 Buttons 1 & 2");
    ESP_LOGI(TAG, "------------------------------------------------");

    // 1. Initialize Peripherals & Codecs
    ESP_ERROR_CHECK(esp_board_init(SAMPLE_RATE, 1, 16));
    rgb_led_init();

    s_playback_ctx.play_dev = esp_ret_play_dev();
    s_playback_ctx.total_audio_read = 0;
    s_playback_ctx.is_playing = false;

    // 64KB Asynchronous Playback RingBuffer (2.0s audio cushion)
    s_audio_play_rb = xRingbufferCreate(65536, RINGBUF_TYPE_BYTEBUF);
    // Pin audio playback to dedicated Core 1 (Priority 10) isolated from Wi-Fi interrupts
    xTaskCreatePinnedToCore(audio_playback_task, "audio_play_task", 4096, NULL, 10, NULL, 1);

    // Pin button listener to Core 0 (Priority 4)
    xTaskCreatePinnedToCore(gpio_button_task, "gpio_button_task", 3072, NULL, 4, NULL, 0);

    // 2. Connect Wi-Fi (Runs on Core 0)
    rgb_led_set_all(0, 0, 255); // Blue = Connecting Wi-Fi
    ESP_LOGI(TAG, "Connecting to Wi-Fi...");
    if (wifi_init_sta() != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi connection failed! Please check SSID & Password.");
        rgb_led_set_all(255, 0, 0);
        return;
    }
    rgb_led_clear();

    // 3. Connect Persistent WebSocket right at boot
    ESP_LOGI(TAG, "Initializing persistent Live WebSocket connection at boot...");
    ensure_websocket_connected(&s_playback_ctx);

    ESP_LOGI(TAG, "✅ Free Heap Memory: %lu bytes", (unsigned long)esp_get_free_heap_size());
    ESP_LOGI(TAG, "System Ready! Continuous Voice Assistant active.");

    // 4. Launch Continuous Microphone Stream Task pinned to dedicated Core 1 (Priority 9)
    xTaskCreatePinnedToCore(continuous_mic_stream_task, "mic_stream_task", 5120, NULL, 9, NULL, 1);
}
