#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_websocket_client.h"
#include "bsp_board.h"
#include "rgb_led_driver.h"
#include "wifi_connect.h"

static const char *TAG = "GEMINI_ASSISTANT";

#include <math.h>

// ==========================================================
// CONFIGURATION
// ==========================================================
#define SERVER_IP   "192.168.50.53"  // Laptop IP address on WiFi network
#define SERVER_PORT "8008"           // Relay server port
#define SESSION_ID  "673334d939a1270181963600"

#define MAX_RECORD_TIME_SEC 5        // Maximum 5 seconds duration (fits internal DRAM)
#define MIN_SPEECH_MS       2000     // Record at least 2.0 seconds minimum
#define SILENCE_TIMEOUT_MS  2000     // 2.0s silence cutoff
#define VAD_THRESHOLD_RMS   25.0f    // Sensitive VAD threshold to catch all spoken questions

#define SAMPLE_RATE 16000
#define CHANNELS    4                // ES7210 raw input channels

// Mono PCM 16-bit 16kHz buffer size: 5 sec * 16000 * 2 bytes = 160,000 bytes (160 KB)
#define MONO_PCM_BUF_SIZE (MAX_RECORD_TIME_SEC * SAMPLE_RATE * sizeof(int16_t))
#define CHUNK_SAMPLES 512            // Read 512 samples per chunk from ADC

// WAV Header struct helper
typedef struct {
    char chunk_id[4];        // "RIFF"
    uint32_t chunk_size;     // File size - 8
    char format[4];          // "WAVE"
    char subchunk1_id[4];   // "fmt "
    uint32_t subchunk1_size;// 16 for PCM
    uint16_t audio_format;   // 1 for PCM
    uint16_t num_channels;   // 1 for Mono
    uint32_t sample_rate;    // 16000
    uint32_t byte_rate;      // sample_rate * num_channels * bits_per_sample / 8
    uint16_t block_align;    // num_channels * bits_per_sample / 8
    uint16_t bits_per_sample;// 16
    char subchunk2_id[4];   // "data"
    uint32_t subchunk2_size;// PCM Data length
} __attribute__((packed)) wav_header_t;

static void create_wav_header(wav_header_t *header, uint32_t pcm_data_len) {
    memcpy(header->chunk_id, "RIFF", 4);
    header->chunk_size = pcm_data_len + sizeof(wav_header_t) - 8;
    memcpy(header->format, "WAVE", 4);
    memcpy(header->subchunk1_id, "fmt ", 4);
    header->subchunk1_size = 16;
    header->audio_format = 1; // PCM
    header->num_channels = 1; // Mono
    header->sample_rate = 16000;
    header->bits_per_sample = 16;
    header->byte_rate = 16000 * 1 * 2;
    header->block_align = 1 * 2;
    memcpy(header->subchunk2_id, "data", 4);
    header->subchunk2_size = pcm_data_len;
}

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

static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    ws_playback_ctx_t *ctx = (ws_playback_ctx_t *)handler_args;

    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "⚡ WebSocket Connected to Relay Server!");
            break;
        case WEBSOCKET_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "⚡ WebSocket Disconnected");
            break;
        case WEBSOCKET_EVENT_DATA:
            if (data->op_code == 0x01) { // Text JSON frame
                ESP_LOGI(TAG, "📩 WS Metadata Received (%d bytes): %.*s", data->data_len, data->data_len, data->data_ptr);
            } else if (data->op_code == 0x02 || data->op_code == 0x00) { // Binary PCM audio frame (or continuation)
                if (data->data_len > 0 && ctx && ctx->play_dev) {
                    ctx->is_playing = true;
                    int current_vol = bsp_board_get_volume();
                    float vol_scale = (float)current_vol / 100.0f;

                    int16_t *pcm_samples = (int16_t *)data->data_ptr;
                    int num_samples = data->data_len / sizeof(int16_t);
                    for (int i = 0; i < num_samples; i++) {
                        int32_t sample = (int32_t)(pcm_samples[i] * vol_scale);
                        if (sample > 32767) sample = 32767;
                        if (sample < -32768) sample = -32768;
                        pcm_samples[i] = (int16_t)sample;
                    }

                    esp_codec_dev_write(ctx->play_dev, (void *)pcm_samples, data->data_len);
                    ctx->total_audio_read += data->data_len;
                }
            }
            break;
        case WEBSOCKET_EVENT_ERROR:
            ESP_LOGE(TAG, "⚡ WebSocket Error");
            break;
    }
}

// Real-time audio streaming queue chunk definition (20ms per frame at 16kHz mono = 320 samples = 640 bytes)
#define STREAM_FRAME_SAMPLES 320 
#define STREAM_FRAME_BYTES   (STREAM_FRAME_SAMPLES * sizeof(int16_t))

typedef struct {
    int16_t pcm_data[STREAM_FRAME_SAMPLES];
    uint16_t length_bytes;
    bool is_speech;
} audio_frame_t;

static QueueHandle_t s_audio_stream_queue = NULL;
static bool s_is_assistant_speaking = false;
static bool s_barge_in_triggered = false;

static void send_audio_and_play_response(int16_t *mono_pcm_buf, int pcm_mono_bytes)
{
    ESP_LOGI(TAG, "⚡ Connecting to Gemini Live WebSocket Backend...");
    rgb_led_set_all(200, 150, 0); // Yellow = Thinking / Sending to Gemini

    wav_header_t wav_hdr;
    create_wav_header(&wav_hdr, pcm_mono_bytes);
    int total_wav_size = sizeof(wav_header_t) + pcm_mono_bytes;

    char ws_url[160];
    snprintf(ws_url, sizeof(ws_url), "ws://%s:%s/ws/live/%s", SERVER_IP, SERVER_PORT, SESSION_ID);
    ESP_LOGI(TAG, "Connecting Live WebSocket to: %s (%d bytes payload)", ws_url, total_wav_size);

    ws_playback_ctx_t ctx = {
        .play_dev = esp_ret_play_dev(),
        .total_audio_read = 0,
        .is_playing = false
    };

    esp_websocket_client_config_t ws_cfg = {
        .uri = ws_url,
        .buffer_size = 8192,
        .reconnect_timeout_ms = 10000,
        .network_timeout_ms = 10000,
    };

    esp_websocket_client_handle_t client = esp_websocket_client_init(&ws_cfg);
    esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY, websocket_event_handler, (void *)&ctx);

    esp_err_t err = esp_websocket_client_start(client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start Live WebSocket client: %s", esp_err_to_name(err));
        esp_websocket_client_destroy(client);
        rgb_led_set_all(255, 0, 0);
        return;
    }

    // Wait for connection handshake (up to 8 seconds for SSL / Gemini API handshake)
    int wait_count = 0;
    while (!esp_websocket_client_is_connected(client) && wait_count++ < 80) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    if (!esp_websocket_client_is_connected(client)) {
        ESP_LOGE(TAG, "WebSocket handshake timed out!");
        esp_websocket_client_stop(client);
        esp_websocket_client_destroy(client);
        rgb_led_set_all(255, 0, 0);
        return;
    }

    // Send initial 44-byte WAV header binary frame
    esp_websocket_client_send_bin(client, (const char *)&wav_hdr, sizeof(wav_header_t), portMAX_DELAY);
    
    // Stream PCM Audio chunks from queue
    int bytes_sent = 0;
    while (bytes_sent < pcm_mono_bytes) {
        int chunk = (pcm_mono_bytes - bytes_sent > 1024) ? 1024 : (pcm_mono_bytes - bytes_sent);
        esp_websocket_client_send_bin(client, (const char *)mono_pcm_buf + (bytes_sent / 2), chunk, portMAX_DELAY);
        bytes_sent += chunk;
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    // Wait for Gemini response playback
    s_is_assistant_speaking = true;
    rgb_led_set_all(0, 200, 0); // Green = Playing response
    int timeout_ticks = 0;
    while (timeout_ticks++ < 150) { // Up to 15 seconds wait
        vTaskDelay(pdMS_TO_TICKS(100));
        if (s_barge_in_triggered) {
            ESP_LOGI(TAG, "⚡ [BARGE-IN] Flushed playback due to user speech!");
            break;
        }
        if (ctx.is_playing && ctx.total_audio_read > 0 && timeout_ticks > 40) {
            vTaskDelay(pdMS_TO_TICKS(1000));
            break;
        }
    }

    s_is_assistant_speaking = false;
    s_barge_in_triggered = false;
    ESP_LOGI(TAG, "Finished WebSocket Live response cycle (%d bytes played).", ctx.total_audio_read);

    esp_websocket_client_stop(client);
    esp_websocket_client_destroy(client);
    vTaskDelay(pdMS_TO_TICKS(300));
    rgb_led_clear();
}


/**
 * Record speech into mono_buf using VAD.
 * Returns the actual recorded payload size in bytes.
 */
static int record_speech_mono_vad(int16_t *mono_buf, int max_mono_bytes)
{
    int chunk_samples = CHUNK_SAMPLES;
    int chunk_raw_bytes = chunk_samples * CHANNELS * sizeof(int32_t); // 32-bit per sample from ES7210
    int32_t *raw_chunk = (int32_t *)malloc(chunk_raw_bytes);
    if (!raw_chunk) {
        ESP_LOGE(TAG, "Failed to allocate raw chunk buffer!");
        return 0;
    }

    // Flush leftover ADC DMA queue (discard 5 chunks ~160ms of stale audio/echo)
    for (int f = 0; f < 5; f++) {
        esp_get_feed_data(true, (int16_t *)raw_chunk, chunk_raw_bytes);
    }

    int total_mono_samples_max = max_mono_bytes / sizeof(int16_t);
    int recorded_samples = 0;

    bool speech_started = false;
    int silence_accum_ms = 0;
    int chunk_duration_ms = (CHUNK_SAMPLES * 1000) / SAMPLE_RATE; // ~32ms per 512 samples at 16kHz

    int16_t chunk_mono[CHUNK_SAMPLES];

    while (recorded_samples < total_mono_samples_max) {
        esp_err_t ret = esp_get_feed_data(true, (int16_t *)raw_chunk, chunk_raw_bytes);
        if (ret != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        int samples_this_chunk = chunk_samples;
        if (recorded_samples + samples_this_chunk > total_mono_samples_max) {
            samples_this_chunk = total_mono_samples_max - recorded_samples;
        }

        // Multi-Mic Mix: Pick peak amplitude across all ES7210 microphone channels for crystal-clear voice
        for (int i = 0; i < samples_this_chunk; i++) {
            int16_t ch0 = (int16_t)(raw_chunk[CHANNELS * i + 0] >> 16);
            int16_t ch1 = (int16_t)(raw_chunk[CHANNELS * i + 1] >> 16);
            int16_t ch2 = (int16_t)(raw_chunk[CHANNELS * i + 2] >> 16);
            int16_t ch3 = (int16_t)(raw_chunk[CHANNELS * i + 3] >> 16);

            int16_t max_s = ch0;
            if (abs(ch1) > abs(max_s)) max_s = ch1;
            if (abs(ch2) > abs(max_s)) max_s = ch2;
            if (abs(ch3) > abs(max_s)) max_s = ch3;

            chunk_mono[i] = max_s;
            mono_buf[recorded_samples + i] = max_s;
        }

        recorded_samples += samples_this_chunk;

        // VAD Analysis on current chunk
        float rms = compute_pcm_rms(chunk_mono, samples_this_chunk);
        int current_duration_ms = (recorded_samples * 1000) / SAMPLE_RATE;

        if (rms >= VAD_THRESHOLD_RMS) {
            if (!speech_started) {
                ESP_LOGI(TAG, "🗣️ Speech detected! (RMS: %.1f)", rms);
                speech_started = true;
            }
            silence_accum_ms = 0; // Reset silence timer on active speech
        } else if (speech_started) {
            silence_accum_ms += chunk_duration_ms;

            // Stop recording if silence exceeds SILENCE_TIMEOUT_MS after MIN_SPEECH_MS of audio
            if (current_duration_ms >= MIN_SPEECH_MS && silence_accum_ms >= SILENCE_TIMEOUT_MS) {
                ESP_LOGI(TAG, "⏹️ Silence detected (%d ms). Stopping recording early at %.2f seconds.", 
                         silence_accum_ms, (float)current_duration_ms / 1000.0f);
                break;
            }
        }
    }

    free(raw_chunk);

    if (!speech_started) {
        ESP_LOGI(TAG, "🤫 No speech detected. Waiting for voice prompt...");
        return 0;
    }

    int final_bytes = recorded_samples * sizeof(int16_t);
    return final_bytes;
}

#define GPIO_BTN_BOOT     GPIO_NUM_0   // BOOT Button
#define GPIO_BTN_VOL_UP   GPIO_NUM_1   // GPIO 1
#define GPIO_BTN_VOL_DOWN GPIO_NUM_2   // GPIO 2

static void gpio_button_task(void *pvParameters)
{
    // Enable pullups on all potential hardware button pins for ESP32-S3 boards
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

    bool is_muted = false;
    int last_vol = 70;
    uint16_t prev_tca_val = 0xFFFF;

    while (1) {
        // Read GPIO pin levels
        int p0  = gpio_get_level(GPIO_NUM_0);  // BOOT button
        int p1  = gpio_get_level(GPIO_NUM_1);  // Vol Up / Custom
        int p2  = gpio_get_level(GPIO_NUM_2);  // Vol Down / Custom
        int p3  = gpio_get_level(GPIO_NUM_3);  // Custom
        int p4  = gpio_get_level(GPIO_NUM_4);  // Mute / Custom
        int p5  = gpio_get_level(GPIO_NUM_5);  // Vol Up
        int p6  = gpio_get_level(GPIO_NUM_6);  // Vol Down
        int p38 = gpio_get_level(GPIO_NUM_38); // Vol Up (BOX/KorVo)
        int p39 = gpio_get_level(GPIO_NUM_39); // Vol Down (BOX/KorVo)
        int p40 = gpio_get_level(GPIO_NUM_40); // Mute (BOX/KorVo)

        // Read TCA9555 I2C Expander Inputs (Address 0x20, Port 0 & Port 1)
        uint16_t tca_val = bsp_board_read_tca9555_inputs();

        if (tca_val != prev_tca_val && tca_val != 0xFFFF) {
            ESP_LOGI(TAG, "🔍 TCA9555 Raw State Changed: 0x%04X (Port 0: 0x%02X, Port 1: 0x%02X)",
                     tca_val, (uint8_t)(tca_val & 0xFF), (uint8_t)(tca_val >> 8));

            uint8_t port1 = (uint8_t)(tca_val >> 8);
            uint8_t prev_port1 = (uint8_t)(prev_tca_val >> 8);

            // 1. Button 1 (P1_1 / Bit 1): Volume UP (+15%)
            if ((port1 & 0x02) == 0 && (prev_port1 & 0x02) != 0) {
                int cur_vol = bsp_board_get_volume();
                int new_vol = cur_vol + 15;
                if (new_vol > 100) new_vol = 100;
                bsp_board_set_volume(new_vol);
                ESP_LOGI(TAG, "🔊 Waveshare Button 1 (P1_1) Pressed -> Volume UP: %d%%", new_vol);
                
                int num_leds = (new_vol * 7 + 50) / 100;
                if (num_leds < 1 && new_vol > 0) num_leds = 1;
                rgb_led_set_vu_meter(num_leds);
                vTaskDelay(pdMS_TO_TICKS(800));
                rgb_led_clear();
            }

            // 2. Button 2 (P1_2 / Bit 2): Volume DOWN (-15%)
            if ((port1 & 0x04) == 0 && (prev_port1 & 0x04) != 0) {
                int cur_vol = bsp_board_get_volume();
                int new_vol = cur_vol - 15;
                if (new_vol < 0) new_vol = 0;
                bsp_board_set_volume(new_vol);
                ESP_LOGI(TAG, "🔉 Waveshare Button 2 (P1_2) Pressed -> Volume DOWN: %d%%", new_vol);

                int num_leds = (new_vol * 7 + 50) / 100;
                rgb_led_set_vu_meter(num_leds);
                vTaskDelay(pdMS_TO_TICKS(800));
                rgb_led_clear();
            }

            // 3. Button 3 (P1_3 / Bit 3): Mute / Unmute Toggle
            if ((port1 & 0x08) == 0 && (prev_port1 & 0x08) != 0) {
                is_muted = !is_muted;
                if (is_muted) {
                    last_vol = bsp_board_get_volume();
                    bsp_board_set_volume(0);
                    ESP_LOGI(TAG, "🔇 Waveshare Button 3 (P1_3) Pressed -> Speaker MUTED (0%%)");
                    rgb_led_set_all(255, 0, 0); // Solid Red = Muted
                    vTaskDelay(pdMS_TO_TICKS(600));
                    rgb_led_clear();
                } else {
                    int res_vol = last_vol > 0 ? last_vol : 70;
                    bsp_board_set_volume(res_vol);
                    ESP_LOGI(TAG, "🔊 Waveshare Button 3 (P1_3) Pressed -> Speaker UNMUTED (%d%%)", res_vol);
                    rgb_led_set_all(0, 255, 0); // Solid Green = Unmuted
                    vTaskDelay(pdMS_TO_TICKS(600));
                    rgb_led_clear();
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
    ESP_LOGI(TAG, " ESP32-S3 Gemini Voice Assistant Starting...    ");
    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, "🎛️ HARDWARE BUTTON & GPIO CONFIGURATION:");
    ESP_LOGI(TAG, "   • RESET Button : Hardwired EN Pin (Chip Reset)");
    ESP_LOGI(TAG, "   • BOOT Button  : GPIO 0 (Mute / Unmute Toggle)");
    ESP_LOGI(TAG, "   • VOL+ Button  : GPIO 1, 3, 5, 38 & TCA9555 (+10%% Vol)");
    ESP_LOGI(TAG, "   • VOL- Button  : GPIO 2, 6, 39 & TCA9555 (-10%% Vol)");
    ESP_LOGI(TAG, "   • USER Button  : GPIO 4, 40 & TCA9555 (Mute/Action)");
    ESP_LOGI(TAG, "------------------------------------------------");

    // 1. Initialize Peripherals & Codecs
    ESP_ERROR_CHECK(esp_board_init(SAMPLE_RATE, 1, 16));
    rgb_led_init();

    // Launch GPIO Button Listener Task
    xTaskCreate(gpio_button_task, "gpio_button_task", 3072, NULL, 5, NULL);

    // 2. Connect Wi-Fi
    rgb_led_set_all(0, 0, 255); // Blue = Connecting Wi-Fi
    ESP_LOGI(TAG, "Connecting to Wi-Fi...");
    if (wifi_init_sta() != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi connection failed! Please check SSID & Password.");
        rgb_led_set_all(255, 0, 0);
        return;
    }
    rgb_led_clear();

    // 3. Allocate Mono Audio Buffer (up to 8 sec mono 16-bit 16kHz)
    int16_t *mono_pcm_buf = (int16_t *)malloc(MONO_PCM_BUF_SIZE);
    if (!mono_pcm_buf) {
        ESP_LOGE(TAG, "Failed to allocate memory for mono audio buffer!");
        return;
    }

    ESP_LOGI(TAG, "System Ready! Assistant loop starting in 2 seconds...");
    vTaskDelay(pdMS_TO_TICKS(2000));

    while (1) {
        ESP_LOGI(TAG, "---------------------------------------------");
        ESP_LOGI(TAG, "🎙️ Listening with Smart VAD... Speak into microphone! (Up to %d sec)", MAX_RECORD_TIME_SEC);
        rgb_led_set_all(0, 100, 255); // Cyan/Blue = Recording Speech

        int recorded_bytes = record_speech_mono_vad(mono_pcm_buf, MONO_PCM_BUF_SIZE);
        if (recorded_bytes > 0) {
            ESP_LOGI(TAG, "Finished recording speech (%d bytes).", recorded_bytes);
            send_audio_and_play_response(mono_pcm_buf, recorded_bytes);
            ESP_LOGI(TAG, "Waiting 2.5 seconds for room echo to decay...");
            rgb_led_clear();
            vTaskDelay(pdMS_TO_TICKS(2500));
        } else {
            rgb_led_clear();
            vTaskDelay(pdMS_TO_TICKS(200));
        }
    }

    free(mono_pcm_buf);
}

