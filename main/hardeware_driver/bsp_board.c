#include "bsp_board.h"

#include <stdbool.h>
#include <string.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "driver/gpio.h"
#include "esp_log.h"

#define ADC_I2S_CHANNEL 4

static const char *TAG = "bsp_board";
static int s_play_sample_rate = 16000;
static int s_play_channel_format = 1;
static int s_bits_per_chan = 16;

static i2s_chan_handle_t tx_handle = NULL;
static i2s_chan_handle_t rx_handle = NULL;

static const audio_codec_data_if_t *record_data_if  = NULL;
static const audio_codec_ctrl_if_t *record_ctrl_if  = NULL;
static const audio_codec_if_t *record_codec_if      = NULL;
static esp_codec_dev_handle_t record_dev            = NULL;

static const audio_codec_data_if_t *play_data_if    = NULL;
static const audio_codec_ctrl_if_t *play_ctrl_if    = NULL;
static const audio_codec_if_t *play_codec_if        = NULL;
static const audio_codec_gpio_if_t *gpio_if         = NULL;
static esp_codec_dev_handle_t play_dev              = NULL;

static i2c_master_bus_handle_t i2c_bus_handle       = NULL;

esp_codec_dev_handle_t esp_ret_play_dev(void)
{
    return play_dev;
}

i2c_master_bus_handle_t esp_ret_i2c_handle(void)
{
    return i2c_bus_handle;
}

static esp_err_t i2c_master_init(void)
{
    const i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_NUM,
        .sda_io_num = GPIO_I2C_SDA,
        .scl_io_num = GPIO_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
    };

    esp_err_t ret = i2c_new_master_bus(&bus_config, &i2c_bus_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize I2C bus: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "I2C bus initialized successfully");
    return ESP_OK;
}

static i2c_master_dev_handle_t s_tca9555_dev = NULL;

/**
 * Powers ON the Speaker Power Amplifier (PA) via TCA9555 I2C GPIO Expander (Port 1 Pin 0 / Pin 8)
 */
static esp_err_t enable_speaker_pa_tca9555(void)
{
    if (!i2c_bus_handle) {
        ESP_LOGE(TAG, "I2C bus handle is NULL!");
        return ESP_FAIL;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = 0x20, // TCA9555 default I2C address
        .scl_speed_hz = 100000,
    };

    esp_err_t err = i2c_master_bus_add_device(i2c_bus_handle, &dev_cfg, &s_tca9555_dev);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "TCA9555 add device notice: %s", esp_err_to_name(err));
        return err;
    }

    // 1. Configure Port 0 as inputs (Register 0x06 = 0xFF)
    uint8_t cfg_port0[2] = {0x06, 0xFF};
    i2c_master_transmit(s_tca9555_dev, cfg_port0, sizeof(cfg_port0), -1);

    // 2. Configure Port 1: P1_0 (PA Enable) as OUTPUT, P1_1, P1_2, P1_3 (User Buttons) as INPUTS (Register 0x07 = 0xFE)
    uint8_t cfg_cmd[2] = {0x07, 0xFE};
    i2c_master_transmit(s_tca9555_dev, cfg_cmd, sizeof(cfg_cmd), -1);

    // 3. Set Port 1 Pin 0 to HIGH (1) (Register 0x03, Bit 0 = 1) -> Powers ON PA Amplifier!
    uint8_t out_cmd[2] = {0x03, 0x01};
    i2c_master_transmit(s_tca9555_dev, out_cmd, sizeof(out_cmd), -1);

    ESP_LOGI(TAG, "🔊 Waveshare ESP32-S3-AUDIO TCA9555 Port 1 (P1_1, P1_2, P1_3) initialized as Button Inputs!");
    return ESP_OK;
}

uint16_t bsp_board_read_tca9555_inputs(void)
{
    if (!s_tca9555_dev) return 0xFFFF;
    uint8_t reg = 0x00; // Input Port 0
    uint8_t data[2] = {0xFF, 0xFF};
    if (i2c_master_transmit_receive(s_tca9555_dev, &reg, 1, data, 2, 100) == ESP_OK) {
        return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
    }
    return 0xFFFF;
}

static esp_err_t bsp_i2s_init(int i2s_num, uint32_t sample_rate, int channel_format, int bits_per_chan)
{
    esp_err_t ret_val = ESP_OK;
    i2s_slot_mode_t channel_fmt = I2S_SLOT_MODE_STEREO;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(i2s_num, I2S_ROLE_MASTER);
    ret_val |= i2s_new_channel(&chan_cfg, &tx_handle, &rx_handle);

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(16, channel_fmt),
        .gpio_cfg = {
            .mclk = GPIO_I2S_MCLK,
            .bclk = GPIO_I2S_SCLK,
            .ws   = GPIO_I2S_LRCK,
            .dout = GPIO_I2S_DOUT,
            .din  = GPIO_I2S_SDIN,
        },
    };

    ret_val |= i2s_channel_init_std_mode(tx_handle, &std_cfg);
    ret_val |= i2s_channel_init_std_mode(rx_handle, &std_cfg);
    ret_val |= i2s_channel_enable(tx_handle);
    ret_val |= i2s_channel_enable(rx_handle);

    return ret_val;
}

esp_err_t bsp_codec_adc_init(int sample_rate)
{
    audio_codec_i2s_cfg_t i2s_cfg = {
        .port = 1,
        .rx_handle = rx_handle,
        .tx_handle = NULL,
    };
    record_data_if = audio_codec_new_i2s_data(&i2s_cfg);

    audio_codec_i2c_cfg_t i2c_cfg = {
        .addr = ES7210_CODEC_DEFAULT_ADDR,
        .bus_handle = i2c_bus_handle
    };
    record_ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);

    es7210_codec_cfg_t es7210_cfg = {
        .ctrl_if = (audio_codec_ctrl_if_t *)record_ctrl_if,
        .mic_selected = ES7210_SEL_MIC1 | ES7210_SEL_MIC2 | ES7210_SEL_MIC3 | ES7210_SEL_MIC4,
    };  
    record_codec_if = es7210_codec_new(&es7210_cfg);

    esp_codec_dev_cfg_t dev_cfg = {
        .codec_if = (audio_codec_if_t *)record_codec_if,
        .data_if = (audio_codec_data_if_t *)record_data_if,
        .dev_type = ESP_CODEC_DEV_TYPE_IN,
    };
    record_dev = esp_codec_dev_new(&dev_cfg);

    esp_codec_dev_sample_info_t fs = {
        .sample_rate = 16000,
        .channel = 2,
        .bits_per_sample = 32,
    };
    esp_codec_dev_open(record_dev, &fs);

    esp_codec_dev_set_in_channel_gain(record_dev, ESP_CODEC_DEV_MAKE_CHANNEL_MASK(0), RECORD_VOLUME);
    esp_codec_dev_set_in_channel_gain(record_dev, ESP_CODEC_DEV_MAKE_CHANNEL_MASK(1), RECORD_VOLUME);
    esp_codec_dev_set_in_channel_gain(record_dev, ESP_CODEC_DEV_MAKE_CHANNEL_MASK(2), RECORD_VOLUME);
    esp_codec_dev_set_in_channel_gain(record_dev, ESP_CODEC_DEV_MAKE_CHANNEL_MASK(3), RECORD_VOLUME);

    return ESP_OK;
}

esp_err_t bsp_codec_dac_init(int sample_rate, int channel_format, int bits_per_chan)
{
    audio_codec_i2s_cfg_t i2s_cfg = {
        .port = 1,
        .rx_handle = NULL,
        .tx_handle = tx_handle,
    };
    play_data_if = audio_codec_new_i2s_data(&i2s_cfg);

    audio_codec_i2c_cfg_t i2c_cfg = {
        .addr = ES8311_CODEC_DEFAULT_ADDR,
        .bus_handle = i2c_bus_handle
    };
    play_ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    gpio_if = audio_codec_new_gpio();

    es8311_codec_cfg_t es8311_cfg = {
        .codec_mode = ESP_CODEC_DEV_WORK_MODE_DAC,
        .ctrl_if = (audio_codec_ctrl_if_t *)play_ctrl_if,
        .gpio_if = (audio_codec_gpio_if_t *)gpio_if,
        .pa_pin = -1,
        .use_mclk = false,
    };
    play_codec_if = es8311_codec_new(&es8311_cfg);

    esp_codec_dev_cfg_t dev_cfg = {
        .codec_if = (audio_codec_if_t *)play_codec_if,
        .data_if = (audio_codec_data_if_t *)play_data_if,
        .dev_type = ESP_CODEC_DEV_TYPE_OUT,
    };
    play_dev = esp_codec_dev_new(&dev_cfg);

    esp_codec_dev_sample_info_t fs = {
        .sample_rate = sample_rate,
        .channel = channel_format,
        .bits_per_sample = bits_per_chan,
    };
    esp_codec_dev_set_out_vol(play_dev, 80);
    esp_codec_dev_open(play_dev, &fs);

    ESP_LOGI(TAG, "ES8311 DAC Speaker Codec initialized successfully!");
    return ESP_OK;
}

esp_err_t esp_get_feed_data(bool is_get_raw_channel, int16_t *buffer, int buffer_len)
{
    esp_err_t ret = ESP_OK;
    int audio_chunksize = buffer_len / (sizeof(int16_t) * ADC_I2S_CHANNEL);

    ret = esp_codec_dev_read(record_dev, (void *)buffer, buffer_len);
    if (!is_get_raw_channel) {
        for (int i = 0; i < audio_chunksize; i++) {
            int16_t ref = buffer[4 * i + 0];
            buffer[3 * i + 0] = buffer[4 * i + 1];
            buffer[3 * i + 1] = buffer[4 * i + 3];
            buffer[3 * i + 2] = ref;
        }
    }

    return ret;
}

int esp_get_feed_channel(void)
{
    return ADC_I2S_CHANNEL;
}

static int s_current_volume = 70;

esp_err_t bsp_board_set_volume(int volume)
{
    if (volume < 0) volume = 0;
    if (volume > 100) volume = 100;
    s_current_volume = volume;
    if (play_dev) {
        esp_codec_dev_set_out_vol(play_dev, volume);
        ESP_LOGI(TAG, "🔊 Hardware Volume set to: %d%%", volume);
        return ESP_OK;
    }
    return ESP_FAIL;
}

int bsp_board_get_volume(void)
{
    return s_current_volume;
}

esp_err_t esp_board_init(uint32_t sample_rate, int channel_format, int bits_per_chan)
{
    i2c_master_init();
    enable_speaker_pa_tca9555();

    s_play_sample_rate = sample_rate;
    s_play_channel_format = channel_format;
    s_bits_per_chan = bits_per_chan;

    bsp_i2s_init(1, 16000, 2, 16);
    bsp_codec_adc_init(16000);
    bsp_codec_dac_init(16000, 1, 16);

    bsp_board_set_volume(70);

    return ESP_OK;
}
