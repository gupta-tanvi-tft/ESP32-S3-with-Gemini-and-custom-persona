#pragma once

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/i2s_tdm.h"
#include "driver/i2s_common.h"
#include "driver/i2c_master.h"

#include "soc/soc_caps.h"
#include "esp_idf_version.h"

#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "esp_codec_dev_os.h"
#include "sdkconfig.h"

#ifdef __cplusplus
extern "C" {
#endif

#define I2C_NUM         (0)
#define GPIO_I2C_SCL    (GPIO_NUM_10)
#define GPIO_I2C_SDA    (GPIO_NUM_11)

#define FUNC_I2S_EN         (1)
#define GPIO_I2S_LRCK       (GPIO_NUM_14)
#define GPIO_I2S_MCLK       (GPIO_NUM_12)
#define GPIO_I2S_SCLK       (GPIO_NUM_13)
#define GPIO_I2S_SDIN       (GPIO_NUM_15)
#define GPIO_I2S_DOUT       (GPIO_NUM_16)

#define RECORD_VOLUME       (65.0)
#define PLAYER_VOLUME       (60)

#define LED_STRIP_GPIO_PIN  38
#define LED_STRIP_LED_COUNT 7

esp_err_t esp_board_init(uint32_t sample_rate, int channel_format, int bits_per_chan);
esp_err_t esp_get_feed_data(bool is_get_raw_channel, int16_t *buffer, int buffer_len);
int esp_get_feed_channel(void);
esp_err_t bsp_board_set_volume(int volume);
int bsp_board_get_volume(void);
uint16_t bsp_board_read_tca9555_inputs(void);

i2c_master_bus_handle_t esp_ret_i2c_handle(void);
esp_codec_dev_handle_t esp_ret_play_dev(void);

#ifdef __cplusplus
}
#endif
