#include "rgb_led_driver.h"
#include "esp_log.h"
#include "bsp_board.h"

static const char *TAG = "rgb_led";
static led_strip_handle_t s_led_strip = NULL;

void rgb_led_init(void)
{
    led_strip_config_t strip_config = {
        .strip_gpio_num = LED_STRIP_GPIO_PIN,
        .max_leds = LED_STRIP_LED_COUNT,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_RGB,
        .flags = {
            .invert_out = false,
        }
    };

    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .mem_block_symbols = 0,
        .flags = {
            .with_dma = 0,
        }
    };

    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &s_led_strip));
    led_strip_clear(s_led_strip);
    ESP_LOGI(TAG, "RGB LED Strip initialized (GPIO %d, %d LEDs)", LED_STRIP_GPIO_PIN, LED_STRIP_LED_COUNT);
}

void rgb_led_set_pixel(uint32_t index, uint8_t r, uint8_t g, uint8_t b)
{
    if (s_led_strip && index < LED_STRIP_LED_COUNT) {
        led_strip_set_pixel(s_led_strip, index, r, g, b);
        led_strip_refresh(s_led_strip);
    }
}

void rgb_led_set_all(uint8_t r, uint8_t g, uint8_t b)
{
    if (!s_led_strip) return;
    for (int i = 0; i < LED_STRIP_LED_COUNT; i++) {
        led_strip_set_pixel(s_led_strip, i, r, g, b);
    }
    led_strip_refresh(s_led_strip);
}

void rgb_led_clear(void)
{
    if (s_led_strip) {
        led_strip_clear(s_led_strip);
    }
}

void rgb_led_set_vu_meter(int level)
{
    if (!s_led_strip) return;
    if (level < 0) level = 0;
    if (level > LED_STRIP_LED_COUNT) level = LED_STRIP_LED_COUNT;

    for (int i = 0; i < LED_STRIP_LED_COUNT; i++) {
        if (i < level) {
            if (i < 4) {
                led_strip_set_pixel(s_led_strip, i, 0, 180, 0);
            } else if (i < 6) {
                led_strip_set_pixel(s_led_strip, i, 180, 180, 0);
            } else {
                led_strip_set_pixel(s_led_strip, i, 220, 0, 0);
            }
        } else {
            led_strip_set_pixel(s_led_strip, i, 0, 0, 0);
        }
    }
    led_strip_refresh(s_led_strip);
}
