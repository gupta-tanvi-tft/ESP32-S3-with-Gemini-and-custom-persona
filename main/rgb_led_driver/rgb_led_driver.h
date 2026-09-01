#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "led_strip.h"

#ifdef __cplusplus
extern "C" {
#endif

void rgb_led_init(void);
void rgb_led_set_vu_meter(int level); // level: 0 to 7
void rgb_led_set_pixel(uint32_t index, uint8_t r, uint8_t g, uint8_t b);
void rgb_led_set_all(uint8_t r, uint8_t g, uint8_t b);
void rgb_led_clear(void);

#ifdef __cplusplus
}
#endif
