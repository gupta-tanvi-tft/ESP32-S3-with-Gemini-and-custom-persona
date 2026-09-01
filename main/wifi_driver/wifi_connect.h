#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WIFI_SSID "TFTus-WiFi"
#define WIFI_PASS "TFTus@123#$"

esp_err_t wifi_init_sta(void);

#ifdef __cplusplus
}
#endif
