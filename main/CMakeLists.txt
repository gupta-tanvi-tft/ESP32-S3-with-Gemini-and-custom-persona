set(SOURCES
    "main.c"
    "hardeware_driver/bsp_board.c"
    "rgb_led_driver/rgb_led_driver.c"
    "wifi_driver/wifi_connect.c"
)

set(INCLUDE_DIRS
    "."
    "hardeware_driver"
    "rgb_led_driver"
    "wifi_driver"
)

idf_component_register(
    SRCS ${SOURCES}
    INCLUDE_DIRS ${INCLUDE_DIRS}
    REQUIRES driver led_strip esp_codec_dev log freertos esp_wifi esp_event nvs_flash esp_http_client
)
