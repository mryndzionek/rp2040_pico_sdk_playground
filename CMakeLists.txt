cmake_minimum_required(VERSION 3.13)

# initialize the SDK based on PICO_SDK_PATH
# note: this must happen before project()
include(pico_sdk_import.cmake)

project(rp2040_pico_sdk_playground)

# initialize the Raspberry Pi Pico SDK
pico_sdk_init()

include_directories(../include)

file(GLOB U8G2_SRCS
     "external/u8g2/csrc/*.c"
)

add_compile_options(-Wall -Werror -fdata-sections -ffunction-sections)

include_directories(include external/u8g2/csrc/)

add_library(
    u8g2
    ${U8G2_SRCS}
)

add_executable(rpi_lcd_test
        src/rpi_lcd_test.c
        src/u8g2_ili9486_driver.c
        )

pico_generate_pio_header(rpi_lcd_test ${CMAKE_CURRENT_LIST_DIR}/src/ili9486_lcd_8bit_data.pio)

target_link_libraries(rpi_lcd_test
        pico_stdlib
        hardware_pwm
        hardware_pio
        hardware_dma
        hardware_irq
        u8g2)

pico_add_extra_outputs(rpi_lcd_test)

add_executable(rpi_ws2812_lamp
        src/rpi_ws2812_lamp.c
        )

pico_generate_pio_header(rpi_ws2812_lamp ${CMAKE_CURRENT_LIST_DIR}/src/ws2812.pio)

target_link_libraries(rpi_ws2812_lamp
        pico_stdlib
        hardware_pio
        )

pico_add_extra_outputs(rpi_ws2812_lamp)

