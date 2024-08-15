cmake_minimum_required(VERSION 3.13)

# initialize the SDK based on PICO_SDK_PATH
# note: this must happen before project()
include(pico_sdk_import.cmake)

project(rp2040_pico_sdk_playground C CXX ASM)

# initialize the Raspberry Pi Pico SDK
pico_sdk_init()

include_directories(../include)

file(GLOB U8G2_SRCS
     "external/u8g2/csrc/*.c"
)

add_compile_options(-Wall -Werror -fdata-sections -ffunction-sections)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)

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
        hardware_irq
        hardware_pwm
        hardware_pio
        hardware_dma
        u8g2)

pico_add_extra_outputs(rpi_lcd_test)

add_executable(rpi_ws2812_lamp
        src/rpi_ws2812_lamp.c
        src/plasma.c
        )

pico_generate_pio_header(rpi_ws2812_lamp ${CMAKE_CURRENT_LIST_DIR}/src/ws2812.pio)
pico_generate_pio_header(rpi_ws2812_lamp ${CMAKE_CURRENT_LIST_DIR}/src/inmp441.pio)

target_link_libraries(rpi_ws2812_lamp
        pico_stdlib
        pico_rand
        hardware_pio
        hardware_dma
        CMSISDSP
        )

pico_add_extra_outputs(rpi_ws2812_lamp)

set(CMSISDSP "${CMAKE_CURRENT_LIST_DIR}/external/CMSIS-DSP" CACHE STRING "CMSIS-DSP Library")
add_subdirectory(${CMSISDSP}/Source bin_dsp)
target_compile_options(CMSISDSP PRIVATE -Ofast)
target_link_libraries(CMSISDSP PUBLIC pico_stdlib cmsis_core)

add_executable(rpi_inmp441_fft_demo
        src/rpi_inmp441_fft_demo.c
        )

pico_generate_pio_header(rpi_inmp441_fft_demo ${CMAKE_CURRENT_LIST_DIR}/src/inmp441.pio)

target_link_libraries(rpi_inmp441_fft_demo
        pico_stdlib
        pico_rand
        hardware_pio
        hardware_dma
        CMSISDSP
        )

pico_add_extra_outputs(rpi_inmp441_fft_demo)

set(CMSISNN "${CMAKE_CURRENT_LIST_DIR}/external/CMSIS-NN" CACHE STRING "CMSIS-NN Library")
add_subdirectory(${CMSISNN} bin_nn)
target_compile_options(cmsis-nn PRIVATE -Ofast)
target_link_libraries(cmsis-nn PUBLIC pico_stdlib cmsis_core)

add_library(tflitemicro STATIC IMPORTED)
set_property(TARGET tflitemicro  PROPERTY
             IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/external/tflite-micro/\
gen/cortex_m_qemu_cortex-m0_default/lib/\
libtensorflow-microlite.a")

target_link_libraries(tflitemicro INTERFACE cmsis-nn)

set(TFLITE_MODELS_PATH "${CMAKE_CURRENT_LIST_DIR}/external/tflite-micro\
/gen/cortex_m_qemu_cortex-m0_default/genfiles\
/tensorflow/lite/micro/examples/micro_speech/models")

add_executable(rpi_tflm_micro_speech_demo
        src/rpi_tflm_micro_speech_demo.cc
        ${TFLITE_MODELS_PATH}/audio_preprocessor_int8_model_data.cc 
        ${TFLITE_MODELS_PATH}/micro_speech_quantized_model_data.cc
        )

target_include_directories(rpi_tflm_micro_speech_demo
        PRIVATE external/tflite-micro
        PRIVATE external/tflite-micro/gen/cortex_m_qemu_cortex-m0_default/genfiles
        PRIVATE external/tflite-micro/tensorflow/lite/micro/examples/micro_speech
        PRIVATE external/tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
        PRIVATE external/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp
        PRIVATE ${TFLITE_MODELS_PATH}
        )

pico_generate_pio_header(rpi_tflm_micro_speech_demo ${CMAKE_CURRENT_LIST_DIR}/src/inmp441.pio)

target_link_libraries(rpi_tflm_micro_speech_demo
        pico_stdlib
        pico_rand
        hardware_pio
        hardware_dma
        CMSISDSP
        cmsis-nn
        tflitemicro
        )

target_compile_features(rpi_tflm_micro_speech_demo PRIVATE cxx_std_17)
target_compile_definitions(rpi_tflm_micro_speech_demo PRIVATE TF_LITE_STATIC_MEMORY)
pico_add_extra_outputs(rpi_tflm_micro_speech_demo)

add_executable(rpi_inmp441_whistle_demo
        src/rpi_inmp441_whistle_demo.c
        src/whistle_detector.c
        )

pico_generate_pio_header(rpi_inmp441_whistle_demo ${CMAKE_CURRENT_LIST_DIR}/src/inmp441.pio)

target_link_libraries(rpi_inmp441_whistle_demo
        pico_stdlib
        pico_rand
        hardware_pio
        hardware_dma
        )

pico_add_extra_outputs(rpi_inmp441_whistle_demo)

