# rp2040_pico_sdk_playground

## Building

```
git submodule --init --recursive
export PICO_SDK_PATH=/path/to/your/pico-sdk
mkdir build
cd build
cmake ..
make
```

## Applications

### rpi_lcd_test

![rpi_lcd_test](images/rpi_lcd_test.gif)

A simple app demonstrating the use of PIO and DMA for
efficient data transfers to a 480x320 TFT display.
The graphics library creates a monochromatic image
in memory which is expanded to 16-bit color codes
in PIO block and transferred via 8-bit parallel
interface.

### rpi_ws2812_lamp

A simple app controlled by one button, controlling
a WS2812 strip/matrix. Can be used to test strips/matrices.

| Button press/sequence | Action                             |
|-----------------------|------------------------------------|
| Hold                  | Adjust brightness                  |
| Tap and hold          | Adjust color                       |
| One tap               | Toggle 3-minute timer              |
| Two taps              | Toggle between 'Off' and 'Max Red' |
| Three taps            | Cycle through presets              |
| Four taps             | Activate "Doom flicker" feature    |


### rpi_inmp441_fft_demo

Reading audio from a INMP441 MEMS microphone using PIO+DMA,
computing fixed point FFT (CMSIS-DSP) and displaying an ASCII
spectrogram on serial.

https://github.com/mryndzionek/rp2040_pico_sdk_playground/assets/786191/143725ea-1283-4246-8fa4-98fe817371da

### rpi_tflm_micro_speech_demo

#### /Keyword Spotting/Visual Wake Words/ on RP2040

TFLM [Micro Speech](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/README.md)
model is ported to Raspberry Pi Pico (RP2040, Cortex-M0+). Sound is from a MEMS
I2S microphone (INMP441). The CPU is clocked at 250MHz. Data from
the microphone is transferred using PIO+DMA. With this configuration real-time
speech analysis is possible (stride is 20ms at 16kHz sample rate and single inference takes ~19ms).

