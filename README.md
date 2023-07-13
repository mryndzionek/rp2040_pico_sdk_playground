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

