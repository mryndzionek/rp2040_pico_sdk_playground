#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/i2c.h"

#include "u8g2.h"

#define OLED_I2C_SDA_PIN (18)
#define OLED_I2C_SCL_PIN (19)
#define OLED_I2C_SPEED (100UL)
#define OLED_I2C_INST (i2c1)

static u8g2_t u8g2;

static uint8_t u8x8_gpio_and_delay_pico(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, void *arg_ptr)
{
    switch (msg)
    {
    case U8X8_MSG_GPIO_AND_DELAY_INIT:
        break;

    case U8X8_MSG_DELAY_NANO: // delay arg_int * 1 nano second
        break;
    case U8X8_MSG_DELAY_100NANO: // delay arg_int * 100 nano seconds
        break;
    case U8X8_MSG_DELAY_10MICRO: // delay arg_int * 10 micro seconds
        break;
    case U8X8_MSG_DELAY_MILLI: // delay arg_int * 1 milli second
        sleep_ms(arg_int);
        break;
    case U8X8_MSG_DELAY_I2C:
        /* arg_int is 1 or 4: 100KHz (5us) or 400KHz (1.25us) */
        sleep_us(arg_int <= 2 ? 5 : 1);
        break;

    default:
        u8x8_SetGPIOResult(u8x8, 1); // default return value
        break;
    }
    return 1;
}

static uint8_t u8x8_byte_pico_hw_i2c(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, void *arg_ptr)
{
    uint8_t *data;
    static uint8_t buffer[132];
    static uint8_t buf_idx;

    switch (msg)
    {
    case U8X8_MSG_BYTE_SEND:
        data = (uint8_t *)arg_ptr;
        while (arg_int > 0)
        {
            assert(buf_idx < 132);
            buffer[buf_idx++] = *data;
            data++;
            arg_int--;
        }
        break;

    case U8X8_MSG_BYTE_INIT:
        i2c_init(OLED_I2C_INST, OLED_I2C_SPEED * 1000);
        gpio_set_function(OLED_I2C_SDA_PIN, GPIO_FUNC_I2C);
        gpio_set_function(OLED_I2C_SCL_PIN, GPIO_FUNC_I2C);
        gpio_pull_up(OLED_I2C_SDA_PIN);
        gpio_pull_up(OLED_I2C_SCL_PIN);
        break;

    case U8X8_MSG_BYTE_SET_DC:
        break;

    case U8X8_MSG_BYTE_START_TRANSFER:
        buf_idx = 0;
        break;

    case U8X8_MSG_BYTE_END_TRANSFER:
    {
        uint8_t addr = u8x8_GetI2CAddress(u8x8) >> 1;
        int ret = i2c_write_blocking(OLED_I2C_INST, addr, buffer, buf_idx, false);
        printf("%d\n", ret);
        if ((ret == PICO_ERROR_GENERIC) || (ret == PICO_ERROR_TIMEOUT))
        {
            return 0;
        }
    }
    break;

    default:
        return 0;
    }
    return 1;
}

int main()
{
    stdio_init_all();
    // set_sys_clock_khz(280000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    printf("Starting\n");

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    u8g2_Setup_sh1106_i2c_128x64_noname_f(&u8g2, U8G2_R0,
                                          u8x8_byte_pico_hw_i2c,
                                          u8x8_gpio_and_delay_pico);

    u8g2_SetI2CAddress(&u8g2, 0x78);
    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);
    u8g2_SetContrast(&u8g2, 255);
    u8g2_ClearDisplay(&u8g2);

    u8g2_DrawCircle(&u8g2, 64, 32, 10, U8G2_DRAW_ALL);
    u8g2_SendBuffer(&u8g2);

    while (true)
    {
        gpio_put(LED_PIN, 0);
        sleep_ms(100);
        gpio_put(LED_PIN, 1);
        sleep_ms(100);
    }
}
