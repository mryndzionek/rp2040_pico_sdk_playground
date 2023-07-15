#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pico/stdlib.h"

#include "u8g2.h"
#include "u8g2_ili9486_driver.h"

#include "seven_seg_big.h"

static u8g2_t u8g2;

int main()
{
    stdio_init_all();
    set_sys_clock_khz(280000, true);
    setup_default_uart();

    printf("Starting\n");

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    u8g2_Setup_ili9486_8bit_480x320_f(&u8g2);

    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);
    u8g2_ClearDisplay(&u8g2);

    u8g2_DrawLine(&u8g2, 0, 0, 480, 320);
    u8g2_DrawLine(&u8g2, 0, 320, 480, 0);
    u8g2_DrawCircle(&u8g2, 240, 160, 100, U8G2_DRAW_ALL);
    u8g2_DrawCircle(&u8g2, 240, 160, 70, U8G2_DRAW_ALL);
    u8g2_SetDrawColor(&u8g2, 1);

    u8g2_uint_t points[36][2];
    for (size_t i = 0; i < 36; i++)
    {
        points[i][0] = (160 * cosf((i * 10.0) * M_PI / 180)) + 240;
        points[i][1] = (160 * sinf((i * 10.0) * M_PI / 180)) + 160;
    }

    float v = 0;
    size_t i = 0;
    uint8_t ci = 1;
    char str[16];
    float t = 0.0;
    uint fps = 0;

    while (true)
    {
        uint32_t start_time = time_us_32();
        u8g2_SetDrawColor(&u8g2, ci);
        u8g2_DrawLine(&u8g2, points[i][0],
                      points[i][1],
                      points[(i + 18) % 36][0],
                      points[(i + 18) % 36][1]);
        i++;
        if (i == 18)
        {
            i = 0;
            ci ^= 1;
        }

        u8g2_SetDrawColor(&u8g2, 1);
        u8g2_DrawRBox(&u8g2, 10, 30, 450, 150, 20);
        u8g2_SetDrawColor(&u8g2, 0);
        u8g2_DrawBox(&u8g2, 30, 40, 395, 130);
        u8g2_DrawBox(&u8g2, 0, 250, 480, 30);

        u8g2_SetDrawColor(&u8g2, 1);
        v = 100 * sinf(2 * M_PI * 0.1 * t);
        t += 0.1;
        char s = v < 0 ? '-' : ' ';
        snprintf(str, sizeof(str), "%c%03d.%01d", s,
                 abs((int)(v)), abs((int)(v * 10)) % 10);
        u8g2_SetFont(&u8g2, seg_font);
        u8g2_DrawStr(&u8g2, 30, 150, str);
        u8g2_DrawBox(&u8g2, 240 - 2 * (int)(fabs(v)), 250, 4 * (int)fabs(v), 30);

        u8g2_SetFont(&u8g2, u8g2_font_crox5hb_tf);
        snprintf(str, sizeof(str), "FPS: %d", fps);
        u8g2_DrawStr(&u8g2, 40, 315, str);

        u8g2_SendBuffer(&u8g2);
        gpio_put(LED_PIN, i % 2);
        fps = 1000000 / (time_us_32() - start_time);
    }
}
