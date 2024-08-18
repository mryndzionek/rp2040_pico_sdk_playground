#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pico/stdlib.h"
#include "hardware/clocks.h"

#include "u8g2.h"
#include "u8g2_ili9486_driver.h"

#include "seven_seg_big.h"

#include "GL/gl.h"
#include "zbuffer.h"

#define SCR_WIDTH (320)
#define SCR_HEIGHT (213)

extern void gears_draw(void);
extern void gears_init_scene(void);
extern uint8_t *u8g2_fbuf;

static u8g2_t u8g2;

int main()
{
    stdio_init_all();
    set_sys_clock_khz(280000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    printf("Starting\n");
    sleep_ms(1000);

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    u8g2_Setup_ili9486_8bit_480x320_f(&u8g2);

    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);
    u8g2_ClearDisplay(&u8g2);

    ZBuffer *frame_buffer = NULL;
    frame_buffer = ZB_open(SCR_WIDTH, SCR_HEIGHT, ZB_MODE_5R6G5B, 0);

    if (!frame_buffer)
    {
        printf("ZB_open failed!\n");
        for (;;)
        {
            __breakpoint();
        }
    }

    glInit(frame_buffer);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

    glShadeModel(GL_FLAT);
    glEnable(GL_LIGHTING);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    GLfloat h = (GLfloat)SCR_HEIGHT / (GLfloat)SCR_WIDTH;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1.0, 1.0, -h, h, 6.0, 80.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -55.0);

    gears_init_scene();

    glSetEnableSpecular(GL_FALSE);
    ZB_setDitheringMap(frame_buffer, 3);

    uint fps = 0;
    char str[16];

    while (true)
    {
        uint32_t start_time = time_us_32();
        gpio_put(LED_PIN, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gears_draw();
        snprintf(str, sizeof(str), "FPS: %d", fps);
        glDrawText((unsigned char *)str, 0, 0, 0x808080);

        u8g2_ClearBuffer(&u8g2);
        for (size_t y = 0; y < 320; y++)
        {
            for (size_t x = 0; x < 480; x++)
            {
                size_t xx = SCR_WIDTH * x / 480;
                size_t yy = SCR_HEIGHT * y / 320;
                uint8_t p = frame_buffer->pbuf[yy * (SCR_WIDTH >> 3) + (xx >> 3)] & (1 << (xx % 8));
                if (p)
                    u8g2_DrawPixel(&u8g2, x, y);
            }
        }
        u8g2_SendBuffer(&u8g2);
        gpio_put(LED_PIN, 1);
        fps = 1000000 / (time_us_32() - start_time);
    }
}
