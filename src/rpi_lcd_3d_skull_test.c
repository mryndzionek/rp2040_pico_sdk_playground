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

#include "skull.h"

#define SCR_WIDTH (320)
#define SCR_HEIGHT (213)

extern uint8_t *u8g2_fbuf;
static u8g2_t u8g2;
static GLint skull;

static void skull_init_scene(void)
{
    static GLfloat pos[4] = {5, 5, 10, 0.0};

    static GLfloat red[4] = {1.0, 0.0, 0.0, 0.0};
    static GLfloat white[4] = {1.0, 1.0, 1.0, 0.0};
    static GLfloat shininess = 5;

    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glEnable(GL_CULL_FACE);

    glEnable(GL_LIGHT0);

    glDisable(GL_POLYGON_STIPPLE);
    glPointSize(10.0f);
    glTextSize(GL_TEXT_SIZE24x24);

    skull = glGenLists(1);
    glNewList(skull, GL_COMPILE);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT, GL_SPECULAR, white);
    glMaterialfv(GL_FRONT, GL_SHININESS, &shininess);
    glColor3fv(red);

    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < 3 * SKULL_NUM_TRIANGLES; i += 3)
    {
        glNormal3f(skull_tnormals[i], skull_tnormals[i + 1], skull_tnormals[i + 2]);
        glVertex3f(skull_triangles[i], skull_triangles[i + 1], skull_triangles[i + 2]);
    }
    glEnd();

    glBegin(GL_QUADS);
    for (size_t i = 0; i < 3 * SKULL_NUM_QUADS; i += 3)
    {
        glNormal3f(skull_qnormals[i], skull_qnormals[i + 1], skull_qnormals[i + 2]);
        glVertex3f(skull_quads[i], skull_quads[i + 1], skull_quads[i + 2]);
    }
    glEnd();

    glEndList();
}

static void skull_draw(void)
{
    static GLfloat view_roty = 30.0;
    static GLfloat angle = 0.0;

    angle += 8.0;
    glPushMatrix();
    glRotatef(view_roty - angle, 0.0, 1.0, 0.0);
    glTranslatef(0.0, -0.25, 0.0);

    glCallList(skull);
    glPopMatrix();
}

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
    glFrustum(-0.2, 0.2, -h / 5, h / 5, 6.0, 80.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -55.0);

    skull_init_scene();

    glSetEnableSpecular(GL_FALSE);
    ZB_setDitheringMap(frame_buffer, 3);

    uint fps = 0;
    char str[16];

    while (true)
    {
        uint32_t start_time = time_us_32();
        gpio_put(LED_PIN, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        skull_draw();
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
