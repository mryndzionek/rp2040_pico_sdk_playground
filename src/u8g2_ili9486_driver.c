#include "u8g2_ili9486_driver.h"

#include <stdio.h>

#include "pico/stdlib.h"

#include "hardware/pwm.h"
#include "hardware/pio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"

#include "ili9486_lcd_8bit_data.pio.h"

#define BACKLIGHT_PIN (15)

#define LCD_WR_PIN (12)
#define LCD_D0_PIN (2)

#define LCD_RESET_PIN (10)
#define LCD_CS_PIN (11)
#define LCD_DC_PIN (13)

#define LCD_PIO (pio0)
#define LCD_SM (0)

#define DMA_CHANNEL (0)
#define DMA_CHANNEL_MASK (1u << DMA_CHANNEL)

#define RGB_L(_r, _g, _b) (((_r & 0b11111000)) | (((_g >> 2) & 0b111000) >> 3))
#define RGB_H(_r, _g, _b) ((((_g >> 2) & 0b111) << 5) | (_b >> 3))

#define ON_COLOR_L RGB_L(0x00, 0xFF, 0x00)
#define ON_COLOR_H RGB_H(0x00, 0xFF, 0x00)

#define OFF_COLOR_L RGB_L(0x00, 0x00, 0x00)
#define OFF_COLOR_H RGB_H(0x00, 0x00, 0x00)

static const u8x8_display_info_t _display_info =
    {
        /* most of the settings are not required, because this is a serial RS232 printer */

        /* chip_enable_level = */ 0,
        /* chip_disable_level = */ 1,

        /* post_chip_enable_wait_ns = */ 5,
        /* pre_chip_disable_wait_ns = */ 5,
        /* reset_pulse_width_ms = */ 100,
        /* post_reset_wait_ms = */ 100,
        /* sda_setup_time_ns = */ 20,
        /* sck_pulse_width_ns = */ 140,
        /* sck_clock_hz = */ 1000000UL,
        /* spi_mode = */ 0,
        /* i2c_bus_clock_100kHz = */ 4,
        /* data_setup_time_ns = */ 30,
        /* write_pulse_width_ns = */ 40,
        /* tile_width = */ 60,
        /* tile_height = */ 40,
        /* default_x_offset = */ 0,
        /* flipmode_x_offset = */ 0,
        /* pixel_width = */ 480,
        /* pixel_height = */ 320};

static const uint8_t _init_seq[] = {
    U8X8_START_TRANSFER(),

    U8X8_C(0x11),
    U8X8_DLY(100),
    U8X8_CA(0x3A, 0x55),
    U8X8_CA(0xC2, 0x44),
    U8X8_CAAAA(0xC5, 0x00, 0x00, 0x00, 0x00),

    U8X8_CAAA(0xE0, 0x0F, 0x1F, 0x1C),
    U8X8_A8(0x0C, 0x0F, 0x08, 0x48, 0x98, 0x37, 0x0A, 0x13),
    U8X8_A4(0x04, 0x11, 0x0D, 0x00),

    U8X8_CAAA(0xE1, 0x0F, 0x32, 0x2E),
    U8X8_A8(0x0B, 0x0D, 0x05, 0x47, 0x75, 0x37, 0x06, 0x10),
    U8X8_A4(0x03, 0x24, 0x20, 0x00),

    U8X8_C(0x20),
    U8X8_CA(0x36, 0xC8),
    U8X8_C(0x29),
    U8X8_DLY(100),

    U8X8_END_TRANSFER(),

    U8X8_END() /* end of sequence */
};

static uint8_t _gpio_and_delay(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, void *arg_ptr)
{
    switch (msg)
    {
    case U8X8_MSG_GPIO_AND_DELAY_INIT:
    {
        if (pio_can_add_program(LCD_PIO, &ili9486_lcd_8bit_data_program))
        {
            uint offset = pio_add_program(LCD_PIO, &ili9486_lcd_8bit_data_program);
            ili9486_lcd_8bit_data_program_init(LCD_PIO, LCD_SM, offset, LCD_D0_PIN, LCD_WR_PIN);
        }
        else
        {
            printf("Failed to add LCD PIO program\n");
        }

        dma_claim_mask(DMA_CHANNEL_MASK);
        dma_channel_config channel_config = dma_channel_get_default_config(DMA_CHANNEL);
        channel_config_set_dreq(&channel_config, pio_get_dreq(LCD_PIO, LCD_SM, true));
        channel_config_set_transfer_data_size(&channel_config, DMA_SIZE_8);
        channel_config_set_read_increment(&channel_config, true);

        dma_channel_configure(DMA_CHANNEL,
                              &channel_config,
                              &LCD_PIO->txf[LCD_SM],
                              NULL,
                              480,
                              false);

        // irq_set_exclusive_handler(DMA_IRQ_0, dma_complete_handler);
        // dma_channel_set_irq0_enabled(DMA_CHANNEL, true);
        // irq_set_enabled(DMA_IRQ_0, true);

        gpio_init(LCD_RESET_PIN);
        gpio_set_dir(LCD_RESET_PIN, GPIO_OUT);

        gpio_init(LCD_CS_PIN);
        gpio_set_dir(LCD_CS_PIN, GPIO_OUT);

        gpio_init(LCD_DC_PIN);
        gpio_set_dir(LCD_DC_PIN, GPIO_OUT);
    }
    break;

    case U8X8_MSG_GPIO_DC:
        if (arg_int)
        {
            gpio_put(LCD_DC_PIN, 1);
        }
        else
        {
            gpio_put(LCD_DC_PIN, 0);
        }
        break;

    case U8X8_MSG_GPIO_CS:
        if (arg_int)
        {
            gpio_put(LCD_CS_PIN, 1);
        }
        else
        {
            gpio_put(LCD_CS_PIN, 0);
        }
        break;

    case U8X8_MSG_GPIO_RESET:
        if (arg_int)
        {
            gpio_put(LCD_RESET_PIN, 1);
        }
        else
        {
            gpio_put(LCD_RESET_PIN, 0);
        }
        break;

    case U8X8_MSG_DELAY_MILLI:
        sleep_ms(arg_int);
        break;

    case U8X8_MSG_DELAY_10MICRO:
        sleep_us(arg_int);
        break;
    }
    return 0;
}

static uint8_t *u8g2_m_40_60_f(uint8_t *page_cnt)
{
#ifdef U8G2_USE_DYNAMIC_ALLOC
    *page_cnt = 40;
    return 0;
#else
    static uint8_t buf[19200];
    *page_cnt = 40;
    return buf;
#endif
}

static uint8_t _disp_cad(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, void *arg_ptr)
{
    switch (msg)
    {
    case U8X8_MSG_CAD_SEND_CMD:
        u8x8_byte_SetDC(u8x8, 0);
        u8x8_byte_SendByte(u8x8, arg_int);
        break;
    case U8X8_MSG_CAD_SEND_ARG:
        u8x8_byte_SetDC(u8x8, 1);
        u8x8_byte_SendByte(u8x8, arg_int);
        break;
    case U8X8_MSG_CAD_SEND_DATA:
        u8x8_byte_SetDC(u8x8, 1);
        // u8x8_byte_SendBytes(u8x8, arg_int, arg_ptr);
        // break;
        /* fall through */
    case U8X8_MSG_CAD_INIT:
    case U8X8_MSG_CAD_START_TRANSFER:
    case U8X8_MSG_CAD_END_TRANSFER:
        return u8x8->byte_cb(u8x8, msg, arg_int, arg_ptr);
    default:
        return 0;
    }
    return 1;
}

static uint8_t _disp_byte(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, void *arg_ptr)
{
    uint8_t b;
    uint8_t *data;

    switch (msg)
    {
    case U8X8_MSG_BYTE_SEND:
        data = (uint8_t *)arg_ptr;
        while (arg_int > 0)
        {
            b = *data;
            data++;
            arg_int--;
            pio_sm_put_blocking(LCD_PIO, LCD_SM, 0);
            pio_sm_put_blocking(LCD_PIO, LCD_SM, b);
        }
        break;

    case U8X8_MSG_BYTE_INIT:
        /* disable chipselect */
        u8x8_gpio_SetCS(u8x8, u8x8->display_info->chip_disable_level);
        break;

    case U8X8_MSG_BYTE_SET_DC:
        u8x8_gpio_SetDC(u8x8, arg_int);
        break;

    case U8X8_MSG_BYTE_START_TRANSFER:
        u8x8_gpio_SetCS(u8x8, u8x8->display_info->chip_enable_level);
        u8x8->gpio_and_delay_cb(u8x8, U8X8_MSG_DELAY_NANO, u8x8->display_info->post_chip_enable_wait_ns, NULL);
        break;

    case U8X8_MSG_BYTE_END_TRANSFER:
        u8x8->gpio_and_delay_cb(u8x8, U8X8_MSG_DELAY_NANO, u8x8->display_info->pre_chip_disable_wait_ns, NULL);
        u8x8_gpio_SetCS(u8x8, u8x8->display_info->chip_disable_level);
        break;

    default:
        return 0;
    }
    return 1;
}

static uint8_t _disp_handle(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, void *arg_ptr)
{
    switch (msg)
    {
    case U8X8_MSG_DISPLAY_INIT:
        u8x8_d_helper_display_init(u8x8);
        u8x8_cad_SendSequence(u8x8, _init_seq);
        break;

    case U8X8_MSG_DISPLAY_SETUP_MEMORY:
        u8x8_d_helper_display_setup_memory(u8x8, &_display_info);
        break;

    case U8X8_MSG_DISPLAY_DRAW_TILE:
    {
        uint16_t x = ((u8x8_tile_t *)arg_ptr)->x_pos * 8;
        uint16_t y1 = ((u8x8_tile_t *)arg_ptr)->y_pos;
        uint8_t c = ((u8x8_tile_t *)arg_ptr)->cnt;
        uint8_t *ptr = ((u8x8_tile_t *)arg_ptr)->tile_ptr;
        uint16_t y2 = y1 + 1;

        y1 *= 8;
        y2 *= 8;
        y2 -= 1;

        u8x8_cad_StartTransfer(u8x8);

        u8x8_cad_SendCmd(u8x8, 0x2A);
        u8x8_cad_SendArg(u8x8, (y1 >> 8) & 0xFF);
        u8x8_cad_SendArg(u8x8, y1 & 0xff);
        u8x8_cad_SendArg(u8x8, (y2 >> 8) & 0xFF);
        u8x8_cad_SendArg(u8x8, y2 & 0xff);

        u8x8_cad_SendCmd(u8x8, 0x2B);
        u8x8_cad_SendArg(u8x8, (x >> 8) & 0xFF);
        u8x8_cad_SendArg(u8x8, x & 0xff);
        u8x8_cad_SendArg(u8x8, ((c * 8 - 1) >> 8) & 0xFF);
        u8x8_cad_SendArg(u8x8, ((c * 8 - 1) & 0xff));

        u8x8_cad_SendCmd(u8x8, 0x2C);

        u8x8_byte_SetDC(u8x8, 1);
        pio_sm_put_blocking(LCD_PIO, LCD_SM, (c * 8) - 1);

        dma_channel_set_read_addr(DMA_CHANNEL, (void *)ptr, true);
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);

        // Wait for the FIFO to drain
        sleep_us(2);

        u8x8_cad_EndTransfer(u8x8);
    }
    break;

    case U8X8_MSG_DISPLAY_SET_FLIP_MODE:
        break;

    case U8X8_MSG_DISPLAY_SET_POWER_SAVE:
        u8x8_cad_StartTransfer(u8x8);
        if (arg_int == 0)
        {
            u8x8_cad_SendCmd(u8x8, 0x11);
        }
        else
        {
            u8x8_cad_SendCmd(u8x8, 0x10);
        }
        u8x8_cad_EndTransfer(u8x8);
        break;

    default:
        return 0;
    }
    return 1;
}

void u8g2_Setup_ili9486_8bit_480x320_f(u8g2_t *u8g2)
{
    gpio_set_function(BACKLIGHT_PIN, GPIO_FUNC_PWM);
    pwm_config config = pwm_get_default_config();
    pwm_config_set_clkdiv(&config, 4.f);
    uint slice_num = pwm_gpio_to_slice_num(BACKLIGHT_PIN);
    pwm_init(slice_num, &config, true);

    uint8_t tile_buf_height;
    uint8_t *buf;
    u8g2_SetupDisplay(u8g2, _disp_handle, _disp_cad, _disp_byte, _gpio_and_delay);
    buf = u8g2_m_40_60_f(&tile_buf_height);
    u8g2_SetupBuffer(u8g2, buf, tile_buf_height, u8g2_ll_hvline_vertical_top_lsb, U8G2_R0);

    pwm_set_gpio_level(BACKLIGHT_PIN, 0);
}
