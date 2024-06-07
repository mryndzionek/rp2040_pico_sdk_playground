#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hardware/pio.h"
#include "hardware/clocks.h"

#include "pico/stdlib.h"
#include "pico/util/queue.h"
#include "pico/sync.h"
#include "pico/rand.h"

#include "ws2812.pio.h"

#define BUTTON_1_GPIO (15)
#define WS2812_PIN (8)

#define IS_RGBW (false)
#define NUM_PIXELS (64 * 2UL)
#define SLEEP_TIMEOUT_MS (60UL * 3UL * 1000UL)
#define TICK_HZ (200)
#define MS_TO_TICKS(_ms) (_ms * TICK_HZ / 1000)
#define DISCR_TIMEOUT_MS (200)

#define DEFAULT_COLOR ((hsv_t){30, 1.0, 1.0})

#define FIFO_LENGTH (4)

typedef enum
{
    ev_button_1_press = 0,
    ev_button_1_release,
    ev_button_1_short_press,
    ev_button_1_long_press,
    ev_button_1_long_release,
    ev_button_1_discr_alarm,
    ev_tick,
    ev_sleep_alarm,
} event_e;

static const char *const event_to_str[] = {
    "ev_button_1_press",
    "ev_button_1_release",
    "ev_button_1_short_press",
    "ev_button_1_long_press",
    "ev_button_1_long_release",
    "ev_button_1_discr_alarm",
    "ev_tick",
    "ev_sleep_alarm",
};

typedef struct
{
    event_e tag;
    union
    {
        struct
        {
            size_t count;
            bool with_hold;
        } short_press;
    };
} event_t;

typedef enum
{
    state_idle = 0,
    state_color_adjust,
    state_sleep_ack,
    state_brightness_adjust,
    state_dim_out,
    state_plasma,
} state_t;

typedef struct
{
    const event_t on_event;
    const event_t off_event;
    const uint gpio_num;
    size_t count;
} debouncer_t;

typedef struct
{
    event_t short_event;
    const event_t long_event_on;
    const event_t long_event_off;
    const event_t alarm_event;
    alarm_id_t alarm_id;
    bool active;
} discriminator_t;

typedef struct
{
    uint8_t R;
    uint8_t G;
    uint8_t B;
} rgb_t;

typedef struct
{
    uint16_t H;
    float S;
    float V;
} hsv_t;

typedef enum
{
    idle_mode_off = 0,
    idle_mode_default,
    idle_mode_red,
    idle_mode_green,
    idle_mode_blue,
    idle_mode_white,
    idle_mode_max,
} idle_mode_e;

typedef struct
{
    hsv_t hsv;
    hsv_t hsv_tmp;
    state_t state;
    idle_mode_e idle_mode;
    repeating_timer_t tick_timer;
    alarm_id_t off_id;
    uint8_t ack_repeat;
    size_t count;
} ctx_t;

static uint8_t leds[NUM_PIXELS][3];
static uint8_t leds_rem[NUM_PIXELS][3];

void plasma(uint8_t leds[NUM_PIXELS][3]);

static queue_t event_queue;
static critical_section_t lock;

static const uint8_t gamma_map_lo[256] =
    {
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x18, 0x18, 0x19, 0x1f, 0x26, 0x2e, 0x36, 0x40, 0x4a, 0x56, 0x62, 0x70, 0x7f, 0x8e, 0x9f, 0xb1, 0xc4, 0xd8, 0xe8, 0x0, 0x1c, 0x35, 0x50, 0x6c, 0x89, 0xa7, 0xc7, 0xe8, 0x0, 0x2f, 0x55, 0x7c, 0xa4, 0xce, 0x0, 0x27, 0x56, 0x86, 0xb8, 0xe8, 0x20, 0x57, 0x8f, 0xc9, 0x0, 0x43, 0x82, 0xc3, 0x0, 0x4a, 0x90, 0xd9, 0x23, 0x6e, 0xbc, 0x0, 0x5d, 0xb0, 0x0, 0x5d, 0xb6, 0x18, 0x6e, 0xcd, 0x2e, 0x91, 0x0, 0x5d, 0xc6, 0x32, 0x9f, 0x18, 0x80, 0xe8, 0x69, 0xe1, 0x5b, 0xd7, 0x56, 0xd7, 0x59, 0xde, 0x66, 0xe8, 0x7b, 0x0, 0x9a, 0x2c, 0xc1, 0x59, 0xe8, 0x8e, 0x2d, 0xcd, 0x70, 0x18, 0xbe, 0x68, 0x18, 0xc4, 0x76, 0x2a, 0xe0, 0x99, 0x55, 0x18, 0xd4, 0x97, 0x5c, 0x24, 0xe8, 0xbd, 0x8c, 0x5f, 0x34, 0x0, 0xe6, 0xc3, 0xa3, 0x85, 0x6a, 0x52, 0x3c, 0x29, 0x19, 0x0, 0x0, 0x0, 0xe8, 0xe8, 0xe8, 0xe8, 0x0, 0x0, 0x18, 0x1b, 0x2c, 0x40, 0x57, 0x70, 0x8d, 0xac, 0xce, 0xe8, 0x1b, 0x46, 0x74, 0xa5, 0xd8, 0x18, 0x49, 0x85, 0xc5, 0x0, 0x4d, 0x96, 0xe1, 0x30, 0x81, 0xd6, 0x2e, 0x89, 0xe7, 0x48, 0xac, 0x18, 0x7d, 0xe8, 0x5b, 0xcf, 0x46, 0xc0, 0x3d, 0xbd, 0x41, 0xc7, 0x51, 0xde, 0x6f, 0x0, 0x99, 0x33, 0xd0, 0x71, 0x18, 0xbb, 0x66, 0x18, 0xc4, 0x79, 0x30, 0xe8, 0xa9, 0x6b, 0x30, 0x0, 0xc4, 0x93, 0x65, 0x3b, 0x18, 0xe8, 0xd1, 0xb4, 0x9b, 0x85, 0x73, 0x64, 0x59, 0x51, 0x4d, 0x4c, 0x4e, 0x54, 0x5e, 0x6b, 0x7c, 0x90, 0xa8, 0xc3, 0xe2, 0x0, 0x2b, 0x54, 0x81, 0xb2, 0xe6, 0x1f, 0x5a, 0x99, 0xdc, 0x23, 0x6d, 0xbb, 0x0, 0x62, 0xba, 0x18, 0x77, 0xdb, 0x43, 0xae, 0x1e, 0x90, 0x0, 0x81, 0x0};

static const uint8_t gamma_map_hi[256] =
    {
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xa, 0xa, 0xa, 0xb, 0xb, 0xc, 0xc, 0xc, 0xd, 0xd, 0xe, 0xe, 0xe, 0xf, 0xf, 0x10, 0x10, 0x11, 0x11, 0x12, 0x12, 0x13, 0x13, 0x14, 0x15, 0x15, 0x16, 0x16, 0x17, 0x17, 0x18, 0x19, 0x19, 0x1a, 0x1b, 0x1b, 0x1c, 0x1d, 0x1d, 0x1e, 0x1f, 0x1f, 0x20, 0x21, 0x22, 0x22, 0x23, 0x24, 0x25, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x35, 0x36, 0x37, 0x38, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4d, 0x4e, 0x4f, 0x50, 0x52, 0x53, 0x54, 0x55, 0x57, 0x58, 0x59, 0x5b, 0x5c, 0x5d, 0x5f, 0x60, 0x62, 0x63, 0x64, 0x66, 0x67, 0x69, 0x6a, 0x6c, 0x6d, 0x6f, 0x70, 0x72, 0x73, 0x75, 0x77, 0x78, 0x7a, 0x7b, 0x7d, 0x7f, 0x80, 0x82, 0x84, 0x85, 0x87, 0x89, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x93, 0x95, 0x97, 0x99, 0x9b, 0x9c, 0x9e, 0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe, 0xc0, 0xc2, 0xc5, 0xc7, 0xc9, 0xcb, 0xcd, 0xcf, 0xd2, 0xd4, 0xd6, 0xd8, 0xdb, 0xdd, 0xdf, 0xe2, 0xe4, 0xe6, 0xe9, 0xeb, 0xed, 0xf0, 0xf2, 0xf5, 0xf7, 0xfa, 0xfc, 0xff};

static inline uint8_t apply_gamma(uint8_t v, uint8_t *rem)
{
    uint8_t g = gamma_map_lo[v];
    uint8_t data = g;
    g = gamma_map_hi[v];

    if (((uint16_t)(*rem) + data) > 255)
    {
        data = g + 1;
    }
    else
    {
        data = g;
    }

    *rem += data;

    return data;
}

static rgb_t hsv_to_rgb(hsv_t hsv)
{
    float r = 0, g = 0, b = 0;

    if (hsv.S == 0)
    {
        r = hsv.V;
        g = hsv.V;
        b = hsv.V;
    }
    else
    {
        int i;
        float f, p, q, t, H;

        hsv.H %= 360;
        if (hsv.H == 360)
            H = 0.0;
        else
            H = hsv.H / 60.0;

        i = (int)trunc(H);
        f = H - i;

        p = hsv.V * (1.0 - hsv.S);
        q = hsv.V * (1.0 - (hsv.S * f));
        t = hsv.V * (1.0 - (hsv.S * (1.0 - f)));

        switch (i)
        {
        case 0:
            r = hsv.V;
            g = t;
            b = p;
            break;

        case 1:
            r = q;
            g = hsv.V;
            b = p;
            break;

        case 2:
            r = p;
            g = hsv.V;
            b = t;
            break;

        case 3:
            r = p;
            g = q;
            b = hsv.V;
            break;

        case 4:
            r = t;
            g = p;
            b = hsv.V;
            break;

        default:
            r = hsv.V;
            g = p;
            b = q;
            break;
        }
    }

    return (rgb_t){
        .R = r * 255,
        .G = g * 255,
        .B = b * 255,
    };
}

static inline void put_pixel(uint32_t pixel_grb)
{
    pio_sm_put_blocking(pio0, 0, pixel_grb << 8u);
}

static inline uint32_t urgb_u32(uint8_t r, uint8_t g, uint8_t b)
{
    return ((uint32_t)(r) << 8) |
           ((uint32_t)(g) << 16) |
           (uint32_t)(b);
}

static bool led_tick_callback(repeating_timer_t *rt)
{
    bool ret;
    event_t event = {.tag = ev_tick};

    ret = queue_try_add(&event_queue, &event);
    hard_assert(ret);
    return true; // keep repeating
}

static int64_t alarm_callback(alarm_id_t id, void *user_data)
{
    bool ret;
    event_t event = {.tag = ev_sleep_alarm};

    ret = queue_try_add(&event_queue, &event);
    hard_assert(ret);
    return 0;
}

static int64_t discr_callback(alarm_id_t id, void *user_data)
{
    bool ret;
    discriminator_t *discr = user_data;

    ret = queue_try_add(&event_queue, &discr->alarm_event);
    hard_assert(ret);
    return 0;
}

static void _update_leds_rgb(rgb_t rgb)
{
    static uint8_t tmp[3] = {0};

    for (int i = 0; i < NUM_PIXELS; ++i)
    {
        put_pixel(urgb_u32(apply_gamma(rgb.R, &tmp[0]), apply_gamma(rgb.G, &tmp[1]), apply_gamma(rgb.B, &tmp[2])));
    }
    pio_sm_drain_tx_fifo(pio0, 0);
}

static void debounce(debouncer_t *deb)
{
    bool ret;

    if (gpio_get(deb->gpio_num))
    {
        if (deb->count < 4)
        {
            deb->count++;
            if (deb->count == 4)
            {
                ret = queue_try_add(&event_queue, &deb->on_event);
                hard_assert(ret);
            }
        }
    }
    else
    {
        if (deb->count > 0)
        {
            deb->count--;
            if (deb->count == 0)
            {
                ret = queue_try_add(&event_queue, &deb->off_event);
                hard_assert(ret);
            }
        }
    }
}

static void discriminate_input(discriminator_t *discr, bool input)
{
    critical_section_enter_blocking(&lock);
    discr->short_event.short_press.with_hold = input;

    if (discr->active)
    {
        bool ret;

        if (!input)
        {
            if (discr->alarm_id == -1)
            {
                ret = queue_try_add(&event_queue, &discr->long_event_off);
                hard_assert(ret);
                discr->active = false;
                discr->short_event.short_press.count = 0;
            }
            else
            {
                discr->short_event.short_press.count++;
                cancel_alarm(discr->alarm_id);
                discr->alarm_id = add_alarm_in_ms(DISCR_TIMEOUT_MS, discr_callback, discr, false);
                hard_assert(discr->alarm_id > 0);
            }
        }
        else
        {
            if (discr->alarm_id > 0)
            {
                if (discr->alarm_id != -1)
                {
                    cancel_alarm(discr->alarm_id);
                    discr->alarm_id = add_alarm_in_ms(DISCR_TIMEOUT_MS, discr_callback, discr, false);
                    hard_assert(discr->alarm_id > 0);
                }
            }
        }
    }
    else
    {
        if (input)
        {
            hard_assert(discr->alarm_id == -1);
            discr->alarm_id = add_alarm_in_ms(DISCR_TIMEOUT_MS, discr_callback, discr, false);
            hard_assert(discr->alarm_id > 0);
            discr->active = true;
        }
    }
    critical_section_exit(&lock);
}

static void discriminate_alarm(discriminator_t *discr)
{
    bool ret;

    critical_section_enter_blocking(&lock);
    hard_assert(discr->active);
    discr->alarm_id = -1;

    if (discr->short_event.short_press.count == 0)
    {
        ret = queue_try_add(&event_queue, &discr->long_event_on);
        hard_assert(ret);
    }
    else
    {
        ret = queue_try_add(&event_queue, &discr->short_event);
        hard_assert(ret);
        if (!discr->short_event.short_press.with_hold)
        {
            discr->active = false;
        }
        discr->short_event.short_press.count = 0;
    }
    critical_section_exit(&lock);
}

__attribute__((unused)) static void debug_event(event_t const *const ev)
{
    if (ev->tag == ev_button_1_short_press)
    {
        printf("Event: %s, count: %d %d\n", event_to_str[ev->tag],
               ev->short_press.count,
               ev->short_press.with_hold);
    }
    else
    {
        printf("Event: %s\n", event_to_str[ev->tag]);
    }
}

int main()
{
    stdio_init_all();
    set_sys_clock_khz(48000, true);
    setup_default_uart();
    printf("WS2812 lamp\n");
    printf("LED data pin: %d\n", WS2812_PIN);
    printf("Button pin: %d\n", BUTTON_1_GPIO);

    gpio_init(BUTTON_1_GPIO);
    gpio_set_dir(BUTTON_1_GPIO, GPIO_IN);

    queue_init(&event_queue, sizeof(event_t), FIFO_LENGTH);
    critical_section_init(&lock);

    PIO pio = pio0;
    int sm = 0;
    bool ret;
    event_t event;

    ctx_t ctx = {
        .hsv = DEFAULT_COLOR,
        .state = state_idle,
        .idle_mode = idle_mode_default,
        .tick_timer = {.alarm_id = -1},
        .off_id = -1,
        .ack_repeat = 0,
        .count = 0,
    };

    debouncer_t button_1_deb = {
        .off_event = {.tag = ev_button_1_release},
        .on_event = {.tag = ev_button_1_press},
        .count = 0,
        .gpio_num = BUTTON_1_GPIO,
    };

    discriminator_t button_1_discr = {
        .short_event = {.tag = ev_button_1_short_press,
                        .short_press.count = 0,
                        .short_press.with_hold = false},
        .long_event_on = {.tag = ev_button_1_long_press},
        .long_event_off = {.tag = ev_button_1_long_release},
        .alarm_event = {.tag = ev_button_1_discr_alarm},
        .alarm_id = -1,
        .active = false,
    };

    uint offset = pio_add_program(pio, &ws2812_program);
    ws2812_program_init(pio, sm, offset, WS2812_PIN, 800000, IS_RGBW);

    ctx.hsv_tmp = ctx.hsv;
    _update_leds_rgb(hsv_to_rgb(ctx.hsv));
    ret = add_repeating_timer_us(-1000000 / TICK_HZ, led_tick_callback, NULL, &ctx.tick_timer);
    hard_assert(ret);

    while (1)
    {
        queue_remove_blocking(&event_queue, &event);

        // if (event.tag != ev_tick)
        // {
        //     debug_event(&event);
        // }

        switch (event.tag)
        {
        case ev_sleep_alarm:
            ctx.state = state_dim_out;
            ctx.hsv_tmp = ctx.hsv;
            break;

        case ev_tick:
            debounce(&button_1_deb);
            if (ctx.state != state_plasma)
            {
                _update_leds_rgb(hsv_to_rgb(ctx.hsv));
            }
            else
            {
                plasma(leds);
                for (int i = 0; i < NUM_PIXELS; ++i)
                {
                    uint8_t r = apply_gamma(leds[i][0], &leds_rem[i][0]);
                    uint8_t g = apply_gamma(leds[i][1], &leds_rem[i][1]);
                    uint8_t b = apply_gamma(leds[i][2], &leds_rem[i][2]);
                    put_pixel(urgb_u32(r, g, b));
                }
                pio_sm_drain_tx_fifo(pio0, 0);
            }
            break;

        case ev_button_1_press:
        case ev_button_1_release:
            discriminate_input(&button_1_discr, event.tag == ev_button_1_press);
            break;

        case ev_button_1_discr_alarm:
            discriminate_alarm(&button_1_discr);
            break;

        default:
            break;
        }

        switch (ctx.state)
        {
        case state_idle:
            switch (event.tag)
            {
            case ev_button_1_long_press:
                ctx.state = state_brightness_adjust;
                break;

            case ev_button_1_short_press:
                switch (event.short_press.count)
                {
                case 1:
                    if (event.short_press.with_hold)
                    {
                        ctx.hsv.S = 1.0;
                        ctx.state = state_color_adjust;
                    }
                    else if (ctx.idle_mode != idle_mode_off)
                    {
                        ctx.state = state_sleep_ack;
                        ctx.count = 0;

                        ctx.hsv_tmp = ctx.hsv;
                        ctx.hsv = (hsv_t){0, 0.0, 0.0};

                        if (ctx.off_id >= 0)
                        {
                            cancel_alarm(ctx.off_id);
                            ctx.off_id = -1;
                            ctx.ack_repeat = 2;
                        }
                        else
                        {
                            ctx.off_id = add_alarm_in_ms(SLEEP_TIMEOUT_MS, alarm_callback, NULL, false);
                            hard_assert(ret);
                            ctx.ack_repeat = 0;
                        }
                    }
                    break;

                case 2:
                    if (ctx.off_id >= 0)
                    {
                        cancel_alarm(ctx.off_id);
                        ctx.off_id = -1;
                    }

                    if (ctx.idle_mode != idle_mode_off)
                    {
                        ctx.idle_mode = idle_mode_off;
                        ctx.hsv = (hsv_t){0, 0.0, 0.0};
                    }
                    else
                    {
                        ctx.idle_mode = idle_mode_default;
                        ctx.hsv = DEFAULT_COLOR;
                    }
                    break;

                case 3:
                    ctx.idle_mode = (ctx.idle_mode + 1) % idle_mode_max;
                    switch (ctx.idle_mode)
                    {
                    case idle_mode_off:
                        ctx.hsv = (hsv_t){0, 0.0, 0.0};
                        break;

                    case idle_mode_default:
                        ctx.hsv = DEFAULT_COLOR;
                        break;

                    case idle_mode_red:
                        ctx.hsv = (hsv_t){0, 1.0, 1.0};
                        break;

                    case idle_mode_blue:
                        ctx.hsv = (hsv_t){120, 1.0, 1.0};
                        break;

                    case idle_mode_green:
                        ctx.hsv = (hsv_t){240, 1.0, 1.0};
                        break;

                    case idle_mode_white:
                        ctx.hsv = (hsv_t){0, 0.0, 1.0};
                        break;

                    case idle_mode_max:
                        hard_assert(1);
                        break;
                    }
                    break;

                case 4:
                    ctx.state = state_plasma;
                    break;

                default:
                    break;
                }
                break;

            default:
                break;
            }
            break;

        case state_color_adjust:
            switch (event.tag)
            {
            case ev_tick:
            {
                ctx.count++;
                if (ctx.count == MS_TO_TICKS(100))
                {
                    if ((ctx.hsv.H < 15) || (ctx.hsv.H > 350))
                    {
                        ctx.hsv.H += 1;
                    }
                    else
                    {
                        ctx.hsv.H += 5;
                    }
                    ctx.hsv.H %= 360;
                    ctx.count = 0;
                }
            }
            break;

            case ev_button_1_long_release:
                ctx.state = state_idle;
                ctx.count = 0;
                break;

            default:
                break;
            }
            break;

        case state_sleep_ack:
        {
            switch (event.tag)
            {
            case ev_tick:
            {
                ctx.count++;
                if (ctx.count == MS_TO_TICKS(100))
                {
                    ctx.count = 0;
                    if (ctx.ack_repeat)
                    {
                        if ((ctx.ack_repeat % 2) == 1)
                        {
                            ctx.hsv = (hsv_t){0, 0.0, 0.0};
                        }
                        else
                        {
                            ctx.hsv = ctx.hsv_tmp;
                        }
                        ctx.ack_repeat--;
                    }
                    else
                    {
                        ctx.count = 0;
                        ctx.hsv = ctx.hsv_tmp;
                        ctx.state = state_idle;
                    }
                }
            }
            break;

            default:
                break;
            }
        }
        break;

        case state_brightness_adjust:
            switch (event.tag)
            {
            case ev_tick:
            {
                ctx.count++;
                if (ctx.count == MS_TO_TICKS(100))
                {
                    static double increase = 0.05;
                    ctx.hsv.V += increase;
                    if (ctx.hsv.V > 1.0)
                    {
                        ctx.hsv.V = 1.0;
                        increase = -increase;
                    }
                    else if (ctx.hsv.V < 0.0)
                    {
                        ctx.hsv.V = 0.0;
                        increase = -increase;
                    }
                    ctx.count = 0;
                }
            }
            break;

            case ev_button_1_long_release:
                ctx.state = state_idle;
                ctx.count = 0;
                break;

            default:
                break;
            }
            break;

        case state_dim_out:
        {
            static float t = 0.0f;
            switch (event.tag)
            {
            case ev_tick:
            {
                ctx.count++;
                if (ctx.count == MS_TO_TICKS(100))
                {
                    ctx.hsv.V = expf(-t / 8.0f);
                    t += 0.1;
                    if (ctx.hsv.V < 0.004)
                    {
                        ctx.hsv = (hsv_t){0, 0.0, 0.0};
                        ctx.off_id = -1;
                        ctx.idle_mode = idle_mode_off;
                        ctx.state = state_idle;
                        t = 0.0f;
                    }
                    ctx.count = 0;
                }
            }
            break;

            case ev_button_1_long_press:
            case ev_button_1_short_press:
                ctx.hsv = ctx.hsv_tmp;
                ctx.state = state_idle;
                t = 0.0f;
                break;

            default:
                break;
            }
        }
        break;

        case state_plasma:
            switch (event.tag)
            {
            case ev_button_1_short_press:
            case ev_button_1_long_press:
                ctx.state = state_idle;
                break;

            default:
                break;
            }
            break;

        default:
            break;
        }
    }
}
