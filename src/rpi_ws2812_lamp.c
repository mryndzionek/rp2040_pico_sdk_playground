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
#define NUM_PIXELS (64 * 4)
#define SLEEP_TIMEOUT_MS (60UL * 3UL * 1000UL)
#define TICK_HZ (100)
#define MS_TO_TICKS(_ms) (_ms * TICK_HZ / 1000)
#define DISCR_TIMEOUT_MS (200)

#define FLICKER_HZ (10)
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
    ev_flicker_start,
    ev_flicker_tick,
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
    "ev_flicker_start",
    "ev_flicker_tick",
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
    repeating_timer_t flicker_timer;
    bool flicker;
    alarm_id_t flicker_id;
    uint8_t flicker_index;
} ctx_t;

// static const char flicker_pattern[] = "mmmaaammmaaammmabcdefaaaammmmabcdefmmmaaaa";
static const char flicker_pattern[] = "mmamammmmammamamaaamammma";

static queue_t event_queue;
static critical_section_t lock;

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

static bool flicker_tick_callback(repeating_timer_t *rt)
{
    bool ret;
    event_t event = {.tag = ev_flicker_tick};

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

static int64_t flicker_callback(alarm_id_t id, void *user_data)
{
    bool ret;
    event_t event = {.tag = ev_flicker_start};

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

static void update_leds_rgb(rgb_t rgb)
{
    for (int i = 0; i < NUM_PIXELS; ++i)
    {
        put_pixel(urgb_u32(rgb.R, rgb.G, rgb.B));
    }
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

static void stop_flicker(ctx_t *ctx)
{
    if (ctx->flicker_timer.alarm_id != -1)
    {
        cancel_repeating_timer(&ctx->flicker_timer);
    }
    cancel_alarm(ctx->flicker_id);
    ctx->flicker_id = -1;
    ctx->flicker_index = 0;
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
        .hsv = (hsv_t){0, 1.0, 1.0},
        .state = state_idle,
        .idle_mode = idle_mode_red,
        .tick_timer = {.alarm_id = -1},
        .off_id = -1,
        .ack_repeat = 0,
        .count = 0,
        .flicker_timer = {.alarm_id = -1},
        .flicker = false,
        .flicker_id = -1,
        .flicker_index = 0,
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
    update_leds_rgb(hsv_to_rgb(ctx.hsv));
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
            stop_flicker(&ctx);
            ctx.flicker = false;
            ctx.hsv_tmp = ctx.hsv;
            break;

        case ev_tick:
            debounce(&button_1_deb);
            break;

        case ev_button_1_press:
        case ev_button_1_release:
            discriminate_input(&button_1_discr, event.tag == ev_button_1_press);
            break;

        case ev_button_1_discr_alarm:
            discriminate_alarm(&button_1_discr);
            break;

        case ev_flicker_start:
            if (ctx.flicker)
            {
                ret = add_repeating_timer_us(-1000000 / FLICKER_HZ, flicker_tick_callback, NULL, &ctx.flicker_timer);
                hard_assert(ret);
                ctx.hsv_tmp = ctx.hsv;
            }
            break;

        case ev_flicker_tick:
            if (ctx.state == state_idle)
            {
                if (ctx.flicker)
                {
                    if (ctx.flicker_index == sizeof(flicker_pattern))
                    {
                        stop_flicker(&ctx);
                        ctx.flicker_id = add_alarm_in_ms(1000 * (60 + (rand() % 60)), flicker_callback, NULL, false);
                        hard_assert(ctx.flicker_id > 0);
                        ctx.hsv = ctx.hsv_tmp;
                    }
                    else
                    {
                        ctx.hsv.V = (float)(flicker_pattern[ctx.flicker_index++] - 'a') / 25;
                    }
                    update_leds_rgb(hsv_to_rgb(ctx.hsv));
                }
            }
            else
            {
                // skip current flicker
                stop_flicker(&ctx);
                ctx.flicker_id = add_alarm_in_ms(1000 * (60 + (rand() % 60)), flicker_callback, NULL, false);
                hard_assert(ctx.flicker_id > 0);
            }
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

                        update_leds_rgb((rgb_t){0x00, 0x00, 0x00});

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

                    if (ctx.idle_mode == idle_mode_red)
                    {
                        ctx.idle_mode = idle_mode_off;
                        ctx.hsv = (hsv_t){0, 0.0, 0.0};
                        stop_flicker(&ctx);
                        ctx.flicker = false;
                    }
                    else
                    {
                        ctx.idle_mode = idle_mode_red;
                        ctx.hsv = (hsv_t){0, 1.0, 1.0};
                    }
                    update_leds_rgb(hsv_to_rgb(ctx.hsv));
                    break;

                case 3:
                    ctx.idle_mode = (ctx.idle_mode + 1) % idle_mode_max;
                    switch (ctx.idle_mode)
                    {
                    case idle_mode_off:
                        ctx.hsv = (hsv_t){0, 0.0, 0.0};
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
                    update_leds_rgb(hsv_to_rgb(ctx.hsv));
                    break;

                case 4:
                    ctx.flicker ^= 1;
                    if (ctx.flicker)
                    {
                        srand(get_rand_32());
                        ctx.flicker_id = add_alarm_in_ms(50, flicker_callback, NULL, false);
                        hard_assert(ctx.flicker_id > 0);
                        ctx.hsv_tmp = ctx.hsv;
                    }
                    else
                    {
                        stop_flicker(&ctx);
                    }
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
                    update_leds_rgb(hsv_to_rgb(ctx.hsv));
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
                            update_leds_rgb((rgb_t){0x00, 0x00, 0x00});
                        }
                        else
                        {
                            update_leds_rgb(hsv_to_rgb(ctx.hsv));
                        }
                        ctx.ack_repeat--;
                    }
                    else
                    {
                        update_leds_rgb(hsv_to_rgb(ctx.hsv));
                        ctx.count = 0;
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
                    update_leds_rgb(hsv_to_rgb(ctx.hsv));
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
                    update_leds_rgb(hsv_to_rgb(ctx.hsv));
                    ctx.count = 0;
                }
            }
            break;

            case ev_button_1_long_press:
            case ev_button_1_short_press:
                ctx.hsv = ctx.hsv_tmp;
                update_leds_rgb(hsv_to_rgb(ctx.hsv));
                ctx.state = state_idle;
                t = 0.0f;
                break;

            default:
                break;
            }
        }
        break;

        default:
            break;
        }
    }
}
