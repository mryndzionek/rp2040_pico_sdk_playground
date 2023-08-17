#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "pico/util/queue.h"

#include "ws2812.pio.h"

#define BUTTON_1_GPIO (16)
#define BUTTON_2_GPIO (17)

#define IS_RGBW false
#define NUM_PIXELS (64 * 4)
#define SLEEP_TIMEOUT_MS (60UL * 3UL * 1000UL)
#define TICK_HZ (100)
#define MS_TO_TICKS(_ms) (_ms * TICK_HZ / 1000)
#define DISCR_TIMEOUT_MS (200)

#define WS2812_PIN (2)

#define FIFO_LENGTH (4)

typedef enum
{
    ev_button_1_press = 0,
    ev_button_1_release,
    ev_button_1_short_press,
    ev_button_1_long_press,
    ev_button_1_long_release,
    ev_button_1_discr_alarm,
    ev_button_2_press,
    ev_button_2_release,
    ev_button_2_short_press,
    ev_button_2_long_press,
    ev_button_2_long_release,
    ev_button_2_discr_alarm,
    ev_tick,
    ev_sleep_alarm,
} event_e;

typedef struct
{
    event_e tag;
    union
    {
        struct
        {
            size_t count;
        } short_press;
    };
} event_t;

typedef enum
{
    state_idle = 0,
    state_color_adjust,
    state_brightness_pre_adjust,
    state_sleep_ack,
    state_brightness_adjust,
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
    double S;
    double V;
} hsv_t;

typedef enum
{
    idle_mode_hsv = 0,
    idle_mode_off,
    idle_mode_red,
    idle_mode_blue,
    idle_mode_green,
    idle_mode_white,
    idle_mode_max,
} idle_mode_e;

static queue_t event_queue;

static rgb_t HSVToRGB(hsv_t hsv)
{
    double r = 0, g = 0, b = 0;

    if (hsv.S == 0)
    {
        r = hsv.V;
        g = hsv.V;
        b = hsv.V;
    }
    else
    {
        int i;
        double f, p, q, t, H;

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

    rgb_t rgb = {
        .R = r * 255,
        .G = g * 255,
        .B = b * 255,
    };

    return rgb;
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

    if (!gpio_get(deb->gpio_num))
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
                cancel_alarm(discr->alarm_id);
                discr->alarm_id = add_alarm_in_ms(DISCR_TIMEOUT_MS, discr_callback, discr, false);
                hard_assert(discr->alarm_id > 0);
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
}

static void discriminate_alarm(discriminator_t *discr)
{
    bool ret;

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
        discr->active = false;
        discr->short_event.short_press.count = 0;
    }
}

int main()
{
    bool ret;

    stdio_init_all();
    set_sys_clock_khz(48000, true);
    setup_default_uart();
    printf("WS2812 lamp\n");
    printf("LED data pin: %d\n", WS2812_PIN);
    printf("Button pins: %d %d\n", BUTTON_1_GPIO, BUTTON_2_GPIO);

    queue_init(&event_queue, sizeof(event_t), FIFO_LENGTH);
    gpio_pull_up(BUTTON_1_GPIO);
    gpio_pull_up(BUTTON_2_GPIO);

    PIO pio = pio0;
    int sm = 0;
    uint offset = pio_add_program(pio, &ws2812_program);

    ws2812_program_init(pio, sm, offset, WS2812_PIN, 800000, IS_RGBW);

    hsv_t hsv = (hsv_t){0, 1.0, 0.5};
    rgb_t rgb = HSVToRGB(hsv);
    update_leds_rgb(rgb);

    event_t event;
    repeating_timer_t tick_timer;
    state_t state = state_idle;
    alarm_id_t alarm_id = -1;
    uint8_t ack_repeat = 0;
    size_t count = 0;
    idle_mode_e idle_mode = idle_mode_hsv;

    debouncer_t button_1_deb = {
        .off_event = {.tag = ev_button_1_release},
        .on_event = {.tag = ev_button_1_press},
        .count = 0,
        .gpio_num = BUTTON_1_GPIO,
    };

    discriminator_t button_1_discr = {
        .short_event = {.tag = ev_button_1_short_press, .short_press.count = 0},
        .long_event_on = {.tag = ev_button_1_long_press},
        .long_event_off = {.tag = ev_button_1_long_release},
        .alarm_event = {.tag = ev_button_1_discr_alarm},
        .alarm_id = -1,
        .active = false,
    };

    debouncer_t button_2_deb = {
        .off_event = {.tag = ev_button_2_release},
        .on_event = {.tag = ev_button_2_press},
        .count = 0,
        .gpio_num = BUTTON_2_GPIO,
    };

    discriminator_t button_2_discr = {
        .short_event = {.tag = ev_button_2_short_press, .short_press.count = 0},
        .long_event_on = {.tag = ev_button_2_long_press},
        .long_event_off = {.tag = ev_button_2_long_release},
        .alarm_event = {.tag = ev_button_2_discr_alarm},
        .alarm_id = -1,
        .active = false,
    };

    ret = add_repeating_timer_us(-1000000 / TICK_HZ, led_tick_callback, NULL, &tick_timer);
    hard_assert(ret);

    while (1)
    {
        queue_remove_blocking(&event_queue, &event);
        switch (event.tag)
        {
        case ev_sleep_alarm:
            rgb = (rgb_t){0x00, 0x00, 0x00};
            update_leds_rgb(rgb);
            alarm_id = -1;
            break;

        case ev_tick:
            debounce(&button_1_deb);
            debounce(&button_2_deb);
            break;

        case ev_button_1_press:
        case ev_button_1_release:
            discriminate_input(&button_1_discr, event.tag == ev_button_1_press);
            break;

        case ev_button_2_press:
        case ev_button_2_release:
            discriminate_input(&button_2_discr, event.tag == ev_button_2_press);
            break;

        case ev_button_1_discr_alarm:
            discriminate_alarm(&button_1_discr);
            break;

        case ev_button_2_discr_alarm:
            discriminate_alarm(&button_2_discr);
            break;

        default:
            break;
        }

        switch (state)
        {
        case state_idle:
            switch (event.tag)
            {
            case ev_button_1_long_press:
                state = state_color_adjust;
                idle_mode = idle_mode_hsv;
                break;

            case ev_button_1_short_press:
                if (event.short_press.count == 2)
                {
                    state = state_sleep_ack;
                    count = 0;

                    update_leds_rgb((rgb_t){0x00, 0x00, 0x00});

                    if (alarm_id >= 0)
                    {
                        cancel_alarm(alarm_id);
                        alarm_id = -1;
                        ack_repeat = 2;
                    }
                    else
                    {
                        alarm_id = add_alarm_in_ms(SLEEP_TIMEOUT_MS, alarm_callback, NULL, false);
                        hard_assert(ret);
                        ack_repeat = 0;
                    }
                }
                break;

            case ev_button_2_long_press:
                state = state_brightness_adjust;
                idle_mode = idle_mode_hsv;
                break;

            case ev_button_2_short_press:
                if (event.short_press.count == 2)
                {
                    idle_mode = (idle_mode + 1) % idle_mode_max;
                    switch (idle_mode)
                    {
                    case idle_mode_hsv:
                        rgb = HSVToRGB(hsv);
                        break;

                    case idle_mode_off:
                        rgb = (rgb_t){0x00, 0x00, 0x00};
                        break;

                    case idle_mode_red:
                        rgb = (rgb_t){0xFF, 0x00, 0x00};
                        break;

                    case idle_mode_blue:
                        rgb = (rgb_t){0x00, 0x00, 0xFF};
                        break;

                    case idle_mode_green:
                        rgb = (rgb_t){0x00, 0xFF, 0x00};
                        break;

                    case idle_mode_white:
                        rgb = (rgb_t){0xFF, 0xFF, 0xFF};
                        break;

                    case idle_mode_max:
                        hard_assert(1);
                        break;
                    }
                    update_leds_rgb(rgb);
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
                count++;
                if (count == MS_TO_TICKS(100))
                {
                    if ((hsv.H < 15) || (hsv.H > 350))
                    {
                        hsv.H += 1;
                    }
                    else
                    {
                        hsv.H += 5;
                    }
                    hsv.H %= 360;
                    update_leds_rgb(HSVToRGB(hsv));
                    count = 0;
                }
            }
            break;

            case ev_button_1_long_release:
                state = state_idle;
                count = 0;
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
                count++;
                if (count == MS_TO_TICKS(100))
                {
                    count = 0;
                    if (ack_repeat)
                    {
                        if ((ack_repeat % 2) == 1)
                        {
                            update_leds_rgb((rgb_t){0x00, 0x00, 0x00});
                        }
                        else
                        {
                            if (idle_mode_hsv == idle_mode)
                            {
                                update_leds_rgb(HSVToRGB(hsv));
                            }
                            else
                            {
                                update_leds_rgb(rgb);
                            }
                        }
                        ack_repeat--;
                    }
                    else
                    {
                        if (idle_mode_hsv == idle_mode)
                        {
                            update_leds_rgb(HSVToRGB(hsv));
                        }
                        else
                        {
                            update_leds_rgb(rgb);
                        }
                        count = 0;
                        state = state_idle;
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
                count++;
                if (count == MS_TO_TICKS(100))
                {
                    static double increase = 0.05;
                    hsv.V += increase;
                    if (hsv.V > 1.0)
                    {
                        hsv.V = 1.0;
                        increase = -increase;
                    }
                    else if (hsv.V < 0.0)
                    {
                        hsv.V = 0.0;
                        increase = -increase;
                    }
                    update_leds_rgb(HSVToRGB(hsv));
                    count = 0;
                }
            }
            break;

            case ev_button_2_long_release:
                count = 0;
                state = state_idle;
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
