#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <math.h>

#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/pwm.h"
#include "pico/stdlib.h"

#include "inmp441.pio.h"

#define SPEAKER_OUTPUT
#define INMP441_PIN_SD (16)
#define INMP441_PIN_SCK (14)
#define AUDIO_PIN (18)
#define BUTTON_PIN (13)

#define CPU_CLOCK (128000000UL)
#define BASE_RATE_HZ (20)
#define SAMPLERATE (16000UL)
#define CHUNK_READ_SIZE (SAMPLERATE / BASE_RATE_HZ)
#define INTERP_RATE (16)
#define PWM_MAX (CPU_CLOCK / (SAMPLERATE * INTERP_RATE))
#define VIBRATO_HZ (10)
#define OSC_BUFFER_LEN (SAMPLERATE / VIBRATO_HZ)

typedef struct
{
    bool fButton0;
    float fHslider0;
    float fRec0[2];
    int IOTA0;
    int16_t fVec0[32768];
    int16_t fRec1[8192];
    int fSampleRate;
} pitchshifter_t;

typedef enum
{
    SQUELCH_ENABLED = 0,
    SQUELCH_RISE,
    SQUELCH_SIGNALHI,
    SQUELCH_FALL,
    SQUELCH_SIGNALLO,
    SQUELCH_TIMEOUT,
    SQUELCH_DISABLED,
} squelch_mode_e;

static inline int max(int a, int b) { return ((a) > (b) ? a : b); }
static inline int min(int a, int b) { return ((a) < (b) ? a : b); }

static void interp_bresenham(int16_t y1, int16_t y2, uint16_t nx, int16_t *ny)
{
    const int16_t x1 = 0;
    const int16_t x2 = nx - 1;
    const int16_t dx = x2 - x1;

    int16_t d, dy, ai, bi, yi;
    int16_t x = x1;
    int16_t y = y1;

    if (y1 < y2)
    {
        yi = 1;
        dy = y2 - y1;
    }
    else
    {
        yi = -1;
        dy = y1 - y2;
    }

    ny[x] = y;

    if (dx > dy)
    {
        ai = (dy - dx) * 2;
        bi = dy * 2;
        d = bi - dx;

        while (x != x2)
        {
            if (d >= 0)
            {
                x++;
                y += yi;
                d += ai;
            }
            else
            {
                d += bi;
                x++;
            }
            ny[x] = y;
        }
    }
    else
    {
        ai = (dx - dy) * 2;
        bi = dx * 2;
        d = bi - dy;

        while (y != y2)
        {
            if (d >= 0)
            {
                x++;
                y += yi;
                d += ai;
            }
            else
            {
                d += bi;
                y += yi;
            }
            ny[x] = y;
        }
    }
}

pitchshifter_t *pitchshifter_new(int sample_rate)
{
    pitchshifter_t *dsp = (pitchshifter_t *)calloc(1, sizeof(pitchshifter_t));
    if (!dsp)
    {
        return NULL;
    }
    dsp->fSampleRate = sample_rate;
    return dsp;
}

void pitchshifter_destroy(pitchshifter_t **dsp_p)
{
    assert(dsp_p);
    if (*dsp_p)
    {
        pitchshifter_t *dsp = *dsp_p;
        free(dsp);
        dsp_p = NULL;
    }
}

void pitchshifter_reset(pitchshifter_t *dsp)
{
    /* C99 loop */
    {
        int l0;
        for (l0 = 0; l0 < 2; l0 = l0 + 1)
        {
            dsp->fRec0[l0] = 0.0f;
        }
    }
    dsp->IOTA0 = 0;
    /* C99 loop */
    {
        int l1;
        for (l1 = 0; l1 < 32768; l1 = l1 + 1)
        {
            dsp->fVec0[l1] = 0;
        }
    }
    /* C99 loop */
    {
        int l2;
        for (l2 = 0; l2 < 8192; l2 = l2 + 1)
        {
            dsp->fRec1[l2] = 0;
        }
    }
}

void pitchshifter_process(pitchshifter_t *dsp, int count, int16_t *__restrict__ inputs, int16_t *__restrict__ outputs)
{
    int iSlow0 = dsp->fButton0;
    float fSlow1 = powf(2.0f, 0.083333336f * (float)(dsp->fHslider0));
    /* C99 loop */
    {
        int i0;
        for (i0 = 0; i0 < count; i0 = i0 + 1)
        {
            dsp->fRec0[0] = fmodf(dsp->fRec0[1] + 1001.0f - fSlow1, 1e+03f);
            float fTemp0 = fminf(0.002f * dsp->fRec0[0], 1.0f);
            float fTemp1 = dsp->fRec0[0] + 1e+03f;
            float fTemp2 = floorf(fTemp1);
            dsp->fVec0[dsp->IOTA0 & 32767] = inputs[i0];
            int iTemp4 = (int)(fTemp1);
            int iTemp5 = (int)(dsp->fRec0[0]);
            float fTemp6 = floorf(dsp->fRec0[0]);
            float fTemp7 = (dsp->fVec0[(dsp->IOTA0 - min(16385, max(0, iTemp5))) & 32767] * (fTemp6 + (1.0f - dsp->fRec0[0])) + (dsp->fRec0[0] - fTemp6) * dsp->fVec0[(dsp->IOTA0 - min(16385, max(0, iTemp5 + 1))) & 32767]) * fTemp0 + (dsp->fVec0[(dsp->IOTA0 - min(16385, max(0, iTemp4))) & 32767] * (fTemp2 + (-999.0f - dsp->fRec0[0])) + dsp->fVec0[(dsp->IOTA0 - min(16385, max(0, iTemp4 + 1))) & 32767] * (dsp->fRec0[0] + (1e+03f - fTemp2))) * (1.0f - fTemp0);
            dsp->fRec1[dsp->IOTA0 & 8191] = fTemp7 + 0.5f * dsp->fRec1[(dsp->IOTA0 - 4801) & 8191];
            outputs[i0] = (((iSlow0) ? dsp->fRec1[dsp->IOTA0 & 8191] : fTemp7));
            dsp->fRec0[1] = dsp->fRec0[0];
            dsp->IOTA0 = dsp->IOTA0 + 1;
        }
    }
}

static inline int32_t clip(int32_t v)
{
    if (v > (INT16_MAX - (INT16_MAX / 8)))
    {
        v = (INT16_MAX - (INT16_MAX / 8));
    }
    if (v < (INT16_MIN + (INT16_MAX / 8)))
    {
        v = (INT16_MIN + (INT16_MAX / 8));
    }
    return v;
}

static int8_t button_debounce(bool state)
{
    int8_t ret = -1;
    static uint16_t count = 0;

    if (state)
    {
        if (count < 2)
        {
            count++;
            if (count == 2)
            {
                ret = 0;
            }
        }
    }
    else
    {
        if (count > 0)
        {
            count--;
            if (count == 0)
            {
                ret = 1;
            }
        }
    }

    return ret;
}

#ifndef SPEAKER_OUTPUT
static void agc(int16_t *const in, uint32_t n)
{
    static const float alpha = (1e-2f); // BW
    static float y2_prime;
    static float g = 1.0f;
    static squelch_mode_e squelch_mode = SQUELCH_ENABLED;
    static uint32_t squelch_timer;
    static bool squelch_on = false;
    float rssi;

    for (size_t i = 0; i < n; i++)
    {
        const float x = g * ((float)in[i] / INT16_MAX);
        in[i] = (x * INT16_MAX) / 5;
        const float e_hat = x * x;
        y2_prime = ((1.0f - alpha) * y2_prime) + (alpha * e_hat);

        if (y2_prime > 1e-6f)
            g *= expf(-0.5f * alpha * logf(y2_prime));

        g = (g > 1e6f) ? 1e6f : g;

        rssi = -20 * log10f(g);
        const bool threshold_exceeded = rssi > -26.0f;

        switch (squelch_mode)
        {
        case SQUELCH_ENABLED:
            squelch_mode = threshold_exceeded ? SQUELCH_RISE : SQUELCH_ENABLED;
            break;
        case SQUELCH_RISE:
            squelch_mode = threshold_exceeded ? SQUELCH_SIGNALHI : SQUELCH_FALL;
            break;
        case SQUELCH_SIGNALHI:
            squelch_mode = threshold_exceeded ? SQUELCH_SIGNALHI : SQUELCH_FALL;
            break;
        case SQUELCH_FALL:
            squelch_mode = threshold_exceeded ? SQUELCH_SIGNALHI : SQUELCH_SIGNALLO;
            squelch_timer = SAMPLERATE / 5;
            break;
        case SQUELCH_SIGNALLO:
            squelch_timer--;
            if (squelch_timer == 0)
                squelch_mode = SQUELCH_TIMEOUT;
            else if (threshold_exceeded)
                squelch_mode = SQUELCH_SIGNALHI;
            break;
        case SQUELCH_TIMEOUT:
            squelch_mode = SQUELCH_ENABLED;
            break;
        case SQUELCH_DISABLED:
            break;
        }

        if (squelch_mode == SQUELCH_TIMEOUT)
        {
            squelch_on = true;
        }
        else if (squelch_mode == SQUELCH_SIGNALHI)
        {
            squelch_on = false;
        }
        if (squelch_on)
        {
            in[i] = 0;
        }
    }
}
#endif

static int32_t input_samples[2][CHUNK_READ_SIZE];
static int16_t input[CHUNK_READ_SIZE] = {0};
static int16_t output[CHUNK_READ_SIZE] = {0};
static int16_t output_samples[2][CHUNK_READ_SIZE * INTERP_RATE];
static float osc_buffer[OSC_BUFFER_LEN];

int main()
{
    int16_t prev_output = 0;
    uint16_t output_idx = 0;
    uint32_t button_press_time = 0;
    uint16_t led_blink_counter = 0;
    bool vibrato = false;
    uint32_t osc_idx = 0;
    bool btn_state = false;

    size_t bi = 0;
    PIO pio = pio1;
    int sm;

    stdio_init_all();
    set_sys_clock_khz(CPU_CLOCK / 1000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, 0);

    // switch regulator to PWM mode
    // to reduce noise
    const uint PSU_PIN = 23;
    gpio_init(PSU_PIN);
    gpio_set_function(PSU_PIN, GPIO_FUNC_SIO);
    gpio_set_dir(PSU_PIN, GPIO_OUT);
    gpio_put(PSU_PIN, 1);

    gpio_init(BUTTON_PIN);
    gpio_set_dir(BUTTON_PIN, GPIO_IN);
    gpio_pull_up(BUTTON_PIN);

    for (size_t i = 0; i < OSC_BUFFER_LEN; i++)
    {
        osc_buffer[i] = sinf(2 * M_PI * i / OSC_BUFFER_LEN);
    }

    pitchshifter_t *dsp = pitchshifter_new(SAMPLERATE);

    gpio_set_function(AUDIO_PIN, GPIO_FUNC_PWM);
    gpio_set_drive_strength(AUDIO_PIN, GPIO_DRIVE_STRENGTH_12MA);
    int pwm_slice_num = pwm_gpio_to_slice_num(AUDIO_PIN);
    pwm_config config = pwm_get_default_config();
    pwm_config_set_clkdiv(&config, 1.f);
    pwm_config_set_wrap(&config, PWM_MAX - 1);
    pwm_init(pwm_slice_num, &config, true);

    int pwm_dma_ch = dma_claim_unused_channel(true);
    dma_channel_config pwm_dma_cfg = dma_channel_get_default_config(pwm_dma_ch);

    channel_config_set_transfer_data_size(&pwm_dma_cfg, DMA_SIZE_16);
    channel_config_set_read_increment(&pwm_dma_cfg, true);
    channel_config_set_write_increment(&pwm_dma_cfg, false);
    channel_config_set_dreq(&pwm_dma_cfg, DREQ_PWM_WRAP0 + pwm_slice_num);

    uint offset = pio_add_program(pio, &inmp441_program);
    sm = pio_claim_unused_sm(pio, true);

    int dma_ch = dma_claim_unused_channel(true);
    dma_channel_config channel_config = dma_channel_get_default_config(dma_ch);
    channel_config_set_dreq(&channel_config, pio_get_dreq(pio, sm, false));
    channel_config_set_transfer_data_size(&channel_config, DMA_SIZE_32);
    channel_config_set_write_increment(&channel_config, true);
    channel_config_set_read_increment(&channel_config, false);

    dma_channel_configure(dma_ch,
                          &channel_config,
                          NULL,
                          &pio->rxf[sm],
                          CHUNK_READ_SIZE,
                          false);

    inmp441_program_init(pio, sm, offset, SAMPLERATE, INMP441_PIN_SD, INMP441_PIN_SCK);

    printf("Starting\n");
    dma_channel_set_write_addr(dma_ch, (void *)input_samples[bi], true);
    dma_channel_configure(pwm_dma_ch, &pwm_dma_cfg, &pwm_hw->slice[pwm_slice_num].cc,
                          (void *)output_samples[bi], CHUNK_READ_SIZE * INTERP_RATE, true);

    while (1)
    {
        dma_channel_wait_for_finish_blocking(dma_ch);
        dma_channel_wait_for_finish_blocking(pwm_dma_ch);
        dma_channel_set_write_addr(dma_ch, (void *)input_samples[bi ^ 1], true);
        dma_channel_configure(pwm_dma_ch, &pwm_dma_cfg, &pwm_hw->slice[pwm_slice_num].cc,
                              (void *)output_samples[bi ^ 1], CHUNK_READ_SIZE * INTERP_RATE, true);

        uint32_t start_time = time_us_32();
        int8_t btn_event = button_debounce(gpio_get(BUTTON_PIN));
        if (btn_event == 1)
        {
            gpio_put(LED_PIN, 1);
            button_press_time = time_us_32();
            btn_state = true;
        }
        else if ((btn_event == 0) && btn_state)
        {
            gpio_put(LED_PIN, 1);
            btn_state = false;
            uint32_t now = time_us_32();
            if ((now - button_press_time) > 2000000)
            {
                // very long press
                vibrato ^= 1;
                if (vibrato)
                {
                    osc_idx = 0;
                }
            }
            else if ((now - button_press_time) > 500000)
            {
                // long press
                dsp->fButton0 ^= 1;
            }
            else
            {
                // short press
                dsp->fHslider0 += 2.0f;
                if (dsp->fHslider0 > 12.0f)
                {
                    dsp->fHslider0 = -12.0f;
                }
            }
        }
        else if (led_blink_counter == 0)
        {
            gpio_put(LED_PIN, 1);
        }
        else if (btn_event < 0)
        {
            gpio_put(LED_PIN, 0);
        }
        led_blink_counter = (led_blink_counter + 1) % BASE_RATE_HZ;

        for (size_t i = 0; i < CHUNK_READ_SIZE; i++)
        {
            input[i] = clip(((int32_t)(input_samples[bi][i])) >> 11);
        }

#ifndef SPEAKER_OUTPUT
        agc(input, CHUNK_READ_SIZE);
#endif
        pitchshifter_process(dsp, CHUNK_READ_SIZE, input, output);
        if (vibrato)
        {
            for (size_t i = 0; i < CHUNK_READ_SIZE; i++)
            {
                output[i] *= osc_buffer[osc_idx];
                osc_idx = (osc_idx + 1) % OSC_BUFFER_LEN;
            }
        }
        
        output_idx = 0;
        for (size_t i = 0; i < CHUNK_READ_SIZE; i++)
        {
            int32_t audio = clip(2 * output[i]);

            audio += INT16_MAX;
            audio /= PWM_MAX - 1;
            interp_bresenham(prev_output, audio, INTERP_RATE, &output_samples[bi][output_idx]);
            output_idx += INTERP_RATE;
            prev_output = audio;
        }
        start_time = time_us_32() - start_time;
        // printf("Took %ld us (%ld%% load)\n", start_time, 100 * start_time / (1000000 / BASE_RATE_HZ));
        bi ^= 1;
    }
}
