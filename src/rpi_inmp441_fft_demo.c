#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "arm_math.h"

#include "pico/stdlib.h"

#include "hardware/dma.h"
#include "inmp441.pio.h"

#define INMP441_PIN_SD (16)
#define INMP441_PIN_SCK (14)

#define SAMPLERATE (16000)
#define FFTSIZE (1024)

#define DMA_CHANNEL (0)
#define DMA_CHANNEL_MASK (1u << DMA_CHANNEL)
#define DISP_SIG_SIZE (sizeof(disp_signs) - 1)

static const char disp_signs[] = " .,-+*&NM#";

static uint32_t samples[2][FFTSIZE];

static char float2disp(float32_t x)
{
    size_t i = 10 * log10f(40 * x);
    if (i >= DISP_SIG_SIZE)
    {
        i = DISP_SIG_SIZE - 1;
    }

    return disp_signs[i];
}

static void get_window(q31_t *const y, size_t ny)
{
    float32_t w[ny];
    arm_hanning_f32(w, ny);
    arm_float_to_q31(w, y, ny);
}

int main()
{
    PIO pio = pio0;
    int sm;
    size_t bi = 0;

    arm_status status = ARM_MATH_SUCCESS;
    arm_rfft_instance_q31 ffti;
    q31_t tmp[FFTSIZE];
    q31_t tmp2[2 * FFTSIZE];
    q31_t win[FFTSIZE];
    uint32_t idx;
    q31_t maxAmp;
    float32_t maxFreq;

    stdio_init_all();
    // set_sys_clock_khz(240000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    uint offset = pio_add_program(pio, &inmp441_program);
    sm = pio_claim_unused_sm(pio, true);

    dma_claim_mask(DMA_CHANNEL_MASK);
    dma_channel_config channel_config = dma_channel_get_default_config(DMA_CHANNEL);
    channel_config_set_dreq(&channel_config, pio_get_dreq(pio, sm, false));
    channel_config_set_transfer_data_size(&channel_config, DMA_SIZE_32);
    channel_config_set_write_increment(&channel_config, true);
    channel_config_set_read_increment(&channel_config, false);

    dma_channel_configure(DMA_CHANNEL,
                          &channel_config,
                          NULL,
                          &pio->rxf[sm],
                          FFTSIZE,
                          false);

    inmp441_program_init(pio, sm, offset, SAMPLERATE, INMP441_PIN_SD, INMP441_PIN_SCK);

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    status = arm_rfft_init_1024_q31(&ffti, 0, 1);
    // assert(status == ARM_MATH_SUCCESS);
    (void)status;
    get_window(win, FFTSIZE);

    printf("Starting\n");

    dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi], true);

    while (true)
    {
        gpio_put(LED_PIN, 1);
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);
        dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi ^ 1], true);
        gpio_put(LED_PIN, 0);

        for (size_t j = 0; j < FFTSIZE; j++)
        {
            tmp2[j] = *((q31_t *)&samples[bi][j]);
        }

        arm_mult_q31(tmp2, win, tmp, FFTSIZE);

        arm_rfft_q31(&ffti, tmp, tmp2);
        arm_cmplx_mag_q31(tmp2, tmp, FFTSIZE);
        arm_max_q31(tmp, FFTSIZE / 2, &maxAmp, &idx);
        maxFreq = (float32_t)idx * SAMPLERATE / FFTSIZE;

        // print spectrogram
        printf("|");
        for (size_t j = 0; j < 128; j++)
        {
            float32_t v = (float32_t)tmp[j * (FFTSIZE / 512)] / (1UL << 22);
            printf("%c", float2disp(v));
        }
        printf("| %06.1f Hz\n", maxFreq);
        bi ^= 1;
    }
}
