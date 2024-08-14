#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "arm_math.h"

#include "pico/stdlib.h"
#include "pico/multicore.h"

#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "inmp441.pio.h"

#include "fbank.h"
#include "sha_rnn_intf.h"

#define INMP441_PIN_SD (16)
#define INMP441_PIN_SCK (14)

#define BRICK_SIZE (11)

#define SAMPLERATE (16000UL)

#define FRAME_OFFSET (FRAME_LEN - FRAME_STEP)
#define CHUNK_SIZE ((BRICK_SIZE * FRAME_STEP) + (FRAME_OFFSET))
#define CHUNK_READ_SIZE (CHUNK_SIZE - FRAME_OFFSET)

#define DMA_CHANNEL (0)
#define DMA_CHANNEL_MASK (1u << DMA_CHANNEL)

static int32_t samples[2][CHUNK_SIZE];
static float input[CHUNK_SIZE] = {0.0};

#define DISP_SIG_SIZE (sizeof(disp_signs) - 1)

static const char disp_signs[] = " .,-+*&NM#";

char float2disp(float x)
{
    int32_t i = ((x + 0.3) * (float)DISP_SIG_SIZE) / 3.5;

    if (i >= (int32_t)DISP_SIG_SIZE)
    {
        i = DISP_SIG_SIZE - 1;
    }
    else if (i < 0)
    {
        i = 0;
    }
    return disp_signs[i];
}

static void(core1_entry)(void)
{
    size_t debounce_count = 0;
    size_t label;
    float logit;

    printf("Core 1 running!\n");

    while (true)
    {
        multicore_fifo_push_blocking(true);
        float(*fbins)[NUM_FILT] = (float(*)[NUM_FILT])multicore_fifo_pop_blocking();

        uint32_t start_time = time_us_32();
        sha_rnn_norm(fbins);

        // for (size_t i = 0; i < NUM_FRAMES; i++)
        // {
        //     float vmin = 100.0f;
        //     float vmax = -100.0f;

        //     printf(" | ");
        //     for (size_t j = 0; j < NUM_FILT; j++)
        //     {
        //         printf("%c", float2disp(fbins[i][j]));
        //         if (fbins[i][j] > vmax)
        //         {
        //             vmax = fbins[i][j];
        //         }
        //         if (fbins[i][j] < vmin)
        //         {
        //             vmin = fbins[i][j];
        //         }
        //     }
        //     printf(" | %d %f %f\n", i, vmin, vmax);
        // }

        sha_rnn_process(fbins, &logit, &label);
        if (debounce_count)
        {
            if (label == 0)
            {
                debounce_count--;
            }
        }
        else
        {
            printf(" | %.3f | Core 1: %.3f ms %f, %d %s |\n", time_us_32() / 1000.0, (time_us_32() - start_time) / 1000.0, logit, label, fbank_label_idx_to_str(label));
            if (label > 0)
            {
                if (logit >= 3.0)
                {
                    printf(" | %.3f | Detected keyword: %s |\n", time_us_32() / 1000.0, fbank_label_idx_to_str(label));
                    debounce_count = 1;
                }
            }
        }
    }
}

static uint32_t core1_stack[8 * 1024UL];

static float fbins_out[NUM_FRAMES][NUM_FILT];

int main()
{
    PIO pio = pio0;
    int sm;
    size_t bi = 0;
    static float fbins[NUM_FRAMES][NUM_FILT];

    stdio_init_all();
    set_sys_clock_khz(280000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);
    multicore_launch_core1_with_stack(core1_entry, core1_stack, sizeof(core1_stack));

    fbank_init();

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
                          CHUNK_READ_SIZE,
                          false);

    inmp441_program_init(pio, sm, offset, SAMPLERATE, INMP441_PIN_SD, INMP441_PIN_SCK);

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    printf("Starting\n");

    dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi], true);

    uint32_t start_time;
    uint32_t busy_time = 0;
    size_t brick_num = 0;

    while (true)
    {
        gpio_put(LED_PIN, 1);
        start_time = time_us_32();
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);
        start_time = time_us_32();
        dma_channel_set_write_addr(DMA_CHANNEL, (void *)&samples[bi ^ 1][FRAME_OFFSET], true);
        gpio_put(LED_PIN, 0);

        // float mmin = 100.0;
        // float mmax = -100.0;

        for (size_t i = FRAME_OFFSET; i < CHUNK_SIZE; i++)
        {
            input[i] = (float)(samples[bi][i]);
            input[i] /= (1UL << 28);

            // if (input[i] > mmax)
            // {
            //     mmax = input[i];
            // }
            // if (input[i] < mmin)
            // {
            //     mmin = input[i];
            // }
        }
        // printf("%f %f\n", mmin, mmax);

        fbank(input, (float(*)[32])fbins[brick_num], CHUNK_READ_SIZE);
        brick_num += BRICK_SIZE;
        memmove(input, &input[CHUNK_SIZE - FRAME_OFFSET], FRAME_OFFSET * sizeof(float));

        if (multicore_fifo_rvalid())
        {
            multicore_fifo_pop_blocking();
            memcpy(fbins_out, fbins, sizeof(fbins));
            multicore_fifo_push_blocking((uint32_t)fbins_out);
            // printf(" | Core 0: %.3f ms |\n", busy_time / 1000.0);
            busy_time = 0;
        }

        if (brick_num == NUM_FRAMES)
        {
            brick_num = 0;
        }

        busy_time += (time_us_32() - start_time);
        bi ^= 1;
    }
}
