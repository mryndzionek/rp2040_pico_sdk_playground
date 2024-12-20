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

#define SAMPLERATE (16000UL)

#define CHUNK_READ_SIZE (SHARNN_BRICK_SIZE * FRAME_STEP)
#define FRAME_OFFSET (FRAME_LEN - FRAME_STEP)
#define CHUNK_SIZE (CHUNK_READ_SIZE + FRAME_OFFSET)

#define UPDATE_TIME_MS (1000UL * CHUNK_READ_SIZE / SAMPLERATE)

#define DMA_CHANNEL (0)
#define DMA_CHANNEL_MASK (1u << DMA_CHANNEL)

#define DISP_SIG_SIZE (sizeof(disp_signs) - 1)

// #define DEBUG_PRINTF

static int32_t samples[2][CHUNK_READ_SIZE];
static float input[CHUNK_SIZE] = {0.0};

static const char disp_signs[] = " .,-:+*&NM#";

static char float2disp(float x)
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

void print_features(const float *feat, size_t w, size_t h)
{
    float v;
    float vmin = *feat;
    float vmax = *feat;

    for (size_t i = 0; i < h; i++)
    {
        printf(" | ");
        for (size_t j = 0; j < w; j++)
        {
            v = feat[w * i + j];
            printf("%c", float2disp(v));
            if (v > vmax)
            {
                vmax = v;
            }
            if (v < vmin)
            {
                vmin = v;
            }
        }
        printf(" | %d %f %f\n", i, vmin, vmax);
    }
}

static void(core1_entry)(void)
{
    size_t debounce_count = 0;
    size_t label;
    float prob;

    printf("Core 1 running!\n");

    while (true)
    {
        multicore_fifo_push_blocking(true); // Signal to Core0 that we're ready for more work
        float(*fbins)[NUM_FILT] = (float(*)[NUM_FILT])multicore_fifo_pop_blocking();
#ifdef DEBUG_PRINTF
        uint32_t start_time = time_us_32();
#endif
        sha_rnn_process(fbins, &prob, &label);
        // print_features((const float *)fbins, NUM_FILT, SHARNN_BRICK_SIZE);
#ifdef DEBUG_PRINTF
        float t_ms = (time_us_32() - start_time) / 1000.0;
        printf(" | %.3f | Core 1: inf. time: %.3f ms, util: %.2f%%, prob: %f, label: %d %s |\n",
               time_us_32() / 1000.0, t_ms, 100 * t_ms / UPDATE_TIME_MS,
               prob,
               label, fbank_label_idx_to_str(label));
#endif
        if (debounce_count)
        {
            if (label == 0)
            {
                debounce_count--;
            }
        }
        else
        {
            if (label > 0)
            {
                if (prob >= 0.8)
                {
                    printf(" | %.3f | Detected keyword: %s |\n", time_us_32() / 1000.0, fbank_label_idx_to_str(label));
                    debounce_count = 1;
                }
            }
        }
    }
}

static float fbins_out[SHARNN_BRICK_SIZE][NUM_FILT];

int main()
{
    PIO pio = pio0;
    int sm;
    size_t bi = 0;
    static float fbins[SHARNN_BRICK_SIZE][NUM_FILT];

    stdio_init_all();
    set_sys_clock_khz(280000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    printf("Keyword spotting demo\n");
    printf("Recognized keywords:\n\t");
    for (size_t i = 1; i < NUM_LABELS; i++)
    {
        printf("%s, ", fbank_label_idx_to_str(i));
    }
    printf("\n");

    multicore_launch_core1(core1_entry);

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

    while (true)
    {
        gpio_put(LED_PIN, 1);
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);
#ifdef DEBUG_PRINTF
        uint32_t start_time = time_us_32();
#endif
        dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi ^ 1], true);
        gpio_put(LED_PIN, 0);

        for (size_t i = 0; i < CHUNK_READ_SIZE; i++)
        {
            input[FRAME_OFFSET + i] = (float)(samples[bi][i]);
            input[FRAME_OFFSET + i] /= (1UL << 24);
            input[FRAME_OFFSET + i] *= 0.075;
        }

        fbank_prep(&input[FRAME_OFFSET], CHUNK_READ_SIZE);
        fbank(input, (float(*)[32])fbins, CHUNK_SIZE);
        sha_rnn_norm((float *)fbins, SHARNN_BRICK_SIZE);
        memmove(input, &input[CHUNK_READ_SIZE], FRAME_OFFSET * sizeof(float));

#ifdef DEBUG_PRINTF
        float t_ms = (time_us_32() - start_time) / 1000.0;
        printf(" | %.3f | Core 0: proc. time: %.3f ms, util: %.2f%% |\n",
               time_us_32() / 1000.0, t_ms, 100 * t_ms / UPDATE_TIME_MS);
#endif

        // Check if Core1 is ready for more work
        if (multicore_fifo_rvalid())
        {
            multicore_fifo_pop_blocking();
            // Send a copy of the feature buffer to Core1
            memcpy(fbins_out, fbins, SHARNN_BRICK_SIZE * NUM_FILT * sizeof(float));
            multicore_fifo_push_blocking((uint32_t)fbins_out);
        }
        else
        {
            printf("Core1 not keeping up!\n");
        }
        bi ^= 1;
    }
}
