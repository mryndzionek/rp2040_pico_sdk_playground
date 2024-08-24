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

#include "u8g2.h"
#include "u8g2_ili9486_driver.h"

#define INMP441_PIN_SD (16)
#define INMP441_PIN_SCK (14)

#define SAMPLERATE (16000UL)

#define CHUNK_READ_SIZE (SHARNN_BRICK_SIZE * FRAME_STEP)
#define FRAME_OFFSET (FRAME_LEN - FRAME_STEP)
#define CHUNK_SIZE (CHUNK_READ_SIZE + FRAME_OFFSET)

#define UPDATE_TIME_MS (1000UL * CHUNK_READ_SIZE / SAMPLERATE)

// #define DEBUG_PRINTF

static int32_t samples[2][CHUNK_READ_SIZE];
static float input[CHUNK_SIZE] = {0.0};

static u8g2_t u8g2;

static void(core1_entry)(void)
{
    size_t debounce_count = 0;
    size_t label;
    size_t result = 0;
    float prob;

    printf("Core 1 running!\n");

    while (true)
    {
        multicore_fifo_push_blocking(result); // Signal to Core0 that we're ready for more work
        float(*fbins)[NUM_FILT] = (float(*)[NUM_FILT])multicore_fifo_pop_blocking();
#ifdef DEBUG_PRINTF
        uint32_t start_time = time_us_32();
#endif
        sha_rnn_process(fbins, &prob, &label);
        if (debounce_count)
        {
            if (label == 0)
            {
                debounce_count--;
                if (!debounce_count)
                {
                    result = 0;
                }
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
                    result = label;
                }
            }
        }

#ifdef DEBUG_PRINTF
        float t_ms = (time_us_32() - start_time) / 1000.0;
        printf(" | %.3f | Core 1: inf. time: %.3f ms, util: %.2f%%, prob: %f, label: %d %s |\n",
               time_us_32() / 1000.0, t_ms, 100 * t_ms / UPDATE_TIME_MS,
               prob,
               label, fbank_label_idx_to_str(label));
#endif
    }
}

static float fbins_out[SHARNN_BRICK_SIZE][NUM_FILT];

int main()
{
    char tmp[64] = "    ";
    size_t n = 4;
    PIO pio = pio1;
    int sm;
    size_t bi = 0;
    static float fbins[SHARNN_BRICK_SIZE][NUM_FILT];

    stdio_init_all();
    set_sys_clock_khz(280000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    u8g2_Setup_ili9486_8bit_480x320_f(&u8g2);

    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);
    u8g2_ClearDisplay(&u8g2);

    u8g2_SetFont(&u8g2, u8g2_font_logisoso22_tr);
    u8g2_DrawStr(&u8g2, 40, 40, "Keyword spotting demo");
    u8g2_SetFont(&u8g2, u8g2_font_luRS18_te);
    u8g2_DrawStr(&u8g2, 40, 80, "Recognized keywords:");

    for (size_t i = 1; i < NUM_LABELS; i++)
    {
        n += snprintf(&tmp[n], sizeof(tmp) - n, "%s, ", fbank_label_idx_to_str(i));
        assert(n < sizeof(tmp));
    }

    u8g2_DrawStr(&u8g2, 40, 120, tmp);
    u8g2_SendBuffer(&u8g2);
    u8g2_SetFont(&u8g2, u8g2_font_inb49_mf);

    multicore_launch_core1(core1_entry);

    fbank_init();

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

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    printf("Starting\n");
    dma_channel_set_write_addr(dma_ch, (void *)samples[bi], true);

    while (true)
    {
        gpio_put(LED_PIN, 1);
        dma_channel_wait_for_finish_blocking(dma_ch);
#ifdef DEBUG_PRINTF
        uint32_t start_time = time_us_32();
#endif
        dma_channel_set_write_addr(dma_ch, (void *)samples[bi ^ 1], true);
        gpio_put(LED_PIN, 0);

        for (size_t i = 0; i < CHUNK_READ_SIZE; i++)
        {
            input[FRAME_OFFSET + i] = (float)(samples[bi][i]);
            input[FRAME_OFFSET + i] /= (1UL << 24);
            input[FRAME_OFFSET + i] *= 0.2;
        }

        fbank_prep(&input[FRAME_OFFSET], CHUNK_READ_SIZE);
        fbank(input, (float(*)[32])fbins, CHUNK_SIZE);
        sha_rnn_norm((float *)fbins, SHARNN_BRICK_SIZE);
        memmove(input, &input[CHUNK_READ_SIZE], FRAME_OFFSET * sizeof(float));

        // Check if Core1 is ready for more work
        if (multicore_fifo_rvalid())
        {
            size_t label = multicore_fifo_pop_blocking();
            // Send a copy of the feature buffer to Core1
            memcpy(fbins_out, fbins, SHARNN_BRICK_SIZE * NUM_FILT * sizeof(float));
            multicore_fifo_push_blocking((uint32_t)fbins_out);
            
            if (label > 0)
            {
                u8g2_DrawStr(&u8g2, (480 - 176) / 2, 220, fbank_label_idx_to_str(label));
            }
            else
            {
                u8g2_SetDrawColor(&u8g2, 0);
                u8g2_DrawBox(&u8g2, (480 - 176) / 2, 150, 220, 100);
                u8g2_SetDrawColor(&u8g2, 1);
            }

            u8g2_SendBuffer(&u8g2);
        }
        else
        {
            printf("Core1 not keeping up!\n");
        }
#ifdef DEBUG_PRINTF
        float t_ms = (time_us_32() - start_time) / 1000.0;
        printf(" | %.3f | Core 0: proc. time: %.3f ms, util: %.2f%% |\n",
               time_us_32() / 1000.0, t_ms, 100 * t_ms / UPDATE_TIME_MS);
#endif
        bi ^= 1;
    }
}
