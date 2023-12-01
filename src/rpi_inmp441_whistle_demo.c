#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pico/stdlib.h"

#include "hardware/dma.h"
#include "inmp441.pio.h"

#include "whistle_detector.h"

#define INMP441_PIN_SD (16)
#define INMP441_PIN_SCK (14)

#define DMA_CHANNEL (0)
#define DMA_CHANNEL_MASK (1u << DMA_CHANNEL)

static const uint CONTROL_LED_PIN = 18;

int main()
{
    PIO pio = pio0;
    int sm;
    size_t bi = 0;
    int32_t samples[2][WIN_SIZE];
    float data[WIN_SIZE];
    whistle_detector_t *detector;

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
                          WIN_SIZE,
                          false);

    inmp441_program_init(pio, sm, offset, SAMPLERATE, INMP441_PIN_SD, INMP441_PIN_SCK);

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    gpio_init(CONTROL_LED_PIN);
    gpio_set_dir(CONTROL_LED_PIN, GPIO_OUT);
    gpio_put(CONTROL_LED_PIN, 1);

    gpio_init(19);
    gpio_set_dir(19, GPIO_OUT);
    gpio_put(19, 1);

    detector = whistle_detector_create(&(size_t[N_FREQS]){1300, 1850});
    assert(detector);

    printf("Starting\n");

    dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi], true);

    while (true)
    {
        gpio_put(LED_PIN, 1);
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);
        dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi ^ 1], true);
        gpio_put(LED_PIN, 0);
        // uint32_t start_time = time_us_32();

        for (size_t i = 0; i < WIN_SIZE; i++)
        {
            data[i] = (float)samples[bi][i] / 2147483.648;
        }

        whistle_detector_out_e ret = whistle_detector_update(detector, data, WIN_SIZE);
        if (ret == whistle_detector_off)
        {
            gpio_put(CONTROL_LED_PIN, 1);
        }
        else if (ret == whistle_detector_on)
        {
            gpio_put(CONTROL_LED_PIN, 0);
        }

        bi ^= 1;
    }
}
