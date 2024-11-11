#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#include "hardware/clocks.h"
#include "hardware/dma.h"

#include "pico/stdlib.h"
#include "pico/multicore.h"

#include "bipbuffer.h"

void bipbuffer_basic_test(size_t size)
{
    bipbuffer_t *buf = bipbuffer_new(2 * size);
    hard_assert(buf);
    bipbuffer_rw_pair_t rw = bipbuffer_split(buf);
    bipbuffer_reader_t *reader = rw.reader;
    bipbuffer_writer_t *writer = rw.writer;

    const size_t ITERS = 1000;

    for (size_t i = 0; i < ITERS; i++)
    {
        uint8_t data[size];
        for (size_t s = 0; s < size; s++)
        {
            data[s] = s % 255;
        }
        bipbuffer_wgrant_t wgr = bipbuffer_writer_grant_exact(writer, size);
        hard_assert(wgr.data);
        hard_assert(wgr.len == size);
        memcpy(wgr.data, data, size);
        bipbuffer_writer_commit(writer, &wgr, size);

        bipbuffer_rgrant_t rgr = bipbuffer_reader_read(reader);
        hard_assert(rgr.data);
        hard_assert(rgr.len == size);
        hard_assert(memcmp(rgr.data, data, size) == 0);
        bipbuffer_reader_release(reader, &rgr, size);
    }

    const bool ret = bipbuffer_release(buf, reader, writer);
    hard_assert(ret);
}

static const size_t MULTI_ITERS = 1000;

static void core1_entry()
{
    while (true)
    {
        size_t size = multicore_fifo_pop_blocking();
        bipbuffer_t *buf = bipbuffer_new(2 * size);
        hard_assert(buf);
        bipbuffer_rw_pair_t rw = bipbuffer_split(buf);
        bipbuffer_reader_t *reader = rw.reader;
        bipbuffer_writer_t *writer = rw.writer;

        bipbuffer_reader_t *reader_ = (bipbuffer_reader_t *)multicore_fifo_pop_blocking();
        multicore_fifo_push_blocking((uint32_t)reader);

        for (size_t i = 0; i < MULTI_ITERS; i++)
        {
            while (true)
            {
                bipbuffer_rgrant_t rgr = bipbuffer_reader_read(reader_);
                if (rgr.data)
                {
                    bipbuffer_wgrant_t wgr = bipbuffer_writer_grant_exact(writer, rgr.len);
                    hard_assert(wgr.data);
                    hard_assert(wgr.len == rgr.len);
                    memcpy(wgr.data, rgr.data, rgr.len);
                    bipbuffer_writer_commit(writer, &wgr, rgr.len);
                    bipbuffer_reader_release(reader_, &rgr, rgr.len);
                    break;
                }
            }
        }

        multicore_fifo_pop_blocking();
        multicore_fifo_push_blocking(true);

        const bool ret = bipbuffer_release(buf, reader, writer);
        hard_assert(ret);
    }
}

static void bipbuffer_multi_test(size_t size)
{
    multicore_fifo_push_blocking(size);
    bipbuffer_t *buf = bipbuffer_new(2 * size);
    hard_assert(buf);
    bipbuffer_rw_pair_t rw = bipbuffer_split(buf);
    bipbuffer_reader_t *reader = rw.reader;
    bipbuffer_writer_t *writer = rw.writer;

    multicore_fifo_push_blocking((uint32_t)reader);
    bipbuffer_reader_t *reader_ = (bipbuffer_reader_t *)multicore_fifo_pop_blocking();

    for (size_t i = 0; i < MULTI_ITERS; i++)
    {
        uint8_t *data = calloc(1, size);
        for (size_t s = 0; s < size; s++)
        {
            data[s] = s % 255;
        }
        bipbuffer_wgrant_t wgr = bipbuffer_writer_grant_exact(writer, size);
        hard_assert(wgr.data);
        hard_assert(wgr.len == size);
        memcpy(wgr.data, data, size);
        bipbuffer_writer_commit(writer, &wgr, size);
        free(data);

        while (true)
        {
            bipbuffer_rgrant_t rgr = bipbuffer_reader_read(reader_);
            if (rgr.data)
            {
                hard_assert(memcmp(rgr.data, data, rgr.len) == 0);
                bipbuffer_reader_release(reader_, &rgr, rgr.len);
                break;
            }
        }
    }

    multicore_fifo_push_blocking(true);
    multicore_fifo_pop_blocking();

    const bool ret = bipbuffer_release(buf, reader, writer);
    hard_assert(ret);
}

int main()
{
    stdio_init_all();
    set_sys_clock_khz(133000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, 0);
    bool led_state = false;

    for (uint32_t s = 2; s < 10000; s += 1000)
    {
        printf("Basic test number: %ld\n", s);
        bipbuffer_basic_test(s);
        gpio_put(LED_PIN, led_state);
        led_state ^= 1;
    }

    multicore_launch_core1(core1_entry);
    for (uint32_t s = 2; s < 10000; s += 1000)
    {
        printf("Multi test number: %ld\n", s);
        bipbuffer_multi_test(s);
        gpio_put(LED_PIN, led_state);
        led_state ^= 1;
    }

    while (true)
    {
        gpio_put(LED_PIN, led_state);
        led_state ^= 1;
        sleep_ms(100);
        tight_loop_contents();
    }
}
