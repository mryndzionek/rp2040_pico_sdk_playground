#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define LEDS_NUM (2UL * 64)

#define qsuba(x, b) ((x > b) ? x - b : 0)
#define pproc(x, b) ((x > b) ? x - b : 2)

static uint32_t _time;

static uint8_t sin8(uint8_t theta)
{
    static const uint8_t b_m16_interleave[] = {0, 49, 49, 41, 90, 27, 117, 10};
    uint8_t offset = theta;

    if (theta & 0x40)
    {
        offset = (uint8_t)255 - offset;
    }
    offset &= 0x3F; // 0..63

    uint8_t secoffset = offset & 0x0F; // 0..15
    if (theta & 0x40)
        secoffset++;

    uint8_t section = offset >> 4; // 0..3
    uint8_t s2 = section * 2;
    const uint8_t *p = b_m16_interleave;
    p += s2;
    uint8_t b = *p;
    p++;
    uint8_t m16 = *p;

    uint8_t mx = (m16 * secoffset) >> 4;

    int8_t y = mx + b;
    if (theta & 0x80)
        y = -y;

    y += 128;

    return y;
}

static uint8_t cos8(uint8_t theta)
{
    return sin8(theta + 64);
}

static uint16_t beat88(uint16_t beats_per_minute_88, uint32_t timebase)
{
    return ((_time - timebase) * beats_per_minute_88 * 280) >> 16;
}

static uint16_t beat16(uint16_t beats_per_minute, uint32_t timebase)
{
    if (beats_per_minute < 256)
        beats_per_minute <<= 8;
    return beat88(beats_per_minute, timebase);
}

static uint8_t beat8(uint16_t beats_per_minute, uint32_t timebase)
{
    return beat16(beats_per_minute, timebase) >> 8;
}

static inline __attribute__((always_inline)) uint8_t scale8(uint8_t i, uint8_t scale)
{
    return (((uint16_t)i) * (1 + (uint16_t)(scale))) >> 8;
}

static inline __attribute__((always_inline)) uint8_t scale8_LEAVING_R1_DIRTY(uint8_t i, uint8_t scale)
{
    return (((uint16_t)i) * ((uint16_t)(scale) + 1)) >> 8;
}

static uint8_t beatsin8(uint16_t beats_per_minute, uint8_t lowest, uint8_t highest)
{
    uint32_t timebase = 0;
    uint8_t phase_offset = 0;
    uint8_t beat = beat8(beats_per_minute, timebase);
    uint8_t beatsin = sin8(beat + phase_offset);
    uint8_t rangewidth = highest - lowest;
    uint8_t scaledbeat = scale8(beatsin, rangewidth);
    uint8_t result = lowest + scaledbeat;
    return result;
}

static uint8_t ease8InOutCubic(uint8_t i)
{
    uint8_t ii = scale8_LEAVING_R1_DIRTY(i, i);
    uint8_t iii = scale8_LEAVING_R1_DIRTY(ii, i);

    uint16_t r1 = (3 * (uint16_t)(ii)) - (2 * (uint16_t)(iii));

    /* the code generated for the above *'s automatically
       cleans up R1, so there's no need to explicitily call
       cleanup_R1(); */

    uint8_t result = r1;

    if (r1 & 0x100)
    {
        result = 255;
    }
    return result;
}

static uint8_t triwave8(uint8_t in)
{
    if (in & 0x80)
    {
        in = 255 - in;
    }
    uint8_t out = in << 1;
    return out;
}

static uint8_t cubicwave8(uint8_t in)
{
    return ease8InOutCubic(triwave8(in));
}

static uint32_t color_wheel(uint8_t pos)
{
    pos = 255 - pos;
    if (pos < 85)
    {
        return ((uint32_t)(255 - pos * 3) << 16) | ((uint32_t)(0) << 8) | (pos * 3);
    }
    else if (pos < 170)
    {
        pos -= 85;
        return ((uint32_t)(0) << 16) | ((uint32_t)(pos * 3) << 8) | (255 - pos * 3);
    }
    else
    {
        pos -= 170;
        return ((uint32_t)(pos * 3) << 16) | ((uint32_t)(255 - pos * 3) << 8) | (0);
    }
}

void plasma(uint8_t leds[LEDS_NUM][3])
{
    int thisPhase = beatsin8(3, -64, 64);
    int thatPhase = beatsin8(7, -64, 64);

    for (int k = 0; k < LEDS_NUM; k++)
    {
        int colorIndex = cubicwave8((k * 2) + thisPhase) / 2 + cos8((k * 1) + thatPhase) / 2;
        int thisBright = qsuba(colorIndex, beatsin8(7, 0, 220));

        uint32_t c = color_wheel(colorIndex % 256);
        uint8_t r = (c >> 16) & 0xFF;
        uint8_t g = (c >> 8) & 0xFF;
        uint8_t b = c & 0xFF;

        leds[k][0] = r;
        leds[k][1] = pproc(g, thisBright);
        leds[k][2] = pproc(b, thisBright);
    }
    _time += 1;
}