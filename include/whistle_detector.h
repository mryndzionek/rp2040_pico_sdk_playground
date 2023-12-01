#ifndef _WHISTLE_DETECTOR_H_
#define _WHISTLE_DETECTOR_H_

#include <stddef.h>
#include <stdbool.h>

#define SAMPLERATE (16000)
#define WIN_SIZE (SAMPLERATE / 50) // 50 updates per second
#define N_FREQS (2)

typedef struct _whistle_detector_t whistle_detector_t;

typedef enum
{
    whistle_detector_none = 0,
    whistle_detector_on = 1,
    whistle_detector_off = 2,
} whistle_detector_out_e;

whistle_detector_t *whistle_detector_create(size_t const (*freqs)[N_FREQS]);
whistle_detector_out_e whistle_detector_update(whistle_detector_t *self, const float *x, size_t nx);

#endif // _WHISTLE_DETECTOR_H_