#include "whistle_detector.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pico/stdlib.h"

typedef struct
{
    float coeff;
    float q0;
    float q1;
    float q2;
} Goertzel_state_t;

typedef struct
{
    Goertzel_state_t states[N_FREQS];
    float powers[N_FREQS];
} Goertzel_t;

typedef struct
{
    bool active;
    size_t tres_up;
    size_t tres_down;
    size_t count;
} lpf_t;

typedef enum
{
    detector_idle = 0,
    detector_f1_detect,
    detector_f2_detect,
    detector_stop,
} detector_e;

typedef struct
{
    lpf_t lpf_f1;
    lpf_t lpf_f2;
    detector_e state;
    size_t count;
    size_t timer;
    size_t tres;
} detector_t;

struct _whistle_detector_t
{
    Goertzel_t gt;
    detector_t detector;
};

static void lpf_init(lpf_t *self, size_t up_tres, size_t down_tres)
{
    self->active = false;
    self->tres_down = down_tres;
    self->tres_up = up_tres;
    self->count = 0;
}

static void lpf_update(lpf_t *self, bool s)
{
    if (self->active)
    {
        if (!s)
        {
            self->count++;
            if (self->tres_down == self->count)
            {
                self->active = false;
                self->count = 0;
            }
        }
        else
        {
            if (self->count > 0)
            {
                self->count--;
            }
        }
    }
    else
    {
        if (s)
        {
            self->count++;
            if (self->tres_up == self->count)
            {
                self->active = true;
                self->count = 0;
            }
        }
        else
        {
            if (self->count > 0)
            {
                self->count--;
            }
        }
    }
}

static void detector_init(detector_t *self, size_t up_tres, size_t down_tres, size_t tres)
{
    lpf_init(&self->lpf_f1, up_tres, down_tres);
    lpf_init(&self->lpf_f2, up_tres, down_tres);
    self->state = detector_idle;
    self->count = 0;
    self->timer = 0;
    self->tres = tres;
}

static whistle_detector_out_e detector_update(detector_t *self, bool f1, bool f2)
{
    whistle_detector_out_e ret = whistle_detector_none;

    if (f1 && (!f2))
    {
        lpf_update(&self->lpf_f1, true);
    }
    else if (f2 && (!f1))
    {
        lpf_update(&self->lpf_f2, true);
    }
    else
    {
        lpf_update(&self->lpf_f1, false);
        lpf_update(&self->lpf_f2, false);
    }

    switch (self->state)
    {
    case detector_idle:
        if (self->lpf_f1.active || self->lpf_f2.active)
        {
            self->count++;
            if (self->count > self->tres)
            {
                self->count = 0;
                if (self->lpf_f1.active)
                {
                    self->state = detector_f2_detect;
                }
                else
                {
                    self->state = detector_f1_detect;
                }
                printf("----\n");
            }
        }
        else
        {
            self->count = 0;
        }
        break;

    case detector_f1_detect:
        if (self->lpf_f1.active)
        {
            self->count++;
            if (self->count > self->tres)
            {
                self->count = 0;
                self->timer = 0;
                self->state = detector_stop;
                printf("Off!!!!\n");
                ret |= whistle_detector_off;
            }
        }
        else
        {
            self->timer++;
            if (self->timer >= (4 * self->tres))
            {
                self->count = 0;
                self->timer = 0;
                self->state = detector_idle;
            }
        }
        break;

    case detector_f2_detect:
        if (self->lpf_f2.active)
        {
            self->count++;
            if (self->count > self->tres)
            {
                self->count = 0;
                self->timer = 0;
                self->state = detector_stop;
                printf("On!!!!\n");
                ret |= whistle_detector_on;
            }
        }
        else
        {
            self->timer++;
            if (self->timer >= (4 * self->tres))
            {
                self->count = 0;
                self->timer = 0;
                self->state = detector_idle;
            }
        }
        break;

    case detector_stop:
        self->timer++;
        if (self->timer >= (2 * self->tres))
        {
            self->timer = 0;
            self->state = detector_idle;
        }
        break;
    }

    return ret;
}

static void Goertzel_state_reset(Goertzel_state_t *self)
{
    self->q1 = 0.0f;
    self->q2 = 0.0f;
}

static void Goertzel_state_init(Goertzel_state_t *self, size_t freq)
{
    int k = 0.5 + ((WIN_SIZE * freq) / SAMPLERATE);
    float w = 2.0f * M_PI * k / WIN_SIZE;
    self->coeff = 2.0f * cosf(w);
}

static float Goertzel_state_update(Goertzel_state_t *self, const float *x, size_t nx)
{
    Goertzel_state_reset(self);

    for (size_t i = 0; i < nx; i++)
    {
        self->q0 = self->coeff * self->q1 - self->q2 + x[i];
        self->q2 = self->q1;
        self->q1 = self->q0;
    }

    // return the power
    return 2.0f * ((self->q1 * self->q1) + (self->q2 * self->q2) - (self->q1 * self->q2 * self->coeff)) / SAMPLERATE;
}

static void Goertzel_init(Goertzel_t *self, size_t const (*freqs)[N_FREQS])
{
    for (size_t i = 0; i < N_FREQS; i++)
    {
        Goertzel_state_init(&self->states[i], (*freqs)[i]);
        self->powers[i] = 0.0f;
    }
}

static void Goertzel_update(Goertzel_t *self, const float *x, size_t nx)
{
    for (size_t i = 0; i < N_FREQS; i++)
    {
        self->powers[i] = Goertzel_state_update(&self->states[i], x, nx);
    }
}

whistle_detector_t *whistle_detector_create(size_t const (*freqs)[N_FREQS])
{
    static whistle_detector_t self;
    Goertzel_init(&self.gt, freqs);
    detector_init(&self.detector, 5, 8, 10);

    return &self;
}

whistle_detector_out_e whistle_detector_update(whistle_detector_t *self, const float *x, size_t nx)
{
    Goertzel_update(&self->gt, x, nx);
    return detector_update(&self->detector, self->gt.powers[0] > 0.8, self->gt.powers[1] > 0.8);
}
