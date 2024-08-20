#include "sha_rnn_intf.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "fastgrnn_rnn0_params.h"
#include "fastgrnn_rnn1_params.h"
#include "fastgrnn_fc_params.h"

#define EULER_NUMBER_F (2.71828182846f)

// clang-format off

const float INPUT_MEANS[32] = 
   {-2.20714022e+01, -2.10418987e+01, -2.01226244e+01, -1.98376712e+01, -1.96531247e+01, -1.94665661e+01, -1.93195476e+01, -1.89841527e+01,
    -1.90280114e+01, -1.90856069e+01, -1.90287364e+01, -1.90039459e+01, -1.90815247e+01, -1.90292351e+01, -1.89478379e+01, -1.89638440e+01,
    -1.88551630e+01, -1.88364735e+01, -1.88426662e+01, -1.86974404e+01, -1.86674291e+01, -1.86341649e+01, -1.85360960e+01, -1.85488060e+01,
    -1.85736517e+01, -1.86158047e+01, -1.86675265e+01, -1.87325428e+01, -1.88939340e+01, -1.91419879e+01, -1.93809577e+01, -2.00635581e+01};

const float INPUT_STDEVS[32] = 
   { 1.51904685e+00,  2.56263570e+00,  3.26918455e+00,  3.49894536e+00,  3.65298700e+00,  3.83852754e+00,  3.97563294e+00,  4.16645257e+00,
     4.12782323e+00,  4.04010193e+00,  3.99454258e+00,  3.93411565e+00,  3.84118601e+00,  3.79329999e+00,  3.77533532e+00,  3.72341660e+00,
     3.70462574e+00,  3.66291410e+00,  3.65237271e+00,  3.69993782e+00,  3.67954402e+00,  3.66206700e+00,  3.68731804e+00,  3.65337917e+00,
     3.58808535e+00,  3.53377602e+00,  3.49063573e+00,  3.44829142e+00,  3.40913938e+00,  3.41278226e+00,  3.33891566e+00,  3.10446943e+00};

// clang-format on

static inline float sigmoidf(float n)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

static inline float expo(float y)
{
    if (y > 80)
        y = 80;
    return exp(y);
}

static float softmax(const float *xs, size_t n, size_t len)
{
    float sum = 0;
    for (size_t i = 0; i < len; i++)
        sum += expo(xs[i]);
    if (sum == 0)
        sum = 0.001;
    return (expo(xs[n])) / sum;
}

static void rnn0_process(const float input[32], const float hidden[64], float output[64])
{
    float z;
    float c;

    for (size_t i = 0; i < 32; i++)
    {
        for (size_t j = 0; j < 64; j++)
        {
            output[j] += GRNN0_W[j][i] * input[i];
        }
    }

    for (size_t j = 0; j < 64; j++)
    {
        for (size_t i = 0; i < 64; i++)
        {
            output[j] += GRNN0_U[j][i] * hidden[i];
        }
    }

    for (size_t j = 0; j < 64; j++)
    {
        z = output[j] + GRNN0_BIAS_GATE[j];
        z = sigmoidf(z);
        c = output[j] + GRNN0_BIAS_UPDATE[j];
        c = tanhf(c);

        output[j] = z * hidden[j] + (sigmoidf(GRNN0_ZETA) * (1.0 - z) + sigmoidf(GRNN0_NU)) * c;
    }
}

void sha_rnn_rnn0_process(const sha_rnn_input_t input, sha_rnn_rnn1_input_t output)
{
    float hidden[64] = {0.0f};

    for (size_t k = 0; k < SHARNN_BRICK_SIZE; k++)
    {
        memset(output, 0, sizeof(sha_rnn_rnn1_input_t));
        rnn0_process(input[k], hidden, output);
        memcpy(hidden, output, sizeof(hidden));
    }
}

static void rnn1_process(const float input[64], const float hidden[32], float output[32])
{
    float z;
    float c;

    for (size_t j = 0; j < 32; j++)
    {
        for (size_t i = 0; i < 64; i++)
        {
            output[j] += GRNN1_W[j][i] * input[i];
        }
    }

    for (size_t j = 0; j < 32; j++)
    {
        for (size_t i = 0; i < 32; i++)
        {
            output[j] += GRNN1_U[j][i] * hidden[i];
        }
    }

    for (size_t j = 0; j < 32; j++)
    {
        z = output[j] + GRNN1_BIAS_GATE[j];
        z = sigmoidf(z);
        c = output[j] + GRNN1_BIAS_UPDATE[j];
        c = tanhf(c);

        output[j] = z * hidden[j] + (sigmoidf(GRNN1_ZETA) * (1.0 - z) + sigmoidf(GRNN1_NU)) * c;
    }
}

void sha_rnn_rnn1_process(const sha_rnn_rnn1_input_t input, sha_rnn_fc_input_t output)
{
    static sha_rnn_rnn1_input_t rnn1_input_hist[9];
    static size_t rnn1_hist_idx = 0;

    float rnn1_hidden[32] = {0.0};

    memcpy(rnn1_input_hist[rnn1_hist_idx], input, sizeof(sha_rnn_rnn1_input_t));

    for (size_t i = 0; i < 9; i++)
    {
        size_t j = (rnn1_hist_idx + 1 + i) % 9;
        memset(output, 0, sizeof(sha_rnn_fc_input_t));
        rnn1_process(rnn1_input_hist[j], rnn1_hidden, output);
        memcpy(rnn1_hidden, output, sizeof(sha_rnn_fc_input_t));
    }

    rnn1_hist_idx++;

    if (rnn1_hist_idx == 9)
    {
        rnn1_hist_idx = 0;
    }
}

void sha_rnn_fc_process(const sha_rnn_fc_input_t input, sha_rnn_output_t output)
{
    memset(output, 0, 6 * sizeof(float));

    for (size_t j = 0; j < FC_OUT_DIM; j++)
    {
        for (size_t i = 0; i < FC_IN_DIM; i++)
        {
            output[j] += input[i] * FC_W[j][i];
        }
        output[j] += FC_B[j];
    }
}

void sha_rnn_get_max_prob(const sha_rnn_output_t input, float *max_prob, size_t *max_idx)
{
    float max_logit = input[0];
    *max_idx = 0;

    for (size_t j = 0; j < FC_OUT_DIM; j++)
    {
        if (input[j] > max_logit)
        {
            max_logit = input[j];
            *max_idx = j;
        }
    }

    *max_prob = softmax(input, *max_idx, FC_OUT_DIM);
}

void sha_rnn_process(const sha_rnn_input_t input, float *max_prob, size_t *max_idx)
{
    static float output[64] = {0.0f};
    static float output2[32] = {0.0f};
    static float output3[6] = {0.0f};

    sha_rnn_rnn0_process(input, output);
    sha_rnn_rnn1_process(output, output2);
    sha_rnn_fc_process(output2, output3);
    sha_rnn_get_max_prob(output3, max_prob, max_idx);
}

void sha_rnn_norm(float *input, size_t num)
{
    for (size_t i = 0; i < num; i++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            input[i * 32 + j] = (input[i * 32 + j] - INPUT_MEANS[j]) / INPUT_STDEVS[j];
        }
    }
}
