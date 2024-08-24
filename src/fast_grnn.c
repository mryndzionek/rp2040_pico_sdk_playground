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
   {-2.12331905e+01, -1.99339143e+01, -1.88493105e+01, -1.85132621e+01, -1.83019754e+01, -1.80862333e+01, -1.78963678e+01, -1.74733286e+01,
    -1.74941011e+01, -1.75390417e+01, -1.74435942e+01, -1.73745611e+01, -1.74294784e+01, -1.73334774e+01, -1.72274376e+01, -1.72476427e+01,
    -1.71092561e+01, -1.70597725e+01, -1.70682271e+01, -1.68957978e+01, -1.68249507e+01, -1.67537905e+01, -1.66493864e+01, -1.66498491e+01,
    -1.66505395e+01, -1.66824747e+01, -1.67300516e+01, -1.67505526e+01, -1.68517453e+01, -1.71253653e+01, -1.75678692e+01, -1.82739482e+01};

const float INPUT_STDEVS[32] = 
   { 2.20001016e+00,  3.21028437e+00,  3.87447227e+00,  4.06303800e+00,  4.17243623e+00,  4.33675750e+00,  4.46849674e+00,  4.62265511e+00,
     4.57323127e+00,  4.45845353e+00,  4.37913642e+00,  4.29543066e+00,  4.20241054e+00,  4.14233502e+00,  4.09889076e+00,  4.02453706e+00,
     3.98647849e+00,  3.94190537e+00,  3.91626348e+00,  3.93815674e+00,  3.92972060e+00,  3.91623566e+00,  3.90864320e+00,  3.85885353e+00,
     3.80008495e+00,  3.74413825e+00,  3.68913930e+00,  3.63978736e+00,  3.58673602e+00,  3.56004917e+00,  3.58654384e+00,  3.43574113e+00};

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
