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
   {-2.19537185e+01, -2.08062092e+01, -1.98128379e+01, -1.95351861e+01, -1.93877978e+01, -1.91910107e+01, -1.90876014e+01, -1.87987882e+01,
    -1.88987945e+01, -1.89908786e+01, -1.89535294e+01, -1.89469632e+01, -1.90421560e+01, -1.89754335e+01, -1.88555629e+01, -1.88489780e+01,
    -1.87705871e+01, -1.87722245e+01, -1.87616748e+01, -1.85906146e+01, -1.85888466e+01, -1.85712229e+01, -1.84645891e+01, -1.84787809e+01,
    -1.85427585e+01, -1.86246621e+01, -1.87080788e+01, -1.87635485e+01, -1.88147655e+01, -1.88572737e+01, -1.90581653e+01, -1.98974529e+01};

const float INPUT_STDEVS[32] = 
   { 1.59539227e+00,  2.70295791e+00,  3.42594533e+00,  3.66004782e+00,  3.79484445e+00,  3.99340194e+00,  4.09076405e+00,  4.23794266e+00,
     4.15982048e+00,  4.02225128e+00,  3.94903479e+00,  3.89277474e+00,  3.80861770e+00,  3.77094379e+00,  3.78045410e+00,  3.75068293e+00,
     3.72967475e+00,  3.69791716e+00,  3.73229353e+00,  3.82730964e+00,  3.80400104e+00,  3.78268974e+00,  3.83031155e+00,  3.80337893e+00,
     3.71575590e+00,  3.64716962e+00,  3.59968490e+00,  3.57114090e+00,  3.54200996e+00,  3.51933571e+00,  3.44956075e+00,  3.15522720e+00};

// clang-format on

static inline float sigmoidf(float n)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
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
        memset(output, 0, sizeof(float) * 64);
        rnn0_process(input[k], hidden, output);
        memcpy(hidden, output, sizeof(float) * 64);
    }
}

static void rnn1_process(const float input[64], const float hidden[32], float output[32])
{
    float z;
    float c;

    for (size_t i = 0; i < 64; i++)
    {
        for (size_t j = 0; j < 32; j++)
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
    static float rnn1_input_hist[9][64];
    static size_t rnn1_hist_idx;

    float rnn1_hidden[32] = {0.0};

    memcpy(rnn1_input_hist[rnn1_hist_idx], input, sizeof(sha_rnn_rnn1_input_t));
    memset(output, 0, sizeof(sha_rnn_fc_input_t));

    for (size_t i = 0; i < 9; i++)
    {
        size_t j = (rnn1_hist_idx + 1 + i) % 9;
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

    for (size_t j = 0; j < 6; j++)
    {
        for (size_t i = 0; i < 32; i++)
        {
            output[j] += input[i] * FC_W[j][i];
        }
        output[j] += FC_B[j];
    }
}

void sha_rnn_get_max_logit(const sha_rnn_output_t input, float *max_logit, size_t *max_idx)
{
    *max_logit = input[0];
    *max_idx = 0;

    for (size_t j = 0; j < 6; j++)
    {
        if (input[j] > *max_logit)
        {
            *max_logit = input[j];
            *max_idx = j;
        }
    }
}

void sha_rnn_process(const sha_rnn_input_t input, float *max_logit, size_t *max_idx)
{
    float output[64] = {0.0f};
    float output2[32] = {0.0f};
    float output3[6] = {0.0f};

    sha_rnn_rnn0_process(input, output);
    sha_rnn_rnn1_process(output, output2);
    sha_rnn_fc_process(output2, output3);
    sha_rnn_get_max_logit(output3, max_logit, max_idx);
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
