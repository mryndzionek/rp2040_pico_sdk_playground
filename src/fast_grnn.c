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
   {-2.19515185e+01, -2.08044217e+01, -1.98109261e+01, -1.95327370e+01, -1.93849368e+01, -1.91873338e+01, -1.90828581e+01, -1.87937778e+01,
    -1.88934836e+01, -1.89848071e+01, -1.89467470e+01, -1.89396496e+01, -1.90344348e+01, -1.89672144e+01, -1.88473633e+01, -1.88410380e+01,
    -1.87624591e+01, -1.87641735e+01, -1.87535443e+01, -1.85824398e+01, -1.85806396e+01, -1.85628294e+01, -1.84558778e+01, -1.84695477e+01,
    -1.85329455e+01, -1.86145172e+01, -1.86973441e+01, -1.87517974e+01, -1.88024679e+01, -1.88446191e+01, -1.90448788e+01, -1.98816874e+01};

const float INPUT_STDEVS[32] = 
   { 1.59865445e+00,  2.70373763e+00,  3.42596410e+00,  3.66015493e+00,  3.79518352e+00,  3.99374483e+00,  4.09168577e+00,  4.23915447e+00,
     4.16156086e+00,  4.02488782e+00,  3.95275120e+00,  3.89742751e+00,  3.81411585e+00,  3.77758735e+00,  3.78746846e+00,  3.75780111e+00,
     3.73753786e+00,  3.70626031e+00,  3.74089806e+00,  3.83585089e+00,  3.81300976e+00,  3.79241512e+00,  3.84050112e+00,  3.81456447e+00,
     3.72884485e+00,  3.66182427e+00,  3.61507565e+00,  3.58707859e+00,  3.55794731e+00,  3.53523306e+00,  3.46639427e+00,  3.17800313e+00};

// clang-format on

static inline float sigmoidf(float n)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

static void rnn0_process(const float input[9][32], const float hidden[9][64], float output[9][64])
{
    float z;
    float c;

    for (size_t t = 0; t < 9; t++)
    {
        {
            float tmp[8] = {0.0};

            for (size_t j = 0; j < 8; j++)
            {
                for (size_t i = 0; i < 32; i++)
                {
                    tmp[j] += GRNN0_W1[j][i] * input[t][i];
                }
            }

            for (size_t i = 0; i < 8; i++)
            {
                for (size_t j = 0; j < 64; j++)
                {
                    output[t][j] += GRNN0_W2[j][i] * tmp[i];
                }
            }
        }

        {
            float tmp[16] = {0.0};

            for (size_t j = 0; j < 16; j++)
            {
                for (size_t i = 0; i < 64; i++)
                {
                    tmp[j] += GRNN0_U1[j][i] * hidden[t][i];
                }
            }

            for (size_t i = 0; i < 16; i++)
            {
                for (size_t j = 0; j < 64; j++)
                {
                    output[t][j] += GRNN0_U2[j][i] * tmp[i];
                }
            }
        }

        for (size_t j = 0; j < 64; j++)
        {
            z = output[t][j] + GRNN0_BIAS_GATE[j];
            z = sigmoidf(z);
            c = output[t][j] + GRNN0_BIAS_UPDATE[j];
            c = tanhf(c);

            output[t][j] = z * hidden[t][j] + (sigmoidf(GRNN0_ZETA) * (1.0 - z) + sigmoidf(GRNN0_NU)) * c;
        }
    }
}

void sha_rnn_rnn0_process(const sha_rnn_input_t input, sha_rnn_rnn1_input_t output)
{
    float frame[9][32] = {{0.0f}};
    float hidden[9][64] = {{0.0f}};

    for (size_t k = 0; k < SHARNN_BRICK_SIZE; k++)
    {
        for (size_t i = 0; i < 9; i++)
        {
            const float *src = input[k + (i * SHARNN_BRICK_SIZE)];
            for (size_t j = 0; j < 32; j++)
            {
                frame[i][j] = src[j];
            }
        }

        memset(output, 0, sizeof(float) * 9 * 64);
        rnn0_process(frame, hidden, output);
        memcpy(hidden, output, sizeof(float) * 9 * 64);
    }
}

void sha_rnn_rnn1_process(const sha_rnn_rnn1_input_t input, sha_rnn_fc_input_t output)
{
    float z;
    float c;

    for (size_t t = 0; t < 9; t++)
    {
        {
            float tmp[16] = {0.0};

            for (size_t j = 0; j < 16; j++)
            {
                for (size_t i = 0; i < 64; i++)
                {
                    tmp[j] += GRNN1_W1[j][i] * input[t][i];
                }
            }

            for (size_t i = 0; i < 16; i++)
            {
                for (size_t j = 0; j < 32; j++)
                {
                    output[t][j] += GRNN1_W2[j][i] * tmp[i];
                }
            }
        }

        if (t > 0)
        {
            float tmp[8] = {0.0};

            for (size_t j = 0; j < 8; j++)
            {
                for (size_t i = 0; i < 32; i++)
                {
                    tmp[j] += GRNN1_U1[j][i] * output[t - 1][i];
                }
            }
            for (size_t i = 0; i < 8; i++)
            {
                for (size_t j = 0; j < 32; j++)
                {
                    output[t][j] += GRNN1_U2[j][i] * tmp[i];
                }
            }
        }

        for (size_t j = 0; j < 32; j++)
        {
            z = output[t][j] + GRNN1_BIAS_GATE[j];
            z = sigmoidf(z);
            c = output[t][j] + GRNN1_BIAS_UPDATE[j];
            c = tanhf(c);

            output[t][j] = z * (t > 0 ? output[t - 1][j] : 0.0f) + (sigmoidf(GRNN1_ZETA) * (1.0 - z) + sigmoidf(GRNN1_NU)) * c;
        }
    }
}

void sha_rnn_fc_process(const sha_rnn_fc_input_t input, sha_rnn_output_t output)
{
    memset(output, 0, 9 * 6 * sizeof(float));

    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 6; j++)
        {
            for (size_t i = 0; i < 32; i++)
            {
                output[t][j] += input[t][i] * FC_W[j][i];
            }
            output[t][j] += FC_B[j];
        }
    }
}

void sha_rnn_get_max_logit(const sha_rnn_output_t input, float *max_logit, size_t *max_idx)
{
    *max_logit = input[0][0];
    *max_idx = 0;

    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 6; j++)
        {
            if (input[t][j] > *max_logit)
            {
                *max_logit = input[t][j];
                *max_idx = j;
            }
        }
    }
}

void sha_rnn_process(const sha_rnn_input_t input, float *max_logit, size_t *max_idx)
{
    float output[9][64] = {{0.0f}};
    float output2[9][32] = {{0.0f}};
    float output3[9][6] = {{0.0f}};

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
