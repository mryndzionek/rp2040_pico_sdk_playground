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
   {-2.19515412e+01, -2.08044265e+01, -1.98109013e+01, -1.95327326e+01, -1.93849298e+01, -1.91873826e+01, -1.90829301e+01, -1.87938066e+01,
    -1.88934953e+01, -1.89848798e+01, -1.89468397e+01, -1.89397208e+01, -1.90345433e+01, -1.89673649e+01, -1.88475054e+01, -1.88411368e+01,
    -1.87625272e+01, -1.87641597e+01, -1.87535152e+01, -1.85824503e+01, -1.85806894e+01, -1.85629260e+01, -1.84559814e+01, -1.84696631e+01,
    -1.85330695e+01, -1.86146780e+01, -1.86974835e+01, -1.87519411e+01, -1.88026378e+01, -1.88447854e+01, -1.90450149e+01, -1.98818332e+01};

const float INPUT_STDEVS[32] = 
   { 1.59858290e+00,  2.70366216e+00,  3.42592312e+00,  3.66016417e+00,  3.79519976e+00,  3.99374771e+00,  4.09164130e+00,  4.23915963e+00,
     4.16159911e+00,  4.02485525e+00,  3.95268956e+00,  3.89733413e+00,  3.81398529e+00,  3.77745963e+00,  3.78735383e+00,  3.75776813e+00,
     3.73754094e+00,  3.70637776e+00,  3.74105971e+00,  3.83593445e+00,  3.81301609e+00,  3.79236109e+00,  3.84044565e+00,  3.81449422e+00,
     3.72874419e+00,  3.66162727e+00,  3.61492548e+00,  3.58691758e+00,  3.55774937e+00,  3.53504266e+00,  3.46625903e+00,  3.17779207e+00};

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
        for (size_t j = 0; j < 64; j++)
        {
            for (size_t i = 0; i < 32; i += 4)
            {
                output[t][j] += GRNN0_W[j][i] * input[t][i];
                output[t][j] += GRNN0_W[j][i + 1] * input[t][i + 1];
                output[t][j] += GRNN0_W[j][i + 2] * input[t][i + 2];
                output[t][j] += GRNN0_W[j][i + 3] * input[t][i + 3];
            }
        }

        for (size_t j = 0; j < 64; j++)
        {
            for (size_t i = 0; i < 64; i += 4)
            {
                output[t][j] += GRNN0_U[j][i] * hidden[t][i];
                output[t][j] += GRNN0_U[j][i + 1] * hidden[t][i + 1];
                output[t][j] += GRNN0_U[j][i + 2] * hidden[t][i + 2];
                output[t][j] += GRNN0_U[j][i + 3] * hidden[t][i + 3];
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
        for (size_t j = 0; j < 32; j++)
        {
            for (size_t i = 0; i < 64; i += 4)
            {
                output[t][j] += GRNN1_W[j][i] * input[t][i];
                output[t][j] += GRNN1_W[j][i + 1] * input[t][i + 1];
                output[t][j] += GRNN1_W[j][i + 2] * input[t][i + 2];
                output[t][j] += GRNN1_W[j][i + 3] * input[t][i + 3];
            }
        }

        if (t > 0)
        {
            for (size_t j = 0; j < 32; j++)
            {
                for (size_t i = 0; i < 32; i += 4)
                {
                    output[t][j] += GRNN1_U[j][i] * output[t - 1][i];
                    output[t][j] += GRNN1_U[j][i + 1] * output[t - 1][i + 1];
                    output[t][j] += GRNN1_U[j][i + 2] * output[t - 1][i + 2];
                    output[t][j] += GRNN1_U[j][i + 3] * output[t - 1][i + 3];
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
            for (size_t i = 0; i < 32; i += 4)
            {
                output[t][j] += input[t][i] * FC_W[j][i];
                output[t][j] += input[t][i + 1] * FC_W[j][i + 1];
                output[t][j] += input[t][i + 2] * FC_W[j][i + 2];
                output[t][j] += input[t][i + 3] * FC_W[j][i + 3];
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
