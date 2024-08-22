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
   {-2.20803794e+01, -2.10498204e+01, -2.01332458e+01, -1.98503641e+01, -1.96702833e+01, -1.94815025e+01, -1.93244276e+01, -1.89872353e+01,
    -1.90294171e+01, -1.90944312e+01, -1.90412636e+01, -1.90157023e+01, -1.90879129e+01, -1.90210206e+01, -1.89361924e+01, -1.89605638e+01,
    -1.88481369e+01, -1.88089601e+01, -1.88221544e+01, -1.86716544e+01, -1.86144728e+01, -1.85539447e+01, -1.84524235e+01, -1.84498015e+01,
    -1.84490421e+01, -1.84700759e+01, -1.85023571e+01, -1.85113608e+01, -1.85909222e+01, -1.87795075e+01, -1.90431321e+01, -1.95383847e+01};

const float INPUT_STDEVS[32] = 
   { 1.50980659e+00,  2.55662721e+00,  3.26256848e+00,  3.49234653e+00,  3.64175440e+00,  3.82773382e+00,  3.97485435e+00,  4.17046555e+00,
     4.13257386e+00,  4.03648201e+00,  3.98446850e+00,  3.91718947e+00,  3.82393333e+00,  3.78335525e+00,  3.76489474e+00,  3.69870426e+00,
     3.67093190e+00,  3.62545966e+00,  3.60124870e+00,  3.63486845e+00,  3.62076255e+00,  3.60298550e+00,  3.59497209e+00,  3.53529199e+00,
     3.46006539e+00,  3.38678962e+00,  3.30762921e+00,  3.24055474e+00,  3.17114118e+00,  3.09351798e+00,  2.99109959e+00,  2.69346914e+00};

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
