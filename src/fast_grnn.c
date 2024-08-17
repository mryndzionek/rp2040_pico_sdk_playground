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
   {-2.19509236e+01, -2.08036722e+01, -1.98098399e+01, -1.95315112e+01, -1.93835309e+01, -1.91858462e+01, -1.90817526e+01, -1.87921466e+01,
    -1.88916665e+01, -1.89826100e+01, -1.89447061e+01, -1.89380033e+01, -1.90328330e+01, -1.89655568e+01, -1.88461177e+01, -1.88400139e+01,
    -1.87615224e+01, -1.87633237e+01, -1.87530108e+01, -1.85819958e+01, -1.85805503e+01, -1.85631627e+01, -1.84566843e+01, -1.84705400e+01,
    -1.85335962e+01, -1.86155988e+01, -1.86984266e+01, -1.87530468e+01, -1.88037968e+01, -1.88459160e+01, -1.90462253e+01, -1.98828904e+01};

const float INPUT_STDEVS[32] = 
   { 1.59909965e+00,  2.70424305e+00,  3.42706803e+00,  3.66150334e+00,  3.79670104e+00,  3.99551076e+00,  4.09254905e+00,  4.23990748e+00,
     4.16237435e+00,  4.02640128e+00,  3.95418326e+00,  3.89847204e+00,  3.81504459e+00,  3.77877013e+00,  3.78803548e+00,  3.75840238e+00,
     3.73858670e+00,  3.70742574e+00,  3.74186200e+00,  3.83679068e+00,  3.81366515e+00,  3.79282824e+00,  3.84046295e+00,  3.81423285e+00,
     3.72891246e+00,  3.66153880e+00,  3.61471378e+00,  3.58643547e+00,  3.55685932e+00,  3.53377246e+00,  3.46451099e+00,  3.17561747e+00};

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
    static float output[64] = {0.0f};
    static float output2[32] = {0.0f};
    static float output3[6] = {0.0f};

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
