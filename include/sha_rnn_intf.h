#ifndef __SHA_RNN_INTF__
#define __SHA_RNN_INTF__

#include <stddef.h>

#define SHARNN_BRICK_SIZE (11)
#define SHARNN_FEATURE_DIM0 (99)
#define SHARNN_FEATURE_DIM1 (32)
#define SHARNN_INPUT_DIM0 (SHARNN_FEATURE_DIM0 / SHARNN_BRICK_SIZE)
#define SHARNN_HIDD_DIM0 (64)
#define SHARNN_HIDD_DIM1 (32)
#define SHARNN_OUTPUT_DIM (6)

typedef float sha_rnn_input_t[SHARNN_FEATURE_DIM0][SHARNN_FEATURE_DIM1];
typedef float sha_rnn_rnn1_input_t[SHARNN_INPUT_DIM0][SHARNN_HIDD_DIM0];
typedef float sha_rnn_fc_input_t[SHARNN_INPUT_DIM0][SHARNN_HIDD_DIM1];
typedef float sha_rnn_output_t[SHARNN_INPUT_DIM0][SHARNN_OUTPUT_DIM];

void sha_rnn_norm(float *input, size_t num);
void sha_rnn_rnn0_process(const sha_rnn_input_t input, sha_rnn_rnn1_input_t output);
void sha_rnn_rnn1_process(const sha_rnn_rnn1_input_t input, sha_rnn_fc_input_t output);
void sha_rnn_fc_process(const sha_rnn_fc_input_t input, sha_rnn_output_t output);
void sha_rnn_get_max_logit(const sha_rnn_output_t input, float *max_logit, size_t *max_idx);
void sha_rnn_process(const sha_rnn_input_t input, float *max_logit, size_t *max_idx);

#endif // __SHA_RNN_INTF__
