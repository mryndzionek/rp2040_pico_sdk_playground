#ifndef __FBANK_H__
#define __FBANK_H__

#define SAMPLE_LEN (16000UL)
#define NUM_FRAMES (99UL)
#define NUM_FILT (32UL)
#define FRAME_STEP (160UL)
#define FRAME_LEN (400UL)

#define NUM_LABELS (6)

void fbank_init(void);
void fbank(float *input, float (*output)[NUM_FILT], size_t size);
char const *const fbank_label_idx_to_str(size_t label);

#endif // __FBANK_H__