#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "arm_math.h"

#include "pico/stdlib.h"

#include "hardware/dma.h"
#include "inmp441.pio.h"

#include "tensorflow/lite/core/c/common.h"
#include "micro_model_settings.h"

#include "audio_preprocessor_int8_model_data.h"
#include "micro_speech_quantized_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define INMP441_PIN_SD (16)
#define INMP441_PIN_SCK (14)

#define DMA_CHANNEL (0)
#define DMA_CHANNEL_MASK (1u << DMA_CHANNEL)

#define DISP_SIG_SIZE (sizeof(disp_signs) - 1)

typedef enum
{
    catSilence = 0,
    catUnknown,
    catYes,
    catNo,
} category_e;

typedef struct
{
    bool state;
    size_t tres;
    size_t count;
    size_t timeout;
    category_e last_cat;
} filter_t;

static constexpr size_t kArenaSize = 9464;
alignas(16) static uint8_t g_audio_arena[kArenaSize];
alignas(16) static uint8_t g_speech_arena[kArenaSize];

using Features = int8_t[kFeatureCount][kFeatureSize];
static Features g_features;

static constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
static constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;

static int32_t samples[2][kAudioSampleStrideCount];

using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

static const char disp_signs[] = " .,-+*&NM#";

static char int8_to_disp(int8_t x)
{
    size_t i = (float)((size_t)x + 128) / 20;
    if (i >= DISP_SIG_SIZE)
    {
        i = DISP_SIG_SIZE - 1;
    }

    return disp_signs[i];
}

static void filter_update(filter_t *self, category_e category, int *res)
{
    *res = 0;
    if (self->state)
    {
        if (self->timeout == 0)
        {
            if (category != self->last_cat)
            {
                self->count++;
                if (self->count >= self->tres)
                {
                    self->state = false;
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
            self->timeout--;
            if (self->timeout == 0)
            {
                *res = 2;
            }
        }
    }
    else
    {
        if (category > catUnknown)
        {
            self->count++;
            // printf("%s - %d\n", kCategoryLabels[category], self->count);
            if (self->count >= self->tres)
            {
                self->state = true;
                self->last_cat = category;
                self->timeout = 50;
                self->count = 0;
                *res = 1;
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

static TfLiteStatus RegisterOps(MicroSpeechOpResolver &op_resolver)
{
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    return kTfLiteOk;
}

static TfLiteStatus RegisterOps(AudioPreprocessorOpResolver &op_resolver)
{
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
    TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
    TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
    TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
    TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
    TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
    TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
    return kTfLiteOk;
}

static TfLiteStatus get_feature(const int16_t *audio_data,
                                const size_t audio_data_size,
                                int8_t *feature_output,
                                tflite::MicroInterpreter *interpreter)
{
    TfLiteStatus status = kTfLiteError;

    TfLiteTensor *input = interpreter->input(0);

    assert(input != nullptr);
    assert(kAudioSampleDurationCount == audio_data_size);
    assert(kAudioSampleDurationCount == input->dims->data[input->dims->size - 1]);

    TfLiteTensor *output = interpreter->output(0);
    assert(output != nullptr);
    assert(kFeatureSize == output->dims->data[output->dims->size - 1]);

    std::copy_n(audio_data, audio_data_size,
                tflite::GetTensorData<int16_t>(input));
    status = interpreter->Invoke();
    assert(status == kTfLiteOk);
    std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize,
                feature_output);

    return status;
}

static TfLiteStatus get_inference_results(const Features &features, tflite::MicroInterpreter *interpreter, float *category_predictions)
{
    TfLiteStatus status = kTfLiteError;

    TfLiteTensor *input = interpreter->input(0);
    assert(input != nullptr);

    // check input shape is compatible with our feature data size
    assert(kFeatureElementCount ==
           input->dims->data[input->dims->size - 1]);

    TfLiteTensor *output = interpreter->output(0);
    assert(output != nullptr);

    // check output shape is compatible with our number of prediction categories
    assert(kCategoryCount == output->dims->data[output->dims->size - 1]);

    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;

    std::copy_n(&features[0][0], kFeatureElementCount,
                tflite::GetTensorData<int8_t>(input));
    status = interpreter->Invoke();
    assert(status == kTfLiteOk);

    // Dequantize output values
    // printf("MicroSpeech category predictions\n");
    for (int i = 0; i < kCategoryCount; i++)
    {
        category_predictions[i] =
            (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
            output_scale;
    }

    return status;
}

static void print_features(int8_t *feature)
{
    printf("|");
    for (size_t i = 0; i < kFeatureSize; i++)
    {
        printf("%c", int8_to_disp(feature[i]));
    }
    printf("|");
}

int main(int argc, char **argv)
{
    PIO pio = pio0;
    int sm;
    size_t bi = 0;
    TfLiteStatus status;
    int16_t audio_data[kAudioSampleDurationCount] = {0};
    filter_t out_filter = {
        .state = false,
        .tres = 5,
        .count = 0,
    };

    stdio_init_all();
    set_sys_clock_khz(250000, true);
    stdio_uart_init_full(uart0, 921600, 0, 1);

    (void)status;

    uint offset = pio_add_program(pio, &inmp441_program);
    sm = pio_claim_unused_sm(pio, true);

    dma_claim_mask(DMA_CHANNEL_MASK);
    dma_channel_config channel_config = dma_channel_get_default_config(DMA_CHANNEL);
    channel_config_set_dreq(&channel_config, pio_get_dreq(pio, sm, false));
    channel_config_set_transfer_data_size(&channel_config, DMA_SIZE_32);
    channel_config_set_write_increment(&channel_config, true);
    channel_config_set_read_increment(&channel_config, false);

    dma_channel_configure(DMA_CHANNEL,
                          &channel_config,
                          NULL,
                          &pio->rxf[sm],
                          kAudioSampleStrideCount,
                          false);

    inmp441_program_init(pio, sm, offset, kAudioSampleFrequency, INMP441_PIN_SD, INMP441_PIN_SCK);

    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    const uint YES_LED_PIN = 18;
    const uint NO_LED_PIN = 19;

    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    gpio_init(YES_LED_PIN);
    gpio_set_dir(YES_LED_PIN, GPIO_OUT);
    gpio_put(YES_LED_PIN, 1);

    gpio_init(NO_LED_PIN);
    gpio_set_dir(NO_LED_PIN, GPIO_OUT);
    gpio_put(NO_LED_PIN, 1);

    // Initialize audio processor
    const tflite::Model *audioProcModel =
        tflite::GetModel(g_audio_preprocessor_int8_model_data);
    assert(audioProcModel->version() == TFLITE_SCHEMA_VERSION);

    AudioPreprocessorOpResolver audio_op_resolver;
    status = RegisterOps(audio_op_resolver);
    assert(status == kTfLiteOk);

    tflite::MicroInterpreter audioProcInterpreter(audioProcModel, audio_op_resolver, g_audio_arena, kArenaSize);
    status = audioProcInterpreter.AllocateTensors();
    assert(status == kTfLiteOk);

    printf("AudioPreprocessor model arena size = %u\n",
           audioProcInterpreter.arena_used_bytes());

    // Initialize the NN model
    const tflite::Model *speechModel =
        tflite::GetModel(g_micro_speech_quantized_model_data);
    assert(speechModel->version() == TFLITE_SCHEMA_VERSION);

    MicroSpeechOpResolver op_resolver;
    status = RegisterOps(op_resolver);
    assert(status == kTfLiteOk);

    tflite::MicroInterpreter speechInterpreter(speechModel, op_resolver, g_speech_arena, kArenaSize);
    status = speechInterpreter.AllocateTensors();
    assert(status == kTfLiteOk);

    printf("MicroSpeech model arena size = %u\n",
           speechInterpreter.arena_used_bytes());

    printf("Starting\n");

    dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi], true);

    int res;
    float category_predictions[kCategoryCount];
    category_e category;
    const int strideOffset = (kAudioSampleDurationCount - kAudioSampleStrideCount);
    const float alpha = 0.97;

    while (true)
    {
        gpio_put(LED_PIN, 1);
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);
        dma_channel_set_write_addr(DMA_CHANNEL, (void *)samples[bi ^ 1], true);
        gpio_put(LED_PIN, 0);

        uint32_t start_time = time_us_32();
        memmove(audio_data, &audio_data[kAudioSampleStrideCount], (kAudioSampleDurationCount - kAudioSampleStrideCount) * sizeof(audio_data[0]));

        for (size_t i = 0; i < kAudioSampleStrideCount; i++)
        {
            samples[bi][i] >>= 16;
            audio_data[i + strideOffset] = *((int32_t *)&samples[bi][i]);
            // preemphasis
            audio_data[i + strideOffset] -= alpha * audio_data[i + strideOffset - 1];
        }

        memmove(g_features[0], g_features[1], kFeatureSize * (kFeatureCount - 1) * sizeof(int8_t));

        status = get_feature(audio_data, kAudioSampleDurationCount,
                             g_features[kFeatureCount - 1], &audioProcInterpreter);
        assert(status == kTfLiteOk);

        status = get_inference_results(g_features, &speechInterpreter, category_predictions);
        assert(status == kTfLiteOk);

        category = catSilence;
        for (int i = 0; i < kCategoryCount; i++)
        {
            if (((i == 2) && (category_predictions[i] >= 0.62)) || ((i == 3) && (category_predictions[i] >= 0.85)))
            {
                category = (category_e)i;
            }
        }

        filter_update(&out_filter, category, &res);
        if (res == 1)
        {
            if (category == 2)
            {
                gpio_put(NO_LED_PIN, 0);
            }
            else if (category == 3)
            {
                gpio_put(YES_LED_PIN, 0);
            }
        }
        else if (res == 2)
        {
            gpio_put(NO_LED_PIN, 1);
            gpio_put(YES_LED_PIN, 1);
        }

        print_features(g_features[kFeatureCount - 1]);
        for (int i = 0; i < kCategoryCount; i++)
        {
            char mark = ' ';
            if ((out_filter.state) && (out_filter.last_cat == (category_e)i))
            {
                mark = '>';
            }
            printf("  %.4f %c%s,", static_cast<double>(category_predictions[i]), mark,
                   kCategoryLabels[i]);
        }
        printf(" | %.3f ms |\n", (time_us_32() - start_time) / 1000.0);

        bi ^= 1;
    }
}
