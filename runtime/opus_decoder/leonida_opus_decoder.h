#ifndef LEONIDA_OPUS_DECODER_H
#define LEONIDA_OPUS_DECODER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 120 ms at 48 kHz: the longest Opus packet, in samples per channel. */
#define LEONIDA_OPUS_MAX_FRAMES 5760u

typedef struct LeonidaOpusDecoder LeonidaOpusDecoder;

enum LeonidaOpusStatus {
  LEONIDA_OPUS_OK = 0,
  LEONIDA_OPUS_INVALID_ARGUMENT = 1,
  LEONIDA_OPUS_INVALID_FORMAT = 2,
  LEONIDA_OPUS_DECODER_FAILURE = 3,
  LEONIDA_OPUS_OUT_OF_MEMORY = 4,
  LEONIDA_OPUS_PANIC = 5,
  LEONIDA_OPUS_BUFFER_TOO_SMALL = 6,
};

int32_t leonida_opus_decoder_create(uint32_t sample_rate, uint32_t channels,
                                    LeonidaOpusDecoder **out_decoder);
uint32_t leonida_opus_decoder_channels(const LeonidaOpusDecoder *decoder);
int32_t leonida_opus_decode_packet(LeonidaOpusDecoder *decoder,
                                   const uint8_t *input, size_t input_len,
                                   float *out, size_t out_capacity_frames,
                                   uint32_t *out_frames);
int32_t leonida_opus_decoder_reset(LeonidaOpusDecoder *decoder);
void leonida_opus_decoder_destroy(LeonidaOpusDecoder *decoder);
/* Test helper: encodes frame_count interleaved f32 frames to one packet at 48 kHz. */
int32_t leonida_opus_encode_test(const float *input, uint32_t frame_count,
                                 uint32_t channels, uint8_t *out,
                                 size_t out_capacity, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif
