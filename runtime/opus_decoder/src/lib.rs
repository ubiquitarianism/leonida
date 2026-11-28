#![deny(unsafe_op_in_unsafe_fn)]

//! C ABI around the pure-Rust `opus-pure` decoder, for the console audio decoder (`xapu.dll`).
//!
//! The command protocol lives in `runtime/shims/xapu_chad.cpp`; this crate only turns one Opus
//! packet into interleaved f32 PCM, resets the decoder, and (for tests) encodes a synthetic tone.
//! `opus-pure` is BSD-3-Clause; this wrapper is GPL-3.0-or-later, like the rest of the runtime.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Mutex;

use opus_pure::{Application, OpusDecoder, OpusEncoder};

pub const LEONIDA_OPUS_OK: i32 = 0;
pub const LEONIDA_OPUS_INVALID_ARGUMENT: i32 = 1;
pub const LEONIDA_OPUS_INVALID_FORMAT: i32 = 2;
pub const LEONIDA_OPUS_DECODER_FAILURE: i32 = 3;
pub const LEONIDA_OPUS_OUT_OF_MEMORY: i32 = 4;
pub const LEONIDA_OPUS_PANIC: i32 = 5;
pub const LEONIDA_OPUS_BUFFER_TOO_SMALL: i32 = 6;

/// 120 ms at 48 kHz, the longest Opus packet, in samples per channel.
pub const LEONIDA_OPUS_MAX_FRAMES: usize = 5760;

#[repr(C)]
pub struct LeonidaOpusDecoder {
    decoder: Mutex<OpusDecoder>,
    channels: usize,
}

#[no_mangle]
pub unsafe extern "C" fn leonida_opus_decoder_channels(decoder: *const LeonidaOpusDecoder) -> u32 {
    if decoder.is_null() {
        return 0;
    }
    unsafe { (*decoder).channels as u32 }
}

/// Creates a decoder for one mono or stereo Opus stream at `sample_rate` (the title uses 48000).
///
/// # Safety
/// `out` must be writable for one pointer. The returned decoder must be destroyed once, after
/// every call that uses it has finished.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_decoder_create(
    sample_rate: u32,
    channels: u32,
    out: *mut *mut LeonidaOpusDecoder,
) -> i32 {
    if out.is_null() {
        return LEONIDA_OPUS_INVALID_ARGUMENT;
    }
    unsafe { *out = ptr::null_mut() };
    if sample_rate == 0 || sample_rate > i32::MAX as u32 || !(1..=2).contains(&channels) {
        return LEONIDA_OPUS_INVALID_FORMAT;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        OpusDecoder::new(sample_rate as i32, channels as usize).map(|decoder| {
            Box::new(LeonidaOpusDecoder {
                decoder: Mutex::new(decoder),
                channels: channels as usize,
            })
        })
    })) {
        Ok(Ok(decoder)) => {
            unsafe { *out = Box::into_raw(decoder) };
            LEONIDA_OPUS_OK
        }
        Ok(Err(_)) => LEONIDA_OPUS_INVALID_FORMAT,
        Err(_) => LEONIDA_OPUS_PANIC,
    }
}

/// Decodes one Opus packet into `out` as interleaved f32 frames.
///
/// # Safety
/// `decoder` must come from `leonida_opus_decoder_create`. `input` must reference `input_len`
/// readable bytes; `out` must hold `out_capacity_frames * channels` floats; `out_frames` must be
/// writable. Do not destroy the decoder concurrently with this call.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_decode_packet(
    decoder: *mut LeonidaOpusDecoder,
    input: *const u8,
    input_len: usize,
    out: *mut f32,
    out_capacity_frames: usize,
    out_frames: *mut u32,
) -> i32 {
    if decoder.is_null()
        || input.is_null()
        || input_len == 0
        || out.is_null()
        || out_frames.is_null()
    {
        return LEONIDA_OPUS_INVALID_ARGUMENT;
    }
    let decoder = unsafe { &*decoder };
    let channels = decoder.channels;
    let Ok(mut guard) = decoder.decoder.lock() else {
        return LEONIDA_OPUS_DECODER_FAILURE;
    };
    let packet = unsafe { std::slice::from_raw_parts(input, input_len) };
    let mut pcm = vec![0f32; LEONIDA_OPUS_MAX_FRAMES * channels];
    match guard.decode(packet, LEONIDA_OPUS_MAX_FRAMES, &mut pcm) {
        Ok(frames) => {
            if frames > out_capacity_frames {
                return LEONIDA_OPUS_BUFFER_TOO_SMALL;
            }
            let samples = frames * channels;
            unsafe {
                ptr::copy_nonoverlapping(pcm.as_ptr(), out, samples);
                *out_frames = frames as u32;
            }
            LEONIDA_OPUS_OK
        }
        Err(_) => LEONIDA_OPUS_DECODER_FAILURE,
    }
}

/// # Safety
/// `decoder` must come from `leonida_opus_decoder_create` and not be used concurrently.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_decoder_reset(decoder: *mut LeonidaOpusDecoder) -> i32 {
    if decoder.is_null() {
        return LEONIDA_OPUS_INVALID_ARGUMENT;
    }
    let decoder = unsafe { &*decoder };
    let Ok(mut guard) = decoder.decoder.lock() else {
        return LEONIDA_OPUS_DECODER_FAILURE;
    };
    let _ = guard.reset_state();
    LEONIDA_OPUS_OK
}

/// # Safety
/// `decoder` must be a live handle and must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_decoder_destroy(decoder: *mut LeonidaOpusDecoder) {
    if !decoder.is_null() {
        unsafe { drop(Box::from_raw(decoder)) };
    }
}

/// Test helper: encodes `frame_count` interleaved f32 frames to one Opus packet at 48 kHz.
///
/// # Safety
/// `input` must hold `frame_count * channels` floats; `out` must hold `out_capacity` bytes;
/// `out_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn leonida_opus_encode_test(
    input: *const f32,
    frame_count: u32,
    channels: u32,
    out: *mut u8,
    out_capacity: usize,
    out_len: *mut usize,
) -> i32 {
    if input.is_null()
        || out.is_null()
        || out_len.is_null()
        || frame_count == 0
        || !(1..=2).contains(&channels)
    {
        return LEONIDA_OPUS_INVALID_ARGUMENT;
    }
    let samples = frame_count as usize * channels as usize;
    let input = unsafe { std::slice::from_raw_parts(input, samples) };
    let mut encoder = match OpusEncoder::new(48_000, channels as usize, Application::Audio) {
        Ok(encoder) => encoder,
        Err(_) => return LEONIDA_OPUS_INVALID_FORMAT,
    };
    let mut packet = vec![0u8; out_capacity.max(4000)];
    match encoder.encode(input, frame_count as usize, &mut packet) {
        Ok(bytes) if bytes <= out_capacity => {
            unsafe {
                ptr::copy_nonoverlapping(packet.as_ptr(), out, bytes);
                *out_len = bytes;
            }
            LEONIDA_OPUS_OK
        }
        Ok(_) => LEONIDA_OPUS_BUFFER_TOO_SMALL,
        Err(_) => LEONIDA_OPUS_DECODER_FAILURE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_decodes_a_tone() {
        const FRAMES: u32 = 960;
        const CHANNELS: u32 = 2;
        let mut tone = Vec::with_capacity((FRAMES * CHANNELS) as usize);
        for i in 0..FRAMES {
            let sample = ((i as f32) * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5;
            tone.push(sample);
            tone.push(sample);
        }
        let mut packet = [0u8; 1500];
        let mut packet_len = 0usize;
        // SAFETY: the buffers are owned by this test.
        let encoded = unsafe {
            leonida_opus_encode_test(
                tone.as_ptr(),
                FRAMES,
                CHANNELS,
                packet.as_mut_ptr(),
                packet.len(),
                &mut packet_len,
            )
        };
        assert_eq!(encoded, LEONIDA_OPUS_OK);
        assert!(packet_len > 0);

        let mut decoder = ptr::null_mut();
        // SAFETY: `decoder` is writable for one pointer.
        assert_eq!(
            unsafe { leonida_opus_decoder_create(48_000, CHANNELS, &mut decoder) },
            LEONIDA_OPUS_OK
        );
        let mut pcm = vec![0f32; LEONIDA_OPUS_MAX_FRAMES * CHANNELS as usize];
        let mut frames = 0u32;
        // SAFETY: the packet and PCM buffers are owned by this test.
        let decoded = unsafe {
            leonida_opus_decode_packet(
                decoder,
                packet.as_ptr(),
                packet_len,
                pcm.as_mut_ptr(),
                LEONIDA_OPUS_MAX_FRAMES,
                &mut frames,
            )
        };
        assert_eq!(decoded, LEONIDA_OPUS_OK);
        assert_eq!(frames, FRAMES);
        let energy: f32 =
            pcm[..(frames * CHANNELS) as usize].iter().map(|s| s * s).sum::<f32>() / (frames * CHANNELS) as f32;
        assert!(energy > 0.01, "decoded a tone, energy {energy}");
        // SAFETY: created above, destroyed once.
        unsafe { leonida_opus_decoder_destroy(decoder) };
    }
}
