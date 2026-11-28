# Opus decoder C ABI

This crate exposes a C ABI around the BSD-3-Clause `opus-pure` package for the console audio
decoder (`xapu.dll`). The command protocol lives in `runtime/shims/xapu_chad.cpp`; this crate only
decodes one Opus packet into interleaved f32 PCM, resets the decoder, and (for tests) encodes a
synthetic tone.

## Build

```sh
cargo fmt --manifest-path runtime/opus_decoder/Cargo.toml -- --check
cargo test --manifest-path runtime/opus_decoder/Cargo.toml --locked
cargo build --release --manifest-path runtime/opus_decoder/Cargo.toml --locked
```

`runtime/CMakeLists.txt` builds this crate for `x86_64-pc-windows-msvc` at configure time when
`cargo` is on `PATH` and links it into `gta6rt` and `xapu_chad_test` with
`LEONIDA_HAVE_OPUS_DECODER` defined. The first build fetches `opus-pure` from crates.io. Set
`CARGO_NET_OFFLINE=true` for a configure that uses only the local Cargo cache.

Without `cargo`, when the build fails, or with the `cl.exe` fallback of `scripts/build_runtime.ps1`,
the XAPU path stays silent. Use `-DLEONIDA_REQUIRE_OPUS_DECODER=ON` to require the decoder in a
CMake build.

To link an already-built `leonida_opus_decoder.lib` without invoking `cargo` during CMake
configure, pass `-DLEONIDA_OPUS_DECODER_LIB=<path>`.

## Facts and credit

The `xapu.dll` command protocol (connect input/output, per-stream command budget, activate /
process / reset / deactivate, result order) follows Marvisak/VIctory
`crates/victory-hle/src/modules/apu.rs` (GPL-3.0-or-later), which this repository already credits for
other modules. `opus-pure` is BSD-3-Clause and stays a dependency; no code from it is copied.
