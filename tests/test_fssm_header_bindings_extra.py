"""Additional edge-case tests for tools/fssm_header.py."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fssm_header as fh


def test_wave_size_handles_short_non_magic_and_wave32_flag():
    assert fh.wave_size(b"FSSM" + b"\x00" * 16) is None
    non_magic = bytearray(0x100)
    assert fh.wave_size(bytes(non_magic)) is None

    w64 = bytearray(0x100)
    w64[0:4] = fh.MAGIC
    assert fh.wave_size(bytes(w64)) == 64

    w32 = bytearray(0x100)
    w32[0:4] = fh.MAGIC
    struct.pack_into("<I", w32, fh.WAVE_FLAGS_OFFSET, fh.FLAG_WAVE32)
    assert fh.wave_size(bytes(w32)) == 32


def test_binding_table_with_unset_code_end_and_trailing_bytes():
    code_off = 0x160
    code_sz = 0x20
    tail_off = code_off + code_sz
    name = b"PS_ClearF32_Wrapped"
    tail = (
        struct.pack("<4I", *([fh.TAIL_FILLER] * 4))
        + struct.pack("<2I", 1, 0)
        + struct.pack("<5I", 2, 0, 8, 0, 0)
        + struct.pack("<IIHH", fh.BINDING_SENTINEL, 0, 0, len(name))
        + name
        + b"\xaa\xbb"
    )
    blob = bytearray(tail_off)
    blob[0:4] = fh.MAGIC
    struct.pack_into("<Q", blob, fh.CODE_OFFSET, code_off)
    struct.pack_into("<Q", blob, fh.CODE_SIZE, code_sz)
    struct.pack_into("<Q", blob, fh.CODE_END, fh.CODE_END_UNSET)
    full = bytes(blob) + tail

    parsed = fh.binding_table(full)
    assert parsed is not None
    assert parsed["code_end"] == tail_off
    assert parsed["header"] == 1
    assert parsed["records"] == [(2, 0, 8, 0, 0)]
    assert parsed["name"] == "PS_ClearF32_Wrapped"
    assert parsed["trailing"] == 2

    # Missing filler must return None.
    broken = bytearray(full)
    broken[tail_off] = 1
    assert fh.binding_table(bytes(broken)) is None


def test_interpolants_and_read_methods_multi_entry():
    blob = bytearray(fh.TABLE_OFFSET + 3 * fh.RECORD_SIZE)
    blob[0:4] = fh.MAGIC
    # Entry 0: flat (READ_MOV), 32-bit
    blob[fh.TABLE_OFFSET : fh.TABLE_OFFSET + 8] = bytes([fh.RECORD_KIND, 1, 0xF, 0, 0, 0, 1, fh.READ_MOV])
    # Entry 1: interpolated (READ_INTERP), 16-bit (FLAG_FP16)
    blob[fh.TABLE_OFFSET + 8 : fh.TABLE_OFFSET + 16] = bytes(
        [fh.RECORD_KIND, 2, 0x3, 0, 0, 0, 2, fh.READ_INTERP | fh.FLAG_FP16]
    )
    # Sentinel non-6 record stops table parsing
    blob[fh.TABLE_OFFSET + 16] = 0

    count, fp16_mask = fh.interpolants(bytes(blob))
    assert count == 2
    assert fp16_mask == 0b10
    assert fh.read_methods(bytes(blob)) == [fh.READ_MOV, fh.READ_INTERP]


if __name__ == "__main__":
    test_wave_size_handles_short_non_magic_and_wave32_flag()
    test_binding_table_with_unset_code_end_and_trailing_bytes()
    test_interpolants_and_read_methods_multi_entry()
