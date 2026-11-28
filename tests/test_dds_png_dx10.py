"""Tests for DX10 alpha modes using synthetic DDS buffers (no game data).

    python3 -I tests/test_dds_png_dx10.py
"""
import importlib.util
import os
import struct
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "dds_png.py")
    spec = importlib.util.spec_from_file_location("dds_png_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dds = load_module()


class Dx10AlphaTests(unittest.TestCase):
    def test_rgba8_alpha_mode_in_png(self):
        header = bytearray(148)
        header[:4] = b"DDS "
        struct.pack_into("<IIIII", header, 4, 124, 0x100F, 1, 1, 4)
        struct.pack_into("<II4s", header, 76, 32, 4, b"DX10")
        struct.pack_into("<I", header, 108, 0x1000)  # DDSCAPS_TEXTURE
        struct.pack_into("<IIII", header, 128, 28, 3, 0, 1)  # RGBA8, 2D, one slice
        pixel = b"\xff\x00\x00\x00"

        for mode, alpha in ((3, 255), (1, 0)):  # Opaque and straight alpha
            with self.subTest(mode=mode):
                struct.pack_into("<I", header, 144, mode)
                png = dds.dds_to_png_bytes(bytes(header) + pixel)
                # The first chunk after IHDR contains the compressed row.
                pos = 8 + 12 + 13
                size, kind = struct.unpack_from(">I4s", png, pos)
                self.assertEqual(kind, b"IDAT")
                row = zlib.decompress(png[pos + 8:pos + 8 + size])
                self.assertEqual(row, bytes((0, 255, 0, 0, alpha)))


def _dx10(width, height, dxgi, payload, pitch=0):
    header = bytearray(148)
    header[:4] = b"DDS "
    struct.pack_into("<IIIII", header, 4, 124, 0x100F | (8 if pitch else 0), height, width, pitch)
    struct.pack_into("<II4s", header, 76, 32, 4, b"DX10")
    struct.pack_into("<IIII", header, 128, dxgi, 3, 0, 1)
    return bytes(header) + payload


class SingleChannelTests(unittest.TestCase):
    def test_r32_float_linear_values_with_padding(self):
        payload = (struct.pack("<4f", -1, 0, 0.5, 0.25) + b"skip"
                   + struct.pack("<4f", 2, float("inf"), float("-inf"), float("nan")))
        data = _dx10(4, 2, 41, payload, pitch=20)
        self.assertEqual(dds.parse_dds_header(data)["format"], "R32_FLOAT")
        width, height, rgba = dds.decode_dds(data)
        self.assertEqual((width, height), (4, 2))
        self.assertEqual(rgba, b"".join(bytes((r, 0, 0, 255)) for r in (0, 0, 128, 64, 255, 255, 0, 0)))

    def test_byte_unorm_masks_respect_pitch_and_channel_order(self):
        for dxgi, fmt, payload, expected in (
            (61, "R8_UNORM", bytes([1, 2, 99, 99, 3, 4]), bytes([1,0,0,255, 2,0,0,255, 3,0,0,255, 4,0,0,255])),
            (49, "R8G8_UNORM", bytes([1,2,3,4, 99,99, 5,6,7,8]), bytes([1,2,0,255, 3,4,0,255, 5,6,0,255, 7,8,0,255])),
        ):
            with self.subTest(fmt=fmt):
                data = _dx10(2, 2, dxgi, payload, pitch=4 if dxgi == 61 else 6)
                self.assertEqual(dds.parse_dds_header(data)["format"], fmt)
                self.assertEqual(dds.decode_dds(data), (2, 2, expected))
                self.assertEqual(dds.dds_to_png_bytes(data), dds.encode_png(2, 2, expected))
                with self.assertRaises(dds.DdsError):
                    dds.decode_dds(data[:-1])

    def test_r16_unorm_red_opaque(self):
        px = struct.pack("<HHHH", 0, 65535, 32768, 257)
        _, _, rgba = dds.decode_dds(_dx10(2, 2, 56, px))
        self.assertEqual(rgba, bytes([0, 0, 0, 255, 255, 0, 0, 255, 128, 0, 0, 255, 1, 0, 0, 255]))

    def test_a8_unorm_alpha_only_with_pitch(self):
        payload = bytes([1, 2, 3, 99, 4, 5, 6, 99])  # 3x2, pitch 4
        _, _, rgba = dds.decode_dds(_dx10(3, 2, 65, payload, pitch=4))
        self.assertEqual(rgba[3::4], bytes([1, 2, 3, 4, 5, 6]))
        self.assertEqual(set(rgba[0::4]) | set(rgba[1::4]) | set(rgba[2::4]), {0})

    def test_rgba16_unorm_normalizes_all_channels_with_row_padding(self):
        payload = struct.pack("<4H", 0, 65535, 32768, 257) + b"pad!"
        payload += struct.pack("<4H", 65535, 0, 32767, 32768)
        data = _dx10(1, 2, 11, payload, pitch=12)
        self.assertEqual(dds.parse_dds_header(data)["format"], "RGBA16_UNORM")
        self.assertEqual(dds.decode_dds(data)[2], bytes((
            0, 255, 128, 1,
            255, 0, 127, 128,
        )))

    def test_rgba16_snorm_maps_signed_extremes_and_zero(self):
        payload = struct.pack("<4h", -32768, -32767, 0, 32767)
        data = _dx10(1, 1, 13, payload)
        self.assertEqual(dds.parse_dds_header(data)["format"], "RGBA16_SNORM")
        self.assertEqual(dds.decode_dds(data)[2], bytes((0, 0, 128, 255)))

    def test_rgba16_snorm_padded_rows_and_truncation(self):
        payload = struct.pack("<4h", -16384, 16384, -32768, 32767) + b"pad!"
        payload += struct.pack("<4h", 0, 0, 0, 0)
        data = _dx10(1, 2, 13, payload, pitch=12)
        self.assertEqual(dds.decode_dds(data)[2], bytes((64, 191, 0, 255, 128, 128, 128, 128)))
        with self.assertRaises(dds.DdsError):
            dds.decode_dds(_dx10(1, 1, 13, bytes(7)))

    def test_truncated_rejected(self):
        for dxgi, n in ((56, 7), (65, 3), (41, 15)):
            with self.assertRaises(dds.DdsError):
                dds.decode_dds(_dx10(2, 2, dxgi, bytes(n)))


def _opaque(*pairs):
    """RGBA bytes for (red, green) pairs with blue 0 and alpha 255."""
    return b"".join(bytes((r, g, 0, 255)) for r, g in pairs)


class PackedAndIntegerTests(unittest.TestCase):
    """Hand-built texels for the formats added by #2218, #2314, #2317, #2319, #2320, #2321, #2323 and #2324.

    Each case is 2x2 with a padded row pitch, so the second row starts after the padding bytes,
    and a payload one byte short must be refused.
    """

    def check(self, dxgi, fmt, row0, row1, pitch, expected):
        payload = row0 + b"\xee" * (pitch - len(row0)) + row1
        data = _dx10(2, 2, dxgi, payload, pitch=pitch)
        self.assertEqual(dds.parse_dds_header(data)["format"], fmt)
        self.assertEqual(dds.decode_dds(data), (2, 2, expected))
        self.assertEqual(dds.dds_to_png_bytes(data), dds.encode_png(2, 2, expected))
        with self.assertRaises(dds.DdsError):
            dds.decode_dds(data[:-1])

    def test_r9g9b9e5_sharedexp(self):
        def texel(r, g, b, e):
            return struct.pack("<I", r | (g << 9) | (b << 18) | (e << 27))
        # value = mantissa * 2^(e - 24): (256, 128, 0) at e=16 is (1, 0.5, 0); 511/512 rounds to 255, 64/512 to 32,
        # 1/512 to 0; 300 * 2^-4 saturates; e=0 is 2^-24 per step, so 511 * 2^-24 is still 0.
        self.check(67, "R9G9B9E5_FLOAT",
                   texel(256, 128, 0, 16) + texel(511, 64, 1, 15),
                   texel(300, 0, 0, 20) + texel(511, 511, 511, 0), 12,
                   bytes((255, 128, 0, 255, 255, 32, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255)))

    def test_b5g6r5_unorm(self):
        # Blue in bits 0-4, green 5-10, red 11-15; 5 and 6 bits widen by bit replication.
        texels = struct.pack("<4H", 31 << 11, 63 << 5, 1, (16 << 11) | (32 << 5) | 16)
        self.check(85, "B5G6R5_UNORM", texels[:4], texels[4:], 6,
                   bytes((255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 8, 255, 132, 130, 132, 255)))

    def test_b5g5r5a1_unorm(self):
        # Blue in bits 0-4, green 5-9, red 10-14, alpha bit 15.
        texels = struct.pack("<4H", 0x8000 | (31 << 10), 0x7FFF, 31 << 5, 0x8001)
        self.check(86, "B5G5R5A1_UNORM", texels[:4], texels[4:], 6,
                   bytes((255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0, 8, 255)))

    def test_r8g8_snorm(self):
        # round((v / 127 / 2 + 0.5) * 255), with -128 clamped to -127 as for BC5_SNORM.
        self.check(51, "RG8_SNORM", struct.pack("<4b", -128, -127, 0, 127), struct.pack("<4b", 64, -64, 1, -1), 6,
                   _opaque((0, 0), (128, 255), (192, 63), (129, 126)))

    def test_r8g8_uint_and_sint(self):
        self.check(50, "RG8_UINT", bytes((0, 255, 7, 128)), bytes((1, 2, 254, 3)), 5,
                   _opaque((0, 255), (7, 128), (1, 2), (254, 3)))
        self.check(52, "RG8_SINT", struct.pack("<4b", -128, 127, 0, -1), struct.pack("<4b", 1, -127, 64, -64), 5,
                   _opaque((0, 255), (128, 127), (129, 1), (192, 64)))

    def test_r8_snorm_uint_and_sint(self):
        red = lambda *values: b"".join(bytes((v, 0, 0, 255)) for v in values)  # noqa: E731
        self.check(63, "R8_SNORM", struct.pack("<2b", -128, -127), struct.pack("<2b", 0, 127), 3, red(0, 0, 128, 255))
        self.check(63, "R8_SNORM", struct.pack("<2b", 64, -64), struct.pack("<2b", 1, -1), 3, red(192, 63, 129, 126))
        self.check(62, "R8_UINT", bytes((0, 255)), bytes((7, 128)), 3, red(0, 255, 7, 128))
        self.check(64, "R8_SINT", struct.pack("<2b", -128, 127), struct.pack("<2b", -1, 0), 3, red(0, 255, 127, 128))

    def test_r10g10b10a2_uint_and_typeless_read_like_unorm(self):
        # R in bits 0-9, G 10-19, B 20-29, A 30-31; each field scaled to 0-255 from its own bit width.
        def texel(r, g, b, a):
            return struct.pack("<I", r | (g << 10) | (b << 20) | (a << 30))
        row0, row1 = texel(1023, 0, 512, 3) + texel(0, 1023, 0, 1), texel(1, 2, 3, 0) + texel(511, 1022, 0, 2)
        expected = bytes((255, 0, 128, 255, 0, 255, 0, 85, 0, 0, 1, 0, 127, 255, 0, 170))
        for dxgi in (23, 24, 25):  # TYPELESS, UNORM, UINT
            with self.subTest(dxgi=dxgi):
                self.check(dxgi, "RGBA8", row0, row1, 12, expected)


if __name__ == "__main__":
    unittest.main()
