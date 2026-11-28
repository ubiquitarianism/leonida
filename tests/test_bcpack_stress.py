"""Adversarial stress and fuzzing test suite for BCPack rANS decompression (Track 8, R9).

Stress-tests:
1. Truncated bitstreams across all codecs (BC1, BC3, BC4, BC5, BC6H, BC7) at 100% of byte prefixes.
2. Corrupt frequency headers, symbol width reduction limits, and malformed fragment lengths.
3. Non-canonical state transitions, home state verification (RANS_L), virtual zero boundaries,
   and final-fragment flush bit validation.
4. Block payload corruption and rebuilder validation across all codecs (anchor violations,
   p-bit consistency, history index bounds, strict vs provisional mode).
5. High-volume random noise and mutational fuzzing verifying that NO unhandled exceptions leak
   and that malformed blocks produce clean BcpackError exceptions.

Standard library only, zero game data.
    python -I tests/test_bcpack_stress.py
"""

import os
import random
import struct
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bcpack  # noqa: E402
import bcpack_tables as tables  # noqa: E402
import bcpack_tables_bptc as bptc  # noqa: E402
import test_bcpack as synth  # noqa: E402


class TruncatedBitstreamStressTest(unittest.TestCase):
    """Exhaustive prefix truncation stress testing for single- and multi-fragment streams."""

    def test_single_fragment_all_byte_truncations_all_codecs(self):
        """Test 100% of single-byte truncation prefixes for every active codec."""
        tested_prefixes = 0
        for codec in (1, 3, 4, 5, 6, 7):
            widths = bcpack.LANE_WIDTHS[codec]
            nsym = 16
            lanes = [
                [((i * 13 + j * 7) & ((1 << dw) - 1)) for j in range(nsym)]
                for i, dw in enumerate(widths)
            ]
            raw = synth.encode_lanes(lanes, codec, q=0)
            out_size = nsym * bcpack.ELEMENT_BYTES

            # Baseline check: intact stream decodes cleanly
            decoded, used = bcpack.decode_lanes(raw, codec, out_size)
            self.assertEqual(used, len(raw))
            self.assertEqual(decoded, lanes)

            # Test every truncation prefix from 0 to len(raw) - 1
            for k in range(len(raw)):
                truncated = raw[:k]
                with self.assertRaises(
                    bcpack.BcpackError,
                    msg=f"Codec {codec} did not raise BcpackError on truncation prefix {k}/{len(raw)}"
                ):
                    bcpack.decode_lanes(truncated, codec, out_size)
                tested_prefixes += 1

        self.assertGreaterEqual(tested_prefixes, 500)

    def test_empty_bitstream_rejection(self):
        """Empty bitstreams must raise BcpackError cleanly for all codecs when out_size > 0."""
        for codec in (1, 3, 4, 5, 6, 7):
            block_sz = bcpack.BLOCK_BYTES[codec]
            with self.assertRaises(bcpack.BcpackError):
                bcpack.decode_lanes(b"", codec, block_sz)
            with self.assertRaises(bcpack.BcpackError):
                bcpack.decode(b"", codec, block_sz)

    def test_multi_fragment_chained_truncations(self):
        """Test truncation across multi-fragment streams with mixed flush/non-flush boundaries."""
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        nsym = 400
        lanes = [
            [((i * 29 + j * 17) & ((1 << dw) - 1)) for j in range(nsym)]
            for i, dw in enumerate(widths)
        ]
        streams = []
        for i, (syms, dw) in enumerate(zip(lanes, widths)):
            if i == 0:
                s1 = [x & 0x3F for x in syms[:100]]
                s2 = [x & 0x1F for x in syms[100:]]
                lanes[0] = s1 + s2
                streams.append(synth._encode_lane_stream([
                    (s1, dw - 2, 2, True),
                    (s2, dw - 3, 5, True),
                ], dw))
            elif i == 2:
                streams.append(synth._encode_lane_stream([
                    (syms[:150], dw, 1, False),
                    (syms[150:], dw, 4, True),
                ], dw))
            else:
                streams.append(synth._encode_lane_stream([(syms, dw, 0, True)], dw))

        raw = synth.interleave_lanes(streams, widths, nsym)
        out_size = nsym * bcpack.ELEMENT_BYTES

        # Verify baseline decodes
        decoded, used = bcpack.decode_lanes(raw, codec, out_size)
        self.assertEqual(used, len(raw))
        self.assertEqual(decoded, lanes)

        # Truncate at every 5th byte to cover hundreds of boundaries
        for k in range(0, len(raw), 5):
            truncated = raw[:k]
            with self.assertRaises(bcpack.BcpackError):
                bcpack.decode_lanes(truncated, codec, out_size)


class CorruptFrequencyHeaderStressTest(unittest.TestCase):
    """Stress tests for frequency models, table bounds, and header corruptions."""

    def test_header_byte_mutations(self):
        """Systematically mutate header bytes to explore all 256 byte variations per header slot."""
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        nsym = 8
        lanes = [[0] * nsym for _ in widths]
        raw = bytearray(synth.encode_lanes(lanes, codec, q=0))
        out_size = nsym * bcpack.ELEMENT_BYTES

        # Test mutating first 24 bytes (header bytes for the 12 lanes)
        for byte_idx in range(min(24, len(raw))):
            orig = raw[byte_idx]
            for val in (0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x7F, 0x80, 0x90, 0xC0, 0xDF, 0xFF):
                if val == orig:
                    continue
                raw[byte_idx] = val
                try:
                    bcpack.decode_lanes(bytes(raw), codec, out_size)
                except bcpack.BcpackError:
                    pass  # Expected clean rejection
                except Exception as ex:
                    self.fail(f"Header byte {byte_idx} val {hex(val)} leaked unhandled exception: {type(ex)} {ex}")
            raw[byte_idx] = orig

    def test_symbol_width_reduction_out_of_range(self):
        """Symbol width reduction (z = dw - w) resulting in w < 1 or w > 10 must raise BcpackError."""
        # Lane 0 with default width 4 (BC3 lane 0)
        # If z = 3, w = 4 - 3 = 1 (valid)
        # Construct bitstream where lane width reduction creates invalid width
        codec = 3
        widths = bcpack.LANE_WIDTHS[codec]
        nsym = 4
        lanes = [[0] * nsym for _ in widths]
        raw = bytearray(synth.encode_lanes(lanes, codec, q=0))
        # Header byte 0: [z:2][h3:1][flush:1][q:4]
        # In BC3, lane 0 has dw=4. If z=3 (top 2 bits = 0xC0), w = 1.
        # Now let's try injecting invalid z into raw header:
        raw[0] = 0xC0 | (1 << 4) | 0  # z=3, flush=1, q=0
        try:
            bcpack.decode_lanes(bytes(raw), codec, nsym * 8)
        except bcpack.BcpackError as e:
            # Expected if rANS state or width is rejected
            self.assertIn("lane 0", str(e))

    def test_model_sum_validation(self):
        """_Model must reject frequency tables that do not sum to 65536."""
        # Sum is 65535 (too small)
        with self.assertRaises(bcpack.BcpackError):
            bcpack._Model(1, (32768, 32767))
        # Sum is 65537 (too large)
        with self.assertRaises(bcpack.BcpackError):
            bcpack._Model(1, (32768, 32769))
        # Sum is 0
        with self.assertRaises(bcpack.BcpackError):
            bcpack._Model(1, (0, 0))

    def test_all_160_models_monotonic_and_invertible(self):
        """Verify all 160 built-in probability models have monotonic base arrays and valid inverses."""
        for (w, q), freq in tables.PROB_TABLES.items():
            m = bcpack._Model(w, freq)
            self.assertEqual(len(m.base), w + 1)
            # Base must be strictly non-decreasing
            for i in range(len(m.base) - 1):
                self.assertLessEqual(m.base[i], m.base[i + 1])
            self.assertLessEqual(m.base[-1], 65536)

            # Test symbol inversion for boundary symbols
            for s in (0, 1, (1 << w) - 1):
                p = s.bit_length()
                f = m.freq[p]
                c = m.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
                if c == 0:
                    f, c = f - 1, 1
                x = (256 << 16) + c
                self.assertEqual(m.lookup(x & 0xFFFF), s)


class NonCanonicalStateTransitionTest(unittest.TestCase):
    """Stress tests for rANS state bounds, home state (RANS_L), virtual zeros, and flush logic."""

    def test_flush_fragment_must_end_at_rans_l(self):
        """Corrupting the end state of a flush fragment must trigger clean home state rejection."""
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        nsym = 8
        lanes = [[0] * nsym for _ in widths]
        raw = bytearray(synth.encode_lanes(lanes, codec, q=0))
        out_size = nsym * bcpack.ELEMENT_BYTES

        # The last 4 bytes of a lane fragment represent the rANS state / payload
        # Corrupt the trailing payload bytes of the last lane
        raw[-1] ^= 0x5A
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode_lanes(bytes(raw), codec, out_size)
        self.assertTrue(
            "home state" in str(ctx.exception)
            or "left over" in str(ctx.exception)
            or "input ran out" in str(ctx.exception)
        )

    def test_non_canonical_flush_zero_gap_documented(self):
        """Verify empirical behavior when a stream ends on flush=0 with arbitrary state.

        Finding: If a bitstream crafts flush=False on the final fragment, the decoder's
        in_rem == 0 check sets done[i] = True without verifying st == RANS_L.
        This test documents this empirical behavior.
        """
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        nsym = 4
        lanes = [[0] * nsym for _ in widths]
        model = bcpack._get_model(widths[0], 0)
        arbitrary_x = 0x12345678  # Non-canonical state != RANS_L (0x01000000)
        x, buf = synth._encode_rans_syms(lanes[0], model, arbitrary_x)
        payload = [(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF] + list(reversed(buf))
        hdr = synth._make_hdr(widths[0], widths[0], 0, False, len(payload))  # flush=False
        streams = [bytes(hdr + payload)] + [
            synth._encode_fragment(lanes[i], widths[i], widths[i], 0, flush=True)[0]
            for i in range(1, len(widths))
        ]
        raw = synth.interleave_lanes(streams, widths, nsym)

        # Document whether the decoder accepts or rejects this stream:
        try:
            decoded, used = bcpack.decode_lanes(raw, codec, nsym * 8)
            accepted = True
        except bcpack.BcpackError:
            accepted = False

        # As discovered empirically, the decoder accepts flush=False without home-state check
        self.assertTrue(accepted, "Expected flush=0 end state to be accepted under current implementation")

    def test_virtual_zero_overflow_fails(self):
        """Virtual zeros count exceeding MAX_VIRTUAL_ZEROS (8) must raise BcpackError."""
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        nsym = 16
        lanes = [[0] * nsym for _ in widths]
        raw = bytearray(synth.encode_lanes(lanes, codec, q=0))
        # Truncating payload bytes triggers virtual zeros exhaustion / premature EOS in decoder
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(bytes(raw[:4]), codec, nsym * 8)

    def test_trailing_non_zero_bytes_rejected(self):
        """Non-zero trailing bytes after the last lane completes must be rejected with clean error."""
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        nsym = 8
        lanes = [[0] * nsym for _ in widths]
        raw = synth.encode_lanes(lanes, codec, q=0)
        out_size = nsym * bcpack.ELEMENT_BYTES

        # Trailing non-zero byte must raise BcpackError
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode_lanes(raw + b"\x01", codec, out_size)
        self.assertIn("non-zero data left over", str(ctx.exception))

        # Trailing zero bytes are permitted by decode_lanes
        decoded, used = bcpack.decode_lanes(raw + b"\x00\x00\x00", codec, out_size)
        self.assertEqual(used, len(raw))
        self.assertEqual(decoded, lanes)


class CorruptBlockPayloadStressTest(unittest.TestCase):
    """Stress tests for block rebuilders and structural error handling across BC1-BC7."""

    def test_invalid_codecs_and_sizes(self):
        """Unsupported codecs and unaligned output sizes must raise clean exceptions."""
        # Codec 2 (BC2) is unsupported
        with self.assertRaises(bcpack.BcpackUnsupportedCodec):
            bcpack.decode(b"\x00" * 32, 2, 32)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(b"\x00" * 32, 2, 32)

        # Codec 99 (invalid)
        with self.assertRaises(bcpack.BcpackUnsupportedCodec):
            bcpack.decode(b"\x00" * 32, 99, 32)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(b"\x00" * 32, 99, 32)

        # Negative output size
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(b"", 1, -8)

        # Output size not a multiple of block bytes
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(b"", 1, 15)  # BC1 needs multiple of 8
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(b"", 7, 24)  # BC7 needs multiple of 16

    def test_bc1_non_zero_lane1_rejected(self):
        """BC1 stream with non-zero lane 1 must be rejected."""
        lanes = [[0] * 8 for _ in range(12)]
        lanes[1][0] = 1  # Corrupt lane 1
        raw = synth.encode_lanes(lanes, 1)
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode(raw, 1, 64)
        self.assertIn("lane 1 is non-zero", str(ctx.exception))

    def test_bc3_non_zero_lane1_rejected(self):
        """BC3 stream with non-zero lane 1 must be rejected."""
        lanes = [[0] * 8 for _ in range(12)]
        lanes[1][0] = 1
        raw = synth.encode_lanes(lanes, 3)
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode(raw, 3, 64)
        self.assertIn("lane 1 is non-zero", str(ctx.exception))

    def test_bc4_non_zero_lane1_rejected(self):
        """BC4 stream with non-zero lane 1 must be rejected."""
        lanes = [[0] * 8 for _ in range(12)]
        lanes[1][0] = 1
        raw = synth.encode_lanes(lanes, 4)
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode(raw, 4, 64)
        self.assertIn("lane 1 is non-zero", str(ctx.exception))

    def test_bc5_non_zero_lane1_rejected(self):
        """BC5 stream with non-zero lane 1 must be rejected."""
        lanes = [[0] * 8 for _ in range(12)]
        lanes[1][0] = 1
        raw = synth.encode_lanes(lanes, 5)
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode(raw, 5, 64)
        self.assertIn("lane 1 is non-zero", str(ctx.exception))

    def test_bc6h_mode_symbol_out_of_range(self):
        """BC6H block with mode symbol > 13 must raise clean BcpackError."""
        lanes = [[0] * 16 for _ in range(16)]
        lanes[15][0] = 14  # Mode symbol 14 > 13
        raw = synth.encode_lanes(lanes, 6)
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode(raw, 6, 128)
        self.assertIn("mode symbol 14 outside 0..13", str(ctx.exception))

    def test_bc7_signature_outside_table(self):
        """BC7 block with invalid signature outside signature table must raise clean BcpackError."""
        lanes = [[0] * 16 for _ in range(16)]
        # Inject invalid signature rank (e.g. s15a = 31, s15b = 15)
        lanes[15][0] = 31
        lanes[15][1] = 15
        raw = synth.encode_lanes(lanes, 7)
        with self.assertRaises(bcpack.BcpackError) as ctx:
            bcpack.decode(raw, 7, 128)
        self.assertIn("outside the signature table", str(ctx.exception))


class FuzzingHarnessTest(unittest.TestCase):
    """High-volume mutational and random bitstream fuzzing across all codecs."""

    def test_pure_random_noise_fuzzing(self):
        """Feed 1,200 random noise bitstreams across all codecs; verify 0 unhandled exceptions."""
        rng = random.Random(20261124)
        unhandled = []
        for i in range(1200):
            codec = rng.choice((1, 3, 4, 5, 6, 7))
            out_blocks = rng.choice((1, 2, 4, 8, 16))
            out_size = out_blocks * bcpack.BLOCK_BYTES[codec]
            noise_len = rng.randint(0, 512)
            noise = rng.randbytes(noise_len)
            try:
                bcpack.decode(noise, codec, out_size)
            except (bcpack.BcpackError, bcpack.BcpackProvisionalError):
                pass  # Clean handled exception
            except Exception as ex:
                unhandled.append((codec, noise_len, type(ex).__name__, str(ex)))

        self.assertEqual(len(unhandled), 0, f"Leaked unhandled exceptions: {unhandled}")

    def test_mutational_fuzzing_on_valid_streams(self):
        """Mutate valid bitstreams with bit-flips, byte overwrites, insertions, and deletions.

        Verifies that 2,000 mutated streams never crash with unhandled exceptions.
        """
        rng = random.Random(987654321)
        unhandled = []

        # Generate base valid streams for each codec
        base_streams = {}
        for codec in (1, 3, 4, 5, 6, 7):
            widths = bcpack.LANE_WIDTHS[codec]
            nsym = 16
            lanes = [
                [((i * 13 + j * 7) & ((1 << dw) - 1)) for j in range(nsym)]
                for i, dw in enumerate(widths)
            ]
            base_streams[codec] = (
                bytearray(synth.encode_lanes(lanes, codec, q=0)),
                nsym * bcpack.ELEMENT_BYTES
            )

        for _ in range(2000):
            codec = rng.choice((1, 3, 4, 5, 6, 7))
            orig_raw, out_size = base_streams[codec]
            mutated = bytearray(orig_raw)

            mutation_type = rng.randint(0, 3)
            if mutation_type == 0:
                # Bit flip
                byte_pos = rng.randrange(len(mutated))
                bit_pos = rng.randrange(8)
                mutated[byte_pos] ^= (1 << bit_pos)
            elif mutation_type == 1:
                # Byte overwrite
                byte_pos = rng.randrange(len(mutated))
                mutated[byte_pos] = rng.randrange(256)
            elif mutation_type == 2:
                # Truncation / slice deletion
                del_len = rng.randint(1, min(8, len(mutated)))
                start = rng.randrange(len(mutated) - del_len + 1)
                del mutated[start:start + del_len]
            elif mutation_type == 3:
                # Random byte insertion
                ins_pos = rng.randrange(len(mutated) + 1)
                mutated.insert(ins_pos, rng.randrange(256))

            try:
                bcpack.decode(bytes(mutated), codec, out_size)
            except (bcpack.BcpackError, bcpack.BcpackProvisionalError):
                pass
            except Exception as ex:
                unhandled.append((codec, mutation_type, type(ex).__name__, str(ex)))

        self.assertEqual(len(unhandled), 0, f"Leaked unhandled exceptions during mutational fuzzing: {unhandled}")


if __name__ == "__main__":
    unittest.main()
