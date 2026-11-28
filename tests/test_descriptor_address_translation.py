"""32-bit descriptor address translation reference contract model and production verification.

(docs/d3d12_vulkan.md §3.3 and §3.4, docs/boot_leads.md BL-4).

Exercises the pure-Python reference contract for descriptor bitfield packing/unpacking
and address arena partitioning, and statically verifies that production headers
(runtime/src/gfx10_descriptors.h) and translator logic (tools/rdna2spirv.py)
conform to the documented bitfield layouts and bounds constraints.

No game data, no GPU, no game executables. All descriptors and addresses are
hand-written to match the documented contract. Standard library unittest only. Run with:

    python -I tests/test_descriptor_address_translation.py
"""
import os
import re
import struct
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GFX10_HEADER = os.path.join(ROOT, "runtime", "src", "gfx10_descriptors.h")
RDNA2SPIRV = os.path.join(ROOT, "tools", "rdna2spirv.py")

# ---------------------------------------------------------------------------
# Contract constants (from docs/d3d12_vulkan.md §3.3)
# ---------------------------------------------------------------------------

FOUR_GIB = 1 << 32
ARENA_CAP_MAX = (4095 * 1024 * 1024)   # 4095 MiB — the documented arena cap
IMAGE_KEY_BASE = FOUR_GIB              # image keys start at exactly 4 GiB (§3.3)
TWO_POW_48 = 1 << 48
TWO_POW_40 = 1 << 40


# ---------------------------------------------------------------------------
# Pure-Python helpers mirroring the runtime's descriptor encoding
# ---------------------------------------------------------------------------

def encode_vsharp_base(addr48: int) -> tuple[int, int]:
    """Pack a 48-bit buffer address into V# word0 / word1 lower 16 bits.

    V# BASE_ADDRESS is 48 bits, split as:
        word0 = addr[31:0]
        word1[15:0] = addr[47:32]
    See gfx10_descriptors.h, tools/rdna2spirv.py BASE_ADDRESS comment.
    """
    assert 0 <= addr48 < TWO_POW_48, f"V# base must be < 2^48, got {addr48:#x}"
    word0 = addr48 & 0xFFFF_FFFF
    word1_lo16 = (addr48 >> 32) & 0xFFFF
    return word0, word1_lo16


def decode_vsharp_base(word0: int, word1: int) -> int:
    """Extract the 48-bit buffer base address from V# word0 and word1."""
    return word0 | ((word1 & 0xFFFF) << 32)


def encode_tsharp_base(addr: int) -> tuple[int, int]:
    """Pack a 256-byte-aligned image address into T# word0 / word1 lower 8 bits.

    T# BASE_ADDRESS is 40 bits, stored as address >> 8 split across:
        word0 = addr[39:8]   (i.e., addr >> 8, low 32 bits)
        word1[7:0] = addr[47:40] (high 8 bits of addr >> 8)
    gfx10_descriptors.h: "address >> 8 … 48 bits; word 0 … word 1[7:0]"
    """
    assert addr % 256 == 0, f"T# base must be 256-byte aligned, got {addr:#x}"
    assert 0 <= addr < TWO_POW_48, f"T# base must be < 2^48, got {addr:#x}"
    shifted = addr >> 8
    word0 = shifted & 0xFFFF_FFFF
    word1_lo8 = (shifted >> 32) & 0xFF
    return word0, word1_lo8


def decode_tsharp_base(word0: int, word1: int) -> int:
    """Extract the image base address from T# word0 and word1[7:0]."""
    shifted = word0 | ((word1 & 0xFF) << 32)
    return shifted << 8


# ---------------------------------------------------------------------------
# §3.3: Arena GPU address contract  host_address = heap_base + console_va
# ---------------------------------------------------------------------------

class TestArenaAddressContract(unittest.TestCase):
    """Verify host_address = heap_base + console_va for arena-resident resources."""

    def test_buffer_address_round_trip_low_arena(self):
        """A buffer at a low arena console VA reconstructs correctly."""
        heap_base = 0x0000_0001_0000_0000  # an example BDA above 4 GiB for the arena buffer itself
        console_va = 0x0001_0000            # 64 KiB offset inside the arena (a 32-bit value)
        host_address = heap_base + console_va
        recovered_va = (host_address - heap_base) & 0xFFFF_FFFF_FFFF_FFFF
        self.assertEqual(recovered_va, console_va)

    def test_buffer_address_at_1gib_offset(self):
        heap_base = 0x0000_0002_0000_0000
        console_va = 0x4000_0000  # 1 GiB — still 32-bit
        self.assertLess(console_va, FOUR_GIB)
        host_address = heap_base + console_va
        recovered_va = (host_address - heap_base) & 0xFFFF_FFFF_FFFF_FFFF
        self.assertEqual(recovered_va, console_va)

    def test_ptr32_truncation_gives_same_host_address(self):
        """25,865 shaders zero the high dword before the heap_base + va add."""
        heap_base = 0x0000_0001_0000_0000
        console_va = 0x0080_0000  # 8 MiB — fits in 32 bits
        va32 = console_va & 0xFFFF_FFFF
        self.assertEqual(heap_base + console_va, heap_base + va32)

    def test_ptr32_truncation_fails_for_high_arena_va(self):
        """A VA >= 4 GiB cannot survive ptr32 truncation."""
        heap_base = 0x0000_0001_0000_0000
        console_va_above_4gib = 0x1_0000_0000  # 4 GiB
        va32 = console_va_above_4gib & 0xFFFF_FFFF  # truncated to 0
        self.assertNotEqual(heap_base + va32, heap_base + console_va_above_4gib)

    def test_all_descriptor_heap_vas_must_be_below_4gib(self):
        """Descriptor heaps are allocated in the low 4 GiB of the arena."""
        arena_cap = ARENA_CAP_MAX
        self.assertLess(arena_cap, FOUR_GIB)

    def test_image_key_range_starts_above_arena(self):
        """Image keys start at 4 GiB, above the 4095 MiB arena cap (§3.3)."""
        self.assertGreater(IMAGE_KEY_BASE, ARENA_CAP_MAX)
        self.assertEqual(IMAGE_KEY_BASE, FOUR_GIB)

    def test_heap_base_plus_console_va_parametric(self):
        heap_base = 0x0000_0001_8000_0000
        test_vas = [0x0001_0000, 0x0010_0000, 0x0100_0000, 0x1000_0000, 0xF000_0000]
        for console_va in test_vas:
            with self.subTest(console_va=console_va):
                host_address = heap_base + console_va
                self.assertEqual(host_address - heap_base, console_va)
                self.assertEqual(console_va & 0xFFFF_FFFF, console_va)


# ---------------------------------------------------------------------------
# §3.4: V# (buffer descriptor) base address encoding
# ---------------------------------------------------------------------------

class TestVSharpBaseAddress(unittest.TestCase):
    """Verify the 48-bit V# base address encode/decode contract."""

    def test_zero_base(self):
        w0, w1_lo = encode_vsharp_base(0)
        self.assertEqual(w0, 0)
        self.assertEqual(w1_lo, 0)
        self.assertEqual(decode_vsharp_base(w0, w1_lo), 0)

    def test_low_32bit_base(self):
        addr = 0x0010_0000  # 1 MiB
        w0, w1_lo = encode_vsharp_base(addr)
        self.assertEqual(w0, addr)
        self.assertEqual(w1_lo, 0)
        self.assertEqual(decode_vsharp_base(w0, w1_lo), addr)

    def test_high_16bit_spills_into_word1(self):
        addr = 0x0001_0000_0000  # 4 GiB
        w0, w1_lo = encode_vsharp_base(addr)
        self.assertEqual(w0, 0)
        self.assertEqual(w1_lo, 0x0001)
        self.assertEqual(decode_vsharp_base(w0, w1_lo), addr)

    def test_max_representable_48bit_address(self):
        addr = TWO_POW_48 - 1  # 0xFFFF_FFFF_FFFF
        w0, w1_lo = encode_vsharp_base(addr)
        self.assertEqual(w0, 0xFFFF_FFFF)
        self.assertEqual(w1_lo, 0xFFFF)
        self.assertEqual(decode_vsharp_base(w0, w1_lo), addr)

    def test_typical_arena_buffer_address(self):
        addr = 0x0004_0000  # 256 KiB
        w0, w1_lo = encode_vsharp_base(addr)
        self.assertEqual(decode_vsharp_base(w0, w1_lo), addr)

    def test_word1_upper_16bits_are_not_base(self):
        addr = 0x0001_0000_0000  # 4 GiB
        w0, w1_lo = encode_vsharp_base(addr)
        word1_full = w1_lo | (0x20 << 16)
        self.assertEqual(decode_vsharp_base(w0, word1_full), addr)

    def test_round_trip_parametric(self):
        test_addrs = [
            0x0000_0000,
            0x0001_0000,
            0x0040_0000,
            0x0FFF_0000,
            0x0001_0000_0000,
            0x000F_FFFF_FF00,
        ]
        for addr in test_addrs:
            with self.subTest(addr=addr):
                w0, w1_lo = encode_vsharp_base(addr)
                self.assertEqual(decode_vsharp_base(w0, w1_lo), addr)


# ---------------------------------------------------------------------------
# §3.4: T# (image descriptor) base address encoding
# ---------------------------------------------------------------------------

class TestTSharpBaseAddress(unittest.TestCase):
    """Verify the 48-bit T# BASE_ADDRESS encoding."""

    def test_zero_base(self):
        w0, w1_lo8 = encode_tsharp_base(0)
        self.assertEqual(w0, 0)
        self.assertEqual(w1_lo8, 0)
        self.assertEqual(decode_tsharp_base(w0, w1_lo8), 0)

    def test_minimum_nonzero_address(self):
        addr = 256
        w0, w1_lo8 = encode_tsharp_base(addr)
        self.assertEqual(w0, 1)
        self.assertEqual(w1_lo8, 0)
        self.assertEqual(decode_tsharp_base(w0, w1_lo8), addr)

    def test_misaligned_raises(self):
        with self.assertRaises(AssertionError):
            encode_tsharp_base(0x100 + 1)

    def test_image_key_base_at_4gib(self):
        addr = IMAGE_KEY_BASE
        w0, w1_lo8 = encode_tsharp_base(addr)
        self.assertEqual(decode_tsharp_base(w0, w1_lo8), addr)

    def test_typical_key_just_above_4gib(self):
        addr = IMAGE_KEY_BASE + 256
        w0, w1_lo8 = encode_tsharp_base(addr)
        self.assertEqual(decode_tsharp_base(w0, w1_lo8), addr)

    def test_max_representable_tsharp_address(self):
        addr = 0xFFFF_FFFF_FF00
        w0, w1_lo8 = encode_tsharp_base(addr)
        self.assertEqual(decode_tsharp_base(w0, w1_lo8), addr)

    def test_word1_upper_bits_not_read(self):
        addr = IMAGE_KEY_BASE
        w0, w1_lo8 = encode_tsharp_base(addr)
        word1_full = w1_lo8 | (0x14 << 20) | (0x1 << 30)
        self.assertEqual(decode_tsharp_base(w0, word1_full), addr)

    def test_round_trip_parametric(self):
        test_addrs = [
            0,
            256,
            0x1000,
            0x0010_0000,
            0x1_0000_0000,
            0x1_0000_0100,
            0xFFFF_FFFF_FF00,
        ]
        for addr in test_addrs:
            with self.subTest(addr=addr):
                w0, w1_lo8 = encode_tsharp_base(addr)
                self.assertEqual(decode_tsharp_base(w0, w1_lo8), addr)


# ---------------------------------------------------------------------------
# §3.3 Descriptor-table high-dword truncation census (ptr32_lo contract)
# ---------------------------------------------------------------------------

class TestPtr32LoContract(unittest.TestCase):
    """Model the 25,865-shader census finding: descriptor-table pointers are ptr32_lo."""

    HEAP_BASE = 0x0000_0002_0000_0000

    def _ptr32_host(self, va: int) -> int:
        return self.HEAP_BASE + (va & 0xFFFF_FFFF)

    def test_ptr32_host_address_is_correct_for_low_arena_vas(self):
        test_cases = [
            (0x0001_0000, "64 KiB"),
            (0x0010_0000, "1 MiB"),
            (0x0100_0000, "16 MiB"),
            (0x1000_0000, "256 MiB"),
            (0x3FFF_0000, "near 1 GiB"),
            (0xEFFF_0000, "near 3.75 GiB"),
        ]
        for va, label in test_cases:
            with self.subTest(label=label, va=va):
                self.assertLess(va, FOUR_GIB)
                full_host = self.HEAP_BASE + va
                ptr32_host = self._ptr32_host(va)
                self.assertEqual(full_host, ptr32_host)

    def test_ptr32_breaks_for_high_va(self):
        va_above_4gib = 0x1_8000_0000
        full_host = self.HEAP_BASE + va_above_4gib
        ptr32_host = self._ptr32_host(va_above_4gib)
        self.assertNotEqual(full_host, ptr32_host)

    def test_arena_cap_below_4gib_satisfies_ptr32_contract(self):
        self.assertLess(ARENA_CAP_MAX, FOUR_GIB)
        for offset in [0, 0x1000, 0x0010_0000, ARENA_CAP_MAX]:
            with self.subTest(offset=offset):
                self.assertEqual(offset & 0xFFFF_FFFF, offset)


# ---------------------------------------------------------------------------
# §3.3: BVH builder 40-bit address constraint
# ---------------------------------------------------------------------------

class TestBvhBuilderAddressConstraint(unittest.TestCase):
    """The BVH builder rejects destination addresses >= 2^40 (measured, §3.3)."""

    def _bvh_validate(self, addr: int) -> bool:
        return (addr < TWO_POW_40) and (addr % 256 == 0)

    def test_arena_vas_pass_bvh_check(self):
        test_vas = [0, 256, 0x1000, 0xFFFF_FF00, ARENA_CAP_MAX & ~0xFF]
        for va in test_vas:
            with self.subTest(va=va):
                aligned = va & ~0xFF
                self.assertTrue(self._bvh_validate(aligned))

    def test_high_vas_fail_bvh_check(self):
        test_vas = [0x01_0000_0000_0000, TWO_POW_40, TWO_POW_40 + 256]
        for bad_va in test_vas:
            with self.subTest(bad_va=bad_va):
                aligned = bad_va & ~0xFF
                if aligned >= TWO_POW_40:
                    self.assertFalse(self._bvh_validate(aligned))


# ---------------------------------------------------------------------------
# Production Implementation Conformance Verification
# ---------------------------------------------------------------------------

class TestProductionImplementationConformance(unittest.TestCase):
    """Verify that runtime/src/gfx10_descriptors.h and tools/rdna2spirv.py
    strictly implement the documented bitfield layouts and bounds constraints.
    """

    def test_production_gfx10_header_vsharp_encoding_matches(self):
        """V# base address in gfx10_descriptors.h:
        words[0] = address, words[1] lower 16 bits = address >> 32
        """
        self.assertTrue(os.path.exists(GFX10_HEADER), f"Missing {GFX10_HEADER}")
        with open(GFX10_HEADER, encoding="utf-8") as f:
            content = f.read()

        # Check Buffer encoding lines
        self.assertRegex(
            content,
            r"words\[0\]\s*=\s*std::uint32_t\(buffer\.address\);",
            "gfx10_descriptors.h must pack buffer.address low 32 bits into word0",
        )
        self.assertRegex(
            content,
            r"field\(std::uint32_t\(buffer\.address\s*>>\s*32\),\s*0,\s*16\)",
            "gfx10_descriptors.h must pack buffer.address high 16 bits into word1[15:0]",
        )
        # Check Buffer decoding lines
        self.assertRegex(
            content,
            r"buffer\.address\s*=\s*\(std::uint64_t\(get\(words\[1\],\s*0,\s*16\)\)\s*<<\s*32\)\s*\|\s*words\[0\];",
            "gfx10_descriptors.h must reconstruct buffer.address from word0 and word1[15:0]",
        )

    def test_production_gfx10_header_tsharp_encoding_matches(self):
        """T# base address in gfx10_descriptors.h:
        words[0] = units (address >> 8), words[1] lower 8 bits = units >> 32 (address >> 40)
        """
        self.assertTrue(os.path.exists(GFX10_HEADER), f"Missing {GFX10_HEADER}")
        with open(GFX10_HEADER, encoding="utf-8") as f:
            content = f.read()

        self.assertRegex(
            content,
            r"const\s+std::uint64_t\s+units\s*=\s*image\.address\s*>>\s*8;",
            "gfx10_descriptors.h must convert image.address to 256-byte units",
        )
        self.assertRegex(
            content,
            r"words\[0\]\s*=\s*std::uint32_t\(units\);",
            "gfx10_descriptors.h must store low 32 bits of units in word0",
        )
        self.assertRegex(
            content,
            r"field\(std::uint32_t\(units\s*>>\s*32\),\s*0,\s*8\)",
            "gfx10_descriptors.h must store high 8 bits of units in word1[7:0]",
        )
        self.assertRegex(
            content,
            r"image\.address\s*=\s*\(\(std::uint64_t\(get\(words\[1\],\s*0,\s*8\)\)\s*<<\s*32\)\s*\|\s*words\[0\]\)\s*<<\s*8;",
            "gfx10_descriptors.h must reconstruct image.address as (word1[7:0]<<32 | word0) << 8",
        )

    def test_production_translator_buffer_address_matches(self):
        """tools/rdna2spirv.py extracts V# base with 16-bit mask on d1."""
        self.assertTrue(os.path.exists(RDNA2SPIRV), f"Missing {RDNA2SPIRV}")
        with open(RDNA2SPIRV, encoding="utf-8") as f:
            content = f.read()

        self.assertRegex(
            content,
            r"\{hi\}\s*=\s*OpBitwiseAnd\s+%uint\s+\{d1\}\s+\{self\.cu\(0xFFFF\)\}",
            "rdna2spirv.py must mask word1 with 0xFFFF for V# base high bits",
        )


if __name__ == "__main__":
    unittest.main()
