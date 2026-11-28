"""image_sample reads a UINT/SINT image through its integer view in every dimension and every stage (SHADER-1404).

Standard library only (spirv-as and spirv-val are used when they are on PATH); no GPU, no game files.

The hardware picks the sampled view from T# word 1 FORMAT in every dimension. image_sample_l and image_sample_lz take
that typed read in all of them (test_sample_typed_dims.py); the implicit image_sample took it only in a 2D pixel
shader, so a UINT/SINT 2D-array, 3D, cube or 1D texture, or any integer texture sampled outside the pixel stage, was
read through the float view and the VGPRs held a converted float instead of the integer texel. Here: the alias each
dimension loads, the Grad width the pixel stage samples with (the quad gradients are taken before the per-format
branch, as the 2D form does), LOD 0 outside the pixel stage, and that the modules validate. The GPU pixel fixtures
are runtime/test/shaders/ps_sample_implicit_uint.asm (2D) and ps_sample_array_uint.asm, ps_sample_array_sint.asm,
ps_sample_volume_uint.asm and ps_sample_volume_sint.asm (image_sample, image_sample_l and image_sample_lz of a
2D-array and a 3D integer texture, through vk_texture_test).
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support

DESC = "s[8:15], s[16:19]"
PRO = ["v_mov_b32 v4, 1.0", "v_mov_b32 v5, 0.5", "v_mov_b32 v6, 0.25", "v_mov_b32 v7, 0.75",
       "v_mov_b32 v8, 0"]

# dim -> (address range, vdata, dmask, STORE_DIMS index of the typed aliases, sampled binding, Grad type)
DIMS = {
    "1D": ("v4", "v9", "0x1", 3, 5, "%float"),
    "2D": ("v[4:5]", "v[9:12]", "0xf", 0, 0, "%v2float"),
    "2D_ARRAY": ("v[4:6]", "v[9:12]", "0xf", 1, 3, "%v2float"),     # the layer has no derivative
    "3D": ("v[4:6]", "v[9:12]", "0xf", 2, 6, "%v3float"),
    "CUBE": ("v[4:6]", "v[9:12]", "0xf", 1, 3, "%v2float"),         # a cube samples its 6-layer 2D-array view
}
# One 32-thread compute wave (RSRC2 and group size in the FSSM header).
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate(name, line):
    if name.startswith("cs_"):
        return support.translate_lines(name, PRO + [line, "s_endpgm"], fssm=CS_FSSM)
    return support.translate_lines(name, PRO + [line, "exp mrt0 v9, v9, v9, v9 done vm", "s_endpgm"])


def sample(op, dim):
    addr, vdata, dmask = DIMS[dim][:3]
    if op != "image_sample":       # the explicit LOD follows the coordinates
        addr = {"v4": "v[4:5]", "v[4:5]": "v[4:6]", "v[4:6]": "v[4:7]"}[addr] if op == "image_sample_l" else addr
    return f"{op} {vdata}, {addr}, {DESC} dmask:{dmask} dim:SQ_RSRC_IMG_{dim}"


def sample_lines(spv):
    return [ln for ln in spv.splitlines() if "OpImageSample" in ln]


class SampleImplicitDims(unittest.TestCase):
    def assert_typed_aliases(self, spv, di, binding):
        for cls in (1, 2):
            self.assertRegex(spv, rf"OpAccessChain %_ptr_UC_itx{cls}_{di} %itx{cls}_{di} ")
            self.assertIn(f"OpDecorate %itx{cls}_{di} Binding {binding}", spv)
            self.assertIn(f"%itxsamp{cls}_{di} = OpTypeSampledImage %itximg{cls}_{di}", spv)

    def test_pixel_stage_samples_each_view_with_the_dimension_gradients(self):
        for dim, (_, _, _, di, binding, gty) in DIMS.items():
            with self.subTest(dim=dim):
                spv = translate("ps_sampleimplicit", sample("image_sample", dim))
                self.assert_typed_aliases(spv, di, binding)
                dx = re.findall(rf"(%\w+) = OpDPdx {gty} ", spv)
                dy = re.findall(rf"(%\w+) = OpDPdy {gty} ", spv)
                self.assertEqual((len(dx), len(dy)), (1, 1), "one pair of quad gradients, before the branch")
                lines = sample_lines(spv)
                # float, UINT and SINT views, all with the same gradients and no implicit-LOD sample in the branch
                self.assertEqual(len(lines), 3, lines)
                for ty in ("%v4float", "%v4uint_st", "%v4int_st"):
                    self.assertTrue(any(f"OpImageSampleExplicitLod {ty} " in ln and
                                        ln.endswith(f"Grad {dx[0]} {dy[0]}") for ln in lines), (ty, lines))
                self.assertNotIn("OpImageSampleImplicitLod", spv)

    def test_array_and_cube_gradients_leave_out_the_layer(self):
        for dim in ("2D_ARRAY", "CUBE"):
            with self.subTest(dim=dim):
                spv = translate("ps_sampleimplicitarr", sample("image_sample", dim))
                uv = re.search(r"(%\w+) = OpCompositeConstruct %v3float ", spv).group(1)
                st = re.search(rf"(%\w+) = OpVectorShuffle %v2float {uv} {uv} 0 1", spv)
                self.assertIsNotNone(st, spv)
                self.assertRegex(spv, rf"= OpDPdx %v2float {st.group(1)}\n")
                # the sample itself still takes all three coordinates
                self.assertTrue(all(f" {uv} Grad " in ln for ln in sample_lines(spv)))

    def test_the_2d_pixel_form_differentiates_the_coordinates_directly(self):
        spv = translate("ps_sampleimplicit2d", sample("image_sample", "2D"))
        self.assertNotIn("OpVectorShuffle %v2float", spv)
        self.assertRegex(spv, r"OpAccessChain %_ptr_UC_itx1_0 %itx1_0 ")
        self.assertNotIn("%itx1_1", spv)

    def test_compute_stage_samples_each_view_at_lod_zero(self):
        for dim, (_, _, _, di, binding, _) in DIMS.items():
            with self.subTest(dim=dim):
                spv = translate("cs_sampleimplicit", sample("image_sample", dim))
                self.assert_typed_aliases(spv, di, binding)
                self.assertNotIn("OpDPdx", spv)
                lines = sample_lines(spv)
                self.assertEqual(len(lines), 3, lines)
                for ty in ("%v4float", "%v4uint_st", "%v4int_st"):
                    self.assertTrue(any(f"OpImageSampleExplicitLod {ty} " in ln and ln.endswith("Lod %f0")
                                        for ln in lines), (ty, lines))

    def test_a_form_off_the_typed_path_still_samples_the_float_view(self):
        # image_sample_b is not one of the typed forms (yet): one float sample, no integer alias.
        spv = translate("ps_samplebias", f"image_sample_b v[9:12], v[4:7], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY")
        self.assertNotIn("%itx1_", spv)
        self.assertEqual(len(sample_lines(spv)), 1)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"),
                         "needs spirv-as and spirv-val on PATH")
    def test_modules_validate(self):
        for stage in ("ps", "cs"):
            for op in ("image_sample", "image_sample_l", "image_sample_lz"):
                for dim in DIMS:
                    for mods in ("", " d16", " a16"):
                        line = sample(op, dim) + mods
                        if mods == " a16":      # two f16 coordinates (and the LOD) to a VGPR
                            line = re.sub(r"v\[4:[5-7]\]", "v[4:5]", line)
                        with self.subTest(stage=stage, op=op, dim=dim, mods=mods):
                            spv = translate(f"{stage}_sampleimplicitval", line)
                            with tempfile.TemporaryDirectory() as tmp:
                                a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
                                with open(a, "w", encoding="utf-8", newline="\n") as f:
                                    f.write(spv)
                                subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
                                subprocess.run(["spirv-val", "--target-env", "vulkan1.2", b], check=True)


if __name__ == "__main__":
    unittest.main()
