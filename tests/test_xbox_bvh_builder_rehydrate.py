"""Tests for RT-XDXR-BVH-REHYDRATE-BATCHED2X.

Verifies that runtime/shims/xbox_bvh_builder.cpp implements
XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X,
validates arguments (null list, null descs, count=0), rebuilds
structures from geometry via ID3D12GraphicsCommandList4, refuses
source-only dehydrated console streams without geometry (docs/bvh.md §4/§4a),
and registers the mangled export.
Synthetic test only; no retail game data or GPU required.
"""
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
XBOX_BVH_CPP = ROOT / "runtime" / "shims" / "xbox_bvh_builder.cpp"
XBOX_BVH_TEST_CPP = ROOT / "runtime" / "test" / "xbox_bvh_builder_test.cpp"


class XboxBvhBuilderRehydrateTest(unittest.TestCase):
    def test_rehydrate_implementation_exists(self):
        self.assertTrue(XBOX_BVH_CPP.is_file(), f"{XBOX_BVH_CPP} must exist")
        content = XBOX_BVH_CPP.read_text(encoding="utf-8")

        # Function definition
        m = re.search(r"HRESULT builder_rehydrate_batched2x\([^)]*\)\s*\{(.*?)\n\}", content, re.DOTALL)
        self.assertIsNotNone(m, "builder_rehydrate_batched2x must be defined in xbox_bvh_builder.cpp")
        body = m.group(1)

        # Must return S_OK on empty count
        self.assertIn("if (count == 0) return S_OK;", body)

        # Must validate descs and list
        self.assertIn("if (!descs) return E_INVALIDARG;", body)
        self.assertIn("if (!list) return E_INVALIDARG;", body)

        # Must convert to list4 and call BuildRaytracingAccelerationStructure for geometry
        self.assertIn("ID3D12GraphicsCommandList4* list4 = list4_of(list);", body)
        self.assertIn("BuildRaytracingAccelerationStructure", body)

        # Must refuse source-only dehydrated streams without geometry (docs/bvh.md §4/§4a)
        self.assertIn("E_NOTIMPL", body)
        self.assertIn("docs/bvh.md", body)

        # Must release list4 and return S_OK on success
        self.assertIn("list4->Release();", body)
        self.assertIn("return S_OK;", body)

    def test_mangled_export_registered(self):
        content = XBOX_BVH_CPP.read_text(encoding="utf-8")
        # In C++, string literals split across lines are concatenated by the compiler.
        compact = re.sub(r'"\s*\n\s*"', '', content)
        symbol = "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z"
        self.assertIn(symbol, compact, f"Symbol {symbol} must be registered in kEntries")
        self.assertIn("&builder_rehydrate_batched2x", content)

    def test_test_harness_validates_rehydrate(self):
        self.assertTrue(XBOX_BVH_TEST_CPP.is_file(), f"{XBOX_BVH_TEST_CPP} must exist")
        test_content = XBOX_BVH_TEST_CPP.read_text(encoding="utf-8")
        self.assertIn("rehydrate(builder, list.p, 1, &rehydrate_desc, &source) == S_OK", test_content)
        self.assertIn("rehydrate(builder, nullptr, 1, &rehydrate_desc, &source) == E_INVALIDARG", test_content)
        self.assertIn("rehydrate(builder, list.p, 0, nullptr, nullptr) == S_OK", test_content)
        self.assertIn("rehydrate(builder, list.p, 1, &source_only_desc, &source) == E_NOTIMPL", test_content)
        self.assertIn("rehydrate(builder, list.p, 1, &empty_desc, nullptr) == S_OK", test_content)
        self.assertIn("rehydrate(builder, list.p, 1, &zero_scratch_desc, &source) == S_OK", test_content)
        self.assertIn("rehydrate(builder, list.p, 2, pair_desc, nullptr) == S_OK", test_content)
        # The zero-scratch build and both batch elements are read back and handed to check-build.
        for name in ("shim_blas_rehydrated_zero_scratch", "shim_blas_rehydrated_batch"):
            self.assertIn(name, test_content)

    def test_scratch_regions_and_fresh_build(self):
        content = XBOX_BVH_CPP.read_text(encoding="utf-8")
        m = re.search(r"HRESULT builder_rehydrate_batched2x\([^)]*\)\s*\{(.*?)\n\}", content, re.DOTALL)
        self.assertIsNotNone(m)
        body = m.group(1)
        # One scratch buffer per call, a region per element at the D3D12 alignment.
        self.assertIn("rehydrate_scratch(self, device, list4, scratch_bytes)", body)
        self.assertIn("D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT", body)
        # The list holds the scratch until it is Reset; no buffer is kept or regrown on the builder.
        self.assertIn("rt::d3d12::hold_until_reset(list, res)", content)
        self.assertNotIn("builder_ensure_scratch", content)
        # A rehydrate is a fresh build: the update flag and the source are cleared.
        self.assertIn("&= ~D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE", body)
        self.assertIn("build.SourceAccelerationStructureData = 0;", body)


if __name__ == "__main__":
    unittest.main()
