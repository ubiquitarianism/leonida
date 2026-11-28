"""Local sidecar generation and resume from a hand-built RDS5 pool."""
from pathlib import Path
from array import array
import json
import os
import shutil
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"tools"), str(ROOT/"tests")]
from test_bcpack_history_calib import (_encode_bc7_mode6_tiled, _pack_etpf, _write_custom_rds5_pool)
import build_dstorage_history as history


class RuntimeHistoryTest(unittest.TestCase):
    def test_constraints_preserve_confirmed_bits(self):
        docs = {"bc6h": history.calib.Bc6hCalibrator().build_document()[0],
                "bc7": history.calib.Bc7Calibrator().build_document()[0]}
        for name in ("bc6h", "bc7"):
            row = docs[name]
            values, confirmed = (("indexHistory", "indexConfirmedBits") if name == "bc6h" else
                                 ("historyIndices", "historyIndexConfirmedBits"))
            row[values][8][5] = 12
            row[confirmed][8][5] = 2
        votes = array("Q", [0]) * (2*256*35*16)
        for base in (0, 256*35*16):
            start = base + (8*32+5)*16
            # Only candidates 3 and 14 satisfy the anchor. The known top two bits select 14.
            for v in range(16):
                votes[start+v] = int(v not in (3, 14))
        with tempfile.TemporaryDirectory() as name:
            model = Path(name)/"history.json"
            model.write_text(json.dumps(docs))
            history.refine(docs, votes, model)
        self.assertEqual(docs["bc6h"]["indexHistory"][8][5], 14)
        self.assertEqual(docs["bc7"]["historyIndices"][8][5], 14)
        self.assertEqual(docs["bc7"]["historyIndexConfirmedBits"][8][5], 2)

    def test_shared_red_parity_preserves_partial_endpoint_precision(self):
        docs = {"bc6h": history.calib.Bc6hCalibrator().build_document()[0],
                "bc7": history.calib.Bc7Calibrator().build_document()[0]}
        bc7 = docs["bc7"]
        for field in (0, 4):
            bc7["historyEndpoints"][9][field] = 96
            bc7["historyEndpointBits"][9][field] = 5
            bc7["historyEndpointConfirmedBits"][9][field] = 4
        votes = array("Q", [0]) * (2*256*35*16)
        start = (256*35 + 256*32 + 9*3)*16
        for pair in range(16):
            votes[start + pair] = int(not ((pair // 4 ^ pair % 4) & 2))
        with tempfile.TemporaryDirectory() as name:
            model = Path(name)/"profile.json"
            model.write_text(json.dumps(docs))
            history.refine(docs, votes, model)
        loaded = history.bptc.load_bc7(docs)
        a, b = (loaded["history_endpoints"][9][p] for p in (0, 4))
        self.assertEqual((a ^ b) & 2, 2)
        for field in (0, 4):
            self.assertEqual(loaded["history_endpoints"][9][field] >> 3, 96 >> 3)
            self.assertEqual(loaded["history_endpoint_bits"][9][field], 5)
            self.assertEqual(loaded["history_endpoint_confirmed_bits"][9][field], 4)
        bc7["historyEndpointBits"][9][0] = 7
        for field in (0, 1):
            # Field zero's known prefix now includes bit 1. Field one is green.
            bc7["historyEndpointLow2Fits"] = [[9, field, 2]]
            with self.assertRaises(ValueError):
                history.bptc.load_bc7(docs)

    def test_visible_2d_mip_starts_after_first_256_compressed_blocks(self):
        # Mip 1 begins 4 KiB into the compressed tail; its larger sibling is raw BC7.
        # Carry the synthetic pre-start alpha value to blocks 256..258.
        solid = bytes([0x40]) + bytes(15)
        secret = (0, 0, 0, 128) * 6
        tail = _encode_bc7_mode6_tiled(solid * 512, secret_entry=255, secret_ep=secret, ref_count=255)
        with tempfile.TemporaryDirectory() as name:
            pool = Path(name)/"late-mip.ttpf"
            record = _pack_etpf(128, 128, 4, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 8192, tail), (8, 16384, zlib.compress(solid * 1024))])
            bc7 = history.fit_slices([pool])["bc7"]
        for field in (3, 7):
            self.assertGreaterEqual(bc7["historyEndpointBits"][255][field], 4)
            self.assertEqual(bc7["historyEndpointConfirmedBits"][255][field], 0)
            self.assertLessEqual(abs(bc7["historyEndpoints"][255][field] - 128), 4)

    def test_packed_tail_fits_against_decoded_bcpack_sibling(self):
        # Mip 0 is zlib, mip 1 a self-contained BCPack stream and the packed tail (mips 2-4) starts its
        # chains at history entry 255, whose alpha no raw or zlib sibling of the tail can show.
        solid = bytes([0x40]) + bytes(15)
        secret = (0, 0, 0, 128) * 6
        tail = _encode_bc7_mode6_tiled(solid * 256, secret_entry=255, secret_ep=secret, ref_count=255)
        with tempfile.TemporaryDirectory() as name:
            pool = Path(name)/"bcpack-parent.ttpf"
            record = _pack_etpf(128, 128, 4, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 4096, tail), (7, 4096, _encode_bc7_mode6_tiled(solid * 256)),
                                                     (8, 16384, zlib.compress(solid * 1024))])
            fits = {6: history.calib.Bc6hCalibrator(), 7: history.calib.Bc7Calibrator()}
            history.calib.calibrate_from_pools([str(pool)], fits)
            docs = {history.calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in fits.items()}
            history.extend_missing(docs, history.fit_slices([pool]))
            bcpack = history.calib.bcpack
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(tail, 7, 4096, allow_provisional=True, history=docs)
            history.fit_decoded_neighbors([pool], docs)
        bc7 = docs["bc7"]
        for field in (3, 7):
            self.assertGreaterEqual(bc7["historyEndpointBits"][255][field], 4)
            self.assertEqual(bc7["historyEndpointConfirmedBits"][255][field], 0)
            self.assertLessEqual(abs(bc7["historyEndpoints"][255][field] - 128), 4)
        self.assertEqual(bcpack.decode(tail, 7, 4096, allow_provisional=True, history=docs), solid * 256)

    def test_packed_tail_fits_against_its_own_refused_stream(self):
        # One BCPack stream holds the packed tail (blocks 0-255) and mip 1 (blocks 256-511). Chains from history
        # entry 255 stay inside the tail, so the stream is refused and no fit reaches the entry, but mip 1 is
        # determined and is the reference.
        solid = bytes([0x40]) + bytes(15)
        secret = (0, 0, 0, 128) * 6
        stream = _encode_bc7_mode6_tiled(solid * 512, secret_entry=255, secret_ep=secret, ref_count=200)
        with tempfile.TemporaryDirectory() as name:
            pool = Path(name)/"one-stream.ttpf"
            record = _pack_etpf(128, 128, 4, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 8192, stream), (8, 16384, zlib.compress(solid * 1024))])
            fits = {6: history.calib.Bc6hCalibrator(), 7: history.calib.Bc7Calibrator()}
            history.calib.calibrate_from_pools([str(pool)], fits)
            docs = {history.calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in fits.items()}
            history.extend_missing(docs, history.fit_slices([pool]))
            bcpack = history.calib.bcpack
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(stream, 7, 8192, allow_provisional=True, history=docs)
            history.fit_decoded_neighbors([pool], docs)
        bc7 = docs["bc7"]
        for field in (3, 7):
            self.assertGreaterEqual(bc7["historyEndpointBits"][255][field], 4)
            self.assertEqual(bc7["historyEndpointConfirmedBits"][255][field], 0)
            self.assertLessEqual(abs(bc7["historyEndpoints"][255][field] - 128), 4)
        self.assertEqual(bcpack.decode(stream, 7, 8192, allow_provisional=True, history=docs), solid * 512)

    def test_build_resume_and_repair(self):
        trainer = os.environ.get("LEONIDA_TEST_BCPACK_TRAINER")
        if not trainer and not shutil.which("cl" if os.name == "nt" else "c++"):
            self.skipTest("a host C++ compiler or prebuilt fitter is required")
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            pool = work/"synthetic.ttpf"
            # Solid mode-6 BC7 blocks. No copyrighted payload or title files.
            blocks = (bytes([0x40]) + bytes(15)) * 256
            encoded = _encode_bc7_mode6_tiled(blocks)
            record = _pack_etpf(64, 64, 1, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 4096, encoded)])
            sidecar = work/"shaders"/"bcpack"/"history.bcph"
            def build():
                history.build(pool, sidecar, Path(trainer) if trainer else None, if_needed=True)
            build()
            original = sidecar.read_bytes()
            self.assertEqual((original[:8], len(original)), (b"BCPHIST1", 60936))
            when = sidecar.stat().st_mtime_ns
            build()
            self.assertEqual(when, sidecar.stat().st_mtime_ns)
            sidecar.write_bytes(bytes(len(original)))
            build()
            self.assertEqual(original, sidecar.read_bytes())
