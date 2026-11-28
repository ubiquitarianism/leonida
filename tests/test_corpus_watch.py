"""Counts-only corpus watch: counting and report-path safety.

Standard library only, no game data, no GPU. Run with:
    python3 -I tests/test_corpus_watch.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import corpus_watch  # noqa: E402


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)


class CorpusCounts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = self.tmp.name
        os.makedirs(os.path.join(self.work, "shaders", "dxbc"))
        os.makedirs(os.path.join(self.work, "shaders", "spirv"))

    def test_counts_only_validated_entries_and_per_stage(self):
        write_json(os.path.join(self.work, "shaders", "dxbc", "index.json"), [
            {"stage": "ps"}, {"stage": "ps"}, {"stage": "vs"},
        ])
        write_json(os.path.join(self.work, "shaders", "spirv", "index.json"), {
            "a": {"stage": "ps"}, "b": {"stage": "vs"}, "c": {"stage": "ps", "error": "bad"},
        })
        got = corpus_watch.corpus_counts(self.work)
        self.assertEqual(got["carved"], 3)
        self.assertEqual(got["spirv_entries"], 3)
        self.assertEqual(got["validated"], 2)
        self.assertEqual(got["per_stage"]["ps"], [2, 1])
        self.assertEqual(got["per_stage"]["vs"], [1, 1])

    def test_dict_dxbc_index_and_legacy_layout(self):
        legacy = os.path.join(self.tmp.name, "legacy")
        os.makedirs(os.path.join(legacy, "dxbc"))
        os.makedirs(os.path.join(legacy, "spirv"))
        write_json(os.path.join(legacy, "dxbc", "index.json"), {"x": {"stage": "cs"}})
        write_json(os.path.join(legacy, "spirv", "index.json"), {"x": {"stage": "cs"}})
        got = corpus_watch.corpus_counts(legacy)
        self.assertEqual(got["carved"], 1)
        self.assertEqual(got["validated"], 1)
        self.assertEqual(got["per_stage"]["cs"], [1, 1])

    def test_no_indexes_returns_none(self):
        self.assertIsNone(corpus_watch.corpus_counts(self.work))

    def test_report_paths_must_not_collide(self):
        report = os.path.join(self.work, "report")
        results = os.path.join(self.work, "counts.jsonl")
        self.assertFalse(corpus_watch.outputs_collide(self.work, report, results, None))
        self.assertTrue(corpus_watch.outputs_collide(self.work, report, results, results))
        self.assertTrue(corpus_watch.outputs_collide(self.work, self.work, results, None))


if __name__ == "__main__":
    unittest.main()
