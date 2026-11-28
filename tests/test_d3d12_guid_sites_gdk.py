"""Unit tests for GDK GUID loading in tools/d3d12_guid_sites.py."""

from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import d3d12_guid_sites as dgs  # noqa: E402


class TestD3d12GuidSitesGdk(unittest.TestCase):
    def test_read_gdk_guids_parses_classes_and_interfaces(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "gdk.json"
            p.write_text(
                json.dumps(
                    {
                        "classes": {"XStoreImpl": {"id": "0dd112ac-7c24-448c-b92b-3960fb5bd30c"}},
                        "interfaces": {
                            "IXStoreImpl6": {"id": "b09d803c-2414-4a05-82c6-66dfdc9e9a44"},
                            "BadEntry": {"id": "not-a-uuid"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            guids = dgs.read_gdk_guids(str(p))
            self.assertEqual(guids["XStoreImpl"], uuid.UUID("0dd112ac-7c24-448c-b92b-3960fb5bd30c"))
            self.assertEqual(guids["IXStoreImpl6"], uuid.UUID("b09d803c-2414-4a05-82c6-66dfdc9e9a44"))
            self.assertNotIn("BadEntry", guids)

    def test_read_gdk_guids_missing_or_invalid_returns_empty(self) -> None:
        self.assertEqual(dgs.read_gdk_guids("/nonexistent/gdk.json"), {})

    def test_read_gdk_guids_non_dict_json_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "gdk.json"
            for doc in ("[1, 2, 3]", '"text"', "42", "null"):
                p.write_text(doc, encoding="utf-8")
                self.assertEqual(dgs.read_gdk_guids(str(p)), {})


if __name__ == "__main__":
    unittest.main()
