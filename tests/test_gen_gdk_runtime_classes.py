"""Unit tests for tools/gen_gdk_runtime_classes.py."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import gen_gdk_runtime_classes as gen  # noqa: E402


class TestParseLayout(unittest.TestCase):
    def test_returns_none_without_layout_section(self) -> None:
        self.assertIsNone(gen.parse_layout("# IFoo\nNo known methods.\n"))

    def test_parses_methods_and_unknown_slots(self) -> None:
        doc = (
            "# IExample\n\n"
            "Layout:\n"
            "- [Query](Query.md)\n"
            "- *unknown*\n"
            "- PlainMethod\n"
            "\nTrailing paragraph stops parsing.\n"
            "- IgnoredBullet\n"
        )
        self.assertEqual(gen.parse_layout(doc), ["Query", None, "PlainMethod"])


class TestParseDocsAndCheck(unittest.TestCase):
    def test_missing_com_folder_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                gen.parse_docs(tmp)

    def test_parses_classes_and_interfaces_and_detects_missing_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cls_dir = root / "COM" / "XUserImpl"
            cls_dir.mkdir(parents=True)

            (cls_dir / "README.md").write_text(
                "Class Id: `AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE`\n"
                "Implements: [IXUser](IXUser.md), [IMissing](IMissing.md)\n",
                encoding="utf-8",
            )
            (cls_dir / "IXUser.md").write_text(
                "Interface Id: `11111111-2222-3333-4444-555555555555`\n"
                "Inherits: IUnknown\n"
                "Layout:\n"
                "- GetId\n",
                encoding="utf-8",
            )
            (cls_dir / "IXUser2.md").write_text(
                "Interface Id: `66666666-7777-8888-9999-000000000000`\n"
                "Inherits: [IXMissingBase](IXMissingBase.md)\n",
                encoding="utf-8",
            )

            classes, interfaces = gen.parse_docs(tmp)
            self.assertEqual(
                classes["XUserImpl"],
                {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "implements": ["IXUser", "IMissing"]},
            )
            self.assertIsNone(interfaces["IXUser"]["inherits"])
            self.assertEqual(interfaces["IXUser"]["methods"], ["GetId"])
            self.assertEqual(interfaces["IXUser2"]["inherits"], "IXMissingBase")
            self.assertIsNone(interfaces["IXUser2"]["methods"])

            problems = gen.check(classes, interfaces)
            self.assertIn("XUserImpl implements IMissing, which has no page", problems)
            self.assertIn("IXUser2 inherits IXMissingBase, which has no page", problems)


class TestBuildAndMain(unittest.TestCase):
    def test_revision_fallback_when_not_git_repo(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertEqual(gen.revision(tmp), "unknown revision")

    def test_main_writes_json_and_handles_missing_docs(self) -> None:
        with TemporaryDirectory() as tmp:
            missing_dir = str(Path(tmp) / "does_not_exist")
            self.assertEqual(gen.main([missing_dir]), 1)

            docs = Path(tmp) / "docs"
            cls_dir = docs / "COM" / "Demo"
            cls_dir.mkdir(parents=True)
            (cls_dir / "README.md").write_text(
                "Class Id: `00000000-0000-0000-0000-000000000001`\n"
                "Implements: [IDemo](IDemo.md)\n",
                encoding="utf-8",
            )
            (cls_dir / "IDemo.md").write_text(
                "Interface Id: `00000000-0000-0000-0000-000000000002`\n"
                "Inherits: IUnknown\n"
                "Layout:\n"
                "- Ping\n",
                encoding="utf-8",
            )
            out_json = Path(tmp) / "out.json"
            with mock.patch.object(gen, "revision", return_value="deadbeef"):
                rc = gen.main([str(docs), "--out", str(out_json)])
            self.assertEqual(rc, 0)
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(data["source"].endswith("deadbeef"))
            self.assertEqual(data["interfaces"]["IDemo"]["methods"], ["Ping"])


if __name__ == "__main__":
    unittest.main()
