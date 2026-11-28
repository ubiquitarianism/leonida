"""A Vulkan device extension is enabled under its own header guard, never under another extension's.

    python -I tests/test_vulkan_extension_guards.py

The runtime builds against whatever Vulkan headers the machine has, so each optional extension sits behind
``#ifdef VK_..._EXTENSION_NAME``.  When the line that enables extension A sits inside the guard of extension B,
headers that have A but not B drop A from the device while the code that detected A still uses it.  That happened
with ``VK_KHR_maintenance5`` (the A8_UNORM render-target path, ``VulkanDeviceInfo::a8_render_target``), whose
enable line was nested in the ``VK_KHR_maintenance8`` guard of ``create_vulkan_device_standalone``: with
Vulkan-Headers 1.4.304 the device used ``VK_FORMAT_A8_UNORM_KHR`` without the extension (review of #428).

The check reads the runtime's C++ sources, tracks the ``#if``/``#ifdef`` stack, and fails on any
``push_back(VK_X_EXTENSION_NAME)`` with a ``VK_Y_EXTENSION_NAME`` guard (Y != X) open around it.  Standard library
only; no build, no GPU.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = (os.path.join("runtime", "shims"), os.path.join("runtime", "src"), os.path.join("runtime", "test"))
PUSH = re.compile(r"push_back\(\s*(VK_\w+_EXTENSION_NAME)\s*\)")
GUARD_NAME = re.compile(r"\b(VK_\w+_EXTENSION_NAME)\b")
DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def foreign_guards(text):
    """[(line, pushed, guard)] for every extension enabled inside another extension's guard."""
    stack = []  # one list of guard names per open #if
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        directive = DIRECTIVE.match(line)
        if directive:
            kind, rest = directive.groups()
            if kind in ("if", "ifdef"):
                stack.append(GUARD_NAME.findall(rest))
            elif kind == "ifndef":
                stack.append([])  # the extension is absent inside: not a guard that enables anything
            elif kind in ("elif", "else"):
                if stack:
                    stack[-1] = [] if kind == "else" else GUARD_NAME.findall(rest)
            elif kind == "endif" and stack:
                stack.pop()
            continue
        for pushed in PUSH.findall(line):
            for guards in stack:
                for guard in guards:
                    if guard != pushed:
                        found.append((number, pushed, guard))
    return found


def runtime_sources(root):
    for folder in SOURCES:
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            for name in sorted(files):
                if name.endswith((".cpp", ".h", ".hpp")):
                    yield os.path.join(dirpath, name)


class ExtensionGuards(unittest.TestCase):
    def test_detects_a_foreign_guard(self):
        nested = (
            "#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME\n"
            "    if (m8) v.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);\n"
            "#ifdef VK_KHR_MAINTENANCE_5_EXTENSION_NAME\n"
            "    if (a8) v.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);\n"
            "#endif\n"
            "#endif\n")
        self.assertEqual(foreign_guards(nested),
                         [(4, "VK_KHR_MAINTENANCE_5_EXTENSION_NAME", "VK_KHR_MAINTENANCE_8_EXTENSION_NAME")])

    def test_accepts_own_guards_and_else_branches(self):
        flat = (
            "#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME\n"
            "    if (m8) v.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);\n"
            "#endif\n"
            "#if defined(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) && defined(_WIN32)\n"
            "    if (a8) v.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);\n"
            "#endif\n"
            "#ifdef VK_EXT_FOO_EXTENSION_NAME\n"
            "#else\n"
            "    v.push_back(VK_EXT_BAR_EXTENSION_NAME);\n"
            "#endif\n")
        self.assertEqual(foreign_guards(flat), [])

    def test_runtime_sources(self):
        problems = []
        for path in runtime_sources(ROOT):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line, pushed, guard in foreign_guards(f.read()):
                    problems.append("%s:%d enables %s inside the guard of %s"
                                    % (os.path.relpath(path, ROOT), line, pushed, guard))
        self.assertEqual(problems, [], "\n".join(problems))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
