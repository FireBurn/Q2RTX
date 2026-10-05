# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location("normalize", Path(__file__).parents[1] / "tools/normalize_spirv_bindings.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def binary(*instructions):
    words = [0x07230203, 0x00010300, 0, 100, 0]
    for opcode, operands in instructions:
        words += [((len(operands) + 1) << 16) | opcode] + operands
    return struct.pack(f"<{len(words)}I", *words)


class BindingTests(unittest.TestCase):
    def test_split_aliases_preserves_every_other_instruction(self):
        original = binary((71, [10, 34, 1]), (71, [10, 33, 1]),
            (71, [20, 34, 1]), (71, [20, 33, 1]),
            (59, [3, 10, 0]), (59, [4, 20, 0]), (0, []))
        result, mapping = tool.normalize(original)
        self.assertEqual([m["new_binding"] for m in mapping], [0, 1])
        self.assertEqual([m["old_binding"] for m in mapping], [1, 1])
        before, after = tool.instructions(original)[1], tool.instructions(result)[1]
        self.assertEqual([w for o, w in before if o != 71], [w for o, w in after if o != 71])
        self.assertEqual(tool.normalize(result)[0], result)

    def test_rejects_grouped_and_incomplete_decorations(self):
        for original in (binary((73, [2])), binary((71, [10, 33, 1]), (59, [3, 10, 0]))):
            with self.assertRaises(ValueError):
                tool.normalize(original)

    def test_rejects_bad_instruction_lengths(self):
        valid = binary((0, []))
        for original in (valid[:-1], valid[:20] + struct.pack('<I', 0), valid[:20] + struct.pack('<I', 5 << 16)):
            with self.assertRaises(ValueError):
                tool.instructions(original)
