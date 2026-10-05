import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("cuda_fatbin_tool", Path(__file__).parents[1] / "cuda_fatbin_tool.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def envelope(payload):
    return struct.pack("<IHHQ", 0xba55ed50, 1, 16, len(payload)) + payload


class FatbinTests(unittest.TestCase):
    def test_truncated_and_oversized_envelopes(self):
        for data in (tool.MAGIC, envelope(b"123")[:-1], envelope(b"")):
            result = tool.scan(data)
            self.assertFalse(result["envelopes"])
            self.assertEqual(len(result["rejected_markers"]), 1)

    def test_skips_nested_marker_and_finds_following_envelope(self):
        result = tool.scan(b"prefix" + envelope(envelope(b"inner")) + b"gap" + envelope(b"last"))
        self.assertEqual(len(result["envelopes"]), 2)
        self.assertEqual(result["envelopes"][0]["offset"], 6)
        self.assertFalse(result["rejected_markers"])

    def test_unknown_header_does_not_hide_later_valid_envelope(self):
        bad = struct.pack("<IHHQ", 0xba55ed50, 2, 16, 100000)
        result = tool.scan(bad + envelope(b"valid"))
        self.assertEqual(len(result["envelopes"]), 1)
        self.assertEqual(len(result["rejected_markers"]), 1)

    def test_extract_duplicate_and_protect_existing_file(self):
        data = envelope(b"test") * 2
        inventory = tool.scan(data)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tool.extract(data, inventory, directory)
            files = list(directory.glob("*.fatbin"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), envelope(b"test"))
            tool.extract(data, inventory, directory)
            files[0].write_bytes(b"different")
            with self.assertRaises(ValueError):
                tool.extract(data, inventory, directory)
            self.assertEqual(files[0].read_bytes(), b"different")
