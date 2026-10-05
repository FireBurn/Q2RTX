import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location("dlss_nr_weights", Path(__file__).parents[1] / "dlss_nr_weights.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def pack(entries):
    data = b""
    for name, blob in entries:
        data += struct.pack("<Q", len(name)) + name + struct.pack("<Q", len(blob)) + blob
    return struct.pack("<Q", len(data) + 8) + data


class WeightsTests(unittest.TestCase):
    def test_offsets_and_blob_boundaries(self):
        data = pack([(b"block0", b"abc"), (b"block1", b"defg")])
        result = tool.index_weights(data)
        self.assertEqual([r["name"] for r in result], ["block0", "block1"])
        self.assertEqual([data[r["offset"]:r["offset"] + r["size"]] for r in result], [b"abc", b"defg"])

    def test_rejects_truncation_even_with_adjusted_total(self):
        data = pack([(b"block0", b"abc")])
        for length in range(1, len(data)):
            truncated = data[:length]
            if length >= 8:
                truncated = struct.pack("<Q", length) + truncated[8:]
            if length == 8:  # Empty table is structurally valid.
                continue
            with self.assertRaises(ValueError):
                tool.index_weights(truncated)

    def test_rejects_duplicate_names_and_empty_blobs(self):
        for entries in ([(b"a", b"x"), (b"a", b"y")], [(b"a", b"")], [(b"", b"x")]):
            with self.assertRaises(ValueError):
                tool.index_weights(pack(entries))
