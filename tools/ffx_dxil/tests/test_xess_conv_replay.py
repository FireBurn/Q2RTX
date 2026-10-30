# SPDX-License-Identifier: MIT
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xess_conv_replay as replay


class ReplayCaptureTests(unittest.TestCase):
    def test_copies_require_complete_coverage(self):
        with self.assertRaisesRegex(ValueError, "Uncaptured"):
            replay.materialize(16, [{"offset": 0, "bytes": b"captured"}])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            replay.materialize(4, [{"offset": 1, "bytes": b"1234"}])
        with self.assertRaisesRegex(ValueError, "Unbounded"):
            replay.materialize(0, [])

    def test_copy_order_and_nonzero_offsets(self):
        self.assertEqual(replay.materialize(8, [
            {"offset": 4, "bytes": b"abcd"},
            {"offset": 0, "bytes": b"1234"},
            {"offset": 2, "bytes": b"XY"},
        ]), b"12XYabcd")

    def test_upload_snapshot_and_dispatch_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "uploads").mkdir()
            (path / "uploads/a.bin").write_bytes(b"old-data")
            (path / "uploads/b.bin").write_bytes(b"new-data")
            (path / "vkd3d.log").write_text(
                "REFERENCE_UPLOAD resource=aa va=1000 size=8 file=Z:\\dump\\a.bin\n"
                "CopyBufferRegion: iface 1, dst_resource bb, dst_offset 0, src_resource aa, src_offset 4, byte_count 4.\n"
                "REFERENCE_UPLOAD resource=aa va=1000 size=8 file=Z:\\dump\\b.bin\n"
                "CopyBufferRegion: iface 1, dst_resource bb, dst_offset 0, src_resource aa, src_offset 0, byte_count 4.\n")
            early = replay.copied_buffers(path, 4)
            late = replay.copied_buffers(path, 5)
            self.assertEqual(replay.materialize(4, early["bb"]), b"data")
            self.assertEqual(replay.materialize(4, late["bb"]), b"new-")
            self.assertEqual(late["bb"][0]["upload_file"], "a.bin")
            self.assertEqual(late["bb"][1]["upload_line"], 3)

    def test_shader_identity_and_truncation(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            replay.verify_shader(b"abcd", {"output_sha256": "wrong"})
        empty_hash = hashlib.sha256(b"").hexdigest()
        with self.assertRaisesRegex(ValueError, "Invalid SPIR-V size"):
            replay.verify_shader(b"", {"output_sha256": empty_hash})


if __name__ == "__main__":
    unittest.main()
