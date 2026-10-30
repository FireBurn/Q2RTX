# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xess_resolve_replay as replay


class ResolveCaptureTests(unittest.TestCase):
    def fixture(self, root):
        trace, shaders = root / "trace", root / "shaders"
        (trace / "uploads").mkdir(parents=True)
        shaders.mkdir()
        shader = shaders / "065ebeb5c40e7a3d.spv"
        shader.write_bytes(b"synthetic payload for materializer tests; not executed")
        shader.with_suffix(".json").write_text(json.dumps({
            "output_sha256": hashlib.sha256(shader.read_bytes()).hexdigest(),
            "bindings": [{"new_binding": i} for i in range(5)]}))
        logs, cbvs = [], []
        for i, (copied, size) in enumerate(zip((320,80,80),(512,256,256))):
            (trace / "uploads" / f"{i}.bin").write_bytes(bytes([i+1])*copied)
            logs.extend([
                f"GetGPUVirtualAddress: iface b{i}, va #{4096+i*4096:x}",
                f"REFERENCE_UPLOAD resource=a{i} va=1000 size={copied} file=/local/{i}.bin",
                f"CopyBufferRegion: iface 1, dst_resource b{i}, dst_offset 0, src_resource a{i}, src_offset 0, byte_count {copied}."])
            cbvs.append({"register": i, "view": {"kind": "CBV", "address": hex(4096+i*4096), "size": size}})
        (trace / "vkd3d.log").write_text("\n".join(logs))
        external = [{"view": {"resource_description": dict(zip(("width","height","format"), shape))}}
            for shape in [(753,424,10),(753,424,34),(1280,720,10),(753,424,41)]]
        dispatch = {"shader": shader.stem, "line": 1000,
            "constants": {"3": {str(i): "00000000" for i in range(20)}},
            "resource_views": {"0": external, "1": [], "2": cbvs}, "groups": [160,90,1]}
        (trace / "dispatch-index.json").write_text(json.dumps({"dispatches": [dispatch]*56}))
        return trace, shaders

    def test_padding_is_explicit_and_captured_prefix_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); trace, shaders = self.fixture(root)
            result = replay.build(trace, shaders, root / "output", 0xa5, 2)
            self.assertEqual(result["frame"], 2)
            self.assertEqual(result["push"][-3:], [0,4,9])
            self.assertEqual(sum(b["uncaptured_padding_bytes"] for b in result["buffers"]),544)
            for i, buffer in enumerate(result["buffers"]):
                payload = Path(buffer["file"]).read_bytes(); n = buffer["captured_bytes"]
                self.assertEqual(payload[:n],bytes([i+1])*n)
                self.assertEqual(payload[n:],bytes([0xa5])*(buffer["view_bytes"]-n))

    def test_truncated_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); trace, shaders = self.fixture(root)
            (trace / "uploads/0.bin").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError,"Upload size mismatch"):
                replay.build(trace,shaders,root / "output")

    def test_out_of_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); trace, shaders = self.fixture(root)
            with self.assertRaisesRegex(ValueError,"Padding byte"):
                replay.build(trace,shaders,root / "output",256)
            with self.assertRaisesRegex(ValueError,"Frame outside"):
                replay.build(trace,shaders,root / "output",frame=4)
            path = trace / "dispatch-index.json"
            data = json.loads(path.read_text())
            data["dispatches"][13]["resource_views"]["0"][0]["view"]["resource_description"]["width"] = 100
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"profile changed"):
                replay.build(trace,shaders,root / "output")


if __name__ == "__main__":
    unittest.main()
