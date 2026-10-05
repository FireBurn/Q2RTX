#!/usr/bin/env python3
"""Tests for xess_model_tool."""
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "ffx_dxil"))

import xess_model_tool as tool


class XeSSModelToolTests(unittest.TestCase):
    def test_unet_topology_integrity(self):
        summary = tool.validate_unet_topology()
        self.assertEqual(summary["dispatches"], 14)
        self.assertEqual(summary["layers"], 13)
        self.assertEqual(summary["total_bytes"], 253280)

    def test_synthetic_pack_and_unpack(self):
        # Create synthetic valid data for all 13 layers
        synthetic_map = {}
        for l in tool.XESS_UNET_LAYERS:
            synthetic_map[l["name"]] = {
                "weights": bytes([(i % 256) for i in range(l["weights_bytes"])]),
                "scale": bytes([(i % 256) for i in range(l["scale_bytes"])]),
                "bias": bytes([(i % 256) for i in range(l["bias_bytes"])]),
            }

        packed = tool.pack_model(synthetic_map)
        self.assertTrue(packed.startswith(b"XESSMOD2"))

        unpacked = tool.unpack_model(packed)
        self.assertEqual(len(unpacked), 13)
        for name, data in synthetic_map.items():
            self.assertIn(name, unpacked)
            self.assertEqual(unpacked[name]["weights"], data["weights"])
            self.assertEqual(unpacked[name]["scale"], data["scale"])
            self.assertEqual(unpacked[name]["bias"], data["bias"])

    def test_invalid_layer_rejected(self):
        incomplete_map = {"XeSS_i8": {"weights": b"\x00" * 320, "scale": b"\x00" * 80, "bias": b"\x00" * 80}}
        with self.assertRaises(ValueError):
            tool.pack_model(incomplete_map)

    def test_corrupted_magic_rejected(self):
        with self.assertRaises(ValueError):
            tool.unpack_model(b"BADMAGIC" + b"\x00" * 200)

    def test_provenance_extraction_if_available(self):
        prov_path = REPO_ROOT / "build" / "xess-audit" / "trace-3" / "upload-copy-provenance.json"
        uploads_dir = REPO_ROOT / "build" / "xess-audit" / "trace-3" / "uploads"
        if prov_path.exists() and uploads_dir.exists():
            extracted = tool.extract_from_provenance(prov_path, uploads_dir)
            self.assertEqual(len(extracted), 13)
            packed = tool.pack_model(extracted)
            unpacked = tool.unpack_model(packed)
            self.assertEqual(len(unpacked), 13)


if __name__ == "__main__":
    unittest.main()
