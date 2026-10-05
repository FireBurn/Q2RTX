import struct
import unittest
from tools.ffx_dxil.dlss_model_tool import (
    parse_d4r_manifest,
    decode_dlss_nr_weights,
    generate_vulkan_schedule,
)


class DlssModelToolTests(unittest.TestCase):
    def test_parse_d4r_manifest_valid(self):
        lines = [
            "# d4r kernels manifest",
            "",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef hiluma_swin_encoder gfx1100 native_swin",
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789 hiluma_output_kernel gfx1201 direct_output native_fp8",
        ]
        entries = parse_d4r_manifest(lines)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["gpu_arch"], "gfx1100")
        self.assertEqual(entries[0]["kernel_name"], "hiluma_swin_encoder")
        self.assertIn("native_swin", entries[0]["flags"])
        self.assertEqual(entries[1]["gpu_arch"], "gfx1201")
        self.assertIn("native_fp8", entries[1]["flags"])

    def test_parse_d4r_manifest_invalid_hash_or_arch(self):
        # Short hash
        with self.assertRaises(ValueError):
            parse_d4r_manifest(["short_hash kernel gfx1100"])

        # Non-hex characters
        with self.assertRaises(ValueError):
            parse_d4r_manifest(["z" * 64 + " kernel gfx1100"])

        # Unknown architecture
        with self.assertRaises(ValueError):
            parse_d4r_manifest(["0" * 64 + " kernel unknown_arch"])

        # Duplicate entry
        with self.assertRaises(ValueError):
            parse_d4r_manifest([
                "0" * 64 + " k1 gfx1100",
                "0" * 64 + " k2 gfx1100",
            ])

    def test_decode_dlss_nr_weights_valid(self):
        # Construct synthetic serialization
        payload = b""
        layer1_name = "block0.layer0.conv"
        layer1_data = b"\x01\x02\x03\x04" * 4
        payload += struct.pack("<Q", len(layer1_name)) + layer1_name.encode("utf-8")
        payload += struct.pack("<Q", len(layer1_data)) + layer1_data

        layer2_name = "head.dense"
        layer2_data = b"\x05\x06" * 8
        payload += struct.pack("<Q", len(layer2_name)) + layer2_name.encode("utf-8")
        payload += struct.pack("<Q", len(layer2_data)) + layer2_data

        full_data = struct.pack("<Q", len(payload) + 8) + payload
        res = decode_dlss_nr_weights(full_data)

        self.assertEqual(res["blob_count"], 2)
        self.assertEqual(res["total_size"], len(full_data))
        self.assertEqual(res["layer_prefixes"]["block0"], 1)
        self.assertEqual(res["layer_prefixes"]["head"], 1)
        self.assertEqual(res["blobs"][0]["name"], layer1_name)
        self.assertEqual(res["blobs"][1]["name"], layer2_name)

    def test_decode_dlss_nr_weights_truncated(self):
        with self.assertRaises(ValueError):
            decode_dlss_nr_weights(b"\x00\x00\x00")

        # Header says size is 100, but buffer is only 8 bytes
        with self.assertRaises(ValueError):
            decode_dlss_nr_weights(struct.pack("<Q", 100))

    def test_generate_vulkan_schedule(self):
        sched_k = generate_vulkan_schedule("K", 1280, 720, 2560, 1440, "gfx1100")
        self.assertEqual(sched_k["model_id"], "K")
        self.assertFalse(sched_k["is_neural_rendering"])
        self.assertEqual(len(sched_k["descriptor_bindings"]), 7)

        sched_5 = generate_vulkan_schedule("5", 1280, 720, 2560, 1440, "gfx1201")
        self.assertEqual(sched_5["model_id"], "5")
        self.assertTrue(sched_5["is_neural_rendering"])
        # Neural rendering adds 9 bindings: 7 + 9 = 16 bindings
        self.assertEqual(len(sched_5["descriptor_bindings"]), 16)
        names = [b["name"] for b in sched_5["descriptor_bindings"]]
        self.assertIn("NormalsRoughnessMaterial", names)
        self.assertIn("DirectDiffuseRadiance", names)
        self.assertIn("IndirectDiffuseRadiance", names)
        self.assertIn("WeightsTensorBuffer", names)


if __name__ == "__main__":
    unittest.main()
