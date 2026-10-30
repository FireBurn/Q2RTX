#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import struct
import unittest
from tools.ffx_dxil.pack_dlss_model import pack_dlss_model, pack_dlssnr_model


class PackDlssModelTests(unittest.TestCase):
    def test_pack_dlss_model_k_valid(self):
        data = pack_dlss_model("K")
        self.assertGreaterEqual(len(data), 65536)
        magic, version, model_family, entry_count = struct.unpack_from("<8sIII", data, 0)
        self.assertEqual(magic, b"DLSSMOD1")
        self.assertEqual(version, 1)
        self.assertEqual(model_family, 1)  # Model K
        self.assertGreater(entry_count, 0)

    def test_pack_dlss_model_all_families(self):
        for name, expected_fam in [("E", 0), ("K", 1), ("M", 2), ("L", 3)]:
            data = pack_dlss_model(name)
            magic, version, family, _ = struct.unpack_from("<8sIII", data, 0)
            self.assertEqual(magic, b"DLSSMOD1")
            self.assertEqual(version, 1)
            self.assertEqual(family, expected_fam)

    def test_pack_dlssnr_model_valid(self):
        data = pack_dlssnr_model()
        self.assertGreaterEqual(len(data), 1048576)
        magic, version = struct.unpack_from("<8sI", data, 0)
        self.assertEqual(magic, b"DLSSNR1\0")
        self.assertEqual(version, 1)


if __name__ == "__main__":
    unittest.main()
