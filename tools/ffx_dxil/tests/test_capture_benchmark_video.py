#!/usr/bin/env python3
"""Unit tests for tools/capture_benchmark_video.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import capture_benchmark_video as cbv


class CaptureBenchmarkVideoTests(unittest.TestCase):
    def test_upscaler_profiles_integrity(self):
        expected_keys = ["native", "fsr314", "fsr4", "fsr315", "xess", "dlss_d4r", "unified_sr"]
        for k in expected_keys:
            self.assertIn(k, cbv.UPSCALER_PROFILES)
            prof = cbv.UPSCALER_PROFILES[k]
            self.assertIn("name", prof)
            self.assertIn("tech", prof)
            self.assertIn("fps_720p", prof)
            self.assertIn("ms_720p", prof)
            self.assertIn("color", prof)
            self.assertGreater(prof["fps_720p"], 0.0)
            self.assertGreater(prof["ms_720p"], 0.0)

    def test_load_fonts(self):
        fonts = cbv.load_fonts()
        self.assertIn("title", fonts)
        self.assertIn("body", fonts)
        self.assertIn("small", fonts)
        self.assertIsNotNone(fonts["title"])

    def test_render_split_frame(self):
        fonts = cbv.load_fonts()
        img_a = Image.new("RGB", (640, 360), (30, 40, 60))
        img_b = Image.new("RGB", (640, 360), (10, 80, 50))
        prof_a = cbv.UPSCALER_PROFILES["native"]
        prof_b = cbv.UPSCALER_PROFILES["fsr4"]

        frame = cbv.render_split_frame(img_a, img_b, 640, 360, 0.25, prof_a, prof_b, fonts, sweep=True)
        self.assertEqual(frame.size, (640, 360))
        self.assertEqual(frame.mode, "RGB")

    def test_render_grid_frame(self):
        fonts = cbv.load_fonts()
        images = {
            "native": Image.new("RGB", (320, 180), (20, 20, 20)),
            "fsr4": Image.new("RGB", (320, 180), (40, 40, 40)),
            "xess": Image.new("RGB", (320, 180), (60, 60, 60)),
            "dlss_d4r": Image.new("RGB", (320, 180), (80, 80, 80)),
        }

        frame = cbv.render_grid_frame(images, 640, 360, 0.5, fonts)
        self.assertEqual(frame.size, (640, 360))
        self.assertEqual(frame.mode, "RGB")

    def test_end_to_end_short_encode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_mp4 = os.path.join(tmp_dir, "test_render.mp4")
            fonts = cbv.load_fonts()
            img_a = Image.new("RGB", (320, 240), (20, 30, 40))
            img_b = Image.new("RGB", (320, 240), (40, 50, 60))
            prof_a = cbv.UPSCALER_PROFILES["native"]
            prof_b = cbv.UPSCALER_PROFILES["xess"]

            total_frames = 15  # 0.5s at 30 fps
            def frames():
                for i in range(total_frames):
                    yield cbv.render_split_frame(img_a, img_b, 320, 240, i / float(total_frames), prof_a, prof_b, fonts)

            cbv.encode_video_stream(frames(), total_frames, 320, 240, 30, out_mp4)
            self.assertTrue(os.path.exists(out_mp4))
            self.assertGreater(os.path.getsize(out_mp4), 1000)

            # Check format with ffprobe
            result = subprocess.run(
                ["/usr/bin/ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", out_mp4],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("h264", result.stdout)


if __name__ == "__main__":
    unittest.main()
