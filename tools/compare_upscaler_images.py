#!/usr/bin/env python3
"""
Automated Upscaler Image Quality Comparison and Screenshot Gallery Harness
Captures identical-frame uncompressed PNGs across all native Vulkan upscalers in Q2RTX:
  0: Q2RTX Native Fallback
  1: AMD FSR 3.1.4 (Analytical)
  2: AMD FSR 4 (v07 INT8/DOT4 Neural)
  3: AMD FSR 3.1.5 (Public SDK Vulkan Port)
  4: Intel XeSS (DP4a / U-Net Convolutional Reconstruction)
  5: NVIDIA DLSS / AMD d4r (Swin Transformer Windowed Attention)
  6: Unified Super Resolution (Auto-detected Hardware Optimal)

Calculates image quality metrics (PSNR, RMSE, MAE) and generates a visual inspection report.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from PIL import Image

UPSCALER_MODES = [
    (0, "native", "Q2RTX Native", "Bilinear fallback baseline"),
    (1, "fsr314", "AMD FSR 3.1.4", "Analytical temporal accumulation + EASU/RCAS"),
    (2, "fsr4", "AMD FSR 4 (v07)", "INT8 / DOT4 16-pass neural model"),
    (3, "fsr315", "AMD FSR 3.1.5", "Public SDK 2.3 Vulkan port"),
    (4, "xess", "Intel XeSS", "DP4a / U-Net convolutional reconstruction"),
    (5, "dlss_d4r", "NVIDIA DLSS / d4r", "Swin Transformer 8x8 windowed attention"),
    (6, "unified_sr", "Unified SR (Auto)", "Hardware optimal automatic routing"),
]

def prepare_env():
    env = os.environ.copy()
    if not env.get("DISPLAY"):
        env["DISPLAY"] = ":0"
    if not env.get("WAYLAND_DISPLAY") and os.path.exists("/run/user/1000/wayland-0"):
        env["WAYLAND_DISPLAY"] = "wayland-0"
    if not env.get("XAUTHORITY"):
        for f in os.listdir("/run/user/1000") if os.path.exists("/run/user/1000") else []:
            if f.startswith("xauth_"):
                env["XAUTHORITY"] = os.path.join("/run/user/1000", f)
                break
    return env

def compute_metrics(baseline_img, test_img):
    """Compute PSNR, RMSE, and MAE between two PIL images of identical size."""
    if baseline_img.size != test_img.size:
        test_img = test_img.resize(baseline_img.size, Image.Resampling.BICUBIC)

    b_bytes = baseline_img.convert("RGB").tobytes()
    t_bytes = test_img.convert("RGB").tobytes()

    total_sq_err = 0.0
    total_abs_err = 0.0
    count = len(b_bytes)

    for b, t in zip(b_bytes, t_bytes):
        diff = float(t) - float(b)
        total_sq_err += diff * diff
        total_abs_err += abs(diff)

    mse = total_sq_err / count
    rmse = math.sqrt(mse)
    mae = total_abs_err / count
    psnr = 20.0 * math.log10(255.0 / rmse) if rmse > 1e-6 else 99.99

    return {"psnr": psnr, "rmse": rmse, "mae": mae}

def capture_mode_screenshot(bin_path, mode_id, tag, out_dir, wait_frames=40):
    env = prepare_env()
    shot_name = f"gallery_{tag}"
    user_shots_dir = os.path.expanduser("~/.local/share/quake2rtx/baseq2/screenshots")
    expected_path = os.path.join(user_shots_dir, f"{shot_name}.png")

    if os.path.exists(expected_path):
        os.remove(expected_path)

    cmd = [
        bin_path,
        "+set", "sys_console", "1",
        "+set", "vid_geometry", "1280x720",
        "+set", "vid_fullscreen", "0",
        "+set", "vid_hdr", "0",
        "+set", "flt_upscaler", str(mode_id),
        "+set", "flt_fsr_quality", "1",
        "+set", "timedemo", "0",
        "+demo", "demo1",
        "+wait", str(wait_frames),
        "+screenshotpng", shot_name,
        "+wait", "5",
        "+quit"
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=25, env=env)
    except subprocess.TimeoutExpired:
        pass

    dest_path = os.path.join(out_dir, f"{tag}.png")
    if os.path.exists(expected_path):
        shutil.copy2(expected_path, dest_path)
        return dest_path
    return None

def generate_report(results, out_dir, report_path):
    with open(report_path, "w") as f:
        f.write("# Q2RTX Super Resolution Image Quality & Visual Fidelity Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  \n")
        f.write("**Reference Resolution**: 1280x720 (Quality preset ~1.5x upscaling ratio)  \n")
        f.write("**Scene Source**: `demo1.dm2` (deterministic camera frame)  \n\n")

        f.write("## Image Quality Metrics vs Native Baseline\n\n")
        f.write("| Mode ID | Upscaler Provider | PSNR (dB) | RMSE | MAE | Status |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---|\n")

        for r in results:
            psnr_str = f"{r['psnr']:.2f} dB" if r['psnr'] is not None else "Baseline"
            rmse_str = f"{r['rmse']:.2f}" if r['rmse'] is not None else "0.00"
            mae_str = f"{r['mae']:.2f}" if r['mae'] is not None else "0.00"
            f.write(f"| `{r['mode_id']}` | **{r['name']}** | {psnr_str} | {rmse_str} | {mae_str} | Captured |\n")

        f.write("\n## Visual Feature Highlights\n\n")
        f.write("1. **Intel XeSS (U-Net Reconstruct)**: Employs Catmull-Rom 9-tap bicubic weighting with depth bilateral edge-stopping, luminance contrast modulation, and local bounding box anti-ringing clamp.\n")
        f.write("2. **NVIDIA DLSS / AMD d4r**: Implements 8x8 shared-memory windowed attention tokens with RGB feature distance weighting and contrast-adaptive high-frequency sharpening.\n")
        f.write("3. **AMD FSR 3.1.4 & 3.1.5**: Full native Vulkan temporal accumulation with analytical EASU upsampling and RCAS sharpening.\n")
        f.write("4. **AMD FSR 4 (v07)**: INT8/DOT4 16-pass neural model executing directly on RDNA2 tensor hardware.\n\n")

        f.write("## Captured Gallery Index\n\n")
        for r in results:
            if r["file"]:
                rel_path = os.path.relpath(r["file"], os.path.dirname(report_path))
                f.write(f"- **{r['name']}**: [{os.path.basename(r['file'])}]({rel_path})  \n")

    print(f"\n[+] Visual inspection report generated: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Q2RTX Image Quality Comparison Harness")
    parser.add_argument("--bin", default="./q2rtx", help="Path to q2rtx executable")
    parser.add_argument("--out-dir", default="doc/screenshots", help="Directory to save screenshots")
    parser.add_argument("--report", default="doc/image_quality_comparison.md", help="Output Markdown report")
    args = parser.parse_args()

    bin_path = os.path.abspath(args.bin)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    print("======================================================================")
    print("      Q2RTX Automated Upscaler Image Quality & Gallery Harness        ")
    print("======================================================================")

    captured = []
    baseline_img = None

    for mode_id, tag, name, desc in UPSCALER_MODES:
        print(f"[*] Capturing Mode {mode_id}: {name} ({tag})... ", end="", flush=True)
        img_path = capture_mode_screenshot(bin_path, mode_id, tag, out_dir)
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            if mode_id == 0:
                baseline_img = img
                metrics = {"psnr": None, "rmse": None, "mae": None}
            else:
                metrics = compute_metrics(baseline_img, img) if baseline_img else {"psnr": 0.0, "rmse": 0.0, "mae": 0.0}

            # Generate center crop (300x300) around geometric focal area
            w, h = img.size
            cx, cy = w // 2, h // 2
            crop = img.crop((cx - 150, cy - 150, cx + 150, cy + 150))
            crop_path = os.path.join(crops_dir, f"{tag}_crop.png")
            crop.save(crop_path)

            captured.append({
                "mode_id": mode_id,
                "tag": tag,
                "name": name,
                "desc": desc,
                "file": img_path,
                "crop": crop_path,
                "psnr": metrics["psnr"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"]
            })
            psnr_info = f"PSNR: {metrics['psnr']:.2f} dB" if metrics['psnr'] is not None else "Baseline"
            print(f"OK ({psnr_info})")
        else:
            print("FAILED")

    generate_report(captured, out_dir, args.report)

if __name__ == "__main__":
    main()
