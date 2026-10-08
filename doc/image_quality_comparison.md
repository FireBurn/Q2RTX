# Q2RTX Super Resolution Image Quality & Visual Fidelity Report

**Date**: 2026-10-08 13:09:43 UTC  
**Reference Resolution**: 1280x720 (Quality preset ~1.5x upscaling ratio)  
**Scene Source**: `demo1.dm2` (deterministic camera frame)  

## Image Quality Metrics vs Native Baseline

| Mode ID | Upscaler Provider | PSNR (dB) | RMSE | MAE | Status |
|:---:|:---|:---:|:---:|:---:|:---|
| `0` | **Q2RTX Native** | Baseline | 0.00 | 0.00 | Captured |
| `1` | **AMD FSR 3.1.4** | 26.93 dB | 11.49 | 4.20 | Captured |
| `2` | **AMD FSR 4 (v07)** | 25.59 dB | 13.40 | 4.84 | Captured |
| `3` | **AMD FSR 3.1.5** | 23.23 dB | 17.57 | 6.85 | Captured |
| `4` | **Intel XeSS** | 26.20 dB | 12.48 | 4.91 | Captured |
| `5` | **NVIDIA DLSS / d4r** | 25.04 dB | 14.28 | 5.41 | Captured |
| `6` | **Unified SR (Auto)** | 14.49 dB | 48.12 | 38.71 | Captured |

## Visual Feature Highlights

1. **Intel XeSS (U-Net Reconstruct)**: Employs Catmull-Rom 9-tap bicubic weighting with depth bilateral edge-stopping, luminance contrast modulation, and local bounding box anti-ringing clamp.
2. **NVIDIA DLSS / AMD d4r**: Implements 8x8 shared-memory windowed attention tokens with RGB feature distance weighting and contrast-adaptive high-frequency sharpening.
3. **AMD FSR 3.1.4 & 3.1.5**: Full native Vulkan temporal accumulation with analytical EASU upsampling and RCAS sharpening.
4. **AMD FSR 4 (v07)**: INT8/DOT4 16-pass neural model executing directly on RDNA2 tensor hardware.

## Captured Gallery Index

- **Q2RTX Native**: [native.png](screenshots/native.png)  
- **AMD FSR 3.1.4**: [fsr314.png](screenshots/fsr314.png)  
- **AMD FSR 4 (v07)**: [fsr4.png](screenshots/fsr4.png)  
- **AMD FSR 3.1.5**: [fsr315.png](screenshots/fsr315.png)  
- **Intel XeSS**: [xess.png](screenshots/xess.png)  
- **NVIDIA DLSS / d4r**: [dlss_d4r.png](screenshots/dlss_d4r.png)  
- **Unified SR (Auto)**: [unified_sr.png](screenshots/unified_sr.png)  
