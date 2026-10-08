# Unified Super Resolution Minimal Sample

This sample demonstrates how an independent game engine or application can integrate `ffx-vulkan::unified-sr` in just a few lines of C code.

## Key Highlights

- **Vendor-Neutral Super Resolution**: Integrates AMD FSR, NVIDIA DLSS / d4r, and Intel XeSS under a single clean Vulkan C API.
- **Auto-Detection**: Automatically detects GPU hardware vendor and architecture capabilities (`ffxVkUnifiedSrQueryGpuCapabilities`) and selects the optimal upscaler (`FFX_VK_UPSCALER_AUTO`).
- **No Proprietary Dependencies**: Operates natively in Vulkan without requiring DirectX 12, Wine, or closed-source vendor SDK runtimes.

## Building and Running

When `FFX_VK_PORTABLE_BUILD_TESTS` is enabled in `ffx-vulkan`, this target is built as `ffx_vk_unified_sr_sample` and executed under CTest.
