# d4r and DLSS 5 Vulkan Integration Guide

This document describes how to use [countervolts/d4r](https://github.com/countervolts/d4r) (DLSS 4 Radeon)
and DLSS 5 Neural Rendering in Vulkan engines using the reusable `ffx-vulkan::dlss-contract` target.

---

## 1. Overview and Architecture

### d4r (DLSS 4 Radeon)
Upstream: <https://github.com/countervolts/d4r>

d4r executes NVIDIA's official DLSS Super Resolution library (`nvngx_dlss.dll`) on AMD Radeon GPUs (RDNA3 and RDNA4) under Linux/Proton:
- **CUDA NGX Driver Path:** Instead of the proprietary NVIDIA D3D12 driver path, d4r initializes NGX via the CUDA driver API (`NVSDK_NGX_CUDA_Init_ProjectID`) backed by ZLUDA/HIP.
- **Hardware Architecture Spoofing:** NGX expects an NVIDIA architecture to select optimal weights. Setting `AD100` / `sm_89` pairs the transformer model with compatible PTX.
- **Native RDNA3/RDNA4 Compute Kernels:** Translating NVIDIA's warp-level tensor core instructions (`mma.sync`) directly to generic GPU code suffers from lane shuffles and register spilling. d4r replaces the heaviest network layers with hand-written AMD Wave Matrix Multiply Accumulate (WMMA) compute kernels:
  - `k/`: DLSS 4 Swin transformer layers (Model K).
  - `m/`: DLSS 4.5 transformer layers (Model M).
  - `l/`: DLSS 4.5 Ultra Performance layers (Model L).
  - `common/`: AMD WMMA layout mappings for `gfx1100`-`gfx1103` (RDNA3, FP8 widened to FP16) and `gfx1200`-`gfx1201` (RDNA4, native FP8 WMMA).
- **VRAM Interop:** Inputs (color, depth, motion) and output stay in GPU VRAM using Vulkan external memory handles (POSIX opaque file descriptors or Win32 handles).

### DLSS 5 Neural Rendering (DLSSNR)
While d4r focuses on DLSS 3/4/4.5 super resolution, NVIDIA's DLSS 5 introduces **Neural Rendering / Neural Reconstruction** (`nvngx_dlssnr.dll`).
- **Neural Radiance Replay:** Reconstructs lighting, view-dependent reflections, and geometry using neural networks.
- **Weights Structure:** Uses a 147.6 MB `WEIGHTS_HT` resource containing 153 named tensor blobs (layer hierarchy: `block0.layer0.layer`, etc.).
- **Hardware Requirements:** Blackwell (`sm_120`) and RDNA4 (`gfx120x`) native FP8 matrix instructions.
- **Input Signals:** Requires multi-channel G-buffer inputs (normals, linear roughness, material class, diffuse/specular albedos) and direct/indirect radiance partitions with first-lobe hit distances and dominant light blocker visibility.

---

## 2. Using `ffx-vulkan::dlss-contract` in Other Vulkan Projects

The reusable target `ffx-vulkan::dlss-contract` provides a clean, dependency-free Vulkan C11/C++ contract that any Vulkan application can use to prepare, validate, and bridge DLSS 3/4/4.5 and DLSS 5 workloads.

### CMake Integration
In your project's `CMakeLists.txt`:
```cmake
find_package(ffx-vulkan REQUIRED)
target_link_libraries(my_vulkan_engine PRIVATE ffx-vulkan::dlss-contract)
```
Or if vendoring `extern/ffx-vulkan`:
```cmake
add_subdirectory(extern/ffx-vulkan)
target_link_libraries(my_vulkan_engine PRIVATE ffx-vulkan::dlss-contract)
```

### C/C++ Example: Initializing DLSS / d4r

```c
#include <ffx_vk_dlss_contract.h>

// 1. Query optimal render resolution for target display
FfxVkPortableExtent2D displayExtent = { 2560, 1440 };
FfxVkPortableExtent2D renderExtent;
ffxVkDlssGetOptimalRenderResolution(displayExtent, FFX_VK_DLSS_PRESET_QUALITY, &renderExtent);
// renderExtent is resolved to 1707x960 (1.5x scale)

// 2. Set up CreateInfo with d4r options
FfxVkDlssCreateInfo createInfo = {
    .structSize = sizeof(FfxVkDlssCreateInfo),
    .contractVersion = FFX_VK_DLSS_CONTRACT_VERSION,
    .model = FFX_VK_DLSS_MODEL_4_5_TRANSFORMER_M, // DLSS 4.5 Model M
    .preset = FFX_VK_DLSS_PRESET_QUALITY,
    .flags = FFX_VK_DLSS_FLAG_NATIVE_SWIN_ENCODERS |
             FFX_VK_DLSS_FLAG_VRAM_INTEROP |
             FFX_VK_DLSS_FLAG_AUTO_EXPOSURE |
             FFX_VK_DLSS_FLAG_HDR_INPUT,
    .gpuArch = FFX_VK_DLSS_ARCH_RDNA3,            // or FFX_VK_DLSS_ARCH_RDNA4 on RX 9000
    .maxRenderSize = renderExtent,
    .displaySize = displayExtent,
    .motionVectorDilation = 2                      // d4r default 2-pixel dilation
};

uint64_t issues = 0;
if (ffxVkDlssValidateCreateInfo(&createInfo, &issues) != FFX_VK_PORTABLE_OK) {
    // Handle validation issues (e.g. unsupported architecture or invalid extents)
}
```

### C/C++ Example: Recording a Dispatch

```c
// 3. Populate per-frame DispatchInfo
FfxVkDlssDispatchInfo dispatchInfo = {
    .structSize = sizeof(FfxVkDlssDispatchInfo),
    .contractVersion = FFX_VK_DLSS_CONTRACT_VERSION,
    .flags = createInfo.flags,
    .renderSize = renderExtent,

    // Vulkan images owned by the application
    .color = {
        .structSize = sizeof(FfxVkPortableImage),
        .image = appColorImage,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .extent = renderExtent,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .state = FFX_VK_PORTABLE_RESOURCE_STATE_COMPUTE_READ
    },
    .depth = {
        .structSize = sizeof(FfxVkPortableImage),
        .image = appDepthImage,
        .format = VK_FORMAT_D32_SFLOAT,
        .extent = renderExtent,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .state = FFX_VK_PORTABLE_RESOURCE_STATE_COMPUTE_READ
    },
    .motionVectors = {
        .structSize = sizeof(FfxVkPortableImage),
        .image = appMotionImage,
        .format = VK_FORMAT_R16G16_SFLOAT,
        .extent = renderExtent,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .state = FFX_VK_PORTABLE_RESOURCE_STATE_COMPUTE_READ
    },
    .output = {
        .structSize = sizeof(FfxVkPortableImage),
        .image = appOutputImage,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .extent = displayExtent,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT,
        .state = FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS
    },

    // Camera and motion metadata
    .jitterOffset = { jitterX, jitterY },
    .motionVectorScale = { 1.0f / (float)renderExtent.width, 1.0f / (float)renderExtent.height },
    .verticalFov = 1.047f, // radians
    .nearZ = 0.1f,
    .farZ = 1000.0f,
    .preExposure = 1.0f,
    .frameReset = isCameraCut ? VK_TRUE : VK_FALSE
};

// 4. Validate fail-closed before execution
if (ffxVkDlssValidateDispatchInfo(&createInfo, &dispatchInfo, &issues) == FFX_VK_PORTABLE_OK) {
    // Safe to execute DLSS / d4r dispatch
}
```

---

## 3. DLSS 5 Neural Rendering Dispatch

When selecting `FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING`, populate the extended neural rendering signals:
- `.normalsRoughnessMaterial`: Octahedral normal, linear roughness, and material classification.
- `.diffuseAlbedo` & `.specularAlbedo`: Sqrt-encoded albedo textures.
- `.directDiffuse` & `.directSpecular`: Linear direct radiance partitions.
- `.indirectDiffuse` & `.indirectSpecular`: Linear indirect radiance partitions (alpha channel contains the physically traced first-lobe hit distance).
- `.dominantLightBlockerDistance`: Dense shadow blocker distance.
- `.weightsTensorBuffer`: Storage buffer mapped to the decoded `WEIGHTS_HT` resource.

---

## 4. Manifest and Model Tooling

The python utility `tools/ffx_dxil/dlss_model_tool.py` provides standalone inspection and validation:

```sh
# Validate a d4r kernel manifest
python3 tools/ffx_dxil/dlss_model_tool.py d4r-manifest path/to/d4r-kernels.txt

# Inspect and index DLSS 5 WEIGHTS_HT binary resource
python3 tools/ffx_dxil/dlss_model_tool.py nr-weights path/to/WEIGHTS_HT.bin --manifest dlss5-manifest.json

# Generate a Vulkan descriptor binding schedule for a game engine
python3 tools/ffx_dxil/dlss_model_tool.py vulkan-schedule --model M --arch gfx1100 --render-size 1280 720 --display-size 2560 1440
```
