/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 *
 * Universal Single-Header Entrypoint for All Scalers and Frame Generation
 * Designed for cross-vendor integration into any Vulkan engine or project:
 * - AMD FidelityFX Super Resolution (FSR 1.0, FSR 2.2, FSR 3.1, FSR 4 Neural)
 * - Intel Xe Super Sampling (XeSS 1.3/1.4 DP4a / U-Net)
 * - NVIDIA Deep Learning Super Sampling (DLSS / d4r Swin Attention)
 * - Universal Multi-Frame Generation Presenter (2x, 3x, 4x)
 * - Unified Super Resolution (Unified SR) with Dynamic Resolution Scaling (DRS)
 */

#ifndef FFX_VK_ALL_SCALERS_H
#define FFX_VK_ALL_SCALERS_H

#include "ffx_vk_portable.h"
#include "ffx_vk_unified_sr.h"
#include "ffx_vk_xess_contract.h"
#include "ffx_vk_dlss_contract.h"
#include "ffx_vk_fsr4_v07.h"
#include "ffx_vk_rayregeneration_contract.h"
#include "ffx_vk_framegeneration_presenter.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define FFX_VK_ALL_SCALERS_VERSION_MAJOR 1
#define FFX_VK_ALL_SCALERS_VERSION_MINOR 0
#define FFX_VK_ALL_SCALERS_VERSION_PATCH 0

/* Master upscaler enumeration across all vendor offerings */
typedef enum FfxVkMasterScalerId {
    FFX_VK_SCALER_OFF = 0,
    FFX_VK_SCALER_FSR3_1_4 = 1,
    FFX_VK_SCALER_FSR4_NEURAL = 2,
    FFX_VK_SCALER_FSR3_1_5 = 3,
    FFX_VK_SCALER_INTEL_XESS = 4,
    FFX_VK_SCALER_NVIDIA_DLSS = 5,
    FFX_VK_SCALER_UNIFIED_AUTO = 6,
    FFX_VK_SCALER_COUNT = 7
} FfxVkMasterScalerId;

/* Universal upscaler descriptor for external engines (e.g. OptiScaler/game engines) */
typedef struct FfxVkMasterScalerInfo {
    FfxVkMasterScalerId id;
    const char *name;
    const char *vendor;
    const char *algorithm;
    bool works_on_amd;
    bool works_on_nvidia;
    bool works_on_intel;
    bool supports_drs;
    bool supports_framegen;
} FfxVkMasterScalerInfo;

static inline FfxVkMasterScalerInfo ffxVkGetScalerInfo(FfxVkMasterScalerId id)
{
    FfxVkMasterScalerInfo info = { id, "Off", "None", "None", true, true, true, false, false };
    switch (id) {
    case FFX_VK_SCALER_FSR3_1_4:
        info.name = "AMD FidelityFX Super Resolution 3.1.4 (FSR 3.1.4)";
        info.vendor = "AMD";
        info.algorithm = "Analytical Temporal Filter";
        info.works_on_amd = true;
        info.works_on_nvidia = true;
        info.works_on_intel = true;
        info.supports_drs = false;
        info.supports_framegen = true;
        break;
    case FFX_VK_SCALER_FSR4_NEURAL:
        info.name = "AMD FidelityFX Super Resolution 4 (FSR 4 Neural ML)";
        info.vendor = "AMD";
        info.algorithm = "INT8 / DOT4 Convolutional Neural Model";
        info.works_on_amd = true;
        info.works_on_nvidia = true;
        info.works_on_intel = true;
        info.supports_drs = true;
        info.supports_framegen = true;
        break;
    case FFX_VK_SCALER_FSR3_1_5:
        info.name = "AMD FidelityFX Super Resolution 3.1.5 (FSR 3.1.5)";
        info.vendor = "AMD";
        info.algorithm = "Public SDK 2.3 Temporal Pipeline";
        info.works_on_amd = true;
        info.works_on_nvidia = true;
        info.works_on_intel = true;
        info.supports_drs = false;
        info.supports_framegen = true;
        break;
    case FFX_VK_SCALER_INTEL_XESS:
        info.name = "Intel Xe Super Sampling (XeSS)";
        info.vendor = "Intel";
        info.algorithm = "DP4a / U-Net Neural Reconstruction";
        info.works_on_amd = true;
        info.works_on_nvidia = true;
        info.works_on_intel = true;
        info.supports_drs = true;
        info.supports_framegen = true;
        break;
    case FFX_VK_SCALER_NVIDIA_DLSS:
        info.name = "NVIDIA Deep Learning Super Sampling (DLSS / d4r)";
        info.vendor = "NVIDIA";
        info.algorithm = "Swin Transformer Windowed Attention";
        info.works_on_amd = true;
        info.works_on_nvidia = true;
        info.works_on_intel = true;
        info.supports_drs = true;
        info.supports_framegen = true;
        break;
    case FFX_VK_SCALER_UNIFIED_AUTO:
        info.name = "Unified Super Resolution (Auto-Selector)";
        info.vendor = "Universal";
        info.algorithm = "Automatic Hardware-Optimal Governor";
        info.works_on_amd = true;
        info.works_on_nvidia = true;
        info.works_on_intel = true;
        info.supports_drs = true;
        info.supports_framegen = true;
        break;
    default:
        break;
    }
    return info;
}

#if defined(__cplusplus)
}
#endif

#endif /* FFX_VK_ALL_SCALERS_H */
