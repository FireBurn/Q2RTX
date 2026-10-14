/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_all_scalers.h"
#include <assert.h>
#include <string.h>

int main(void)
{
    /* Test master scaler enumeration and query descriptors */
    for (int i = 0; i < (int)FFX_VK_SCALER_COUNT; ++i) {
        FfxVkMasterScalerInfo info = ffxVkGetScalerInfo((FfxVkMasterScalerId)i);
        assert(info.id == (FfxVkMasterScalerId)i);
        assert(info.name != NULL && strlen(info.name) > 0);
        assert(info.vendor != NULL && strlen(info.vendor) > 0);
        assert(info.algorithm != NULL && strlen(info.algorithm) > 0);
        assert(info.works_on_amd);
        assert(info.works_on_nvidia);
        assert(info.works_on_intel);
    }

    FfxVkMasterScalerInfo fsr4 = ffxVkGetScalerInfo(FFX_VK_SCALER_FSR4_NEURAL);
    assert(strstr(fsr4.name, "FSR 4") != NULL);
    assert(fsr4.supports_drs == true);
    assert(fsr4.supports_framegen == true);

    FfxVkMasterScalerInfo xess = ffxVkGetScalerInfo(FFX_VK_SCALER_INTEL_XESS);
    assert(strstr(xess.name, "XeSS") != NULL);
    assert(xess.supports_drs == true);

    FfxVkMasterScalerInfo dlss = ffxVkGetScalerInfo(FFX_VK_SCALER_NVIDIA_DLSS);
    assert(strstr(dlss.name, "DLSS") != NULL);
    assert(dlss.supports_drs == true);

    FfxVkMasterScalerInfo unified = ffxVkGetScalerInfo(FFX_VK_SCALER_UNIFIED_AUTO);
    assert(strstr(unified.name, "Unified") != NULL);
    assert(unified.supports_drs == true);

    return 0;
}
