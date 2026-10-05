/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_unified_sr.h"
#include <assert.h>
#include <string.h>

int main(void)
{
    /* 1. Test upscaler and tier names */
    assert(strcmp(ffxVkUnifiedSrGetUpscalerName(FFX_VK_UPSCALER_AUTO), "Automatic (Hardware Optimal)") == 0);
    assert(strcmp(ffxVkUnifiedSrGetUpscalerName(FFX_VK_UPSCALER_FSR3), "AMD FidelityFX Super Resolution 3.1") == 0);
    assert(strcmp(ffxVkUnifiedSrGetUpscalerName(FFX_VK_UPSCALER_FSR4), "AMD FidelityFX Super Resolution 4 (INT8/DOT4)") == 0);
    assert(strcmp(ffxVkUnifiedSrGetUpscalerName(FFX_VK_UPSCALER_DLSS), "NVIDIA DLSS / AMD d4r (Tensor/WMMA)") == 0);
    assert(strcmp(ffxVkUnifiedSrGetUpscalerName(FFX_VK_UPSCALER_XESS), "Intel XeSS (DP4a/XMX)") == 0);

    assert(strstr(ffxVkUnifiedSrGetGpuTierName(FFX_VK_GPU_TIER_GENERIC_COMPUTE), "Generic Compute") != NULL);
    assert(strstr(ffxVkUnifiedSrGetGpuTierName(FFX_VK_GPU_TIER_INT8_DOT4), "Integer Dot Product") != NULL);
    assert(strstr(ffxVkUnifiedSrGetGpuTierName(FFX_VK_GPU_TIER_TENSOR_WMMA), "AMD Wave Matrix") != NULL);
    assert(strstr(ffxVkUnifiedSrGetGpuTierName(FFX_VK_GPU_TIER_TENSOR_CORES), "NVIDIA Tensor Cores") != NULL);

    /* 2. Test resolution scaling */
    VkExtent2D display = { 2560, 1440 };
    VkExtent2D render = { 0, 0 };

    assert(ffxVkUnifiedSrGetRenderResolution(FFX_VK_QUALITY_PRESET_NATIVE, display, &render));
    assert(render.width == 2560 && render.height == 1440);

    assert(ffxVkUnifiedSrGetRenderResolution(FFX_VK_QUALITY_PRESET_QUALITY, display, &render));
    assert(render.width == 1708 && render.height == 960); /* Even */

    assert(ffxVkUnifiedSrGetRenderResolution(FFX_VK_QUALITY_PRESET_PERFORMANCE, display, &render));
    assert(render.width == 1280 && render.height == 720);

    assert(ffxVkUnifiedSrGetRenderResolution(FFX_VK_QUALITY_PRESET_ULTRA_PERFORMANCE, display, &render));
    assert(render.width == 854 && render.height == 480);

    /* 3. Test resolution auto-selection and fallback logic */
    FfxVkGpuCapabilities rtxCaps;
    memset(&rtxCaps, 0, sizeof(rtxCaps));
    rtxCaps.vendorId = FFX_VK_VENDOR_NVIDIA;
    rtxCaps.tier = FFX_VK_GPU_TIER_TENSOR_CORES;
    rtxCaps.supportsTensorCores = true;
    rtxCaps.supportsDot4 = true;
    rtxCaps.supportedUpscalersBitmask = (1u << FFX_VK_UPSCALER_FSR3) |
                                       (1u << FFX_VK_UPSCALER_FSR4) |
                                       (1u << FFX_VK_UPSCALER_XESS) |
                                       (1u << FFX_VK_UPSCALER_DLSS);
    rtxCaps.recommendedUpscaler = FFX_VK_UPSCALER_DLSS;

    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_AUTO, &rtxCaps) == FFX_VK_UPSCALER_DLSS);
    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_FSR4, &rtxCaps) == FFX_VK_UPSCALER_FSR4);

    /* AMD RDNA3 with WMMA */
    FfxVkGpuCapabilities rdna3Caps;
    memset(&rdna3Caps, 0, sizeof(rdna3Caps));
    rdna3Caps.vendorId = FFX_VK_VENDOR_AMD;
    rdna3Caps.tier = FFX_VK_GPU_TIER_TENSOR_WMMA;
    rdna3Caps.supportsWmma = true;
    rdna3Caps.supportsDot4 = true;
    rdna3Caps.supportedUpscalersBitmask = (1u << FFX_VK_UPSCALER_FSR3) |
                                         (1u << FFX_VK_UPSCALER_FSR4) |
                                         (1u << FFX_VK_UPSCALER_XESS) |
                                         (1u << FFX_VK_UPSCALER_DLSS); /* d4r supported */
    rdna3Caps.recommendedUpscaler = FFX_VK_UPSCALER_DLSS;

    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_AUTO, &rdna3Caps) == FFX_VK_UPSCALER_DLSS);

    /* AMD RDNA2 (Navi 22 / RX 6800M) */
    FfxVkGpuCapabilities rdna2Caps;
    memset(&rdna2Caps, 0, sizeof(rdna2Caps));
    rdna2Caps.vendorId = FFX_VK_VENDOR_AMD;
    rdna2Caps.tier = FFX_VK_GPU_TIER_INT8_DOT4;
    rdna2Caps.supportsDot4 = true;
    rdna2Caps.supportedUpscalersBitmask = (1u << FFX_VK_UPSCALER_FSR3) |
                                         (1u << FFX_VK_UPSCALER_FSR4) |
                                         (1u << FFX_VK_UPSCALER_XESS);
    rdna2Caps.recommendedUpscaler = FFX_VK_UPSCALER_FSR4;

    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_AUTO, &rdna2Caps) == FFX_VK_UPSCALER_FSR4);
    /* Requesting DLSS on RDNA2 gracefully falls back to FSR4 */
    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_DLSS, &rdna2Caps) == FFX_VK_UPSCALER_FSR4);

    /* Legacy GPU (e.g. Polaris / Pascal) without DOT4 */
    FfxVkGpuCapabilities legacyCaps;
    memset(&legacyCaps, 0, sizeof(legacyCaps));
    legacyCaps.tier = FFX_VK_GPU_TIER_GENERIC_COMPUTE;
    legacyCaps.supportedUpscalersBitmask = (1u << FFX_VK_UPSCALER_FSR3);
    legacyCaps.recommendedUpscaler = FFX_VK_UPSCALER_FSR3;

    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_AUTO, &legacyCaps) == FFX_VK_UPSCALER_FSR3);
    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_DLSS, &legacyCaps) == FFX_VK_UPSCALER_FSR3);
    assert(ffxVkUnifiedSrResolveUpscaler(FFX_VK_UPSCALER_FSR4, &legacyCaps) == FFX_VK_UPSCALER_FSR3);

    /* 4. Test Unified Context Lifecycle */
    FfxVkUnifiedSrContext ctx;
    FfxVkUnifiedSrCreateInfo createInfo;
    memset(&createInfo, 0, sizeof(createInfo));
    createInfo.structSize = sizeof(FfxVkUnifiedSrCreateInfo);
    createInfo.version = FFX_VK_UNIFIED_SR_VERSION;
    createInfo.preferredUpscaler = FFX_VK_UPSCALER_AUTO;
    createInfo.quality = FFX_VK_QUALITY_PRESET_QUALITY;
    createInfo.maxInputExtent = (VkExtent2D){ 1708, 960 };
    createInfo.maxOutputExtent = (VkExtent2D){ 2560, 1440 };

    assert(ffxVkUnifiedSrCreate(&ctx, &createInfo) == VK_SUCCESS);
    assert(ctx.initialized == 1);

    FfxVkUnifiedSrDispatchInfo dispatchInfo;
    memset(&dispatchInfo, 0, sizeof(dispatchInfo));
    dispatchInfo.structSize = sizeof(FfxVkUnifiedSrDispatchInfo);
    dispatchInfo.version = FFX_VK_UNIFIED_SR_VERSION;
    dispatchInfo.commandBuffer = (VkCommandBuffer)(uintptr_t)1;
    dispatchInfo.inputExtent = (VkExtent2D){ 1708, 960 };
    dispatchInfo.outputExtent = (VkExtent2D){ 2560, 1440 };

    assert(ffxVkUnifiedSrDispatch(&ctx, &dispatchInfo) == VK_SUCCESS);

    ffxVkUnifiedSrDestroy(&ctx);
    assert(ctx.initialized == 0);

    return 0;
}
