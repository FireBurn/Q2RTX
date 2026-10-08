/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 *
 * Standalone Minimal Sample for ffx-vulkan::unified-sr
 * Demonstrates 1-line provider-neutral Super Resolution integration for any Vulkan engine.
 */

#include <ffx_vk_unified_sr.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    printf("=====================================================\n");
    printf(" FidelityFX Vulkan: Unified Super Resolution Sample  \n");
    printf("=====================================================\n\n");

    /* 1. Calculate input render resolution given display resolution and quality preset */
    VkExtent2D displayExtent = { 2560, 1440 };
    VkExtent2D renderExtent = { 0, 0 };

    if (!ffxVkUnifiedSrGetRenderResolution(FFX_VK_QUALITY_PRESET_QUALITY, displayExtent, &renderExtent)) {
        fprintf(stderr, "ERROR: Failed to compute render resolution.\n");
        return 1;
    }
    printf("[1] Display Extent: %ux%u -> Render Extent (Quality): %ux%u (%.1f%% scale)\n",
           displayExtent.width, displayExtent.height,
           renderExtent.width, renderExtent.height,
           (float)renderExtent.width / (float)displayExtent.width * 100.0f);

    /* 2. Configure Unified SR Context Create Info */
    FfxVkUnifiedSrCreateInfo createInfo;
    memset(&createInfo, 0, sizeof(createInfo));
    createInfo.structSize = sizeof(FfxVkUnifiedSrCreateInfo);
    createInfo.version = FFX_VK_UNIFIED_SR_VERSION;
    createInfo.preferredUpscaler = FFX_VK_UPSCALER_AUTO; /* Auto-detect hardware tier */
    createInfo.quality = FFX_VK_QUALITY_PRESET_QUALITY;
    createInfo.maxInputExtent = renderExtent;
    createInfo.maxOutputExtent = displayExtent;
    createInfo.device = VK_NULL_HANDLE; /* In headless verification or offline initialization */

    /* 3. Initialize Unified SR Context */
    FfxVkUnifiedSrContext srContext;
    VkResult res = ffxVkUnifiedSrCreate(&srContext, &createInfo);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to initialize Unified SR context (%d)\n", (int)res);
        return 1;
    }

    printf("[2] Active Upscaler: %s\n", ffxVkUnifiedSrGetUpscalerName(srContext.activeUpscaler));
    printf("[3] GPU Feature Tier: %s\n", ffxVkUnifiedSrGetGpuTierName(srContext.capabilities.tier));

    /* 4. Prepare Per-Frame Dispatch Descriptor */
    FfxVkUnifiedSrDispatchInfo dispatchInfo;
    memset(&dispatchInfo, 0, sizeof(dispatchInfo));
    dispatchInfo.structSize = sizeof(FfxVkUnifiedSrDispatchInfo);
    dispatchInfo.version = FFX_VK_UNIFIED_SR_VERSION;
    dispatchInfo.commandBuffer = (VkCommandBuffer)(uintptr_t)1;
    dispatchInfo.inputExtent = renderExtent;
    dispatchInfo.outputExtent = displayExtent;
    dispatchInfo.jitterOffsetX = 0.25f;
    dispatchInfo.jitterOffsetY = -0.125f;
    dispatchInfo.sharpness = 0.4f;
    dispatchInfo.frameTimeMs = 16.6f;
    dispatchInfo.resetHistory = false;

    /* Setup mock portable image descriptors */
    dispatchInfo.colorIn.structSize = sizeof(FfxVkPortableImage);
    dispatchInfo.colorIn.image = (VkImage)(uintptr_t)2;
    dispatchInfo.colorIn.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    dispatchInfo.colorIn.extent = (FfxVkPortableExtent2D){ renderExtent.width, renderExtent.height };
    dispatchInfo.colorIn.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    dispatchInfo.depth.structSize = sizeof(FfxVkPortableImage);
    dispatchInfo.depth.image = (VkImage)(uintptr_t)3;
    dispatchInfo.depth.format = VK_FORMAT_D32_SFLOAT;
    dispatchInfo.depth.extent = (FfxVkPortableExtent2D){ renderExtent.width, renderExtent.height };
    dispatchInfo.depth.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    dispatchInfo.motionVectors.structSize = sizeof(FfxVkPortableImage);
    dispatchInfo.motionVectors.image = (VkImage)(uintptr_t)4;
    dispatchInfo.motionVectors.format = VK_FORMAT_R16G16_SFLOAT;
    dispatchInfo.motionVectors.extent = (FfxVkPortableExtent2D){ renderExtent.width, renderExtent.height };
    dispatchInfo.motionVectors.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    dispatchInfo.colorOut.structSize = sizeof(FfxVkPortableImage);
    dispatchInfo.colorOut.image = (VkImage)(uintptr_t)5;
    dispatchInfo.colorOut.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    dispatchInfo.colorOut.extent = (FfxVkPortableExtent2D){ displayExtent.width, displayExtent.height };
    dispatchInfo.colorOut.usage = VK_IMAGE_USAGE_STORAGE_BIT;

    printf("[4] Dispatching frame upscaling (%ux%u -> %ux%u)...\n",
           dispatchInfo.inputExtent.width, dispatchInfo.inputExtent.height,
           dispatchInfo.outputExtent.width, dispatchInfo.outputExtent.height);

    res = ffxVkUnifiedSrDispatch(&srContext, &dispatchInfo);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Dispatch failed with result %d\n", (int)res);
        ffxVkUnifiedSrDestroy(&srContext);
        return 1;
    }
    printf("[5] Frame dispatch recorded successfully.\n");

    /* 5. Destroy Context */
    ffxVkUnifiedSrDestroy(&srContext);
    printf("[6] Unified SR context destroyed.\n\n");
    printf("Sample completed successfully.\n");
    return 0;
}
