/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_mlframegen_contract.h"
#include <assert.h>
#include <math.h>
#include <string.h>

static FfxVkPortableImage make_test_image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, uint64_t handle)
{
    FfxVkPortableImage img;
    memset(&img, 0, sizeof(img));
    img.structSize = sizeof(img);
    img.image = (VkImage)(uintptr_t)handle;
    img.format = format;
    img.extent.width = width;
    img.extent.height = height;
    img.usage = usage;
    return img;
}

int main(void)
{
    /* 1. Test interpolation factors */
    assert(fabsf(ffxVkMlFrameGenGetInterpolationFactor(FFX_VK_MLFRAMEGEN_MODE_2X, 0) - 0.5f) < 1e-5f);
    assert(fabsf(ffxVkMlFrameGenGetInterpolationFactor(FFX_VK_MLFRAMEGEN_MODE_3X, 0) - 0.333333f) < 1e-4f);
    assert(fabsf(ffxVkMlFrameGenGetInterpolationFactor(FFX_VK_MLFRAMEGEN_MODE_3X, 1) - 0.666667f) < 1e-4f);
    assert(fabsf(ffxVkMlFrameGenGetInterpolationFactor(FFX_VK_MLFRAMEGEN_MODE_4X, 0) - 0.25f) < 1e-5f);
    assert(fabsf(ffxVkMlFrameGenGetInterpolationFactor(FFX_VK_MLFRAMEGEN_MODE_4X, 1) - 0.50f) < 1e-5f);
    assert(fabsf(ffxVkMlFrameGenGetInterpolationFactor(FFX_VK_MLFRAMEGEN_MODE_4X, 2) - 0.75f) < 1e-5f);

    /* 2. Test CreateInfo validation */
    FfxVkMlFrameGenCreateInfo createInfo;
    memset(&createInfo, 0, sizeof(createInfo));
    createInfo.structSize = sizeof(FfxVkMlFrameGenCreateInfo);
    createInfo.contractVersion = FFX_VK_MLFRAMEGEN_CONTRACT_VERSION;
    createInfo.mode = FFX_VK_MLFRAMEGEN_MODE_2X;
    createInfo.tier = FFX_VK_MLFRAMEGEN_TIER_INT8_DOT4;
    createInfo.displaySize.width = 1920;
    createInfo.displaySize.height = 1080;

    /* Without INT8 support, tier INT8 fails */
    uint64_t issues = ffxVkMlFrameGenValidateCreateInfo(&createInfo, false, false);
    assert(issues & FFX_VK_MLFG_VALIDATION_MISSING_INT8);

    /* With INT8 support, tier INT8 passes */
    issues = ffxVkMlFrameGenValidateCreateInfo(&createInfo, true, false);
    assert(issues == FFX_VK_MLFG_VALIDATION_NONE);

    /* Test WMMA requirement */
    createInfo.tier = FFX_VK_MLFRAMEGEN_TIER_FP8_WMMA;
    issues = ffxVkMlFrameGenValidateCreateInfo(&createInfo, true, false);
    assert(issues & FFX_VK_MLFG_VALIDATION_MISSING_WMMA);
    issues = ffxVkMlFrameGenValidateCreateInfo(&createInfo, true, true);
    assert(issues == FFX_VK_MLFG_VALIDATION_NONE);

    /* 3. Test DispatchInfo validation */
    FfxVkMlFrameGenDispatchInfo dispatch;
    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.structSize = sizeof(FfxVkMlFrameGenDispatchInfo);
    dispatch.contractVersion = FFX_VK_MLFRAMEGEN_CONTRACT_VERSION;
    dispatch.commandBuffer = (VkCommandBuffer)(uintptr_t)1;
    dispatch.interpolationFactor = 0.5f;
    dispatch.frameTimeMs = 16.6f;

    dispatch.currentRealColor = make_test_image(1920, 1080, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, 10);
    dispatch.previousRealColor = make_test_image(1920, 1080, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, 11);
    dispatch.motionVectors = make_test_image(1920, 1080, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, 12);
    dispatch.depth = make_test_image(1920, 1080, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, 13);
    dispatch.outputGeneratedColor = make_test_image(1920, 1080, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT, 14);

    issues = ffxVkMlFrameGenValidateDispatchInfo(&createInfo, &dispatch);
    assert(issues == FFX_VK_MLFG_VALIDATION_NONE);

    /* Check invalid interpolation factor */
    dispatch.interpolationFactor = 1.0f;
    assert(ffxVkMlFrameGenValidateDispatchInfo(&createInfo, &dispatch) & FFX_VK_MLFG_VALIDATION_INVALID_FACTOR);
    dispatch.interpolationFactor = 0.0f;
    assert(ffxVkMlFrameGenValidateDispatchInfo(&createInfo, &dispatch) & FFX_VK_MLFG_VALIDATION_INVALID_FACTOR);
    dispatch.interpolationFactor = 0.5f;

    /* Check alias conflict (output == currentRealColor) */
    dispatch.outputGeneratedColor.image = dispatch.currentRealColor.image;
    assert(ffxVkMlFrameGenValidateDispatchInfo(&createInfo, &dispatch) & FFX_VK_MLFG_VALIDATION_ALIAS_CONFLICT);
    dispatch.outputGeneratedColor.image = (VkImage)(uintptr_t)14;

    /* Check extent mismatch */
    dispatch.outputGeneratedColor.extent.width = 1280;
    assert(ffxVkMlFrameGenValidateDispatchInfo(&createInfo, &dispatch) & FFX_VK_MLFG_VALIDATION_EXTENT_MISMATCH);

    return 0;
}
