/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_xess_contract.h"
#include <assert.h>
#include <math.h>
#include <string.h>

static FfxVkPortableImage make_test_image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage)
{
    FfxVkPortableImage img;
    memset(&img, 0, sizeof(img));
    img.structSize = sizeof(img);
    img.image = (VkImage)(uintptr_t)1;
    img.format = format;
    img.extent.width = width;
    img.extent.height = height;
    img.usage = usage;
    return img;
}

int main(void)
{
    uint64_t issues = 0;
    VkExtent2D display = { 1280, 720 };
    VkExtent2D input = { 0, 0 };

    /* 1. Test input resolution computation */
    assert(ffxVkXessGetInputResolution(FFX_VK_XESS_QUALITY_QUALITY, display, &input));
    assert(input.width == 754 && input.height == 424); /* Even extents */

    assert(ffxVkXessGetInputResolution(FFX_VK_XESS_QUALITY_BALANCED, display, &input));
    assert(input.width == 640 && input.height == 360);

    assert(ffxVkXessGetInputResolution(FFX_VK_XESS_QUALITY_PERFORMANCE, display, &input));
    assert(input.width == 558 && input.height == 314);

    /* 2. Test layer metadata and U-Net topology */
    assert(ffxVkXessGetTotalWeightBytes() == 253280);
    const FfxVkXessLayerMeta *m0 = ffxVkXessGetLayerMeta(FFX_VK_XESS_LAYER_I10);
    assert(m0 != NULL);
    assert(strcmp(m0->name, "XeSS_i10") == 0);
    assert(m0->dispatchId == 1);
    assert(m0->weightsBytes == 2304);

    const FfxVkXessLayerMeta *mResolve = ffxVkXessGetLayerMeta(FFX_VK_XESS_LAYER_I8);
    assert(mResolve != NULL);
    assert(strcmp(mResolve->name, "XeSS_i8") == 0);
    assert(mResolve->dispatchId == 13);
    assert(mResolve->weightsBytes == 320);

    /* 3. Test CreateInfo validation with DP4a support */
    FfxVkXessCreateInfo createInfo;
    memset(&createInfo, 0, sizeof(createInfo));
    createInfo.structSize = sizeof(FfxVkXessCreateInfo);
    createInfo.contractVersion = FFX_VK_XESS_CONTRACT_VERSION;
    createInfo.path = FFX_VK_XESS_PATH_DP4A;
    createInfo.quality = FFX_VK_XESS_QUALITY_QUALITY;
    createInfo.maxInputExtent = (VkExtent2D){ 754, 424 };
    createInfo.maxOutputExtent = (VkExtent2D){ 1280, 720 };

    issues = ffxVkXessValidateCreateInfo(&createInfo, true, true);
    assert(issues == FFX_VK_XESS_VALIDATION_NONE);

    /* Missing DP4a triggers validation error */
    issues = ffxVkXessValidateCreateInfo(&createInfo, false, true);
    assert(issues & FFX_VK_XESS_VALIDATION_MISSING_DP4A);

    /* Missing formatless write triggers validation error */
    issues = ffxVkXessValidateCreateInfo(&createInfo, true, false);
    assert(issues & FFX_VK_XESS_VALIDATION_MISSING_FORMATLESS_WRITE);

    /* Invalid struct size */
    createInfo.structSize = sizeof(FfxVkXessCreateInfo) - 4;
    issues = ffxVkXessValidateCreateInfo(&createInfo, true, true);
    assert(issues == FFX_VK_XESS_VALIDATION_STRUCT_SIZE);
    createInfo.structSize = sizeof(FfxVkXessCreateInfo);

    /* 4. Test DispatchInfo validation */
    FfxVkXessDispatchInfo dispatchInfo;
    memset(&dispatchInfo, 0, sizeof(dispatchInfo));
    dispatchInfo.structSize = sizeof(FfxVkXessDispatchInfo);
    dispatchInfo.contractVersion = FFX_VK_XESS_CONTRACT_VERSION;
    dispatchInfo.inputExtent = (VkExtent2D){ 754, 424 };
    dispatchInfo.outputExtent = (VkExtent2D){ 1280, 720 };
    dispatchInfo.colorIn = make_test_image(754, 424, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT);
    dispatchInfo.velocity = make_test_image(754, 424, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT);
    dispatchInfo.depth = make_test_image(754, 424, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT);
    dispatchInfo.colorOut = make_test_image(1280, 720, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
    dispatchInfo.jitterOffsetX = 0.25f;
    dispatchInfo.jitterOffsetY = -0.15f;
    dispatchInfo.sharpness = 0.5f;

    issues = ffxVkXessValidateDispatchInfo(&createInfo, &dispatchInfo);
    assert(issues == FFX_VK_XESS_VALIDATION_NONE);

    /* Test non-finite constants detection */
    dispatchInfo.jitterOffsetX = NAN;
    issues = ffxVkXessValidateDispatchInfo(&createInfo, &dispatchInfo);
    assert(issues & FFX_VK_XESS_VALIDATION_NONFINITE_CONSTANTS);
    dispatchInfo.jitterOffsetX = 0.25f;

    /* Test extent overflow detection */
    dispatchInfo.inputExtent = (VkExtent2D){ 800, 424 };
    issues = ffxVkXessValidateDispatchInfo(&createInfo, &dispatchInfo);
    assert(issues & FFX_VK_XESS_VALIDATION_EXTENT_EXCEEDS_MAX);

    return 0;
}
