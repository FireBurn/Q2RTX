/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_xess_contract.h"

#include <math.h>
#include <string.h>

static const FfxVkXessLayerMeta s_xessLayers[FFX_VK_XESS_LAYER_COUNT] = {
    { "XeSS_i10", 1,  2304,  64,  64 },
    { "XeSS_i12", 2,  4608, 128, 128 },
    { "XeSS_i13", 3, 18432, 256, 256 },
    { "XeSS_i14", 4, 73728, 512, 512 },
    { "XeSS_i15", 5, 73728, 256, 256 },
    { "XeSS_i16", 6, 36864, 256, 256 },
    { "XeSS_i17", 7, 18432, 128, 128 },
    { "XeSS_i18", 8,  9216, 128, 128 },
    { "XeSS_i19", 9,  4608,  64,  64 },
    { "XeSS_i20", 10, 2304,  64,  64 },
    { "XeSS_i21", 11, 2304,  64,  64 },
    { "XeSS_i23", 12, 2304,  64,  64 },
    { "XeSS_i8",  13,  320,  80,  80 },
};

const FfxVkXessLayerMeta *ffxVkXessGetLayerMeta(FfxVkXessLayerId layerId)
{
    if ((uint32_t)layerId >= FFX_VK_XESS_LAYER_COUNT)
        return NULL;
    return &s_xessLayers[layerId];
}

uint32_t ffxVkXessGetTotalWeightBytes(void)
{
    uint32_t total = 0;
    for (uint32_t i = 0; i < FFX_VK_XESS_LAYER_COUNT; ++i) {
        total += s_xessLayers[i].weightsBytes +
                 s_xessLayers[i].scaleBytes +
                 s_xessLayers[i].biasBytes;
    }
    return total; /* 253,280 bytes */
}

bool ffxVkXessGetInputResolution(
    FfxVkXessQuality quality,
    VkExtent2D outputExtent,
    VkExtent2D *outInputExtent)
{
    if (!outInputExtent || !outputExtent.width || !outputExtent.height)
        return false;

    float scale = 1.7f; /* Quality default */
    switch (quality) {
    case FFX_VK_XESS_QUALITY_ULTRA_QUALITY_PLUS:
        scale = 1.3f;
        break;
    case FFX_VK_XESS_QUALITY_ULTRA_QUALITY:
        scale = 1.5f;
        break;
    case FFX_VK_XESS_QUALITY_QUALITY:
        scale = 1.7f;
        break;
    case FFX_VK_XESS_QUALITY_BALANCED:
        scale = 2.0f;
        break;
    case FFX_VK_XESS_QUALITY_PERFORMANCE:
        scale = 2.3f;
        break;
    case FFX_VK_XESS_QUALITY_ULTRA_PERFORMANCE:
        scale = 3.0f;
        break;
    default:
        return false;
    }

    outInputExtent->width = (uint32_t)ceilf((float)outputExtent.width / scale);
    outInputExtent->height = (uint32_t)ceilf((float)outputExtent.height / scale);

    /* Guarantee even extents for downsampled/upsampled convolutional pipelines */
    if (outInputExtent->width & 1u)
        outInputExtent->width += 1u;
    if (outInputExtent->height & 1u)
        outInputExtent->height += 1u;

    return true;
}

static bool ffx_vk_xess_validate_image(const FfxVkPortableImage *img, bool isOutput)
{
    if (!img || img->image == VK_NULL_HANDLE)
        return false;
    if (!img->extent.width || !img->extent.height)
        return false;
    if (img->format == VK_FORMAT_UNDEFINED)
        return false;

    if (isOutput) {
        if (!(img->usage & (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)))
            return false;
    } else {
        if (!(img->usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT)))
            return false;
    }
    return true;
}

uint64_t ffxVkXessValidateCreateInfo(
    const FfxVkXessCreateInfo *info,
    bool supportsDot4,
    bool supportsFormatlessWrite)
{
    uint64_t issues = FFX_VK_XESS_VALIDATION_NONE;

    if (!info || info->structSize != sizeof(FfxVkXessCreateInfo))
        return FFX_VK_XESS_VALIDATION_STRUCT_SIZE;

    if (info->contractVersion != FFX_VK_XESS_CONTRACT_VERSION)
        issues |= FFX_VK_XESS_VALIDATION_CONTRACT_VERSION;

    if (!info->maxInputExtent.width || !info->maxInputExtent.height ||
        !info->maxOutputExtent.width || !info->maxOutputExtent.height)
        issues |= FFX_VK_XESS_VALIDATION_ZERO_EXTENT;

    if (info->maxInputExtent.width > info->maxOutputExtent.width ||
        info->maxInputExtent.height > info->maxOutputExtent.height)
        issues |= FFX_VK_XESS_VALIDATION_EXTENT_EXCEEDS_MAX;

    /* XeSS DP4a execution path requires integer dot product and formatless storage write */
    if (info->path == FFX_VK_XESS_PATH_AUTO || info->path == FFX_VK_XESS_PATH_DP4A) {
        if (!supportsDot4)
            issues |= FFX_VK_XESS_VALIDATION_MISSING_DP4A;
        if (!supportsFormatlessWrite)
            issues |= FFX_VK_XESS_VALIDATION_MISSING_FORMATLESS_WRITE;
    }

    /* Validate model container if provided */
    if (info->modelContainerData) {
        const uint8_t *bytes = (const uint8_t *)info->modelContainerData;
        if (info->modelContainerSizeBytes < 24) {
            issues |= FFX_VK_XESS_VALIDATION_MODEL_CONTAINER_INVALID;
        } else if (memcmp(bytes, "XESSMOD2", 8) != 0) {
            issues |= FFX_VK_XESS_VALIDATION_MODEL_CONTAINER_INVALID;
        } else {
            uint32_t version = 0, numLayers = 0;
            memcpy(&version, bytes + 8, sizeof(version));
            memcpy(&numLayers, bytes + 12, sizeof(numLayers));
            if (version != 2 || numLayers != FFX_VK_XESS_LAYER_COUNT) {
                issues |= FFX_VK_XESS_VALIDATION_MODEL_CONTAINER_INVALID;
            } else if (info->modelContainerSizeBytes < (size_t)ffxVkXessGetTotalWeightBytes()) {
                issues |= FFX_VK_XESS_VALIDATION_MODEL_WEIGHTS_TRUNCATED;
            }
        }
    }

    return issues;
}

uint64_t ffxVkXessValidateDispatchInfo(
    const FfxVkXessCreateInfo *createInfo,
    const FfxVkXessDispatchInfo *dispatchInfo)
{
    uint64_t issues = FFX_VK_XESS_VALIDATION_NONE;

    if (!createInfo || createInfo->structSize != sizeof(FfxVkXessCreateInfo))
        return FFX_VK_XESS_VALIDATION_STRUCT_SIZE;

    if (!dispatchInfo || dispatchInfo->structSize != sizeof(FfxVkXessDispatchInfo))
        return FFX_VK_XESS_VALIDATION_STRUCT_SIZE;

    if (dispatchInfo->contractVersion != FFX_VK_XESS_CONTRACT_VERSION)
        issues |= FFX_VK_XESS_VALIDATION_CONTRACT_VERSION;

    if (!dispatchInfo->inputExtent.width || !dispatchInfo->inputExtent.height ||
        !dispatchInfo->outputExtent.width || !dispatchInfo->outputExtent.height) {
        issues |= FFX_VK_XESS_VALIDATION_ZERO_EXTENT;
    } else {
        if (dispatchInfo->inputExtent.width > createInfo->maxInputExtent.width ||
            dispatchInfo->inputExtent.height > createInfo->maxInputExtent.height ||
            dispatchInfo->outputExtent.width > createInfo->maxOutputExtent.width ||
            dispatchInfo->outputExtent.height > createInfo->maxOutputExtent.height)
            issues |= FFX_VK_XESS_VALIDATION_EXTENT_EXCEEDS_MAX;
    }

    if (!ffx_vk_xess_validate_image(&dispatchInfo->colorIn, false))
        issues |= FFX_VK_XESS_VALIDATION_IMAGE_HANDLE;
    if (!ffx_vk_xess_validate_image(&dispatchInfo->velocity, false))
        issues |= FFX_VK_XESS_VALIDATION_IMAGE_HANDLE;
    if (!ffx_vk_xess_validate_image(&dispatchInfo->depth, false))
        issues |= FFX_VK_XESS_VALIDATION_IMAGE_HANDLE;
    if (!ffx_vk_xess_validate_image(&dispatchInfo->colorOut, true))
        issues |= FFX_VK_XESS_VALIDATION_IMAGE_HANDLE;

    if (!isfinite(dispatchInfo->jitterOffsetX) ||
        !isfinite(dispatchInfo->jitterOffsetY) ||
        !isfinite(dispatchInfo->sharpness))
        issues |= FFX_VK_XESS_VALIDATION_NONFINITE_CONSTANTS;

    return issues;
}
