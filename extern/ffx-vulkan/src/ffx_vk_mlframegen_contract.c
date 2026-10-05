/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_mlframegen_contract.h"
#include <math.h>
#include <string.h>

float ffxVkMlFrameGenGetInterpolationFactor(FfxVkMlFrameGenMode mode, uint32_t slotIndex)
{
    switch (mode) {
    case FFX_VK_MLFRAMEGEN_MODE_2X:
        return 0.5f;
    case FFX_VK_MLFRAMEGEN_MODE_3X:
        if (slotIndex == 0) return 1.0f / 3.0f;
        if (slotIndex == 1) return 2.0f / 3.0f;
        return 0.5f;
    case FFX_VK_MLFRAMEGEN_MODE_4X:
        if (slotIndex == 0) return 0.25f;
        if (slotIndex == 1) return 0.50f;
        if (slotIndex == 2) return 0.75f;
        return 0.5f;
    default:
        return 0.5f;
    }
}

static bool ffx_vk_mlfg_validate_image(const FfxVkPortableImage *img, bool isOutput)
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
        if (!(img->usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)))
            return false;
    }
    return true;
}

uint64_t ffxVkMlFrameGenValidateCreateInfo(
    const FfxVkMlFrameGenCreateInfo *info,
    bool supportsInt8,
    bool supportsWmma)
{
    uint64_t issues = FFX_VK_MLFG_VALIDATION_NONE;

    if (!info || info->structSize != sizeof(FfxVkMlFrameGenCreateInfo))
        return FFX_VK_MLFG_VALIDATION_STRUCT_SIZE;

    if (info->contractVersion != FFX_VK_MLFRAMEGEN_CONTRACT_VERSION)
        issues |= FFX_VK_MLFG_VALIDATION_CONTRACT_VERSION;

    if (!info->displaySize.width || !info->displaySize.height)
        issues |= FFX_VK_MLFG_VALIDATION_ZERO_EXTENT;

    if (info->tier == FFX_VK_MLFRAMEGEN_TIER_INT8_DOT4 && !supportsInt8)
        issues |= FFX_VK_MLFG_VALIDATION_MISSING_INT8;

    if (info->tier == FFX_VK_MLFRAMEGEN_TIER_FP8_WMMA && !supportsWmma)
        issues |= FFX_VK_MLFG_VALIDATION_MISSING_WMMA;

    return issues;
}

uint64_t ffxVkMlFrameGenValidateDispatchInfo(
    const FfxVkMlFrameGenCreateInfo *createInfo,
    const FfxVkMlFrameGenDispatchInfo *dispatchInfo)
{
    uint64_t issues = FFX_VK_MLFG_VALIDATION_NONE;

    if (!createInfo || !dispatchInfo || dispatchInfo->structSize != sizeof(FfxVkMlFrameGenDispatchInfo))
        return FFX_VK_MLFG_VALIDATION_STRUCT_SIZE;

    if (dispatchInfo->contractVersion != FFX_VK_MLFRAMEGEN_CONTRACT_VERSION)
        issues |= FFX_VK_MLFG_VALIDATION_CONTRACT_VERSION;

    if (!dispatchInfo->commandBuffer)
        issues |= FFX_VK_MLFG_VALIDATION_IMAGE_HANDLE;

    /* Validate interpolation factor (must be finite and strictly between 0 and 1) */
    if (!isfinite(dispatchInfo->interpolationFactor) ||
        dispatchInfo->interpolationFactor <= 0.0f ||
        dispatchInfo->interpolationFactor >= 1.0f) {
        issues |= FFX_VK_MLFG_VALIDATION_INVALID_FACTOR;
    }

    /* Validate current and previous real images */
    if (!ffx_vk_mlfg_validate_image(&dispatchInfo->currentRealColor, false) ||
        !ffx_vk_mlfg_validate_image(&dispatchInfo->previousRealColor, false) ||
        !ffx_vk_mlfg_validate_image(&dispatchInfo->motionVectors, false) ||
        !ffx_vk_mlfg_validate_image(&dispatchInfo->depth, false) ||
        !ffx_vk_mlfg_validate_image(&dispatchInfo->outputGeneratedColor, true)) {
        issues |= FFX_VK_MLFG_VALIDATION_IMAGE_HANDLE;
    }

    /* Validate extent matches displaySize */
    if (dispatchInfo->outputGeneratedColor.extent.width != createInfo->displaySize.width ||
        dispatchInfo->outputGeneratedColor.extent.height != createInfo->displaySize.height) {
        issues |= FFX_VK_MLFG_VALIDATION_EXTENT_MISMATCH;
    }

    /* Disallow output alias with input frames */
    if (dispatchInfo->outputGeneratedColor.image != VK_NULL_HANDLE) {
        if (dispatchInfo->outputGeneratedColor.image == dispatchInfo->currentRealColor.image ||
            dispatchInfo->outputGeneratedColor.image == dispatchInfo->previousRealColor.image ||
            dispatchInfo->outputGeneratedColor.image == dispatchInfo->motionVectors.image ||
            dispatchInfo->outputGeneratedColor.image == dispatchInfo->depth.image) {
            issues |= FFX_VK_MLFG_VALIDATION_ALIAS_CONFLICT;
        }
    }

    /* Validate camera metadata */
    if (!isfinite(dispatchInfo->jitterOffset.x) || !isfinite(dispatchInfo->jitterOffset.y) ||
        !isfinite(dispatchInfo->cameraPositionDelta.x) || !isfinite(dispatchInfo->cameraPositionDelta.y) ||
        !isfinite(dispatchInfo->cameraPositionDelta.z) || !isfinite(dispatchInfo->frameTimeMs)) {
        issues |= FFX_VK_MLFG_VALIDATION_NONFINITE_METADATA;
    }

    return issues;
}
