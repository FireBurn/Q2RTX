/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_dlss_contract.h"
#include <math.h>
#include <string.h>

static int is_valid_float(float v) {
    return isfinite(v) != 0;
}

static int is_valid_positive_float(float v) {
    return isfinite(v) && v > 0.0f;
}

static int is_valid_color_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return 1;
    default:
        return 0;
    }
}

static int is_valid_depth_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_R32_SFLOAT:
        return 1;
    default:
        return 0;
    }
}

static int is_valid_motion_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 1;
    default:
        return 0;
    }
}

FfxVkPortableResult ffxVkDlssValidateCreateInfo(
    const FfxVkDlssCreateInfo* createInfo, uint64_t* issues)
{
    uint64_t accumulated = FFX_VK_DLSS_VALIDATION_NONE;

    if (!issues) {
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    }
    *issues = FFX_VK_DLSS_VALIDATION_NONE;

    if (!createInfo) {
        *issues = FFX_VK_DLSS_VALIDATION_STRUCT_SIZE;
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    }

    if (createInfo->structSize != sizeof(FfxVkDlssCreateInfo)) {
        accumulated |= FFX_VK_DLSS_VALIDATION_STRUCT_SIZE;
    }

    if (createInfo->contractVersion != FFX_VK_DLSS_CONTRACT_VERSION) {
        accumulated |= FFX_VK_DLSS_VALIDATION_CONTRACT_VERSION;
    }

    if (createInfo->model > FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING) {
        accumulated |= FFX_VK_DLSS_VALIDATION_UNSUPPORTED_MODEL;
    }

    if (createInfo->gpuArch == FFX_VK_DLSS_ARCH_UNKNOWN ||
        createInfo->gpuArch > FFX_VK_DLSS_ARCH_GENERIC_VULKAN) {
        accumulated |= FFX_VK_DLSS_VALIDATION_UNSUPPORTED_ARCH;
    }

    /* Native FP8 requires RDNA4 or NVIDIA Tensor (sm_120) or Generic Vulkan */
    if ((createInfo->flags & FFX_VK_DLSS_FLAG_NATIVE_FP8) &&
        createInfo->gpuArch != FFX_VK_DLSS_ARCH_RDNA4 &&
        createInfo->gpuArch != FFX_VK_DLSS_ARCH_NVIDIA_TENSOR &&
        createInfo->gpuArch != FFX_VK_DLSS_ARCH_GENERIC_VULKAN) {
        accumulated |= FFX_VK_DLSS_VALIDATION_UNSUPPORTED_ARCH;
    }

    if (createInfo->maxRenderSize.width == 0 || createInfo->maxRenderSize.height == 0 ||
        createInfo->displaySize.width == 0 || createInfo->displaySize.height == 0) {
        accumulated |= FFX_VK_DLSS_VALIDATION_ZERO_EXTENT;
    }

    *issues = accumulated;
    return (accumulated == FFX_VK_DLSS_VALIDATION_NONE)
        ? FFX_VK_PORTABLE_OK
        : FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
}

FfxVkPortableResult ffxVkDlssValidateDispatchInfo(
    const FfxVkDlssCreateInfo* createInfo,
    const FfxVkDlssDispatchInfo* dispatchInfo, uint64_t* issues)
{
    uint64_t accumulated = FFX_VK_DLSS_VALIDATION_NONE;

    if (!issues) {
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    }
    *issues = FFX_VK_DLSS_VALIDATION_NONE;

    if (!createInfo || !dispatchInfo) {
        *issues = FFX_VK_DLSS_VALIDATION_STRUCT_SIZE;
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    }

    if (dispatchInfo->structSize != sizeof(FfxVkDlssDispatchInfo)) {
        accumulated |= FFX_VK_DLSS_VALIDATION_STRUCT_SIZE;
    }

    if (dispatchInfo->contractVersion != FFX_VK_DLSS_CONTRACT_VERSION) {
        accumulated |= FFX_VK_DLSS_VALIDATION_CONTRACT_VERSION;
    }

    if (dispatchInfo->renderSize.width == 0 || dispatchInfo->renderSize.height == 0) {
        accumulated |= FFX_VK_DLSS_VALIDATION_ZERO_EXTENT;
    }

    if (dispatchInfo->renderSize.width > createInfo->maxRenderSize.width ||
        dispatchInfo->renderSize.height > createInfo->maxRenderSize.height) {
        accumulated |= FFX_VK_DLSS_VALIDATION_EXTENT_EXCEEDS_MAX;
    }

    /* Color image check */
    if (dispatchInfo->color.image == VK_NULL_HANDLE) {
        accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_HANDLE;
    } else {
        if (dispatchInfo->color.extent.width < dispatchInfo->renderSize.width ||
            dispatchInfo->color.extent.height < dispatchInfo->renderSize.height) {
            accumulated |= FFX_VK_DLSS_VALIDATION_ZERO_EXTENT;
        }
        if (!is_valid_color_format(dispatchInfo->color.format)) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_FORMAT;
        }
        if (!(dispatchInfo->color.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT))) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_USAGE;
        }
    }

    /* Depth image check */
    if (dispatchInfo->depth.image == VK_NULL_HANDLE) {
        accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_HANDLE;
    } else {
        if (dispatchInfo->depth.extent.width < dispatchInfo->renderSize.width ||
            dispatchInfo->depth.extent.height < dispatchInfo->renderSize.height) {
            accumulated |= FFX_VK_DLSS_VALIDATION_ZERO_EXTENT;
        }
        if (!is_valid_depth_format(dispatchInfo->depth.format)) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_FORMAT;
        }
    }

    /* Motion vectors check */
    if (dispatchInfo->motionVectors.image == VK_NULL_HANDLE) {
        accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_HANDLE;
    } else {
        if (dispatchInfo->motionVectors.extent.width < dispatchInfo->renderSize.width ||
            dispatchInfo->motionVectors.extent.height < dispatchInfo->renderSize.height) {
            accumulated |= FFX_VK_DLSS_VALIDATION_ZERO_EXTENT;
        }
        if (!is_valid_motion_format(dispatchInfo->motionVectors.format)) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_FORMAT;
        }
    }

    /* Output image check */
    if (dispatchInfo->output.image == VK_NULL_HANDLE) {
        accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_HANDLE;
    } else {
        if (dispatchInfo->output.extent.width < createInfo->displaySize.width ||
            dispatchInfo->output.extent.height < createInfo->displaySize.height) {
            accumulated |= FFX_VK_DLSS_VALIDATION_ZERO_EXTENT;
        }
        if (!is_valid_color_format(dispatchInfo->output.format)) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_FORMAT;
        }
        if (!(dispatchInfo->output.usage & (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_USAGE;
        }
    }

    /* Optional exposure check */
    if (!(createInfo->flags & FFX_VK_DLSS_FLAG_AUTO_EXPOSURE) &&
        !(dispatchInfo->flags & FFX_VK_DLSS_FLAG_AUTO_EXPOSURE)) {
        if (dispatchInfo->exposure.image == VK_NULL_HANDLE) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_HANDLE;
        }
    }

    /* Camera and metadata numerical validation */
    if (!is_valid_float(dispatchInfo->jitterOffset.x) ||
        !is_valid_float(dispatchInfo->jitterOffset.y) ||
        !is_valid_float(dispatchInfo->cameraPositionDelta.x) ||
        !is_valid_float(dispatchInfo->cameraPositionDelta.y) ||
        !is_valid_float(dispatchInfo->cameraPositionDelta.z)) {
        accumulated |= FFX_VK_DLSS_VALIDATION_NONFINITE_VALUE;
    }

    if (!is_valid_positive_float(dispatchInfo->motionVectorScale.x) ||
        !is_valid_positive_float(dispatchInfo->motionVectorScale.y) ||
        !is_valid_positive_float(dispatchInfo->verticalFov) ||
        !is_valid_positive_float(dispatchInfo->nearZ) ||
        !is_valid_positive_float(dispatchInfo->farZ) ||
        !is_valid_positive_float(dispatchInfo->preExposure) ||
        dispatchInfo->nearZ == dispatchInfo->farZ) {
        accumulated |= FFX_VK_DLSS_VALIDATION_CAMERA_METADATA;
    }

    /* VRAM interop handle validation (if enabled) */
    if ((createInfo->flags & FFX_VK_DLSS_FLAG_VRAM_INTEROP) ||
        (dispatchInfo->flags & FFX_VK_DLSS_FLAG_VRAM_INTEROP)) {
        if (dispatchInfo->colorExportHandle) {
            if (dispatchInfo->colorExportHandle->memory == VK_NULL_HANDLE ||
                dispatchInfo->colorExportHandle->size == 0 ||
                (dispatchInfo->colorExportHandle->fd < 0 &&
                 dispatchInfo->colorExportHandle->win32Handle == NULL)) {
                accumulated |= FFX_VK_DLSS_VALIDATION_VRAM_INTEROP_HANDLE;
            }
        }
        if (dispatchInfo->outputExportHandle) {
            if (dispatchInfo->outputExportHandle->memory == VK_NULL_HANDLE ||
                dispatchInfo->outputExportHandle->size == 0 ||
                (dispatchInfo->outputExportHandle->fd < 0 &&
                 dispatchInfo->outputExportHandle->win32Handle == NULL)) {
                accumulated |= FFX_VK_DLSS_VALIDATION_VRAM_INTEROP_HANDLE;
            }
        }
    }

    /* DLSS 5 Neural Rendering validation */
    if (createInfo->model == FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING) {
        /* G-buffer auxiliary inputs */
        if (dispatchInfo->normalsRoughnessMaterial.image == VK_NULL_HANDLE ||
            dispatchInfo->diffuseAlbedo.image == VK_NULL_HANDLE ||
            dispatchInfo->specularAlbedo.image == VK_NULL_HANDLE) {
            accumulated |= FFX_VK_DLSS_VALIDATION_NEURAL_RENDERING_MISSING_SIGNAL;
        }

        /* At least one direct or indirect radiance partition must be present */
        const int hasRadiance =
            (dispatchInfo->directDiffuse.image != VK_NULL_HANDLE) ||
            (dispatchInfo->directSpecular.image != VK_NULL_HANDLE) ||
            (dispatchInfo->indirectDiffuse.image != VK_NULL_HANDLE) ||
            (dispatchInfo->indirectSpecular.image != VK_NULL_HANDLE);
        if (!hasRadiance) {
            accumulated |= FFX_VK_DLSS_VALIDATION_NEURAL_RENDERING_MISSING_SIGNAL;
        }

        /* Indirect radiance partitions require RGBA16F (alpha = first hit distance) */
        if (dispatchInfo->indirectDiffuse.image != VK_NULL_HANDLE &&
            dispatchInfo->indirectDiffuse.format != VK_FORMAT_R16G16B16A16_SFLOAT) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_FORMAT;
        }
        if (dispatchInfo->indirectSpecular.image != VK_NULL_HANDLE &&
            dispatchInfo->indirectSpecular.format != VK_FORMAT_R16G16B16A16_SFLOAT) {
            accumulated |= FFX_VK_DLSS_VALIDATION_IMAGE_FORMAT;
        }

        /* Tensor weights buffer: required for DLSS 5 */
        if (dispatchInfo->weightsTensorBuffer.buffer == VK_NULL_HANDLE ||
            dispatchInfo->weightsTensorBuffer.size < (1024ull * 1024ull)) {
            accumulated |= FFX_VK_DLSS_VALIDATION_NEURAL_RENDERING_WEIGHTS;
        }
    }

    *issues = accumulated;
    return (accumulated == FFX_VK_DLSS_VALIDATION_NONE)
        ? FFX_VK_PORTABLE_OK
        : FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
}

FfxVkPortableResult ffxVkDlssGetOptimalRenderResolution(
    FfxVkPortableExtent2D displaySize,
    FfxVkDlssPreset preset,
    FfxVkPortableExtent2D* outRenderSize)
{
    float scale = 1.0f;

    if (!outRenderSize) {
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    }
    if (displaySize.width == 0 || displaySize.height == 0) {
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    }

    switch (preset) {
    case FFX_VK_DLSS_PRESET_ULTRA_PERFORMANCE:
        scale = 3.0f;
        break;
    case FFX_VK_DLSS_PRESET_PERFORMANCE:
        scale = 2.0f;
        break;
    case FFX_VK_DLSS_PRESET_BALANCED:
        scale = 1.7f;
        break;
    case FFX_VK_DLSS_PRESET_QUALITY:
        scale = 1.5f;
        break;
    case FFX_VK_DLSS_PRESET_DLAA:
        scale = 1.0f;
        break;
    default:
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    }

    outRenderSize->width = (uint32_t)ceilf((float)displaySize.width / scale);
    outRenderSize->height = (uint32_t)ceilf((float)displaySize.height / scale);

    if (outRenderSize->width == 0) outRenderSize->width = 1;
    if (outRenderSize->height == 0) outRenderSize->height = 1;

    return FFX_VK_PORTABLE_OK;
}

const char* ffxVkDlssGetModelName(FfxVkDlssModel model) {
    switch (model) {
    case FFX_VK_DLSS_MODEL_3_CNN_E:
        return "DLSS 3 CNN (Model E)";
    case FFX_VK_DLSS_MODEL_4_SWIN_K:
        return "DLSS 4 Swin Transformer (Model K)";
    case FFX_VK_DLSS_MODEL_4_5_TRANSFORMER_M:
        return "DLSS 4.5 Transformer (Model M)";
    case FFX_VK_DLSS_MODEL_4_5_TRANSFORMER_L:
        return "DLSS 4.5 Ultra Performance (Model L)";
    case FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING:
        return "DLSS 5 Neural Rendering (DLSSNR)";
    default:
        return "DLSS Unknown Model";
    }
}

const char* ffxVkDlssGetArchName(FfxVkDlssGpuArch arch) {
    switch (arch) {
    case FFX_VK_DLSS_ARCH_NVIDIA_TENSOR:
        return "NVIDIA Tensor Cores (sm_80/89/120)";
    case FFX_VK_DLSS_ARCH_RDNA3:
        return "AMD RDNA3 (gfx110x WMMA FP16)";
    case FFX_VK_DLSS_ARCH_RDNA4:
        return "AMD RDNA4 (gfx120x Native FP8 WMMA)";
    case FFX_VK_DLSS_ARCH_GENERIC_VULKAN:
        return "Vulkan Cooperative Matrix";
    default:
        return "Unknown Architecture";
    }
}
