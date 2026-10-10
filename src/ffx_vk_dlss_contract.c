/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_dlss_contract.h"
#include "d4r_swin_reconstruct_spv.h"
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

    /* Validate model container if provided */
    if (createInfo->modelContainerData) {
        const uint8_t* bytes = (const uint8_t*)createInfo->modelContainerData;
        if (createInfo->modelContainerSizeBytes < 24) {
            accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_CONTAINER_INVALID;
        } else if (memcmp(bytes, "DLSSMOD1", 8) == 0) {
            uint32_t version = 0, modelFamily = 0;
            memcpy(&version, bytes + 8, sizeof(version));
            memcpy(&modelFamily, bytes + 12, sizeof(modelFamily));
            if (version != 1 || modelFamily > FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING) {
                accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_CONTAINER_INVALID;
            } else if (createInfo->model == FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING) {
                accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_CONTAINER_INVALID;
            } else if (createInfo->modelContainerSizeBytes < 65536) {
                accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_WEIGHTS_TRUNCATED;
            }
        } else if (memcmp(bytes, "DLSSNR1\0", 8) == 0 || memcmp(bytes, "DLSSNR1", 7) == 0) {
            uint32_t version = 0;
            memcpy(&version, bytes + 8, sizeof(version));
            if (version != 1) {
                accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_CONTAINER_INVALID;
            } else if (createInfo->model != FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING) {
                accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_CONTAINER_INVALID;
            } else if (createInfo->modelContainerSizeBytes < 1048576) {
                accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_WEIGHTS_TRUNCATED;
            }
        } else {
            accumulated |= FFX_VK_DLSS_VALIDATION_MODEL_CONTAINER_INVALID;
        }
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

typedef struct FfxVkDlssPushConstants {
    uint32_t inputExtent[2];
    uint32_t outputExtent[2];
    float jitterOffset[2];
    float motionVectorScale[2];
    float sharpness;
    float preExposure;
    uint32_t resetHistory;
    uint32_t flags;
} FfxVkDlssPushConstants;

const uint32_t* ffxVkDlssGetReconstructSpirv(size_t* outWordCount)
{
    if (outWordCount)
        *outWordCount = g_d4r_swin_reconstruct_spv_word_count;
    return g_d4r_swin_reconstruct_spv;
}

FfxVkPortableResult ffxVkDlssCreatePipeline(
    VkDevice device,
    FfxVkDlssPipeline* outPipeline)
{
    if (device == VK_NULL_HANDLE || !outPipeline)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    memset(outPipeline, 0, sizeof(*outPipeline));
    outPipeline->device = device;

    /* 1. Linear clamp sampler */
    VkSamplerCreateInfo samplerInfo = {0};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 1.0f;
    if (vkCreateSampler(device, &samplerInfo, NULL, &outPipeline->linearSampler) != VK_SUCCESS) {
        ffxVkDlssDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 2. Descriptor set layout: bindings 0,1,2 = SAMPLER, binding 3 = STORAGE_IMAGE */
    VkDescriptorSetLayoutBinding bindings[4];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 4;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &outPipeline->descriptorSetLayout) != VK_SUCCESS) {
        ffxVkDlssDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 3. Pipeline layout with push constants */
    VkPushConstantRange pushRange = {0};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(FfxVkDlssPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &outPipeline->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &outPipeline->pipelineLayout) != VK_SUCCESS) {
        ffxVkDlssDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 4. Shader module */
    VkShaderModuleCreateInfo moduleInfo = {0};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = g_d4r_swin_reconstruct_spv_size;
    moduleInfo.pCode = g_d4r_swin_reconstruct_spv;
    if (vkCreateShaderModule(device, &moduleInfo, NULL, &outPipeline->shaderModule) != VK_SUCCESS) {
        ffxVkDlssDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 5. Compute pipeline */
    VkComputePipelineCreateInfo computeInfo = {0};
    computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeInfo.stage.module = outPipeline->shaderModule;
    computeInfo.stage.pName = "main";
    computeInfo.layout = outPipeline->pipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, NULL, &outPipeline->pipeline) != VK_SUCCESS) {
        ffxVkDlssDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 6. Descriptor pool and descriptor sets */
    VkDescriptorPoolSize poolSizes[2];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 3 * FFX_VK_DLSS_DESCRIPTOR_SET_COUNT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1 * FFX_VK_DLSS_DESCRIPTOR_SET_COUNT;

    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = FFX_VK_DLSS_DESCRIPTOR_SET_COUNT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, NULL, &outPipeline->descriptorPool) != VK_SUCCESS) {
        ffxVkDlssDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    VkDescriptorSetLayout layouts[FFX_VK_DLSS_DESCRIPTOR_SET_COUNT];
    for (uint32_t i = 0; i < FFX_VK_DLSS_DESCRIPTOR_SET_COUNT; ++i)
        layouts[i] = outPipeline->descriptorSetLayout;

    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = outPipeline->descriptorPool;
    allocInfo.descriptorSetCount = FFX_VK_DLSS_DESCRIPTOR_SET_COUNT;
    allocInfo.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(device, &allocInfo, outPipeline->descriptorSets) != VK_SUCCESS) {
        ffxVkDlssDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    return FFX_VK_PORTABLE_OK;
}

void ffxVkDlssDestroyPipeline(
    VkDevice device,
    FfxVkDlssPipeline* pipeline)
{
    if (!pipeline)
        return;
    if (device != VK_NULL_HANDLE) {
        if (pipeline->pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, pipeline->pipeline, NULL);
        if (pipeline->shaderModule != VK_NULL_HANDLE)
            vkDestroyShaderModule(device, pipeline->shaderModule, NULL);
        if (pipeline->pipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device, pipeline->pipelineLayout, NULL);
        if (pipeline->descriptorPool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device, pipeline->descriptorPool, NULL);
        if (pipeline->descriptorSetLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device, pipeline->descriptorSetLayout, NULL);
        if (pipeline->linearSampler != VK_NULL_HANDLE)
            vkDestroySampler(device, pipeline->linearSampler, NULL);
    }
    memset(pipeline, 0, sizeof(*pipeline));
}

FfxVkPortableResult ffxVkDlssPipelineSetModel(
    FfxVkDlssPipeline* pipeline,
    const void* modelContainerData,
    size_t modelContainerSizeBytes)
{
    if (!pipeline || !modelContainerData)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    if (modelContainerSizeBytes < 24)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    const uint8_t* bytes = (const uint8_t*)modelContainerData;
    if (memcmp(bytes, "DLSSMOD1", 8) == 0) {
        uint32_t version = 0, modelFamily = 0;
        memcpy(&version, bytes + 8, sizeof(version));
        memcpy(&modelFamily, bytes + 12, sizeof(modelFamily));
        if (version != 1 || modelFamily > FFX_VK_DLSS_MODEL_5_NEURAL_RENDERING || modelContainerSizeBytes < 65536)
            return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    } else if (memcmp(bytes, "DLSSNR1\0", 8) == 0 || memcmp(bytes, "DLSSNR1", 7) == 0) {
        uint32_t version = 0;
        memcpy(&version, bytes + 8, sizeof(version));
        if (version != 1 || modelContainerSizeBytes < 1048576)
            return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    } else {
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    }

    pipeline->hasPretrainedWeights = true;
    pipeline->weightsSizeBytes = modelContainerSizeBytes;
    return FFX_VK_PORTABLE_OK;
}

FfxVkPortableResult ffxVkDlssExecuteDispatch(
    VkCommandBuffer cmdBuf,
    FfxVkDlssPipeline* pipeline,
    const FfxVkDlssDispatchInfo* dispatchInfo)
{
    if (cmdBuf == VK_NULL_HANDLE || !pipeline || pipeline->pipeline == VK_NULL_HANDLE || !dispatchInfo)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    if (dispatchInfo->color.view == VK_NULL_HANDLE ||
        dispatchInfo->motionVectors.view == VK_NULL_HANDLE ||
        dispatchInfo->depth.view == VK_NULL_HANDLE ||
        dispatchInfo->output.view == VK_NULL_HANDLE) {
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    }

    uint32_t setIndex = pipeline->currentSetIndex % FFX_VK_DLSS_DESCRIPTOR_SET_COUNT;
    VkDescriptorSet dstSet = pipeline->descriptorSets[setIndex];
    pipeline->currentSetIndex = (setIndex + 1) % FFX_VK_DLSS_DESCRIPTOR_SET_COUNT;

    VkDescriptorImageInfo imageInfos[4];
    imageInfos[0].sampler = pipeline->linearSampler;
    imageInfos[0].imageView = dispatchInfo->color.view;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[1].sampler = pipeline->linearSampler;
    imageInfos[1].imageView = dispatchInfo->motionVectors.view;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[2].sampler = pipeline->linearSampler;
    imageInfos[2].imageView = dispatchInfo->depth.view;
    imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[3].sampler = VK_NULL_HANDLE;
    imageInfos[3].imageView = dispatchInfo->output.view;
    imageInfos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[4];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dstSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = (i == 3) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &imageInfos[i];
    }
    vkUpdateDescriptorSets(pipeline->device, 4, writes, 0, NULL);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 1, &dstSet, 0, NULL);

    FfxVkDlssPushConstants pc;
    pc.inputExtent[0] = dispatchInfo->renderSize.width;
    pc.inputExtent[1] = dispatchInfo->renderSize.height;
    pc.outputExtent[0] = dispatchInfo->output.extent.width;
    pc.outputExtent[1] = dispatchInfo->output.extent.height;
    pc.jitterOffset[0] = dispatchInfo->jitterOffset.x;
    pc.jitterOffset[1] = dispatchInfo->jitterOffset.y;
    pc.motionVectorScale[0] = dispatchInfo->motionVectorScale.x;
    pc.motionVectorScale[1] = dispatchInfo->motionVectorScale.y;
    pc.sharpness = 0.25f;
    pc.preExposure = dispatchInfo->preExposure > 0.0f ? dispatchInfo->preExposure : 1.0f;
    pc.resetHistory = dispatchInfo->frameReset ? 1u : 0u;
    pc.flags = dispatchInfo->flags;

    vkCmdPushConstants(cmdBuf, pipeline->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groupCountX = (dispatchInfo->output.extent.width + 7) / 8;
    uint32_t groupCountY = (dispatchInfo->output.extent.height + 7) / 8;
    vkCmdDispatch(cmdBuf, groupCountX, groupCountY, 1);

    return FFX_VK_PORTABLE_OK;
}

