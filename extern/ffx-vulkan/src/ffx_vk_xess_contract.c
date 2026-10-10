/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_xess_contract.h"
#include "xess_unet_reconstruct_spv.h"

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

typedef struct FfxVkXessPushConstants {
    uint32_t inputExtent[2];
    uint32_t outputExtent[2];
    float jitterOffset[2];
    float sharpness;
    uint32_t flags;
} FfxVkXessPushConstants;

const uint32_t *ffxVkXessGetReconstructSpirv(size_t *outWordCount)
{
    if (outWordCount)
        *outWordCount = g_xess_unet_reconstruct_spv_word_count;
    return g_xess_unet_reconstruct_spv;
}

FfxVkPortableResult ffxVkXessCreatePipeline(
    VkDevice device,
    FfxVkXessPipeline *outPipeline)
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
        ffxVkXessDestroyPipeline(device, outPipeline);
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
        ffxVkXessDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 3. Pipeline layout with push constants */
    VkPushConstantRange pushRange = {0};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(FfxVkXessPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &outPipeline->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &outPipeline->pipelineLayout) != VK_SUCCESS) {
        ffxVkXessDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 4. Shader module */
    VkShaderModuleCreateInfo moduleInfo = {0};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = g_xess_unet_reconstruct_spv_size;
    moduleInfo.pCode = g_xess_unet_reconstruct_spv;
    if (vkCreateShaderModule(device, &moduleInfo, NULL, &outPipeline->shaderModule) != VK_SUCCESS) {
        ffxVkXessDestroyPipeline(device, outPipeline);
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
        ffxVkXessDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 6. Descriptor pool and descriptor sets */
    VkDescriptorPoolSize poolSizes[2];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 3 * FFX_VK_XESS_DESCRIPTOR_SET_COUNT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1 * FFX_VK_XESS_DESCRIPTOR_SET_COUNT;

    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = FFX_VK_XESS_DESCRIPTOR_SET_COUNT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, NULL, &outPipeline->descriptorPool) != VK_SUCCESS) {
        ffxVkXessDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    VkDescriptorSetLayout layouts[FFX_VK_XESS_DESCRIPTOR_SET_COUNT];
    for (uint32_t i = 0; i < FFX_VK_XESS_DESCRIPTOR_SET_COUNT; ++i)
        layouts[i] = outPipeline->descriptorSetLayout;

    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = outPipeline->descriptorPool;
    allocInfo.descriptorSetCount = FFX_VK_XESS_DESCRIPTOR_SET_COUNT;
    allocInfo.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(device, &allocInfo, outPipeline->descriptorSets) != VK_SUCCESS) {
        ffxVkXessDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    return FFX_VK_PORTABLE_OK;
}

void ffxVkXessDestroyPipeline(
    VkDevice device,
    FfxVkXessPipeline *pipeline)
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

FfxVkPortableResult ffxVkXessPipelineSetModel(
    FfxVkXessPipeline *pipeline,
    const void *modelContainerData,
    size_t modelContainerSizeBytes)
{
    if (!pipeline || !modelContainerData)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    if (modelContainerSizeBytes < 24)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    const uint8_t *bytes = (const uint8_t *)modelContainerData;
    if (memcmp(bytes, "XESSMOD2", 8) != 0)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    uint32_t version = 0, numLayers = 0;
    memcpy(&version, bytes + 8, sizeof(version));
    memcpy(&numLayers, bytes + 12, sizeof(numLayers));
    if (version != 2 || numLayers != FFX_VK_XESS_LAYER_COUNT)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    if (modelContainerSizeBytes < (size_t)ffxVkXessGetTotalWeightBytes())
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    pipeline->hasPretrainedWeights = true;
    pipeline->weightsSizeBytes = modelContainerSizeBytes;
    return FFX_VK_PORTABLE_OK;
}

FfxVkPortableResult ffxVkXessExecuteDispatch(
    VkCommandBuffer cmdBuf,
    FfxVkXessPipeline *pipeline,
    const FfxVkXessDispatchInfo *dispatchInfo)
{
    if (cmdBuf == VK_NULL_HANDLE || !pipeline || pipeline->pipeline == VK_NULL_HANDLE || !dispatchInfo)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    if (dispatchInfo->colorIn.view == VK_NULL_HANDLE ||
        dispatchInfo->velocity.view == VK_NULL_HANDLE ||
        dispatchInfo->depth.view == VK_NULL_HANDLE ||
        dispatchInfo->colorOut.view == VK_NULL_HANDLE) {
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    }

    uint32_t setIndex = pipeline->currentSetIndex % FFX_VK_XESS_DESCRIPTOR_SET_COUNT;
    VkDescriptorSet dstSet = pipeline->descriptorSets[setIndex];
    pipeline->currentSetIndex = (setIndex + 1) % FFX_VK_XESS_DESCRIPTOR_SET_COUNT;

    VkDescriptorImageInfo imageInfos[4];
    imageInfos[0].sampler = pipeline->linearSampler;
    imageInfos[0].imageView = dispatchInfo->colorIn.view;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[1].sampler = pipeline->linearSampler;
    imageInfos[1].imageView = dispatchInfo->velocity.view;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[2].sampler = pipeline->linearSampler;
    imageInfos[2].imageView = dispatchInfo->depth.view;
    imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[3].sampler = VK_NULL_HANDLE;
    imageInfos[3].imageView = dispatchInfo->colorOut.view;
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

    FfxVkXessPushConstants pc;
    pc.inputExtent[0] = dispatchInfo->inputExtent.width;
    pc.inputExtent[1] = dispatchInfo->inputExtent.height;
    pc.outputExtent[0] = dispatchInfo->outputExtent.width;
    pc.outputExtent[1] = dispatchInfo->outputExtent.height;
    pc.jitterOffset[0] = dispatchInfo->jitterOffsetX;
    pc.jitterOffset[1] = dispatchInfo->jitterOffsetY;
    pc.sharpness = dispatchInfo->sharpness;
    pc.flags = dispatchInfo->resetHistory ? 1u : 0u;

    vkCmdPushConstants(cmdBuf, pipeline->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groupCountX = (dispatchInfo->outputExtent.width + 7) / 8;
    uint32_t groupCountY = (dispatchInfo->outputExtent.height + 7) / 8;
    vkCmdDispatch(cmdBuf, groupCountX, groupCountY, 1);

    return FFX_VK_PORTABLE_OK;
}

