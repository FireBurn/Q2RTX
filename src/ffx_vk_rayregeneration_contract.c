/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_rayregeneration_contract.h"
#include "rr_reconstruct_spv.h"

#include <math.h>
#include <string.h>

static int image_is_readable(const FfxVkPortableImage* image,
                             FfxVkPortableExtent2D required_size,
                             VkFormat format_a, VkFormat format_b,
                             uint64_t* issues) {
    if (image->structSize != sizeof(*image)) {
        *issues |= FFX_VK_RR_VALIDATION_STRUCT_SIZE;
        return 0;
    }
    if (image->image == VK_NULL_HANDLE) {
        *issues |= FFX_VK_RR_VALIDATION_IMAGE_HANDLE;
        return 0;
    }
    if (image->extent.width < required_size.width ||
        image->extent.height < required_size.height) {
        *issues |= FFX_VK_RR_VALIDATION_IMAGE_EXTENT;
    }
    if ((image->usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
        *issues |= FFX_VK_RR_VALIDATION_IMAGE_USAGE;
    }
    if (image->state != FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ &&
        image->state != FFX_VK_PORTABLE_RESOURCE_STATE_COMPUTE_READ) {
        *issues |= FFX_VK_RR_VALIDATION_IMAGE_STATE;
    }
    if (image->format != format_a && image->format != format_b) {
        *issues |= FFX_VK_RR_VALIDATION_IMAGE_FORMAT;
    }
    return 1;
}

static int images_alias(const FfxVkPortableImage* input,
                        const FfxVkPortableImage* output) {
    return input->image != VK_NULL_HANDLE && input->image == output->image;
}

static void image_is_writable(const FfxVkPortableImage* input,
                              const FfxVkPortableImage* output,
                              FfxVkPortableExtent2D render_size,
                              VkFormat format, uint64_t* issues) {
    if (output->structSize != sizeof(*output) ||
        output->image == VK_NULL_HANDLE ||
        output->extent.width < render_size.width ||
        output->extent.height < render_size.height ||
        (output->usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0 ||
        output->format != format ||
        (!images_alias(input, output) &&
         output->state != FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS))
        *issues |= FFX_VK_RR_VALIDATION_OUTPUT_IMAGE;
}

static FfxVkPortableExtent2D signal_extent(
    const FfxVkRayRegenerationInputs* inputs, uint32_t signal) {
    FfxVkPortableExtent2D result = inputs->renderSize;
    if (inputs->checkerboardSignalFlags & signal)
        result.width = (result.width + 1u) / 2u;
    return result;
}

static int finite_float3(FfxVkPortableFloat3 value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

static int finite_float2(FfxVkPortableFloat2 value) {
    return isfinite(value.x) && isfinite(value.y);
}

static int finite_matrix(const float matrix[16]) {
    uint32_t i;
    for (i = 0; i < 16; ++i) {
        if (!isfinite(matrix[i]))
            return 0;
    }
    return 1;
}

FfxVkPortableResult ffxVkRayRegenerationValidateInputs(
    const FfxVkRayRegenerationInputs* inputs, uint64_t* issues) {
    uint64_t result = FFX_VK_RR_VALIDATION_NONE;
    const uint32_t known_signals = FFX_VK_RR_SIGNAL_DIRECT_DIFFUSE |
        FFX_VK_RR_SIGNAL_DIRECT_SPECULAR |
        FFX_VK_RR_SIGNAL_INDIRECT_DIFFUSE |
        FFX_VK_RR_SIGNAL_INDIRECT_SPECULAR |
        FFX_VK_RR_SIGNAL_DOMINANT_LIGHT_VISIBILITY |
        FFX_VK_RR_SIGNAL_AMBIENT_OCCLUSION |
        FFX_VK_RR_SIGNAL_SPECULAR_OCCLUSION;
    const uint32_t primary_signals = FFX_VK_RR_SIGNAL_DIRECT_DIFFUSE |
        FFX_VK_RR_SIGNAL_DIRECT_SPECULAR |
        FFX_VK_RR_SIGNAL_INDIRECT_DIFFUSE |
        FFX_VK_RR_SIGNAL_INDIRECT_SPECULAR |
        FFX_VK_RR_SIGNAL_DOMINANT_LIGHT_VISIBILITY;

    if (!issues)
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    *issues = FFX_VK_RR_VALIDATION_NONE;
    if (!inputs)
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    if (inputs->structSize != sizeof(*inputs) ||
        inputs->contractVersion != FFX_VK_RAYREGENERATION_CONTRACT_VERSION)
        result |= FFX_VK_RR_VALIDATION_STRUCT_SIZE;
    if (!inputs->renderSize.width || !inputs->renderSize.height)
        result |= FFX_VK_RR_VALIDATION_ZERO_EXTENT;

    /* AO and specular occlusion are optional additions. A real RR provider
     * requires at least one primary radiance or dominant-light signal. */
    if (!(inputs->signalFlags & primary_signals))
        result |= FFX_VK_RR_VALIDATION_REQUIRED_SIGNAL;
    if (inputs->signalFlags & ~known_signals)
        result |= FFX_VK_RR_VALIDATION_SIGNAL_FLAGS;
    if ((inputs->checkerboardSignalFlags & ~known_signals) ||
        (inputs->checkerboardSignalFlags & ~inputs->signalFlags) ||
        (inputs->checkerboardOriginBits & ~inputs->checkerboardSignalFlags))
        result |= FFX_VK_RR_VALIDATION_CHECKERBOARD;
    image_is_readable(&inputs->linearDepth, inputs->renderSize,
        VK_FORMAT_R32_SFLOAT, VK_FORMAT_UNDEFINED, &result);
    image_is_readable(&inputs->motionVectors, inputs->renderSize,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, &result);
    image_is_readable(&inputs->normalsRoughnessMaterial, inputs->renderSize,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, &result);
    image_is_readable(&inputs->diffuseAlbedo, inputs->renderSize,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, &result);
    image_is_readable(&inputs->specularAlbedo, inputs->renderSize,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, &result);
    if (!finite_float3(inputs->motionVectorScale) ||
        !finite_float2(inputs->jitterOffset) ||
        !finite_float3(inputs->cameraPositionDelta) ||
        !finite_matrix(inputs->view) || !finite_matrix(inputs->projection) ||
        !isfinite(inputs->linearDepthMin) || !isfinite(inputs->linearDepthMax))
        result |= FFX_VK_RR_VALIDATION_NONFINITE_METADATA;
    if (inputs->motionVectorScale.x == 0.0f ||
        inputs->motionVectorScale.y == 0.0f ||
        inputs->motionVectorScale.z == 0.0f ||
        inputs->linearDepthMin < 0.0f ||
        inputs->linearDepthMax <= inputs->linearDepthMin)
        result |= FFX_VK_RR_VALIDATION_CAMERA_METADATA;

    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_DIRECT_DIFFUSE) != 0)
        image_is_readable(&inputs->directDiffuse, signal_extent(inputs,
            FFX_VK_RR_SIGNAL_DIRECT_DIFFUSE),
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, &result);
    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_DIRECT_SPECULAR) != 0)
        image_is_readable(&inputs->directSpecular, signal_extent(inputs,
            FFX_VK_RR_SIGNAL_DIRECT_SPECULAR),
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, &result);
    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_INDIRECT_DIFFUSE) != 0)
        image_is_readable(&inputs->indirectDiffuse, signal_extent(inputs,
            FFX_VK_RR_SIGNAL_INDIRECT_DIFFUSE),
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, &result);
    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_INDIRECT_SPECULAR) != 0)
        image_is_readable(&inputs->indirectSpecular, signal_extent(inputs,
            FFX_VK_RR_SIGNAL_INDIRECT_SPECULAR),
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, &result);
    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_AMBIENT_OCCLUSION) != 0)
        image_is_readable(&inputs->ambientOcclusion, signal_extent(inputs,
            FFX_VK_RR_SIGNAL_AMBIENT_OCCLUSION),
            VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, &result);
    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_SPECULAR_OCCLUSION) != 0)
        image_is_readable(&inputs->specularOcclusion, signal_extent(inputs,
            FFX_VK_RR_SIGNAL_SPECULAR_OCCLUSION),
            VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, &result);
    if (inputs->directAlphaSemantic != FFX_VK_RR_ALPHA_NONNEGATIVE_UNDEFINED ||
        inputs->indirectAlphaSemantic != FFX_VK_RR_ALPHA_FIRST_LOBE_HIT_DISTANCE ||
        !isfinite(inputs->noHitDistance) || inputs->noHitDistance <= 0.0f)
        result |= FFX_VK_RR_VALIDATION_ALPHA_SEMANTIC;

    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_DOMINANT_LIGHT_VISIBILITY) != 0) {
        image_is_readable(&inputs->dominantLightVisibility, signal_extent(inputs,
            FFX_VK_RR_SIGNAL_DOMINANT_LIGHT_VISIBILITY),
            VK_FORMAT_R16_SFLOAT, VK_FORMAT_UNDEFINED, &result);
        if (!finite_float3(inputs->dominantLightDirection) ||
            !finite_float3(inputs->dominantLightEmission) ||
            !isfinite(inputs->dominantLightAngularRadius) ||
            inputs->dominantLightAngularRadius <= 0.0f ||
            (inputs->dominantLightDirection.x * inputs->dominantLightDirection.x +
             inputs->dominantLightDirection.y * inputs->dominantLightDirection.y +
             inputs->dominantLightDirection.z * inputs->dominantLightDirection.z) < 1e-6f)
            result |= FFX_VK_RR_VALIDATION_DOMINANT_LIGHT;
    }

    *issues = result;
    return result == FFX_VK_RR_VALIDATION_NONE ? FFX_VK_PORTABLE_OK
                                                 : FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
}

FfxVkPortableResult ffxVkRayRegenerationValidateOutputs(
    const FfxVkRayRegenerationInputs* inputs,
    const FfxVkRayRegenerationOutputs* outputs, uint64_t* issues) {
    uint64_t result = FFX_VK_RR_VALIDATION_NONE;

    if (!issues)
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    *issues = FFX_VK_RR_VALIDATION_NONE;
    if (!inputs || !outputs)
        return FFX_VK_PORTABLE_ERROR_INVALID_POINTER;
    if (ffxVkRayRegenerationValidateInputs(inputs, &result) !=
        FFX_VK_PORTABLE_OK) {
        *issues = result;
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    }
    if (outputs->structSize != sizeof(*outputs) ||
        outputs->contractVersion != FFX_VK_RAYREGENERATION_CONTRACT_VERSION)
        result |= FFX_VK_RR_VALIDATION_STRUCT_SIZE;
    if (inputs->signalFlags & FFX_VK_RR_SIGNAL_DIRECT_DIFFUSE)
        image_is_writable(&inputs->directDiffuse, &outputs->directDiffuse,
            inputs->renderSize, VK_FORMAT_R16G16B16A16_SFLOAT, &result);
    if (inputs->signalFlags & FFX_VK_RR_SIGNAL_DIRECT_SPECULAR)
        image_is_writable(&inputs->directSpecular, &outputs->directSpecular,
            inputs->renderSize, VK_FORMAT_R16G16B16A16_SFLOAT, &result);
    if (inputs->signalFlags & FFX_VK_RR_SIGNAL_INDIRECT_DIFFUSE)
        image_is_writable(&inputs->indirectDiffuse, &outputs->indirectDiffuse,
            inputs->renderSize, VK_FORMAT_R16G16B16A16_SFLOAT, &result);
    if (inputs->signalFlags & FFX_VK_RR_SIGNAL_INDIRECT_SPECULAR)
        image_is_writable(&inputs->indirectSpecular, &outputs->indirectSpecular,
            inputs->renderSize, VK_FORMAT_R16G16B16A16_SFLOAT, &result);
    if (inputs->signalFlags & FFX_VK_RR_SIGNAL_AMBIENT_OCCLUSION)
        image_is_writable(&inputs->ambientOcclusion, &outputs->ambientOcclusion,
            inputs->renderSize, VK_FORMAT_R8_UNORM, &result);
    if (inputs->signalFlags & FFX_VK_RR_SIGNAL_SPECULAR_OCCLUSION)
        image_is_writable(&inputs->specularOcclusion, &outputs->specularOcclusion,
            inputs->renderSize, VK_FORMAT_R8_UNORM, &result);
    if (inputs->signalFlags & FFX_VK_RR_SIGNAL_DOMINANT_LIGHT_VISIBILITY)
        image_is_writable(&inputs->dominantLightVisibility,
            &outputs->dominantLightVisibility, inputs->renderSize,
            VK_FORMAT_R8_UNORM, &result);

    *issues = result;
    return result == FFX_VK_RR_VALIDATION_NONE ? FFX_VK_PORTABLE_OK
                                                 : FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
}

typedef struct FfxVkRayRegenerationPushConstants {
    uint32_t renderSize[2];
    float jitterOffset[2];
    float cameraPositionDelta[3];
    float linearDepthMin;
    float linearDepthMax;
    float normalSigma;
    float depthSigma;
    float roughnessSigma;
    uint32_t signalFlags;
} FfxVkRayRegenerationPushConstants;

const uint32_t* ffxVkRayRegenerationGetReconstructSpirv(size_t* outWordCount)
{
    if (outWordCount)
        *outWordCount = g_rr_reconstruct_spv_word_count;
    return g_rr_reconstruct_spv;
}

FfxVkPortableResult ffxVkRayRegenerationCreatePipeline(
    VkDevice device,
    FfxVkRayRegenerationPipeline* outPipeline)
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
        ffxVkRayRegenerationDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 2. Descriptor set layout: bindings 0,1,2,3 = SAMPLER, binding 4 = STORAGE_IMAGE */
    VkDescriptorSetLayoutBinding bindings[5];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 5;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &outPipeline->descriptorSetLayout) != VK_SUCCESS) {
        ffxVkRayRegenerationDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 3. Pipeline layout with push constants */
    VkPushConstantRange pushRange = {0};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(FfxVkRayRegenerationPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &outPipeline->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &outPipeline->pipelineLayout) != VK_SUCCESS) {
        ffxVkRayRegenerationDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 4. Shader module */
    VkShaderModuleCreateInfo moduleInfo = {0};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = g_rr_reconstruct_spv_size;
    moduleInfo.pCode = g_rr_reconstruct_spv;
    if (vkCreateShaderModule(device, &moduleInfo, NULL, &outPipeline->shaderModule) != VK_SUCCESS) {
        ffxVkRayRegenerationDestroyPipeline(device, outPipeline);
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
        ffxVkRayRegenerationDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    /* 6. Descriptor pool and descriptor sets */
    VkDescriptorPoolSize poolSizes[2];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 4 * FFX_VK_RR_DESCRIPTOR_SET_COUNT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1 * FFX_VK_RR_DESCRIPTOR_SET_COUNT;

    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = FFX_VK_RR_DESCRIPTOR_SET_COUNT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, NULL, &outPipeline->descriptorPool) != VK_SUCCESS) {
        ffxVkRayRegenerationDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    VkDescriptorSetLayout layouts[FFX_VK_RR_DESCRIPTOR_SET_COUNT];
    for (uint32_t i = 0; i < FFX_VK_RR_DESCRIPTOR_SET_COUNT; ++i)
        layouts[i] = outPipeline->descriptorSetLayout;

    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = outPipeline->descriptorPool;
    allocInfo.descriptorSetCount = FFX_VK_RR_DESCRIPTOR_SET_COUNT;
    allocInfo.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(device, &allocInfo, outPipeline->descriptorSets) != VK_SUCCESS) {
        ffxVkRayRegenerationDestroyPipeline(device, outPipeline);
        return FFX_VK_PORTABLE_ERROR_VULKAN;
    }

    return FFX_VK_PORTABLE_OK;
}

void ffxVkRayRegenerationDestroyPipeline(
    VkDevice device,
    FfxVkRayRegenerationPipeline* pipeline)
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

FfxVkPortableResult ffxVkRayRegenerationExecuteDispatch(
    VkCommandBuffer cmdBuf,
    FfxVkRayRegenerationPipeline* pipeline,
    const FfxVkRayRegenerationInputs* inputs,
    const FfxVkRayRegenerationOutputs* outputs)
{
    if (cmdBuf == VK_NULL_HANDLE || !pipeline || pipeline->pipeline == VK_NULL_HANDLE || !inputs || !outputs)
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;

    VkImageView inRadiance = VK_NULL_HANDLE;
    VkImageView outRadiance = VK_NULL_HANDLE;

    if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_DIRECT_DIFFUSE) && inputs->directDiffuse.view != VK_NULL_HANDLE) {
        inRadiance = inputs->directDiffuse.view;
        outRadiance = outputs->directDiffuse.view;
    } else if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_INDIRECT_DIFFUSE) && inputs->indirectDiffuse.view != VK_NULL_HANDLE) {
        inRadiance = inputs->indirectDiffuse.view;
        outRadiance = outputs->indirectDiffuse.view;
    } else if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_DIRECT_SPECULAR) && inputs->directSpecular.view != VK_NULL_HANDLE) {
        inRadiance = inputs->directSpecular.view;
        outRadiance = outputs->directSpecular.view;
    } else if ((inputs->signalFlags & FFX_VK_RR_SIGNAL_INDIRECT_SPECULAR) && inputs->indirectSpecular.view != VK_NULL_HANDLE) {
        inRadiance = inputs->indirectSpecular.view;
        outRadiance = outputs->indirectSpecular.view;
    } else if (inputs->directDiffuse.view != VK_NULL_HANDLE) {
        inRadiance = inputs->directDiffuse.view;
        outRadiance = outputs->directDiffuse.view;
    }

    if (inRadiance == VK_NULL_HANDLE || outRadiance == VK_NULL_HANDLE ||
        inputs->linearDepth.view == VK_NULL_HANDLE ||
        inputs->normalsRoughnessMaterial.view == VK_NULL_HANDLE ||
        inputs->motionVectors.view == VK_NULL_HANDLE) {
        return FFX_VK_PORTABLE_ERROR_INVALID_ARGUMENT;
    }

    uint32_t setIndex = pipeline->currentSetIndex % FFX_VK_RR_DESCRIPTOR_SET_COUNT;
    VkDescriptorSet dstSet = pipeline->descriptorSets[setIndex];
    pipeline->currentSetIndex = (setIndex + 1) % FFX_VK_RR_DESCRIPTOR_SET_COUNT;

    VkDescriptorImageInfo imageInfos[5];
    imageInfos[0].sampler = pipeline->linearSampler;
    imageInfos[0].imageView = inRadiance;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[1].sampler = pipeline->linearSampler;
    imageInfos[1].imageView = inputs->linearDepth.view;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[2].sampler = pipeline->linearSampler;
    imageInfos[2].imageView = inputs->normalsRoughnessMaterial.view;
    imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[3].sampler = pipeline->linearSampler;
    imageInfos[3].imageView = inputs->motionVectors.view;
    imageInfos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    imageInfos[4].sampler = VK_NULL_HANDLE;
    imageInfos[4].imageView = outRadiance;
    imageInfos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[5];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dstSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = (i == 4) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &imageInfos[i];
    }
    vkUpdateDescriptorSets(pipeline->device, 5, writes, 0, NULL);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 1, &dstSet, 0, NULL);

    FfxVkRayRegenerationPushConstants pc;
    pc.renderSize[0] = inputs->renderSize.width;
    pc.renderSize[1] = inputs->renderSize.height;
    pc.jitterOffset[0] = inputs->jitterOffset.x;
    pc.jitterOffset[1] = inputs->jitterOffset.y;
    pc.cameraPositionDelta[0] = inputs->cameraPositionDelta.x;
    pc.cameraPositionDelta[1] = inputs->cameraPositionDelta.y;
    pc.cameraPositionDelta[2] = inputs->cameraPositionDelta.z;
    pc.linearDepthMin = inputs->linearDepthMin;
    pc.linearDepthMax = inputs->linearDepthMax;
    pc.normalSigma = 16.0f;
    pc.depthSigma = 1.0f;
    pc.roughnessSigma = 0.25f;
    pc.signalFlags = inputs->signalFlags;

    vkCmdPushConstants(cmdBuf, pipeline->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groupCountX = (inputs->renderSize.width + 7) / 8;
    uint32_t groupCountY = (inputs->renderSize.height + 7) / 8;
    vkCmdDispatch(cmdBuf, groupCountX, groupCountY, 1);

    return FFX_VK_PORTABLE_OK;
}
