/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef FFX_VK_XESS_CONTRACT_H
#define FFX_VK_XESS_CONTRACT_H

#include "ffx_vk_portable.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Provider-neutral Vulkan contract for Intel XeSS 2 / 3 Super Resolution.
 * It models the 14-dispatch U-Net convolutional pipeline, 13 neural weight layers
 * (253,280 bytes), cross-vendor DP4a and Intel XMX execution paths, and memory
 * requirements for binary model containers (XESSMOD2).
 * Enables native integration into any Vulkan project without Intel Windows runtimes. */
#define FFX_VK_XESS_CONTRACT_VERSION 1u

/* XeSS Execution Paths */
typedef enum FfxVkXessPath {
    FFX_VK_XESS_PATH_AUTO = 0,
    FFX_VK_XESS_PATH_DP4A = 1,              /* Cross-vendor INT8 dot product (AMD RDNA2+, NVIDIA, Intel) */
    FFX_VK_XESS_PATH_XMX = 2,               /* Intel Arc Xe Matrix Extensions */
    FFX_VK_XESS_PATH_GENERIC_COMPUTE = 3     /* Fallback FP16/FP32 compute path */
} FfxVkXessPath;

/* XeSS Quality Presets */
typedef enum FfxVkXessQuality {
    FFX_VK_XESS_QUALITY_ULTRA_QUALITY_PLUS = 0, /* ~1.3x scale */
    FFX_VK_XESS_QUALITY_ULTRA_QUALITY = 1,      /* ~1.5x scale */
    FFX_VK_XESS_QUALITY_QUALITY = 2,            /* ~1.7x scale (e.g. 753x424 -> 1280x720) */
    FFX_VK_XESS_QUALITY_BALANCED = 3,           /* ~2.0x scale */
    FFX_VK_XESS_QUALITY_PERFORMANCE = 4,        /* ~2.3x scale */
    FFX_VK_XESS_QUALITY_ULTRA_PERFORMANCE = 5   /* ~3.0x scale */
} FfxVkXessQuality;

/* Feature and capability flags */
typedef enum FfxVkXessFlagBits {
    FFX_VK_XESS_FLAG_ENABLE_AUTO_EXPOSURE = 1u << 0,
    FFX_VK_XESS_FLAG_HDR_INPUT = 1u << 1,
    FFX_VK_XESS_FLAG_INVERTED_DEPTH = 1u << 2,
    FFX_VK_XESS_FLAG_JITTERED_MOTION = 1u << 3,
    FFX_VK_XESS_FLAG_HIGH_RES_MOTION = 1u << 4,
    FFX_VK_XESS_FLAG_RESPONSIVE_PIXEL_MASK = 1u << 5,
} FfxVkXessFlagBits;

/* Validation issue bits */
typedef enum FfxVkXessValidationIssueBits {
    FFX_VK_XESS_VALIDATION_NONE = 0,
    FFX_VK_XESS_VALIDATION_STRUCT_SIZE = 1ull << 0,
    FFX_VK_XESS_VALIDATION_CONTRACT_VERSION = 1ull << 1,
    FFX_VK_XESS_VALIDATION_ZERO_EXTENT = 1ull << 2,
    FFX_VK_XESS_VALIDATION_EXTENT_EXCEEDS_MAX = 1ull << 3,
    FFX_VK_XESS_VALIDATION_IMAGE_HANDLE = 1ull << 4,
    FFX_VK_XESS_VALIDATION_IMAGE_FORMAT = 1ull << 5,
    FFX_VK_XESS_VALIDATION_IMAGE_USAGE = 1ull << 6,
    FFX_VK_XESS_VALIDATION_MISSING_DP4A = 1ull << 7,
    FFX_VK_XESS_VALIDATION_MISSING_FORMATLESS_WRITE = 1ull << 8,
    FFX_VK_XESS_VALIDATION_MODEL_CONTAINER_INVALID = 1ull << 9,
    FFX_VK_XESS_VALIDATION_MODEL_WEIGHTS_TRUNCATED = 1ull << 10,
    FFX_VK_XESS_VALIDATION_NONFINITE_CONSTANTS = 1ull << 11,
} FfxVkXessValidationIssueBits;

/* U-Net Layer Identification (13 layers) */
typedef enum FfxVkXessLayerId {
    FFX_VK_XESS_LAYER_I10 = 0,  /* Dispatch 1  (encoder down_1) */
    FFX_VK_XESS_LAYER_I12 = 1,  /* Dispatch 2  (encoder down_2) */
    FFX_VK_XESS_LAYER_I13 = 2,  /* Dispatch 3  (encoder down_3) */
    FFX_VK_XESS_LAYER_I14 = 3,  /* Dispatch 4  (encoder down_4) */
    FFX_VK_XESS_LAYER_I15 = 4,  /* Dispatch 5  (bottleneck) */
    FFX_VK_XESS_LAYER_I16 = 5,  /* Dispatch 6  (decoder up_4 skip 3) */
    FFX_VK_XESS_LAYER_I17 = 6,  /* Dispatch 7  (decoder up_3a) */
    FFX_VK_XESS_LAYER_I18 = 7,  /* Dispatch 8  (decoder up_3 skip 2) */
    FFX_VK_XESS_LAYER_I19 = 8,  /* Dispatch 9  (decoder up_2a) */
    FFX_VK_XESS_LAYER_I20 = 9,  /* Dispatch 10 (decoder up_2 skip 1) */
    FFX_VK_XESS_LAYER_I21 = 10, /* Dispatch 11 (decoder up_1a) */
    FFX_VK_XESS_LAYER_I23 = 11, /* Dispatch 12 (decoder up_1 skip 0) */
    FFX_VK_XESS_LAYER_I8  = 12, /* Dispatch 13 (resolve pass) */
    FFX_VK_XESS_LAYER_COUNT = 13
} FfxVkXessLayerId;

/* Metadata for a single neural layer in the U-Net */
typedef struct FfxVkXessLayerMeta {
    const char *name;
    uint32_t dispatchId;
    uint32_t weightsBytes;
    uint32_t scaleBytes;
    uint32_t biasBytes;
} FfxVkXessLayerMeta;

/* XeSS Context Creation Descriptor */
typedef struct FfxVkXessCreateInfo {
    uint32_t structSize;
    uint32_t contractVersion;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    FfxVkXessPath path;
    FfxVkXessQuality quality;
    uint32_t flags;
    VkExtent2D maxInputExtent;
    VkExtent2D maxOutputExtent;
    /* Optional pointer to packed XESSMOD2 model container in host memory */
    const void *modelContainerData;
    size_t modelContainerSizeBytes;
} FfxVkXessCreateInfo;

/* XeSS Per-Frame Dispatch Descriptor */
typedef struct FfxVkXessDispatchInfo {
    uint32_t structSize;
    uint32_t contractVersion;
    VkCommandBuffer commandBuffer;
    FfxVkPortableImage colorIn;
    FfxVkPortableImage velocity;
    FfxVkPortableImage depth;
    FfxVkPortableImage exposure;           /* Optional, 1x1 float */
    FfxVkPortableImage responsiveMask;     /* Optional */
    FfxVkPortableImage colorOut;           /* 1280x720 RGBA16F */
    VkExtent2D inputExtent;
    VkExtent2D outputExtent;
    float jitterOffsetX;
    float jitterOffsetY;
    float sharpness;               /* 0.0 - 1.0 */
    bool resetHistory;
} FfxVkXessDispatchInfo;

/* Contract verification functions */
uint64_t ffxVkXessValidateCreateInfo(
    const FfxVkXessCreateInfo *info,
    bool supportsDot4,
    bool supportsFormatlessWrite);

uint64_t ffxVkXessValidateDispatchInfo(
    const FfxVkXessCreateInfo *createInfo,
    const FfxVkXessDispatchInfo *dispatchInfo);

/* Query canonical layer metadata for the XeSS U-Net */
const FfxVkXessLayerMeta *ffxVkXessGetLayerMeta(FfxVkXessLayerId layerId);

/* Query total weight byte requirement across all 13 layers (253,280 bytes) */
uint32_t ffxVkXessGetTotalWeightBytes(void);

/* Calculate input render resolution given target output resolution and quality */
bool ffxVkXessGetInputResolution(
    FfxVkXessQuality quality,
    VkExtent2D outputExtent,
    VkExtent2D *outInputExtent);

#define FFX_VK_XESS_DESCRIPTOR_SET_COUNT 4

/* XeSS Native Compute Pipeline and Descriptor State */
typedef struct FfxVkXessPipeline {
    VkDevice device;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSets[FFX_VK_XESS_DESCRIPTOR_SET_COUNT];
    VkSampler linearSampler;
    VkShaderModule shaderModule;
    uint32_t currentSetIndex;
} FfxVkXessPipeline;

/* Access the embedded XeSS U-Net reconstruction SPIR-V binary */
const uint32_t *ffxVkXessGetReconstructSpirv(size_t *outWordCount);

/* Create compute pipeline for XeSS U-Net reconstruction pass */
FfxVkPortableResult ffxVkXessCreatePipeline(
    VkDevice device,
    FfxVkXessPipeline *outPipeline);

/* Destroy compute pipeline resources */
void ffxVkXessDestroyPipeline(
    VkDevice device,
    FfxVkXessPipeline *pipeline);

/* Execute XeSS U-Net reconstruction compute dispatch */
FfxVkPortableResult ffxVkXessExecuteDispatch(
    VkCommandBuffer cmdBuf,
    FfxVkXessPipeline *pipeline,
    const FfxVkXessDispatchInfo *dispatchInfo);

#if defined(__cplusplus)
}
#endif

#endif /* FFX_VK_XESS_CONTRACT_H */
