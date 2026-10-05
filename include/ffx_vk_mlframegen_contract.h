/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef FFX_VK_MLFRAMEGEN_CONTRACT_H
#define FFX_VK_MLFRAMEGEN_CONTRACT_H

#include "ffx_vk_portable.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Provider-neutral Vulkan contract for AMD ML Frame Generation 4.0.1 (FSR 4 MLFG).
 * It models the neural bidirectional motion estimation, multi-frame interpolation
 * ratios (2x, 3x, 4x), hardware execution tiers (INT8, FP8/WMMA), and WSI
 * presentation synchronization before dispatch.
 * It is easily usable by any Vulkan application without requiring Windows 11 or DX12. */
#define FFX_VK_MLFRAMEGEN_CONTRACT_VERSION 1u

/* ML Frame Generation Interpolation Multipliers */
typedef enum FfxVkMlFrameGenMode {
    FFX_VK_MLFRAMEGEN_MODE_2X = 0,               /* 1 generated frame per real frame (t = 0.5) */
    FFX_VK_MLFRAMEGEN_MODE_3X = 1,               /* 2 generated frames per real frame (t = 0.333, 0.667) */
    FFX_VK_MLFRAMEGEN_MODE_4X = 2                /* 3 generated frames per real frame (t = 0.25, 0.5, 0.75) */
} FfxVkMlFrameGenMode;

/* ML Frame Generation Hardware Tiers */
typedef enum FfxVkMlFrameGenTier {
    FFX_VK_MLFRAMEGEN_TIER_GENERIC_COMPUTE = 0,  /* Standard compute shader fallback */
    FFX_VK_MLFRAMEGEN_TIER_INT8_DOT4 = 1,        /* INT8 DP4a dot-product path (RDNA2+, Turing+, Arc) */
    FFX_VK_MLFRAMEGEN_TIER_FP8_WMMA = 2          /* Native FP8 / WMMA tensor path (RDNA4, Blackwell) */
} FfxVkMlFrameGenTier;

/* Context creation flags */
typedef enum FfxVkMlFrameGenFlags {
    FFX_VK_MLFRAMEGEN_FLAG_NONE = 0,
    FFX_VK_MLFRAMEGEN_FLAG_HDR_COLOR = 1u << 0,
    FFX_VK_MLFRAMEGEN_FLAG_DEPTH_INVERTED = 1u << 1,
    FFX_VK_MLFRAMEGEN_FLAG_ALLOW_ASYNC_COMPUTE = 1u << 2,
    FFX_VK_MLFRAMEGEN_FLAG_LOW_LATENCY = 1u << 3
} FfxVkMlFrameGenFlags;

/* Validation issue bitmask */
typedef enum FfxVkMlFrameGenValidationIssueBits {
    FFX_VK_MLFG_VALIDATION_NONE = 0,
    FFX_VK_MLFG_VALIDATION_STRUCT_SIZE = 1ull << 0,
    FFX_VK_MLFG_VALIDATION_CONTRACT_VERSION = 1ull << 1,
    FFX_VK_MLFG_VALIDATION_ZERO_EXTENT = 1ull << 2,
    FFX_VK_MLFG_VALIDATION_EXTENT_MISMATCH = 1ull << 3,
    FFX_VK_MLFG_VALIDATION_INVALID_FACTOR = 1ull << 4,
    FFX_VK_MLFG_VALIDATION_IMAGE_HANDLE = 1ull << 5,
    FFX_VK_MLFG_VALIDATION_IMAGE_USAGE = 1ull << 6,
    FFX_VK_MLFG_VALIDATION_IMAGE_FORMAT = 1ull << 7,
    FFX_VK_MLFG_VALIDATION_ALIAS_CONFLICT = 1ull << 8,
    FFX_VK_MLFG_VALIDATION_MISSING_INT8 = 1ull << 9,
    FFX_VK_MLFG_VALIDATION_MISSING_WMMA = 1ull << 10,
    FFX_VK_MLFG_VALIDATION_NONFINITE_METADATA = 1ull << 11
} FfxVkMlFrameGenValidationIssueBits;

/* Context Creation Descriptor */
typedef struct FfxVkMlFrameGenCreateInfo {
    uint32_t structSize;
    uint32_t contractVersion;
    FfxVkMlFrameGenMode mode;
    FfxVkMlFrameGenTier tier;
    uint32_t flags;
    FfxVkPortableExtent2D displaySize;
    /* Optional pointer to packed MLFG 4.0.1 neural model container in host memory */
    const void *modelContainerData;
    size_t modelContainerSizeBytes;
} FfxVkMlFrameGenCreateInfo;

/* Per-Frame Interpolation Dispatch Descriptor */
typedef struct FfxVkMlFrameGenDispatchInfo {
    uint32_t structSize;
    uint32_t contractVersion;
    VkCommandBuffer commandBuffer;

    /* Current and previous real frames (display-resolution HDR scene color) */
    FfxVkPortableImage currentRealColor;
    FfxVkPortableImage previousRealColor;

    /* Temporal motion and depth */
    FfxVkPortableImage motionVectors;
    FfxVkPortableImage depth;

    /* Output target for generated interpolated frame */
    FfxVkPortableImage outputGeneratedColor;

    /* Fractional interpolation phase in range (0.0, 1.0) */
    float interpolationFactor;

    /* Frame sequencing & camera metadata */
    uint64_t frameId;
    VkBool32 resetHistory;
    FfxVkPortableFloat2 jitterOffset;
    FfxVkPortableFloat3 cameraPositionDelta;
    float frameTimeMs;
} FfxVkMlFrameGenDispatchInfo;

/* Contract verification functions */
uint64_t ffxVkMlFrameGenValidateCreateInfo(
    const FfxVkMlFrameGenCreateInfo *info,
    bool supportsInt8,
    bool supportsWmma);

uint64_t ffxVkMlFrameGenValidateDispatchInfo(
    const FfxVkMlFrameGenCreateInfo *createInfo,
    const FfxVkMlFrameGenDispatchInfo *dispatchInfo);

/* Helpers to compute interpolation factor for slot index */
float ffxVkMlFrameGenGetInterpolationFactor(FfxVkMlFrameGenMode mode, uint32_t slotIndex);

#if defined(__cplusplus)
}
#endif

#endif /* FFX_VK_MLFRAMEGEN_CONTRACT_H */
