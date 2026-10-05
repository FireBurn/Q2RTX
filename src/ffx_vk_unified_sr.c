/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "ffx_vk_unified_sr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *ffxVkUnifiedSrGetUpscalerName(FfxVkUpscalerType upscaler)
{
    switch (upscaler) {
    case FFX_VK_UPSCALER_AUTO:
        return "Automatic (Hardware Optimal)";
    case FFX_VK_UPSCALER_FSR3:
        return "AMD FidelityFX Super Resolution 3.1";
    case FFX_VK_UPSCALER_FSR4:
        return "AMD FidelityFX Super Resolution 4 (INT8/DOT4)";
    case FFX_VK_UPSCALER_DLSS:
        return "NVIDIA DLSS / AMD d4r (Tensor/WMMA)";
    case FFX_VK_UPSCALER_XESS:
        return "Intel XeSS (DP4a/XMX)";
    default:
        return "Unknown Upscaler";
    }
}

const char *ffxVkUnifiedSrGetGpuTierName(FfxVkGpuTier tier)
{
    switch (tier) {
    case FFX_VK_GPU_TIER_GENERIC_COMPUTE:
        return "Tier 1: Generic Compute (All GPUs)";
    case FFX_VK_GPU_TIER_INT8_DOT4:
        return "Tier 2: Integer Dot Product / DP4a";
    case FFX_VK_GPU_TIER_TENSOR_WMMA:
        return "Tier 3: AMD Wave Matrix (RDNA3/RDNA4 WMMA/FP8)";
    case FFX_VK_GPU_TIER_TENSOR_CORES:
        return "Tier 3: NVIDIA Tensor Cores (RTX 20/30/40/50)";
    default:
        return "Unknown Tier";
    }
}

bool ffxVkUnifiedSrGetRenderResolution(
    FfxVkQualityPreset quality,
    VkExtent2D outputExtent,
    VkExtent2D *outInputExtent)
{
    if (!outInputExtent || !outputExtent.width || !outputExtent.height)
        return false;

    float scale = 1.5f; /* Default Quality */
    switch (quality) {
    case FFX_VK_QUALITY_PRESET_NATIVE:
        scale = 1.0f;
        break;
    case FFX_VK_QUALITY_PRESET_QUALITY:
        scale = 1.5f;
        break;
    case FFX_VK_QUALITY_PRESET_BALANCED:
        scale = 1.7f;
        break;
    case FFX_VK_QUALITY_PRESET_PERFORMANCE:
        scale = 2.0f;
        break;
    case FFX_VK_QUALITY_PRESET_ULTRA_PERFORMANCE:
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

bool ffxVkUnifiedSrQueryGpuCapabilities(
    VkPhysicalDevice physicalDevice,
    FfxVkGpuCapabilities *outCaps)
{
    if (!physicalDevice || !outCaps)
        return false;

    memset(outCaps, 0, sizeof(*outCaps));

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);

    outCaps->vendorId = props.vendorID;
    outCaps->deviceId = props.deviceID;
    snprintf(outCaps->deviceName, sizeof(outCaps->deviceName), "%s", props.deviceName);

    switch (props.vendorID) {
    case FFX_VK_VENDOR_AMD:
        snprintf(outCaps->vendorName, sizeof(outCaps->vendorName), "AMD");
        break;
    case FFX_VK_VENDOR_NVIDIA:
        snprintf(outCaps->vendorName, sizeof(outCaps->vendorName), "NVIDIA");
        break;
    case FFX_VK_VENDOR_INTEL:
        snprintf(outCaps->vendorName, sizeof(outCaps->vendorName), "Intel");
        break;
    case FFX_VK_VENDOR_ARM:
        snprintf(outCaps->vendorName, sizeof(outCaps->vendorName), "ARM");
        break;
    case FFX_VK_VENDOR_QUALCOMM:
        snprintf(outCaps->vendorName, sizeof(outCaps->vendorName), "Qualcomm");
        break;
    default:
        snprintf(outCaps->vendorName, sizeof(outCaps->vendorName), "Unknown");
        break;
    }

    /* Probe features */
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physicalDevice, &features);

    /* Baseline compute capability */
    outCaps->supportsFp16 = true;
    outCaps->supportsFormatlessWrite = features.shaderStorageImageWriteWithoutFormat;

    /* Vendor and architecture heuristics */
    if (outCaps->vendorId == FFX_VK_VENDOR_NVIDIA) {
        /* Check for RTX (Turing, Ampere, Ada Lovelace, Blackwell) */
        if (strstr(outCaps->deviceName, "RTX") || strstr(outCaps->deviceName, "rtx") ||
            strstr(outCaps->deviceName, "TITAN V") || strstr(outCaps->deviceName, "Quadro RTX")) {
            outCaps->supportsTensorCores = true;
            outCaps->supportsInt8 = true;
            outCaps->supportsDot4 = true;
            outCaps->supportsBufferDeviceAddress = true;
            outCaps->tier = FFX_VK_GPU_TIER_TENSOR_CORES;
        } else {
            outCaps->tier = FFX_VK_GPU_TIER_GENERIC_COMPUTE;
        }
    } else if (outCaps->vendorId == FFX_VK_VENDOR_AMD) {
        /* Check for RDNA3 (Navi 3x, gfx110x, RX 7xxx) and RDNA4 (Navi 4x, gfx120x, RX 8xxx) */
        if (strstr(outCaps->deviceName, "RX 7") || strstr(outCaps->deviceName, "RX 8") ||
            strstr(outCaps->deviceName, "Navi 3") || strstr(outCaps->deviceName, "Navi 4") ||
            strstr(outCaps->deviceName, "NAVI3") || strstr(outCaps->deviceName, "NAVI4") ||
            strstr(outCaps->deviceName, "gfx11") || strstr(outCaps->deviceName, "gfx12") ||
            strstr(outCaps->deviceName, "Radeon 7") || strstr(outCaps->deviceName, "Radeon 8")) {
            outCaps->supportsWmma = true;
            outCaps->supportsInt8 = true;
            outCaps->supportsDot4 = true;
            outCaps->supportsBufferDeviceAddress = true;
            outCaps->tier = FFX_VK_GPU_TIER_TENSOR_WMMA;
        } else if (strstr(outCaps->deviceName, "RX 6") || strstr(outCaps->deviceName, "Navi 2") ||
                   strstr(outCaps->deviceName, "NAVI2") || strstr(outCaps->deviceName, "gfx103")) {
            /* RDNA2 (Navi 2x, RX 6xxx) supports INT8/DOT4 but lacks dedicated WMMA */
            outCaps->supportsInt8 = true;
            outCaps->supportsDot4 = true;
            outCaps->supportsBufferDeviceAddress = true;
            outCaps->tier = FFX_VK_GPU_TIER_INT8_DOT4;
        } else {
            outCaps->tier = FFX_VK_GPU_TIER_GENERIC_COMPUTE;
        }
    } else if (outCaps->vendorId == FFX_VK_VENDOR_INTEL) {
        if (strstr(outCaps->deviceName, "Arc") || strstr(outCaps->deviceName, "arc") ||
            strstr(outCaps->deviceName, "Alchemist") || strstr(outCaps->deviceName, "Battlemage") ||
            strstr(outCaps->deviceName, "Xe")) {
            outCaps->supportsInt8 = true;
            outCaps->supportsDot4 = true;
            outCaps->supportsBufferDeviceAddress = true;
            outCaps->tier = FFX_VK_GPU_TIER_INT8_DOT4;
        } else {
            outCaps->tier = FFX_VK_GPU_TIER_GENERIC_COMPUTE;
        }
    } else {
        outCaps->tier = FFX_VK_GPU_TIER_GENERIC_COMPUTE;
    }

    /* Bitmask of supported upscalers */
    outCaps->supportedUpscalersBitmask = (1u << FFX_VK_UPSCALER_FSR3); /* FSR 3 is universally supported */

    if (outCaps->supportsDot4 || outCaps->tier >= FFX_VK_GPU_TIER_INT8_DOT4) {
        outCaps->supportedUpscalersBitmask |= (1u << FFX_VK_UPSCALER_FSR4);
        outCaps->supportedUpscalersBitmask |= (1u << FFX_VK_UPSCALER_XESS); /* XeSS DP4a path */
    }

    if (outCaps->supportsTensorCores || outCaps->supportsWmma) {
        outCaps->supportedUpscalersBitmask |= (1u << FFX_VK_UPSCALER_DLSS); /* NVNGX or d4r */
    }

    /* Recommended upscaler choice */
    if (outCaps->supportsTensorCores) {
        outCaps->recommendedUpscaler = FFX_VK_UPSCALER_DLSS;
    } else if (outCaps->supportsWmma) {
        outCaps->recommendedUpscaler = FFX_VK_UPSCALER_DLSS; /* d4r on RDNA3/RDNA4 */
    } else if (outCaps->tier == FFX_VK_GPU_TIER_INT8_DOT4) {
        outCaps->recommendedUpscaler = FFX_VK_UPSCALER_FSR4; /* FSR4 INT8 on RDNA2 / Intel Arc */
    } else {
        outCaps->recommendedUpscaler = FFX_VK_UPSCALER_FSR3; /* FSR 3.1 on legacy GPUs */
    }

    return true;
}

FfxVkUpscalerType ffxVkUnifiedSrResolveUpscaler(
    FfxVkUpscalerType preferred,
    const FfxVkGpuCapabilities *caps)
{
    if (!caps)
        return FFX_VK_UPSCALER_FSR3;

    if (preferred == FFX_VK_UPSCALER_AUTO)
        return caps->recommendedUpscaler;

    /* If preferred upscaler is supported on this GPU, honor it */
    if (caps->supportedUpscalersBitmask & (1u << preferred))
        return preferred;

    /* Fallback chain: DLSS -> FSR4 -> XeSS -> FSR3 */
    if (preferred == FFX_VK_UPSCALER_DLSS) {
        if (caps->supportedUpscalersBitmask & (1u << FFX_VK_UPSCALER_FSR4))
            return FFX_VK_UPSCALER_FSR4;
        if (caps->supportedUpscalersBitmask & (1u << FFX_VK_UPSCALER_XESS))
            return FFX_VK_UPSCALER_XESS;
    } else if (preferred == FFX_VK_UPSCALER_FSR4 || preferred == FFX_VK_UPSCALER_XESS) {
        if (caps->supportedUpscalersBitmask & (1u << FFX_VK_UPSCALER_FSR4))
            return FFX_VK_UPSCALER_FSR4;
    }

    return FFX_VK_UPSCALER_FSR3;
}

VkResult ffxVkUnifiedSrCreate(
    FfxVkUnifiedSrContext *ctx,
    const FfxVkUnifiedSrCreateInfo *info)
{
    if (!ctx || !info)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (info->structSize != sizeof(FfxVkUnifiedSrCreateInfo) ||
        info->version != FFX_VK_UNIFIED_SR_VERSION)
        return VK_ERROR_INITIALIZATION_FAILED;

    memset(ctx, 0, sizeof(*ctx));
    ctx->createInfo = *info;

    /* Query GPU capabilities */
    if (!ffxVkUnifiedSrQueryGpuCapabilities(info->physicalDevice, &ctx->capabilities)) {
        /* Fallback synthetic capabilities if physical device properties query fails */
        ctx->capabilities.tier = FFX_VK_GPU_TIER_GENERIC_COMPUTE;
        ctx->capabilities.supportedUpscalersBitmask = (1u << FFX_VK_UPSCALER_FSR3);
        ctx->capabilities.recommendedUpscaler = FFX_VK_UPSCALER_FSR3;
    }

    ctx->activeUpscaler = ffxVkUnifiedSrResolveUpscaler(info->preferredUpscaler, &ctx->capabilities);
    ctx->initialized = 1;

    return VK_SUCCESS;
}

VkResult ffxVkUnifiedSrDispatch(
    FfxVkUnifiedSrContext *ctx,
    const FfxVkUnifiedSrDispatchInfo *dispatchInfo)
{
    if (!ctx || !ctx->initialized || !dispatchInfo)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (dispatchInfo->structSize != sizeof(FfxVkUnifiedSrDispatchInfo) ||
        dispatchInfo->version != FFX_VK_UNIFIED_SR_VERSION)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (!dispatchInfo->commandBuffer)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (!dispatchInfo->inputExtent.width || !dispatchInfo->inputExtent.height ||
        !dispatchInfo->outputExtent.width || !dispatchInfo->outputExtent.height)
        return VK_ERROR_INITIALIZATION_FAILED;

    /* In a live engine, this dispatches the active backend (FSR3, FSR4, DLSS/d4r, or XeSS) */
    return VK_SUCCESS;
}

void ffxVkUnifiedSrDestroy(FfxVkUnifiedSrContext *ctx)
{
    if (!ctx || !ctx->initialized)
        return;

    ctx->initialized = 0;
    ctx->backendContext = NULL;
}
