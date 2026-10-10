#include "ffx_vk_framegeneration_presenter.h"

#include <limits.h>

VkPresentModeKHR ffxVkFrameGenerationSelectPresentModeEx(
    bool frameGenerationEnabled, bool vsyncEnabled,
    FfxVkFrameGenerationSyncPolicy syncPolicy,
    const VkPresentModeKHR *availableModes, uint32_t availableModeCount)
{
    if (vsyncEnabled)
        return VK_PRESENT_MODE_FIFO_KHR;

    if (frameGenerationEnabled) {
        if (syncPolicy == FFX_VK_FRAME_GENERATION_SYNC_POLICY_RELAXED_FIFO) {
            for (uint32_t i = 0; i < availableModeCount; ++i)
                if (availableModes && availableModes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
                    return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    for (uint32_t i = 0; i < availableModeCount; ++i)
        if (availableModes && availableModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
    return VK_PRESENT_MODE_MAILBOX_KHR;
}

VkPresentModeKHR ffxVkFrameGenerationSelectPresentMode(
    bool frameGenerationEnabled, bool vsyncEnabled,
    const VkPresentModeKHR *availableModes, uint32_t availableModeCount)
{
    return ffxVkFrameGenerationSelectPresentModeEx(
        frameGenerationEnabled, vsyncEnabled,
        FFX_VK_FRAME_GENERATION_SYNC_POLICY_STRICT_FIFO,
        availableModes, availableModeCount);
}

uint32_t ffxVkFrameGenerationRequiredImageCount(uint32_t minImageCount,
                                                bool frameGenerationEnabled)
{
    if (!frameGenerationEnabled)
        return minImageCount > 2u ? minImageCount : 2u;
    if (minImageCount > UINT32_MAX - 2u)
        return 0;
    return minImageCount + 2u;
}

uint32_t ffxVkFrameGenerationRequestedImageCount(uint32_t minImageCount,
                                                 uint32_t maxImageCount,
                                                 bool frameGenerationEnabled)
{
    uint32_t requested = ffxVkFrameGenerationRequiredImageCount(
        minImageCount, frameGenerationEnabled);
    if (!requested)
        return 0;
    if (maxImageCount && requested > maxImageCount)
        requested = maxImageCount;
    return requested < minImageCount ? minImageCount : requested;
}

bool ffxVkFrameGenerationValidateAcquiredPair(uint32_t generatedImageIndex,
                                              uint32_t realImageIndex,
                                              uint32_t swapchainImageCount)
{
    return swapchainImageCount > 1u &&
           generatedImageIndex < swapchainImageCount &&
           realImageIndex < swapchainImageCount &&
           generatedImageIndex != realImageIndex;
}

static bool ffx_vk_framegeneration_acquire_succeeded(VkResult result)
{
    return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
}

VkResult ffxVkFrameGenerationAcquirePair(
    FfxVkFrameGenerationAcquireImageFn acquireImage,
    void *userData,
    VkSemaphore generatedAvailableSemaphore,
    VkSemaphore realAvailableSemaphore,
    uint32_t swapchainImageCount,
    FfxVkFrameGenerationAcquiredPair *outPair)
{
    if (!outPair)
        return VK_ERROR_INITIALIZATION_FAILED;

    *outPair = (FfxVkFrameGenerationAcquiredPair) {
        .generatedAvailableSemaphore = generatedAvailableSemaphore,
        .realAvailableSemaphore = realAvailableSemaphore,
        .generatedAcquireResult = VK_ERROR_INITIALIZATION_FAILED,
        .realAcquireResult = VK_ERROR_INITIALIZATION_FAILED,
    };
    if (!acquireImage || swapchainImageCount < 2u)
        return VK_ERROR_INITIALIZATION_FAILED;

    outPair->generatedAcquireResult = acquireImage(userData,
        generatedAvailableSemaphore, &outPair->generatedImageIndex);
    if (!ffx_vk_framegeneration_acquire_succeeded(
            outPair->generatedAcquireResult))
        return outPair->generatedAcquireResult;
    outPair->generatedImageAcquired = true;

    outPair->realAcquireResult = acquireImage(userData,
        realAvailableSemaphore, &outPair->realImageIndex);
    if (!ffx_vk_framegeneration_acquire_succeeded(outPair->realAcquireResult))
        return outPair->realAcquireResult;
    outPair->realImageAcquired = true;
    outPair->paired = ffxVkFrameGenerationValidateAcquiredPair(
        outPair->generatedImageIndex, outPair->realImageIndex,
        swapchainImageCount);
    if (!outPair->paired)
        return VK_ERROR_INITIALIZATION_FAILED;
    return outPair->generatedAcquireResult == VK_SUBOPTIMAL_KHR ||
           outPair->realAcquireResult == VK_SUBOPTIMAL_KHR
        ? VK_SUBOPTIMAL_KHR : VK_SUCCESS;
}

bool ffxVkFrameGenerationBuildPresentPlan(
    const FfxVkFrameGenerationAcquiredPair *acquiredPair,
    bool interpolationDispatched,
    bool reset,
    FfxVkFrameGenerationPresentPlan *outPlan)
{
    if (!outPlan)
        return false;
    *outPlan = (FfxVkFrameGenerationPresentPlan) { 0 };
    if (!acquiredPair || !acquiredPair->generatedImageAcquired)
        return false;

    outPlan->slots[0] = (FfxVkFrameGenerationPresentSlot) {
        .imageIndex = acquiredPair->generatedImageIndex,
        .imageAvailableSemaphore = acquiredPair->generatedAvailableSemaphore,
        .useInterpolatedScene = acquiredPair->paired &&
            ffxVkFrameGenerationShouldPresentGenerated(interpolationDispatched,
                reset),
    };
    outPlan->slotCount = 1;

    if (!acquiredPair->paired)
        return !acquiredPair->realImageAcquired;
    if (!acquiredPair->realImageAcquired ||
        acquiredPair->generatedImageIndex == acquiredPair->realImageIndex)
        return false;

    outPlan->slots[1] = (FfxVkFrameGenerationPresentSlot) {
        .imageIndex = acquiredPair->realImageIndex,
        .imageAvailableSemaphore = acquiredPair->realAvailableSemaphore,
        .useInterpolatedScene = false,
    };
    outPlan->slotCount = 2;
    return true;
}

bool ffxVkFrameGenerationShouldPresentGenerated(bool interpolationDispatched,
                                                bool reset)
{
    return interpolationDispatched && !reset;
}

bool ffxVkFrameGenerationTransitionNeedsQuiescence(bool wasFrameGenerationActive,
                                                   bool isFrameGenerationActive)
{
    return wasFrameGenerationActive != isFrameGenerationActive;
}

size_t ffxVkFrameGenerationRenderFinishedSemaphoreIndex(uint32_t imageIndex,
                                                         uint32_t gpuIndex,
                                                         uint32_t deviceCount)
{
    if (!deviceCount || gpuIndex >= deviceCount)
        return SIZE_MAX;
    if ((size_t)imageIndex > (SIZE_MAX - (size_t)gpuIndex) / (size_t)deviceCount)
        return SIZE_MAX;
    return (size_t)imageIndex * (size_t)deviceCount + (size_t)gpuIndex;
}

uint32_t ffxVkFrameGenerationRequiredImageCountMulti(uint32_t minImageCount,
                                                     uint32_t multiplier,
                                                     bool frameGenerationEnabled)
{
    if (!frameGenerationEnabled || multiplier <= 1u)
        return minImageCount > 2u ? minImageCount : 2u;
    if (multiplier > FFX_VK_FRAME_GENERATION_MAX_SLOTS)
        multiplier = FFX_VK_FRAME_GENERATION_MAX_SLOTS;
    if (minImageCount > UINT32_MAX - multiplier)
        return 0;
    return minImageCount + multiplier;
}

uint32_t ffxVkFrameGenerationRequestedImageCountMulti(uint32_t minImageCount,
                                                      uint32_t maxImageCount,
                                                      uint32_t multiplier,
                                                      bool frameGenerationEnabled)
{
    uint32_t requested = ffxVkFrameGenerationRequiredImageCountMulti(
        minImageCount, multiplier, frameGenerationEnabled);
    if (!requested)
        return 0;
    if (maxImageCount && requested > maxImageCount)
        requested = maxImageCount;
    return requested < minImageCount ? minImageCount : requested;
}

bool ffxVkFrameGenerationBuildMultiPresentPlan(
    uint32_t multiplier,
    const uint32_t *acquiredImageIndices,
    const VkSemaphore *acquiredSemaphores,
    uint32_t acquiredCount,
    bool interpolationDispatched,
    bool reset,
    FfxVkFrameGenerationMultiPresentPlan *outPlan)
{
    if (!outPlan)
        return false;
    *outPlan = (FfxVkFrameGenerationMultiPresentPlan){ 0 };

    if (multiplier < 2u || multiplier > FFX_VK_FRAME_GENERATION_MAX_SLOTS)
        return false;
    if (!acquiredImageIndices || !acquiredSemaphores || acquiredCount == 0)
        return false;

    /* Fallback if insufficient frames were acquired to form full multiplier plan */
    if (acquiredCount < multiplier) {
        outPlan->slots[0] = (FfxVkFrameGenerationPresentSlot){
            .imageIndex = acquiredImageIndices[0],
            .imageAvailableSemaphore = acquiredSemaphores[0],
            .useInterpolatedScene = false,
        };
        outPlan->interpolationPhases[0] = 1.0f;
        outPlan->slotCount = 1;
        outPlan->multiplier = 1;
        return true;
    }

    /* Verify uniqueness of all acquired image indices */
    for (uint32_t i = 0; i < multiplier; ++i) {
        for (uint32_t j = i + 1; j < multiplier; ++j) {
            if (acquiredImageIndices[i] == acquiredImageIndices[j])
                return false;
        }
    }

    /* Build ordered sequence: (multiplier - 1) generated slots, followed by 1 real slot */
    bool shouldInterpolate = ffxVkFrameGenerationShouldPresentGenerated(interpolationDispatched, reset);
    for (uint32_t i = 0; i < multiplier - 1; ++i) {
        outPlan->slots[i] = (FfxVkFrameGenerationPresentSlot){
            .imageIndex = acquiredImageIndices[i],
            .imageAvailableSemaphore = acquiredSemaphores[i],
            .useInterpolatedScene = shouldInterpolate,
        };
        outPlan->interpolationPhases[i] = (float)(i + 1) / (float)multiplier;
    }

    /* Final real scene slot */
    outPlan->slots[multiplier - 1] = (FfxVkFrameGenerationPresentSlot){
        .imageIndex = acquiredImageIndices[multiplier - 1],
        .imageAvailableSemaphore = acquiredSemaphores[multiplier - 1],
        .useInterpolatedScene = false,
    };
    outPlan->interpolationPhases[multiplier - 1] = 1.0f;
    outPlan->slotCount = multiplier;
    outPlan->multiplier = multiplier;
    return true;
}
