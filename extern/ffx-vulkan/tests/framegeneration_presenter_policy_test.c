#include "ffx_vk_framegeneration_presenter.h"

#include <stdio.h>

static int failures;

typedef struct acquire_sequence_s {
    VkResult results[2];
    uint32_t indices[2];
    unsigned int calls;
} acquire_sequence_t;

static VkResult acquire_sequence(void *user_data, VkSemaphore semaphore,
    uint32_t *out_image_index)
{
    acquire_sequence_t *sequence = user_data;
    const unsigned int call = sequence->calls++;
    (void)semaphore;
    if (call >= 2u)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (sequence->results[call] == VK_SUCCESS ||
        sequence->results[call] == VK_SUBOPTIMAL_KHR)
        *out_image_index = sequence->indices[call];
    return sequence->results[call];
}

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

int main(void)
{
    const VkPresentModeKHR immediate[] = { VK_PRESENT_MODE_FIFO_KHR,
                                           VK_PRESENT_MODE_IMMEDIATE_KHR };
    const VkPresentModeKHR mailbox[] = { VK_PRESENT_MODE_FIFO_KHR,
                                         VK_PRESENT_MODE_MAILBOX_KHR };

    const VkPresentModeKHR relaxed[] = { VK_PRESENT_MODE_FIFO_KHR,
                                          VK_PRESENT_MODE_FIFO_RELAXED_KHR };

    CHECK(ffxVkFrameGenerationSelectPresentMode(true, false, immediate, 2) ==
          VK_PRESENT_MODE_FIFO_KHR);
    CHECK(ffxVkFrameGenerationSelectPresentMode(false, true, immediate, 2) ==
          VK_PRESENT_MODE_FIFO_KHR);
    CHECK(ffxVkFrameGenerationSelectPresentMode(false, false, immediate, 2) ==
          VK_PRESENT_MODE_IMMEDIATE_KHR);
    CHECK(ffxVkFrameGenerationSelectPresentMode(false, false, mailbox, 2) ==
          VK_PRESENT_MODE_MAILBOX_KHR);

    /* Relaxed FIFO / VRR policy tests */
    CHECK(ffxVkFrameGenerationSelectPresentModeEx(
              true, false, FFX_VK_FRAME_GENERATION_SYNC_POLICY_RELAXED_FIFO,
              relaxed, 2) == VK_PRESENT_MODE_FIFO_RELAXED_KHR);
    CHECK(ffxVkFrameGenerationSelectPresentModeEx(
              true, true, FFX_VK_FRAME_GENERATION_SYNC_POLICY_RELAXED_FIFO,
              relaxed, 2) == VK_PRESENT_MODE_FIFO_KHR);
    CHECK(ffxVkFrameGenerationSelectPresentModeEx(
              true, false, FFX_VK_FRAME_GENERATION_SYNC_POLICY_RELAXED_FIFO,
              immediate, 2) == VK_PRESENT_MODE_FIFO_KHR);
    CHECK(ffxVkFrameGenerationSelectPresentModeEx(
              true, false, FFX_VK_FRAME_GENERATION_SYNC_POLICY_STRICT_FIFO,
              relaxed, 2) == VK_PRESENT_MODE_FIFO_KHR);

    CHECK(ffxVkFrameGenerationRequiredImageCount(3, true) == 5);
    CHECK(ffxVkFrameGenerationRequiredImageCount(3, false) == 3);
    CHECK(ffxVkFrameGenerationRequestedImageCount(3, 0, true) == 5);
    CHECK(ffxVkFrameGenerationRequestedImageCount(3, 4, true) == 4);
    CHECK(ffxVkFrameGenerationRequestedImageCount(1, 0, false) == 2);

    CHECK(ffxVkFrameGenerationValidateAcquiredPair(0, 1, 4));
    CHECK(!ffxVkFrameGenerationValidateAcquiredPair(0, 0, 4));
    CHECK(!ffxVkFrameGenerationValidateAcquiredPair(0, 4, 4));
    CHECK(!ffxVkFrameGenerationValidateAcquiredPair(0, 1, 1));

    {
        acquire_sequence_t sequence = {
            .results = { VK_SUCCESS, VK_SUCCESS }, .indices = { 1, 3 },
        };
        FfxVkFrameGenerationAcquiredPair pair;
        FfxVkFrameGenerationPresentPlan plan;
        CHECK(ffxVkFrameGenerationAcquirePair(acquire_sequence, &sequence,
            (VkSemaphore)(uintptr_t)1, (VkSemaphore)(uintptr_t)2, 4, &pair) == VK_SUCCESS);
        CHECK(sequence.calls == 2u);
        CHECK(pair.generatedImageAcquired && pair.realImageAcquired && pair.paired);
        CHECK(pair.generatedImageIndex == 1u && pair.realImageIndex == 3u);
        CHECK(ffxVkFrameGenerationBuildPresentPlan(&pair, true, false, &plan));
        CHECK(plan.slotCount == 2u);
        CHECK(plan.slots[0].imageIndex == 1u &&
              plan.slots[0].imageAvailableSemaphore == (VkSemaphore)(uintptr_t)1 &&
              plan.slots[0].useInterpolatedScene);
        CHECK(plan.slots[1].imageIndex == 3u &&
              plan.slots[1].imageAvailableSemaphore == (VkSemaphore)(uintptr_t)2 &&
              !plan.slots[1].useInterpolatedScene);
        CHECK(ffxVkFrameGenerationBuildPresentPlan(&pair, true, true, &plan));
        CHECK(plan.slotCount == 2u && !plan.slots[0].useInterpolatedScene);
    }
    {
        acquire_sequence_t sequence = {
            .results = { VK_SUBOPTIMAL_KHR, VK_SUCCESS }, .indices = { 0, 2 },
        };
        FfxVkFrameGenerationAcquiredPair pair;
        CHECK(ffxVkFrameGenerationAcquirePair(acquire_sequence, &sequence,
            VK_NULL_HANDLE, VK_NULL_HANDLE, 4, &pair) == VK_SUBOPTIMAL_KHR);
        CHECK(pair.generatedImageAcquired && pair.realImageAcquired && pair.paired);
    }
    {
        acquire_sequence_t sequence = {
            .results = { VK_SUCCESS, VK_NOT_READY }, .indices = { 2, 0 },
        };
        FfxVkFrameGenerationAcquiredPair pair;
        FfxVkFrameGenerationPresentPlan plan;
        CHECK(ffxVkFrameGenerationAcquirePair(acquire_sequence, &sequence,
            VK_NULL_HANDLE, VK_NULL_HANDLE, 4, &pair) == VK_NOT_READY);
        CHECK(sequence.calls == 2u);
        CHECK(pair.generatedImageAcquired && !pair.realImageAcquired && !pair.paired);
        CHECK(pair.generatedImageIndex == 2u);
        CHECK(ffxVkFrameGenerationBuildPresentPlan(&pair, true, false, &plan));
        CHECK(plan.slotCount == 1u && plan.slots[0].imageIndex == 2u &&
              !plan.slots[0].useInterpolatedScene);
    }
    {
        acquire_sequence_t sequence = {
            .results = { VK_ERROR_OUT_OF_DATE_KHR, VK_SUCCESS }, .indices = { 0, 1 },
        };
        FfxVkFrameGenerationAcquiredPair pair;
        CHECK(ffxVkFrameGenerationAcquirePair(acquire_sequence, &sequence,
            VK_NULL_HANDLE, VK_NULL_HANDLE, 4, &pair) == VK_ERROR_OUT_OF_DATE_KHR);
        CHECK(sequence.calls == 1u && !pair.generatedImageAcquired);
    }
    {
        acquire_sequence_t sequence = {
            .results = { VK_SUCCESS, VK_SUCCESS }, .indices = { 1, 1 },
        };
        FfxVkFrameGenerationAcquiredPair pair;
        CHECK(ffxVkFrameGenerationAcquirePair(acquire_sequence, &sequence,
            VK_NULL_HANDLE, VK_NULL_HANDLE, 4, &pair) == VK_ERROR_INITIALIZATION_FAILED);
        CHECK(pair.generatedImageAcquired && pair.realImageAcquired && !pair.paired);
        CHECK(!ffxVkFrameGenerationBuildPresentPlan(&pair, true, false, NULL));
    }

    CHECK(ffxVkFrameGenerationShouldPresentGenerated(true, false));
    CHECK(!ffxVkFrameGenerationShouldPresentGenerated(true, true));
    CHECK(!ffxVkFrameGenerationShouldPresentGenerated(false, false));

    CHECK(ffxVkFrameGenerationTransitionNeedsQuiescence(true, false));
    CHECK(ffxVkFrameGenerationTransitionNeedsQuiescence(false, true));
    CHECK(!ffxVkFrameGenerationTransitionNeedsQuiescence(true, true));
    CHECK(!ffxVkFrameGenerationTransitionNeedsQuiescence(false, false));

    CHECK(ffxVkFrameGenerationRenderFinishedSemaphoreIndex(3, 0, 2) == 6u);
    CHECK(ffxVkFrameGenerationRenderFinishedSemaphoreIndex(3, 1, 2) == 7u);
    CHECK(ffxVkFrameGenerationRenderFinishedSemaphoreIndex(0, 1, 0) == SIZE_MAX);
    CHECK(ffxVkFrameGenerationRenderFinishedSemaphoreIndex(0, 2, 2) == SIZE_MAX);

    /* Multi-frame generation (3x, 4x) tests */
    CHECK(ffxVkFrameGenerationRequiredImageCountMulti(3, 2, true) == 5);
    CHECK(ffxVkFrameGenerationRequiredImageCountMulti(3, 3, true) == 6);
    CHECK(ffxVkFrameGenerationRequiredImageCountMulti(3, 4, true) == 7);
    CHECK(ffxVkFrameGenerationRequiredImageCountMulti(3, 3, false) == 3);

    CHECK(ffxVkFrameGenerationRequestedImageCountMulti(3, 0, 3, true) == 6);
    CHECK(ffxVkFrameGenerationRequestedImageCountMulti(3, 5, 3, true) == 5);

    {
        /* Test 3x plan: 2 generated slots (t=0.333, t=0.667) + 1 real slot (t=1.0) */
        const uint32_t indices3x[] = { 1, 2, 3 };
        const VkSemaphore sems3x[] = { (VkSemaphore)(uintptr_t)10, (VkSemaphore)(uintptr_t)20, (VkSemaphore)(uintptr_t)30 };
        FfxVkFrameGenerationMultiPresentPlan plan3x;

        CHECK(ffxVkFrameGenerationBuildMultiPresentPlan(3, indices3x, sems3x, 3, true, false, &plan3x));
        CHECK(plan3x.slotCount == 3);
        CHECK(plan3x.multiplier == 3);
        CHECK(plan3x.slots[0].imageIndex == 1 && plan3x.slots[0].useInterpolatedScene);
        CHECK(plan3x.slots[1].imageIndex == 2 && plan3x.slots[1].useInterpolatedScene);
        CHECK(plan3x.slots[2].imageIndex == 3 && !plan3x.slots[2].useInterpolatedScene);
        CHECK(plan3x.interpolationPhases[0] > 0.33f && plan3x.interpolationPhases[0] < 0.34f);
        CHECK(plan3x.interpolationPhases[1] > 0.66f && plan3x.interpolationPhases[1] < 0.67f);
        CHECK(plan3x.interpolationPhases[2] == 1.0f);

        /* Test 4x plan: 3 generated slots (t=0.25, 0.5, 0.75) + 1 real slot */
        const uint32_t indices4x[] = { 0, 1, 2, 3 };
        const VkSemaphore sems4x[] = { (VkSemaphore)(uintptr_t)1, (VkSemaphore)(uintptr_t)2,
                                       (VkSemaphore)(uintptr_t)3, (VkSemaphore)(uintptr_t)4 };
        FfxVkFrameGenerationMultiPresentPlan plan4x;

        CHECK(ffxVkFrameGenerationBuildMultiPresentPlan(4, indices4x, sems4x, 4, true, false, &plan4x));
        CHECK(plan4x.slotCount == 4);
        CHECK(plan4x.multiplier == 4);
        CHECK(plan4x.slots[0].imageIndex == 0 && plan4x.slots[0].useInterpolatedScene);
        CHECK(plan4x.slots[1].imageIndex == 1 && plan4x.slots[1].useInterpolatedScene);
        CHECK(plan4x.slots[2].imageIndex == 2 && plan4x.slots[2].useInterpolatedScene);
        CHECK(plan4x.slots[3].imageIndex == 3 && !plan4x.slots[3].useInterpolatedScene);
        CHECK(plan4x.interpolationPhases[0] == 0.25f);
        CHECK(plan4x.interpolationPhases[1] == 0.50f);
        CHECK(plan4x.interpolationPhases[2] == 0.75f);
        CHECK(plan4x.interpolationPhases[3] == 1.0f);

        /* Test duplicate index rejection */
        const uint32_t dupIndices[] = { 1, 2, 1 };
        CHECK(!ffxVkFrameGenerationBuildMultiPresentPlan(3, dupIndices, sems3x, 3, true, false, &plan3x));

        /* Test insufficient acquire fallback */
        CHECK(ffxVkFrameGenerationBuildMultiPresentPlan(3, indices3x, sems3x, 2, true, false, &plan3x));
        CHECK(plan3x.slotCount == 1);
        CHECK(plan3x.multiplier == 1);
        CHECK(!plan3x.slots[0].useInterpolatedScene);
    }

    return failures ? 1 : 0;
}
