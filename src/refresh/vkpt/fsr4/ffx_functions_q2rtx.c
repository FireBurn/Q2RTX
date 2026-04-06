/*
 * ffx_functions_q2rtx.c
 *
 * Implements the four FFX API entry points (ffxCreateContext, ffxDestroyContext,
 * ffxQuery, ffxDispatch) without linking AMD's signed DLL.
 *
 * Instead of going through the provider/DLL layer, all calls route through
 * g_vkBackendOverride — the FfxInterface set up by fsr.c before every call.
 *
 * The FSR4 upscaler context is modelled as a thin wrapper that stores the
 * FfxInterface pointer and the per-frame cbuffer data.  The 14-pass compute
 * dispatch is issued directly by calling fpExecuteGpuJobs via the backend.
 *
 * Jitter is computed with a Halton[2,3] sequence matching what the AMD SDK
 * generates (see ffxQueryDescUpscaleGetJitterOffset handling below).
 */

#include "ffx_types_q2rtx.h"
#include "ffx_fsr4_vk.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ── global override pointer (defined in fsr.c) ──────────────────────────── */
extern FfxInterface *g_vkBackendOverride;

/* ── internal context ────────────────────────────────────────────────────── */

typedef struct Fsr4Context {
    FfxInterface        iface;          /* copy of backend interface        */
    FfxUInt32           effectId;       /* effectContextId from backend     */
    FfxApiDimensions2D  maxRenderSize;
    FfxApiDimensions2D  maxUpscaleSize;
    uint32_t            flags;
    bool                valid;

    /* ── Internal resources (created once, persist across frames) ────────── */
    FfxResourceInternal ri_scratch;       /* Large SSBO for NN activations   */
    FfxResourceInternal ri_history;       /* Previous upscaled output (dpy)  */
    FfxResourceInternal ri_recurrent[2];  /* Recurrent feature maps (PP)     */
    FfxResourceInternal ri_reprojected;   /* Pre-pass → post-pass bridge     */
    FfxResourceInternal ri_exposure;      /* Auto-exposure (1×1 R32F)        */
    int                 recurrent_idx;    /* 0 or 1: which is "current read" */

    bool                resources_ok;     /* true if internal resources exist */
} Fsr4Context;

/* ── Halton sequence helper ──────────────────────────────────────────────── */

static float halton(int base, int index)
{
    float f = 1.0f, r = 0.0f;
    int   i = index;
    while (i > 0) {
        f /= (float)base;
        r += f * (float)(i % base);
        i /= base;
    }
    return r;
}

static int jitter_phase_count(uint32_t display_w, uint32_t render_w)
{
    /* FSR4 formula: ceil(8 * (display_w / render_w)^2), clamped to sensible range */
    if (render_w == 0) return 8;
    float ratio = (float)display_w / (float)render_w;
    int   count = (int)ceilf(8.0f * ratio * ratio);
    if (count < 1)   count = 1;
    if (count > 256) count = 256;
    return count;
}

/* ── Internal resource creation helper ───────────────────────────────────── */

static FfxErrorCode create_internal_tex(Fsr4Context *c, uint32_t w, uint32_t h,
    FfxSurfaceFormat fmt, FfxResourceInternal *out)
{
    FfxCreateResourceDescription desc;
    memset(&desc, 0, sizeof(desc));
    desc.type    = FFX_RESOURCE_TYPE_TEXTURE2D;
    desc.format  = fmt;
    desc.width   = w;
    desc.height  = h;
    return c->iface.fpCreateResource(&c->iface, &desc, c->effectId, out);
}

static FfxErrorCode create_internal_buf(Fsr4Context *c, uint32_t size_bytes,
    FfxResourceInternal *out)
{
    FfxCreateResourceDescription desc;
    memset(&desc, 0, sizeof(desc));
    desc.type         = FFX_RESOURCE_TYPE_BUFFER;
    desc.format       = FFX_SURFACE_FORMAT_UNKNOWN;
    desc.width        = size_bytes;  /* buffer "width" = size in bytes */
    desc.height       = 0;
    desc.initDataSize = size_bytes;  /* allocate at least this much */
    desc.initData     = NULL;        /* zero-fill is fine (no upload) */
    return c->iface.fpCreateResource(&c->iface, &desc, c->effectId, out);
}

static FfxErrorCode create_internal_resources(Fsr4Context *c)
{
    FfxErrorCode err;
    uint32_t rw = c->maxRenderSize.width;
    uint32_t rh = c->maxRenderSize.height;
    uint32_t dw = c->maxUpscaleSize.width;
    uint32_t dh = c->maxUpscaleSize.height;

    /* Scratch buffer for NN activations.
       Size estimate: largest model pass needs ~(width/2 * height/2 * 64_channels * 1_byte)
       for the bottleneck.  64 MB covers up to 4K input comfortably. */
    uint32_t scratch_bytes = 64u * 1024u * 1024u;
    /* Scale by resolution: use 32 MB for ≤1080p, 64 MB for ≤2160p, 128 MB for 4320p */
    if (rh <= 1080) scratch_bytes = 32u * 1024u * 1024u;
    else if (rh > 2160) scratch_bytes = 128u * 1024u * 1024u;

    err = create_internal_buf(c, scratch_bytes, &c->ri_scratch);
    if (err != FFX_OK) return err;

    /* History color — previous frame's upscaled output (display resolution) */
    err = create_internal_tex(c, dw, dh, FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                              &c->ri_history);
    if (err != FFX_OK) return err;

    /* Recurrent feature maps — two copies for ping-pong (render resolution)
       The post-pass shader writes rw_recurrent_0 with OpImage format Rgba32f,
       so we must use R32G32B32A32_FLOAT. */
    for (int i = 0; i < 2; i++) {
        err = create_internal_tex(c, rw, rh, FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT,
                                  &c->ri_recurrent[i]);
        if (err != FFX_OK) return err;
    }

    /* Reprojected color — pre-pass output (render resolution) */
    err = create_internal_tex(c, rw, rh, FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                              &c->ri_reprojected);
    if (err != FFX_OK) return err;

    /* Auto-exposure — 1×1 texture, initialised to 1.0 by the GPU
       (the SPD pass will fill it when enabled; for now 1.0 = no exposure adjust) */
    err = create_internal_tex(c, 1, 1, FFX_SURFACE_FORMAT_R32_FLOAT,
                              &c->ri_exposure);
    if (err != FFX_OK) return err;

    c->recurrent_idx = 0;
    c->resources_ok  = true;
    return FFX_OK;
}

/* ── ffxCreateContext ─────────────────────────────────────────────────────── */

ffxReturnCode_t ffxCreateContext(ffxContext *ctx,
    ffxCreateContextDescHeader *desc,
    const ffxAllocationCallbacks *mem)
{
    (void)mem;

    if (!ctx || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    if (!g_vkBackendOverride) return FFX_API_RETURN_ERROR;

    /* Find the upscale descriptor in the linked list */
    ffxCreateContextDescUpscale *upscaleDesc = NULL;
    for (ffxCreateContextDescHeader *h = desc; h; h = h->pNext) {
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
            upscaleDesc = (ffxCreateContextDescUpscale *)h;
            break;
        }
    }
    if (!upscaleDesc) return FFX_API_RETURN_ERROR_PARAMETER;

    Fsr4Context *c = (Fsr4Context *)calloc(1, sizeof(Fsr4Context));
    if (!c) return FFX_API_RETURN_ERROR;

    /* Copy the interface so we can use it after g_vkBackendOverride is cleared */
    c->iface          = *g_vkBackendOverride;
    c->maxRenderSize  = upscaleDesc->maxRenderSize;
    c->maxUpscaleSize = upscaleDesc->maxUpscaleSize;
    c->flags          = upscaleDesc->flags;
    c->valid          = true;

    /* Initialise the Vulkan backend context (creates pipelines) */
    FfxEffectBindlessConfig bindlessCfg = {0, 0, 0, 1};
    FfxErrorCode err = c->iface.fpCreateBackendContext(
        &c->iface, FFX_EFFECT_FSR4UPSCALER, &bindlessCfg, &c->effectId);
    if (err != FFX_OK) {
        free(c);
        return FFX_API_RETURN_ERROR;
    }

    /* Create internal GPU resources (scratch, history, recurrent, etc.) */
    err = create_internal_resources(c);
    if (err != FFX_OK) {
        c->iface.fpDestroyBackendContext(&c->iface, c->effectId);
        free(c);
        return FFX_API_RETURN_ERROR;
    }

    *ctx = (ffxContext)c;
    return FFX_API_RETURN_OK;
}

/* ── ffxDestroyContext ────────────────────────────────────────────────────── */

ffxReturnCode_t ffxDestroyContext(ffxContext *ctx,
    const ffxAllocationCallbacks *mem)
{
    (void)mem;
    if (!ctx || !*ctx) return FFX_API_RETURN_ERROR_PARAMETER;

    Fsr4Context *c = (Fsr4Context *)(*ctx);
    if (c->valid)
        c->iface.fpDestroyBackendContext(&c->iface, c->effectId);

    free(c);
    *ctx = NULL;
    return FFX_API_RETURN_OK;
}

/* ── ffxQuery ─────────────────────────────────────────────────────────────── */

ffxReturnCode_t ffxQuery(ffxContext *ctx, ffxQueryDescHeader *desc)
{
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;

    Fsr4Context *c = (ctx && *ctx) ? (Fsr4Context *)(*ctx) : NULL;

    for (ffxQueryDescHeader *h = desc; h; h = h->pNext) {

        if (h->type == FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_JITTER_PHASE_COUNT) {
            ffxQueryDescUpscaleGetJitterPhaseCount *q =
                (ffxQueryDescUpscaleGetJitterPhaseCount *)h;
            if (q->pOutPhaseCount)
                *q->pOutPhaseCount = jitter_phase_count(
                    q->displayWidth, q->renderWidth);
            continue;
        }

        if (h->type == FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_JITTER_OFFSET) {
            ffxQueryDescUpscaleGetJitterOffset *q =
                (ffxQueryDescUpscaleGetJitterOffset *)h;
            if (q->pOutX && q->pOutY) {
                /* Halton[2,3], 1-indexed, shifted to [-0.5, 0.5] */
                int idx = (q->index % q->phaseCount) + 1;
                *q->pOutX = halton(2, idx) - 0.5f;
                *q->pOutY = halton(3, idx) - 0.5f;
            }
            continue;
        }
    }

    (void)c;
    return FFX_API_RETURN_OK;
}

/* ── ffxDispatch ─────────────────────────────────────────────────────────── */

/*
 * Dispatch a single upscale frame.
 *
 * The FSR4 INT8 model runs as 14 sequential compute passes (pre + 12 model
 * passes + post), optionally followed by RCAS sharpening.  The backend
 * (ffx_fsr4_vk.c) already has the VkPipeline objects built for each pass;
 * we just need to schedule the right dispatch jobs.
 *
 * Resource → register → Vulkan binding mapping:
 *
 *   FULL passes (pre/post) compiled with -fvk-t-shift 0 -fvk-u-shift 21 -fvk-b-shift 43:
 *     srvTextures[i]  →  binding i            (COMBINED_IMAGE_SAMPLER)
 *     uavTextures[i]  →  binding 21+i         (STORAGE_IMAGE)
 *     cbs[0]          →  binding 43            (UNIFORM_BUFFER)
 *     uavBuffers (scratch) → binding 32        (STORAGE_BUFFER, via bindingIndex)
 *
 *   writeFullDs iterates srvTextures[0..srvTextureCount-1] and writes each
 *   at Vulkan binding = SLOT_SRV_TEX_BASE + loop_index.  So srvTextures[]
 *   MUST be indexed by t-register number: srvTextures[3] → t3 → binding 3.
 *   Unused slots must have internalIndex = UINT32_MAX so writeFullDs skips them.
 *
 *   Same for uavTextures: uavTextures[i] → binding 21+i.  So uavTextures[3]
 *   → u3 → binding 24.  Pad unused slots with UINT32_MAX.
 *
 *   MODEL passes compiled with -fvk-t-shift 0 -fvk-u-shift 2:
 *     srvBuffers[0] → binding 0 (t0 = input)
 *     srvBuffers[1] → binding 1 (t1 = weights/initializer)
 *     uavBuffers[0] → binding 2 (u0 = output)
 *     uavBuffers[1] → binding 3 (u1 = scratch)
 */

/* Dispatch sizes for the 12 inner model passes (16x16 thread groups) */
static void compute_model_dispatch(uint32_t w, uint32_t h,
    uint32_t pass, /* 1..12 */
    uint32_t *dx, uint32_t *dy)
{
    /* The inner model passes operate on 1/2-res (encoder) or full-res
       (decoder) tiles.  Passes 1-6 = encoder (half-res), 7-12 = decoder
       (full-res).  Thread group size is 32x1 for inner passes. */
    if (pass <= 6) {
        *dx = (w/2 + 31) / 32;
        *dy = (h/2 + 1)  / 2;
    } else {
        *dx = (w + 31) / 32;
        *dy = (h + 1)  / 2;
    }
    if (*dx == 0) *dx = 1;
    if (*dy == 0) *dy = 1;
}

/* Register one FfxApiResource and return the internal index */
static FfxResourceInternal reg_resource(Fsr4Context *c,
    const FfxApiResource *res, FfxUInt32 effectId)
{
    FfxResourceInternal out = {UINT32_MAX};
    if (res && res->resource)
        c->iface.fpRegisterResource(&c->iface, res, effectId, &out);
    return out;
}

/* Sentinel for "no resource at this slot" — writeFullDs/writeModelDs skip these */
static const FfxResourceInternal RI_NONE = { UINT32_MAX };

ffxReturnCode_t ffxDispatch(ffxContext *ctx, const ffxDispatchDescHeader *desc)
{
    if (!ctx || !*ctx || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    Fsr4Context *c = (Fsr4Context *)(*ctx);
    if (!c->valid || !c->resources_ok) return FFX_API_RETURN_ERROR;

    /* Find the upscale dispatch descriptor */
    const ffxDispatchDescUpscale *d = NULL;
    for (const ffxDispatchDescHeader *h = desc; h; h = h->pNext) {
        if (h->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
            d = (const ffxDispatchDescUpscale *)h;
            break;
        }
    }
    if (!d) return FFX_API_RETURN_ERROR_PARAMETER;

    VkCommandBuffer cmd = (VkCommandBuffer)d->commandList;
    uint32_t rw = d->renderSize.width;
    uint32_t rh = d->renderSize.height;
    uint32_t dw = d->upscaleSize.width  ? d->upscaleSize.width  : c->maxUpscaleSize.width;
    uint32_t dh = d->upscaleSize.height ? d->upscaleSize.height : c->maxUpscaleSize.height;

    /* Register external resources for this frame */
    FfxResourceInternal ri_color  = reg_resource(c, &d->color,         c->effectId);
    FfxResourceInternal ri_depth  = reg_resource(c, &d->depth,         c->effectId);
    FfxResourceInternal ri_mv     = reg_resource(c, &d->motionVectors, c->effectId);
    FfxResourceInternal ri_output = reg_resource(c, &d->output,        c->effectId);

    /* Shorthand for internal resources */
    FfxResourceInternal ri_history     = c->ri_history;
    FfxResourceInternal ri_recurrent_r = c->ri_recurrent[c->recurrent_idx];      /* read  */
    FfxResourceInternal ri_recurrent_w = c->ri_recurrent[c->recurrent_idx ^ 1];  /* write */
    FfxResourceInternal ri_reprojected = c->ri_reprojected;
    FfxResourceInternal ri_exposure    = c->ri_exposure;
    FfxResourceInternal ri_scratch     = c->ri_scratch;

    /* ── HACK: clear history & reprojected to zero each frame ───────────────
     * Background:
     *   - ri_history and ri_reprojected are allocated at display resolution
     *     (dw x dh = e.g. 2560x1440 at 50% upscale).
     *   - Empirically we see that the FSR4 pre-pass only writes these in the
     *     render-resolution region (upper-left rw x rh = 1280x720), leaving
     *     the other ~75% of pixels with whatever content they had before.
     *   - The next frame's pre-pass then samples full-display UVs from those
     *     buffers and reads stale old-frame data outside the render region,
     *     which blends into the output and produces a sharp rectangular
     *     artifact at (rw, rh) of the display.
     *
     * Without access to the SDK's HLSL sources we can't fix the write bounds
     * in the shader.  Cheap workaround: clear these buffers to zero each
     * frame before the pre-pass runs.  The shader will then see zero instead
     * of stale data outside the render region, and either blend less history
     * there or reject it as disocclusion.
     *
     * Trade-off: on pans/camera motion this effectively disables history
     * reprojection in the newly-revealed outer band, so we lose some
     * temporal quality there.  The rectangular artifact disappears.
     */
    {
        VkImage img_history     = ffxFsr4VkGetImage(&c->iface, ri_history);
        VkImage img_reprojected = ffxFsr4VkGetImage(&c->iface, ri_reprojected);

        VkClearColorValue clear = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange sr = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1
        };

        /* For each image, pick the correct oldLayout:
         *   - If the backend hasn't transitioned it yet (needsInit=1),
         *     it's in UNDEFINED.  We transition UNDEFINED→TRANSFER_DST
         *     and mark it initialized so ExecuteGpuJobs doesn't re-transition.
         *   - Otherwise it's in GENERAL (from the previous frame's compute
         *     work, sync'd via queue submit semaphore), and we transition
         *     GENERAL→TRANSFER_DST. */
        VkImageMemoryBarrier bar[2];
        memset(bar, 0, sizeof(bar));
        uint32_t bc = 0;

        #define ADD_BAR(img, ri)                                                  \
            do {                                                                   \
                if ((img)) {                                                       \
                    bar[bc].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;       \
                    int needs_init = ffxFsr4VkNeedsInitialTransition(&c->iface, (ri)); \
                    bar[bc].oldLayout = needs_init                                 \
                        ? VK_IMAGE_LAYOUT_UNDEFINED                                \
                        : VK_IMAGE_LAYOUT_GENERAL;                                 \
                    bar[bc].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;      \
                    bar[bc].srcAccessMask = needs_init ? 0                          \
                        : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT); \
                    bar[bc].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;          \
                    bar[bc].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;        \
                    bar[bc].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;        \
                    bar[bc].image = (img);                                         \
                    bar[bc].subresourceRange = sr;                                 \
                    if (needs_init) ffxFsr4VkMarkImageInitialized(&c->iface, (ri));\
                    bc++;                                                          \
                }                                                                   \
            } while(0)

        ADD_BAR(img_history,     ri_history);
        ADD_BAR(img_reprojected, ri_reprojected);
        #undef ADD_BAR

        if (bc > 0) {
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, NULL, 0, NULL,
                bc, bar);

            if (img_history)
                vkCmdClearColorImage(cmd, img_history,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &sr);
            if (img_reprojected)
                vkCmdClearColorImage(cmd, img_reprojected,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &sr);

            /* Transition back to GENERAL for the compute passes */
            for (uint32_t i = 0; i < bc; i++) {
                bar[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                bar[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                bar[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                bar[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, NULL, 0, NULL,
                bc, bar);
        }
    }

    /* Build the OptimizedConstants cbuffer (88 bytes) */
    struct {
        float    inv_size[2];
        float    scale[2];
        float    inv_scale[2];
        float    jitter[2];
        float    mv_scale[2];
        float    tex_size[2];
        float    max_renderSize[2];
        float    fMotionVectorJitterCancellation[2];
        uint32_t width;
        uint32_t height;
        uint32_t reset;
        uint32_t width_lr;
        uint32_t height_lr;
        float    preExposure;
        float    previous_preExposure;
        uint32_t rcas_enabled;
        float    rcas_sharpness;
        float    _pad1;
    } cb;
    memset(&cb, 0, sizeof(cb));
    cb.inv_size[0]    = 1.0f / (float)dw;
    cb.inv_size[1]    = 1.0f / (float)dh;
    cb.scale[0]       = (float)dw / (float)rw;
    cb.scale[1]       = (float)dh / (float)rh;
    cb.inv_scale[0]   = (float)rw / (float)dw;
    cb.inv_scale[1]   = (float)rh / (float)dh;
    cb.jitter[0]      = d->jitterOffset.x;
    cb.jitter[1]      = d->jitterOffset.y;
    cb.mv_scale[0]    = d->motionVectorScale.x;
    cb.mv_scale[1]    = d->motionVectorScale.y;
    cb.tex_size[0]    = (float)dw;
    cb.tex_size[1]    = (float)dh;
    cb.max_renderSize[0] = (float)rw;
    cb.max_renderSize[1] = (float)rh;
    cb.width          = dw;
    cb.height         = dh;
    cb.reset          = d->reset ? 1u : 0u;
    cb.width_lr       = rw;
    cb.height_lr      = rh;
    cb.preExposure    = (d->preExposure > 0.0f) ? d->preExposure : 1.0f;
    cb.previous_preExposure = cb.preExposure;
    cb.rcas_enabled   = d->enableSharpening ? 1u : 0u;
    cb.rcas_sharpness = d->sharpness;

    /* Stage the cbuffer */
    FfxConstantBuffer cbHandle = {{0}};
    c->iface.fpStageConstantBufferDataFunc(&c->iface, &cb, (FfxUInt32)sizeof(cb), &cbHandle);

    /* ── Schedule all 14 compute passes ──────────────────────────────────── */
    for (int pass = 0; pass <= 13; pass++) {

        FfxGpuJobDescription job;
        memset(&job, 0, sizeof(job));
        job.jobType = FFX_GPU_JOB_COMPUTE;

        FfxComputeJobDescription *cj = &job.computeJobDescriptor;

        /* Pipeline: encode pass index as the pipeline void* */
        cj->pipeline.pipeline = (FfxPipeline)(uintptr_t)(uint32_t)pass;

        if (pass == 0) {
            /* ── PRE-PASS ────────────────────────────────────────────────
             * SRV textures indexed by t-register (writeFullDs uses SLOT_SRV_TEX_BASE+i):
             *   t0 = r_history_color     (previous upscaled output)
             *   t1 = r_velocity          (motion vectors)
             *   t3 = r_input_color       (current frame)
             *   t4 = r_recurrent_0       (recurrent feature map, read)
             *   t6 = r_input_exposure    (exposure)
             * UAV textures indexed by u-register (writeFullDs uses SLOT_UAV_BASE+i):
             *   u3 = rw_reprojected_color
             * UAV buffers (via bindingIndex):
             *   u11 = ScratchBuffer
             * CBV:
             *   b0 = OptimizedConstants
             */
            cj->pipeline.srvTextureCount = 7;  /* t0..t6, pad unused */
            cj->srvTextures[0].resource = ri_history;       /* t0 */
            cj->srvTextures[1].resource = ri_mv;            /* t1 */
            cj->srvTextures[2].resource = RI_NONE;          /* t2 unused */
            cj->srvTextures[3].resource = ri_color;         /* t3 */
            cj->srvTextures[4].resource = ri_recurrent_r;   /* t4 */
            cj->srvTextures[5].resource = RI_NONE;          /* t5 unused */
            cj->srvTextures[6].resource = ri_exposure;      /* t6 */

            cj->pipeline.uavTextureCount = 4;  /* u0..u3, pad unused */
            cj->uavTextures[0].resource = RI_NONE;          /* u0 unused */
            cj->uavTextures[1].resource = RI_NONE;          /* u1 unused */
            cj->uavTextures[2].resource = RI_NONE;          /* u2 unused */
            cj->uavTextures[3].resource = ri_reprojected;   /* u3 → bind 24 */

            /* ScratchBuffer at u11 */
            cj->pipeline.uavBufferCount = 1;
            cj->uavBuffers[0].resource = ri_scratch;
            cj->pipeline.uavBufferBindings[0].bindingIndex = 11;  /* u11 → bind 32 */

            cj->pipeline.constCount = 1;
            cj->cbs[0] = cbHandle;

            cj->dimensions[0] = (dw + 7) / 8;  /* pre-pass uses display resolution */
            cj->dimensions[1] = (dh + 7) / 8;
            cj->dimensions[2] = 1;

        } else if (pass == 13) {
            /* ── POST-PASS ───────────────────────────────────────────────
             *   t3 = r_input_color
             *   t6 = r_input_exposure
             *   t9 = r_reprojected_color  (from pre-pass)
             *   u1 = rw_history_color     (write history for next frame)
             *   u2 = rw_mlsr_output_color (final output)
             *   u6 = rw_recurrent_0       (write recurrent for next frame)
             *   u11 = ScratchBuffer
             *   b0 = OptimizedConstants
             */
            cj->pipeline.srvTextureCount = 10;  /* t0..t9, pad unused */
            cj->srvTextures[0].resource = RI_NONE;          /* t0 unused */
            cj->srvTextures[1].resource = RI_NONE;          /* t1 unused */
            cj->srvTextures[2].resource = RI_NONE;          /* t2 unused */
            cj->srvTextures[3].resource = ri_color;         /* t3 */
            cj->srvTextures[4].resource = RI_NONE;          /* t4 unused */
            cj->srvTextures[5].resource = RI_NONE;          /* t5 unused */
            cj->srvTextures[6].resource = ri_exposure;      /* t6 */
            cj->srvTextures[7].resource = RI_NONE;          /* t7 unused */
            cj->srvTextures[8].resource = RI_NONE;          /* t8 unused */
            cj->srvTextures[9].resource = ri_reprojected;   /* t9 */

            cj->pipeline.uavTextureCount = 7;  /* u0..u6, pad unused */
            cj->uavTextures[0].resource = RI_NONE;          /* u0 unused */
            cj->uavTextures[1].resource = ri_history;       /* u1 → bind 22 (write) */
            cj->uavTextures[2].resource = ri_output;        /* u2 → bind 23 */
            cj->uavTextures[3].resource = RI_NONE;          /* u3 unused */
            cj->uavTextures[4].resource = RI_NONE;          /* u4 unused */
            cj->uavTextures[5].resource = RI_NONE;          /* u5 unused */
            cj->uavTextures[6].resource = ri_recurrent_w;   /* u6 → bind 27 (write) */

            /* ScratchBuffer at u11 */
            cj->pipeline.uavBufferCount = 1;
            cj->uavBuffers[0].resource = ri_scratch;
            cj->pipeline.uavBufferBindings[0].bindingIndex = 11;  /* u11 → bind 32 */

            cj->pipeline.constCount = 1;
            cj->cbs[0] = cbHandle;

            cj->dimensions[0] = (dw + 7) / 8;  /* post-pass uses display resolution */
            cj->dimensions[1] = (dh + 7) / 8;
            cj->dimensions[2] = 1;

        } else {
            /* ── MODEL PASSES 1-12 ───────────────────────────────────────
             * Compiled with -fvk-t-shift 0 -fvk-u-shift 2:
             *   bind 0 (t0): input scratch buffer  → srvBuffers[0]
             *   bind 1 (t1): weights/initializer   → srvBuffers[1]
             *   bind 2 (u0): output scratch buffer → uavBuffers[0]
             *   bind 3 (u1): scratch/aux buffer    → uavBuffers[1]
             *
             * The scratch buffer is used for all four bindings.
             * The shader reads from different offsets than it writes to.
             * srvBuffers[1] (InitializerBuffer) is used for weight data
             * that's embedded in the shader; we bind scratch as a placeholder
             * so the descriptor is valid.
             */
            cj->pipeline.srvBufferCount = 2;
            cj->pipeline.uavBufferCount = 2;
            cj->srvBuffers[0].resource = ri_scratch;   /* bind 0: input */
            cj->srvBuffers[1].resource = ri_scratch;   /* bind 1: initializer/weights */
            cj->uavBuffers[0].resource = ri_scratch;   /* bind 2: output */
            cj->uavBuffers[1].resource = ri_scratch;   /* bind 3: scratch */

            uint32_t dx, dy;
            compute_model_dispatch(rw, rh, (uint32_t)pass, &dx, &dy);
            cj->dimensions[0] = dx;
            cj->dimensions[1] = dy;
            cj->dimensions[2] = 1;
        }

        c->iface.fpScheduleGpuJob(&c->iface, &job);
    }

    /* ── Execute all jobs ──────────────────────────────────────────────────── */
    c->iface.fpExecuteGpuJobs(&c->iface, (FfxCommandList)cmd, c->effectId);

    /* ── Unregister per-frame external resources ──────────────────────────── */
    c->iface.fpUnregisterResources(&c->iface, (FfxCommandList)cmd, c->effectId);

    /* ── Swap recurrent ping-pong for next frame ─────────────────────────── */
    c->recurrent_idx ^= 1;

    return FFX_API_RETURN_OK;
}

/* ── ffxConfigure (stub — not needed for basic upscaling) ─────────────────── */

ffxReturnCode_t ffxConfigure(ffxContext *ctx, const ffxApiHeader *desc)
{
    (void)ctx; (void)desc;
    return FFX_API_RETURN_OK;
}
