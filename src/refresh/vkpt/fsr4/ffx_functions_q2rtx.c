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

    /* Initialise the Vulkan backend context */
    FfxEffectBindlessConfig bindlessCfg = {0, 0, 0, 1};
    FfxErrorCode err = c->iface.fpCreateBackendContext(
        &c->iface, FFX_EFFECT_FSR4UPSCALER, &bindlessCfg, &c->effectId);
    if (err != FFX_OK) {
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
 * We build FfxGpuJobDescription entries — one per pass — and let the backend's
 * fpScheduleGpuJob / fpExecuteGpuJobs record the actual vkCmdDispatch calls.
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

ffxReturnCode_t ffxDispatch(ffxContext *ctx, const ffxDispatchDescHeader *desc)
{
    if (!ctx || !*ctx || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    Fsr4Context *c = (Fsr4Context *)(*ctx);
    if (!c->valid) return FFX_API_RETURN_ERROR;

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
    FfxResourceInternal ri_color  = reg_resource(c, &d->color,        c->effectId);
    FfxResourceInternal ri_depth  = reg_resource(c, &d->depth,        c->effectId);
    FfxResourceInternal ri_mv     = reg_resource(c, &d->motionVectors, c->effectId);
    FfxResourceInternal ri_output = reg_resource(c, &d->output,        c->effectId);

    /* Build the OptimizedConstants cbuffer (88 bytes, matches the push-constant
       layout in the pre/post pass shaders):
       float inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2],
       tex_size[2], max_renderSize[2], fMotionVectorJitterCancellation[2],
       uint width, height, reset, width_lr,
       uint height_lr, float preExposure, float previous_preExposure, uint rcas_enabled,
       float rcas_sharpness, float _pad1 */
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

        /* Pipeline: encode pass index as the pipeline void* (see ffx_fsr4_vk.c) */
        cj->pipeline.pipeline = (FfxPipeline)(uintptr_t)(uint32_t)pass;

        /* Pass 0 = pre, 13 = post: use full resource set + cbuffer */
        if (pass == 0 || pass == 13) {
            /* SRV textures: color (t0), depth (t2), motion vectors (t1) */
            cj->pipeline.srvTextureCount = 3;
            cj->srvTextures[0].resource = ri_color;
            cj->srvTextures[1].resource = ri_mv;
            cj->srvTextures[2].resource = ri_depth;

            /* UAV textures: output (u2) */
            cj->pipeline.uavTextureCount = (pass == 13) ? 1 : 0;
            if (pass == 13)
                cj->uavTextures[0].resource = ri_output;

            /* cbuffer */
            cj->pipeline.constCount = 1;
            cj->cbs[0] = cbHandle;

            /* Dispatch: full display resolution in 8x8 groups */
            cj->dimensions[0] = (dw + 7) / 8;
            cj->dimensions[1] = (dh + 7) / 8;
            cj->dimensions[2] = 1;

        } else {
            /* Inner model passes 1-12: just scratch buffer ping-pong */
            cj->pipeline.srvBufferCount = 1;
            cj->pipeline.uavBufferCount = 1;
            /* The backend maps srvBuffers[0] → binding 0 (input)
               and uavBuffers[0] → binding 1 (output scratch).
               The scratch buffer resource is managed by the backend internally;
               we pass sentinel indices and let it resolve them. */
            cj->srvBuffers[0].resource.internalIndex = 0; /* scratch read  */
            cj->uavBuffers[0].resource.internalIndex = 1; /* scratch write */

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

    return FFX_API_RETURN_OK;
}

/* ── ffxConfigure (stub — not needed for basic upscaling) ─────────────────── */

ffxReturnCode_t ffxConfigure(ffxContext *ctx, const ffxApiHeader *desc)
{
    (void)ctx; (void)desc;
    return FFX_API_RETURN_OK;
}
