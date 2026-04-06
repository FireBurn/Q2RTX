/*
Copyright (C) 2018 Christoph Schied
Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
Copyright (C) 2021, Frank Richter. All rights reserved.
Copyright (C) 2025, FSR4 ML upscaler port.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include "vkpt.h"
#include "fsr4/ffx_fsr4_vk.h"

/*
    FidelityFX Super Resolution 4 (FSR4) integration
    =================================================

    Replaces the FSR1 EASU+RCAS spatial upscaler with FSR4, an INT8
    machine-learning temporal upscaler from the AMD FSR4 SDK.

    Signal path:
        Rendered frame (render resolution)
          → TAA (Q2RTX asvgf_taau.comp)
          → VKPT_IMG_TAA_OUTPUT
          → FSR4 vkpt_fsr_do()
          → VKPT_IMG_FSR_EASU_OUTPUT   (upscaled, display resolution)
          → vkpt_final_blit()
          → swapchain

    VKPT_IMG_FSR_RCAS_OUTPUT is repurposed as the FSR4 recurrent-state
    ping-pong buffer managed internally by the FSR4 backend.

    Cvars
    -----
    flt_fsr_enable    0 = off, 1 = on (same name as before)
    flt_fsr_sharpness float [0,2], fed to FSR4 RCAS sharpening (default 0.2)

    Shader files expected in baseq2/fsr4_shaders/
    ---------------------------------------------
    fsr4_pre.spv
    fsr4_1080_pass1.spv  ... fsr4_1080_pass12.spv   (for ≤1080p output)
    fsr4_2160_pass1.spv  ... fsr4_2160_pass12.spv   (for ≤4K output)
    fsr4_4320_pass1.spv  ... fsr4_4320_pass12.spv   (for 8K output)
    fsr4_post.spv
    fsr4_rcas.spv        (optional)
    fsr4_spd.spv         (optional)

    Generate with:
        ./compile_shaders.sh <fsr4_sdk_root> performance baseq2/fsr4_shaders
*/

/* ── backend state ───────────────────────────────────────────────────────── */

/* Definition of the global override pointer declared extern in vkpt.h.
   Set to &fsr4_backend around every ffx::* call so the FSR4 provider
   routes through our Vulkan backend instead of the DX12 path. */
FfxInterface *g_vkBackendOverride = NULL;

static FfxInterface     fsr4_backend;
static void *           fsr4_scratch     = NULL;
static ffxContext        fsr4_context;
static bool             fsr4_context_ok  = false;
static bool             fsr4_backend_ok  = false;
static bool             fsr4_reset_next  = true;
static uint32_t         fsr4_last_rw     = 0;
static uint32_t         fsr4_last_rh     = 0;

cvar_t *cvar_flt_fsr_enable    = NULL;
cvar_t *cvar_flt_fsr_sharpness = NULL;
/* Compatibility stubs - profiler.c externs these */
cvar_t *cvar_flt_fsr_easu = NULL;
cvar_t *cvar_flt_fsr_rcas = NULL;

/* ── shader loading ──────────────────────────────────────────────────────── */

static bool load_spv(const char *rel_path, const char *entry_point, FfxFsr4VkShaderBlob *out)
{
    char full[MAX_OSPATH];
    Q_snprintf(full, sizeof(full), "fsr4_shaders/%s", rel_path);

    byte *buf = NULL;
    int   len = FS_LoadFile(full, (void **)&buf);
    if (len <= 0 || !buf) {
        Com_WPrintf("FSR4: shader not found: %s\n", full);
        return false;
    }
    if (len & 3) {
        Com_WPrintf("FSR4: shader %s size %d not DWORD-aligned\n", full, len);
        FS_FreeFile(buf);
        return false;
    }

    uint32_t *words = Z_Malloc((size_t)len);
    memcpy(words, buf, (size_t)len);
    FS_FreeFile(buf);

    out->spirv      = words;
    out->sizeBytes  = (size_t)len;
    out->entryPoint = entry_point;
    return true;
}

static void free_blobs(FfxFsr4VkShaderBlob blobs[FFX_FSR4_VK_PASS_COUNT])
{
    for (int i = 0; i < FFX_FSR4_VK_PASS_COUNT; i++) {
        if (blobs[i].spirv) {
            Z_Free((void *)blobs[i].spirv);
            blobs[i].spirv = NULL;
        }
    }
}

static const char *display_res_tag(void)
{
    uint32_t h = qvk.extent_unscaled.height;
    if (h > 2160) return "4320";
    if (h > 1080) return "2160";
    return "1080";
}

/* ── context management ──────────────────────────────────────────────────── */

static VkResult fsr4_recreate_context(void)
{
    if (!fsr4_backend_ok) return VK_SUCCESS;

    if (fsr4_context_ok) {
        g_vkBackendOverride = &fsr4_backend;
        ffxDestroyContext(&fsr4_context, NULL);
        g_vkBackendOverride = NULL;
        fsr4_context_ok = false;
    }

    g_vkBackendOverride = &fsr4_backend;

    ffxCreateContextDescUpscale desc;
    memset(&desc, 0, sizeof(desc));
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    desc.header.pNext = NULL;
    desc.maxRenderSize.width   = qvk.extent_render.width;
    desc.maxRenderSize.height  = qvk.extent_render.height;
    desc.maxUpscaleSize.width  = qvk.extent_unscaled.width;
    desc.maxUpscaleSize.height = qvk.extent_unscaled.height;
    desc.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE
               | FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;

    ffxReturnCode_t ret = ffxCreateContext(&fsr4_context, &desc.header, NULL);
    g_vkBackendOverride = NULL;

    if (ret != FFX_API_RETURN_OK) {
        Com_WPrintf("FSR4: ffx::CreateContext failed (%d)\n", (int)ret);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    fsr4_context_ok    = true;
    fsr4_reset_next    = true;
    fsr4_last_rw       = qvk.extent_render.width;
    fsr4_last_rh       = qvk.extent_render.height;

    Com_Printf("FSR4: context ready %ux%u -> %ux%u\n",
               qvk.extent_render.width,   qvk.extent_render.height,
               qvk.extent_unscaled.width, qvk.extent_unscaled.height);
    return VK_SUCCESS;
}

/* ── vkpt public API ─────────────────────────────────────────────────────── */

void vkpt_fsr_init_cvars(void)
{
    cvar_flt_fsr_enable    = Cvar_Get("flt_fsr_enable",    "0",   CVAR_ARCHIVE);
    cvar_flt_fsr_sharpness = Cvar_Get("flt_fsr_sharpness", "0.2", CVAR_ARCHIVE);
    cvar_flt_fsr_easu      = Cvar_Get("flt_fsr_easu",      "1",   CVAR_ARCHIVE);
    cvar_flt_fsr_rcas      = Cvar_Get("flt_fsr_rcas",      "1",   CVAR_ARCHIVE);
}

VkResult vkpt_fsr_initialize(void)
{
    size_t scratch_sz = ffxFsr4VkGetScratchMemorySize();
    fsr4_scratch = Z_Malloc(scratch_sz);

    FfxFsr4VkShaderBlob blobs[FFX_FSR4_VK_PASS_COUNT];
    memset(blobs, 0, sizeof(blobs));

    bool ok = true;
    char name[64];
    const char *res = display_res_tag();

    ok &= load_spv("fsr4_pre.spv",  "main", &blobs[0]);
    for (int i = 1; i <= 12 && ok; i++) {
        /* Entry point names must outlive the blobs array.  Using string
         * literals (static storage) avoids a dangling pointer when the
         * local `entry` buffer goes out of scope before the pipelines are
         * built by ffxCreateContext → cbCreateBackendCtx → mkPipeline. */
        static const char *pass_entries[13] = {
            NULL,   /* [0] unused – pre-pass uses "main" */
            "fsr4_model_v07_i8_pass1",  "fsr4_model_v07_i8_pass2",
            "fsr4_model_v07_i8_pass3",  "fsr4_model_v07_i8_pass4",
            "fsr4_model_v07_i8_pass5",  "fsr4_model_v07_i8_pass6",
            "fsr4_model_v07_i8_pass7",  "fsr4_model_v07_i8_pass8",
            "fsr4_model_v07_i8_pass9",  "fsr4_model_v07_i8_pass10",
            "fsr4_model_v07_i8_pass11", "fsr4_model_v07_i8_pass12",
        };
        Q_snprintf(name, sizeof(name), "fsr4_%s_pass%d.spv", res, i);
        ok &= load_spv(name, pass_entries[i], &blobs[i]);
    }
    ok &= load_spv("fsr4_post.spv", "main", &blobs[13]);
    /* optional passes – don't fail if absent */
    load_spv("fsr4_rcas.spv", "main", &blobs[14]);
    load_spv("fsr4_spd.spv",  "main", &blobs[15]);

    if (!ok) {
        Com_WPrintf("FSR4: required shaders missing – FSR4 disabled.\n"
                    "      Run compile_shaders.sh to build baseq2/fsr4_shaders/\n");
        free_blobs(blobs);
        Z_Free(fsr4_scratch);
        fsr4_scratch = NULL;
        return VK_SUCCESS;
    }

    FfxFsr4VkCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.device            = qvk.device;
    ci.physicalDevice    = qvk.physical_device;
    ci.scratchBuffer     = fsr4_scratch;
    ci.scratchBufferSize = scratch_sz;
    memcpy(ci.shaders, blobs, sizeof(blobs));

    VkResult r = ffxFsr4VkCreateContext(&ci, &fsr4_backend);
    free_blobs(blobs);

    if (r != VK_SUCCESS) {
        Com_WPrintf("FSR4: ffxFsr4VkCreateContext failed (%d) – FSR4 disabled.\n", r);
        Z_Free(fsr4_scratch);
        fsr4_scratch = NULL;
        return VK_SUCCESS;
    }

    fsr4_backend_ok = true;
    Com_Printf("FSR4: backend ready (INT8/dot4add path).\n");
    return VK_SUCCESS;
}

VkResult vkpt_fsr_destroy(void)
{
    if (fsr4_context_ok) {
        g_vkBackendOverride = &fsr4_backend;
        ffxDestroyContext(&fsr4_context, NULL);
        g_vkBackendOverride = NULL;
        fsr4_context_ok = false;
    }
    if (fsr4_backend_ok) {
        ffxFsr4VkDestroyContext((FfxFsr4VkContext *)fsr4_scratch);
        fsr4_backend_ok = false;
    }
    if (fsr4_scratch) {
        Z_Free(fsr4_scratch);
        fsr4_scratch = NULL;
    }
    fsr4_reset_next = true;
    fsr4_last_rw = fsr4_last_rh = 0;
    return VK_SUCCESS;
}

VkResult vkpt_fsr_create_pipelines(void)
{
    return fsr4_recreate_context();
}

VkResult vkpt_fsr_destroy_pipelines(void)
{
    if (fsr4_context_ok) {
        g_vkBackendOverride = &fsr4_backend;
        ffxDestroyContext(&fsr4_context, NULL);
        g_vkBackendOverride = NULL;
        fsr4_context_ok = false;
    }
    return VK_SUCCESS;
}

bool vkpt_fsr_is_enabled(void)
{
    if (!fsr4_backend_ok || !fsr4_context_ok)
        return false;
    if (!cvar_flt_fsr_enable->integer)
        return false;
    if (cvar_flt_fsr_enable->integer == 1 &&
        (qvk.extent_render.width  >= qvk.extent_unscaled.width ||
         qvk.extent_render.height >= qvk.extent_unscaled.height))
        return false;
    return true;
}

/* FSR4 handles its own upscaling – no separate pre-upscale step needed */
bool vkpt_fsr_needs_upscale(void)
{
    return false;
}

/* No-op: FSR4 drives its own constants via the FfxInterface */
void vkpt_fsr_update_ubo(QVKUniformBuffer_t *ubo)
{
    (void)ubo;
}

VkResult vkpt_fsr_do(VkCommandBuffer cmd_buf)
{
    if (!vkpt_fsr_is_enabled())
        return VK_SUCCESS;

    /* Recreate if render resolution changed */
    if (qvk.extent_render.width  != fsr4_last_rw ||
        qvk.extent_render.height != fsr4_last_rh)
    {
        _VK(fsr4_recreate_context());
        if (!fsr4_context_ok) return VK_SUCCESS;
    }

    BEGIN_PERF_MARKER(cmd_buf, PROFILER_FSR);

    /* Query the Halton jitter offset for this frame */
    int32_t phase_count = 0;
    {
        ffxQueryDescUpscaleGetJitterPhaseCount q;
        memset(&q, 0, sizeof(q));
        q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_JITTER_PHASE_COUNT;
        q.header.pNext = NULL;
        q.displayWidth   = qvk.extent_unscaled.width;
        q.renderWidth    = qvk.extent_render.width;
        q.pOutPhaseCount = &phase_count;
        g_vkBackendOverride = &fsr4_backend;
        ffxQuery(&fsr4_context, &q.header);
        g_vkBackendOverride = NULL;
    }

    float jx = 0.f, jy = 0.f;
    if (phase_count > 0) {
        ffxQueryDescUpscaleGetJitterOffset q;
        memset(&q, 0, sizeof(q));
        q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_JITTER_OFFSET;
        q.header.pNext = NULL;
        q.index        = (int32_t)(qvk.frame_counter % (uint64_t)phase_count);
        q.phaseCount   = phase_count;
        q.pOutX        = &jx;
        q.pOutY        = &jy;
        g_vkBackendOverride = &fsr4_backend;
        ffxQuery(&fsr4_context, &q.header);
        g_vkBackendOverride = NULL;
    }

    /* Transition TAA output: GENERAL -> SHADER_READ_ONLY_OPTIMAL */
    {
        VkImageSubresourceRange sr = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
        };
        IMAGE_BARRIER(cmd_buf,
            .image            = qvk.images[VKPT_IMG_TAA_OUTPUT],
            .subresourceRange = sr,
            .srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        );
    }

    /* Build the dispatch descriptor */
    ffxDispatchDescUpscale d;
    memset(&d, 0, sizeof(d));
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.header.pNext = NULL;

#define FILL_TEX(field, img_idx, fmt, w, h)                                 \
    do {                                                                     \
        d.field.resource              = (void *)qvk.images_views[img_idx]; \
        d.field.description.type      = FFX_RESOURCE_TYPE_TEXTURE2D;       \
        d.field.description.format    = (fmt);                              \
        d.field.description.width     = (w);                                \
        d.field.description.height    = (h);                                \
    } while(0)

    FILL_TEX(color,        VKPT_IMG_TAA_OUTPUT,      FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
             qvk.extent_render.width,   qvk.extent_render.height);

    FILL_TEX(depth,        VKPT_IMG_PT_MOTION,       FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
             qvk.extent_render.width,   qvk.extent_render.height);

    FILL_TEX(motionVectors, VKPT_IMG_PT_MOTION,      FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
             qvk.extent_render.width,   qvk.extent_render.height);

    FILL_TEX(output,       VKPT_IMG_FSR_EASU_OUTPUT, FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
             qvk.extent_unscaled.width, qvk.extent_unscaled.height);

#undef FILL_TEX

    d.renderSize.width   = qvk.extent_render.width;
    d.renderSize.height  = qvk.extent_render.height;
    d.upscaleSize.width  = qvk.extent_unscaled.width;
    d.upscaleSize.height = qvk.extent_unscaled.height;

    d.jitterOffset.x = jx;
    d.jitterOffset.y = jy;

    /* PT_MOTION is in pixel-space, scaled by the render resolution */
    d.motionVectorScale.x = (float)qvk.extent_render.width;
    d.motionVectorScale.y = (float)qvk.extent_render.height;

    d.frameTimeDelta   = 16.667f;   /* TODO: pass actual delta from main.c */
    d.sharpness        = cvar_flt_fsr_sharpness->value;
    d.enableSharpening = (cvar_flt_fsr_rcas->integer != 0) ? 1 : 0;
    d.reset            = fsr4_reset_next ? 1 : 0;
    d.commandList      = (FfxCommandList)cmd_buf;

    fsr4_reset_next = false;

    g_vkBackendOverride = &fsr4_backend;
    ffxDispatch(&fsr4_context, &d.header);
    g_vkBackendOverride = NULL;

    /* Restore TAA output: SHADER_READ_ONLY_OPTIMAL -> GENERAL */
    {
        VkImageSubresourceRange sr = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
        };
        IMAGE_BARRIER(cmd_buf,
            .image            = qvk.images[VKPT_IMG_TAA_OUTPUT],
            .subresourceRange = sr,
            .srcAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
        );
    }

    END_PERF_MARKER(cmd_buf, PROFILER_FSR);
    return VK_SUCCESS;
}

VkResult vkpt_fsr_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
    return vkpt_final_blit(cmd_buf, VKPT_IMG_FSR_EASU_OUTPUT,
                           qvk.extent_unscaled, false, warp);
}
