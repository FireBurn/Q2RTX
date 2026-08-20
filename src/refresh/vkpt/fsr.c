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
#include "system/system.h"
#include "fsr4/ffx_fsr4_vk.h"
#include "fsr4/ffx_fsr4_schedule.h"
#include "fsr4/ffx_fsr4_assets.h"
#ifdef VKPT_FSR3
#include "ffx_vk_portable.h"
#include "ffx_vk_fsr3_3_1_5_bridge.h"
#endif
#include <math.h>

/*
    FidelityFX temporal upscaler integration
    =========================================

    Replaces the old FSR1 EASU+RCAS spatial path with selectable native Vulkan
    FSR3 or experimental INT8 FSR4 temporal reconstruction.

    Signal path:
        Rendered frame (render resolution)
          → provider (FSR3 or FSR4; Q2RTX TAA is bypassed)
          → VKPT_IMG_FSR_EASU_OUTPUT   (upscaled, display resolution)
          → copy to VKPT_IMG_TAA_OUTPUT for the existing post chain
          → vkpt_final_blit()
          → swapchain

    VKPT_IMG_FSR_RCAS_OUTPUT is repurposed as the FSR4 recurrent-state
    ping-pong buffer managed internally by the FSR4 backend.

    Cvars
    -----
    flt_upscaler      0 = Q2RTX, 1 = FSR3 3.1.4, 2 = FSR4 v07 model family,
                      3 = FSR3 3.1.5 public-SDK Vulkan experiment
    flt_fsr_enable    deprecated alias that migrates to flt_upscaler 2
    flt_fsr_sharpness deprecated compatibility cvar; it has no effect

    Shader files expected in baseq2/fsr4_shaders/
    ---------------------------------------------
    fsr4_model_v07_i8_<preset>_<tier>_{pre,pass1..pass12,post}.spv
    fsr4_model_v07_i8_<preset>_{initializers,pre_weights}.bin
    rcas.spv
    spd_auto_exposure.spv

    Generate with:
        src/refresh/vkpt/fsr4/compile_shaders_fsr4.sh \
            <fsr4_sdk_root> all baseq2/fsr4_shaders
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
static uint32_t         fsr4_ctx_dw      = 0;  /* display res context was created at */
static uint32_t         fsr4_ctx_dh      = 0;
static char             fsr4_shader_tier[5] = {0};
static char             fsr4_shader_model[16] = {0};
/* Preserve a precise initialization failure for the requested-versus-active
 * menu status. Distribution packages intentionally omit incomplete v07 model
 * blobs, so a generic "context unavailable" is not actionable. */
static char             fsr4_unavailable_reason[128] =
    "fallback: FSR4 v07 context unavailable";

enum {
    VKPT_UPSCALER_Q2RTX = 0,
    VKPT_UPSCALER_FSR3 = 1,
    VKPT_UPSCALER_FSR4 = 2,
    VKPT_UPSCALER_FSR3_315 = 3,
};

static int requested_fsr_quality(void);
static const char *fsr4_requested_model(void);
static int fsr4_requested_render_scale(void);
static const char *fsr_quality_label(int quality);

static bool fsr4_dynamic_resolution_requested(void)
{
    return cvar_flt_fsr4_dynamic_resolution &&
           cvar_flt_fsr4_dynamic_resolution->integer != 0;
}

enum {
    VKPT_FSR_QUALITY_NATIVE_AA = 0,
    VKPT_FSR_QUALITY_QUALITY = 1,
    VKPT_FSR_QUALITY_BALANCED = 2,
    VKPT_FSR_QUALITY_PERFORMANCE = 3,
    VKPT_FSR_QUALITY_ULTRA_PERFORMANCE = 4,
};

#ifdef VKPT_FSR3
static FfxVkPortableUpscaleContext *fsr3_context = NULL;
static bool fsr3_context_ok = false;
static bool fsr3_reset_next = true;
static uint32_t fsr3_ctx_dw = 0;
static uint32_t fsr3_ctx_dh = 0;
static FfxVkPortableFrameGenerationContext *fsr3_frame_generation_context = NULL;
static bool fsr3_frame_generation_context_ok = false;
static FfxVkFsr3_3_1_6FrameGenerationContext *fsr3_316_frame_generation_context = NULL;
static bool fsr3_316_frame_generation_context_ok = false;
static uint64_t fsr3_316_frame_generation_frame_ids[MAX_FRAMES_IN_FLIGHT];
static bool fsr3_frame_generation_reset_next = true;
static uint32_t fsr3_frame_generation_dw = 0;
static uint32_t fsr3_frame_generation_dh = 0;
static FfxVkFsr3_3_1_5UpscalerContext *fsr3_315_context = NULL;
static bool fsr3_315_context_ok = false;
static bool fsr3_315_reset_next = true;
static uint32_t fsr3_315_ctx_dw = 0;
static uint32_t fsr3_315_ctx_dh = 0;
static FfxVkFsr3_3_1_5Resource fsr3_315_color;
static FfxVkFsr3_3_1_5Resource fsr3_315_depth;
static FfxVkFsr3_3_1_5Resource fsr3_315_motion;
static FfxVkFsr3_3_1_5Resource fsr3_315_reactive;
static FfxVkFsr3_3_1_5Resource fsr3_315_composition;
static FfxVkFsr3_3_1_5Resource fsr3_315_dilated_depth;
static FfxVkFsr3_3_1_5Resource fsr3_315_dilated_motion;
static FfxVkFsr3_3_1_5Resource fsr3_315_previous_depth;
static FfxVkFsr3_3_1_5Resource fsr3_315_output;
#endif

static uint32_t align_up_8(uint32_t value)
{
    return (value + 7u) & ~7u;
}

static bool fsr4_output_extent_supported(VkExtent2D extent)
{
    /* The largest v07 tensor specialization is the 4320/8K graph.  AMD's
     * provider rejects anything larger rather than reusing its fixed scratch
     * layout, and the Vulkan path must do the same. */
    return extent.width > 0u && extent.height > 0u &&
           extent.width <= FFX_FSR4_DOT4_MAX_OUTPUT_WIDTH &&
           extent.height <= FFX_FSR4_DOT4_MAX_OUTPUT_HEIGHT;
}

static bool fsr4_format_supports(VkFormat format, VkFormatFeatureFlags required)
{
    VkFormatProperties properties;

    memset(&properties, 0, sizeof(properties));
    vkGetPhysicalDeviceFormatProperties(qvk.physical_device, format,
                                        &properties);
    return (properties.optimalTilingFeatures & required) == required;
}

static bool fsr4_device_is_capable(void)
{
    const VkFormatFeatureFlags sampled_storage =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;

    if (!qvk.supports_fp16 || !qvk.supports_int16 || !qvk.supports_int8 ||
        !qvk.supports_dot4 || !qvk.supports_compute_derivatives ||
        !qvk.supports_storage_image_extended_formats ||
        !qvk.supports_storage_image_write_without_format) {
        Com_WPrintf("FSR4: required FP16/INT16/INT8/DOT4/derivative/storage "
                    "features are unavailable; using fallback.\n");
        return false;
    }

    /* The v07 INT8 graph samples and stores all of these formats.  Validate
     * the actual optimal-tiling format feature table instead of assuming that
     * the corresponding arithmetic feature bits make images usable. */
    if (!fsr4_format_supports(VK_FORMAT_R8G8B8A8_UNORM,
                              sampled_storage) ||
        !fsr4_format_supports(VK_FORMAT_R16G16_SFLOAT,
                              sampled_storage) ||
        !fsr4_format_supports(VK_FORMAT_R16G16B16A16_SFLOAT,
                              sampled_storage) ||
        !fsr4_format_supports(VK_FORMAT_R32_SFLOAT,
                              sampled_storage)) {
        Com_WPrintf("FSR4: a required sampled/storage image format is "
                    "unsupported; using fallback.\n");
        return false;
    }

    return true;
}

cvar_t *cvar_flt_fsr_enable    = NULL;
cvar_t *cvar_flt_upscaler     = NULL;
cvar_t *cvar_flt_fsr_quality  = NULL;
cvar_t *cvar_flt_fsr3_sharpening = NULL;
cvar_t *cvar_flt_fsr4_sharpening = NULL;
cvar_t *cvar_flt_fsr4_auto_exposure = NULL;
cvar_t *cvar_flt_fsr4_dynamic_resolution = NULL;
cvar_t *cvar_flt_frame_generation = NULL;
cvar_t *cvar_flt_frame_generation_backend = NULL;
cvar_t *cvar_flt_frame_generation_min_rendered_fps = NULL;
cvar_t *cvar_flt_frame_generation_active = NULL;
cvar_t *cvar_flt_frame_generation_reason = NULL;
cvar_t *cvar_flt_frame_generation_rendered_fps = NULL;
cvar_t *cvar_flt_frame_generation_generated_fps = NULL;
static unsigned fsr3_fg_last_present_msec;
static float fsr3_fg_filtered_rendered_fps;
static unsigned fsr3_fg_low_rate_frames;
static unsigned fsr3_fg_recovery_frames;
static bool fsr3_fg_rate_blocked;
cvar_t *cvar_flt_upscaler_active = NULL;
cvar_t *cvar_flt_upscaler_reason = NULL;
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
        Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
                   "fallback: FSR4 v07 shader asset missing (%s)", rel_path);
        return false;
    }
    if (len & 3) {
        Com_WPrintf("FSR4: shader %s size %d not DWORD-aligned\n", full, len);
        Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
                   "fallback: FSR4 v07 shader asset is invalid (%s)", rel_path);
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

static bool load_model_asset(const char *rel_path, void **out_data, size_t *out_size)
{
    char full[MAX_OSPATH];
    byte *file_data = NULL;
    int file_size;

    if (!out_data || !out_size)
        return false;
    *out_data = NULL;
    *out_size = 0;
    Q_snprintf(full, sizeof(full), "fsr4_shaders/%s", rel_path);
    file_size = FS_LoadFile(full, (void **)&file_data);
    if (file_size <= 0 || !file_data) {
        Com_WPrintf("FSR4: model asset not found: %s\n", full);
        Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
                   "fallback: FSR4 v07 model asset missing (%s)", rel_path);
        return false;
    }

    *out_data = Z_Malloc((size_t)file_size);
    memcpy(*out_data, file_data, (size_t)file_size);
    *out_size = (size_t)file_size;
    FS_FreeFile(file_data);
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
    uint32_t w = qvk.extent_unscaled.width;
    uint32_t h = qvk.extent_unscaled.height;
    /* Match AMD's provider: either dimension crossing a tier boundary selects
     * the next generated scratch/tensor layout (important for ultrawide). */
    if (w > 3840 || h > 2160) return "4320";
    if (w > 1920 || h > 1080) return "2160";
    return "1080";
}

static void fsr4_destroy_provider_context(void)
{
    if (!fsr4_context_ok)
        return;
    g_vkBackendOverride = &fsr4_backend;
    ffxDestroyContext(&fsr4_context, NULL);
    g_vkBackendOverride = NULL;
    fsr4_context_ok = false;
    fsr4_ctx_dw = fsr4_ctx_dh = 0;
}

static void fsr4_destroy_backend(void)
{
    fsr4_destroy_provider_context();
    if (fsr4_backend_ok) {
        ffxFsr4VkDestroyContext((FfxFsr4VkContext *)fsr4_scratch);
        fsr4_backend_ok = false;
    }
    memset(&fsr4_backend, 0, sizeof(fsr4_backend));
    fsr4_shader_tier[0] = '\0';
    fsr4_shader_model[0] = '\0';
}

static VkResult fsr4_create_backend_for_tier(const char *tier,
                                              const char *model)
{
    FfxFsr4VkShaderBlob blobs[FFX_FSR4_VK_PASS_COUNT];
    FfxFsr4V07AssetSet assets;
    FfxFsr4VkCreateInfo ci;
    void *model_initializer = NULL;
    void *pre_pass_weights = NULL;
    size_t model_initializer_size = 0;
    size_t pre_pass_weights_size = 0;
    bool ok = true;
    VkResult result;
    FfxFsr4ModelPreset preset;

    if (!tier || !model || !fsr4_scratch) {
        Q_strlcpy(fsr4_unavailable_reason,
                  "fallback: FSR4 v07 backend initialization failed",
                  sizeof(fsr4_unavailable_reason));
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (!strcmp(model, "native"))
        preset = FFX_FSR4_MODEL_PRESET_NATIVE_AA;
    else if (!strcmp(model, "quality"))
        preset = FFX_FSR4_MODEL_PRESET_QUALITY;
    else if (!strcmp(model, "balanced"))
        preset = FFX_FSR4_MODEL_PRESET_BALANCED;
    else if (!strcmp(model, "performance"))
        preset = FFX_FSR4_MODEL_PRESET_PERFORMANCE;
    else if (!strcmp(model, "ultraperf"))
        preset = FFX_FSR4_MODEL_PRESET_ULTRA_PERFORMANCE;
    else if (!strcmp(model, "drs"))
        preset = FFX_FSR4_MODEL_PRESET_DRS;
    else {
        Q_strlcpy(fsr4_unavailable_reason,
                  "fallback: FSR4 v07 requested model is unsupported",
                  sizeof(fsr4_unavailable_reason));
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (!ffxFsr4V07BuildAssetSet(preset, qvk.extent_unscaled.width,
                                 qvk.extent_unscaled.height, &assets) ||
        strcmp(assets.tier, tier) != 0) {
        Com_WPrintf("FSR4: reusable asset selector rejected %s/%s.\n",
                    model, tier);
        Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
                   "fallback: FSR4 v07 asset selector rejected %s/%s", model, tier);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
               "fallback: FSR4 v07 %s/%s assets unavailable", model, tier);
    memset(blobs, 0, sizeof(blobs));

    ok &= load_spv(assets.pre, "main", &blobs[0]);
    for (int i = 1; i <= 12 && ok; ++i) {
        static const char *pass_entries[13] = {
            NULL,
            "fsr4_model_v07_i8_pass1",  "fsr4_model_v07_i8_pass2",
            "fsr4_model_v07_i8_pass3",  "fsr4_model_v07_i8_pass4",
            "fsr4_model_v07_i8_pass5",  "fsr4_model_v07_i8_pass6",
            "fsr4_model_v07_i8_pass7",  "fsr4_model_v07_i8_pass8",
            "fsr4_model_v07_i8_pass9",  "fsr4_model_v07_i8_pass10",
            "fsr4_model_v07_i8_pass11", "fsr4_model_v07_i8_pass12",
        };
        ok &= load_spv(assets.model[i - 1], pass_entries[i], &blobs[i]);
    }
    ok &= load_spv(assets.post, "main", &blobs[13]);

    /* RCAS is a real optional dispatch, but its pipeline must be present so a
     * user-visible sharpening request cannot turn into a partial graph. */
    ok &= load_spv(assets.rcas, "main", &blobs[14]);
    ok &= load_spv(assets.spdAutoExposure, "main", &blobs[15]);
    ok &= load_model_asset(assets.initializer, &model_initializer,
                           &model_initializer_size);
    ok &= load_model_asset(assets.prePassWeights, &pre_pass_weights,
                           &pre_pass_weights_size);
    if (model_initializer_size != FFX_FSR4_V07_INITIALIZER_BYTES) {
        Com_WPrintf("FSR4: initializer has unexpected size %zu (expected %u)\n",
                    model_initializer_size,
                    (unsigned)FFX_FSR4_V07_INITIALIZER_BYTES);
        if (model_initializer) {
            Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
                       "fallback: FSR4 v07 initializer size is invalid (%s)",
                       assets.initializer);
        }
        ok = false;
    }
    if (pre_pass_weights_size != FFX_FSR4_V07_PRE_PASS_WEIGHTS_BYTES) {
        Com_WPrintf("FSR4: pre-pass weights have unexpected size %zu (expected %u)\n",
                    pre_pass_weights_size,
                    (unsigned)FFX_FSR4_V07_PRE_PASS_WEIGHTS_BYTES);
        if (pre_pass_weights) {
            Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
                       "fallback: FSR4 v07 pre-pass weights size is invalid (%s)",
                       assets.prePassWeights);
        }
        ok = false;
    }
    if (!ok) {
        Com_WPrintf("FSR4: coherent %s/%s shader/model set is unavailable; FSR4 disabled.\n"
                    "      Build target fsr4_shaders or run "
                    "src/refresh/vkpt/fsr4/compile_shaders_fsr4.sh.\n",
                    model, tier);
        result = VK_ERROR_INITIALIZATION_FAILED;
        goto cleanup;
    }

    memset(&ci, 0, sizeof(ci));
    ci.device = qvk.device;
    ci.physicalDevice = qvk.physical_device;
    ci.scratchBuffer = fsr4_scratch;
    ci.scratchBufferSize = ffxFsr4VkGetScratchMemorySize();
    memcpy(ci.shaders, blobs, sizeof(blobs));
    ci.modelInitializer = model_initializer;
    ci.modelInitializerSize = model_initializer_size;
    ci.prePassWeights = pre_pass_weights;
    ci.prePassWeightsSize = pre_pass_weights_size;

    result = ffxFsr4VkCreateContext(&ci, &fsr4_backend);
    if (result == VK_SUCCESS) {
        fsr4_backend_ok = true;
        Q_strlcpy(fsr4_shader_tier, tier, sizeof(fsr4_shader_tier));
        Q_strlcpy(fsr4_shader_model, model, sizeof(fsr4_shader_model));
        fsr4_unavailable_reason[0] = '\0';
        Com_Printf("FSR4: Vulkan backend ready (INT8/DOT4, %s, %s tier).\n",
                   model, tier);
    } else {
        Com_WPrintf("FSR4: Vulkan backend creation failed (%d); FSR4 disabled.\n",
                    result);
        Q_snprintf(fsr4_unavailable_reason, sizeof(fsr4_unavailable_reason),
                   "fallback: FSR4 v07 Vulkan backend creation failed (%s)",
                   qvk_result_to_string(result));
        ffxFsr4VkDestroyContext((FfxFsr4VkContext *)fsr4_scratch);
        memset(&fsr4_backend, 0, sizeof(fsr4_backend));
    }

cleanup:
    free_blobs(blobs);
    if (model_initializer) Z_Free(model_initializer);
    if (pre_pass_weights) Z_Free(pre_pass_weights);
    return result;
}

/* ── context management ──────────────────────────────────────────────────── */

static VkResult fsr4_recreate_context(void)
{
    const char *required_tier = display_res_tag();
    const char *required_model = fsr4_requested_model();
    if (!fsr4_output_extent_supported(qvk.extent_unscaled)) {
        Com_WPrintf("FSR4: output extent %ux%u exceeds the v07 7680x4320 "
                    "limit; using fallback.\n",
                    qvk.extent_unscaled.width, qvk.extent_unscaled.height);
        Q_strlcpy(fsr4_unavailable_reason,
                  "fallback: FSR4 v07 output exceeds 8K limit",
                  sizeof(fsr4_unavailable_reason));
        fsr4_destroy_backend();
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    if (!fsr4_backend_ok || strcmp(fsr4_shader_tier, required_tier) != 0 ||
        strcmp(fsr4_shader_model, required_model) != 0) {
        /* A quality model owns persistent images and descriptors.  Settings
         * changes can occur while older frames are still in flight, unlike a
         * normal swapchain recreation, so retire them before replacing that
         * graph. This is deliberately a rare user-triggered hitch. */
        if (fsr4_backend_ok)
            _VK(vkDeviceWaitIdle(qvk.device));
        fsr4_destroy_backend();
        if (!fsr4_scratch ||
            fsr4_create_backend_for_tier(required_tier, required_model) != VK_SUCCESS)
            return VK_ERROR_INITIALIZATION_FAILED;
    }

    fsr4_destroy_provider_context();

    g_vkBackendOverride = &fsr4_backend;

    ffxCreateContextDescUpscale desc;
    memset(&desc, 0, sizeof(desc));
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    desc.header.pNext = NULL;
    /* Use display resolution as maxRenderSize so the context handles any
       dynamic resolution without needing recreation.  The actual render
       resolution is passed per-frame in the dispatch descriptor. */
    desc.maxRenderSize.width   = align_up_8(qvk.extent_unscaled.width);
    desc.maxRenderSize.height  = align_up_8(qvk.extent_unscaled.height);
    desc.maxUpscaleSize.width  = align_up_8(qvk.extent_unscaled.width);
    desc.maxUpscaleSize.height = align_up_8(qvk.extent_unscaled.height);
    /* The provider always owns a complete SPD resource graph. The per-frame
     * cvar selects its optional dispatch; the default remains Q2RTX's explicit
     * exposure identity path. */
    desc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                 FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    if (fsr4_dynamic_resolution_requested())
        desc.flags |= FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;

    ffxReturnCode_t ret = ffxCreateContext(&fsr4_context, &desc.header, NULL);
    g_vkBackendOverride = NULL;

    if (ret != FFX_API_RETURN_OK) {
        Com_WPrintf("FSR4: ffx::CreateContext failed (%d)\n", (int)ret);
        Q_strlcpy(fsr4_unavailable_reason,
                  "fallback: FSR4 v07 provider context creation failed",
                  sizeof(fsr4_unavailable_reason));
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    fsr4_context_ok    = true;
    fsr4_reset_next    = true;
    fsr4_last_rw       = qvk.extent_render.width;
    fsr4_last_rh       = qvk.extent_render.height;
    fsr4_ctx_dw        = qvk.extent_unscaled.width;
    fsr4_ctx_dh        = qvk.extent_unscaled.height;

    Com_Printf("FSR4: context ready %ux%u -> %ux%u\n",
               qvk.extent_render.width,   qvk.extent_render.height,
               qvk.extent_unscaled.width, qvk.extent_unscaled.height);
    return VK_SUCCESS;
}

/* ── vkpt public API ─────────────────────────────────────────────────────── */

static int requested_upscaler(void)
{
    int requested = cvar_flt_upscaler ? cvar_flt_upscaler->integer : 0;

    if (requested >= VKPT_UPSCALER_FSR3 && requested <= VKPT_UPSCALER_FSR3_315)
        return requested;
    /* flt_fsr_enable is retained as a migration/console alias for the old
     * prototype.  The new provider cvar takes precedence whenever nonzero. */
    if (cvar_flt_fsr_enable && cvar_flt_fsr_enable->integer != 0)
        return VKPT_UPSCALER_FSR4;
    return VKPT_UPSCALER_Q2RTX;
}

static int requested_fsr_quality(void)
{
    int quality = cvar_flt_fsr_quality
        ? cvar_flt_fsr_quality->integer
        : VKPT_FSR_QUALITY_PERFORMANCE;
    return Q_clip(quality, VKPT_FSR_QUALITY_NATIVE_AA,
                  VKPT_FSR_QUALITY_ULTRA_PERFORMANCE);
}

static int fsr3_requested_render_scale(void)
{
    switch (requested_fsr_quality()) {
    case VKPT_FSR_QUALITY_NATIVE_AA: return 100;
    case VKPT_FSR_QUALITY_QUALITY: return 67;
    case VKPT_FSR_QUALITY_BALANCED: return 59;
    case VKPT_FSR_QUALITY_ULTRA_PERFORMANCE: return 33;
    default: return 50;
    }
}

static const char *fsr4_requested_model(void)
{
    if (fsr4_dynamic_resolution_requested())
        return "drs";
    switch (requested_fsr_quality()) {
    case VKPT_FSR_QUALITY_NATIVE_AA: return "native";
    case VKPT_FSR_QUALITY_QUALITY: return "quality";
    case VKPT_FSR_QUALITY_BALANCED: return "balanced";
    case VKPT_FSR_QUALITY_ULTRA_PERFORMANCE: return "ultraperf";
    default: return "performance";
    }
}

static int fsr4_requested_render_scale(void)
{
    /* A DRS-model selection returns control to Q2RTX's bounded, profiler-driven
     * DRS controller. Static model selections remain exact model contracts. */
    if (fsr4_dynamic_resolution_requested())
        return 0;
    return fsr3_requested_render_scale();
}

static const char *fsr_quality_label(int quality)
{
    switch (quality) {
    case VKPT_FSR_QUALITY_NATIVE_AA: return "Native AA";
    case VKPT_FSR_QUALITY_QUALITY: return "Quality";
    case VKPT_FSR_QUALITY_BALANCED: return "Balanced";
    case VKPT_FSR_QUALITY_ULTRA_PERFORMANCE: return "Ultra Performance";
    default: return "Performance";
    }
}

int vkpt_fsr_requested_render_scale(void)
{
    switch (requested_upscaler()) {
    case VKPT_UPSCALER_FSR3: return fsr3_requested_render_scale();
    case VKPT_UPSCALER_FSR3_315: return fsr3_requested_render_scale();
    case VKPT_UPSCALER_FSR4: return fsr4_requested_render_scale();
    default: return 0;
    }
}

static bool upscaler_frame_is_eligible(int provider)
{
    float ratio_x;
    float ratio_y;

    if (qvk.device_count != 1 ||
        vkpt_refdef.uniform_buffer.pt_projection != PROJECTION_RECTILINEAR)
        return false;
    if (!qvk.extent_render.width || !qvk.extent_render.height ||
        qvk.extent_render.width > qvk.extent_unscaled.width ||
        qvk.extent_render.height > qvk.extent_unscaled.height)
        return false;

    /* Both providers use the same fixed user ratios. FSR4 selects a distinct
     * trained INT8 graph for each; it must never be dispatched at a ratio
     * belonging to another graph. */
    if (provider == VKPT_UPSCALER_FSR3 || provider == VKPT_UPSCALER_FSR3_315)
        return true;
    if (fsr4_dynamic_resolution_requested())
        return true;

    ratio_x = (float)qvk.extent_unscaled.width /
              (float)max(1u, qvk.extent_render.width);
    ratio_y = (float)qvk.extent_unscaled.height /
              (float)max(1u, qvk.extent_render.height);
    const float expected_ratio = 100.0f /
        (float)fsr4_requested_render_scale();
    return fabsf(ratio_x - expected_ratio) <= 0.02f &&
           fabsf(ratio_y - expected_ratio) <= 0.02f;
}

#ifdef VKPT_FSR3
static void fsr3_fill_device_info(FfxVkPortableDeviceInfo *device_info)
{
    memset(device_info, 0, sizeof(*device_info));
    device_info->structSize = sizeof(*device_info);
    device_info->instance = qvk.instance;
    device_info->physicalDevice = qvk.physical_device;
    device_info->device = qvk.device;
    device_info->getDeviceProcAddr = vkGetDeviceProcAddr;
    device_info->queue = qvk.queue_graphics;
    device_info->queueFamilyIndex = (uint32_t)qvk.queue_idx_graphics;
    /* Report the logical-device feature set, not merely what the physical
     * adapter advertises.  The portable backend uses these bits to avoid
     * selecting optional Vulkan paths that Q2RTX did not enable. */
    device_info->shaderFloat16Enabled = qvk.supports_fp16 ? VK_TRUE : VK_FALSE;
    device_info->debugUtilsEnabled = VK_TRUE;
    device_info->shaderStorageBufferArrayNonUniformIndexingEnabled = VK_TRUE;
    device_info->accelerationStructureEnabled = VK_TRUE;
    device_info->shaderStorageImageWriteWithoutFormatEnabled =
        qvk.supports_storage_image_write_without_format ? VK_TRUE : VK_FALSE;
}

static FfxVkFsr3_3_1_5Resource fsr3_315_import_image(
    VkImage image, VkFormat format, VkExtent2D extent, uint32_t state)
{
    FfxVkFsr3_3_1_5ImportedImageDescription description;
    FfxVkFsr3_3_1_5Bridge *bridge;

    memset(&description, 0, sizeof(description));
    if (!fsr3_315_context || image == VK_NULL_HANDLE ||
        !extent.width || !extent.height)
        return (FfxVkFsr3_3_1_5Resource){0};
    bridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(fsr3_315_context);
    if (!bridge)
        return (FfxVkFsr3_3_1_5Resource){0};
    description.image = image;
    description.format = format;
    description.width = extent.width;
    description.height = extent.height;
    description.mipCount = 1;
    description.arrayLayers = 1;
    /* Q2RTX's global images are held in GENERAL across compute passes. The
     * renderer inserts the producer->compute dependency immediately before
     * dispatch; the bridge returns every imported image to this same layout. */
    description.layout = VK_IMAGE_LAYOUT_GENERAL;
    description.state = state;
    description.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return ffxVkFsr3_3_1_5BridgeImportImage(bridge, &description);
}

static void fsr3_315_release_images(void)
{
    FfxVkFsr3_3_1_5Bridge *bridge;
    FfxVkFsr3_3_1_5Resource *resources[] = {
        &fsr3_315_color, &fsr3_315_depth, &fsr3_315_motion,
        &fsr3_315_reactive, &fsr3_315_composition,
        &fsr3_315_dilated_depth, &fsr3_315_dilated_motion,
        &fsr3_315_previous_depth, &fsr3_315_output,
    };

    if (!fsr3_315_context)
        return;
    bridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(fsr3_315_context);
    if (bridge) {
        for (size_t index = 0; index < LENGTH(resources); ++index) {
            ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, *resources[index]);
            memset(resources[index], 0, sizeof(*resources[index]));
        }
    }
}

static void fsr3_315_destroy_context(void)
{
    if (fsr3_315_context) {
        _VK(vkDeviceWaitIdle(qvk.device));
        /* Descriptor sets retain imported image views until the GPU is done.
         * The context teardown fence above makes releasing those views safe. */
        fsr3_315_release_images();
        ffxVkFsr3_3_1_5UpscalerContextDestroy(fsr3_315_context);
    }
    fsr3_315_context = NULL;
    fsr3_315_context_ok = false;
    fsr3_315_ctx_dw = fsr3_315_ctx_dh = 0;
    fsr3_315_reset_next = true;
}

static VkResult fsr3_315_create_context(void)
{
    FfxVkFsr3_3_1_5UpscalerCreateInfo create_info;
    FfxVkFsr3_3_1_5SharedResourceDescriptions shared;
    FfxVkFsr3_3_1_5Result result;
    VkExtent2D extent = qvk.extent_screen_images;

    fsr3_315_destroy_context();
    if (!qvk.extent_unscaled.width || !qvk.extent_unscaled.height ||
        !extent.width || !extent.height)
        return VK_ERROR_INITIALIZATION_FAILED;
    memset(&create_info, 0, sizeof(create_info));
    create_info.physicalDevice = qvk.physical_device;
    create_info.device = qvk.device;
    create_info.maxRenderWidth = extent.width;
    create_info.maxRenderHeight = extent.height;
    create_info.maxUpscaleWidth = extent.width;
    create_info.maxUpscaleHeight = extent.height;
    create_info.hdrColorInput = VK_TRUE;
    create_info.autoExposure = VK_TRUE;
    result = ffxVkFsr3_3_1_5UpscalerContextCreate(&create_info, &fsr3_315_context);
    if (result != FFX_VK_FSR3_3_1_5_OK) {
        Com_WPrintf("FSR3.1.5: public Vulkan context unavailable (%d); using fallback.\n",
                    (int)result);
        fsr3_315_context = NULL;
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (ffxVkFsr3_3_1_5UpscalerContextGetSharedResourceDescriptions(
            fsr3_315_context, &shared) != FFX_VK_FSR3_3_1_5_OK ||
        shared.dilatedDepth.format != VK_FORMAT_R32_SFLOAT ||
        shared.dilatedMotionVectors.format != VK_FORMAT_R16G16_SFLOAT ||
        shared.reconstructedPrevNearestDepth.format != VK_FORMAT_R32_UINT ||
        shared.dilatedDepth.width > extent.width || shared.dilatedDepth.height > extent.height ||
        shared.dilatedMotionVectors.width > extent.width ||
        shared.dilatedMotionVectors.height > extent.height ||
        shared.reconstructedPrevNearestDepth.width > extent.width ||
        shared.reconstructedPrevNearestDepth.height > extent.height) {
        Com_WPrintf("FSR3.1.5: SDK shared-resource contract is incompatible with Q2RTX images.\n");
        fsr3_315_destroy_context();
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    fsr3_315_color = fsr3_315_import_image(qvk.images[VKPT_IMG_FLAT_COLOR],
        VK_FORMAT_R16G16B16A16_SFLOAT, extent,
        FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    fsr3_315_depth = fsr3_315_import_image(qvk.images[VKPT_IMG_TEMPORAL_DEVICE_DEPTH],
        VK_FORMAT_R32_SFLOAT, extent,
        FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    fsr3_315_motion = fsr3_315_import_image(qvk.images[VKPT_IMG_FLAT_MOTION],
        VK_FORMAT_R16G16B16A16_SFLOAT, extent,
        FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    fsr3_315_reactive = fsr3_315_import_image(qvk.images[VKPT_IMG_TEMPORAL_REACTIVE_MASK],
        VK_FORMAT_R8_UNORM, extent,
        FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    fsr3_315_composition = fsr3_315_import_image(
        qvk.images[VKPT_IMG_TEMPORAL_COMPOSITION_MASK], VK_FORMAT_R8_UNORM,
        extent, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ);
    fsr3_315_dilated_depth = fsr3_315_import_image(
        qvk.images[VKPT_IMG_FSR3_315_DILATED_DEPTH], VK_FORMAT_R32_SFLOAT,
        extent, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    fsr3_315_dilated_motion = fsr3_315_import_image(
        qvk.images[VKPT_IMG_FSR3_315_DILATED_MOTION], VK_FORMAT_R16G16_SFLOAT,
        extent, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    fsr3_315_previous_depth = fsr3_315_import_image(
        qvk.images[VKPT_IMG_FSR3_315_PREVIOUS_DEPTH], VK_FORMAT_R32_UINT,
        extent, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    fsr3_315_output = fsr3_315_import_image(qvk.images[VKPT_IMG_FSR_EASU_OUTPUT],
        VK_FORMAT_R16G16B16A16_SFLOAT, extent,
        FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!fsr3_315_color.resource || !fsr3_315_depth.resource ||
        !fsr3_315_motion.resource || !fsr3_315_reactive.resource ||
        !fsr3_315_composition.resource || !fsr3_315_dilated_depth.resource ||
        !fsr3_315_dilated_motion.resource || !fsr3_315_previous_depth.resource ||
        !fsr3_315_output.resource) {
        Com_WPrintf("FSR3.1.5: failed to import Q2RTX temporal images.\n");
        fsr3_315_destroy_context();
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    fsr3_315_context_ok = true;
    fsr3_315_reset_next = true;
    fsr3_315_ctx_dw = qvk.extent_unscaled.width;
    fsr3_315_ctx_dh = qvk.extent_unscaled.height;
    Com_Printf("FSR3.1.5: public Vulkan experiment ready (max %ux%u).\n",
               extent.width, extent.height);
    return VK_SUCCESS;
}

static void fsr3_destroy_context(void)
{
    if (fsr3_context) {
        /* Most callers have already drained the device (swapchain/pipeline
         * rebuild), but a display extent can also change during startup while
         * the previous context's command buffers are still retiring.  The
         * portable context owns image views referenced by those descriptors;
         * make teardown safe at the ownership boundary.  This is only on
         * context destruction, never on a normal FSR3 dispatch. */
        _VK(vkDeviceWaitIdle(qvk.device));
        FfxVkPortableResult result =
            ffxVkPortableUpscaleContextDestroy(fsr3_context);
        if (result != FFX_VK_PORTABLE_OK)
            Com_WPrintf("FSR3: context destruction failed (%d)\n", (int)result);
    }
    fsr3_context = NULL;
    fsr3_context_ok = false;
    fsr3_ctx_dw = fsr3_ctx_dh = 0;
    fsr3_reset_next = true;
}

static VkResult fsr3_create_context(void)
{
    FfxVkPortableDeviceInfo device_info;
    FfxVkPortableUpscaleCreateInfo create_info;
    FfxVkPortableResult result;

    fsr3_destroy_context();
    if (!qvk.extent_unscaled.width || !qvk.extent_unscaled.height)
        return VK_ERROR_INITIALIZATION_FAILED;

    fsr3_fill_device_info(&device_info);

    memset(&create_info, 0, sizeof(create_info));
    create_info.structSize = sizeof(create_info);
    /* The upscaler receives pre-tone-map FLAT_COLOR, not the presentation
     * surface. Its value scale is linear HDR irrespective of SDR/HDR output. */
    /* The current SDK 3.1.5 fallback reports exposure as a required input
     * unless the provider owns its automatic-exposure graph. Q2RTX does not
     * have a separate sampled 1x1 exposure texture for FSR3, so enable the
     * portable implementation's exact internal auto-exposure path instead of
     * passing a null resource and relying on a provider-specific identity
     * fallback. */
    create_info.flags = FFX_VK_PORTABLE_CONTEXT_HDR_COLOR_INPUT |
        FFX_VK_PORTABLE_CONTEXT_AUTO_EXPOSURE;
    if (qvk.enable_validation)
        create_info.flags |= FFX_VK_PORTABLE_CONTEXT_DEBUG_CHECKING;
    create_info.maxRenderSize.width = qvk.extent_unscaled.width;
    create_info.maxRenderSize.height = qvk.extent_unscaled.height;
    create_info.maxOutputSize = create_info.maxRenderSize;

    result = ffxVkPortableUpscaleContextCreate(
        &device_info, &create_info, &fsr3_context);
    if (result != FFX_VK_PORTABLE_OK) {
        Com_WPrintf("FSR3: native Vulkan 3.1.4 context unavailable (%d); "
                    "using fallback.\n", (int)result);
        fsr3_context = NULL;
        return result == FFX_VK_PORTABLE_ERROR_OUT_OF_MEMORY
            ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_FEATURE_NOT_PRESENT;
    }

    fsr3_context_ok = true;
    fsr3_reset_next = true;
    fsr3_ctx_dw = qvk.extent_unscaled.width;
    fsr3_ctx_dh = qvk.extent_unscaled.height;
    Com_Printf("FSR3: native Vulkan upscaler 3.1.4 ready (max %ux%u).\n",
               fsr3_ctx_dw, fsr3_ctx_dh);
    return VK_SUCCESS;
}

static void fsr3_destroy_frame_generation_context(void)
{
    if (fsr3_frame_generation_context || fsr3_316_frame_generation_context) {
        /* Generated and real presents use separate submissions.  A resize or
         * startup extent transition must retire both before the FI/OF context
         * frees descriptor-referenced image views. */
        _VK(vkDeviceWaitIdle(qvk.device));
        if (fsr3_frame_generation_context) {
            FfxVkPortableResult result = ffxVkPortableFrameGenerationContextDestroy(
                fsr3_frame_generation_context);
            if (result != FFX_VK_PORTABLE_OK)
                Com_WPrintf("FSR3 FG: legacy context destruction failed (%d)\n", (int)result);
        }
        if (fsr3_316_frame_generation_context)
            ffxVkFsr3_3_1_6FrameGenerationContextDestroy(
                fsr3_316_frame_generation_context);
    }
    fsr3_frame_generation_context = NULL;
    fsr3_frame_generation_context_ok = false;
    fsr3_316_frame_generation_context = NULL;
    fsr3_316_frame_generation_context_ok = false;
    memset(fsr3_316_frame_generation_frame_ids, 0,
        sizeof(fsr3_316_frame_generation_frame_ids));
    fsr3_frame_generation_dw = fsr3_frame_generation_dh = 0;
    fsr3_frame_generation_reset_next = true;
}

static VkResult fsr3_create_frame_generation_context(void)
{
    FfxVkPortableDeviceInfo device_info;
    FfxVkPortableFrameGenerationCreateInfo create_info;
    FfxVkPortableResult result;

    fsr3_destroy_frame_generation_context();
    if (!qvk.extent_unscaled.width || !qvk.extent_unscaled.height)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (cvar_flt_frame_generation_backend &&
        cvar_flt_frame_generation_backend->integer == 1) {
        FfxVkFsr3_3_1_6FrameGenerationCreateInfo create_316;
        FfxVkFsr3_3_1_6FrameGenerationResult result_316;

        /* Q2RTX's HUDless presentation surface remains RGBA16F in either
         * mode. The public FI/OF source selects sRGB or scRGB at dispatch
         * time, so no distinct shader permutation is required for HDR. */
        memset(&create_316, 0, sizeof(create_316));
        create_316.physicalDevice = qvk.physical_device;
        create_316.device = qvk.device;
        create_316.maxRenderWidth = qvk.extent_unscaled.width;
        create_316.maxRenderHeight = qvk.extent_unscaled.height;
        create_316.displayWidth = qvk.extent_unscaled.width;
        create_316.displayHeight = qvk.extent_unscaled.height;
        create_316.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        result_316 = ffxVkFsr3_3_1_6FrameGenerationContextCreate(
            &create_316, &fsr3_316_frame_generation_context);
        if (result_316 != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
            Com_WPrintf("FSR3 FG 3.1.6: Vulkan context unavailable (%d)\n", (int)result_316);
            fsr3_316_frame_generation_context = NULL;
            return result_316 == FFX_VK_FSR3_3_1_6_FRAMEGEN_ERROR_OUT_OF_MEMORY
                ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_FEATURE_NOT_PRESENT;
        }
        fsr3_316_frame_generation_context_ok = true;
        fsr3_frame_generation_reset_next = true;
        fsr3_frame_generation_dw = qvk.extent_unscaled.width;
        fsr3_frame_generation_dh = qvk.extent_unscaled.height;
        Com_Printf("FSR3 FG: SDK 3.1.6 Vulkan Optical Flow/Frame Interpolation ready "
                   "(max %ux%u).\n", fsr3_frame_generation_dw,
                   fsr3_frame_generation_dh);
        return VK_SUCCESS;
    }

    fsr3_fill_device_info(&device_info);
    memset(&create_info, 0, sizeof(create_info));
    create_info.structSize = sizeof(create_info);
    /* FI sees HUDless TAA_OUTPUT after Q2RTX's tone mapper. Its dynamic range
     * must match that presentation-domain source, unlike the upscaler which
     * consumes pre-tone-map linear HDR scene colour. */
    create_info.flags = qvk.surf_is_hdr
        ? FFX_VK_PORTABLE_CONTEXT_HDR_COLOR_INPUT : 0;
    if (qvk.enable_validation)
        create_info.flags |= FFX_VK_PORTABLE_CONTEXT_DEBUG_CHECKING;
    /* Max render size deliberately uses presentation size so DRS/preset
     * changes do not invalidate the FI allocation. */
    create_info.maxRenderSize.width = qvk.extent_unscaled.width;
    create_info.maxRenderSize.height = qvk.extent_unscaled.height;
    create_info.displaySize.width = qvk.extent_unscaled.width;
    create_info.displaySize.height = qvk.extent_unscaled.height;
    create_info.interpolationSourceFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    create_info.outputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    result = ffxVkPortableFrameGenerationContextCreate(
        &device_info, &create_info, &fsr3_frame_generation_context);
    if (result != FFX_VK_PORTABLE_OK) {
        Com_WPrintf("FSR3 FG: native Vulkan context unavailable (%d)\n", (int)result);
        fsr3_frame_generation_context = NULL;
        return result == FFX_VK_PORTABLE_ERROR_OUT_OF_MEMORY
            ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_FEATURE_NOT_PRESENT;
    }

    fsr3_frame_generation_context_ok = true;
    fsr3_frame_generation_reset_next = true;
    fsr3_frame_generation_dw = qvk.extent_unscaled.width;
    fsr3_frame_generation_dh = qvk.extent_unscaled.height;
    Com_Printf("FSR3 FG: native Vulkan Optical Flow/Frame Interpolation ready "
               "(max %ux%u).\n", fsr3_frame_generation_dw,
               fsr3_frame_generation_dh);
    return VK_SUCCESS;
}

static bool fsr3_is_enabled(void)
{
    return requested_upscaler() == VKPT_UPSCALER_FSR3 &&
           fsr3_context_ok && fsr3_context &&
           fsr3_ctx_dw == qvk.extent_unscaled.width &&
           fsr3_ctx_dh == qvk.extent_unscaled.height &&
           upscaler_frame_is_eligible(VKPT_UPSCALER_FSR3);
}

static bool fsr3_315_is_enabled(void)
{
    return requested_upscaler() == VKPT_UPSCALER_FSR3_315 &&
           fsr3_315_context_ok && fsr3_315_context &&
           fsr3_315_ctx_dw == qvk.extent_unscaled.width &&
           fsr3_315_ctx_dh == qvk.extent_unscaled.height &&
           upscaler_frame_is_eligible(VKPT_UPSCALER_FSR3_315);
}
#endif

static bool fsr4_is_enabled(void)
{
    if (requested_upscaler() != VKPT_UPSCALER_FSR4)
        return false;
    if (!fsr4_backend_ok || !fsr4_context_ok)
        return false;
    if (!fsr4_output_extent_supported(qvk.extent_unscaled))
        return false;
    if (strcmp(fsr4_shader_tier, display_res_tag()) != 0)
        return false;
    if (strcmp(fsr4_shader_model, fsr4_requested_model()) != 0)
        return false;
    return upscaler_frame_is_eligible(VKPT_UPSCALER_FSR4);
}

static void publish_upscaler_status(int active, const char *reason)
{
    static int last_active = -1;
    static char last_reason[128];

    if (!cvar_flt_upscaler_active || !cvar_flt_upscaler_reason)
        return;
    if (active != last_active || strcmp(reason, last_reason) != 0) {
        char active_text[8];
        Q_snprintf(active_text, sizeof(active_text), "%d", active);
        Cvar_SetByVar(cvar_flt_upscaler_active, active_text, FROM_CODE);
        Cvar_SetByVar(cvar_flt_upscaler_reason, reason, FROM_CODE);
        Com_Printf("Upscaler: %s\n", reason);
        last_active = active;
        Q_strlcpy(last_reason, reason, sizeof(last_reason));
    }
}

static bool resolve_upscaler(int *active, const char **reason)
{
    const int requested = requested_upscaler();

    *active = VKPT_UPSCALER_Q2RTX;
    if (requested == VKPT_UPSCALER_Q2RTX) {
        *reason = "Q2RTX fallback selected";
        return false;
    }
    if (qvk.device_count != 1) {
        *reason = "fallback: temporal inputs require one GPU";
        return false;
    }
    if (vkpt_refdef.uniform_buffer.pt_projection != PROJECTION_RECTILINEAR) {
        *reason = "fallback: non-rectilinear projection";
        return false;
    }
    if (!qvk.extent_render.width || !qvk.extent_render.height ||
        qvk.extent_render.width > qvk.extent_unscaled.width ||
        qvk.extent_render.height > qvk.extent_unscaled.height) {
        *reason = "fallback: invalid temporal render extent";
        return false;
    }

#ifdef VKPT_FSR3
    if (requested == VKPT_UPSCALER_FSR3) {
        if (!fsr3_context_ok || !fsr3_context) {
            *reason = "fallback: FSR3 Vulkan context unavailable";
            return false;
        }
        if (fsr3_ctx_dw != qvk.extent_unscaled.width ||
            fsr3_ctx_dh != qvk.extent_unscaled.height) {
            *reason = "fallback: FSR3 context resize pending";
            return false;
        }
        *active = VKPT_UPSCALER_FSR3;
        *reason = "FSR3 3.1.4 native Vulkan active";
        return true;
    }
    if (requested == VKPT_UPSCALER_FSR3_315) {
        if (!fsr3_315_context_ok || !fsr3_315_context) {
            *reason = "fallback: FSR3 3.1.5 Vulkan context unavailable";
            return false;
        }
        if (fsr3_315_ctx_dw != qvk.extent_unscaled.width ||
            fsr3_315_ctx_dh != qvk.extent_unscaled.height) {
            *reason = "fallback: FSR3 3.1.5 context resize pending";
            return false;
        }
        *active = VKPT_UPSCALER_FSR3_315;
        *reason = "FSR3 3.1.5 public-SDK Vulkan experiment active";
        return true;
    }
#endif

    if (!fsr4_backend_ok || !fsr4_context_ok) {
        *reason = fsr4_unavailable_reason[0]
            ? fsr4_unavailable_reason
            : "fallback: FSR4 v07 context unavailable";
        return false;
    }
    if (!fsr4_output_extent_supported(qvk.extent_unscaled)) {
        *reason = "fallback: FSR4 v07 output exceeds 8K limit";
        return false;
    }
    if (strcmp(fsr4_shader_tier, display_res_tag()) != 0) {
        *reason = "fallback: FSR4 v07 shader tier resize pending";
        return false;
    }
    if (strcmp(fsr4_shader_model, fsr4_requested_model()) != 0) {
        *reason = "fallback: FSR4 v07 quality model switch pending";
        return false;
    }
    if (!upscaler_frame_is_eligible(VKPT_UPSCALER_FSR4)) {
        *reason = "fallback: FSR4 v07 render ratio does not match model";
        return false;
    }
    *active = VKPT_UPSCALER_FSR4;
    {
        static char fsr4_reason[128];
        if (fsr4_dynamic_resolution_requested()) {
            Q_snprintf(fsr4_reason, sizeof(fsr4_reason),
                       "FSR4 v07 INT8/DOT4 DRS model active");
        } else {
            Q_snprintf(fsr4_reason, sizeof(fsr4_reason),
                       "FSR4 v07 INT8/DOT4 %s active",
                       fsr_quality_label(requested_fsr_quality()));
        }
        *reason = fsr4_reason;
    }
    return true;
}

void vkpt_fsr_init_cvars(void)
{
    cvar_flt_upscaler     = Cvar_Get("flt_upscaler",     "0",   CVAR_ARCHIVE);
    cvar_flt_fsr_quality  = Cvar_Get("flt_fsr_quality",  "3",   CVAR_ARCHIVE);
    cvar_flt_fsr3_sharpening = Cvar_Get("flt_fsr3_sharpening", "0",
                                        CVAR_ARCHIVE);
    cvar_flt_fsr4_sharpening = Cvar_Get("flt_fsr4_sharpening", "0",
                                        CVAR_ARCHIVE);
    cvar_flt_fsr4_auto_exposure = Cvar_Get("flt_fsr4_auto_exposure", "0",
                                           CVAR_ARCHIVE);
    cvar_flt_fsr4_dynamic_resolution = Cvar_Get(
        "flt_fsr4_dynamic_resolution", "0", CVAR_ARCHIVE);
    /* 0 = off, 1 = analytical FSR3 Frame Interpolation.  It remains gated by
     * the explicit presenter prerequisites; selecting it never replaces the
     * normal present path until that presenter has acquired two images. */
    cvar_flt_frame_generation = Cvar_Get("flt_frame_generation", "0",
                                         CVAR_ARCHIVE);
    /* Keep the live-validated 1.1.4 provider as the default. The newer SDK
     * 2.3 FI/OF path is separately selectable until its presenter path has
     * equivalent long-duration game coverage. */
    cvar_flt_frame_generation_backend = Cvar_Get("flt_frame_generation_backend", "0",
                                                 CVAR_ARCHIVE);
    /* Analytical interpolation is most convincing at a sustained high input
     * rate. Thirty is a conservative default safety floor; set zero to
     * explicitly disable the gate, or sixty for AMD's recommended target. */
    cvar_flt_frame_generation_min_rendered_fps = Cvar_Get(
        "flt_frame_generation_min_rendered_fps", "30", CVAR_ARCHIVE);
    cvar_flt_frame_generation_active = Cvar_Get("flt_frame_generation_active", "0",
        CVAR_ROM | CVAR_NOARCHIVE);
    cvar_flt_frame_generation_reason = Cvar_Get("flt_frame_generation_reason", "startup",
        CVAR_ROM | CVAR_NOARCHIVE);
    cvar_flt_frame_generation_rendered_fps = Cvar_Get(
        "flt_frame_generation_rendered_fps", "0", CVAR_ROM | CVAR_NOARCHIVE);
    cvar_flt_frame_generation_generated_fps = Cvar_Get(
        "flt_frame_generation_generated_fps", "0", CVAR_ROM | CVAR_NOARCHIVE);
    cvar_flt_upscaler_active = Cvar_Get("flt_upscaler_active", "0",
        CVAR_ROM | CVAR_NOARCHIVE);
    cvar_flt_upscaler_reason = Cvar_Get("flt_upscaler_reason", "startup",
        CVAR_ROM | CVAR_NOARCHIVE);
    cvar_flt_fsr_enable    = Cvar_Get("flt_fsr_enable",    "0",   CVAR_ARCHIVE);
    cvar_flt_fsr_sharpness = Cvar_Get("flt_fsr_sharpness", "0.2", CVAR_ARCHIVE);
    cvar_flt_fsr_easu      = Cvar_Get("flt_fsr_easu",      "1",   CVAR_ARCHIVE);
    cvar_flt_fsr_rcas      = Cvar_Get("flt_fsr_rcas",      "1",   CVAR_ARCHIVE);

    if (cvar_flt_upscaler->integer == VKPT_UPSCALER_Q2RTX &&
        cvar_flt_fsr_enable->integer != 0) {
        Cvar_SetByVar(cvar_flt_upscaler, "2", FROM_CODE);
        Cvar_SetByVar(cvar_flt_fsr_enable, "0", FROM_CODE);
        Com_Printf("Upscaler: migrated flt_fsr_enable to flt_upscaler 2.\n");
    }
}

void vkpt_fsr_request_reset(void)
{
    fsr4_reset_next = true;
#ifdef VKPT_FSR3
    fsr3_reset_next = true;
    fsr3_315_reset_next = true;
    /* Frame interpolation has independent optical-flow/history state. A
     * camera cut, projection change, provider switch, or temporal input reset
     * must invalidate it even when the upscaler remains usable. */
    fsr3_frame_generation_reset_next = true;
    fsr3_fg_low_rate_frames = 0;
    fsr3_fg_recovery_frames = 0;
    fsr3_fg_rate_blocked = false;
#endif
}

VkResult vkpt_fsr_initialize(void)
{
    size_t scratch_sz = ffxFsr4VkGetScratchMemorySize();

    if (!fsr4_device_is_capable())
        return VK_SUCCESS;

    fsr4_scratch = Z_Malloc(scratch_sz);
    if (fsr4_create_backend_for_tier(display_res_tag(),
                                     fsr4_requested_model()) != VK_SUCCESS) {
        Z_Free(fsr4_scratch);
        fsr4_scratch = NULL;
    }
    /* FSR is optional.  A missing/unsupported model leaves the existing TAA
     * path active instead of failing renderer initialization. */
    return VK_SUCCESS;
}

VkResult vkpt_fsr_destroy(void)
{
#ifdef VKPT_FSR3
    fsr3_destroy_frame_generation_context();
    fsr3_destroy_context();
    fsr3_315_destroy_context();
#endif
    fsr4_destroy_backend();
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
    /* Recreate if context doesn't exist or display resolution changed.
       Dynamic render resolution changes are handled without recreation
       since maxRenderSize = display resolution.  But if the DISPLAY resolution
       changed (window resize, video mode change), the context must be rebuilt
       because all internal textures are sized to display resolution.
       Q2RTX calls destroy_pipelines + create_pipelines between frames (after
       waiting for device idle), so it's safe to destroy here. */
    bool display_changed = fsr4_context_ok &&
        (fsr4_ctx_dw != qvk.extent_unscaled.width ||
         fsr4_ctx_dh != qvk.extent_unscaled.height);

#ifdef VKPT_FSR3
    if (!fsr3_context_ok ||
        fsr3_ctx_dw != qvk.extent_unscaled.width ||
        fsr3_ctx_dh != qvk.extent_unscaled.height) {
        (void)fsr3_create_context();
    }
    if (!fsr3_315_context_ok ||
        fsr3_315_ctx_dw != qvk.extent_unscaled.width ||
        fsr3_315_ctx_dh != qvk.extent_unscaled.height) {
        (void)fsr3_315_create_context();
    }
    if (cvar_flt_frame_generation && cvar_flt_frame_generation->integer != 0 &&
        (!fsr3_frame_generation_context_ok ||
         fsr3_frame_generation_dw != qvk.extent_unscaled.width ||
         fsr3_frame_generation_dh != qvk.extent_unscaled.height)) {
        (void)fsr3_create_frame_generation_context();
    }
#endif

    if (fsr4_scratch) {
        if (!fsr4_context_ok || display_changed ||
            strcmp(fsr4_shader_tier, display_res_tag()) != 0 ||
            strcmp(fsr4_shader_model, fsr4_requested_model()) != 0) {
            VkResult result = fsr4_recreate_context();
            if (result != VK_SUCCESS)
                Com_WPrintf("FSR4: pipeline/context creation unavailable (%d); using fallback.\n",
                            result);
        }
    }
    return VK_SUCCESS;
}

VkResult vkpt_fsr_destroy_pipelines(void)
{
    /* Shader reload and swapchain recreation happen after a device-idle wait.
     * Tear down the complete tier-specific backend so coherent pre/model/post
     * pipelines and persistent resource extents are rebuilt together. */
#ifdef VKPT_FSR3
    fsr3_destroy_frame_generation_context();
    fsr3_destroy_context();
    fsr3_315_destroy_context();
#endif
    fsr4_destroy_backend();
    return VK_SUCCESS;
}

bool vkpt_fsr_is_requested(void)
{
    return requested_upscaler() != VKPT_UPSCALER_Q2RTX;
}

bool vkpt_fsr_is_enabled(void)
{
    int active;
    const char *reason;
    bool enabled = resolve_upscaler(&active, &reason);

    publish_upscaler_status(active, reason);
    return enabled;
}

/* FSR4 handles its own upscaling – no separate pre-upscale step needed */
bool vkpt_fsr_needs_upscale(void)
{
    return false;
}

uint32_t vkpt_fsr_jitter_phase_count(void)
{
    int32_t phase_count = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount query;

    if (!vkpt_fsr_is_enabled())
        return 0;
#ifdef VKPT_FSR3
    if (fsr3_is_enabled()) {
        float ratio = (float)qvk.extent_unscaled.width /
                      (float)max(1u, qvk.extent_render.width);
        return (uint32_t)max(1, (int)(8.0f * ratio * ratio));
    }
#endif
    memset(&query, 0, sizeof(query));
    query.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_JITTER_PHASE_COUNT;
    query.displayWidth = qvk.extent_unscaled.width;
    query.renderWidth = qvk.extent_render.width;
    query.pOutPhaseCount = &phase_count;
    if (ffxQuery(&fsr4_context, &query.header) != FFX_API_RETURN_OK ||
        phase_count <= 0)
        return 0;
    return (uint32_t)phase_count;
}

/* No-op: FSR4 drives its own constants via the FfxInterface */
void vkpt_fsr_update_ubo(QVKUniformBuffer_t *ubo)
{
    (void)ubo;
}

#ifdef VKPT_FSR3
static FfxVkPortableImage fsr3_temporal_image(
    const VkptTemporalImage *image, FfxVkPortableResourceState state)
{
    FfxVkPortableImage result;

    memset(&result, 0, sizeof(result));
    result.structSize = sizeof(result);
    result.image = image->image;
    result.format = image->format;
    result.extent.width = image->allocation_extent.width;
    result.extent.height = image->allocation_extent.height;
    result.mipCount = 1;
    result.arrayLayers = 1;
    result.usage = image->usage;
    result.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    result.state = state;
    return result;
}

static FfxVkPortableImage fsr3_output_image(void)
{
    FfxVkPortableImage result;

    memset(&result, 0, sizeof(result));
    result.structSize = sizeof(result);
    result.image = qvk.images[VKPT_IMG_FSR_EASU_OUTPUT];
    result.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    result.extent.width = qvk.extent_screen_images.width;
    result.extent.height = qvk.extent_screen_images.height;
    result.mipCount = 1;
    result.arrayLayers = 1;
    result.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    result.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    result.state = FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS;
    return result;
}

static FfxVkPortableImage fsr3_screen_image(
    unsigned int image_index, VkExtent2D extent,
    FfxVkPortableResourceState state)
{
    FfxVkPortableImage result;

    memset(&result, 0, sizeof(result));
    result.structSize = sizeof(result);
    result.image = qvk.images[image_index];
    result.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    result.extent.width = extent.width;
    result.extent.height = extent.height;
    result.mipCount = 1;
    result.arrayLayers = 1;
    result.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    result.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    result.state = state;
    return result;
}

bool vkpt_fsr_frame_generation_is_ready(void)
{
    const int upscaler = requested_upscaler();

    if (!cvar_flt_frame_generation || cvar_flt_frame_generation->integer == 0)
        return false;
    /* Optical flow and frame interpolation consume the provider-neutral
     * temporal contract and presentation-domain scene color, not either
     * upscaler's private history. The public 3.1.5 upscaler therefore has the
     * same valid analytical-FG input contract as the proven 1.1.4 path. */
    if (!((upscaler == VKPT_UPSCALER_FSR3 && fsr3_is_enabled()) ||
          (upscaler == VKPT_UPSCALER_FSR3_315 && fsr3_315_is_enabled())))
        return false;
    const bool sdk_316 = cvar_flt_frame_generation_backend &&
        cvar_flt_frame_generation_backend->integer == 1;
    const bool provider_ready = sdk_316
        ? fsr3_316_frame_generation_context_ok && fsr3_316_frame_generation_context
        : fsr3_frame_generation_context_ok && fsr3_frame_generation_context;
    return provider_ready &&
           fsr3_frame_generation_dw == qvk.extent_unscaled.width &&
           fsr3_frame_generation_dh == qvk.extent_unscaled.height &&
           qvk.extent_taa_output.width == qvk.extent_unscaled.width &&
           qvk.extent_taa_output.height == qvk.extent_unscaled.height;
}

static bool fsr3_frame_generation_rate_is_eligible(void)
{
    const VkptTemporalFrame *frame = vkpt_temporal_get_frame();
    const float minimum_fps = cvar_flt_frame_generation_min_rendered_fps
        ? Q_clipf(cvar_flt_frame_generation_min_rendered_fps->value, 0.0f, 240.0f)
        : 0.0f;
    float input_fps;

    if (minimum_fps <= 0.0f) {
        fsr3_fg_low_rate_frames = 0;
        fsr3_fg_recovery_frames = 0;
        fsr3_fg_rate_blocked = false;
        return true;
    }
    /* There is no completed frame-rate sample on first use. Let the normal
     * temporal reset warm up, then evaluate completed logical frames. */
    if (!frame || frame->frame_time_ms <= 0.0f || frame->frame_time_ms > 1000.0f)
        return !fsr3_fg_rate_blocked;

    input_fps = 1000.0f / frame->frame_time_ms;
    if (!fsr3_fg_rate_blocked) {
        if (input_fps < minimum_fps) {
            fsr3_fg_recovery_frames = 0;
            if (++fsr3_fg_low_rate_frames >= 4u)
                fsr3_fg_rate_blocked = true;
        } else {
            fsr3_fg_low_rate_frames = 0;
        }
    } else if (input_fps >= minimum_fps + 2.0f) {
        fsr3_fg_low_rate_frames = 0;
        if (++fsr3_fg_recovery_frames >= 8u) {
            fsr3_fg_recovery_frames = 0;
            fsr3_fg_rate_blocked = false;
        }
    } else {
        fsr3_fg_recovery_frames = 0;
    }
    return !fsr3_fg_rate_blocked;
}

void vkpt_fsr_frame_generation_publish_status(bool active, const char *reason)
{
    static int last_active = -1;
    static char last_reason[128];
    char active_text[8];

    /* Publishing a fallback must not leave the last active presentation rate
     * in the diagnostics. The private estimator state is reset here on every
     * transition away from an active generated→real presenter. */
    if (!active) {
        /* A present/acquire/temporal fallback creates a discontinuity in the
         * generated cadence. Do not reuse optical-flow or interpolation
         * history when the two-image presenter becomes available again. */
        fsr3_frame_generation_reset_next = true;
        fsr3_fg_last_present_msec = 0;
        fsr3_fg_filtered_rendered_fps = 0.0f;
        if (cvar_flt_frame_generation_rendered_fps)
            Cvar_SetByVar(cvar_flt_frame_generation_rendered_fps, "0", FROM_CODE);
        if (cvar_flt_frame_generation_generated_fps)
            Cvar_SetByVar(cvar_flt_frame_generation_generated_fps, "0", FROM_CODE);
    }

    if (!cvar_flt_frame_generation_active || !cvar_flt_frame_generation_reason)
        return;
    if (active == (last_active != 0) && reason && !strcmp(reason, last_reason))
        return;
    Q_snprintf(active_text, sizeof(active_text), "%d", active ? 1 : 0);
    Cvar_SetByVar(cvar_flt_frame_generation_active, active_text, FROM_CODE);
    Cvar_SetByVar(cvar_flt_frame_generation_reason,
        reason ? reason : "unspecified", FROM_CODE);
    Com_Printf("Frame generation: %s\n", reason ? reason : "unspecified");
    last_active = active ? 1 : 0;
    Q_strlcpy(last_reason, reason ? reason : "unspecified", sizeof(last_reason));
}

void vkpt_fsr_frame_generation_note_present_pair(void)
{
    /* This is deliberately presentation cadence, not a GPU timestamp or the
     * game tick rate. A sample is recorded only after both generated and real
     * images were accepted by the WSI path, therefore generated FPS is exactly
     * twice the rendered-frame cadence for this 1:1 interpolator. */
    unsigned now;
    unsigned elapsed;
    float instant_fps;
    char rendered_text[32];
    char generated_text[32];

    if (!cvar_flt_frame_generation_rendered_fps ||
        !cvar_flt_frame_generation_generated_fps)
        return;

    now = Sys_Milliseconds();
    if (fsr3_fg_last_present_msec == 0) {
        fsr3_fg_last_present_msec = now;
        return;
    }
    elapsed = now - fsr3_fg_last_present_msec;
    fsr3_fg_last_present_msec = now;
    if (!elapsed || elapsed > 1000u) {
        fsr3_fg_filtered_rendered_fps = 0.0f;
        return;
    }

    instant_fps = 1000.0f / (float)elapsed;
    fsr3_fg_filtered_rendered_fps = fsr3_fg_filtered_rendered_fps > 0.0f
        ? fsr3_fg_filtered_rendered_fps * 0.85f + instant_fps * 0.15f
        : instant_fps;
    Q_snprintf(rendered_text, sizeof(rendered_text), "%.1f", fsr3_fg_filtered_rendered_fps);
    Q_snprintf(generated_text, sizeof(generated_text), "%.1f",
               fsr3_fg_filtered_rendered_fps * 2.0f);
    Cvar_SetByVar(cvar_flt_frame_generation_rendered_fps, rendered_text, FROM_CODE);
    Cvar_SetByVar(cvar_flt_frame_generation_generated_fps, generated_text, FROM_CODE);
}

bool vkpt_fsr_frame_generation_prepare_present(void)
{
    if (!cvar_flt_frame_generation || cvar_flt_frame_generation->integer == 0)
        return false;
    if (!fsr3_frame_generation_rate_is_eligible()) {
        char reason[128];
        Q_snprintf(reason, sizeof(reason),
            "fallback: rendered input below %.0f FPS threshold",
            cvar_flt_frame_generation_min_rendered_fps->value);
        vkpt_fsr_frame_generation_publish_status(false, reason);
        return false;
    }
    /* Context creation is CPU-only. This call is made after Q2RTX waited for
     * the frame-slot fence and before it acquires either presentation image.
     * We intentionally keep a disabled context alive until normal pipeline
     * teardown, rather than destroying resources still referenced by another
     * in-flight frame. */
    const bool sdk_316 = cvar_flt_frame_generation_backend &&
        cvar_flt_frame_generation_backend->integer == 1;
    const bool selected_context_ok = sdk_316
        ? fsr3_316_frame_generation_context_ok && fsr3_316_frame_generation_context
        : fsr3_frame_generation_context_ok && fsr3_frame_generation_context;
    if (!selected_context_ok &&
        fsr3_create_frame_generation_context() != VK_SUCCESS)
        return false;
    return vkpt_fsr_frame_generation_is_ready();
}

static FfxVkFsr3_3_1_6FrameGenerationImage fsr3_316_image(
    unsigned int image_index, VkExtent2D extent, VkImageLayout layout)
{
    FfxVkFsr3_3_1_6FrameGenerationImage result;

    memset(&result, 0, sizeof(result));
    result.image = qvk.images[image_index];
    result.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    result.width = extent.width;
    result.height = extent.height;
    result.layout = layout;
    result.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    return result;
}

static FfxVkFsr3_3_1_6FrameGenerationImage fsr3_316_temporal_image(
    const VkptTemporalImage *image)
{
    FfxVkFsr3_3_1_6FrameGenerationImage result;

    memset(&result, 0, sizeof(result));
    result.image = image->image;
    result.format = image->format;
    result.width = image->allocation_extent.width;
    result.height = image->allocation_extent.height;
    result.layout = VK_IMAGE_LAYOUT_GENERAL;
    result.usage = image->usage;
    return result;
}

static VkResult fsr3_316_frame_generation_record_after_inputs(
    VkCommandBuffer cmd_buf, const VkptTemporalFrame *frame)
{
    FfxVkFsr3_3_1_6FrameGenerationPrepareInfo prepare;
    FfxVkFsr3_3_1_6FrameGenerationDispatchInfo dispatch;
    FfxVkFsr3_3_1_6FrameGenerationResult result;

    if (!fsr3_316_frame_generation_context)
        return VK_NOT_READY;
    /* If either SDK call rejects after recording partial work, the normal
     * fallback still submits this post command buffer. Retire its imported
     * views at this frame slot's fence rather than leaking/stalling the next
     * logical frame. */
    fsr3_316_frame_generation_frame_ids[qvk.current_frame_index] = frame->frame_id;
    memset(&prepare, 0, sizeof(prepare));
    prepare.commandBuffer = cmd_buf;
    prepare.color = fsr3_316_image(VKPT_IMG_TAA_OUTPUT, qvk.extent_taa_output,
        VK_IMAGE_LAYOUT_GENERAL);
    prepare.depth = fsr3_316_temporal_image(&frame->inputs.device_depth);
    prepare.motionVectors = fsr3_316_temporal_image(&frame->inputs.motion_vectors);
    prepare.renderWidth = frame->render_size.width;
    prepare.renderHeight = frame->render_size.height;
    prepare.jitterOffsetX = -frame->camera.jitter_render_pixels[0];
    prepare.jitterOffsetY = -frame->camera.jitter_render_pixels[1];
    prepare.motionVectorScaleX = frame->inputs.motion_description.to_render_pixels[0];
    prepare.motionVectorScaleY = frame->inputs.motion_description.to_render_pixels[1];
    prepare.frameTimeMilliseconds = frame->frame_time_ms > 0.0f
        ? frame->frame_time_ms : 16.667f;
    prepare.minLuminance = 0.0f;
    prepare.maxLuminance = qvk.surf_is_hdr ? 1000.0f : 1.0f;
    prepare.transferFunction = qvk.surf_is_hdr
        ? FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SCRGB
        : FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
    prepare.cameraNear = frame->camera.near_plane;
    prepare.cameraFar = frame->camera.far_plane;
    prepare.viewSpaceToMeters = frame->camera.view_space_to_meters;
    prepare.cameraVerticalFovRadians = frame->camera.vertical_fov_radians;
    memcpy(prepare.cameraPosition, frame->camera.position, sizeof(prepare.cameraPosition));
    memcpy(prepare.cameraUp, frame->camera.up, sizeof(prepare.cameraUp));
    memcpy(prepare.cameraRight, frame->camera.right, sizeof(prepare.cameraRight));
    memcpy(prepare.cameraForward, frame->camera.forward, sizeof(prepare.cameraForward));
    prepare.frameId = frame->frame_id;
    prepare.reset = (fsr3_frame_generation_reset_next || !frame->history_valid)
        ? VK_TRUE : VK_FALSE;
    result = ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(
        fsr3_316_frame_generation_context, &prepare);
    if (result != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
        fsr3_frame_generation_reset_next = true;
        Com_WPrintf("FSR3 FG 3.1.6: prepare failed (%d)\n", (int)result);
        return VK_ERROR_UNKNOWN;
    }
    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.commandBuffer = cmd_buf;
    dispatch.color = prepare.color;
    dispatch.output = fsr3_316_image(VKPT_IMG_FSR_RCAS_OUTPUT,
        qvk.extent_unscaled, VK_IMAGE_LAYOUT_GENERAL);
    /* Q2RTX's only full-screen displacement (underwater water warp) is
     * intentionally applied by final_blit after FI to both the generated and
     * real presentations.  Do not synthesize an input here: the SDK field
     * means UV_after - UV_before for an actual pre-FI distortion pass.  Leave
     * it null until the renderer owns such a surface. */
    dispatch.displayWidth = frame->display_size.width;
    dispatch.displayHeight = frame->display_size.height;
    dispatch.interpolationWidth = frame->display_size.width;
    dispatch.interpolationHeight = frame->display_size.height;
    dispatch.frameTimeMilliseconds = prepare.frameTimeMilliseconds;
    dispatch.cameraNear = prepare.cameraNear;
    dispatch.cameraFar = prepare.cameraFar;
    dispatch.viewSpaceToMeters = prepare.viewSpaceToMeters;
    dispatch.cameraVerticalFovRadians = prepare.cameraVerticalFovRadians;
    dispatch.minLuminance = prepare.minLuminance;
    dispatch.maxLuminance = prepare.maxLuminance;
    dispatch.transferFunction = prepare.transferFunction;
    dispatch.frameId = prepare.frameId;
    dispatch.reset = prepare.reset;
    result = ffxVkFsr3_3_1_6FrameGenerationContextRecordDispatch(
        fsr3_316_frame_generation_context, &dispatch);
    if (result != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
        fsr3_frame_generation_reset_next = true;
        Com_WPrintf("FSR3 FG 3.1.6: dispatch failed (%d)\n", (int)result);
        return VK_ERROR_UNKNOWN;
    }
    fsr3_frame_generation_reset_next = false;
    return VK_SUCCESS;
}

void vkpt_fsr_frame_generation_retire(uint32_t frame_slot)
{
    const uint64_t frame_id = frame_slot < MAX_FRAMES_IN_FLIGHT
        ? fsr3_316_frame_generation_frame_ids[frame_slot] : 0u;

    if (!frame_id || !fsr3_316_frame_generation_context)
        return;
    (void)ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(
        fsr3_316_frame_generation_context, frame_id);
    fsr3_316_frame_generation_frame_ids[frame_slot] = 0u;
}

VkResult vkpt_fsr_frame_generation_record(VkCommandBuffer cmd_buf)
{
    const VkptTemporalFrame *frame = vkpt_temporal_get_frame();
    const uint32_t required_inputs =
        VKPT_TEMPORAL_INPUT_MOTION_VECTORS |
        VKPT_TEMPORAL_INPUT_DEVICE_DEPTH;
    FfxVkPortableFrameGenerationPrepareInfo prepare;
    FfxVkPortableFrameGenerationDispatchInfo dispatch;
    FfxVkPortableImage source;
    FfxVkPortableResult result;
    char temporal_reason[128];
    VkImageSubresourceRange color_range = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };

    if (!vkpt_fsr_frame_generation_is_ready())
        return VK_NOT_READY;
    if (!vkpt_temporal_validate_current_frame(required_inputs,
            temporal_reason, sizeof(temporal_reason)) ||
        !(frame->flags & VKPT_TEMPORAL_FRAME_RECTILINEAR_PROJECTION) ||
        frame->inputs.device_depth_description.convention !=
            VKPT_TEMPORAL_DEPTH_DEVICE_ZERO_TO_ONE) {
        fsr3_frame_generation_reset_next = true;
        Com_WPrintf("FSR3 FG: temporal inputs rejected: %s\n",
            temporal_reason[0] ? temporal_reason : "unsupported projection/depth convention");
        return VK_NOT_READY;
    }

    /* TAA_OUTPUT is the display-resolution, tone-mapped-but-HUDless scene.
     * The UI queue is drawn separately by the presenter after interpolation. */
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .image = qvk.images[VKPT_IMG_TAA_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    );
    const VkptTemporalImage *provider_inputs[] = {
        &frame->inputs.device_depth, &frame->inputs.motion_vectors
    };
    for (size_t i = 0; i < sizeof(provider_inputs) / sizeof(provider_inputs[0]); ++i) {
        const VkptTemporalImage *input = provider_inputs[i];
        IMAGE_BARRIER_STAGES(cmd_buf, input->producer_stage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .image = input->image,
            .subresourceRange = color_range,
            .srcAccessMask = input->producer_access,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = input->layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        );
    }

    if (cvar_flt_frame_generation_backend &&
        cvar_flt_frame_generation_backend->integer == 1)
        return fsr3_316_frame_generation_record_after_inputs(cmd_buf, frame);

    source = fsr3_screen_image(VKPT_IMG_TAA_OUTPUT, qvk.extent_taa_output,
        FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    memset(&prepare, 0, sizeof(prepare));
    prepare.structSize = sizeof(prepare);
    prepare.commandBuffer = cmd_buf;
    prepare.depth = fsr3_temporal_image(&frame->inputs.device_depth,
        FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    prepare.motionVectors = fsr3_temporal_image(&frame->inputs.motion_vectors,
        FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    prepare.renderSize.width = frame->render_size.width;
    prepare.renderSize.height = frame->render_size.height;
    prepare.jitterOffset.x = -frame->camera.jitter_render_pixels[0];
    prepare.jitterOffset.y = -frame->camera.jitter_render_pixels[1];
    prepare.motionVectorScale.x = frame->inputs.motion_description.to_render_pixels[0];
    prepare.motionVectorScale.y = frame->inputs.motion_description.to_render_pixels[1];
    prepare.frameTimeMilliseconds = frame->frame_time_ms > 0.0f
        ? frame->frame_time_ms : 16.667f;
    prepare.cameraNear = frame->camera.near_plane;
    prepare.cameraFar = frame->camera.far_plane;
    prepare.cameraVerticalFovRadians = frame->camera.vertical_fov_radians;
    prepare.viewSpaceToMeters = frame->camera.view_space_to_meters;
    prepare.minLuminance = 0.0f;
    prepare.maxLuminance = qvk.surf_is_hdr ? 1000.0f : 1.0f;
    prepare.transferFunction = qvk.surf_is_hdr
        ? FFX_VK_PORTABLE_TRANSFER_FUNCTION_SCRGB
        : FFX_VK_PORTABLE_TRANSFER_FUNCTION_SRGB;
    prepare.cameraPosition.x = frame->camera.position[0];
    prepare.cameraPosition.y = frame->camera.position[1];
    prepare.cameraPosition.z = frame->camera.position[2];
    prepare.cameraUp.x = frame->camera.up[0];
    prepare.cameraUp.y = frame->camera.up[1];
    prepare.cameraUp.z = frame->camera.up[2];
    prepare.cameraRight.x = frame->camera.right[0];
    prepare.cameraRight.y = frame->camera.right[1];
    prepare.cameraRight.z = frame->camera.right[2];
    prepare.cameraForward.x = frame->camera.forward[0];
    prepare.cameraForward.y = frame->camera.forward[1];
    prepare.cameraForward.z = frame->camera.forward[2];
    prepare.reset = (fsr3_frame_generation_reset_next || !frame->history_valid)
        ? VK_TRUE : VK_FALSE;
    prepare.frameId = frame->frame_id;
    result = ffxVkPortableFrameGenerationContextPrepare(
        fsr3_frame_generation_context, &prepare, &source);
    if (result != FFX_VK_PORTABLE_OK) {
        fsr3_frame_generation_reset_next = true;
        Com_WPrintf("FSR3 FG: prepare failed (%d)\n", (int)result);
        return VK_ERROR_UNKNOWN;
    }

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.structSize = sizeof(dispatch);
    dispatch.commandBuffer = cmd_buf;
    dispatch.currentColor = source;
    dispatch.hudlessColor.structSize = sizeof(dispatch.hudlessColor);
    dispatch.distortionField.structSize = sizeof(dispatch.distortionField);
    dispatch.output = fsr3_screen_image(VKPT_IMG_FSR_RCAS_OUTPUT,
        qvk.extent_unscaled, FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch.displaySize.width = frame->display_size.width;
    dispatch.displaySize.height = frame->display_size.height;
    dispatch.interpolationRect.x = 0;
    dispatch.interpolationRect.y = 0;
    dispatch.interpolationRect.width = frame->display_size.width;
    dispatch.interpolationRect.height = frame->display_size.height;
    dispatch.frameTimeMilliseconds = prepare.frameTimeMilliseconds;
    dispatch.cameraNear = prepare.cameraNear;
    dispatch.cameraFar = prepare.cameraFar;
    dispatch.cameraVerticalFovRadians = prepare.cameraVerticalFovRadians;
    dispatch.viewSpaceToMeters = prepare.viewSpaceToMeters;
    dispatch.minLuminance = prepare.minLuminance;
    dispatch.maxLuminance = prepare.maxLuminance;
    dispatch.transferFunction = prepare.transferFunction;
    dispatch.reset = prepare.reset;
    dispatch.frameId = prepare.frameId;
    result = ffxVkPortableFrameGenerationContextRecordDispatch(
        fsr3_frame_generation_context, &dispatch);
    if (result != FFX_VK_PORTABLE_OK) {
        fsr3_frame_generation_reset_next = true;
        Com_WPrintf("FSR3 FG: dispatch failed (%d)\n", (int)result);
        return VK_ERROR_UNKNOWN;
    }
    fsr3_frame_generation_reset_next = false;
    return VK_SUCCESS;
}

static void copy_upscaled_output_to_taa(
    VkCommandBuffer cmd_buf, const VkptTemporalFrame *frame)
{
    VkImageSubresourceRange color_range = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };
    VkImageCopy copy = {
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .extent = {frame->display_size.width, frame->display_size.height, 1}
    };

    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        .image = qvk.images[VKPT_IMG_FSR_EASU_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    );
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        .image = qvk.images[VKPT_IMG_TAA_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    );
    vkCmdCopyImage(cmd_buf,
        qvk.images[VKPT_IMG_FSR_EASU_OUTPUT],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        qvk.images[VKPT_IMG_TAA_OUTPUT],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .image = qvk.images[VKPT_IMG_TAA_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    );
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .image = qvk.images[VKPT_IMG_FSR_EASU_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    );
}

static VkResult fsr3_dispatch(VkCommandBuffer cmd_buf)
{
    const VkptTemporalFrame *frame = vkpt_temporal_get_frame();
    FfxVkPortableUpscaleDispatchInfo dispatch;
    FfxVkPortableResult result;
    char temporal_reason[128];
    VkImageSubresourceRange color_range = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };
    const uint32_t required_inputs =
        VKPT_TEMPORAL_INPUT_SCENE_COLOR |
        VKPT_TEMPORAL_INPUT_MOTION_VECTORS |
        VKPT_TEMPORAL_INPUT_DEVICE_DEPTH |
        VKPT_TEMPORAL_INPUT_REACTIVE_MASK |
        VKPT_TEMPORAL_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK;

    if (!fsr3_is_enabled())
        return VK_SUCCESS;
    if (!vkpt_temporal_validate_current_frame(required_inputs,
            temporal_reason, sizeof(temporal_reason)) ||
        !(frame->flags & VKPT_TEMPORAL_FRAME_RECTILINEAR_PROJECTION) ||
        frame->inputs.device_depth_description.convention !=
            VKPT_TEMPORAL_DEPTH_DEVICE_ZERO_TO_ONE) {
        Com_WPrintf("FSR3: temporal inputs rejected: %s\n",
            temporal_reason[0] ? temporal_reason : "unsupported projection/depth convention");
        fsr3_reset_next = true;
        return VK_NOT_READY;
    }

    BEGIN_PERF_MARKER(cmd_buf, PROFILER_FSR);

    const VkptTemporalImage *provider_inputs[] = {
        &frame->inputs.scene_color,
        &frame->inputs.motion_vectors,
        &frame->inputs.device_depth,
        &frame->inputs.reactive_mask,
        &frame->inputs.transparency_and_composition_mask
    };
    for (size_t i = 0; i < sizeof(provider_inputs) / sizeof(provider_inputs[0]); ++i) {
        const VkptTemporalImage *input = provider_inputs[i];
        IMAGE_BARRIER_STAGES(cmd_buf, input->producer_stage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .image = input->image,
            .subresourceRange = color_range,
            .srcAccessMask = input->producer_access,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = input->layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        );
    }

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.structSize = sizeof(dispatch);
    dispatch.commandBuffer = cmd_buf;
    dispatch.color = fsr3_temporal_image(
        &frame->inputs.scene_color, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    dispatch.depth = fsr3_temporal_image(
        &frame->inputs.device_depth, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    dispatch.motionVectors = fsr3_temporal_image(
        &frame->inputs.motion_vectors, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    dispatch.exposure.structSize = sizeof(dispatch.exposure);
    dispatch.reactiveMask = fsr3_temporal_image(
        &frame->inputs.reactive_mask,
        FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    dispatch.transparencyAndCompositionMask = fsr3_temporal_image(
        &frame->inputs.transparency_and_composition_mask,
        FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    dispatch.output = fsr3_output_image();
    /* Q2 applies +sub_pixel_jitter while tracing.  FSR's lookup offset
     * cancels that displacement, so the provider receives the inverse. */
    dispatch.jitterOffset.x = -frame->camera.jitter_render_pixels[0];
    dispatch.jitterOffset.y = -frame->camera.jitter_render_pixels[1];
    dispatch.motionVectorScale.x =
        frame->inputs.motion_description.to_render_pixels[0];
    dispatch.motionVectorScale.y =
        frame->inputs.motion_description.to_render_pixels[1];
    dispatch.renderSize.width = frame->render_size.width;
    dispatch.renderSize.height = frame->render_size.height;
    dispatch.outputSize.width = frame->display_size.width;
    dispatch.outputSize.height = frame->display_size.height;
    dispatch.frameTimeMilliseconds = frame->frame_time_ms > 0.0f
        ? frame->frame_time_ms : 16.667f;
    dispatch.preExposure = (float)STORAGE_SCALE_HDR;
    dispatch.cameraNear = frame->camera.near_plane;
    dispatch.cameraFar = frame->camera.far_plane;
    dispatch.cameraVerticalFovRadians = frame->camera.vertical_fov_radians;
    dispatch.viewSpaceToMeters = frame->camera.view_space_to_meters;
    dispatch.sharpness = Q_clipf(cvar_flt_fsr3_sharpening->value, 0.0f, 1.0f);
    dispatch.enableSharpening = dispatch.sharpness > 0.0f ? VK_TRUE : VK_FALSE;
    dispatch.reset = (fsr3_reset_next || !frame->history_valid) ? VK_TRUE : VK_FALSE;
    dispatch.frameId = frame->frame_id;

    result = ffxVkPortableUpscaleContextRecordDispatch(fsr3_context, &dispatch);
    if (result != FFX_VK_PORTABLE_OK) {
        END_PERF_MARKER(cmd_buf, PROFILER_FSR);
        fsr3_reset_next = true;
        Com_WPrintf("FSR3: dispatch failed (%d)\n", (int)result);
        return VK_ERROR_UNKNOWN;
    }
    fsr3_reset_next = false;
    copy_upscaled_output_to_taa(cmd_buf, frame);

    END_PERF_MARKER(cmd_buf, PROFILER_FSR);
    return VK_SUCCESS;
}

static VkResult fsr3_315_dispatch(VkCommandBuffer cmd_buf)
{
    const VkptTemporalFrame *frame = vkpt_temporal_get_frame();
    FfxVkFsr3_3_1_5UpscalerDispatchInfo dispatch;
    FfxVkFsr3_3_1_5Result result;
    char temporal_reason[128];
    VkImageSubresourceRange color_range = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };
    const uint32_t required_inputs =
        VKPT_TEMPORAL_INPUT_SCENE_COLOR |
        VKPT_TEMPORAL_INPUT_MOTION_VECTORS |
        VKPT_TEMPORAL_INPUT_DEVICE_DEPTH |
        VKPT_TEMPORAL_INPUT_REACTIVE_MASK |
        VKPT_TEMPORAL_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK;

    if (!fsr3_315_is_enabled())
        return VK_SUCCESS;
    if (!vkpt_temporal_validate_current_frame(required_inputs,
            temporal_reason, sizeof(temporal_reason)) ||
        !(frame->flags & VKPT_TEMPORAL_FRAME_RECTILINEAR_PROJECTION) ||
        frame->inputs.device_depth_description.convention !=
            VKPT_TEMPORAL_DEPTH_DEVICE_ZERO_TO_ONE ||
        frame->inputs.scene_color.image != qvk.images[VKPT_IMG_FLAT_COLOR] ||
        frame->inputs.motion_vectors.image != qvk.images[VKPT_IMG_FLAT_MOTION] ||
        frame->inputs.device_depth.image != qvk.images[VKPT_IMG_TEMPORAL_DEVICE_DEPTH] ||
        frame->inputs.reactive_mask.image != qvk.images[VKPT_IMG_TEMPORAL_REACTIVE_MASK] ||
        frame->inputs.transparency_and_composition_mask.image !=
            qvk.images[VKPT_IMG_TEMPORAL_COMPOSITION_MASK]) {
        Com_WPrintf("FSR3.1.5: temporal inputs rejected: %s\n",
            temporal_reason[0] ? temporal_reason : "unexpected image/depth/projection contract");
        fsr3_315_reset_next = true;
        return VK_NOT_READY;
    }

    BEGIN_PERF_MARKER(cmd_buf, PROFILER_FSR);
    const VkptTemporalImage *provider_inputs[] = {
        &frame->inputs.scene_color,
        &frame->inputs.motion_vectors,
        &frame->inputs.device_depth,
        &frame->inputs.reactive_mask,
        &frame->inputs.transparency_and_composition_mask
    };
    for (size_t index = 0; index < LENGTH(provider_inputs); ++index) {
        const VkptTemporalImage *input = provider_inputs[index];
        IMAGE_BARRIER_STAGES(cmd_buf, input->producer_stage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .image = input->image,
            .subresourceRange = color_range,
            .srcAccessMask = input->producer_access,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = input->layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        );
    }

    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.commandBuffer = cmd_buf;
    dispatch.color = fsr3_315_color;
    dispatch.depth = fsr3_315_depth;
    dispatch.motionVectors = fsr3_315_motion;
    dispatch.reactive = fsr3_315_reactive;
    dispatch.transparencyAndComposition = fsr3_315_composition;
    dispatch.dilatedDepth = fsr3_315_dilated_depth;
    dispatch.dilatedMotionVectors = fsr3_315_dilated_motion;
    dispatch.reconstructedPrevNearestDepth = fsr3_315_previous_depth;
    dispatch.output = fsr3_315_output;
    dispatch.jitterOffsetX = -frame->camera.jitter_render_pixels[0];
    dispatch.jitterOffsetY = -frame->camera.jitter_render_pixels[1];
    dispatch.motionVectorScaleX = frame->inputs.motion_description.to_render_pixels[0];
    dispatch.motionVectorScaleY = frame->inputs.motion_description.to_render_pixels[1];
    dispatch.renderWidth = frame->render_size.width;
    dispatch.renderHeight = frame->render_size.height;
    dispatch.upscaleWidth = frame->display_size.width;
    dispatch.upscaleHeight = frame->display_size.height;
    dispatch.frameTimeMilliseconds = frame->frame_time_ms > 0.0f
        ? frame->frame_time_ms : 16.667f;
    dispatch.preExposure = (float)STORAGE_SCALE_HDR;
    dispatch.enableSharpening = cvar_flt_fsr3_sharpening->value > 0.0f ? VK_TRUE : VK_FALSE;
    dispatch.sharpness = Q_clipf(cvar_flt_fsr3_sharpening->value, 0.0f, 1.0f);
    dispatch.reset = (fsr3_315_reset_next || !frame->history_valid) ? VK_TRUE : VK_FALSE;
    dispatch.cameraNear = frame->camera.near_plane;
    dispatch.cameraFar = frame->camera.far_plane;
    dispatch.cameraVerticalFovRadians = frame->camera.vertical_fov_radians;
    dispatch.viewSpaceToMeters = frame->camera.view_space_to_meters;
    result = ffxVkFsr3_3_1_5UpscalerContextRecordDispatch(fsr3_315_context, &dispatch);
    if (result != FFX_VK_FSR3_3_1_5_OK) {
        END_PERF_MARKER(cmd_buf, PROFILER_FSR);
        fsr3_315_reset_next = true;
        Com_WPrintf("FSR3.1.5: dispatch failed (%d)\n", (int)result);
        return VK_ERROR_UNKNOWN;
    }
    fsr3_315_reset_next = false;
    copy_upscaled_output_to_taa(cmd_buf, frame);
    END_PERF_MARKER(cmd_buf, PROFILER_FSR);
    return VK_SUCCESS;
}
#endif

static VkResult fsr4_dispatch(VkCommandBuffer cmd_buf)
{
    const VkptTemporalFrame *frame;
    ffxDispatchDescUpscale d;
    ffxReturnCode_t dispatch_result;
    bool preserve_drs_history;
    char temporal_reason[128];
    VkImageSubresourceRange color_range = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };

    /* A preset change replaces the complete model graph (including opaque
     * initializer/weight payloads), so rebuild before testing enabled state.
     * Without this ordering, the old Performance-only early return made a
     * changed quality cvar silently fall back forever. */
    if (requested_upscaler() == VKPT_UPSCALER_FSR4 &&
        (!fsr4_backend_ok ||
         strcmp(fsr4_shader_tier, display_res_tag()) != 0 ||
         strcmp(fsr4_shader_model, fsr4_requested_model()) != 0)) {
        VkResult recreate_result = fsr4_recreate_context();
        if (recreate_result != VK_SUCCESS)
            return recreate_result;
    }

    if (!fsr4_is_enabled())
        return VK_SUCCESS;

    /* Q2RTX calls create_pipelines only on shader reload, not on swapchain
       recreation.  So if the display resolution has changed since the context
       was created (most commonly: the initial 640x480 menu → actual game res),
       we need to recreate it here.  This path is safe because we're at the
       start of a fresh command buffer recording, and Q2RTX has already
       done a full device wait before the new frame started. */
    if (fsr4_context_ok &&
        (fsr4_ctx_dw != qvk.extent_unscaled.width ||
         fsr4_ctx_dh != qvk.extent_unscaled.height))
    {
        _VK(fsr4_recreate_context());
        if (!fsr4_context_ok) return VK_SUCCESS;
    }

    frame = vkpt_temporal_get_frame();
    if (!vkpt_temporal_validate_current_frame(
            VKPT_TEMPORAL_INPUT_SCENE_COLOR |
            VKPT_TEMPORAL_INPUT_MOTION_VECTORS |
            VKPT_TEMPORAL_INPUT_VIEW_Z,
            temporal_reason, sizeof(temporal_reason)) ||
        !(frame->flags & VKPT_TEMPORAL_FRAME_RECTILINEAR_PROJECTION)) {
        Com_WPrintf("FSR4: temporal inputs rejected: %s\n",
            temporal_reason[0] ? temporal_reason : "unsupported projection");
        fsr4_reset_next = true;
        return VK_NOT_READY;
    }

    /* The dedicated DRS graph is expressly trained for a changing render
     * extent while its display-sized recurrent state remains allocated. Q2's
     * generic temporal contract marks every extent change invalid for the
     * benefit of other consumers; preserve FSR4 history only when that is the
     * sole reset reason. Cuts, settings, projection, presentation gaps, and
     * all other invalidation causes still force a full reset. */
    preserve_drs_history = fsr4_dynamic_resolution_requested() &&
        frame->reset_reasons == VKPT_TEMPORAL_RESET_RENDER_SIZE_CHANGED &&
        frame->previous_render_size.width != 0 &&
        frame->previous_render_size.height != 0;

    BEGIN_PERF_MARKER(cmd_buf, PROFILER_FSR);

    /* Interleave/primary-ray writes must be visible to the provider. */
    const VkptTemporalImage *provider_inputs[] = {
        &frame->inputs.scene_color,
        &frame->inputs.motion_vectors,
        &frame->inputs.view_z
    };
    for (size_t i = 0; i < sizeof(provider_inputs) / sizeof(provider_inputs[0]); ++i) {
        const VkptTemporalImage *input = provider_inputs[i];
        IMAGE_BARRIER_STAGES(cmd_buf, input->producer_stage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .image = input->image,
            .subresourceRange = color_range,
            .srcAccessMask = input->producer_access,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = input->layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        );
    }

    memset(&d, 0, sizeof(d));
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = (FfxCommandList)cmd_buf;

#define FILL_TEMPORAL_TEX(field, temporal_image, ffx_format)                    \
    do {                                                                         \
        d.field.resource = (void *)(temporal_image).view;                        \
        d.field.description.type = FFX_RESOURCE_TYPE_TEXTURE2D;                  \
        d.field.description.format = (ffx_format);                               \
        d.field.description.width = (temporal_image).allocation_extent.width;    \
        d.field.description.height = (temporal_image).allocation_extent.height;  \
    } while (0)

    FILL_TEMPORAL_TEX(color, frame->inputs.scene_color,
                      FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT);
    FILL_TEMPORAL_TEX(motionVectors, frame->inputs.motion_vectors,
                      FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT);
    FILL_TEMPORAL_TEX(depth, frame->inputs.view_z,
                      FFX_SURFACE_FORMAT_R32_FLOAT);
#undef FILL_TEMPORAL_TEX

    d.output.resource = (void *)qvk.images_views[VKPT_IMG_FSR_EASU_OUTPUT];
    d.output.description.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    d.output.description.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    d.output.description.width = qvk.extent_taa_images.width;
    d.output.description.height = qvk.extent_taa_images.height;

    d.renderSize.width = frame->render_size.width;
    d.renderSize.height = frame->render_size.height;
    d.upscaleSize.width = frame->display_size.width;
    d.upscaleSize.height = frame->display_size.height;
    /* Q2RTX adds this offset directly to the primary-ray pixel center.  FSR's
     * constants are used as an inverse input lookup offset, so its API
     * convention is the opposite sign (the same inversion Q2 TAAU performs). */
    d.jitterOffset.x = -frame->camera.jitter_render_pixels[0];
    d.jitterOffset.y = -frame->camera.jitter_render_pixels[1];
    d.motionVectorScale.x =
        frame->inputs.motion_description.to_render_pixels[0];
    d.motionVectorScale.y =
        frame->inputs.motion_description.to_render_pixels[1];
    d.frameTimeDelta = frame->frame_time_ms > 0.0f
        ? frame->frame_time_ms : 16.667f;
    /* The renderer stores HDR radiance multiplied by STORAGE_SCALE_HDR.
     * Treating that storage scale as pre-exposure lets FSR reconstruct in
     * semantic linear space and restore the same scale on output, so Q2RTX's
     * existing bloom/tone mapper can consume it unchanged. */
    d.preExposure = (float)STORAGE_SCALE_HDR;
    d.reset = fsr4_reset_next ||
        (!frame->history_valid && !preserve_drs_history);
    d.sharpness = Q_clipf(cvar_flt_fsr4_sharpening->value, 0.0f, 1.0f);
    d.enableSharpening = d.sharpness > 0.0f;
    d.enableAutoExposure = cvar_flt_fsr4_auto_exposure->integer != 0;
    d.cameraNear = frame->camera.near_plane;
    d.cameraFar = frame->camera.far_plane;
    d.cameraFovAngleVertical = frame->camera.vertical_fov_radians;
    d.viewSpaceToMetersFactor = frame->camera.view_space_to_meters;

    g_vkBackendOverride = &fsr4_backend;
    dispatch_result = ffxDispatch(&fsr4_context, &d.header);
    g_vkBackendOverride = NULL;
    if (dispatch_result != FFX_API_RETURN_OK) {
        END_PERF_MARKER(cmd_buf, PROFILER_FSR);
        fsr4_reset_next = true;
        Com_WPrintf("FSR4: dispatch failed (%d)\n", (int)dispatch_result);
        return VK_ERROR_UNKNOWN;
    }
    fsr4_reset_next = false;

    /* Q2RTX's post effects are currently fixed to TAA_OUTPUT.  Copy the
     * reconstructed linear HDR image there, then run bloom/tone mapping at
     * display resolution.  This keeps FSR before all post-processing. */
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        .image = qvk.images[VKPT_IMG_FSR_EASU_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    );
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        .image = qvk.images[VKPT_IMG_TAA_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    );
    VkImageCopy copy = {
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .extent = {frame->display_size.width, frame->display_size.height, 1}
    };
    vkCmdCopyImage(cmd_buf,
        qvk.images[VKPT_IMG_FSR_EASU_OUTPUT],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        qvk.images[VKPT_IMG_TAA_OUTPUT],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .image = qvk.images[VKPT_IMG_TAA_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    );
    IMAGE_BARRIER_STAGES(cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .image = qvk.images[VKPT_IMG_FSR_EASU_OUTPUT],
        .subresourceRange = color_range,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    );

    END_PERF_MARKER(cmd_buf, PROFILER_FSR);
    return VK_SUCCESS;
}

VkResult vkpt_fsr_do(VkCommandBuffer cmd_buf)
{
#ifdef VKPT_FSR3
    if (fsr3_315_is_enabled())
        return fsr3_315_dispatch(cmd_buf);
    if (fsr3_is_enabled())
        return fsr3_dispatch(cmd_buf);
#endif
    return fsr4_dispatch(cmd_buf);
}

VkResult vkpt_fsr_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
    return vkpt_final_blit(cmd_buf, VKPT_IMG_TAA_OUTPUT,
                           qvk.extent_unscaled, false, warp);
}
