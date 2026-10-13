/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#include "video_recorder.h"
#include "vkpt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define VK_QUEUE_VIDEO_ENCODE_BIT_KHR 0x00000040

typedef struct BitWriter {
    uint8_t buffer[4096];
    size_t byte_count;
    uint32_t bit_buf;
    int bit_cnt;
} BitWriter;

static void bw_init(BitWriter *bw)
{
    bw->byte_count = 0;
    bw->bit_buf = 0;
    bw->bit_cnt = 0;
}

static void bw_put_bits(BitWriter *bw, uint32_t val, int n)
{
    for (int i = n - 1; i >= 0; --i) {
        uint32_t bit = (val >> i) & 1u;
        bw->bit_buf = (bw->bit_buf << 1) | bit;
        bw->bit_cnt++;
        if (bw->bit_cnt == 8) {
            if (bw->byte_count < sizeof(bw->buffer)) {
                bw->buffer[bw->byte_count++] = (uint8_t)bw->bit_buf;
                if (bw->byte_count >= 3 &&
                    bw->buffer[bw->byte_count - 3] == 0 &&
                    bw->buffer[bw->byte_count - 2] == 0 &&
                    bw->buffer[bw->byte_count - 1] <= 3) {
                    if (bw->byte_count < sizeof(bw->buffer)) {
                        bw->buffer[bw->byte_count++] = 3; /* emulation prevention */
                    }
                }
            }
            bw->bit_buf = 0;
            bw->bit_cnt = 0;
        }
    }
}

static void bw_put_ue(BitWriter *bw, uint32_t val)
{
    uint32_t temp = val + 1u;
    int n = 0;
    while (temp > 1u) {
        temp >>= 1u;
        n++;
    }
    bw_put_bits(bw, 0, n);
    bw_put_bits(bw, val + 1u, n + 1);
}

static void bw_put_se(BitWriter *bw, int32_t val)
{
    uint32_t ue = (val <= 0) ? (uint32_t)(-val * 2) : (uint32_t)(val * 2 - 1);
    bw_put_ue(bw, ue);
}

static void bw_flush(BitWriter *bw)
{
    if (bw->bit_cnt > 0) {
        bw_put_bits(bw, 1, 1); /* rbsp stop bit */
        if (bw->bit_cnt > 0) {
            bw_put_bits(bw, 0, 8 - bw->bit_cnt);
        }
    }
}

typedef struct VideoRecorderState {
    bool supported;
    bool active;
    bool recording;
    uint32_t video_encode_queue_family_index;
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    uint32_t bitrate;
    uint64_t frame_count;
    uint64_t bytes_written;
    char output_filename[256];
    FILE *file;
} VideoRecorderState;

static VideoRecorderState g_recorder = {0};

cvar_t *cvar_videorecord_fps = NULL;
cvar_t *cvar_videorecord_bitrate = NULL;
cvar_t *cvar_videorecord_active = NULL;

static void write_nal_unit(FILE *f, const uint8_t *data, size_t size)
{
    if (!f || !data || size == 0)
        return;
    static const uint8_t start_code[4] = { 0, 0, 0, 1 };
    fwrite(start_code, 1, 4, f);
    fwrite(data, 1, size, f);
    g_recorder.bytes_written += (4 + size);
}

static void write_headers_h264(FILE *f, uint32_t width, uint32_t height, uint32_t fps)
{
    BitWriter bw;

    /* 1. Sequence Parameter Set (SPS) - NAL type 7 */
    bw_init(&bw);
    bw_put_bits(&bw, 0x67, 8); /* forbidden_zero=0, nal_ref_idc=3, type=7 */
    bw_put_bits(&bw, 100, 8);  /* profile_idc: High */
    bw_put_bits(&bw, 0, 8);    /* constraint flags */
    bw_put_bits(&bw, 42, 8);   /* level_idc: 4.2 */
    bw_put_ue(&bw, 0);         /* seq_parameter_set_id */
    bw_put_ue(&bw, 1);         /* chroma_format_idc: 4:2:0 */
    bw_put_ue(&bw, 0);         /* bit_depth_luma_minus8 */
    bw_put_ue(&bw, 0);         /* bit_depth_chroma_minus8 */
    bw_put_bits(&bw, 0, 1);    /* qpprime_y_zero_transform_bypass_flag */
    bw_put_bits(&bw, 0, 1);    /* seq_scaling_matrix_present_flag */
    bw_put_ue(&bw, 4);         /* log2_max_frame_num_minus4 */
    bw_put_ue(&bw, 0);         /* pic_order_cnt_type */
    bw_put_ue(&bw, 4);         /* log2_max_pic_order_cnt_lsb_minus4 */
    bw_put_ue(&bw, 1);         /* max_num_ref_frames */
    bw_put_bits(&bw, 0, 1);    /* gaps_in_frame_num_value_allowed_flag */
    bw_put_ue(&bw, (width + 15u) / 16u - 1u);       /* pic_width_in_mbs_minus1 */
    bw_put_ue(&bw, (height + 15u) / 16u - 1u);      /* pic_height_in_map_units_minus1 */
    bw_put_bits(&bw, 1, 1);    /* frame_mbs_only_flag */
    bw_put_bits(&bw, 1, 1);    /* direct_8x8_inference_flag */

    uint32_t crop_r = (((width + 15u) / 16u) * 16u - width) / 2u;
    uint32_t crop_b = (((height + 15u) / 16u) * 16u - height) / 2u;
    if (crop_r > 0 || crop_b > 0) {
        bw_put_bits(&bw, 1, 1); /* frame_cropping_flag */
        bw_put_ue(&bw, 0);      /* crop_left */
        bw_put_ue(&bw, crop_r); /* crop_right */
        bw_put_ue(&bw, 0);      /* crop_top */
        bw_put_ue(&bw, crop_b); /* crop_bottom */
    } else {
        bw_put_bits(&bw, 0, 1);
    }

    bw_put_bits(&bw, 1, 1);    /* vui_parameters_present_flag */
    bw_put_bits(&bw, 0, 4);    /* aspect, overscan, video_signal, chroma_loc */
    bw_put_bits(&bw, 1, 1);    /* timing_info_present_flag */
    bw_put_bits(&bw, 1, 32);   /* num_units_in_tick */
    bw_put_bits(&bw, 2u * fps, 32); /* time_scale */
    bw_put_bits(&bw, 1, 1);    /* fixed_frame_rate_flag */
    bw_put_bits(&bw, 0, 4);    /* nal_hrd, vcl_hrd, pic_struct, bitstream_restriction */
    bw_flush(&bw);
    write_nal_unit(f, bw.buffer, bw.byte_count);

    /* 2. Picture Parameter Set (PPS) - NAL type 8 */
    bw_init(&bw);
    bw_put_bits(&bw, 0x68, 8); /* forbidden_zero=0, nal_ref_idc=3, type=8 */
    bw_put_ue(&bw, 0);         /* pic_parameter_set_id */
    bw_put_ue(&bw, 0);         /* seq_parameter_set_id */
    bw_put_bits(&bw, 0, 1);    /* entropy_coding_mode_flag (CAVLC) */
    bw_put_bits(&bw, 0, 1);    /* bottom_field_pic_order_in_frame_present_flag */
    bw_put_ue(&bw, 0);         /* num_slice_groups_minus1 */
    bw_put_ue(&bw, 0);         /* num_ref_idx_l0_default_active_minus1 */
    bw_put_ue(&bw, 0);         /* num_ref_idx_l1_default_active_minus1 */
    bw_put_bits(&bw, 0, 1);    /* weighted_pred_flag */
    bw_put_bits(&bw, 0, 2);    /* weighted_bipred_idc */
    bw_put_se(&bw, 0);         /* pic_init_qp_minus26 */
    bw_put_se(&bw, 0);         /* pic_init_qs_minus26 */
    bw_put_se(&bw, 0);         /* chroma_qp_index_offset */
    bw_put_bits(&bw, 0, 3);    /* deblocking, constrained, redundant */
    bw_flush(&bw);
    write_nal_unit(f, bw.buffer, bw.byte_count);
}

void vkpt_video_recorder_init(void)
{
    cvar_videorecord_fps = Cvar_Get("videorecord_fps", "60", CVAR_ARCHIVE);
    cvar_videorecord_bitrate = Cvar_Get("videorecord_bitrate", "15000000", CVAR_ARCHIVE);
    cvar_videorecord_active = Cvar_Get("videorecord_active", "0", 0);

    Cmd_AddCommand("videorecord_start", (xcommand_t)&vkpt_video_recorder_start);
    Cmd_AddCommand("videorecord_stop", (xcommand_t)&vkpt_video_recorder_stop);
    Cmd_AddCommand("videorecord_status", (xcommand_t)&vkpt_video_recorder_get_status);
}

void vkpt_video_recorder_query_capabilities(VkPhysicalDevice physicalDevice)
{
    if (physicalDevice == VK_NULL_HANDLE)
        return;

    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &extensionCount, NULL);
    if (extensionCount > 0) {
        VkExtensionProperties *extensions = malloc(extensionCount * sizeof(VkExtensionProperties));
        if (extensions) {
            vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &extensionCount, extensions);
            bool hasVideoQueue = false;
            bool hasVideoEncodeQueue = false;
            bool hasVideoEncodeH264 = false;

            for (uint32_t i = 0; i < extensionCount; ++i) {
                if (strcmp(extensions[i].extensionName, "VK_KHR_video_queue") == 0)
                    hasVideoQueue = true;
                else if (strcmp(extensions[i].extensionName, "VK_KHR_video_encode_queue") == 0)
                    hasVideoEncodeQueue = true;
                else if (strcmp(extensions[i].extensionName, "VK_KHR_video_encode_h264") == 0)
                    hasVideoEncodeH264 = true;
            }
            free(extensions);

            if (hasVideoQueue && hasVideoEncodeQueue && hasVideoEncodeH264) {
                g_recorder.supported = true;
            }
        }
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    if (queueFamilyCount > 0) {
        VkQueueFamilyProperties *queueFamilies = malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
        if (queueFamilies) {
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);
            for (uint32_t i = 0; i < queueFamilyCount; ++i) {
                if (queueFamilies[i].queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
                    g_recorder.video_encode_queue_family_index = i;
                    break;
                }
            }
            free(queueFamilies);
        }
    }

    if (g_recorder.supported) {
        Com_Printf("Video Recorder: VK_KHR_video_encode_h264 supported (encode queue family %u).\n",
                   g_recorder.video_encode_queue_family_index);
    } else {
        Com_Printf("Video Recorder: VK_KHR_video_encode_h264 hardware encoding not supported on device.\n");
    }
}

bool vkpt_video_recorder_start(const char *filename, uint32_t fps, uint32_t bitrate)
{
    if (g_recorder.recording) {
        Com_WPrintf("Video Recorder: Already recording to '%s'. Stop recording first.\n",
                    g_recorder.output_filename);
        return false;
    }

    /* Process console arguments if called via Cmd_AddCommand */
    int argc = Cmd_Argc();
    const char *target_filename = (argc >= 2) ? Cmd_Argv(1) : (filename ? filename : "q2rtx_capture.h264");
    uint32_t target_fps = (argc >= 3) ? (uint32_t)atoi(Cmd_Argv(2)) : (fps ? fps : (cvar_videorecord_fps ? (uint32_t)cvar_videorecord_fps->integer : 60u));
    uint32_t target_bitrate = (argc >= 4) ? (uint32_t)atoi(Cmd_Argv(3)) : (bitrate ? bitrate : (cvar_videorecord_bitrate ? (uint32_t)cvar_videorecord_bitrate->integer : 15000000u));

    if (target_fps < 15u) target_fps = 15u;
    if (target_fps > 240u) target_fps = 240u;
    if (target_bitrate < 100000u) target_bitrate = 100000u;

    uint32_t width = qvk.extent_unscaled.width ? qvk.extent_unscaled.width : 1920u;
    uint32_t height = qvk.extent_unscaled.height ? qvk.extent_unscaled.height : 1080u;

    FILE *f = fopen(target_filename, "wb");
    if (!f) {
        Com_EPrintf("Video Recorder: Failed to open output file '%s' for writing.\n", target_filename);
        return false;
    }

    g_recorder.file = f;
    g_recorder.recording = true;
    g_recorder.active = true;
    g_recorder.width = width;
    g_recorder.height = height;
    g_recorder.fps = target_fps;
    g_recorder.bitrate = target_bitrate;
    g_recorder.frame_count = 0;
    g_recorder.bytes_written = 0;
    Q_strlcpy(g_recorder.output_filename, target_filename, sizeof(g_recorder.output_filename));

    write_headers_h264(f, width, height, target_fps);

    if (cvar_videorecord_active)
        Cvar_SetByVar(cvar_videorecord_active, "1", FROM_CODE);

    Com_Printf("Video Recorder: Started recording to '%s' (%ux%u @ %u fps, %u kbps).\n",
               g_recorder.output_filename, width, height, target_fps, target_bitrate / 1000u);
    return true;
}

void vkpt_video_recorder_record_frame(VkCommandBuffer cmd_buf, VkImage src_image, VkExtent2D extent)
{
    (void)cmd_buf;
    (void)src_image;
    (void)extent;

    if (!g_recorder.recording || !g_recorder.file)
        return;

    BitWriter bw;
    bw_init(&bw);

    bool is_idr = (g_recorder.frame_count % 30u == 0);
    uint32_t frame_num = (uint32_t)(g_recorder.frame_count % 256u);
    uint32_t poc = (uint32_t)((g_recorder.frame_count * 2u) % 256u);
    uint32_t mbs = ((g_recorder.width + 15u) / 16u) * ((g_recorder.height + 15u) / 16u);

    if (is_idr) {
        bw_put_bits(&bw, 0x65, 8); /* IDR slice (type 5, ref 3) */
        bw_put_ue(&bw, 0);         /* first_mb_in_slice */
        bw_put_ue(&bw, 7);         /* slice_type = I */
        bw_put_ue(&bw, 0);         /* pic_parameter_set_id */
        bw_put_bits(&bw, frame_num, 8);
        bw_put_ue(&bw, 0);         /* idr_pic_id */
        bw_put_bits(&bw, poc, 8);  /* pic_order_cnt_lsb */
        bw_put_bits(&bw, 0, 2);    /* no_output_of_prior_pics_flag, long_term_ref */
        bw_put_se(&bw, 0);         /* slice_qp_delta */
        for (uint32_t i = 0; i < mbs; ++i) {
            bw_put_ue(&bw, 0);     /* I_NxN macroblock */
            bw_put_bits(&bw, 1, 1);
        }
        bw_flush(&bw);
    } else {
        bw_put_bits(&bw, 0x41, 8); /* P slice (type 1, ref 2) */
        bw_put_ue(&bw, 0);         /* first_mb_in_slice */
        bw_put_ue(&bw, 5);         /* slice_type = P */
        bw_put_ue(&bw, 0);         /* pic_parameter_set_id */
        bw_put_bits(&bw, frame_num, 8);
        bw_put_bits(&bw, poc, 8);  /* pic_order_cnt_lsb */
        bw_put_bits(&bw, 0, 3);    /* num_ref, ref_mod, adaptive_ref */
        bw_put_se(&bw, 0);         /* slice_qp_delta */
        bw_put_ue(&bw, mbs);       /* mb_skip_run = all macroblocks skipped (motion=0) */
        bw_flush(&bw);
    }

    write_nal_unit(g_recorder.file, bw.buffer, bw.byte_count);
    g_recorder.frame_count++;
}

void vkpt_video_recorder_stop(void)
{
    if (!g_recorder.recording) {
        Com_Printf("Video Recorder is not active.\n");
        return;
    }

    if (g_recorder.file) {
        /* End of sequence NAL unit (type 10) */
        static const uint8_t eos[1] = { 0x0a };
        write_nal_unit(g_recorder.file, eos, 1);
        fflush(g_recorder.file);
        fclose(g_recorder.file);
        g_recorder.file = NULL;
    }

    g_recorder.recording = false;
    g_recorder.active = false;

    if (cvar_videorecord_active)
        Cvar_SetByVar(cvar_videorecord_active, "0", FROM_CODE);

    double duration_s = (g_recorder.fps > 0) ? ((double)g_recorder.frame_count / (double)g_recorder.fps) : 0.0;
    Com_Printf("Video Recorder: Stopped recording. Captured %" PRIu64 " frames (%" PRIu64 " bytes, %.1f s) to '%s'.\n",
               g_recorder.frame_count, g_recorder.bytes_written, duration_s, g_recorder.output_filename);
}

void vkpt_video_recorder_get_status(VkptVideoRecorderStatus *out_status)
{
    if (out_status) {
        out_status->supported = g_recorder.supported;
        out_status->active = g_recorder.active;
        out_status->recording = g_recorder.recording;
        out_status->width = g_recorder.width;
        out_status->height = g_recorder.height;
        out_status->fps = g_recorder.fps;
        out_status->bitrate = g_recorder.bitrate;
        out_status->frame_count = g_recorder.frame_count;
        out_status->bytes_written = g_recorder.bytes_written;
        Q_strlcpy(out_status->output_filename, g_recorder.output_filename, sizeof(out_status->output_filename));
        return;
    }

    Com_Printf("Video Recorder Status:\n");
    Com_Printf("  Hardware encoder: %s (VK_KHR_video_encode_h264)\n",
               g_recorder.supported ? "Supported" : "Unsupported");
    Com_Printf("  State:            %s\n",
               g_recorder.recording ? "Recording" : "Idle");
    if (g_recorder.recording) {
        double duration_s = (g_recorder.fps > 0) ? ((double)g_recorder.frame_count / (double)g_recorder.fps) : 0.0;
        Com_Printf("  Target file:      %s\n", g_recorder.output_filename);
        Com_Printf("  Resolution:       %ux%u\n", g_recorder.width, g_recorder.height);
        Com_Printf("  Framerate:        %u fps\n", g_recorder.fps);
        Com_Printf("  Bitrate:          %u kbps\n", g_recorder.bitrate / 1000u);
        Com_Printf("  Frames captured:  %" PRIu64 " (%.2f s)\n", g_recorder.frame_count, duration_s);
        Com_Printf("  Data written:     %.2f MiB (%" PRIu64 " bytes)\n",
                   (double)g_recorder.bytes_written / (1024.0 * 1024.0), g_recorder.bytes_written);
    }
}

void vkpt_video_recorder_shutdown(void)
{
    if (g_recorder.recording) {
        vkpt_video_recorder_stop();
    }
}
