/*
 * Copyright (c) 2026 Q2RTX FSR Vulkan contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef VIDEO_RECORDER_H
#define VIDEO_RECORDER_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct VkptVideoRecorderStatus {
    bool supported;
    bool active;
    bool recording;
    bool hw_session_active;
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    uint32_t bitrate;
    uint64_t frame_count;
    uint64_t bytes_written;
    char output_filename[256];
} VkptVideoRecorderStatus;

/* Initialize video recording subsystem, cvars, and console commands */
void vkpt_video_recorder_init(void);

/* Query capabilities and check for VK_KHR_video_encode_h264 support */
void vkpt_video_recorder_query_capabilities(VkPhysicalDevice physicalDevice);

/* Start hardware video recording */
bool vkpt_video_recorder_start(const char *filename, uint32_t fps, uint32_t bitrate);

/* Record/encode current frame */
void vkpt_video_recorder_record_frame(VkCommandBuffer cmd_buf, VkImage src_image, VkExtent2D extent);

/* Stop video recording */
void vkpt_video_recorder_stop(void);

/* Get current status */
void vkpt_video_recorder_get_status(VkptVideoRecorderStatus *out_status);

/* Shutdown and release resources */
void vkpt_video_recorder_shutdown(void);

#if defined(__cplusplus)
}
#endif

#endif /* VIDEO_RECORDER_H */
