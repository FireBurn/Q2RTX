// SPDX-License-Identifier: MIT
#pragma once
#include "native_replay_common.h"
#include <filesystem>

// Optional tightly packed captures for the fixed 753x424 replay profile.
// Both prepare and resolve must receive the same directory for a frame.
struct CapturedXeSSInputs {
    std::array<Buffer,3> uploads{};
    CapturedXeSSInputs() {
        const char* directory = std::getenv("XESS_REPLAY_INPUTS");
        if (!directory) return;
        const char* names[] = {"color.rgba16f", "motion.rg16f", "depth.r32f"};
        const size_t sizes[] = {753*424*8,753*424*4,753*424*4};
        std::array<std::vector<char>,3> data;
        for (unsigned i=0;i<3;++i) {
            data[i]=read((std::filesystem::path(directory)/names[i]).c_str());
            if (data[i].size()!=sizes[i]) throw std::runtime_error("Captured input shape mismatch");
        }
        for (unsigned i=0;i<3;++i) {
            uploads[i]=buffer(sizes[i],VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            std::memcpy(uploads[i].mapped,data[i].data(),sizes[i]);
        }
    }
    void record(VkCommandBuffer cmd, const Image* images) const {
        if (!uploads[0].handle) return;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask=barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,1,&barrier,0,nullptr,0,nullptr);
        VkBufferImageCopy copy{};
        copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
        copy.imageExtent={753,424,1};
        for (unsigned i=0;i<3;++i)
            vkCmdCopyBufferToImage(cmd,uploads[i].handle,images[i].handle,VK_IMAGE_LAYOUT_GENERAL,1,&copy);
    }
    void release() {
        for (auto& b:uploads) if (b.handle) {
            vkUnmapMemory(device,b.memory); vkDestroyBuffer(device,b.handle,nullptr);
            vkFreeMemory(device,b.memory,nullptr); b={};
        }
    }
};
