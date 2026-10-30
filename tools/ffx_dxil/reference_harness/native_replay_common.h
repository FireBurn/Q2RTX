// SPDX-License-Identifier: MIT
#pragma once
// Shared allocation helpers for isolated native replay experiments.
#include <vulkan/vulkan.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void check(VkResult result) {
    if (result != VK_SUCCESS) throw std::runtime_error("Vulkan result " + std::to_string(result));
}
static std::vector<char> read(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open input");
    auto n = f.tellg();
    if (n <= 0 || n > 64 * 1024 * 1024) throw std::runtime_error("Input size out of bounds");
    std::vector<char> data(static_cast<size_t>(n)); f.seekg(0);
    if (!f.read(data.data(), n)) throw std::runtime_error("Input read failed");
    return data;
}
static VkDevice device;
static VkPhysicalDevice physical;
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties p; vkGetPhysicalDeviceMemoryProperties(physical, &p);
    for (uint32_t i=0;i<p.memoryTypeCount;i++)
        if ((bits & (1u<<i)) && (p.memoryTypes[i].propertyFlags & required)==required) return i;
    throw std::runtime_error("Required memory unavailable");
}
struct Buffer { VkBuffer handle; VkDeviceMemory memory; void* mapped; };
static inline Buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer b{}; VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size=size; info.usage=usage; check(vkCreateBuffer(device,&info,nullptr,&b.handle));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,b.handle,&req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags=(usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)?VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT:0;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; alloc.pNext=&flags;
    alloc.allocationSize=req.size; alloc.memoryTypeIndex=memory_type(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(device,&alloc,nullptr,&b.memory)); check(vkBindBufferMemory(device,b.handle,b.memory,0));
    check(vkMapMemory(device,b.memory,0,size,0,&b.mapped)); return b;
}
struct Image { VkImage handle; VkDeviceMemory memory; VkImageView view; };
static inline Image image(uint32_t width,uint32_t height,VkFormat format, bool array_view=false, uint32_t layers=1) {
    Image im{}; VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType=VK_IMAGE_TYPE_2D; info.format=format; info.extent={width,height,1};
    info.mipLevels=1; info.arrayLayers=layers; info.samples=VK_SAMPLE_COUNT_1_BIT;
    info.usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    check(vkCreateImage(device,&info,nullptr,&im.handle));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(device,im.handle,&req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; alloc.allocationSize=req.size;
    alloc.memoryTypeIndex=memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(device,&alloc,nullptr,&im.memory)); check(vkBindImageMemory(device,im.handle,im.memory,0));
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; view.image=im.handle;
    view.viewType=array_view?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D; view.format=format; view.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,layers};
    check(vkCreateImageView(device,&view,nullptr,&im.view)); return im;
}
