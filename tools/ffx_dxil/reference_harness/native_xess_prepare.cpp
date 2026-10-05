// SPDX-License-Identifier: MIT
// Native XeSS prepare-pass experiment. Does not execute the full upscaler.
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
static Buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
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
static Image image(uint32_t width,uint32_t height,VkFormat format, bool array_view=false) {
    Image im{}; VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType=VK_IMAGE_TYPE_2D; info.format=format; info.extent={width,height,1};
    info.mipLevels=info.arrayLayers=1; info.samples=VK_SAMPLE_COUNT_1_BIT;
    info.usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    check(vkCreateImage(device,&info,nullptr,&im.handle));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(device,im.handle,&req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; alloc.allocationSize=req.size;
    alloc.memoryTypeIndex=memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(device,&alloc,nullptr,&im.memory)); check(vkBindImageMemory(device,im.handle,im.memory,0));
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; view.image=im.handle;
    view.viewType=array_view?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D; view.format=format; view.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    check(vkCreateImageView(device,&view,nullptr,&im.view)); return im;
}

int main(int argc,char** argv) try {
    if (argc!=4) throw std::runtime_error("Usage: native_xess_prepare normalized-40553fb1ea6dd6f3.spv push-120.bin output.rgba32ui");
    const auto shader=read(argv[1]), constants=read(argv[2]);
    if (shader.size()%4 || constants.size()!=120) throw std::runtime_error("Wrong capture sizes");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance instance; check(vkCreateInstance(&ici,nullptr,&instance));
    uint32_t count=0; check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
    std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
    VkPhysicalDeviceVulkan12Features enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    uint32_t family=UINT32_MAX;
    for (auto candidate:devices) {
        VkPhysicalDeviceVulkan12Features available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 probe{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; probe.pNext=&available;
        vkGetPhysicalDeviceFeatures2(candidate,&probe);
        if (!available.runtimeDescriptorArray || !available.bufferDeviceAddress || !probe.features.shaderStorageImageWriteWithoutFormat ||
            !probe.features.shaderSampledImageArrayDynamicIndexing || !probe.features.shaderStorageImageArrayDynamicIndexing) continue;
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(candidate,&props);
        if (props.apiVersion<VK_API_VERSION_1_3 || props.limits.maxPerStageDescriptorStorageImages<32 || props.limits.maxPerStageDescriptorSampledImages<16) continue;
        uint32_t n=0; vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr);
        std::vector<VkQueueFamilyProperties> queues(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,queues.data());
        for (uint32_t i=0;i<n;++i) if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { physical=candidate; family=i; break; }
        if (physical) { std::printf("Native Vulkan device: %s\n",props.deviceName); break; }
    }
    if (!physical) throw std::runtime_error("No device with captured shader requirements");
    enabled.runtimeDescriptorArray=enabled.bufferDeviceAddress=VK_TRUE;
    features.pNext=&enabled;
    features.features.shaderStorageImageWriteWithoutFormat=VK_TRUE;
    features.features.shaderSampledImageArrayDynamicIndexing=VK_TRUE;
    features.features.shaderStorageImageArrayDynamicIndexing=VK_TRUE;
    float priority=1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex=family; qi.queueCount=1; qi.pQueuePriorities=&priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext=&features; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi;
    check(vkCreateDevice(physical,&di,nullptr,&device)); VkQueue queue; vkGetDeviceQueue(device,family,0,&queue);
    // Colour, velocity, depth, history colour, history mask, output, prepared colour, packed features.
    std::array<Image,8> images={image(753,424,VK_FORMAT_R16G16B16A16_SFLOAT),image(753,424,VK_FORMAT_R16G16_SFLOAT),
        image(753,424,VK_FORMAT_R32_SFLOAT),image(1280,720,VK_FORMAT_R16G16B16A16_SFLOAT),image(1280,720,VK_FORMAT_R8_UNORM),
        image(1280,720,VK_FORMAT_R16G16B16A16_SFLOAT),image(1280,720,VK_FORMAT_R16G16B16A16_SFLOAT),
        image(640,360,VK_FORMAT_R32G32B32A32_UINT,true)};
    Buffer upload=buffer(753*424*8,VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    for (unsigned y=0;y<424;++y) for (unsigned x=0;x<753;++x) {
        uint16_t rgba[]={0x3800,0x3000,0x3400,0x3c00};
        if (((x/32)^(y/32))&1) std::swap(rgba[0],rgba[1]);
        std::memcpy(static_cast<char*>(upload.mapped)+(y*753+x)*8,rgba,8);
    }
    Buffer output=buffer(640*360*16,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU=si.addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod=VK_LOD_CLAMP_NONE; VkSampler sampler; check(vkCreateSampler(device,&si,nullptr,&sampler));
    std::array<VkDescriptorSetLayoutBinding,4> bindings={{{0,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,16,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,16,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,16,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
        {3,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount=bindings.size(); li.pBindings=bindings.data();
    VkDescriptorSetLayout set_layout; check(vkCreateDescriptorSetLayout(device,&li,nullptr,&set_layout));
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,16},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,32},{VK_DESCRIPTOR_TYPE_SAMPLER,1}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets=1; pi.poolSizeCount=3; pi.pPoolSizes=sizes;
    VkDescriptorPool pool; check(vkCreateDescriptorPool(device,&pi,nullptr,&pool));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=pool; ai.descriptorSetCount=1; ai.pSetLayouts=&set_layout;
    VkDescriptorSet set; check(vkAllocateDescriptorSets(device,&ai,&set));
    for (unsigned binding=0;binding<4;++binding) for (unsigned slot=0;slot<(binding==3?1:16);++slot) {
        unsigned im=binding==2?7:binding==1?6:0;
        if (binding==0) { if (slot==1) im=1; if (slot==3) im=2; if (slot==8) im=3; if (slot==9) im=4; }
        if (binding==1 && slot==2) im=5;
        VkDescriptorImageInfo info{sampler,images[im].view,VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=set; write.dstBinding=binding; write.dstArrayElement=slot;
        write.descriptorCount=1; write.descriptorType=bindings[binding].descriptorType; write.pImageInfo=&info;
        vkUpdateDescriptorSets(device,1,&write,0,nullptr);
    }
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,120};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=1; pli.pSetLayouts=&set_layout; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&push;
    VkPipelineLayout layout; check(vkCreatePipelineLayout(device,&pli,nullptr,&layout));
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=shader.size(); smi.pCode=reinterpret_cast<const uint32_t*>(shader.data());
    VkShaderModule module; check(vkCreateShaderModule(device,&smi,nullptr,&module));
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pci.layout=layout;
    pci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    VkPipeline pipeline; check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pci,nullptr,&pipeline));
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=family;
    VkCommandPool commands; check(vkCreateCommandPool(device,&cpi,nullptr,&commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool=commands; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device,&cai,&cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(cmd,&bi));
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    for (unsigned i=0;i<images.size();++i) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; barrier.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout=VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.image=images[i].handle; barrier.subresourceRange=range; barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
        VkClearColorValue clear{};
        if (i==2) clear.float32[0]=0.5f;
        if (i==7) for (auto& value:clear.uint32) value=0xdeadbeef;
        vkCmdClearColorImage(cmd,images[i].handle,VK_IMAGE_LAYOUT_GENERAL,&clear,1,&range);
    }
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent={753,424,1};
    vkCmdCopyBufferToImage(cmd,upload.handle,images[0].handle,VK_IMAGE_LAYOUT_GENERAL,1,&copy);
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline); vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
    vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,120,constants.data()); vkCmdDispatch(cmd,80,23,1);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    copy.imageExtent={640,360,1}; vkCmdCopyImageToBuffer(cmd,images[7].handle,VK_IMAGE_LAYOUT_GENERAL,output.handle,1,&copy);
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    check(vkEndCommandBuffer(cmd)); VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&cmd;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; check(vkCreateFence(device,&fi,nullptr,&fence));
    check(vkQueueSubmit(queue,1,&submit,fence)); check(vkWaitForFences(device,1,&fence,VK_TRUE,30000000000ull));
    unsigned sentinels=0; const auto* values=static_cast<const uint32_t*>(output.mapped);
    for (unsigned i=0;i<640*360*4;++i) sentinels+=values[i]==0xdeadbeef;
    std::ofstream file(argv[3],std::ios::binary); file.write(static_cast<const char*>(output.mapped),640*360*16); file.close();
    if (!file) throw std::runtime_error("Output write failed");
    std::printf("NATIVE_XESS_PREPARE sentinel_words=%u total_words=%u reference_match=unverified\n",sentinels,640*360*4);
    vkDestroyFence(device,fence,nullptr); vkDestroyCommandPool(device,commands,nullptr); vkDestroyPipeline(device,pipeline,nullptr);
    vkDestroyShaderModule(device,module,nullptr); vkDestroyPipelineLayout(device,layout,nullptr); vkDestroyDescriptorPool(device,pool,nullptr);
    vkDestroyDescriptorSetLayout(device,set_layout,nullptr); vkDestroySampler(device,sampler,nullptr);
    for (auto im:images) {vkDestroyImageView(device,im.view,nullptr);vkDestroyImage(device,im.handle,nullptr);vkFreeMemory(device,im.memory,nullptr);}
    for (auto b:{upload,output}) {vkUnmapMemory(device,b.memory);vkDestroyBuffer(device,b.handle,nullptr);vkFreeMemory(device,b.memory,nullptr);}
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    return sentinels?1:0;
} catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
