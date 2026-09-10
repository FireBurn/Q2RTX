// SPDX-License-Identifier: MIT
// Isolated native replay of captured 640x360 official-provider SPD, not an upscaler.
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
static Image image(uint32_t width,uint32_t height,VkFormat format) {
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
    view.viewType=VK_IMAGE_VIEW_TYPE_2D; view.format=format; view.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    check(vkCreateImageView(device,&view,nullptr,&im.view)); return im;
}
int main(int argc,char** argv) try {
    if (argc!=4) throw std::runtime_error("Usage: native_spd_replay captured.spv constant-ring.bin colour-upload.bin");
    auto shader=read(argv[1]), constants=read(argv[2]), colour=read(argv[3]);
    if (shader.size()%4 || constants.size()<32 || colour.size()!=640*360*8) throw std::runtime_error("Wrong capture sizes");
    uint32_t magic; std::memcpy(&magic,shader.data(),4);
    if (magic!=0x07230203) throw std::runtime_error("Not SPIR-V");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; instance_info.pApplicationInfo=&app;
    VkInstance instance; check(vkCreateInstance(&instance_info,nullptr,&instance));
    uint32_t count=0; check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
    std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
    uint32_t family=UINT32_MAX;
    for (auto candidate:devices) {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(candidate,&props);
        if (props.vendorID!=0x1002 || props.deviceType!=VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        uint32_t n=0; vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr);
        std::vector<VkQueueFamilyProperties> q(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,q.data());
        for (uint32_t i=0;i<n;i++) if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {physical=candidate; family=i; break;}
        if (physical) {std::printf("Native device: %s\n",props.deviceName); break;}
    }
    if (!physical) throw std::runtime_error("No AMD discrete compute device");
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutable_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; v12.pNext=&mutable_feature;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext=&v12;
    vkGetPhysicalDeviceFeatures2(physical,&features);
    if (!v12.bufferDeviceAddress || !v12.runtimeDescriptorArray || !mutable_feature.mutableDescriptorType ||
        !features.features.shaderStorageImageReadWithoutFormat || !features.features.shaderStorageImageWriteWithoutFormat)
        throw std::runtime_error("Missing captured shader features");
    float priority=1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex=family; qi.queueCount=1; qi.pQueuePriorities=&priority;
    const char* extension=VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext=&features;
    di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi; di.enabledExtensionCount=1; di.ppEnabledExtensionNames=&extension;
    check(vkCreateDevice(physical,&di,nullptr,&device)); VkQueue queue; vkGetDeviceQueue(device,family,0,&queue);
    Buffer cb=buffer(32,VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(cb.mapped,constants.data(),32);
    Buffer upload=buffer(colour.size(),VK_BUFFER_USAGE_TRANSFER_SRC_BIT); std::memcpy(upload.mapped,colour.data(),colour.size());
    Buffer output=buffer(8,VK_BUFFER_USAGE_TRANSFER_DST_BIT); std::memset(output.mapped,0xff,8);
    std::array<Image,4> images={image(1,1,VK_FORMAT_R32_UINT),image(40,23,VK_FORMAT_R32_SFLOAT),
        image(2,1,VK_FORMAT_R32_SFLOAT),image(640,360,VK_FORMAT_R16G16B16A16_SFLOAT)};
    VkDescriptorType types[]={VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
    VkMutableDescriptorTypeListEXT list{2,types};
    VkMutableDescriptorTypeCreateInfoEXT mutable_info{VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT};
    mutable_info.mutableDescriptorTypeListCount=1; mutable_info.pMutableDescriptorTypeLists=&list;
    VkDescriptorSetLayoutBinding binding{1,VK_DESCRIPTOR_TYPE_MUTABLE_EXT,5,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.pNext=&mutable_info;
    li.bindingCount=1; li.pBindings=&binding;
    VkDescriptorSetLayout layouts[2]; check(vkCreateDescriptorSetLayout(device,&li,nullptr,&layouts[1]));
    li.pNext=nullptr; li.bindingCount=0; li.pBindings=nullptr; check(vkCreateDescriptorSetLayout(device,&li,nullptr,&layouts[0]));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_MUTABLE_EXT,5};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.pNext=&mutable_info;
    pi.maxSets=1; pi.poolSizeCount=1; pi.pPoolSizes=&ps;
    VkDescriptorPool pool; check(vkCreateDescriptorPool(device,&pi,nullptr,&pool));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=pool;
    ai.descriptorSetCount=1; ai.pSetLayouts=&layouts[1]; VkDescriptorSet set; check(vkAllocateDescriptorSets(device,&ai,&set));
    for (uint32_t i=0;i<5;i++) {
        VkDescriptorImageInfo ii{VK_NULL_HANDLE,images[i==4?3:(i==3?0:i)].view,VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet=set; w.dstBinding=1; w.dstArrayElement=i;
        w.descriptorCount=1; w.descriptorType=i==4?VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo=&ii; vkUpdateDescriptorSets(device,1,&w,0,nullptr);
    }
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=2; pli.pSetLayouts=layouts;
    pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&push; VkPipelineLayout layout; check(vkCreatePipelineLayout(device,&pli,nullptr,&layout));
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=shader.size(); smi.pCode=reinterpret_cast<const uint32_t*>(shader.data());
    VkShaderModule module; check(vkCreateShaderModule(device,&smi,nullptr,&module));
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pci.layout=layout;
    pci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    VkPipeline pipeline; check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pci,nullptr,&pipeline));
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=family;
    VkCommandPool commands; check(vkCreateCommandPool(device,&cpi,nullptr,&commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool=commands; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device,&cai,&cmd));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(cmd,&begin));
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    for (auto im:images) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; barrier.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout=VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.image=im.handle; barrier.subresourceRange=range; barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
    }
    VkClearColorValue zero{}; for (unsigned i=0;i<3;i++) vkCmdClearColorImage(cmd,images[i].handle,VK_IMAGE_LAYOUT_GENERAL,&zero,1,&range);
    VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent={640,360,1};
    vkCmdCopyBufferToImage(cmd,upload.handle,images[3].handle,VK_IMAGE_LAYOUT_GENERAL,1,&copy);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bai.buffer=cb.handle;
    struct {uint64_t cb; uint32_t uav,srv;} values{vkGetBufferDeviceAddress(device,&bai),0,4};
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline); vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,1,1,&set,0,nullptr);
    vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&values); vkCmdDispatch(cmd,10,6,1);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    copy.imageExtent={2,1,1}; vkCmdCopyImageToBuffer(cmd,images[2].handle,VK_IMAGE_LAYOUT_GENERAL,output.handle,1,&copy);
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    check(vkEndCommandBuffer(cmd)); VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&cmd;
    check(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE)); check(vkQueueWaitIdle(queue));
    float result[2]; std::memcpy(result,output.mapped,8);
    std::printf("NATIVE_SPD exposure=%g previous=%g finite=%u (reference comparison pending)\n",result[0],result[1],unsigned(std::isfinite(result[0])&&std::isfinite(result[1])));
    vkDestroyCommandPool(device,commands,nullptr);
    vkDestroyPipeline(device,pipeline,nullptr); vkDestroyShaderModule(device,module,nullptr);
    vkDestroyPipelineLayout(device,layout,nullptr); vkDestroyDescriptorPool(device,pool,nullptr);
    for (auto l:layouts) vkDestroyDescriptorSetLayout(device,l,nullptr);
    for (auto im:images) {
        vkDestroyImageView(device,im.view,nullptr); vkDestroyImage(device,im.handle,nullptr);
        vkFreeMemory(device,im.memory,nullptr);
    }
    for (auto b:{cb,upload,output}) {
        vkUnmapMemory(device,b.memory); vkDestroyBuffer(device,b.handle,nullptr); vkFreeMemory(device,b.memory,nullptr);
    }
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    return std::isfinite(result[0])&&std::isfinite(result[1])&&result[0]>0 ? 0:1;
} catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
