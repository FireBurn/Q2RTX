// SPDX-License-Identifier: MIT
// Captured first-frame XeSS resolve. Fixed checkerboard profile, not an engine backend.
#include "native_replay_common.h"
#include "native_xess_inputs.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>

using Json = nlohmann::json;

int main(int argc, char** argv) try {
    if (argc != 5) throw std::runtime_error("Usage: native_xess_resolve manifest.json features.rgba32ui prepared.rgba16f NEW_OUTPUT_DIR");
    std::ifstream input_manifest(argv[1]); Json manifest; input_manifest >> manifest;
    if (manifest.at("schema") != "xess.native-resolve-replay" || manifest.at("version") != 1)
        throw std::runtime_error("Unsupported resolve manifest");
    const auto shader = read(manifest.at("shader").get<std::string>().c_str());
    const auto packed = read(argv[2]), prepared = read(argv[3]);
    const auto constants = manifest.at("push").get<std::array<uint32_t, 23>>();
    const auto groups = manifest.at("groups").get<std::array<uint32_t, 3>>();
    if (shader.size() % 4 || packed.size() != 640*360*16 || prepared.size() != 1280*720*8 ||
        groups != std::array<uint32_t, 3>{160, 90, 1}) throw std::runtime_error("Capture shape mismatch");
    if (!std::filesystem::create_directory(argv[4])) throw std::runtime_error("Output directory must be new");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo = &app;
    VkInstance instance; check(vkCreateInstance(&ii, nullptr, &instance));
    uint32_t count = 0; check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    uint32_t family = UINT32_MAX;
    for (auto candidate : devices) {
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &f13;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !f12.runtimeDescriptorArray || !f12.bufferDeviceAddress ||
            !f12.uniformBufferStandardLayout || !f13.shaderIntegerDotProduct ||
            !features.features.shaderStorageImageWriteWithoutFormat ||
            !features.features.shaderSampledImageArrayDynamicIndexing ||
            !features.features.shaderStorageImageArrayDynamicIndexing ||
            !features.features.shaderUniformBufferArrayDynamicIndexing ||
            props.limits.maxPerStageDescriptorUniformBuffers < 24 ||
            props.limits.maxPerStageDescriptorSampledImages < 24 ||
            props.limits.maxPerStageDescriptorStorageImages < 12) continue;
        uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
        std::vector<VkQueueFamilyProperties> queues(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, queues.data());
        for (uint32_t i = 0; i < n; ++i) if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { physical = candidate; family = i; break; }
        if (physical) { std::printf("Native resolve device: %s\n", props.deviceName); break; }
    }
    if (!physical) throw std::runtime_error("No device with captured resolve requirements");
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.shaderIntegerDotProduct = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &f13;
    f12.runtimeDescriptorArray = f12.bufferDeviceAddress = f12.uniformBufferStandardLayout = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext = &f12;
    features.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    features.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    features.features.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
    features.features.shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
    float priority = 1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext = &features; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    check(vkCreateDevice(physical, &di, nullptr, &device)); VkQueue queue; vkGetDeviceQueue(device, family, 0, &queue);

    // Input colour, motion, depth, model features, prepared colour, output, history, mask.
    std::array<Image, 8> images = {image(753,424,VK_FORMAT_R16G16B16A16_SFLOAT), image(753,424,VK_FORMAT_R16G16_SFLOAT),
        image(753,424,VK_FORMAT_R32_SFLOAT), image(640,360,VK_FORMAT_R32G32B32A32_UINT,true),
        image(1280,720,VK_FORMAT_R16G16B16A16_SFLOAT), image(1280,720,VK_FORMAT_R16G16B16A16_SFLOAT),
        image(1280,720,VK_FORMAT_R16G16B16A16_SFLOAT), image(1280,720,VK_FORMAT_R8_UNORM)};
    std::array<Buffer, 3> uploads = {buffer(753*424*8,VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        buffer(packed.size(),VK_BUFFER_USAGE_TRANSFER_SRC_BIT), buffer(prepared.size(),VK_BUFFER_USAGE_TRANSFER_SRC_BIT)};
    for (unsigned y = 0; y < 424; ++y) for (unsigned x = 0; x < 753; ++x) {
        uint16_t rgba[] = {0x3800,0x3000,0x3400,0x3c00};
        if (((x/32)^(y/32))&1) std::swap(rgba[0],rgba[1]);
        std::memcpy(static_cast<char*>(uploads[0].mapped)+(y*753+x)*8,rgba,8);
    }
    std::memcpy(uploads[1].mapped,packed.data(),packed.size()); std::memcpy(uploads[2].mapped,prepared.data(),prepared.size());
    CapturedXeSSInputs captured_inputs;
    std::array<Buffer, 3> models{}; std::array<size_t, 3> model_sizes{};
    for (size_t i = 0; i < models.size(); ++i) {
        auto data = read(manifest.at("buffers").at(i).at("file").get<std::string>().c_str());
        if (data.size() != manifest.at("buffers").at(i).at("view_bytes").get<size_t>()) throw std::runtime_error("CBV size mismatch");
        model_sizes[i] = data.size(); models[i] = buffer(data.size(),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::memcpy(models[i].mapped,data.data(),data.size());
    }
    const std::array<size_t, 3> output_sizes = {1280*720*8,1280*720*8,1280*720};
    std::array<Buffer, 3> outputs{};
    for (size_t i = 0; i < outputs.size(); ++i) outputs[i] = buffer(output_sizes[i],VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    const VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    for (uint32_t i = 0; i < 5; ++i) bindings[i] = {i,types[i],12,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount = 5; li.pBindings = bindings.data();
    VkDescriptorSetLayout set_layout; check(vkCreateDescriptorSetLayout(device,&li,nullptr,&set_layout));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,24},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,12},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,24}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets = 1; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
    VkDescriptorPool pool; check(vkCreateDescriptorPool(device,&pi,nullptr,&pool));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &set_layout;
    VkDescriptorSet set; check(vkAllocateDescriptorSets(device,&ai,&set));
    for (uint32_t binding = 0; binding < 5; ++binding) for (uint32_t slot = 0; slot < 12; ++slot) {
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet = set; write.dstBinding = binding;
        write.dstArrayElement = slot; write.descriptorCount = 1; write.descriptorType = types[binding];
        VkDescriptorImageInfo im{}; VkDescriptorBufferInfo b{};
        if (binding >= 3) {
            unsigned index = slot >= 9 ? slot - 9 : 0;
            b = {models[index].handle,0,model_sizes[index]}; write.pBufferInfo = &b;
        } else {
            unsigned index = 0;
            if (binding == 0) { if (slot == 1) index = 1; if (slot == 3) index = 2; if (slot == 5) index = 4; }
            if (binding == 1) index = 3;
            if (binding == 2) index = slot == 7 ? 6 : slot == 8 ? 7 : 5;
            im.imageView = images[index].view; im.imageLayout = VK_IMAGE_LAYOUT_GENERAL; write.pImageInfo = &im;
        }
        vkUpdateDescriptorSets(device,1,&write,0,nullptr);
    }
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,92};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &push;
    VkPipelineLayout layout; check(vkCreatePipelineLayout(device,&pli,nullptr,&layout));
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = shader.size(); smi.pCode = reinterpret_cast<const uint32_t*>(shader.data());
    VkShaderModule module; check(vkCreateShaderModule(device,&smi,nullptr,&module));
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pci.layout = layout;
    pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    VkPipeline pipeline; check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pci,nullptr,&pipeline));
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = family;
    VkCommandPool commands; check(vkCreateCommandPool(device,&cpi,nullptr,&commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = commands; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device,&cai,&cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(cmd,&bi));
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    for (size_t i = 0; i < images.size(); ++i) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = images[i].handle; b.subresourceRange = range; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
        VkClearColorValue clear{}; if (i == 2) clear.float32[0] = 0.5f;
        if (i == 5) for (auto& value : clear.float32) value = NAN;
        vkCmdClearColorImage(cmd,images[i].handle,VK_IMAGE_LAYOUT_GENERAL,&clear,1,&range);
    }
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&b,0,nullptr,0,nullptr);
    const unsigned destinations[] = {0,3,4};
    const VkExtent3D extents[] = {{753,424,1},{640,360,1},{1280,720,1}};
    VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
    for (unsigned i = 0; i < 3; ++i) { copy.imageExtent = extents[i]; vkCmdCopyBufferToImage(cmd,uploads[i].handle,images[destinations[i]].handle,VK_IMAGE_LAYOUT_GENERAL,1,&copy); }
    captured_inputs.record(cmd,images.data());
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline); vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
    vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,92,constants.data()); vkCmdDispatch(cmd,groups[0],groups[1],groups[2]);
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&b,0,nullptr,0,nullptr);
    copy.imageExtent = {1280,720,1};
    for (unsigned i = 0; i < 3; ++i) vkCmdCopyImageToBuffer(cmd,images[5+i].handle,VK_IMAGE_LAYOUT_GENERAL,outputs[i].handle,1,&copy);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&b,0,nullptr,0,nullptr);
    check(vkEndCommandBuffer(cmd)); VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; check(vkCreateFence(device,&fi,nullptr,&fence));
    check(vkQueueSubmit(queue,1,&submit,fence)); check(vkWaitForFences(device,1,&fence,VK_TRUE,30000000000ull));
    const char* names[] = {"output.rgba16f","history.rgba16f","mask.r8"};
    for (unsigned i = 0; i < 3; ++i) {
        std::ofstream file(std::filesystem::path(argv[4])/names[i],std::ios::binary);
        file.write(static_cast<const char*>(outputs[i].mapped),output_sizes[i]); file.close();
        if (!file) throw std::runtime_error("Output write failed");
    }
    size_t nonfinite = 0; const auto* halves = static_cast<const uint16_t*>(outputs[0].mapped);
    for (unsigned i = 0; i < 1280*720*4; ++i) if ((i&3) != 3) nonfinite += (halves[i]&0x7c00) == 0x7c00;
    std::printf("NATIVE_XESS_RESOLVE nonfinite_rgb=%zu reference_match=unverified\n",nonfinite);
    vkDestroyFence(device,fence,nullptr); vkDestroyCommandPool(device,commands,nullptr); vkDestroyPipeline(device,pipeline,nullptr);
    vkDestroyShaderModule(device,module,nullptr); vkDestroyPipelineLayout(device,layout,nullptr); vkDestroyDescriptorPool(device,pool,nullptr); vkDestroyDescriptorSetLayout(device,set_layout,nullptr);
    for (auto im : images) { vkDestroyImageView(device,im.view,nullptr); vkDestroyImage(device,im.handle,nullptr); vkFreeMemory(device,im.memory,nullptr); }
    for (auto list : {uploads,models,outputs}) for (auto allocation : list) { vkUnmapMemory(device,allocation.memory); vkDestroyBuffer(device,allocation.handle,nullptr); vkFreeMemory(device,allocation.memory,nullptr); }
    captured_inputs.release();
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    return nonfinite ? 1 : 0;
} catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
