// SPDX-License-Identifier: MIT
// Native execution of the captured learned convolution stack (passes 1..12).
// Not a complete upscaler: prepare, final resolve and history are separate.
#include "native_replay_common.h"
#include "ffx_vk_xess_convolution_graph.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <map>
#include <string>

using Json = nlohmann::json;
struct Tensor { Image image; uint32_t width, height, layers; };
struct Pass {
    VkShaderModule module{};
    VkPipeline pipeline{};
    VkDescriptorSet descriptors{};
    std::array<uint32_t, 6> push{};
    std::array<uint32_t, 3> groups{};
    std::string output;
    Buffer readback{};
    size_t bytes{};
};

static void barrier(VkCommandBuffer cmd, VkPipelineStageFlags source,
    VkPipelineStageFlags destination, VkAccessFlags from, VkAccessFlags to)
{
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = from; b.dstAccessMask = to;
    vkCmdPipelineBarrier(cmd, source, destination, 0, 1, &b, 0, nullptr, 0, nullptr);
}

int main(int argc, char** argv) try {
    if (argc != 4 && argc != 5)
        throw std::runtime_error("Usage: native_xess_convolutions manifest.json prepare.rgba32ui NEW_OUTPUT_DIR [--zero-weights|--resident-repeat]");
    const bool zero = argc == 5 && std::strcmp(argv[4], "--zero-weights") == 0;
    const bool resident = argc == 5 && std::strcmp(argv[4], "--resident-repeat") == 0;
    if (argc == 5 && !zero && !resident) throw std::runtime_error("Unknown option");
    std::ifstream mf(argv[1]);
    Json manifest; mf >> manifest;
    if (manifest.at("schema") != "xess.native-convolution-replay" || manifest.at("version") != 1 ||
        manifest.at("passes").size() != 12)
        throw std::runtime_error("Unsupported replay manifest");
    if (!std::filesystem::create_directory(argv[3]))
        throw std::runtime_error("Output directory must be new");
    const auto input = read(argv[2]);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo = &app;
    VkInstance instance; check(vkCreateInstance(&ii, nullptr, &instance));
    uint32_t count = 0; check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> candidates(count);
    check(vkEnumeratePhysicalDevices(instance, &count, candidates.data()));
    uint32_t family = UINT32_MAX;
    for (auto candidate : candidates) {
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &f13;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        const auto& f = features.features;
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !f12.runtimeDescriptorArray || !f12.bufferDeviceAddress ||
            !f12.shaderInt8 || !f13.shaderIntegerDotProduct || !f.shaderStorageImageWriteWithoutFormat ||
            !f.shaderSampledImageArrayDynamicIndexing || !f.shaderStorageImageArrayDynamicIndexing ||
            !f.shaderStorageBufferArrayDynamicIndexing || props.limits.maxPerStageDescriptorStorageBuffers < 6 ||
            props.limits.maxPerStageDescriptorSampledImages < 6 || props.limits.maxPerStageDescriptorStorageImages < 6)
            continue;
        uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
        std::vector<VkQueueFamilyProperties> queues(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, queues.data());
        for (uint32_t i = 0; i < n; ++i)
            if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { physical = candidate; family = i; break; }
        if (physical) { std::printf("Native convolution device: %s\n", props.deviceName); break; }
    }
    if (!physical) throw std::runtime_error("No device with captured convolution requirements");
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.shaderIntegerDotProduct = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &f13;
    f12.runtimeDescriptorArray = f12.bufferDeviceAddress = f12.shaderInt8 = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext = &f12;
    features.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    features.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    features.features.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
    features.features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext = &features;
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    check(vkCreateDevice(physical, &di, nullptr, &device));
    VkQueue queue; vkGetDeviceQueue(device, family, 0, &queue);

    std::map<std::string, Tensor> images;
    for (auto it = manifest.at("images").begin(); it != manifest.at("images").end(); ++it) {
        auto shape = it.value().get<std::array<uint32_t, 3>>();
        if (!shape[0] || !shape[1] || !shape[2] || shape[0] > 4096 || shape[1] > 4096 || shape[2] > 64)
            throw std::runtime_error("Tensor shape exceeds experiment bounds");
        images[it.key()] = {image(shape[0], shape[1], VK_FORMAT_R32G32B32A32_UINT, true, shape[2]),
            shape[0], shape[1], shape[2]};
    }
    std::map<std::string, Buffer> buffers;
    std::map<std::string, size_t> buffer_sizes;
    for (auto it = manifest.at("buffers").begin(); it != manifest.at("buffers").end(); ++it) {
        const auto data = read(it.value().at("file").get<std::string>().c_str());
        if (data.size() != it.value().at("size").get<size_t>() || data.size() % 16)
            throw std::runtime_error("Model buffer size mismatch");
        auto b = buffer(data.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(b.mapped, data.data(), data.size());
        buffers[it.key()] = b; buffer_sizes[it.key()] = data.size();
    }
    if (zero) for (const auto& p : manifest.at("passes")) {
        const auto rid = p.at("buffers").at(0).get<std::string>();
        std::memset(buffers.at(rid).mapped, 0, buffer_sizes.at(rid));
    }
    const auto& first = images.at(manifest.at("input").get<std::string>());
    if (input.size() != size_t(first.width) * first.height * first.layers * 16)
        throw std::runtime_error("Prepare output shape mismatch");
    Buffer upload = buffer(input.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(upload.mapped, input.data(), input.size());

    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount = 3; li.pBindings = bindings;
    VkDescriptorSetLayout set_layout; check(vkCreateDescriptorSetLayout(device, &li, nullptr, &set_layout));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 72},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 72}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 72}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets = 12; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
    VkDescriptorPool pool; check(vkCreateDescriptorPool(device, &pi, nullptr, &pool));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &push;
    VkPipelineLayout layout; check(vkCreatePipelineLayout(device, &pli, nullptr, &layout));
    std::vector<Pass> passes;
    for (const auto& spec : manifest.at("passes")) {
        Pass p;
        p.push = spec.at("push").get<std::array<uint32_t, 6>>();
        p.groups = spec.at("groups").get<std::array<uint32_t, 3>>();
        for (auto n : p.groups) if (!n || n > 65535) throw std::runtime_error("Invalid dispatch size");
        p.output = spec.at("images").at(2).get<std::string>();
        const auto& target = images.at(p.output);
        p.bytes = size_t(target.width) * target.height * target.layers * 16;
        p.readback = buffer(p.bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto code = read(spec.at("shader").get<std::string>().c_str());
        if (code.size() % 4) throw std::runtime_error("Shader alignment");
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
        check(vkCreateShaderModule(device, &smi, nullptr, &p.module));
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pci.layout = layout;
        pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, p.module, "main", nullptr};
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &p.pipeline));
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &set_layout;
        check(vkAllocateDescriptorSets(device, &ai, &p.descriptors));
        for (uint32_t binding = 0; binding < 3; ++binding) for (uint32_t slot = 0; slot < 6; ++slot) {
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = p.descriptors; write.dstBinding = binding; write.dstArrayElement = slot;
            write.descriptorCount = 1; write.descriptorType = bindings[binding].descriptorType;
            VkDescriptorImageInfo im{}; VkDescriptorBufferInfo b{};
            if (binding == 1) {
                const auto rid = spec.at("buffers").at(slot >= 3 ? slot - 3 : 0).get<std::string>();
                b = {buffers.at(rid).handle, 0, buffer_sizes.at(rid)}; write.pBufferInfo = &b;
            } else {
                const auto rid = spec.at("images").at(binding == 2 ? 2 : slot < 2 ? slot : 0).get<std::string>();
                im.imageView = images.at(rid).image.view; im.imageLayout = VK_IMAGE_LAYOUT_GENERAL; write.pImageInfo = &im;
            }
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
        passes.push_back(p);
    }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = family;
    VkCommandPool commands; check(vkCreateCommandPool(device, &cpi, nullptr, &commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = commands; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device, &cai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(cmd, &bi));
    for (const auto& entry : images) {
        const auto& tensor = entry.second;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.image = tensor.image.handle;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, tensor.layers}; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkClearColorValue clear{}; for (auto& value : clear.uint32) value = 0xdeadbeef;
        vkCmdClearColorImage(cmd, tensor.image.handle, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &b.subresourceRange);
    }
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, first.layers}; copy.imageExtent = {first.width, first.height, 1};
    vkCmdCopyBufferToImage(cmd, upload.handle, first.image.handle, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    if (resident) {
        std::array<FfxVkXeSSConvolutionPass,12> graph{};
        for (size_t i=0;i<passes.size();++i) {
            graph[i].pipeline=passes[i].pipeline;
            graph[i].descriptors=passes[i].descriptors;
            std::copy(passes[i].push.begin(),passes[i].push.end(),graph[i].constants);
            std::copy(passes[i].groups.begin(),passes[i].groups.end(),graph[i].groups);
        }
        check(ffxVkXeSSRecordConvolutions(cmd,layout,graph.data(),graph.size()));
        barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    }
    for (const auto& p : passes) {
        if (!resident) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &p.descriptors, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, p.push.data());
        vkCmdDispatch(cmd, p.groups[0], p.groups[1], p.groups[2]);
        barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        const auto& target = images.at(p.output);
        copy.imageSubresource.layerCount = target.layers; copy.imageExtent = {target.width, target.height, 1};
        vkCmdCopyImageToBuffer(cmd, target.image.handle, VK_IMAGE_LAYOUT_GENERAL, p.readback.handle, 1, &copy);
    }
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    check(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; check(vkCreateFence(device, &fi, nullptr, &fence));
    check(vkQueueSubmit(queue, 1, &submit, fence)); check(vkWaitForFences(device, 1, &fence, VK_TRUE, 30000000000ull));
    if (resident) {
        std::vector<std::vector<char>> first_results;
        for (const auto& p:passes) {
            const auto* data=static_cast<const char*>(p.readback.mapped);
            first_results.emplace_back(data,data+p.bytes);
        }
        check(vkResetFences(device,1,&fence));
        check(vkQueueSubmit(queue,1,&submit,fence));
        check(vkWaitForFences(device,1,&fence,VK_TRUE,30000000000ull));
        for (size_t i=0;i<passes.size();++i)
            if (std::memcmp(first_results[i].data(),passes[i].readback.mapped,passes[i].bytes))
                throw std::runtime_error("Persistent graph repeat differs");
        std::printf("NATIVE_XESS_RESIDENT_REPEAT passes=12 submissions=2 identical=1\n");
    }
    size_t total_sentinels = 0;
    for (size_t i = 0; i < passes.size(); ++i) {
        const auto& p = passes[i]; const auto* words = static_cast<const uint32_t*>(p.readback.mapped);
        size_t sentinels = 0; uint64_t hash = 14695981039346656037ull;
        for (size_t j = 0; j < p.bytes / 4; ++j) { sentinels += words[j] == 0xdeadbeef; hash = (hash ^ words[j]) * 1099511628211ull; }
        const auto path = std::filesystem::path(argv[3]) / ("pass-" + std::to_string(i + 1) + ".rgba32ui");
        std::ofstream out(path, std::ios::binary); out.write(static_cast<const char*>(p.readback.mapped), p.bytes); out.close();
        if (!out) throw std::runtime_error("Readback write failed");
        std::printf("CONV_PASS index=%zu bytes=%zu sentinel_words=%zu word_hash=%016llx\n", i + 1, p.bytes, sentinels, static_cast<unsigned long long>(hash));
        total_sentinels += sentinels;
    }
    std::printf("NATIVE_XESS_CONVOLUTIONS passes=12 weights=%s reference_match=unverified\n", zero ? "zeroed-control" : "captured");
    vkDestroyFence(device, fence, nullptr); vkDestroyCommandPool(device, commands, nullptr);
    for (const auto& p : passes) { vkDestroyPipeline(device, p.pipeline, nullptr); vkDestroyShaderModule(device, p.module, nullptr); }
    vkDestroyPipelineLayout(device, layout, nullptr); vkDestroyDescriptorPool(device, pool, nullptr); vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    for (const auto& entry : images) { const auto& im = entry.second.image; vkDestroyImageView(device, im.view, nullptr); vkDestroyImage(device, im.handle, nullptr); vkFreeMemory(device, im.memory, nullptr); }
    std::vector<Buffer> allocations{upload};
    for (const auto& entry : buffers) allocations.push_back(entry.second);
    for (const auto& p : passes) allocations.push_back(p.readback);
    for (auto b : allocations) { vkUnmapMemory(device, b.memory); vkDestroyBuffer(device, b.handle, nullptr); vkFreeMemory(device, b.memory, nullptr); }
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    return total_sentinels ? 1 : 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
