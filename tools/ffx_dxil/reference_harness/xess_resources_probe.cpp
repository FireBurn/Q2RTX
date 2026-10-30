// SPDX-License-Identifier: MIT
// Live lifetime test for the caller-owned-device resource boundary.
#include "ffx_vk_xess_resources.hpp"
#include <cstdio>
#include <cstring>

static void check(VkResult result) {
    if(result!=VK_SUCCESS) throw std::runtime_error("Vulkan failure");
}
int main() try {
    VkInstanceCreateInfo ii{};ii.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkInstance instance;check(vkCreateInstance(&ii,nullptr,&instance));
    uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
    if(!count) throw std::runtime_error("No Vulkan device");
    std::vector<VkPhysicalDevice> physical(count);check(vkEnumeratePhysicalDevices(instance,&count,physical.data()));
    uint32_t families=0;vkGetPhysicalDeviceQueueFamilyProperties(physical[0],&families,nullptr);
    std::vector<VkQueueFamilyProperties> queues(families);vkGetPhysicalDeviceQueueFamilyProperties(physical[0],&families,queues.data());
    uint32_t family=0;while(family<families && !(queues[family].queueFlags&VK_QUEUE_COMPUTE_BIT)) ++family;
    if(family==families) throw std::runtime_error("No compute queue");
    float priority=1;VkDeviceQueueCreateInfo qi{};qi.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
    VkDeviceCreateInfo di{};di.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;
    VkDevice device;check(vkCreateDevice(physical[0],&di,nullptr,&device));
    {
        ffxvk::XeSSResources retained(physical[0],device);
        auto kept=retained.createBuffer(256,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memset(kept.mapped,0x5a,256);
        for(unsigned repeat=0;repeat<3;++repeat) {
            ffxvk::XeSSResources transient(physical[0],device);
            transient.createImage(16,16,VK_FORMAT_R32G32B32A32_UINT,true,2);
            transient.createBuffer(256,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            bool rejected=false;
            try {transient.createBuffer(0,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);}
            catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected) throw std::runtime_error("Invalid allocation accepted");
            rejected=false;
            try {transient.createStage({}, {},24);}
            catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected) throw std::runtime_error("Invalid shader accepted");
            rejected=false;
            try {transient.createBindings(ffxvk::XeSSStage{});}
            catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected) throw std::runtime_error("Unowned descriptor layout accepted");
            VkSamplerCreateInfo sampler{};sampler.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler.magFilter=sampler.minFilter=VK_FILTER_NEAREST;
            sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            transient.createSampler(sampler);
            transient.clear();transient.clear();
            transient.createBuffer(128,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        }
        for(unsigned i=0;i<256;++i)
            if(static_cast<unsigned char*>(kept.mapped)[i]!=0x5a) throw std::runtime_error("Independent resource corrupted");
    }
    check(vkDeviceWaitIdle(device));
    vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
    std::puts("XESS_RESOURCES independent_owners=pass clear_reuse=pass device_retained=pass invalid_shader=pass sampler_lifetime=pass unowned_bindings=pass");return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
