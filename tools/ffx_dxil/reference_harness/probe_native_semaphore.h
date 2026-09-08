// SPDX-License-Identifier: MIT
// Consumes fd on all paths. No queue submission or shared image import yet.
#include <vulkan/vulkan.h>
static int probe_native_semaphore(int fd, const uint8_t uuid[16])
{
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE, devices[16];
    uint32_t count = 16, family = UINT32_MAX;
    int valid = 0;
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "FSR native semaphore probe", .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    if ((result = vkCreateInstance(&ci, NULL, &instance))) goto done;
    if ((result = vkEnumeratePhysicalDevices(instance, &count, devices))) goto done;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
        vkGetPhysicalDeviceProperties2(devices[i], &props);
        if (!memcmp(uuid, id.deviceUUID, 16)) physical = devices[i];
    }
    if (!physical) goto done;
    VkQueueFamilyProperties queues[64];
    count = 64;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
    for (uint32_t i = 0; i < count; ++i)
        if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    if (family == UINT32_MAX) goto done;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &timeline};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    if (!timeline.timelineSemaphore) goto done;
    float priority = 1;
    VkDeviceQueueCreateInfo queue = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    const char* extension = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
    VkDeviceCreateInfo dc = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &timeline,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = &extension};
    if ((result = vkCreateDevice(physical, &dc, NULL, &device))) goto done;
    VkSemaphoreTypeCreateInfo type = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo sc = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type};
    if ((result = vkCreateSemaphore(device, &sc, NULL, &semaphore))) goto done;
    PFN_vkImportSemaphoreFdKHR import = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkImportSemaphoreFdKHR");
    if (!import) goto done;
    VkImportSemaphoreFdInfoKHR info = {.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
        .semaphore = semaphore, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, .fd = fd};
    if ((result = import(device, &info))) goto done;
    fd = -1; // Successful import transfers descriptor ownership to Vulkan.
    uint64_t target = 1, observed = 0;
    VkSemaphoreWaitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1, .pSemaphores = &semaphore, .pValues = &target};
    if ((result = vkWaitSemaphores(device, &wait, UINT64_C(10000000000)))) goto done;
    if ((result = vkGetSemaphoreCounterValue(device, semaphore, &observed))) goto done;
    valid = observed >= target;
    printf("FFX_NATIVE_SEMAPHORE observed=%llu target=1 valid=%d\n", (unsigned long long)observed, valid);
done:
    if (!valid) fprintf(stderr, "Native semaphore import/wait failed: VkResult=%d\n", result);
    if (fd >= 0) close(fd);
    if (semaphore) vkDestroySemaphore(device, semaphore, NULL);
    if (device) vkDestroyDevice(device, NULL);
    if (instance) vkDestroyInstance(instance, NULL);
    return valid;
}
