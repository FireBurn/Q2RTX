// SPDX-License-Identifier: MIT
// Minimal declaration of the published ID3D12DXVKInteropDevice ABI.
// Reference: ValveSoftware/Proton proton_11.0,
// wineopenxr/vkd3d-proton-interop.h. Dispatchable Vulkan handles are opaque
// pointers in the COM declaration; Vulkan calls use the official header types.
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include "probe_unix_loader.h"
struct ProbeInteropDevice : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE GetDXGIAdapter(REFIID, void**) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetInstanceExtensions(UINT*, const char**) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceExtensions(UINT*, const char**) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceFeatures(const void**) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHandles(void**, void**, void**) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanQueueInfo(ID3D12CommandQueue*, void**, UINT*) = 0;
    virtual void STDMETHODCALLTYPE GetVulkanImageLayout(ID3D12Resource*, D3D12_RESOURCE_STATES, int*) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo(ID3D12Resource*, UINT64*, UINT64*) = 0;
    virtual HRESULT STDMETHODCALLTYPE LockCommandQueue(ID3D12CommandQueue*) = 0;
    virtual HRESULT STDMETHODCALLTYPE UnlockCommandQueue(ID3D12CommandQueue*) = 0;
};

struct ProbeInteropDevice1 : ProbeInteropDevice {
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo1(ID3D12Resource*, UINT64*, UINT64*, int*) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandQueue(const D3D12_COMMAND_QUEUE_DESC*, UINT32, ID3D12CommandQueue**) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandAllocator(D3D12_COMMAND_LIST_TYPE, UINT32, ID3D12CommandAllocator**) = 0;
    virtual HRESULT STDMETHODCALLTYPE BeginVkCommandBufferInterop(ID3D12CommandList*, void**) = 0;
    virtual HRESULT STDMETHODCALLTYPE EndVkCommandBufferInterop(ID3D12CommandList*) = 0;
};

struct ProbeInteropDevice3 : ProbeInteropDevice1 {
    virtual HRESULT STDMETHODCALLTYPE LockVulkanQueue(ID3D12CommandQueue*) = 0;
    virtual HRESULT STDMETHODCALLTYPE UnlockVulkanQueue(ID3D12CommandQueue*) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHeapInfo(ID3D12Heap*, UINT64*, UINT64*, UINT32*) = 0;
};

static bool test_owned_export_allocation(ProbeInteropDevice* interop, UINT64 size, UINT32 type)
{
    void *instance = nullptr, *physical = nullptr, *logical = nullptr;
    if (FAILED(interop->GetVulkanHandles(&instance, &physical, &logical))) return false;
    HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
    if (!vulkan) return false;
    auto gipa = probe_ntdll_proc<PFN_vkGetInstanceProcAddr>(vulkan, "vkGetInstanceProcAddr");
    auto gdpa = gipa ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        gipa(static_cast<VkInstance>(instance), "vkGetDeviceProcAddr")) : nullptr;
    const auto device = static_cast<VkDevice>(logical);
    auto allocate = gdpa ? reinterpret_cast<PFN_vkAllocateMemory>(gdpa(device, "vkAllocateMemory")) : nullptr;
    auto release = gdpa ? reinterpret_cast<PFN_vkFreeMemory>(gdpa(device, "vkFreeMemory")) : nullptr;
    auto export_memory = gdpa ? reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        gdpa(device, "vkGetMemoryWin32HandleKHR")) : nullptr;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    HANDLE handle = nullptr;
    VkResult result = VK_ERROR_EXTENSION_NOT_PRESENT;
    if (allocate && release && export_memory) {
        VkExportMemoryAllocateInfo external{};
        external.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkMemoryAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.pNext = &external;
        info.allocationSize = size;
        info.memoryTypeIndex = type;
        result = allocate(device, &info, nullptr, &memory);
        if (result == VK_SUCCESS) {
            VkMemoryGetWin32HandleInfoKHR get{};
            get.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
            get.memory = memory;
            get.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
            result = export_memory(device, &get, &handle);
        }
    }
    const bool valid = result == VK_SUCCESS && handle;
    std::printf("FFX_OWNED_VULKAN_EXPORT valid=%u result=%d size=%llu memory_type=%u\n",
        valid ? 1u : 0u, result, static_cast<unsigned long long>(size), type);
    if (handle) CloseHandle(handle);
    if (memory) release(device, memory, nullptr);
    FreeLibrary(vulkan);
    return valid;
}

static bool inspect_shared_heap(ID3D12Device* device)
{
    const GUID iid = {0x22a70184,0xa6a4,0x4c24,{0xbf,0x97,0x7d,0x6d,0xf9,0xf1,0x2d,0x8a}};
    ProbeInteropDevice3* interop = nullptr;
    if (FAILED(device->QueryInterface(iid, reinterpret_cast<void**>(&interop)))) {
        std::printf("FFX_SHARED_HEAP available=0\n");
        return true;
    }
    ID3D12Heap* heap = nullptr;
    ID3D12Resource* placed = nullptr;
    HANDLE exported = nullptr;
    D3D12_RESOURCE_DESC image{};
    image.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    image.Width = image.Height = 8;
    image.DepthOrArraySize = image.MipLevels = image.SampleDesc.Count = 1;
    image.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    image.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto allocation = device->GetResourceAllocationInfo(0, 1, &image);
    D3D12_HEAP_DESC desc{};
    desc.SizeInBytes = allocation.SizeInBytes;
    desc.Alignment = allocation.Alignment;
    desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    desc.Properties.CreationNodeMask = desc.Properties.VisibleNodeMask = 1;
    desc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    HRESULT hr = device->CreateHeap(&desc, IID_PPV_ARGS(&heap));
    UINT64 memory = 0, offset = 0;
    UINT32 type = UINT32_MAX;
    if (SUCCEEDED(hr)) hr = interop->GetVulkanHeapInfo(heap, &memory, &offset, &type);
    const HRESULT export_hr = SUCCEEDED(hr) ? device->CreateSharedHandle(heap,
        nullptr, GENERIC_ALL, nullptr, &exported) : hr;
    std::printf("FFX_SHARED_HEAP_EXPORT status=0x%08lx\n", static_cast<unsigned long>(export_hr));
    // Heap export is an optional capability, separate from placement support.
    UINT64 vkimage = 0, image_offset = 0;
    if (SUCCEEDED(hr)) hr = device->CreatePlacedResource(heap, 0, &image,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&placed));
    if (SUCCEEDED(hr)) hr = interop->GetVulkanResourceInfo(placed, &vkimage, &image_offset);
    const bool valid = SUCCEEDED(hr) && memory && vkimage && type != UINT32_MAX;
    const bool owned_export = valid && test_owned_export_allocation(interop, desc.SizeInBytes, type);
    std::printf("FFX_SHARED_HEAP available=1 valid=%u status=0x%08lx size=%llu offset=%llu memory_type=%u\n",
        valid ? 1u : 0u, static_cast<unsigned long>(hr),
        static_cast<unsigned long long>(desc.SizeInBytes), static_cast<unsigned long long>(offset), type);
    std::printf("FFX_SHARED_HEAP_IMAGE valid=%u image_offset=%llu\n",
        valid ? 1u : 0u, static_cast<unsigned long long>(image_offset));
    if (exported) CloseHandle(exported);
    if (placed) placed->Release();
    if (heap) heap->Release();
    interop->Release();
    return valid && owned_export;
}

static bool test_interop_buffer(ID3D12Device* device, bool texture = false, bool shared = false)
{
    const GUID iid = {0x90ecf26e,0xb212,0x43f5,{0xb6,0x2a,0x82,0x5a,0xd7,0xb1,0x38,0x5e}};
    ProbeInteropDevice1* interop = nullptr;
    if (FAILED(device->QueryInterface(iid, reinterpret_cast<void**>(&interop))))
        return true; // Native Windows has no vkd3d extension.
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* commands = nullptr;
    ID3D12Fence* fence = nullptr;
    ID3D12Resource* buffer = nullptr;
    ID3D12Resource* image = nullptr;
    HANDLE event = nullptr;
    HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
    bool success = false;
    void *instance = nullptr, *physical = nullptr, *logical = nullptr, *vkqueue = nullptr;
    UINT family = UINT32_MAX;
    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!vulkan || FAILED(interop->GetVulkanHandles(&instance, &physical, &logical)) ||
        FAILED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue))) ||
        FAILED(interop->GetVulkanQueueInfo(queue, &vkqueue, &family)) ||
        FAILED(interop->CreateInteropCommandAllocator(qdesc.Type, family, &allocator)) ||
        FAILED(device->CreateCommandList(0, qdesc.Type, allocator, nullptr, IID_PPV_ARGS(&commands))))
        goto cleanup;
    {
        // Obtain device entry points from Wine's Vulkan dispatch table.
        using Proc = void (WINAPI*)();
        using GetProc = Proc (WINAPI*)(void*, const char*);
        const FARPROC entry = GetProcAddress(vulkan, "vkGetInstanceProcAddr");
        GetProc gipa = nullptr;
        static_assert(sizeof(gipa) == sizeof(entry), "Win64 function pointer size");
        std::memcpy(&gipa, &entry, sizeof(gipa));
        auto gdpa = gipa ? reinterpret_cast<GetProc>(gipa(instance, "vkGetDeviceProcAddr")) : nullptr;
        auto properties = gipa ? reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            gipa(instance, "vkGetPhysicalDeviceProperties2")) : nullptr;
        VkPhysicalDeviceIDProperties identity{};
        identity.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 props{};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &identity;
        if (!properties) goto cleanup;
        properties(static_cast<VkPhysicalDevice>(physical), &props);
        using Fill = void (WINAPI*)(void*, UINT64, UINT64, UINT64, UINT32);
        auto fill = gdpa ? reinterpret_cast<Fill>(gdpa(logical, "vkCmdFillBuffer")) : nullptr;
        ProbePixels memory;
        if (!fill || !memory.buffer(device, 256, false, &buffer)) goto cleanup;
        UINT64 handle = 0, offset = 0;
        void* vkcommands = nullptr;
        if (FAILED(interop->GetVulkanResourceInfo(buffer, &handle, &offset)) || !handle ||
            FAILED(interop->BeginVkCommandBufferInterop(commands, &vkcommands)) || !vkcommands)
            goto cleanup;
        if (!texture) {
            fill(vkcommands, handle, offset, 256, 0x1234abcd);
        } else {
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            heap.CreationNodeMask = heap.VisibleNodeMask = 1;
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = desc.Height = 8;
            desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            if (FAILED(device->CreateCommittedResource(&heap,
                shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&image)))) goto cleanup;
            if (shared) {
                HANDLE exported = nullptr;
                ID3D12Resource* reopened = nullptr;
                HRESULT hr = device->CreateSharedHandle(image, nullptr, GENERIC_ALL, nullptr, &exported);
                if (SUCCEEDED(hr) && !inspect_unix_shared_handle(exported, false, identity.deviceUUID)) hr = E_FAIL;
                if (SUCCEEDED(hr)) hr = device->OpenSharedHandle(exported, IID_PPV_ARGS(&reopened));
                if (exported) CloseHandle(exported);
                if (FAILED(hr)) goto cleanup;
                image->Release();
                image = reopened;
            }
            UINT64 image_handle = 0, image_offset = 0;
            if (FAILED(interop->GetVulkanResourceInfo(image, &image_handle, &image_offset)) || !image_handle)
                goto cleanup;
            auto barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(gdpa(logical, "vkCmdPipelineBarrier"));
            auto clear = reinterpret_cast<PFN_vkCmdClearColorImage>(gdpa(logical, "vkCmdClearColorImage"));
            auto copy = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(gdpa(logical, "vkCmdCopyImageToBuffer"));
            if (!barrier || !clear || !copy) goto cleanup;
            const auto cb = static_cast<VkCommandBuffer>(vkcommands);
            const auto img = reinterpret_cast<VkImage>(image_handle);
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
            VkClearColorValue color{};
            color.float32[0] = color.float32[3] = 1.0f;
            clear(cb, img, b.newLayout, &color, 1, &b.subresourceRange);
            b.oldLayout = b.newLayout;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
            VkBufferImageCopy region{};
            region.bufferOffset = offset;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {8, 8, 1};
            copy(cb, img, b.newLayout, reinterpret_cast<VkBuffer>(handle), 1, &region);
        }
        if (FAILED(interop->EndVkCommandBufferInterop(commands)) || FAILED(commands->Close()) ||
            FAILED(device->CreateFence(0, shared ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(&fence))))
            goto cleanup;
        if (shared) {
            HANDLE exported = nullptr;
            ID3D12Fence* reopened = nullptr;
            HRESULT hr = device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr, &exported);
            if (SUCCEEDED(hr) && !inspect_unix_shared_handle(exported, true, identity.deviceUUID)) hr = E_FAIL;
            if (SUCCEEDED(hr)) hr = device->OpenSharedHandle(exported, IID_PPV_ARGS(&reopened));
            if (exported) CloseHandle(exported);
            if (FAILED(hr)) goto cleanup;
            fence->Release();
            fence = reopened;
        }
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) goto cleanup;
        ID3D12CommandList* lists[] = {commands};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence, 1)) || FAILED(fence->SetEventOnCompletion(1, event)) ||
            WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 ||
            FAILED(device->GetDeviceRemovedReason()) || fence->GetCompletedValue() != 1)
            goto cleanup;
        void* mapped = nullptr;
        D3D12_RANGE range{0, 256};
        if (FAILED(buffer->Map(0, &range, &mapped))) goto cleanup;
        unsigned matched = 0;
        for (unsigned i = 0; i < 64; ++i)
            matched += static_cast<const UINT32*>(mapped)[i] == (texture ? 0xff0000ffu : 0x1234abcdu);
        D3D12_RANGE no_writes{0, 0};
        buffer->Unmap(0, &no_writes);
        success = matched == 64;
        std::printf("FFX_VULKAN_%s_ROUNDTRIP matched=%u expected=64\n",
            shared ? "SHARED_TEXTURE" : texture ? "TEXTURE" : "BUFFER", matched);
    }
cleanup:
    if (!success) std::fprintf(stderr, "Vulkan/DX12 buffer round trip failed.\n");
    if (event) CloseHandle(event);
    if (fence) fence->Release();
    if (commands) commands->Release();
    if (allocator) allocator->Release();
    if (queue) queue->Release();
    if (buffer) buffer->Release();
    if (image) image->Release();
    if (vulkan) FreeLibrary(vulkan);
    interop->Release();
    return success;
}

static bool inspect_interop_resources(ID3D12Device* device,
    ID3D12CommandQueue* queue, ID3D12Resource* image)
{
    const GUID iid = {0x39da4e09, 0xbd1c, 0x4198,
        {0x9f,0xae,0x86,0xbb,0xe3,0xbe,0x41,0xfd}};
    ProbeInteropDevice* interop = nullptr;
    if (FAILED(device->QueryInterface(iid, reinterpret_cast<void**>(&interop)))) {
        std::printf("FFX_INTEROP_RESOURCES available=0\n");
        return true; // Optional diagnostic: ordinary Windows has no vkd3d.
    }
    void* instance = nullptr;
    void* physical = nullptr;
    void* logical = nullptr;
    void* vkqueue = nullptr;
    UINT family = UINT32_MAX;
    UINT64 vkimage = 0, offset = UINT64_MAX;
    int layout = -1;
    const HRESULT handles = interop->GetVulkanHandles(&instance, &physical, &logical);
    const HRESULT queues = interop->GetVulkanQueueInfo(queue, &vkqueue, &family);
    const HRESULT resources = interop->GetVulkanResourceInfo(image, &vkimage, &offset);
    UINT extension_count = 0;
    if (SUCCEEDED(interop->GetDeviceExtensions(&extension_count, nullptr)) && extension_count < 4096) {
        std::vector<const char*> extensions(extension_count);
        if (SUCCEEDED(interop->GetDeviceExtensions(&extension_count, extensions.data()))) {
            bool memory_fd = false, semaphore_fd = false, memory_win32 = false;
            for (const char* extension : extensions) {
                if (!extension) continue;
                memory_fd |= std::strcmp(extension, "VK_KHR_external_memory_fd") == 0;
                semaphore_fd |= std::strcmp(extension, "VK_KHR_external_semaphore_fd") == 0;
                memory_win32 |= std::strcmp(extension, "VK_KHR_external_memory_win32") == 0;
            }
            std::printf("FFX_INTEROP_EXTERNAL enabled_memory_fd=%u enabled_semaphore_fd=%u enabled_memory_win32=%u\n",
                memory_fd ? 1u : 0u, semaphore_fd ? 1u : 0u, memory_win32 ? 1u : 0u);
        }
    }
    interop->GetVulkanImageLayout(image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &layout);
    const bool valid = SUCCEEDED(handles) && SUCCEEDED(queues) && SUCCEEDED(resources) &&
        instance && physical && logical && vkqueue && vkimage && family != UINT32_MAX && layout > 0;
    std::printf("FFX_INTEROP_RESOURCES available=1 valid=%u queue_family=%u output_layout=%d\n",
        valid ? 1u : 0u, family, layout);
    interop->Release();
    return valid;
}
