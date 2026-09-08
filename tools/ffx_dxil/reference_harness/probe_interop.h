// SPDX-License-Identifier: MIT
// Minimal declaration of the published ID3D12DXVKInteropDevice ABI.
// Reference: ValveSoftware/Proton proton_11.0,
// wineopenxr/vkd3d-proton-interop.h. Dispatchable Vulkan handles are opaque
// pointers here; this diagnostic does not call Vulkan or own these handles.
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

static bool test_interop_buffer(ID3D12Device* device)
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
        using Fill = void (WINAPI*)(void*, UINT64, UINT64, UINT64, UINT32);
        auto fill = gdpa ? reinterpret_cast<Fill>(gdpa(logical, "vkCmdFillBuffer")) : nullptr;
        ProbePixels memory;
        if (!fill || !memory.buffer(device, 256, false, &buffer)) goto cleanup;
        UINT64 handle = 0, offset = 0;
        void* vkcommands = nullptr;
        if (FAILED(interop->GetVulkanResourceInfo(buffer, &handle, &offset)) || !handle ||
            FAILED(interop->BeginVkCommandBufferInterop(commands, &vkcommands)) || !vkcommands)
            goto cleanup;
        fill(vkcommands, handle, offset, 256, 0x1234abcd);
        if (FAILED(interop->EndVkCommandBufferInterop(commands)) || FAILED(commands->Close()) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            goto cleanup;
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
            matched += static_cast<const UINT32*>(mapped)[i] == 0x1234abcd;
        D3D12_RANGE no_writes{0, 0};
        buffer->Unmap(0, &no_writes);
        success = matched == 64;
        std::printf("FFX_VULKAN_BUFFER_ROUNDTRIP matched=%u expected=64\n", matched);
    }
cleanup:
    if (!success) std::fprintf(stderr, "Vulkan/DX12 buffer round trip failed.\n");
    if (event) CloseHandle(event);
    if (fence) fence->Release();
    if (commands) commands->Release();
    if (allocator) allocator->Release();
    if (queue) queue->Release();
    if (buffer) buffer->Release();
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
    interop->GetVulkanImageLayout(image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &layout);
    const bool valid = SUCCEEDED(handles) && SUCCEEDED(queues) && SUCCEEDED(resources) &&
        instance && physical && logical && vkqueue && vkimage && family != UINT32_MAX && layout > 0;
    std::printf("FFX_INTEROP_RESOURCES available=1 valid=%u queue_family=%u output_layout=%d\n",
        valid ? 1u : 0u, family, layout);
    interop->Release();
    return valid;
}
