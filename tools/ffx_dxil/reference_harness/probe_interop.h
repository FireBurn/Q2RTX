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
