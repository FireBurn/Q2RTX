// SPDX-License-Identifier: MIT
//
// Minimal, capture-oriented FidelityFX API probe. It deliberately does not
// include AMD DLLs, shader blobs, model data, or a renderer backend. It
// creates a DX12 device, loads the SDK's signed loader from a caller-supplied
// path, enumerates upscaler providers, and can create one FSR 4.1.1 API
// context. Run it under Wine/Proton with VKD3D_SHADER_DUMP_PATH set to observe
// only the work actually selected by AMD's provider.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <cinttypes>
#include <cstring>
#include <cwchar>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ffx_api.h"
#include "ffx_api_dx12.h"
#if defined(__has_include)
#if __has_include("ffx_denoiser.h")
#include "ffx_denoiser.h"
#define FFX_PROVIDER_PROBE_HAS_DENOISER 1
#endif
#endif
#ifndef FFX_PROVIDER_PROBE_HAS_DENOISER
#define FFX_PROVIDER_PROBE_HAS_DENOISER 0
#endif
#include "ffx_framegeneration.h"
#include "ffx_upscale.h"

namespace {

struct FfxFunctions {
    PfnFfxCreateContext createContext = nullptr;
    PfnFfxDestroyContext destroyContext = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
};

/* Keep a deliberately small, machine-readable record of a create operation.
 * This is more useful than inferring feature support from the requested API
 * version: the loader is permitted to select an older analytical provider. */
struct ProviderSelection {
    bool attempted = false;
    uint32_t create_result = UINT32_MAX;
    uint32_t query_result = UINT32_MAX;
    uint64_t version_id = 0;
    char version_name[128] = {};
};

ID3D12Device* g_provider_allocation_device = nullptr;

void print_hr(const char* operation, HRESULT hr)
{
    std::fprintf(stderr, "%s failed: HRESULT 0x%08" PRIx32 "\n", operation,
        static_cast<uint32_t>(hr));
}

void CALLBACK ffx_message(uint32_t type, const wchar_t* message)
{
    const char* severity = type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning";
    ::fwprintf(stderr, L"FFX provider %hs: %ls\n", severity,
        message ? message : L"(null)");
}

ffxReturnCode_t provider_resource_allocate(uint32_t effect_id,
    D3D12_RESOURCE_STATES initial_state,
    const D3D12_HEAP_PROPERTIES* heap_properties,
    const D3D12_RESOURCE_DESC* description,
    const FfxApiResourceDescription*,
    const D3D12_CLEAR_VALUE* clear_value,
    ID3D12Resource** resource)
{
    if (!g_provider_allocation_device || !heap_properties || !description || !resource)
        return FFX_API_RETURN_ERROR_PARAMETER;
    const HRESULT hr = g_provider_allocation_device->CreateCommittedResource(
        heap_properties, D3D12_HEAP_FLAG_NONE, description, initial_state,
        clear_value, IID_PPV_ARGS(resource));
    if (FAILED(hr)) {
        print_hr("FFX provider resource allocation", hr);
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    std::printf("provider resource: effect=0x%08" PRIx32
        " %llux%u format=%u mips=%u flags=0x%x state=0x%x\n",
        effect_id, static_cast<unsigned long long>(description->Width),
        description->Height, static_cast<unsigned>(description->Format),
        description->MipLevels, static_cast<unsigned>(description->Flags),
        static_cast<unsigned>(initial_state));
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t provider_resource_deallocate(uint32_t effect_id,
    ID3D12Resource* resource)
{
    std::printf("provider resource release: effect=0x%08" PRIx32 "\n", effect_id);
    if (resource)
        resource->Release();
    return FFX_API_RETURN_OK;
}

bool create_device(ID3D12Device** out_device)
{
    IDXGIFactory6* factory = nullptr;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        print_hr("CreateDXGIFactory2", hr);
        return false;
    }

    bool success = false;
    for (UINT index = 0; ; ++index) {
        IDXGIAdapter1* adapter = nullptr;
        hr = factory->EnumAdapters1(index, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND)
            break;
        if (FAILED(hr)) {
            print_hr("IDXGIFactory6::EnumAdapters1", hr);
            break;
        }

        DXGI_ADAPTER_DESC1 description = {};
        adapter->GetDesc1(&description);
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            adapter->Release();
            continue;
        }

        ID3D12Device* device = nullptr;
        hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0,
            IID_PPV_ARGS(&device));
        ::fwprintf(stdout, L"adapter %u: %ls (vendor %04x, device %04x): %hs\n",
            index, description.Description, description.VendorId,
            description.DeviceId, SUCCEEDED(hr) ? "usable" : "unavailable");
        adapter->Release();
        if (SUCCEEDED(hr)) {
            *out_device = device;
            success = true;
            break;
        }
    }
    factory->Release();
    if (!success)
        std::fprintf(stderr, "No hardware DX12 adapter could be created.\n");
    return success;
}

template <typename Function>
Function load_function(HMODULE module, const char* name)
{
    const FARPROC proc = GetProcAddress(module, name);
    static_assert(sizeof(proc) == sizeof(Function),
        "Windows function-pointer representation must match FARPROC");
    Function function = nullptr;
    std::memcpy(&function, &proc, sizeof(function));
    return function;
}

bool load_functions(const wchar_t* loader_path, HMODULE* out_module,
    FfxFunctions* out_functions)
{
    HMODULE module = LoadLibraryW(loader_path);
    if (!module) {
        ::fwprintf(stderr, L"LoadLibraryW failed for %ls (error %lu)\n",
            loader_path, GetLastError());
        return false;
    }
    FfxFunctions functions = {
        load_function<PfnFfxCreateContext>(module, "ffxCreateContext"),
        load_function<PfnFfxDestroyContext>(module, "ffxDestroyContext"),
        load_function<PfnFfxConfigure>(module, "ffxConfigure"),
        load_function<PfnFfxQuery>(module, "ffxQuery"),
        load_function<PfnFfxDispatch>(module, "ffxDispatch"),
    };
    if (!functions.createContext || !functions.destroyContext || !functions.configure ||
        !functions.query || !functions.dispatch) {
        std::fprintf(stderr, "The supplied loader is missing an FFX API export.\n");
        FreeLibrary(module);
        return false;
    }
    *out_module = module;
    *out_functions = functions;
    return true;
}

bool enumerate_effect_versions(const FfxFunctions& functions, ID3D12Device* device,
    uint32_t create_desc_type, const char* effect_name,
    std::vector<uint64_t>* out_ids)
{
    ffxQueryDescGetVersions query = {};
    query.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    query.createDescType = create_desc_type;
    query.device = device;
    uint64_t count = 0;
    query.outputCount = &count;
    ffxReturnCode_t result = functions.query(nullptr, &query.header);
    if (result != FFX_API_RETURN_OK) {
        std::fprintf(stderr, "ffxQuery(GetVersions/count, %s) returned %u\n",
            effect_name, result);
        return false;
    }
    if (!count) {
        std::fprintf(stderr, "The loader reported no %s providers.\n", effect_name);
        return false;
    }

    out_ids->assign(static_cast<size_t>(count), 0);
    std::vector<const char*> names(static_cast<size_t>(count), nullptr);
    query.outputCount = &count;
    query.versionIds = out_ids->data();
    query.versionNames = names.data();
    result = functions.query(nullptr, &query.header);
    if (result != FFX_API_RETURN_OK) {
        std::fprintf(stderr, "ffxQuery(GetVersions/list, %s) returned %u\n",
            effect_name, result);
        return false;
    }
    out_ids->resize(static_cast<size_t>(count));
    std::printf("%s providers (%" PRIu64 "):\n", effect_name, count);
    for (uint64_t i = 0; i < count; ++i) {
        std::printf("  [%" PRIu64 "] id=0x%016" PRIx64 " name=%s\n", i,
            (*out_ids)[static_cast<size_t>(i)],
            names[static_cast<size_t>(i)] ? names[static_cast<size_t>(i)] : "(unnamed)");
    }
    return true;
}

struct DispatchResources {
    ID3D12Resource* color = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* motion_vectors = nullptr;
    ID3D12Resource* reactive = nullptr;
    ID3D12Resource* composition = nullptr;
    ID3D12Resource* output = nullptr;

    void release()
    {
        if (output) output->Release();
        if (composition) composition->Release();
        if (reactive) reactive->Release();
        if (motion_vectors) motion_vectors->Release();
        if (depth) depth->Release();
        if (color) color->Release();
        *this = {};
    }
};

bool create_texture(ID3D12Device* device, uint32_t width, uint32_t height,
    DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initial_state, ID3D12Resource** output)
{
    const D3D12_HEAP_PROPERTIES heap_properties = {
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description = {
        D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        0,
        static_cast<UINT64>(width),
        height,
        1,
        1,
        format,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        flags,
    };
    const HRESULT hr = device->CreateCommittedResource(&heap_properties,
        D3D12_HEAP_FLAG_NONE, &description, initial_state, nullptr,
        IID_PPV_ARGS(output));
    if (FAILED(hr))
        print_hr("ID3D12Device::CreateCommittedResource", hr);
    return SUCCEEDED(hr);
}

bool dispatch_once(const FfxFunctions& functions, ffxContext* context,
    ID3D12Device* device)
{
    constexpr uint32_t render_width = 640;
    constexpr uint32_t render_height = 360;
    constexpr uint32_t output_width = 1280;
    constexpr uint32_t output_height = 720;
    DispatchResources resources;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* command_list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE fence_event = nullptr;
    bool success = false;

    const D3D12_RESOURCE_STATES read_state =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (!create_texture(device, render_width, render_height,
            DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE,
            read_state, &resources.color) ||
        !create_texture(device, render_width, render_height,
            DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE,
            read_state, &resources.depth) ||
        !create_texture(device, render_width, render_height,
            DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE,
            read_state, &resources.motion_vectors) ||
        !create_texture(device, render_width, render_height,
            DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_NONE,
            read_state, &resources.reactive) ||
        !create_texture(device, render_width, render_height,
            DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_NONE,
            read_state, &resources.composition) ||
        !create_texture(device, output_width, output_height,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &resources.output))
        goto cleanup;

    {
        const D3D12_COMMAND_QUEUE_DESC queue_description = {
            D3D12_COMMAND_LIST_TYPE_DIRECT, 0,
            D3D12_COMMAND_QUEUE_FLAG_NONE, 0,
        };
        HRESULT hr = device->CreateCommandQueue(&queue_description,
            IID_PPV_ARGS(&queue));
        if (FAILED(hr)) {
            print_hr("ID3D12Device::CreateCommandQueue", hr);
            goto cleanup;
        }
        hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator));
        if (FAILED(hr)) {
            print_hr("ID3D12Device::CreateCommandAllocator", hr);
            goto cleanup;
        }
        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator, nullptr, IID_PPV_ARGS(&command_list));
        if (FAILED(hr)) {
            print_hr("ID3D12Device::CreateCommandList", hr);
            goto cleanup;
        }
    }

    {
        ffxDispatchDescUpscale dispatch = {};
        dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dispatch.commandList = command_list;
        dispatch.color = ffxApiGetResourceDX12(resources.color,
            FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.depth = ffxApiGetResourceDX12(resources.depth,
            FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.motionVectors = ffxApiGetResourceDX12(resources.motion_vectors,
            FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.reactive = ffxApiGetResourceDX12(resources.reactive,
            FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.transparencyAndComposition =
            ffxApiGetResourceDX12(resources.composition,
                FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.output = ffxApiGetResourceDX12(resources.output,
            FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch.motionVectorScale = {static_cast<float>(render_width),
            static_cast<float>(render_height)};
        dispatch.renderSize = {render_width, render_height};
        dispatch.upscaleSize = {output_width, output_height};
        dispatch.frameTimeDelta = 16.667f;
        dispatch.preExposure = 1.0f;
        dispatch.reset = true;
        dispatch.cameraNear = 0.1f;
        dispatch.cameraFar = 1000.0f;
        dispatch.cameraFovAngleVertical = 1.0f;
        dispatch.viewSpaceToMetersFactor = 1.0f;
        const ffxReturnCode_t result = functions.dispatch(context, &dispatch.header);
        std::printf("ffxDispatch(640x360 -> 1280x720) returned %u\n", result);
        if (result != FFX_API_RETURN_OK)
            goto cleanup;
    }

    {
        HRESULT hr = command_list->Close();
        if (FAILED(hr)) {
            print_hr("ID3D12GraphicsCommandList::Close", hr);
            goto cleanup;
        }
        ID3D12CommandList* command_lists[] = {command_list};
        queue->ExecuteCommandLists(1, command_lists);
        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(hr)) {
            print_hr("ID3D12Device::CreateFence", hr);
            goto cleanup;
        }
        hr = queue->Signal(fence, 1);
        if (FAILED(hr)) {
            print_hr("ID3D12CommandQueue::Signal", hr);
            goto cleanup;
        }
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event) {
            std::fprintf(stderr, "CreateEventW failed: %lu\n", GetLastError());
            goto cleanup;
        }
        hr = fence->SetEventOnCompletion(1, fence_event);
        if (FAILED(hr) || WaitForSingleObject(fence_event, 30000) != WAIT_OBJECT_0) {
            if (FAILED(hr))
                print_hr("ID3D12Fence::SetEventOnCompletion", hr);
            else
                std::fprintf(stderr, "Timed out waiting for FFX dispatch.\n");
            goto cleanup;
        }
    }
    success = true;

cleanup:
    if (fence_event)
        CloseHandle(fence_event);
    if (fence)
        fence->Release();
    if (command_list)
        command_list->Release();
    if (allocator)
        allocator->Release();
    if (queue)
        queue->Release();
    resources.release();
    return success;
}

bool create_fsr411_context(const FfxFunctions& functions, ID3D12Device* device,
    uint64_t version_id, bool dispatch, ProviderSelection* selection)
{
    ffxCreateContextDescUpscale create = {};
    ffxCreateBackendDX12Desc backend = {};
    ffxCreateBackendDX12AllocationCallbacksDesc allocation_callbacks = {};
    ffxCreateContextDescUpscaleVersion version = {};
    ffxOverrideVersion override_version = {};
    ffxContext context = nullptr;

    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &backend.header;
    /* The synthetic dispatch intentionally has no application exposure image.
     * Request the provider's documented internal auto-exposure path so the
     * capture records a clean valid dispatch instead of an avoidable warning. */
    create.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
        FFX_UPSCALE_ENABLE_AUTO_EXPOSURE |
        FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    create.maxRenderSize = {640, 360};
    create.maxUpscaleSize = {1280, 720};
    create.fpMessage = ffx_message;

    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.header.pNext = &version.header;
    backend.device = device;

    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    version.header.pNext = &allocation_callbacks.header;
    version.version = FFX_UPSCALER_VERSION;
    allocation_callbacks.header.type =
        FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12_ALLOCATION_CALLBACKS;
    allocation_callbacks.header.pNext = nullptr;
    allocation_callbacks.pfnFfxResourceAllocator = provider_resource_allocate;
    allocation_callbacks.pfnFfxResourceDeallocator = provider_resource_deallocate;
    if (version_id) {
        override_version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        override_version.header.pNext = nullptr;
        override_version.versionId = version_id;
        override_version.header.pNext = &allocation_callbacks.header;
        version.header.pNext = &override_version.header;
    }

    if (selection)
        selection->attempted = true;
    g_provider_allocation_device = device;
    const ffxReturnCode_t result = functions.createContext(
        &context, &create.header, nullptr);
    if (selection)
        selection->create_result = result;
    std::printf("ffxCreateContext(FSR API %u.%u.%u%s) returned %u\n",
        FFX_UPSCALER_VERSION_MAJOR, FFX_UPSCALER_VERSION_MINOR,
        FFX_UPSCALER_VERSION_PATCH, version_id ? ", explicit provider" : "",
        result);
    if (result != FFX_API_RETURN_OK) {
        g_provider_allocation_device = nullptr;
        return false;
    }

    ffxQueryGetProviderVersion provider_version = {};
    provider_version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    const ffxReturnCode_t query_result = functions.query(&context,
        &provider_version.header);
    if (selection) {
        selection->query_result = query_result;
        selection->version_id = provider_version.versionId;
        std::snprintf(selection->version_name, sizeof(selection->version_name), "%s",
            provider_version.versionName ? provider_version.versionName : "(unnamed)");
    }
    std::printf("ffxQuery(GetProviderVersion) returned %u: id=0x%016" PRIx64
        " name=%s\n", query_result, provider_version.versionId,
        provider_version.versionName ? provider_version.versionName : "(unnamed)");

    ffxQueryDescUpscaleGetResourceRequirements resource_requirements = {};
    resource_requirements.header.type =
        FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS;
    const ffxReturnCode_t requirements_result = functions.query(&context,
        &resource_requirements.header);
    std::printf("ffxQuery(GetResourceRequirements) returned %u: "
        "required=0x%016" PRIx64 " optional=0x%016" PRIx64 "\n",
        requirements_result, resource_requirements.required_resources,
        resource_requirements.optional_resources);

    const bool dispatch_success = !dispatch || dispatch_once(functions, &context, device);
    const ffxReturnCode_t destroy_result = functions.destroyContext(&context, nullptr);
    g_provider_allocation_device = nullptr;
    std::printf("ffxDestroyContext returned %u\n", destroy_result);
    return dispatch_success && destroy_result == FFX_API_RETURN_OK;
}

bool report_selected_provider(const FfxFunctions& functions, ffxContext* context,
    const char* effect_name, ProviderSelection* selection)
{
    ffxQueryGetProviderVersion provider_version = {};
    provider_version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    const ffxReturnCode_t query_result = functions.query(context,
        &provider_version.header);
    if (selection) {
        selection->query_result = query_result;
        selection->version_id = provider_version.versionId;
        std::snprintf(selection->version_name, sizeof(selection->version_name), "%s",
            provider_version.versionName ? provider_version.versionName : "(unnamed)");
    }
    std::printf("ffxQuery(GetProviderVersion, %s) returned %u: id=0x%016" PRIx64
        " name=%s\n", effect_name, query_result, provider_version.versionId,
        provider_version.versionName ? provider_version.versionName : "(unnamed)");
    return query_result == FFX_API_RETURN_OK;
}

bool create_framegeneration_context(const FfxFunctions& functions,
    ID3D12Device* device, ProviderSelection* selection)
{
    ffxCreateContextDescFrameGeneration create = {};
    ffxCreateBackendDX12Desc backend = {};
    ffxCreateBackendDX12AllocationCallbacksDesc allocation_callbacks = {};
    ffxCreateContextDescFrameGenerationVersion version = {};
    ffxContext context = nullptr;

    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    create.header.pNext = &backend.header;
    create.displaySize = {1280, 720};
    create.maxRenderSize = {1280, 720};
    create.backBufferFormat = FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.header.pNext = &version.header;
    backend.device = device;
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    version.header.pNext = &allocation_callbacks.header;
    version.version = FFX_FRAMEGENERATION_VERSION;
    allocation_callbacks.header.type =
        FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12_ALLOCATION_CALLBACKS;
    allocation_callbacks.pfnFfxResourceAllocator = provider_resource_allocate;
    allocation_callbacks.pfnFfxResourceDeallocator = provider_resource_deallocate;

    if (selection)
        selection->attempted = true;
    g_provider_allocation_device = device;
    const ffxReturnCode_t result = functions.createContext(
        &context, &create.header, nullptr);
    if (selection)
        selection->create_result = result;
    std::printf("ffxCreateContext(Frame Generation API %u.%u.%u) returned %u\n",
        FFX_FRAMEGENERATION_VERSION_MAJOR, FFX_FRAMEGENERATION_VERSION_MINOR,
        FFX_FRAMEGENERATION_VERSION_PATCH, result);
    if (result != FFX_API_RETURN_OK) {
        g_provider_allocation_device = nullptr;
        return false;
    }
    const bool query_success = report_selected_provider(functions, &context,
        "frame-generation", selection);
    const ffxReturnCode_t destroy_result = functions.destroyContext(&context, nullptr);
    g_provider_allocation_device = nullptr;
    std::printf("ffxDestroyContext(frame-generation) returned %u\n", destroy_result);
    return query_success && destroy_result == FFX_API_RETURN_OK;
}

#if FFX_PROVIDER_PROBE_HAS_DENOISER
bool create_denoiser_context(const FfxFunctions& functions, ID3D12Device* device,
    ProviderSelection* selection)
{
    ffxCreateContextDescDenoiser create = {};
    ffxCreateBackendDX12Desc backend = {};
    ffxCreateBackendDX12AllocationCallbacksDesc allocation_callbacks = {};
    ffxContext context = nullptr;

    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER;
    create.header.pNext = &backend.header;
    create.version = FFX_DENOISER_VERSION;
    create.maxRenderSize = {640, 360};
    create.signalFlags = FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE;
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.header.pNext = &allocation_callbacks.header;
    backend.device = device;
    allocation_callbacks.header.type =
        FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12_ALLOCATION_CALLBACKS;
    allocation_callbacks.pfnFfxResourceAllocator = provider_resource_allocate;
    allocation_callbacks.pfnFfxResourceDeallocator = provider_resource_deallocate;

    if (selection)
        selection->attempted = true;
    g_provider_allocation_device = device;
    const ffxReturnCode_t result = functions.createContext(
        &context, &create.header, nullptr);
    if (selection)
        selection->create_result = result;
    std::printf("ffxCreateContext(Ray Regeneration API %u.%u.%u) returned %u\n",
        FFX_DENOISER_VERSION_MAJOR, FFX_DENOISER_VERSION_MINOR,
        FFX_DENOISER_VERSION_PATCH, result);
    if (result != FFX_API_RETURN_OK) {
        g_provider_allocation_device = nullptr;
        return false;
    }
    const bool query_success = report_selected_provider(functions, &context,
        "denoiser/ray-regeneration", selection);
    const ffxReturnCode_t destroy_result = functions.destroyContext(&context, nullptr);
    g_provider_allocation_device = nullptr;
    std::printf("ffxDestroyContext(denoiser/ray-regeneration) returned %u\n",
        destroy_result);
    return query_success && destroy_result == FFX_API_RETURN_OK;
}
#endif

void print_selection_summary(const char* effect, const ProviderSelection& selection)
{
    char name[sizeof(selection.version_name)] = {};
    for (size_t index = 0; index < sizeof(name) - 1 && selection.version_name[index];
         ++index) {
        const unsigned char c = static_cast<unsigned char>(selection.version_name[index]);
        name[index] = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'
            ? static_cast<char>(c)
            : '_';
    }
    std::printf("FFX_PROVIDER_PROBE_RESULT effect=%s attempted=%u "
                "create_return=%" PRIu32 " query_return=%" PRIu32
                " selected_id=0x%016" PRIx64 " selected_name=%s\n",
        effect, selection.attempted ? 1u : 0u, selection.create_result,
        selection.query_result, selection.version_id,
        name[0] ? name : "none");
}

void print_usage(const wchar_t* executable)
{
    ::fwprintf(stderr,
        L"Usage: %ls <amd_fidelityfx_loader_dx12.dll> [--create|--dispatch|--create-framegeneration|--create-denoiser] [--provider-index N]\n",
        executable);
}

} // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    bool create = false;
    bool dispatch = false;
    bool create_framegeneration = false;
    bool create_denoiser = false;
    uint64_t provider_index = UINT64_MAX;
    for (int index = 2; index < argc; ++index) {
        if (wcscmp(argv[index], L"--create") == 0) {
            create = true;
        } else if (wcscmp(argv[index], L"--dispatch") == 0) {
            create = true;
            dispatch = true;
        } else if (wcscmp(argv[index], L"--create-framegeneration") == 0) {
            create_framegeneration = true;
        } else if (wcscmp(argv[index], L"--create-denoiser") == 0) {
            create_denoiser = true;
        } else if (wcscmp(argv[index], L"--provider-index") == 0 && index + 1 < argc) {
            provider_index = std::wcstoull(argv[++index], nullptr, 10);
            create = true;
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    ID3D12Device* device = nullptr;
    HMODULE module = nullptr;
    FfxFunctions functions;
    std::vector<uint64_t> provider_ids;
    if (!create_device(&device) || !load_functions(argv[1], &module, &functions) ||
        !enumerate_effect_versions(functions, device,
            FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, "upscaler", &provider_ids)) {
        if (module)
            FreeLibrary(module);
        if (device)
            device->Release();
        return EXIT_FAILURE;
    }

    /* Frame generation has a separately selectable provider family.  It is
     * intentionally enumerated even when the caller only creates an
     * upscaler context, so a later driver/SDK change cannot be mistaken for
     * proof that a neural FG provider was selected.  Failure is reported but
     * does not invalidate the required upscaler probe. */
    std::vector<uint64_t> framegeneration_provider_ids;
    if (!enumerate_effect_versions(functions, device,
            FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION,
            "frame-generation", &framegeneration_provider_ids)) {
        std::fprintf(stderr,
            "Frame-generation provider enumeration unavailable for this loader/adapter.\n");
    }

#if FFX_PROVIDER_PROBE_HAS_DENOISER
    /* FSR Ray Regeneration is supplied through the public denoiser API. It
     * nevertheless has an independent provider list, so enumerate it rather
     * than infer it from the upscaler or frame-generation result. */
    std::vector<uint64_t> denoiser_provider_ids;
    if (!enumerate_effect_versions(functions, device,
            FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER,
            "denoiser/ray-regeneration", &denoiser_provider_ids)) {
        std::fprintf(stderr,
            "Denoiser/Ray-Regeneration provider enumeration unavailable for this loader/adapter.\n");
    }
#else
    /* The Q2RTX-pinned FSR-only SDK source closure intentionally omits the
     * denoiser kit. Do not guess a private type value when compiling only
     * against that reduced closure; point the probe at the full SDK instead. */
    std::printf("ray-regeneration providers: not queried (full SDK denoiser header unavailable)\n");
#endif

    ProviderSelection upscaler_selection;
    ProviderSelection framegeneration_selection;
    ProviderSelection denoiser_selection;
    bool success = true;
    if (create) {
        if (provider_index != UINT64_MAX && provider_index >= provider_ids.size()) {
            std::fprintf(stderr, "Provider index %" PRIu64 " is out of range.\n",
                provider_index);
            success = false;
        } else {
            const uint64_t provider_id = provider_index == UINT64_MAX
                ? 0 : provider_ids[static_cast<size_t>(provider_index)];
            success = create_fsr411_context(functions, device, provider_id, dispatch,
                &upscaler_selection);
        }
    }
    if (create_framegeneration)
        success = create_framegeneration_context(functions, device,
            &framegeneration_selection) && success;
    if (create_denoiser) {
#if FFX_PROVIDER_PROBE_HAS_DENOISER
        success = create_denoiser_context(functions, device, &denoiser_selection) && success;
#else
        std::fprintf(stderr,
            "Cannot create Ray Regeneration context: full SDK denoiser header unavailable.\n");
        success = false;
#endif
    }

    print_selection_summary("upscaler", upscaler_selection);
    print_selection_summary("frame-generation", framegeneration_selection);
    print_selection_summary("ray-regeneration", denoiser_selection);

    FreeLibrary(module);
    device->Release();
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
