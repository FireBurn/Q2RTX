// SPDX-License-Identifier: MIT
//
// Minimal legacy FidelityFX API provider probe.  This is intentionally kept
// separate from fsr_provider_probe.cpp: the public API in the source-v07 FSR4
// SDK predates the versioned 4.1.1 descriptors and the two ABIs must not be
// mixed.  It contains no AMD payloads and only loads a caller-supplied loader.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ffx_api.h"
#include "ffx_api_dx12.h"
#include "ffx_upscale.h"

namespace {

struct Functions {
    PfnFfxCreateContext create_context = nullptr;
    PfnFfxDestroyContext destroy_context = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
};

void print_hresult(const char* action, HRESULT result)
{
    std::fprintf(stderr, "%s failed: HRESULT 0x%08" PRIx32 "\n", action,
        static_cast<uint32_t>(result));
}

template <typename Function>
Function load_function(HMODULE module, const char* name)
{
    const FARPROC raw = GetProcAddress(module, name);
    static_assert(sizeof(raw) == sizeof(Function),
        "Windows function-pointer representation must match FARPROC");
    Function result = nullptr;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

bool create_device(ID3D12Device** out_device)
{
    IDXGIFactory6* factory = nullptr;
    HRESULT result = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(result)) {
        print_hresult("CreateDXGIFactory2", result);
        return false;
    }

    bool found = false;
    for (UINT index = 0; ; ++index) {
        IDXGIAdapter1* adapter = nullptr;
        result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND)
            break;
        if (FAILED(result)) {
            print_hresult("IDXGIFactory6::EnumAdapters1", result);
            break;
        }

        DXGI_ADAPTER_DESC1 description = {};
        adapter->GetDesc1(&description);
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            ID3D12Device* device = nullptr;
            result = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0,
                IID_PPV_ARGS(&device));
            ::fwprintf(stdout,
                L"adapter %u: %ls (vendor %04x, device %04x): %hs\n", index,
                description.Description, description.VendorId,
                description.DeviceId, SUCCEEDED(result) ? "usable" : "unavailable");
            if (SUCCEEDED(result)) {
                *out_device = device;
                found = true;
                adapter->Release();
                break;
            }
        }
        adapter->Release();
    }
    factory->Release();
    if (!found)
        std::fprintf(stderr, "No hardware DX12 adapter could be created.\n");
    return found;
}

bool load_functions(const wchar_t* loader_path, HMODULE* out_module,
    Functions* out_functions)
{
    HMODULE module = LoadLibraryW(loader_path);
    if (!module) {
        ::fwprintf(stderr, L"LoadLibraryW failed for %ls (error %lu)\n",
            loader_path, GetLastError());
        return false;
    }

    const Functions functions = {
        load_function<PfnFfxCreateContext>(module, "ffxCreateContext"),
        load_function<PfnFfxDestroyContext>(module, "ffxDestroyContext"),
        load_function<PfnFfxQuery>(module, "ffxQuery"),
        load_function<PfnFfxDispatch>(module, "ffxDispatch"),
    };
    if (!functions.create_context || !functions.destroy_context || !functions.query ||
        !functions.dispatch) {
        std::fprintf(stderr, "The supplied loader is missing a required FFX API export.\n");
        FreeLibrary(module);
        return false;
    }
    *out_module = module;
    *out_functions = functions;
    return true;
}

bool enumerate_versions(const Functions& functions, ID3D12Device* device,
    std::vector<uint64_t>* out_ids, std::vector<const char*>* out_names)
{
    ffxQueryDescGetVersions query = {};
    query.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    query.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    query.device = device;
    uint64_t count = 0;
    query.outputCount = &count;
    ffxReturnCode_t result = functions.query(nullptr, &query.header);
    if (result != FFX_API_RETURN_OK || count == 0) {
        std::fprintf(stderr, "ffxQuery(GetVersions/count) returned %u, count=%" PRIu64 "\n",
            result, count);
        return false;
    }

    out_ids->assign(static_cast<size_t>(count), 0);
    out_names->assign(static_cast<size_t>(count), nullptr);
    query.versionIds = out_ids->data();
    query.versionNames = out_names->data();
    result = functions.query(nullptr, &query.header);
    if (result != FFX_API_RETURN_OK) {
        std::fprintf(stderr, "ffxQuery(GetVersions/list) returned %u\n", result);
        return false;
    }
    out_ids->resize(static_cast<size_t>(count));
    out_names->resize(static_cast<size_t>(count));
    std::printf("legacy upscaler providers (%" PRIu64 "):\n", count);
    for (uint64_t index = 0; index < count; ++index) {
        const char* name = (*out_names)[static_cast<size_t>(index)];
        std::printf("  [%" PRIu64 "] id=0x%016" PRIx64 " name=%s\n", index,
            (*out_ids)[static_cast<size_t>(index)], name ? name : "(unnamed)");
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
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1,
    };
    const D3D12_RESOURCE_DESC description = {
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, static_cast<UINT64>(width), height,
        1, 1, format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, flags,
    };
    const HRESULT result = device->CreateCommittedResource(&heap_properties,
        D3D12_HEAP_FLAG_NONE, &description, initial_state, nullptr,
        IID_PPV_ARGS(output));
    if (FAILED(result))
        print_hresult("ID3D12Device::CreateCommittedResource", result);
    return SUCCEEDED(result);
}

bool dispatch_once(const Functions& functions, ffxContext* context,
    ID3D12Device* device)
{
    constexpr uint32_t render_width = 640;
    constexpr uint32_t render_height = 360;
    constexpr uint32_t output_width = 1280;
    constexpr uint32_t output_height = 720;
    const D3D12_RESOURCE_STATES read_state =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    DispatchResources resources;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* command_list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE fence_event = nullptr;
    bool success = false;

    if (!create_texture(device, render_width, render_height,
            DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE,
            read_state, &resources.color) ||
        !create_texture(device, render_width, render_height, DXGI_FORMAT_R32_FLOAT,
            D3D12_RESOURCE_FLAG_NONE, read_state, &resources.depth) ||
        !create_texture(device, render_width, render_height, DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_FLAG_NONE, read_state, &resources.motion_vectors) ||
        !create_texture(device, render_width, render_height, DXGI_FORMAT_R8_UNORM,
            D3D12_RESOURCE_FLAG_NONE, read_state, &resources.reactive) ||
        !create_texture(device, render_width, render_height, DXGI_FORMAT_R8_UNORM,
            D3D12_RESOURCE_FLAG_NONE, read_state, &resources.composition) ||
        !create_texture(device, output_width, output_height,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &resources.output)) {
        goto cleanup;
    }

    {
        const D3D12_COMMAND_QUEUE_DESC queue_description = {
            D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0,
        };
        HRESULT result = device->CreateCommandQueue(&queue_description,
            IID_PPV_ARGS(&queue));
        if (FAILED(result)) {
            print_hresult("ID3D12Device::CreateCommandQueue", result);
            goto cleanup;
        }
        result = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator));
        if (FAILED(result)) {
            print_hresult("ID3D12Device::CreateCommandAllocator", result);
            goto cleanup;
        }
        result = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator, nullptr, IID_PPV_ARGS(&command_list));
        if (FAILED(result)) {
            print_hresult("ID3D12Device::CreateCommandList", result);
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
        dispatch.transparencyAndComposition = ffxApiGetResourceDX12(
            resources.composition, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatch.output = ffxApiGetResourceDX12(resources.output,
            FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch.motionVectorScale = {
            static_cast<float>(render_width), static_cast<float>(render_height),
        };
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
        std::printf("ffxDispatch(legacy 640x360 -> 1280x720) returned %u\n", result);
        if (result != FFX_API_RETURN_OK)
            goto cleanup;
    }

    {
        HRESULT result = command_list->Close();
        if (FAILED(result)) {
            print_hresult("ID3D12GraphicsCommandList::Close", result);
            goto cleanup;
        }
        ID3D12CommandList* command_lists[] = {command_list};
        queue->ExecuteCommandLists(1, command_lists);
        result = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(result)) {
            print_hresult("ID3D12Device::CreateFence", result);
            goto cleanup;
        }
        result = queue->Signal(fence, 1);
        if (FAILED(result)) {
            print_hresult("ID3D12CommandQueue::Signal", result);
            goto cleanup;
        }
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event) {
            std::fprintf(stderr, "CreateEventW failed: %lu\n", GetLastError());
            goto cleanup;
        }
        result = fence->SetEventOnCompletion(1, fence_event);
        if (FAILED(result) || WaitForSingleObject(fence_event, 30000) != WAIT_OBJECT_0) {
            if (FAILED(result))
                print_hresult("ID3D12Fence::SetEventOnCompletion", result);
            else
                std::fprintf(stderr, "Timed out waiting for legacy FSR4 dispatch.\n");
            goto cleanup;
        }
    }
    success = true;

cleanup:
    if (fence_event) CloseHandle(fence_event);
    if (fence) fence->Release();
    if (command_list) command_list->Release();
    if (allocator) allocator->Release();
    if (queue) queue->Release();
    resources.release();
    return success;
}

bool create_provider(const Functions& functions, ID3D12Device* device,
    uint64_t version_id, const char* requested_name, bool dispatch)
{
    ffxCreateContextDescUpscale create = {};
    ffxCreateBackendDX12Desc backend = {};
    ffxOverrideVersion override_version = {};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &backend.header;
    create.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    create.maxRenderSize = {640, 360};
    create.maxUpscaleSize = {1280, 720};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.header.pNext = &override_version.header;
    backend.device = device;
    override_version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    override_version.versionId = version_id;

    ffxContext context = nullptr;
    const ffxReturnCode_t create_result = functions.create_context(&context,
        &create.header, nullptr);
    ffxReturnCode_t query_result = UINT32_MAX;
    uint64_t selected_id = 0;
    const char* selected_name = nullptr;
    bool dispatch_success = !dispatch;
    if (create_result == FFX_API_RETURN_OK) {
        ffxQueryGetProviderVersion provider = {};
        provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        query_result = functions.query(&context, &provider.header);
        if (query_result == FFX_API_RETURN_OK) {
            selected_id = provider.versionId;
            selected_name = provider.versionName;
        }
        dispatch_success = !dispatch || dispatch_once(functions, &context, device);
        const ffxReturnCode_t destroy_result = functions.destroy_context(&context, nullptr);
        if (destroy_result != FFX_API_RETURN_OK)
            std::fprintf(stderr, "ffxDestroyContext returned %u\n", destroy_result);
    }
    std::printf("FFX_LEGACY_FSR4_PROVIDER_RESULT requested=%s requested_id=0x%016" PRIx64
                " create=%u query=%u dispatch=%s selected_id=0x%016" PRIx64 " selected=%s\n",
        requested_name ? requested_name : "(unnamed)", version_id, create_result,
        query_result, !dispatch ? "not-requested" : dispatch_success ? "ok" : "failed",
        selected_id, selected_name ? selected_name : "(none)");
    return create_result == FFX_API_RETURN_OK && query_result == FFX_API_RETURN_OK &&
        dispatch_success;
}

} // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc < 2 || argc > 3 ||
        (argc == 3 && ::wcscmp(argv[2], L"--dispatch") != 0)) {
        ::fwprintf(stderr, L"usage: %ls <amd_fidelityfx_loader_dx12.dll> [--dispatch]\n",
            argv[0]);
        return 2;
    }
    const bool dispatch = argc == 3;

    HMODULE module = nullptr;
    Functions functions = {};
    if (!load_functions(argv[1], &module, &functions))
        return 1;

    ID3D12Device* device = nullptr;
    if (!create_device(&device)) {
        FreeLibrary(module);
        return 1;
    }

    std::vector<uint64_t> ids;
    std::vector<const char*> names;
    bool all_created = enumerate_versions(functions, device, &ids, &names);
    if (all_created) {
        for (size_t index = 0; index < ids.size(); ++index)
            all_created = create_provider(functions, device, ids[index], names[index],
                dispatch && names[index] && std::strstr(names[index], "4.") != nullptr) &&
                all_created;
    }
    device->Release();
    FreeLibrary(module);
    return all_created ? 0 : 1;
}
