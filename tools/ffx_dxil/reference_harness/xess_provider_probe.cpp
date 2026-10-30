// SPDX-License-Identifier: MIT
// Capture XeSS pipeline creation using a caller-supplied runtime and SDK.
// Optional synthetic execution is a reference, not native replay correctness.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cwchar>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include "xess/xess_d3d12.h"
#include "xess/xess_d3d12_debug.h"
#include "probe_pixels.h"
#include "xess_probe_frames.h"

template<class T> static T symbol(HMODULE module, const char* name)
{
    auto address = GetProcAddress(module, name);
    T function = nullptr;
    static_assert(sizeof(function) == sizeof(address));
    std::memcpy(&function, &address, sizeof(function));
    if (!function) std::fprintf(stderr, "Missing export: %s\n", name);
    return function;
}

static bool output_dimension(const wchar_t* name,uint32_t fallback,uint32_t& result)
{
    wchar_t value[32]{};
    const DWORD length=GetEnvironmentVariableW(name,value,32);
    if(!length) {result=fallback;return true;}
    if(length>=32) return false;
    for(DWORD i=0;i<length;++i) if(value[i]<L'0' || value[i]>L'9') return false;
    wchar_t* end=nullptr;
    const auto parsed=std::wcstoul(value,&end,10);
    if(!end || *end || parsed<64 || parsed>4096) return false;
    result=static_cast<uint32_t>(parsed);return true;
}

int wmain(int argc, wchar_t** argv)
{
    const bool dispatch = argc == 4 && std::wcscmp(argv[2], L"--dispatch") == 0;
    if (argc != 2 && !dispatch) {
        std::fprintf(stderr, "Usage: xess_provider_probe.exe ABSOLUTE_PATH_TO_libxess.dll [--dispatch EXISTING_OUTPUT_DIRECTORY]\n");
        return 2;
    }
    xess_2d_t output{};
    if(!output_dimension(L"XESS_PROBE_WIDTH",1280,output.x) ||
       !output_dimension(L"XESS_PROBE_HEIGHT",720,output.y)) {
        std::fprintf(stderr,"Output dimensions must be integers in [64,4096]\n");return 2;
    }
    HMODULE module = LoadLibraryExW(argv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        std::fprintf(stderr, "LoadLibraryExW failed: %lu\n", GetLastError());
        return 1;
    }
    auto version = symbol<decltype(&xessGetVersion)>(module, "xessGetVersion");
    auto create = symbol<decltype(&xessD3D12CreateContext)>(module, "xessD3D12CreateContext");
    auto init = symbol<decltype(&xessD3D12Init)>(module, "xessD3D12Init");
    auto destroy = symbol<decltype(&xessDestroyContext)>(module, "xessDestroyContext");
    auto execute = symbol<decltype(&xessD3D12Execute)>(module, "xessD3D12Execute");
    auto input_resolution = symbol<decltype(&xessGetInputResolution)>(module, "xessGetInputResolution");
    decltype(&xessD3D12GetResourcesToDump) internal_resources = nullptr;
    decltype(&xessStartDump) start_dump = nullptr;
    if (GetEnvironmentVariableW(L"XESS_PROBE_INTERNALS", nullptr, 0)) {
        internal_resources = symbol<decltype(internal_resources)>(module, "xessD3D12GetResourcesToDump");
        start_dump = symbol<decltype(start_dump)>(module, "xessStartDump");
        if (!internal_resources || !start_dump) { FreeLibrary(module); return 1; }
    }
    if (!version || !create || !init || !destroy || !execute || !input_resolution) { FreeLibrary(module); return 1; }
    xess_version_t v{};
    const auto vr = version(&v);
    std::printf("XESS_VERSION result=%d version=%u.%u.%u\n", int(vr), v.major, v.minor, v.patch);
    IDXGIFactory1* factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::fprintf(stderr, "CreateDXGIFactory1 failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        FreeLibrary(module); return 1;
    }
    int result = 1;
    for (UINT index = 0;; ++index) {
        IDXGIAdapter1* adapter = nullptr;
        hr = factory->EnumAdapters1(index, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr)) break;
        DXGI_ADAPTER_DESC1 desc{};
        hr = adapter->GetDesc1(&desc);
        if (FAILED(hr) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { adapter->Release(); continue; }
        ID3D12Device* device = nullptr;
        hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
        adapter->Release();
        std::printf("XESS_DEVICE adapter=%u vendor=%04x device=%04x hr=0x%08lx\n",
            index, desc.VendorId, desc.DeviceId, static_cast<unsigned long>(hr));
        if (FAILED(hr)) continue;
        xess_context_handle_t context = nullptr;
        auto status = create(device, &context);
        std::printf("XESS_CREATE result=%d\n", int(status));
        if (status == XESS_RESULT_SUCCESS && context) {
            xess_d3d12_init_params_t params{};
            params.outputResolution = output;
            params.qualitySetting = XESS_QUALITY_SETTING_QUALITY;
            status = init(context, &params);
            std::printf("XESS_INIT result=%d output=%ux%u quality=quality flags=0\n", int(status),output.x,output.y);
            bool frames_ok = !dispatch;
            if (status == XESS_RESULT_SUCCESS && dispatch) {
                /* Querying internal resources alone leaves the dump queue
                 * empty. Explicitly request all resources before execution.
                 * SDK dumping may be compiled out: log its status separately
                 * from execution success, rather than claiming a capture. */
                char dump_path[4096] = {};
                if (start_dump) {
                    BOOL substituted = FALSE;
                    if (!WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, argv[3], -1,
                            dump_path, sizeof(dump_path), nullptr, &substituted) || substituted) {
                        std::fprintf(stderr, "Dump directory must have an encodable ASCII-compatible path\n");
                        destroy(context); device->Release(); break;
                    }
                    xess_dump_parameters_t dump{};
                    dump.path = dump_path;
                    dump.frame_idx = 0;
                    dump.frame_count = 4;
                    dump.dump_elements_mask = XESS_DUMP_ALL;
                    const auto dump_result = start_dump(context, &dump);
                    std::printf("XESS_START_DUMP result=%d frames=4 mask=all\n", int(dump_result));
                }
                xess_2d_t input{};
                const auto queried = input_resolution(context, &params.outputResolution, params.qualitySetting, &input);
                std::printf("XESS_INPUT result=%d resolution=%ux%u\n", int(queried), input.x, input.y);
                if (queried == XESS_RESULT_SUCCESS && input.x && input.y)
                    frames_ok = xess_probe_frames(device, context, execute, input, params.outputResolution, argv[3], internal_resources);
            }
            const auto destroyed = destroy(context);
            std::printf("XESS_DESTROY result=%d\n", int(destroyed));
            if (status == XESS_RESULT_SUCCESS && destroyed == XESS_RESULT_SUCCESS && frames_ok) result = 0;
        }
        device->Release();
        if (!result) break;
    }
    factory->Release();
    FreeLibrary(module);
    return result;
}
