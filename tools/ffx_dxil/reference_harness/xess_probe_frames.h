// SPDX-License-Identifier: MIT
// Synthetic reference frames and raw RGBA16F readbacks. No provider payload.
#include <string>

template<class T> struct XessProbeObject {
    T* p = nullptr;
    ~XessProbeObject() { if (p) p->Release(); }
    XessProbeObject() = default;
    XessProbeObject(const XessProbeObject&) = delete;
    XessProbeObject& operator=(const XessProbeObject&) = delete;
};

static bool xess_probe_texture(ID3D12Device* device, xess_2d_t size,
    DXGI_FORMAT format, bool output, ID3D12Resource** resource)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = size.x; desc.Height = size.y;
    desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = format;
    desc.Flags = output ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, output ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        nullptr, IID_PPV_ARGS(resource)));
}

static bool xess_probe_readback(ProbePixels& pixels, xess_2d_t input,
    const wchar_t* directory, unsigned frame, std::vector<uint16_t>& result)
{
    const auto& layout = pixels.output_layout;
    const unsigned width = layout.Footprint.Width, height = layout.Footprint.Height;
    unsigned char* mapped = nullptr;
    if (FAILED(pixels.readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    result.resize(size_t(width) * height * 4);
    for (unsigned y = 0; y < height; ++y)
        std::memcpy(result.data() + size_t(y) * width * 4,
            mapped + layout.Offset + size_t(y) * layout.Footprint.RowPitch, size_t(width) * 8);
    D3D12_RANGE no_writes{0, 0};
    pixels.readback->Unmap(0, &no_writes);
    size_t nonfinite = 0, nonzero = 0, samples = 0, matches = 0;
    for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel)
        for (unsigned c = 0; c < 3; ++c) {
            const auto bits = result[pixel * 4 + c];
            nonfinite += (bits & 0x7c00) == 0x7c00;
            nonzero += (bits & 0x7fff) != 0;
        }
    // Map input tile centres to output coordinates; no assumed scale factor.
    for (unsigned iy = 16; iy + 16 <= input.y; iy += 32)
        for (unsigned ix = 16; ix + 16 <= input.x; ix += 32) {
            const unsigned x = unsigned(uint64_t(ix) * width / input.x);
            const unsigned y = unsigned(uint64_t(iy) * height / input.y);
            const auto* rgba = &result[(size_t(y) * width + x) * 4];
            const bool green = ((ix / 32) ^ (iy / 32)) & 1;
            const bool positive_finite = !(rgba[0] & 0x8000) && !(rgba[1] & 0x8000) &&
                (rgba[0] & 0x7c00) != 0x7c00 && (rgba[1] & 0x7c00) != 0x7c00;
            ++samples;
            matches += positive_finite && (green ? rgba[1] > rgba[0] : rgba[0] > rgba[1]);
        }
    std::printf("XESS_PIXELS frame=%u nonfinite=%zu nonzero=%zu matched=%zu sampled=%zu\n",
        frame, nonfinite, nonzero, matches, samples);
    const auto path = std::wstring(directory) + L"\\frame-" + std::to_wstring(frame) + L".rgba16f";
    // Do not overwrite an earlier capture; a fresh directory is required.
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "Creating readback file failed: %lu\n", GetLastError());
        return false;
    }
    DWORD written = 0;
    const DWORD bytes = static_cast<DWORD>(result.size() * sizeof(uint16_t));
    const bool saved = WriteFile(file, result.data(), bytes, &written, nullptr) && written == bytes;
    const bool closed = CloseHandle(file) != FALSE;
    return saved && closed && !nonfinite && nonzero && samples && matches == samples;
}

static bool xess_probe_frames(ID3D12Device* device, xess_context_handle_t context,
    decltype(&xessD3D12Execute) execute, xess_2d_t input, xess_2d_t output, const wchar_t* directory,
    decltype(&xessD3D12GetResourcesToDump) internal_resources)
{
    XessProbeObject<ID3D12Resource> color, depth, velocity, target;
    XessProbeObject<ID3D12CommandQueue> queue;
    XessProbeObject<ID3D12CommandAllocator> allocator;
    XessProbeObject<ID3D12GraphicsCommandList> commands;
    XessProbeObject<ID3D12Fence> fence;
    if (!xess_probe_texture(device, input, DXGI_FORMAT_R16G16B16A16_FLOAT, false, &color.p) ||
        !xess_probe_texture(device, input, DXGI_FORMAT_R32_FLOAT, false, &depth.p) ||
        !xess_probe_texture(device, input, DXGI_FORMAT_R16G16_FLOAT, false, &velocity.p) ||
        !xess_probe_texture(device, output, DXGI_FORMAT_R16G16B16A16_FLOAT, true, &target.p)) return false;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(&commands.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    bool success = true;
    std::vector<uint16_t> first, current;
    for (unsigned frame = 0; frame < 4 && success; ++frame) {
        ProbePixels pixels;
        if (frame && (FAILED(allocator.p->Reset()) || FAILED(commands.p->Reset(allocator.p, nullptr)))) { success = false; break; }
        if (frame) pixels.transition(commands.p, target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const auto read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (!pixels.upload(device, commands.p, color.p, read, 0) ||
            !pixels.upload(device, commands.p, depth.p, read, 1) ||
            !pixels.upload(device, commands.p, velocity.p, read, 2) ||
            !pixels.upload(device, commands.p, target.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 3)) { success = false; break; }
        xess_d3d12_execute_params_t params{};
        params.pColorTexture = color.p; params.pDepthTexture = depth.p;
        params.pVelocityTexture = velocity.p; params.pOutputTexture = target.p;
        params.inputWidth = input.x; params.inputHeight = input.y;
        params.exposureScale = 1.0f;
        params.resetHistory = frame == 0 || frame == 3;
        const auto status = execute(context, commands.p, &params);
        std::printf("XESS_EXECUTE frame=%u reset=%u result=%d\n", frame, params.resetHistory, int(status));
        if (status == XESS_RESULT_SUCCESS && internal_resources) {
            xess_resources_to_dump_t* resources = nullptr;
            const auto query = internal_resources(context, &resources);
            std::printf("XESS_INTERNALS frame=%u result=%d count=%u\n", frame, int(query),
                query == XESS_RESULT_SUCCESS && resources ? resources->resource_count : 0);
            if (query == XESS_RESULT_SUCCESS && resources)
                for (uint32_t i = 0; i < resources->resource_count; ++i) {
                    const auto desc = resources->resources[i]->GetDesc();
                    std::printf("XESS_INTERNAL_RESOURCE index=%u name=%s width=%llu height=%u layers=%u format=%u\n",
                        i, resources->resource_names[i], static_cast<unsigned long long>(desc.Width),
                        desc.Height, desc.DepthOrArraySize, unsigned(desc.Format));
                }
        }
        if (status != XESS_RESULT_SUCCESS || !pixels.copy_output(device, commands.p, target.p) || FAILED(commands.p->Close())) { success = false; break; }
        ID3D12CommandList* lists[] = {commands.p};
        queue.p->ExecuteCommandLists(1, lists);
        // On failed completion, terminate without releasing in-flight resources.
        if (FAILED(queue.p->Signal(fence.p, frame + 1)) ||
            FAILED(fence.p->SetEventOnCompletion(frame + 1, event)) ||
            WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 ||
            FAILED(device->GetDeviceRemovedReason()) || fence.p->GetCompletedValue() != frame + 1) {
            std::fprintf(stderr, "XeSS GPU completion failed on frame %u\n", frame);
            std::fflush(nullptr);
            ExitProcess(1);
        }
        std::printf("XESS_COMPLETION frame=%u fence=%u device_ok=1\n", frame, frame + 1);
        success = xess_probe_readback(pixels, input, directory, frame, current);
        if (!frame) first = current;
        if (frame == 3) {
            const bool equal = first == current;
            std::printf("XESS_RESET_REPEAT bitwise_equal=%u\n", unsigned(equal));
            success = success && equal;
        }
    }
    CloseHandle(event);
    return success;
}
