// SPDX-License-Identifier: MIT
// Synthetic upload/readback utilities; contains no provider payload.
struct ProbePixels {
    std::vector<ID3D12Resource*> staging;
    ID3D12Resource* readback = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT output_layout{};
    ~ProbePixels() {
        for (auto* resource : staging) resource->Release();
        if (readback) readback->Release();
    }

    bool buffer(ID3D12Device* device, UINT64 size, bool upload, ID3D12Resource** result) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK;
        heap.CreationNodeMask = heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &desc, upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(result)));
    }

    void transition(ID3D12GraphicsCommandList* commands, ID3D12Resource* image,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {image, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        commands->ResourceBarrier(1, &barrier);
    }

    // kind: 0=colour checker, 1=depth, 2=zero motion/mask, 3=NaN output sentinel.
    bool upload(ID3D12Device* device, ID3D12GraphicsCommandList* commands,
        ID3D12Resource* image, D3D12_RESOURCE_STATES state, unsigned kind) {
        auto desc = image->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
        UINT64 size = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &size);
        ID3D12Resource* data = nullptr;
        if (!buffer(device, size, true, &data)) return false;
        staging.push_back(data);
        unsigned char* mapped = nullptr;
        D3D12_RANGE no_reads{0, 0};
        if (FAILED(data->Map(0, &no_reads, reinterpret_cast<void**>(&mapped)))) return false;
        std::memset(mapped, 0, static_cast<size_t>(size));
        for (UINT y = 0; y < desc.Height; ++y) {
            auto* row = mapped + layout.Offset + y * layout.Footprint.RowPitch;
            for (UINT x = 0; x < desc.Width; ++x) {
                if (kind == 0 || kind == 3) {
                    uint16_t rgba[4] = {0x3800, 0x3000, 0x3400, 0x3c00};
                    if (((x / 32) ^ (y / 32)) & 1) std::swap(rgba[0], rgba[1]);
                    if (kind == 3) for (auto& component : rgba) component = 0x7e00;
                    std::memcpy(row + x * 8, rgba, 8);
                } else if (kind == 1) {
                    const float depth = 0.5f;
                    std::memcpy(row + x * 4, &depth, 4);
                }
            }
        }
        data->Unmap(0, nullptr);
        transition(commands, image, state, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = image;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = data;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = layout;
        commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(commands, image, D3D12_RESOURCE_STATE_COPY_DEST, state);
        return true;
    }

    bool copy_output(ID3D12Device* device, ID3D12GraphicsCommandList* commands,
        ID3D12Resource* image) {
        auto desc = image->GetDesc();
        UINT64 size = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &output_layout, nullptr, nullptr, &size);
        if (!buffer(device, size, false, &readback)) return false;
        transition(commands, image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        src.pResource = image;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource = readback;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = output_layout;
        commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        return true;
    }

    bool verify() {
        unsigned char* mapped = nullptr;
        if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
            return false;
        size_t nonfinite = 0, nonzero = 0, checker_samples = 0, checker_matches = 0;
        for (UINT y = 0; y < output_layout.Footprint.Height; ++y)
            for (UINT x = 0; x < output_layout.Footprint.Width; ++x) {
                uint16_t rgba[4];
                std::memcpy(rgba, mapped + output_layout.Offset + y * output_layout.Footprint.RowPitch + x * 8, 8);
                for (unsigned c = 0; c < 3; ++c) {
                    nonfinite += (rgba[c] & 0x7c00) == 0x7c00;
                    nonzero += (rgba[c] & 0x7fff) != 0;
                }
                // Sample tile centres, away from reconstruction boundaries.
                // Input tiles are 32 pixels wide; this probe scales by 2.
                // Positive finite half floats preserve ordering as integers.
                if (x % 64 == 32 && y % 64 == 32) {
                    ++checker_samples;
                    const bool green_tile = ((x / 64) ^ (y / 64)) & 1;
                    const bool positive_finite = (rgba[0] & 0x8000) == 0 &&
                        (rgba[1] & 0x8000) == 0 &&
                        (rgba[0] & 0x7c00) != 0x7c00 &&
                        (rgba[1] & 0x7c00) != 0x7c00;
                    checker_matches += positive_finite &&
                        (green_tile ? rgba[1] > rgba[0] : rgba[0] > rgba[1]);
                }
            }
        D3D12_RANGE no_writes{0, 0};
        readback->Unmap(0, &no_writes);
        std::printf("FFX_PIXEL_CHECK rgb_nonfinite=%zu rgb_nonzero=%zu\n", nonfinite, nonzero);
        std::printf("FFX_CHECKER_CHECK matched=%zu sampled=%zu\n", checker_matches, checker_samples);
        return nonfinite == 0 && nonzero != 0 && checker_samples != 0 &&
            checker_matches == checker_samples;
    }
};
