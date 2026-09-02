# Official FSR provider probe

fsr_provider_probe.cpp is a deliberately small Windows/DX12 program for
observing the public AMD FFX API under Wine/Proton. It does not contain,
extract, or redistribute SDK DLLs, DXIL, SPIR-V, neural weights, or model
payloads.

It creates a DX12 device, loads a caller-selected
amd_fidelityfx_loader_dx12.dll, enumerates upscaler and frame-generation
providers, and can create an FSR 4.1.1 API context. A successful create is
enough to reveal which upscaler provider AMD selected; dispatch capture is a
later, separate step because it needs correctly initialized color/depth/MV
resources and a command queue. The API headers supplied with SDK 2.3 do not
publish a Ray-Regeneration create descriptor, so the harness reports that
limitation explicitly instead of guessing a private provider type.

Build, replacing SDK with the local FidelityFX SDK 2.3 checkout:

    x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -Wextra -Werror \
      -Wno-unknown-pragmas \
      -I"$SDK/Kits/FidelityFX/api/include" \
      -I"$SDK/Kits/FidelityFX/api/include/dx12" \
      -I"$SDK/Kits/FidelityFX/framegeneration/include" \
      -I"$SDK/Kits/FidelityFX/upscalers/include" \
  fsr_provider_probe.cpp -municode -static -static-libgcc -static-libstdc++ \
      -ld3d12 -ldxgi -lole32 -o fsr_provider_probe.exe

`-Wno-unknown-pragmas` is limited to the SDK's public Frame Generation header:
MinGW does not understand AMD's warning-control pragmas. All ordinary harness
warnings remain errors.

Run from the SDK's Kits/FidelityFX/signedbin directory so the loader can find
its sibling provider DLLs:

    mkdir -p /tmp/ffx-vkd3d-dump
    cd "$SDK/Kits/FidelityFX/signedbin"
    VKD3D_SHADER_DUMP_PATH=/tmp/ffx-vkd3d-dump \
      wine /absolute/path/to/fsr_provider_probe.exe \
      "$PWD/amd_fidelityfx_loader_dx12.dll" --dispatch

The --dispatch option records and waits for one 640x360 -> 1280x720 reset
frame. It intentionally uses throwaway, undefined input pixels: this is a
provider-schedule capture, not an image-quality test. Use --provider-index N
only after recording the enumeration output. It is
intentional that the probe does not claim FSR 4.1.1 runs on RDNA2: it records
the provider and return code selected by the installed driver/DLL combination.
See the parent CAPTURE.md for the next-stage capture requirements and
licensing/provenance boundaries.

The harness registers the public DX12 resource allocator/deallocator callbacks.
Consequently its log records provider-owned D3D12 allocation dimensions,
formats, mip counts, flags, and initial states without extracting a provider
payload. It does not observe descriptor writes, root constants, barriers, or
individual dispatch dimensions; those remain RenderDoc/d3d12-replayer or
explicit D3D12-interception work.
