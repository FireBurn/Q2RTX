# Official FSR provider probe

The deliverable is a native Vulkan provider, without Wine/DX12 at runtime.
This executable is a reference for recovering and verifying that provider's
graph. Shared-resource experiments run only with `--interop`; omit it for
shader/dispatch captures so unrelated commands and interop failures cannot
contaminate the reference workload.

### Local upload snapshots for native replay research

`vkd3d-reference-upload.patch` applies to vkd3d-proton `634d341a5a312a3`.
Build with `enable_trace=true` (the optional profiling device wrapper at that
revision is stale; leave `enable_profiling=false`). Set
`VKD3D_REFERENCE_UPLOAD_PATH` to a new private directory when running this
synthetic probe. The Windows-only hook snapshots CPU-visible UPLOAD buffers
up to 64 MiB at Unmap, with resource/GPU-address/size records in the trace.
Files use exclusive creation; never reuse a directory. It is disabled without
that variable and is not a production-runtime feature.

Snapshots contain whole allocations, including padding; treat every byte as
private, potentially proprietary capture data. Do not publish them. Unmap
snapshots are not dispatch-time snapshots: a persistently mapped ring may
already have overwritten earlier constants. Check address reuse and lifetime
before using any snapshot for replay. Resources never unmapped, GPU-local
contents, and buffers above the bound are not captured by this hook.
The patch also adds `REFERENCE_HEAP` trace records to CPU-heap-handle queries,
reporting CPU/GPU bases, count, type and stride. `dispatch_trace.py` uses these
to resolve GPU table starts to CPU descriptor addresses; table lengths and
the copy chain still require separate recovery.

`vkd3d-reference-views.patch` adds raw SRV/UAV descriptor records to the same
opt-in capture on that pinned runtime's **embedded** descriptor path. It emits
`REFERENCE_VIEW` words with descriptor handle, kind, structure size and byte
offset. Other descriptor implementations and null/default view descriptions
are not covered. The tested x64 ABI uses 40-byte SRVs and 48-byte UAVs; decode
using that runtime's D3D12 headers, not a guessed portable C layout. Descriptor
copies and GPU table handles must still be correlated separately. Logs can
include union padding; retain them only as private capture artifacts.

`vkd3d-reference-ranges.patch` logs parsed root-table ranges with resolved
append offsets. The indexer joins these to captured view definitions through
`CopyDescriptorsSimple` and snapshots the resulting entries at each direct
dispatch. Missing definitions remain null in the **index**, meaning unknown,
not a known D3D12 null descriptor. This still lacks resource identity joins,
static samplers and shader-access analysis; broad unused ranges may coexist
with genuinely missing captures. Do not treat the index as an executable ABI.

fsr_provider_probe.cpp is a deliberately small Windows/DX12 program for
observing the public AMD FFX API under Wine/Proton. It does not contain,
extract, or redistribute SDK DLLs, DXIL, SPIR-V, neural weights, or model
payloads.

It creates a DX12 device, loads a caller-selected
amd_fidelityfx_loader_dx12.dll, enumerates upscaler, frame-generation,
denoiser/Ray-Regeneration, and Radiance Caching providers, and can create FSR
4.1.1, Frame Generation 4.0.1, Ray Regeneration 1.2.0, or Radiance Caching
0.9.0 API contexts. Each creation mode queries the
selected provider after creation, so a legacy fallback cannot be mistaken for
the requested neural API version. Dispatch capture is a separate step because
it needs correctly initialized color/depth/MV resources and a command queue.
Ray Regeneration is published through the SDK's denoiser descriptor; when
compiled only against the deliberately reduced Q2RTX FSR-only source closure,
the probe clearly reports that the denoiser header is absent instead of
guessing a private type.

Build, replacing SDK with the local FidelityFX SDK 2.3 checkout:

Also set `VULKAN_HEADERS` to a Vulkan-Headers checkout's `include` directory
(in Q2RTX, `extern/Vulkan-Headers/include`). The interop diagnostic loads
Vulkan functions dynamically; no Vulkan import library is required.

    x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -Wextra -Werror \
      -Wno-unknown-pragmas \
      -I"$VULKAN_HEADERS" \
      -I"$SDK/Kits/FidelityFX/api/include" \
      -I"$SDK/Kits/FidelityFX/api/include/dx12" \
      -I"$SDK/Kits/FidelityFX/denoisers/include" \
      -I"$SDK/Kits/FidelityFX/framegeneration/include" \
      -I"$SDK/Kits/FidelityFX/radiancecache/include" \
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

The --dispatch option records and waits for four 640x360 -> 1280x720 frames
in one provider context: reset, two history-reusing frames, then reset again.
It uploads a two-colour checker pattern, constant device depth, and zero
motion/reactive/composition inputs, and enables internal auto-exposure. Output
starts as NaN sentinels and is copied to a readback buffer after dispatch. The
probe requires finite RGB everywhere, some nonzero RGB, and the correct
red/green dominance at every checker tile centre after verified fence
completion. Both 4.1.1 forced-INT8 and 3.1.5 control pass all 220 sampled
centres at 1280x720. This is a synthetic execution test, not a temporal image-quality
test or proof of the internal neural model. Use --provider-index N
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

Use `--create-framegeneration` to create the Frame Generation 4 API context
and report its selected provider. With the full SDK headers,
`--create-denoiser` does the equivalent Ray Regeneration 1.2 context probe.
With the full SDK Radiance Caching header, `--create-radiancecache` does the
equivalent 0.9.0 provider availability probe. These creation probes record
allocation metadata but intentionally do not dispatch a synthetic neural frame,
denoising workload, or radiance-cache inference/training workload. They are
provider availability tests, not image-quality tests.

Every invocation ends with one stable provider-result line for each of
upscaler, frame-generation, ray-regeneration, and radiance-caching. The fields
record whether that effect was requested, its create/query return values, and
the provider actually selected.  `4294967295` means no query was possible
(for example, creation returned `FFX_API_RETURN_NO_PROVIDER`); it is not a
provider version.  These records are intended for future compatible-adapter
revalidation scripts: require a successful create and query, then inspect the
selected name/ID rather than treating a successful request for a 4.x API as
proof that its neural provider was used.

## Historical source-v07 FSR4 provider probe

### SDK 2.3 forced-INT8 investigation (2026-09-08)

The probe now accepts experimental `--force-int8`. It loads the SDK upscaler
beside the explicitly selected loader, requires exactly one matching signature
in executable PE sections, and changes the eligibility function only in this
process. It never writes a DLL file. A failed signature/protection check stops
the probe. This flag is for isolated unsupported-GPU experiments; leave it off
for the user's RDNA4 machine so the ordinary FP8/provider path can be tested.
The MinGW build with warnings as errors passes. A 2026-09-08 RDNA2 run using
Proton Experimental vkd3d-proton `634d341a5a312a3` selected provider 4.1.1,
returned dispatch success, and passed the explicit fence/device completion
check. The deterministic checker input now produces finite, nonzero RGB across
the entire output in both forced-INT8 and ordinary FSR3 modes. This does not
establish temporal image quality or rule out internal analytical fallback.
Both modes now explicitly load the
same sibling upscaler DLL. With identical loading, the ordinary control selects
3.1.5 and the forced-INT8 run selects 4.1.1; both reach the verified GPU fence.
Thus the hook changes API provider selection independently of DLL discovery.

For RDNA4, first run without overrides using `--dispatch`, then separately
`--create-framegeneration` and (with full SDK headers) `--create-denoiser`.
Retain the complete output and selected-provider records. These experiments
test the DX12 provider; they do not yet constitute native Vulkan integration.

OptiScaler v0.9.4, commit `7534ad00bf9e590eedb99e8dd9fd8c89dae3654f`,
implements `Fsr4ForceEnableInt8` in `OptiScaler/proxies/FfxApi_Proxy.h`.
It locates an internal SDK upscaler GPU-eligibility function by signature and
detours it to return 1. This differs from the historical provider-selection
environment switches below. It does not itself translate shaders to Vulkan.

A read-only scan of the local SDK 2.3 sample upscaler DLL found exactly one
matching signature at file offset `0x8170`. Its SHA-256 is
`d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46`.
This proves the local binary contains the targeted code signature, not that
INT8 execution succeeds. No DLL was modified or executed for this scan.
Next test the hook in an isolated SDK-2.3 probe, checking internal fallback
and completed GPU execution as well as the API-selected provider. A file
offset is not a loaded-module address; any probe must account for PE sections.

Upstream source:
https://github.com/optiscaler/OptiScaler/blob/7534ad00bf9e590eedb99e8dd9fd8c89dae3654f/OptiScaler/proxies/FfxApi_Proxy.h

### Legacy probe usage

fsr4_v07_provider_probe.cpp is deliberately separate from the SDK 2.3 probe.
It targets the older public ABI used by the source-v07 FSR4 SDK and therefore
does not make claims about FSR 4.1.1. It enumerates every old upscaler provider
and explicitly creates each one, including its version override, then records
the provider returned by the loader in an
FFX_LEGACY_FSR4_PROVIDER_RESULT line.

Build it against the caller's historical SDK:

    x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-switch \
      -I"$SDK/Kits/FidelityFX/api/include" \
      -I"$SDK/Kits/FidelityFX/api/include/dx12" \
      -I"$SDK/Kits/FidelityFX/upscalers/include" \
      fsr4_v07_provider_probe.cpp -municode -static -static-libgcc \
      -static-libstdc++ -ld3d12 -ldxgi -lole32 -o fsr4_v07_provider_probe.exe

Run it from an isolated copy of the SDK's signedbin directory. To test a
community RDNA2 DXIL-compiler compatibility DLL, put only a caller-supplied
copy named amdxcffx64.dll beside that isolated loader; do not overwrite an SDK,
Proton cache, or game installation. A successful context creation only proves
this old DX12 provider ran under that exact Wine/vkd3d configuration. It does
not make the provider official, turn it into FSR 4.1.1, or provide a native
Vulkan binary path for Q2RTX.

Pass --dispatch to record one synthetic 640x360 -> 1280x720 reset frame for
only a version whose reported name begins with 4. This is a command-recording
and completion check with undefined throwaway input pixels, not an image
quality test. The known RDNA2 community switch can make the old 4.0.2 provider
enumerate and create, while this controlled command-list close presently fails
under the local Wine/vkd3d stack because it cannot expose the provider's WMMA
requirement. Keep create/query and dispatch results separate.
