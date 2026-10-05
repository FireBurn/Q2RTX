# XeSS portability investigation

The objective is native execution across GPU vendors, including recovery of
the actual model and dispatch graph. Loading Intel's Windows runtime under
Proton is a reference/capture step, not that deliverable. No XeSS renderer
backend has been added to Q2RTX yet.
The requested final runtime uses native Vulkan resources and SPIR-V, with
no DirectX, HLSL compilation, Wine/Proton or vendor DLL dependency.

## Static extraction result

The supplied OptiScaler checkout contains XeSS 3 SDK headers and Windows
runtimes. Scanning these with `../dxil_container_tool.py scan` found:

| File | SHA-256 | Valid embedded DXBC/DXIL containers |
| --- | --- | ---: |
| libxess.dll | 251659dd84a3e84de67c886a4186e01f3eca49b00641906fe38bb6b807e5d5b7 | 0 |
| libxess_dx11.dll | c7cfe86f0c9d94e4fb3696d3cd5035e2bbb6a8b1b0572f8b7395a4cdfd0c625e | 0 |
| libxess_fg.dll | ec5e0c65e075570c6ede72618bb666d0be0c2e10b2ea9762c0fe8cb8e375ab27 | 0 |

The two DXBC markers in libxess.dll failed container bounds validation.
This establishes only that this scanner cannot carve these files; it does
not establish absence of shaders, compression, encryption, or model weights.
The local manifest is `build/xess-audit/containers.json`.

## Pipeline creation probe

From the repository root:

```sh
mkdir -p build/xess-audit
x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -static -municode -I/home/fireburn/OptiScaler/external/xess/inc \
  tools/ffx_dxil/reference_harness/xess_provider_probe.cpp \
  -ld3d12 -ldxgi -ldxguid -o build/xess-audit/xess_provider_probe.exe
```

Run under Proton with `VKD3D_SHADER_DUMP_PATH` pointing to a fresh directory
and `VKD3D_SHADER_CACHE_PATH=0`. Pass the absolute Windows path to
`libxess.dll` as the only argument. Keep runtime sibling DLLs together.
The probe logs version and adapter IDs, creates a D3D12 context, and initializes
1280x720 Quality mode with default flags. It skips software adapters and
returns failure unless initialization and context destruction succeed.

Add `--dispatch EXISTING_OUTPUT_DIRECTORY` to execute four checkerboard
frames (reset, history, history, reset), saving exclusive-created
`frame-{0,1,2,3}.rgba16f` files. The input resolution is queried from XeSS;
the output is 1280x720 tightly packed little-endian RGBA16F. Depth is 0.5,
motion and jitter are zero, exposure is 1, and each output starts as NaN.
Every frame must complete its fence with a healthy device, overwrite all RGB
with finite values, and preserve red/green ordering at mapped tile centres.
The two reset outputs must be bitwise equal. Reuse the context and command
queue across frames, but wait before reusing resources. These static synthetic
checks exercise history reuse/reset, not motion reconstruction quality.

Successful initialization proves neither frame execution nor image quality.
Next steps are capture inspection, controlled frame execution/readback,
resource/model-upload and dispatch-graph recovery, and native Vulkan replay
against the reference. Cross-vendor and temporal tests must follow; translated
SPIR-V alone is not a complete implementation. See [capture requirements](../CAPTURE.md).

### Verified local result

Proton Experimental initialized this runtime on AMD PCI `1002:73df`.
`xessGetVersion` reported **2.0.2** (the enclosing SDK is branded XeSS 3).
Create, initialize and destroy each returned `XESS_RESULT_SUCCESS`.
The explicit Windows stdout capture is `build/xess-audit/probe-runtime.log`.
The first capture contains 16 DXIL/SPIR-V pairs and four root signatures;
`capture-manifest --require-pairs` accepted all pairs. All 16 SPIR-V modules
pass `spirv-val --target-env vulkan1.3 --uniform-buffer-standard-layout`.
Without the uniform-buffer-standard-layout option one module fails layout
validation, so this feature requirement must be preserved or lowered during
native porting. These modules are initialization captures, not yet a verified
complete frame graph or model extraction.

Directly invoking Proton's Wine binary without Proton's environment failed
device creation and reported a different adapter identity. Use the Proton
launcher for this reference. Its default stdout routing omitted probe output;
running through `cmd /c` with redirection to an explicit Windows log path
retained the initialization result.

The four-frame execution subsequently passed on the same adapter/runtime.
The queried Quality input size is 753x424. All four outputs contain zero
nonfinite RGB components and match all 299 tile-centre samples; the two reset
frames match bitwise. Logs and raw reference outputs are in
`build/xess-audit/dispatch-1.log` and `build/xess-audit/frames-1/`.
Model uploads and the full native replay graph remain to be recovered.

### Instrumented dispatch capture

The patched vkd3d runtime in `build/vkd3d-native411-trace-build` also passed
the four-frame probe. Capture directory: `build/xess-audit/trace-1/`.
Its `dispatch-index.json` contains 56 direct compute dispatches: the same
14 shader hashes occur once per frame. The upload hook saved 55 allocations
totalling 50,584,512 bytes, including synthetic inputs/output sentinels and
provider allocations. That byte count is **not** a model-size measurement.

Twelve descriptor entries remain unresolved: three constant-buffer views in
the final pass of each frame. The existing instrumentation captures SRV/UAV
descriptors but not CBV descriptors. Add CBV capture, then correlate buffer
addresses and sizes with uploads before attempting replay. Recorded upload
bytes still require dispatch-time/lifetime validation. `dispatch_trace.py`
correctly marks the current index `replay_ready: false`.

### Constant recovery

`vkd3d-reference-cbv.patch` adds opt-in CBV address/size capture to the
embedded descriptor path. Apply it after the existing capture patches.
`vkd3d-reference-constants.patch` records compute root constant words,
including partial writes, at the setter. The indexer snapshots these words
per dispatch and invalidates them on root-signature changes. Uncaptured
pointer-only writes are explicitly unknown instead of retaining stale values.

The rebuilt runtime passed the four-frame test in `build/xess-audit/trace-3/`.
All 56 dispatches have captured root constants and all descriptor entries
now resolve. The GPU-local CBVs are named `XeSS_i8_weights`, `XeSS_i8_scale`
and `XeSS_i8_bias`. Recording-order copy provenance links them to the saved
initialization uploads:

| Buffer | CBV range bytes | Uploaded bytes | Uncopied bytes in range |
| --- | ---: | ---: | ---: |
| weights | 512 | 320 | 192 |
| scale | 256 | 80 | 176 |
| bias | 256 | 80 | 176 |

The uncopied bytes cannot be assumed to contain zeros or useful data; shader
access analysis must determine whether they are used. Other captured buffer
uploads feed earlier neural passes. The trace contains 39 upload-to-buffer
copies totalling 253,280 bytes. Local `upload-copy-provenance.json` records
their source snapshots, destination resources, offsets and payload hashes.
`build/xess-audit/recover_upload_copies.py` reproduces this analysis for the
isolated serialized probe. It is a recording-order investigation, not a
general resource lifetime or queue submission validator.

The next native replay requirements are resource extents/lifetimes, barrier
and submission reconstruction, mapping the captured root layout into Vulkan
bindings, and intermediate-output comparisons. The first-pass SPIR-V has
120 bytes of push constants (28 application words plus two descriptor-table
indices), sampled/storage images and a sampler. The current vkd3d translation
uses descriptor indexing and aliases image types at one binding; a portable
native layout will need separate bindings or explicit feature requirements.

### Native descriptor conversion

The engine-independent tool
`extern/ffx-vulkan/tools/normalize_spirv_bindings.py` splits the aliased
descriptor variables into distinct bindings and emits the mapping required
by a native Vulkan backend. All 16 captured modules were converted and
passed SPIR-V validation in `build/xess-audit/native-bindings-1/`.
The transformation changes only DescriptorSet/Binding decorations; every
executable instruction, array index, type and push-constant offset remains
identical. The backend must mirror the appropriate captured table entries
into the separate arrays and maintain the recorded table-base indices.
GPU execution with this layout has not yet been verified. Descriptor indexing,
integer dot products and other recorded capabilities have not been lowered.

## DLSS scope

No `nvngx` runtime files were found in the supplied OptiScaler checkout.
Subsequently acquired DLSS runtimes are recorded in [DLSS.md](DLSS.md).
OptiScaler's replacement routing is not a portable implementation of DLSS.
DLSS 4/4.5 super resolution and DLSS 5 neural rendering are distinct targets.
Extracting kernels alone would not supply their model graph, weights, signal
contracts, or temporal state. Distillation would additionally require a
working reference, training data and training/validation work; it is not a
shader translation operation.

Authoritative background:

- [Intel XeSS SDK](https://github.com/intel/xess): cross-vendor paths have
  capability requirements; they do not cover literally every GPU.
- [Intel SR integration guide](https://github.com/intel/xess/blob/main/doc/xess_sr_developer_guide_english.md):
  documents the HLSL cross-vendor implementation and Vulkan requirements.
- [NVIDIA DLSS](https://developer.nvidia.com/rtx/dlss): official RTX implementation.
- [NVIDIA DLSS 5 research](https://research.nvidia.com/labs/adlr/DLSS5/):
  neural rendering is a separate reconstruction problem.
