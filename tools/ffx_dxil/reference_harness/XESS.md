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

### Native convolution replay milestone (2026-10-15)

`../xess_conv_replay.py` builds a local manifest for first-frame passes 1..12
from the serialized trace. It checks normalized shader hashes and binding/
push layouts, resolves descriptor-table slots, preserves skip connections,
and reconstructs buffers from recorded upload/copy order. It refuses missing
bytes rather than silently filling uncaptured model data with zero. This
profile excludes the prepare and final resolve shaders and is not a generic
capture replay system.

```sh
python3 tools/ffx_dxil/xess_conv_replay.py \
  build/xess-audit/trace-4 build/xess-audit/native-bindings-1 \
  build/xess-audit/conv-replay-NEW
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  -Iextern/ffx-vulkan/include \
  tools/ffx_dxil/reference_harness/native_xess_convolutions.cpp -lvulkan \
  -o build/xess-audit/native_xess_convolutions
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  build/xess-audit/native_xess_convolutions \
  build/xess-audit/conv-replay-NEW/manifest.json \
  build/xess-audit/native-prepare.rgba32ui \
  build/xess-audit/conv-replay-NEW/native
```

The harness additionally needs the `nlohmann/json.hpp` development header.
Both output directories must be new. It binds 36 recovered weight/scale/bias
buffers (252800 bytes), rebases two descriptor tables to six slots, and
executes twelve captured convolution shaders with compute barriers and
per-layer readbacks. Shader instructions and types are preserved. Native
execution requires Vulkan 1.3, runtime descriptor arrays, buffer device
address, shaderInt8, integer dot product, dynamic descriptor indexing and
formatless storage-image writes; there is no Wine or vendor DLL dependency
in the native executable. These requirements are not yet lowered for older
GPUs.

Renoir and RX 6800M/Navi22 runs completed with Vulkan validation enabled,
no validation messages and zero untouched sentinel words in all twelve
outputs. Evidence: `build/xess-audit/conv-replay-1/native.log` and
`conv-replay-2/navi22.log`. The `--zero-weights` control preserves scales/biases
but zeros convolution weights; every layer changes, including 2987413 of
3686400 output bytes in pass 12. This establishes weight dependence, not
reference equivalence. `conv-replay-1/weight-sensitivity.json` retains hashes.
Cross-device comparison found identical passes 1..10, then 11 changed bytes
in pass 11 and 146 in pass 12, each differing by one quantized step.
`conv-replay-2/cross-device-comparison.json` records these differences.

`native_xess_prepare` now accepts an optional fourth data-path argument to
save its 1280x720 prepared-colour output as RGBA16F. The packed feature output
is unchanged bitwise after the shared allocation-helper refactor. The native
prepare, convolution and zero-control experiments still report
`reference_match=unverified`. Next: replay final resolve with the three CBVs
(including explicit handling of their uncopied padding), compare the full
output to reference frame 0, then implement temporal history and motion cases.

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

Set `XESS_PROBE_INTERNALS=1` to request `xessStartDump` for all resources
before executing, then query `xessD3D12GetResourcesToDump` after each dispatch.
Use a fresh output directory: the SDK dump API may overwrite its own files.
The probe logs dump status separately from frame execution. On Oct 15,
runtime 2.0.2 returned `XESS_RESULT_ERROR_NOT_IMPLEMENTED` (-7) from
`xessStartDump` and zero internal resources for all four frames. Execution,
299/299 tile checks per frame, and reset-repeat equality still passed.
Evidence: `build/xess-audit/internal-dump-2/probe.log` and `frames/`.
This runtime cannot supply intermediate reference images through that debug
API; use instrumented resource readback or full native-graph comparison.

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
identical. The backend mirrors the captured table entries into the separate
arrays and maintains the recorded table-base indices.

### First native prepare-pass executed

`tools/ffx_dxil/reference_harness/native_xess_prepare.cpp` is a standalone Linux
Vulkan test for the prepare pass (`40553fb1ea6dd6f3.spv`, (80,23,1) dispatch).
It accepts the normalized SPIR-V module, the 120-byte push-constant binary
(28 root constants plus two table-base words), and writes the 640x360
RGBA32UI packed-features output.

Execution verified on AMD discrete GPU (Radeon RX 6800M, Navi 22) and integrated
GPU (Renoir) under Linux:
- `VK_LAYER_KHRONOS_validation` enabled: 0 errors, 0 warnings.
- Output buffer initialized to `0xdeadbeef`: all 921,600 sentinel words were
  overwritten by the compute shader (`sentinel_words=0`).
- Generated packed INT8 weights (`0xb6b6b6b6`) and zero-point biases (`0x80808080`)
  match neural input formatting.

### U-Net topology and neural weights bundle

Analysis of all 14 dispatches in `dispatch-index.json` and upload copies in
`upload-copy-provenance.json` revealed the full encoder-decoder U-Net
convolutional architecture:

- **14 Dispatches per Frame:**
  - Dispatch 00: prepare input pass (`753x424` -> `640x360x1` feature tensor)
  - Dispatches 01-04: encoder downsampling path (`640x360` -> `320x180` -> `160x90` -> `80x45` -> `40x23x8`)
  - Dispatch 05: bottleneck convolution layer (`40x23x8` -> `40x23x4`)
  - Dispatches 06-12: decoder upsampling path with residual skip connections (`40x23` -> `80x45` -> `160x90` -> `320x180` -> `640x360`)
  - Dispatch 13: final resolve pass (`640x360` neural features -> `1280x720` RGBA16F)

- **13 Neural Weight Layers (253,280 bytes total):**
  Each layer consists of quantized INT8 weights, per-channel FP32 scale, and
  per-channel FP32 bias:
  `XeSS_i10` (2432 B), `XeSS_i12` (4864 B), `XeSS_i13` (18944 B), `XeSS_i14` (74752 B),
  `XeSS_i15` (74240 B), `XeSS_i16` (37376 B), `XeSS_i17` (18688 B), `XeSS_i18` (9472 B),
  `XeSS_i19` (4736 B), `XeSS_i20` (2432 B), `XeSS_i21` (2432 B), `XeSS_i23` (2432 B),
  and resolve layer `XeSS_i8` (480 B).

- **Tooling:**
  `tools/ffx_dxil/xess_model_tool.py` parses provenance records, validates U-Net
  topology, extracts all 13 layers, and bundles them into an aligned binary container
  (`XESSMOD2`) with 256-byte offset alignment suitable for Vulkan storage buffers.
  Verified with 34 tests in `tools/ffx_dxil/tests/test_xess_model_tool.py`.

## Complete native fixed-profile replay

`native_xess_resolve.cpp`, `xess_resolve_replay.py` and
`replay_xess_sequence.py` complete the 14-pass graph and feed resolved history
and mask into subsequent prepare passes. Four reset/history/history/reset
frames match their same-device reference RGBA16F images byte-for-byte on both
RX 6800M (Navi22) and integrated Renoir. Vulkan validation is enabled for each
native stage. Captured final CBV prefixes are preserved and uncaptured padding
is explicit; changing padding to 0xff preserves all four Navi22 matches.

Example (use a new output directory):

```sh
python3 tools/ffx_dxil/replay_xess_sequence.py \
  build/xess-audit/trace-4 build/xess-audit/native-bindings-1 \
  build/xess-audit build/xess-audit/native-sequence-new \
  --reference-frames build/xess-audit/reference-renoir-2/frames
```

The reference must match the selected GPU and input profile. Cross-device
outputs differ slightly even in the reference runtime; same-device comparisons
use exact equality, without relaxing error thresholds. This is a fixed
753x424-input/1280x720-output checkerboard experiment with host readbacks
between stages. Moving scenes, jitter, other quality/resolution profiles and
GPU-resident engine integration remain unverified. The Q2RTX spatial prototype
does not yet use this recovered graph. Native replay has no Wine, DirectX or
vendor DLL dependency; those are used only to obtain offline references.

### External input replay

Prepare and resolve accept `XESS_REPLAY_INPUTS`, a directory containing tightly
packed `color.rgba16f`, `motion.rg16f` and `depth.r32f` at 753x424. All three
files are required and exact byte sizes are checked. Without this setting,
the original synthetic checkerboard is used. The sequence runner's `--inputs`
selects a parent directory containing `frame-0` through `frame-3` and records
input SHA-256 hashes in its comparison report. Supply matching captured push
constants and reference frames: changing images alone does not update jitter,
motion conventions or reset metadata. This path enables moving-input research;
it does not by itself establish moving-scene equivalence.

### Translating input validation

Set `XESS_PROBE_MOVING=1` for the reference probe to translate the checkerboard
left by 0/8/16/0 input pixels. History frames carry +8 horizontal input-pixel
motion, and the final frame resets. The probe saves its exact input images in
`frame-N/` beside the output readbacks. Feed that directory to both `--inputs`
and `--reference-frames`. The existing zero-jitter trace constants apply because
only image contents change. On Renoir all four native outputs match the
same-device reference byte-for-byte with Vulkan validation enabled. Zeroing
motion while retaining translated color changes both history frames (maximum
RGB error about 0.33), while reset frames remain exact. These are controlled
translation results; jitter, disocclusion and general scene coverage remain.

`XESS_PROBE_JITTER=1` additionally varies jitter metadata across the four
frames: (0.25,-0.25), (-0.25,0.25), (0.125,0.375), then (0.25,-0.25) with
reset. Capture fresh constants rather than reusing the zero-jitter trace.
The instrumented `trace-jitter-1` and native `native-sequence-jitter-1` match
all four Renoir frames exactly with Vulkan validation. Current jitter occupies
prepare/resolve words 4/5; prepare words 22/23 hold previous jitter. This probe
does not resample checkerboard pixels at subpixel positions, so this validates
the graph's response to jitter metadata rather than camera sampling quality.

### Generate frame constants from engine-style parameters

Build `xess_frame_constants.cpp` with `-Iextern/ffx-vulkan/include` and place
the executable alongside the native replay binaries. The experimental shared
header `ffx_vk_xess_replay_constants.h` supports only the captured fixed-size,
unit-exposure, zero-origin model profile. Pass `--frame-parameters frames.json`
to the sequence runner, where the JSON is four objects of this form:
`{"jitter":[0.25,-0.25],"reset":true}`. Previous jitter is retained across
frames, including reset; initial previous jitter is zero. Nonfinite or out-of-
range jitter is rejected. The builder's static fields are not claimed to be a
general provider ABI. All four jittered outputs match the reference when the
model trace is the original zero-jitter trace and constants are generated this
way (`native-sequence-generated-constants-1`). Captured frame constants are
therefore unnecessary for this profile; model data still comes from the trace.

### Relocatable native bundle

Package a successful local replay using:

```sh
python3 tools/ffx_dxil/xess_native_bundle.py REPLAY_DIRECTORY NORMALIZED_SHADERS NEW_BUNDLE
```

The bundle contains shader/model payloads and must remain a local build
artifact. Its loader verifies asset hashes and rejects paths outside the bundle.
Use `replay_xess_sequence.py` with `--bundle NEW_BUNDLE`, `--frame-parameters`,
`--inputs` and `--reference-frames`. Trace and shader positional paths are
unused in bundle mode. A relocated bundle with nonexistent trace/shader paths
matches all four moving/jitter reference frames on Renoir under validation
(`native-sequence-bundle-1`). This eliminates runtime trace dependence, while
host readbacks and the fixed profile remain limitations of the harness.

### Shared convolution command recorder

`ffx_vk_xess_convolution_graph.h` records the twelve passes on caller-owned
Vulkan handles, with compute barriers and no intermediate readbacks. The
convolution executable's `--resident-repeat` mode uses it and submits twice
with the same model and pipeline resources. Diagnostic tensor copies happen
after the complete graph. All twelve outputs match the original path; the
sequence runner's `--resident-convolutions` retains exact reference equality
for all moving/jitter frames under validation. Input visibility, image layouts,
resource lifetime and external-stage barriers remain caller responsibilities.
This recorder does not yet own resources or include prepare/resolve/history.

### Persistent fourteen-pass graph

`native_xess_graph.cpp` keeps all pipelines, weights, tensors and history on
one device for the whole sequence. Each frame uses one command buffer and
submission; only input uploads and final output readback involve the host.

```sh
cmake -S extern/ffx-vulkan -B build/ffx-vulkan-project
cmake --build build/ffx-vulkan-project --target ffx_vulkan_xess_recovered
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  -Iextern/ffx-vulkan/include \
  tools/ffx_dxil/reference_harness/native_xess_graph.cpp \
  build/ffx-vulkan-project/libffx_vulkan_xess_recovered.a -lvulkan \
  -o build/xess-audit/native_xess_graph
python3 tools/ffx_dxil/replay_xess_graph.py BUNDLE \
  build/xess-audit/native_xess_graph FRAMES_JSON INPUTS REFERENCE NEW_OUTPUT
```

The wrapper checks bundle hashes, enables synchronization validation and
requires exact RGBA16F reference equality. Four moving/jittered frames on
Renoir and four static frames on Navi22 pass exactly with clean validation
logs (`persistent-graph-2`, `persistent-graph-navi22-1`). This supersedes the
intermediate-readback limitation of the older three-executable sequence runner.
It remains fixed-profile experimental code with process-owned device lifetime;
it is not yet the caller-owned library context or a Q2RTX backend.

### Shared resource ownership

The persistent graph uses `ffx_vk_xess_resources.hpp` for all images/buffers.
Its C++ resource owner takes a borrowed physical/logical device pair; it never
submits, waits or destroys the device. Retire commands before clearing or
destroying it. Independent instances and clear/reuse are exercised by
`xess_resources_probe.cpp`, compiled with `-Iextern/ffx-vulkan/include -lvulkan`.
The full four-frame moving/jitter sequence remains exact with synchronization
validation (`persistent-shared-resources-1`). Pipeline/descriptor ownership
and complete graph lifecycle are still harness code, not a public SR context.

The resource owner now also creates and destroys compute pipelines, shader
modules, descriptor layouts/pools/sets and samplers. Shader words are supplied
in memory; the shared component does not read JSON or files. Partially created
stages remain tracked for cleanup. The harness uses it for all fourteen stages;
Renoir moving/jitter and Navi22 static sequences remain exact with clean
synchronization validation (`persistent-shared-stages-1` and
`persistent-shared-stages-navi22-1`). Graph construction, descriptor bindings
and frame scheduling remain harness responsibilities pending context extraction.

### Shared complete context

The persistent harness now calls `ffxvk::XeSSFixedContext` from
`ffx_vk_xess_fixed_context.hpp`. Model bytes and shader words enter through
typed structures; the library owns graph construction, weights, intermediate
images, history, descriptors, pipelines and dispatch constants. Caller-owned
input/output images are bound explicitly. The context records into a caller
command buffer and performs no filesystem access, submission, wait or readback.
The bundle/JSON loader remains harness code. See the reusable
[integration example](../../../extern/ffx-vulkan/examples/xess-recovered/README.md).

`shared-context-final-1` and `shared-context-final-navi22-1` match all four
same-device reference frames exactly with clean synchronization validation.
The live harness alternates two image sets, rejects invalid extent/jitter,
requires bound images and initial reset, and recovers from discarded commands.
It releases CPU model data immediately after context creation. Captured address
words are replaced by zero; fixed dispatch sizes/constants are generated inside
the context. An installed-headers-only rebuild also matches every reference
frame (`shared-context-installed-1`). Q2RTX's spatial prototype is not yet
connected to this fixed-size context, and broader profiles remain unverified.

### Multiple quality extents

The shared context accepts `FfxVkXeSSReplayExtent` at construction. Enabled
input/output pairs are 565x318 -> 960x540, 753x424 -> 1280x720 and
1130x636 -> 1920x1080. Each tensor downsampling level rounds up; dispatch
counts and frame constants derive from that extent. Other extents are rejected.
Recreate the context after retiring its work to change size. The original
constructor/constant helper keep their 720p defaults.

Set `XESS_PROBE_WIDTH` and `XESS_PROBE_HEIGHT` (integer [64,4096]) on the offline
probe to capture another output size. `xess_geometry_probe.cpp` emits generated
tensor/dispatch geometry and four fixture constant blocks for independent
comparison against the trace. All generated values match the three captured
traces (`geometry-comparison.json`). Both new sizes use the same shaders and
original native model bundle, with no additional recovered weight set.

The persistent wrapper accepts, for example, `--extent 1130 636 1920 1080`.
Renoir moving/jitter comparisons pass exactly at 540p and 1080p
(`shared-context-960x540-1`, `shared-context-1920x1080-1`); the post-change 720p
regression also passes. A fresh RX 6800M 1080p capture and native comparison
pass exactly (`shared-context-1920x1080-navi22-1`). Synchronization validation
and live binding/reset/discard contracts remain clean. A staged installed-header
rebuild also passes at 540p (`shared-context-multi-installed-1`). These are
controlled checkerboard fixtures; broader scenes and quality modes remain open.

### Variable depth and newly exposed pixels

`XESS_PROBE_DEPTH_SCENE=1` selects a near blue rectangle moving right by
0/32/64/0 input pixels, over a static checkerboard with varying far depth.
Foreground depth is 0.25; background ranges from 0.75 toward 0.875. History
frames supply -32-pixel foreground motion and zero background motion. Inputs
are saved exactly; the final frame resets to initial inputs. Combine with
`XESS_PROBE_JITTER=1` for varying jitter metadata.

At 1080p the input contains 1131 distinct depths and exposes 10176 background
pixels on each history frame. Four native outputs match every reference pixel
exactly on both Renoir and RX 6800M, with clean synchronization validation:
`shared-context-depth-scene-renoir-1` and `shared-context-depth-scene-navi22-1`.
The reference's stable-checker assertions avoid the foreground footprint;
the native comparison still covers the full image. A flat-depth control
preserves reset frames but changes both history frames, confirming depth
affects temporal reconstruction. This remains a controlled analytic fixture;
physical camera jitter and general scene quality remain unverified.

### Frames in flight

The context's optional frame-slot count defaults to 1 and accepts 1..8.
`bind(images, slot)` and `record(cmd, frame, slot)` use independent mutable
prepare/resolve descriptor sets for the chosen slot. Pipelines, model buffers
and ordered temporal resources remain shared. Retire that slot before rebinding;
retire all uses before destruction or history discard. Submit frames once, in
order, on the same queue. No queue wait happens inside the library.

The harness now records both slot-0 and slot-1 command buffers before submitting
either, then submits a pair and waits once. This explicitly checks descriptor
updates cannot invalidate a previously recorded frame. Four moving-depth/jitter
frames remain byte-identical to the same-device references on both GPUs, with
clean synchronization validation (`shared-context-two-flight-renoir-1`,
`shared-context-two-flight-navi22-1`). Invalid-slot and all previous context
contracts remain enabled. This addresses one Q2RTX integration prerequisite;
engine input conversion, asset loading and actual dispatch wiring remain.

### C interface and one-file model assets

The harness now creates, binds, records, discards history and destroys through
the opaque C API in `ffx_vk_xess_recovered.h`. The exported
`ffx-vulkan::xess-recovered` target carries the implementation. Bounded diagnostics,
ABI/blob validation and exception-to-VkResult translation keep C++ internals
inside the library. The copied installed-package consumer compiles its ABI
checks as C.

`pack_xess_model.py BUNDLE OUTPUT.xessvk` packs fourteen shaders and thirty-nine
buffers into an XESSVK1 file. `ffxVkXeSSRecoveredModelFromBlob` checks table bounds
and CRCs, returning borrowed model spans; creation copies all payloads. The
engine can release the file after creation. See the shared
[format documentation](../../../extern/ffx-vulkan/examples/xess-recovered/MODEL_FORMAT.md).

Use `replay_xess_graph.py /nonexistent-bundle BINARY FRAMES INPUTS REFERENCE OUT`
with `--packed-model FILE` and the usual `--extent` to verify independence from
bundle directories. `shared-context-c-blob-renoir-1` and
`shared-context-c-blob-navi22-1` retain exact four-frame depth-scene output with
clean synchronization validation. The harness also rejects truncated model
buffers, accepts unaligned shader storage, and releases source bytes before
recording. A C container test covers corruption, overflow, overlap, truncation,
wrong count, reserved fields and trailing bytes. Its synthetic entries test
only parsing; full model execution is covered by the live comparisons.

### Live Q2RTX engine integration (2026-10-27)

`src/refresh/vkpt/fsr.c` and `src/refresh/vkpt/main.c` now integrate
`FfxVkXeSSRecoveredContext` into Q2RTX (`flt_upscaler 4`):
- Enables `shaderStorageImageReadWithoutFormat`, `uniformBufferStandardLayout`,
  `shaderUniformBufferArrayNonUniformIndexing`, and
  `shaderStorageImageArrayNonUniformIndexing` on `qvk.device`.
- Loads `baseq2/xess/xess-quality-1.xessvk` (`XESSVK1` container, git-ignored)
  via `ffxVkXeSSRecoveredModelFromBlob` and creates a 2-slot context
  (`MAX_FRAMES_IN_FLIGHT = 2`).
- Uses `ffxVkXeSSRecoveredGetExtent` to size Q2RTX's render extent (`566x318`,
  `754x424`, `1130x636` even checkerboard width, cropped to `565x318`,
  `753x424`, `1130x636` valid input extent) and `ffxVkXeSSRecoveredBindConverted`
  to convert Q2RTX scaled HDR color (`value_scale = 1/128`) and UV motion
  vectors (`motionScale = {renderWidth, renderHeight}`) into model units.
- Live runs with `+set vk_validation 1` on AMD Radeon RX 6800M (`1002:73df`)
  complete with zero validation errors at `960x540` (`1.260 ms`), `1280x720`
  (`1.692 ms`), and `1920x1080`, and fall back cleanly to the spatial prototype
  outside Quality `540p`/`720p`/`1080p`.

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
