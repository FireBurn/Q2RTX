# FidelityFX Vulkan handover

Last updated: 2026-08-19, Europe/London.  Update this file at every meaningful
milestone and immediately before ending or transferring the session.

## Objective and truth status

The user asked for FSR3 and FSR4 plus all related features in Q2RTX, with
reusable native-Vulkan components and a demonstrable Vulkan implementation.

Current truth:

- Existing renderer fallback builds and runs.
- The reusable tree now contains a separate, pinned public SDK v2.3.0 FSR3.1.5
  source closure (`upstream/ffx-2.3.0`) with both pristine and current
  SHA-256 manifests (103 source files). The new host source compiles as an
  object-only Linux scaffold after narrowly disabling the unpublished watermark,
  making the public DLL-export macro portable, and expanding opaque context
  storage for four-byte `wchar_t`. Its always-on host-graph test records a
  reset and a temporal-RCAS frame against a mock interface, proving 19
  persistent resources, four initialization uploads, 11 live pipelines, and
  the 27/21-job graphs. All 11 generated public modules now create verified
  Vulkan compute pipelines and descriptor sets on the RX 6800M. A new
  SDK-2.3 pipeline callback bridge maps the scheduler's pass names to the
  checked embedded modules and feeds their actual reflected SRV/UAV/CBV slots
  back into `FfxPipelineState`; all 11 callbacks create and destroy on the
  RX 6800M. The same bridge now owns SDK-created Vulkan buffers/images and
  creates full SRV plus mip-level UAV views; a real mipmapped storage-image
  allocation/export/destroy test passes. Its ordered copy-job layer now stages
  and uploads the SDK's four initialized resources; a 3.1.5 host context
  creates 19 persistent resources, records/submits the uploads, and destroys
  cleanly on RX 6800M. The bridge now also stages the three per-frame constant
  buffers, translates clear/copy/transition/UAV/discard jobs, builds reflected
  descriptor sets, and records complete 64x64 -> 128x128 reset and temporal
  sharpened frames using bridge-owned stand-ins for all nine
  application/shared images. Both command buffers submit and complete on RX
  6800M. Khronos validation is enabled and clean after explicitly annotating
  all generated storage images to match the public resource contract (R8,
  RGBA8, R16, RG16, and RGBA16F as appropriate). The test now seeds
  deterministic HDR/depth/motion/mask inputs, initializes the 128x128 RGBA16F
  output with a half-float poison value, and copies the temporal-RCAS result
  back through Vulkan. All 49,152 RGB components are finite, nonzero, and no
  component retains poison; Khronos validation remains clean. This is a real
  graph-and-output invariant, not a visual-quality claim: it retains
  per-dispatch descriptors/constants until context destruction, cannot import
  application VkImages, and is not linked into Q2RTX. It must not be mixed
  with the working 1.1.4 backend.
  The bridge now exports a borrowed native `VkImage` for an owned API-resource
  token, solely for explicit diagnostics/readback and synchronization; the
  smoke test uses it for the output invariant above.
- The first SDK-2.3 Vulkan shader milestone is also present: ten public FSR3
  3.1.5 HLSL pass wrappers plus AccumulateSharpen (11 modules) are generated
  as a fixed Q2-compatible linear-HDR,
  low-resolution, unjittered-motion, conventional-depth SPIR-V profile under
  `generated/ffx-2.3.0/vk/fsr3upscaler-q2-v2`. A pinned DXC generator packs
  SRV/UAV/CBV bindings per pass, retains AMD's proven static sampler range,
  rejects descriptor collisions, and produces a checked manifest. Reflection,
  Vulkan 1.2 validation, a byte-for-byte checked embedded-C bundle, and a real
  11/11 compute-pipeline creation test are part of the standalone suite. These
  modules are not connected to a
  resource/job backend or Q2RTX yet; generic profile/wave/FP16 permutations
  remain unfinished.
- `extern/ffx-vulkan` now has a small tested C SPIR-V reflection helper for
  that generated profile. It consumes `OpName`/`Binding` directly from the
  shader bytes and returns sorted SRV/UAV/sampler/CBV descriptors; a short
  output array is rejected without a partial write. The forthcoming backend
  must use it instead of reproducing a DX12 root-signature table by hand.
- The reusable 3.1.5 descriptor bridge is now implemented and tested. It uses
  those reflected names to create an exact descriptor pool/set, writes every
  SRV/UAV/CBV, leaves immutable samplers in the layout, and rejects absent,
  duplicated, or unexpected resources. The 11/11 RX 6800M pipeline test uses
  real image views and uniform buffers; it is used by the recorded reset-frame
  graph described above.
- The next SDK-2.3 bridge layer has begun: `ffx-vulkan::fsr3-vk-backend-3.1.5`
  owns the host scheduler plus a callback that constructs the exact embedded
  pipeline selected by the host, keeps immutable samplers in its layout, and
  returns reflected binding names/slots for the host's authoritative resource
  patch-up. Its standalone GPU test passes all eleven scheduler pipeline names.
  It now owns persistent SDK-created buffers/images and their image views, and
  its ordered copy queue initializes the four host resources requiring data.
  It stages per-frame UBOs and records the complete ordered scheduler job
  reset and temporal-sharpened scheduler graphs (clear/copy/barrier/discard/
  compute) against bridge-owned images. Its
  temporary descriptor/UBO lifetime is deliberately conservative until an
  embedding fence can reclaim it. Its deterministic temporal output readback
  has a full RGB finite/nonzero/no-poison invariant with a clean validation-layer
  run. It still has no application-resource import or Q2RTX integration.
- The generated profile now has a tested pass/permutation catalogue. It maps
  the SDK's exact base permutation (Lanczos + HDR + low-resolution motion) and
  the separate `ACCUMULATE_SHARPEN` permutation to checked module names, and
  rejects wave64/FP16/jittered/inverted/other unsupported profiles explicitly.
- The old FSR4 prototype was never correct.  Its primary failures were fixed
  rather than hidden with per-frame history resets.
- The rewritten v07 INT8/DOT4 FSR4 model family has passed a live
  1280x720 -> 2560x1440 visual test on the RX 6800M with Vulkan validation,
  explicit exposure, and sharpening on or off.  A 0.50 RCAS live validation
  run at 640x360 -> 1280x720 has no Vulkan-validation messages. Sustained forward/turning motion was
  coherent; the old quadrant boundary, black output, and history train were not
  present.  Repeat GPU-assisted validation is clean at that target resolution.
  This is a working experimental 4.0.2-era model path, **not yet an FSR 4.1.1
  implementation** and not yet a production-quality feature.
- The FSR4 backend now fails closed on missing pipelines and descriptor,
  registration, scheduling, or execution failures; uses the reflected
  104-byte binding-43 UBO without the obsolete push-constant ABI; unwinds
  partial allocations; and has idempotent teardown.  It remains a bespoke
  provider backend with fixed limits, not AMD's current binary provider.
- FFX clear jobs now preserve their requested float colour. The sampled
  explicit-exposure fallback is initialized to 1.0 as required by the shader,
  instead of being accidentally zeroed along with temporal history. The
  correction passed a fresh RCAS-on Vulkan-validation run.
- FSR4 v07 now has a complete, optional SPD auto-exposure resource graph:
  2×1 current/previous exposure, R32_UINT atomic counter, R32F mip-5, exact
  32-byte constants, and 64×64 workgroup scheduling. It is exposed as the
  default-off `flt_fsr4_auto_exposure` experiment, was live-validated with
  RCAS on, and restores a sampled 1.0 identity when disabled.
- FSR3 has a reusable ABI, the pinned AMD FSR3 1.1.4 host runtime, native
  Vulkan compute backend, generated permutations, and an opaque public C
  create/record/destroy API.  A public-API-only live RX 6800M smoke owns its
  backend/scratch/shared resources and performs two real 64x64 -> 128x128
  dispatch/readbacks.  Both frames pass full-output invariants with zero
  validation warnings/errors.  The low-level backend also passed 25
  consecutive runs.  It is now integrated into Q2RTX as the selectable native
  Vulkan `flt_upscaler 1` path with its discrete quality surface. A live
  RX 6800M 640x360 -> 1280x720
  `base1` run is visually coherent and has no Vulkan-validation messages.
  Proof: `/home/fireburn/Screenshot_FSR3_Q2RTX_20260819.png`.
- The temporal contract now exposes both positive-forward primary view-Z and
  conventional finite zero-to-one device depth, plus corrected positive
  vertical FOV and one-inch-to-metre camera scaling.  Ray-query and ray-pipeline
  producer stages are distinguished and all rounded-up ray-query launches have
  bounds guards; both paths validate, with an explicit ray-query GPU-AV run
  clean on the RX 6800M.
- Analytical Frame Generation compute is implemented and live-validated in the
  reusable Vulkan module. Q2RTX now has an experimental active two-acquire,
  generated-then-real single-queue presentation path. Read-only rolling
  rendered/generated presentation cadence is published only after successful
  interpolated-generated→real pairs and resets on fallback; explicit WSI pacing, Ray
  Regeneration, and neural frame generation remain incomplete.
- Frame generation's HDR flag now matches its actual post-tone-map HUDless
  `TAA_OUTPUT` source (the upscaler independently remains pre-tone-map HDR).
  Generic temporal resets plus acquire/present/interpolation fallbacks reset
  the separate FI/OF history before generation resumes. A fresh 36-second
  1280x720 RX 6800M generated→real run was active with Vulkan validation and
  no VUID/error output.
- `flt_frame_generation_min_rendered_fps` defaults to 30 FPS and makes the
  experimental presenter pause after four below-threshold completed frames;
  eight samples at least 2 FPS above the threshold resume it. `0` explicitly
  disables the gate for R&D and 60 reflects AMD's recommendation. Both the
  open-gate and deliberately impossible 240-FPS fallback branches were live
  validated at 1280x720 without a VUID; the local archived test setting was
  restored to the 30-FPS default.
- DXIL extraction/capture tooling is complete and tested; no extracted payload
  is checked into the repository.
- The source-only official SDK 2.3 DX12 probe now cross-compiles under MinGW
  and runs through Wine/vkd3d-proton. On the RX 6800M, the API advertised only
  analytical upscaler providers 3.1.5 and 2.3.4; creating a 4.1.1 API context
  selected 3.1.5 and one 640x360 -> 1280x720 dispatch succeeded. It generated
  11 paired DXIL/SPIR-V capture artifacts and recorded 25 provider-owned D3D12
  resource allocations outside the repo. This is direct evidence that the
  current official path falls back to analytical FSR3 on RDNA2; it is not
  neural FSR4. See tools/ffx_dxil/reference_harness/.
- Following that measured resource query, Q2RTX's FSR3 context now enables its
  internal auto-exposure graph rather than passing a null external exposure
  image. A fresh 28-second 1280x720 RX 6800M Vulkan-validation run recreated
  the 640x480 and 1280x720 contexts, activated FSR3, and emitted no VUID/error.
- The invalid startup transition of every unacquired swapchain image is fixed;
  a fresh non-FSR live validation run renders and presents with no messages.
- Render-complete binary semaphores are now allocated per swapchain image (and
  GPU), rather than per frame slot.  This fixes
  `VUID-vkQueueSubmit-pSignalSemaphores-00067`: a present operation may retain
  its semaphore longer than the frame fence.  The correction is prerequisite
  infrastructure for frame generation's multi-present scheduler.
- The source video menu and `doc/client.md` now describe the current path
  honestly: source-v07 discrete model ratios plus an explicit DRS graph,
  independently controlled
  RCAS, fallback
  TAA/TAAU, and explicitly not FSR 4.1.1.  The installed user
  `q2rtx_media.pkz` can still override the source menu until release assets are
  repackaged. On this development machine the installed user archive was still
  stale, so the verified source `baseq2/q2rtx.menu` is also deployed as the
  higher-priority loose `/home/fireburn/.local/share/quake2rtx/baseq2/q2rtx.menu`
  override (SHA-256 `a66b0744ec168e289a7845b7e0ffbb27e1b0cb4939e5ffe941250170f9008a86`).
- Both FSR3 and FSR4 v07 expose Native AA (100%), Quality (67%), Balanced
  (59%), Performance (50%), and Ultra Performance (33%) through
  `flt_fsr_quality`. FSR4 selects a complete matching model/initializer/weight
  set for each. A Quality live validation run at 858x482 -> 1280x720 reported
  `FSR4 v07 INT8/DOT4 Quality active` with no VUIDs.
- Fresh 1280x720 RX 6800M Vulkan-validation runs also selected and activated
  `Native AA` and `Balanced` FSR4 v07 model graphs with no VUID/error output.
- The temporal adapter now validates the current contract immediately before
  FSR3, FSR4, or analytical-FG imports it. It rejects ABI/stage/extent/image
  producer-state, required-slot, camera/motion, or unavailable-depth mistakes
  with a concrete console reason instead of dispatching undefined data. A
  live Quality FSR4 and FSR3+analytical-FG runs passed this validator and
  Vulkan validation cleanly.
- The temporal contract is now version 3 and carries two dense R8 FSR3
  history-control inputs. `TEMPORAL_REACTIVE_MASK` and
  `TEMPORAL_COMPOSITION_MASK` are written at the primary surface: transparent,
  water/glass, warped, screen/camera paths set both; view-model pixels set
  reactive. Sky and ordinary opaque geometry remain zero. A new RX 6800M
  1280x720 validation run recreated FSR3 from the startup 640x480 context,
  activated it with both masks registered, and logged no VUID/error output.
- Contract version 3 additionally publishes dense `TEMPORAL_NORMALS`,
  `TEMPORAL_ALBEDO`, and `TEMPORAL_ROUGHNESS` from checkerboard interleave.
  They are geometric linear XYZ normals, primary material albedo, and scalar
  roughness—useful reusable inputs for a future RR preparation layer, not an
  assertion of the official RR ABI. A fresh 1280x720 FSR3 live run with these
  allocations/writes active was Vulkan-validation clean.
- FSR4 v07 also exposes its distinct DRS graph with
  `flt_fsr4_dynamic_resolution`. It returns render-scale ownership to Q2RTX's
  existing profiler-driven bounded DRS controller instead of dispatching a
  fixed model at an arbitrary ratio. A 50%-bounded RX 6800M live run reported
  `FSR4 v07 INT8/DOT4 DRS model active` with no VUIDs. The DRS model now keeps
  its display-sized recurrent history when render-size change is the *only*
  temporal reset reason; all cuts/settings/projection/display/presentation
  invalidations still reset it.
- FSR4 DRS clamps Q2RTX's legacy 25–200% DRS/viewsize controls to a maximum
  100% input scale, including its feedback value. Supersampling remains valid
  for the old TAA path but is invalid for an upscaler input.  The fallback to
  `viewsize` occurs before this clamp (the initial controller scale is zero),
  preventing a 0x0 input at startup. A live `viewsize 150`, DRS-disabled
  1280x720 run resolved to exactly 1280x720 -> 1280x720, activated FSR4 DRS,
  and produced no validation errors.
- `ffx-vulkan::fsr4-v07-assets` is now a reusable dependency-free C target
  which selects one coherent generated v07 graph, tier, initializer, pass-0
  weights, RCAS, and SPD asset set without filesystem/Vulkan policy. Q2RTX uses
  this exact selector too; the standalone suite has 16 passing tests and a
  post-refactor RX 6800M Ultra Performance live run was validation-clean.
- FSR3's built-in RCAS is exposed as `flt_fsr3_sharpening` [0,1], default off,
  and FSR4 v07's separately scheduled RCAS is exposed as
  `flt_fsr4_sharpening` [0,1], default off.  They are independent from the
  deprecated FSR1 slider.  The FSR4 0.50 setting passed a 640x360 -> 1280x720
  live Vulkan-validation run on the RX 6800M. A desktop capture of its
  sustained moving `base1` run is
  `/home/fireburn/Screenshot_FSR4_RCAS_20260819.png`; it has no old extent
  rectangle, black output, or history train.
- A successful temporal provider dispatch now preserves temporal validity even
  when `flt_enable 0`; this prevents FSR3/FSR4 from needlessly resetting every
  frame in the composited/no-ASVGF mode. A 28-second FSR3 Performance + RCAS
  source-tree run with the denoiser disabled stayed active with no validation
  message.
- `flt_upscaler_active` and `flt_upscaler_reason` are read-only resolver
  diagnostics.  A live run reports `FSR3 3.1.4 native Vulkan active`; a
  non-rectilinear projection reports its exact fallback reason.  The menu now
  has accurate selectable providers, FSR3 quality, and RCAS controls, but it
  cannot yet render the dynamic reason string inline.

## Environment

- Repository: `/home/fireburn/Q2RTX`
- Branch: `master`
- Starting HEAD: `3b81f570` (`Add FSR4`)
- Pre-FSR4 baseline parent: `f2526e9a`
- Primary GPU: RX 6800M, RADV NAVI22 (RDNA2)
- Build directory: `/home/fireburn/Q2RTX/build`
- Old source-bearing FSR4 tree:
  `/home/fireburn/FidelityFX-SDK_WithFSR4` (FSR 4.0.2-era v07 INT8 source)
- Official SDK 2.3 extraction: `/tmp/fsr-sdk-2.3.1hSK7U`
- DXC source: `/home/fireburn/DirectXShaderCompiler`
- Known-good native DXC:
  `/home/fireburn/DirectXShaderCompiler/build-release/bin/dxc`
  (`libdxcompiler.so 1.10`, build id `5322-c64a5165`).  This local DXC tree was
  configured with LLVM exceptions/RTTI and has two small Objective-C rewriter
  compatibility fixes outside the Q2RTX repository; preserve or document them
  before rebuilding DXC.
- Wine: `/etc/eselect/wine/bin/wine` (11.15 staging)
- Proton Experimental FFX DLL:
  `/home/fireburn/.local/share/Steam/steamapps/common/Proton - Experimental/contrib/amdxcffx64.dll`

No approval prompts are available.  The worktree contains intentional changes
from parallel agents; do not reset or discard them.

## Screenshot/root-cause audit

Thirty-one `/home/fireburn/Screenshot_*.jpeg` files were inspected (one is
zero bytes and several are unrelated).  The hard upper-left corruption boundary
matches the active internal render extent exactly:

- 1356x763 overlay -> boundary at x=1356/y=763;
- initial 640x480 context -> boundary around 640x480 after resize;
- 50% scale -> boundary at 1280x720 on a 2560x1440 output.

Forced reset every frame removed the rectangle, and the later per-frame history
clear suppressed it while making output soft.  This proves a persistent-resource
write-domain/scheduling defect, not ordinary jitter tuning.  Negative motion
scale produced catastrophic trails; Q2 motion direction is correctly
current-to-previous.  `FLAT_MOTION` is jitter-free because Q2 constructs it
from unjittered `P`/`P_prev`; shaders must use
`FFX_MLSR_JITTERED_MOTION_VECTORS=0`.  Q2 adds jitter directly to the primary
ray's pixel center, while FSR performs an inverse input lookup; the dispatch
must negate Q2's jitter (as Q2 TAAU already does).  Weapon/emissive trails also implicate
missing masks and viewmodel motion/rejection, but they are secondary to the
provider failure.

Highest-confidence code failures in the old provider:

1. It hand-guessed a 14-pass graph instead of implementing AMD's provider.
2. It never uploaded the 89,216-byte initializer and bound activation scratch
   as fake weights, although model passes 6-9 read the initializer binding.
3. It never staged the 1,024-byte `cbPass_Weights` read by the pre-pass.
4. Every model/pre/post dispatch dimension was wrong.
5. Scratch was 32/64/128 MiB instead of exact generated-code sizes.
6. Recurrent/reprojected resources used wrong extents/formats.
7. Pre SPIR-V placed a standalone sampler and sampled image at binding 0.
8. Motion-vector permutation and texture extent disagreed, causing OOB reads.
9. Auto exposure was declared but SPD never ran and exposure was undefined.
10. Main constants used raw instead of normalized motion scale and lost prior
    exposure.
11. Q2 TAA, bloom, and tone mapping ran before FSR; depth was null.
12. Upload staging buffers were destroyed immediately after command recording.

## Current implementation changes

### Q2RTX FSR3 live integration (2026-08-19)

- `flt_upscaler` selects `0 = Q2RTX fallback`, `1 = FSR3 3.1.4`, or
  `2 = experimental source-v07 FSR4`; old `flt_fsr_enable` migrates to 2.
- FSR3 receives pre-TAA/pre-tone-map `FLAT_COLOR`, jitter-free
  current-to-previous `FLAT_MOTION`, and conventional device depth.  Q2RTX
  TAA is bypassed, then reconstructed linear HDR is copied into the existing
  display-resolution post chain.
- The adapter uses the inverse of Q2's ray-sample jitter, `GENERAL` state for
  external sampled inputs, `STORAGE_SCALE_HDR` pre-exposure, a one-inch world
  scale, and a 32-phase Performance jitter cycle.  It owns/reset histories
  independently of ASVGF, so `flt_enable 0` does not reset FSR every frame.
- Portable FSR3 now receives the *logical-device* enabled feature set.  It
  avoids AMD device-coherent memory, wave64 subgroup control, synchronization2
  Breadcrumbs, and optional paths unless Q2RTX explicitly enabled them.  This
  prevents the prior `VUID-VkMemoryAllocateInfo-memoryTypeIndex-02790`.
- Portable FSR3 now zero-initializes its aligned backend scratch arenas. AMD's
  Vulkan backend reads `refCount` before clearing its caller-owned scratch;
  using `malloc` caused resize-time context creation to spuriously fail with
  `FFX_VK_PORTABLE_ERROR_BACKEND (-7)` when heap memory was recycled. `calloc`
  makes context creation deterministic for both upscaling and FI/OF backends.
- Root build/ctest and standalone reusable suite pass. Standalone is now 20/20;
  live command used `vk_validation 1`, then ran for 55 seconds with no VUID.

The remaining FSR3 work is quality/preset/feature expansion, proper explicit
failure transaction handling, RenderDoc inspection, and full resize/reset/motion
coverage—not basic native integration.

### Provider-neutral Q2 inputs

- `src/refresh/vkpt/temporal_contract.h`: versioned Vulkan-only frame ABI.
- `src/refresh/vkpt/temporal.c`: Q2 adapter, frame/reset state, resources.
- `src/refresh/vkpt/shader/global_textures.h`: dense R32F
  `TEMPORAL_VIEW_Z` and `TEMPORAL_DEVICE_DEPTH` images.
- `src/refresh/vkpt/shader/primary_rays.rgen`: writes positive-forward primary
  view Z and conventional zero-to-one device depth before reflections alter
  denoiser depth; sky device depth is 1.
- `src/refresh/vkpt/path_tracer.c`, `main.c`, `vkpt.h`, `src/CMakeLists.txt`:
  synchronization, lifecycle, and build hooks.

The adapter publishes the ray-pipeline producer stage in pipeline mode and the
compute stage in ray-query mode.  Rounded-up ray-query invocations are guarded
in primary, reflection/refraction, direct, and half-resolution indirect ray
generation shaders.  The camera contract derives vertical FOV from
`abs(P[5])`; the previous signed calculation was approximately 179.9 degrees
because Q2RTX flips projection Y.  `view_space_to_meters` is currently 0.0254.

Verified before later FSR edits: full build, ray-pipeline/ray-query `spirv-val`,
and a ~20-second live `base1` session with FSR disabled on RX 6800M.

### FSR4 exact scheduler/provider/backend rewrite

- `src/refresh/vkpt/fsr4/ffx_fsr4_schedule.{h,c}`: exact DOT4 schedule,
  scratch sizes, preset selection.
- `src/refresh/vkpt/fsr4/tests/test_fsr4_schedule.c`: verified 2560x1440
  reference dimensions, alignment, limits, and presets.
- `ffx_functions_q2rtx.c`: now uses exact schedule/scratch sizes, separate model
  initializer, 1 KiB pre weights, normalized MV constants, prior exposure,
  display-sized persistent resources, one-time clears, error propagation, and
  no per-frame history clear.
- Persistent recurrent images are display-sized `R8G8B8A8_UNORM`, matching the
  reflected `Rgba8` storage-image ABI and AMD's provider.
- `ffx_fsr4_vk.{h,c}`: accepts/deep-copies model data, fixes pre-weight range,
  retains upload staging until safe teardown, queries required Vulkan features,
  separates sampled-image/sampler descriptors (sampler binding 35), and reports
  actual device capabilities.
- Mandatory pre/model/post pipelines now fail context creation if absent;
  pipeline handles encode pass+1, descriptor/job failures propagate, failed job
  batches roll back, sampled formats are checked for linear filtering, and
  partial resource/pipeline allocations unwind safely.  Generated constants
  use only the reflected 104-byte binding-43 UBO; the stale 88-byte
  push-constant range was removed.
- `fsr.c`: requires generated model assets, uses the temporal contract's
  pre-TAA HDR/motion/view-Z, uses storage scale as pre-exposure, dispatches before
  post effects, and copies reconstructed HDR to `TAA_OUTPUT` for existing
  display-resolution bloom/tone mapping.
- `main.c`: bypasses Q2 TAA/TAAU while FSR is active and puts FSR immediately
  after interleave.
- FSR selection occurs before the denoiser-disabled early return, so
  `flt_enable 0` no longer silently suppresses an otherwise valid FSR request.
- The FSR enable cvar uses the temporal-settings callback, which invalidates
  Q2 history and requests an FSR reset before a disable/re-enable cycle can
  reuse stale recurrent state.

FSR4 RCAS is now an optional post-model dispatch: it uses its reflected
32-byte constant buffer, the display-resolution intermediate, and the normal
HDR post-processing output. `flt_fsr4_sharpening` is clamped to [0,1]. The old
code set `rcas_enabled` in the post shader without scheduling RCAS; that
unsafe behaviour is gone. Explicit exposure is used by default; the optional
SPD auto-exposure graph is implemented behind `flt_fsr4_auto_exposure`.

### Generated FSR4 shader/model set

- `compile_shaders_fsr4.sh` now generates coherent 1080/2160/4320 pre, 12
  model, and post tiers, plus RCAS and future SPD assets.
- Runtime loading uses canonical manifest artifact names, not compatibility
  symlinks, so Windows checkouts without symlink privileges remain viable.
- `prepare_fsr4_spirv_sources.py` creates a temporary source overlay rather
  than modifying the SDK.  It assigns explicit Vulkan storage-image formats,
  unique sampler binding 35, correct positive-forward depth, and widens RGBA
  storage writes to four components.
- `validate_fsr4_spirv.py` checks all 44 modules, descriptor ABI, capabilities,
  image formats, and `OpImageWrite` component counts.  The old generated set is
  correctly rejected because DX12-style `float3` writes are invalid for an
  explicit Vulkan `Rgba16f` image.
- Model assets are 89,216-byte `fsr4_initializers.bin` and 1,024-byte
  `fsr4_pre_weights.bin`.  They remain ignored by the root `*.bin` rule; do not
  publish them without resolving provenance/licensing.

### Reusable module and capture tools

- `extern/ffx-vulkan/`: standalone CMake project, MIT provenance/import plan,
  portable C ABI, strict FSR3 upscale/FG validators, capability probe/tests,
  pinned AMD FSR3 1.1.4 host runtime, native Vulkan backend, and generated
  shader permutations.  Pristine source hashes are in
  `upstream/ffx-1.1.4/ORIGINAL_SHA256SUMS`; the exact SDK 2.3/backend/presenter
  port plan is in `UPSTREAM.md`.
- At that milestone, GCC Release and Clang Release passed all nine standalone
  tests; a backend-off
  C-only build passes its two applicable tests.  The
  shader test validates 2,816 references / 67 unique modules with hashes and
  `spirv-val`.  The live backend test dispatches 64x64 -> 128x128, reads the
  entire RGBA16F result back, rejects finite/alpha/poison/extent failures, and
  treats validation warnings and errors as failures.  Twenty-five consecutive
  Release runs passed on the RX 6800M.
- Two real upstream Linux defects were fixed: unqualified `abs(float)`
  corrupted Lanczos LUT weights, and the scratch allocator failed to align an
  `alignas(32)` `EffectContext`, causing a Release-only `movaps` crash.
- The portable public C ABI now provides opaque upscaler and analytical
  frame-generation contexts. The frame-generation path imports all 18
  FI/OF wrappers and 164 generated tables, validates 352 FI + 56 OF lookup
  permutations, and records Optical Flow, FI prepare, and interpolation into
  a caller command buffer. Its reset/temporal live RX 6800M readbacks are
  finite/nonzero/full-extent with zero validation warnings/errors. It fixes
  the upstream descriptor-pool storage-buffer omission and the dormant FP16
  blob-selector error. The upscaler context continues to
  create/record-dispatch/destroy, owns aligned scratch/backend/temporal and
  three shared images, and returns unsupported stubs when the backend is not
  built.  Its public-only two-frame hashes are `396a5392cdfa2325` and
  `447c2f165a412325`; each frame has 49,152 nonzero RGB components and zero
  validation warnings/errors.  Context destruction requires prior GPU
  completion.  SDK 2.3 algorithm port, Q2 integration, frame
  portable presenter remain outstanding.
- RX 6800M probe: FSR3 compute=yes, analytical FG=yes, FSR4 INT8 prerequisites
  yes (FP16/INT8/signed DOT4/timeline/sync2).  Renoir iGPU reports FSR4 INT8=no.
- `tools/ffx_dxil/`: scan/verify/hash/optional extract and vkd3d pairing tool,
  capture guide, protective ignore, 9 passing tests.
- Validated SDK 2.3: upscaler 1028/885, denoiser 630/630, frame generation
  487/486 occurrences/unique; Proton DLL 1294/1150; zero malformed containers.
- tools/ffx_dxil/reference_harness/fsr_provider_probe.cpp: a source-only
  MinGW/Wine probe that creates a DX12 device, enumerates SDK provider versions,
  creates a 4.1.1 API context, and optionally records one controlled dispatch.
  It registers public resource allocation callbacks. The RX 6800M result is
  provider 3.1.5 fallback with 11 paired capture shaders and 25 logged
  allocations; no provider/model/capture binary is tracked.

## Current coordination

The completed shader-generation, FSR4 backend/live-validation, portable FSR3
API, DXIL tooling, temporal-contract/device-depth, ray-query, menu, and GPU-AV
renderer milestones are reflected here. The active integration frontier is the
portable single-queue Frame Generation presenter: the first Q2RTX path now
acquires generated plus real images, runs analytical FI over HUDless
`TAA_OUTPUT`, replays queued UI over both, and presents in generated→real
order. It waits for the reserved second image instead of treating an immediate
WSI miss as a permanent fallback. The remaining work is timeline/presentation
pacing, FPS telemetry, and broader live visual validation. The presenter owns
two final-blit descriptor slots per frame and uploads replayed UI once before
recording generated and real draws; queue order and the normal real-frame fence
now safely cover both commands without the former CPU replay wait. The FFX
dynamic-view ring retains eight effect calls because FI prepares and dispatches
twice per real frame, fixing the live in-flight image-view VUID.

Check `git status --short` before editing because agent changes are shared
immediately.

## Validation performed after current rewrite

Commands run successfully:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target fsr4_schedule_test -j"$(nproc)"
ctest --test-dir build --output-on-failure -R fsr4_schedule
cmake --build build -j"$(nproc)"
python3 src/refresh/vkpt/fsr4/validate_fsr4_spirv.py \
  baseq2/fsr4_shaders --preset performance
cmake --build build/ffx-vulkan -j"$(nproc)"
ctest --test-dir build/ffx-vulkan --output-on-failure
python3 -m unittest discover -s tools/ffx_dxil/tests -v
```

Root build/test, 44-module FSR4 ABI validation, all sixteen reusable FSR3 tests,
all nine DXIL-tool tests, and `git diff --check` pass.  The FSR3 public-only
smoke records and submits two consecutive frames through only the C ABI.

### Experimental FSR3 Frame Generation presenter (2026-08-19)

- `flt_frame_generation 1` is now an explicitly labelled experimental control.
  It is gated to active native FSR3, one GPU, rectilinear projection,
  display-sized `TAA_OUTPUT`, valid device depth/motion, and a swapchain with
  at least `minImageCount + 2` images (five on the RX 6800M/RADV test surface).
- Q2RTX acquires generated then real swapchain images, records Optical Flow +
  Frame Interpolation into `FSR_RCAS_OUTPUT`, composites the same queued UI on
  both HUDless scene images, and presents generated→real. The second acquire
  waits for the deliberately reserved image; if it or interpolation fails, the
  real scene is presented safely instead.
- The presenter owns distinct generated/real final-blit descriptor slots per
  frame and uploads the replayed UI once, so it does not update a descriptor or
  host-visible UI buffer while the generated command consumes it. The generated
  and real submissions are ordered on the same queue; the ordinary real-frame
  fence therefore retires both without a CPU replay wait. The reusable backend
  now retains eight dynamic-view generations because FI uses Prepare and
  Dispatch in each rendered frame, preventing an in-flight view from being
  recycled on Q2RTX's two-frame host. It now publishes both the active state
  and cadence only for complete interpolated-generated→real WSI pairs; a
  future scheduler still needs explicit present pacing. A two-present
  real-frame fallback cannot inflate these diagnostics: cadence is published
  only when the FI output was successfully recorded.
- A 1280x720 RX 6800M run with `vk_validation=1`, FSR3 Performance, and
  `flt_frame_generation=1` rebuilt 640x480 then 1280x720 FSR3/FG contexts and
  exited cleanly with no FSR3-FG/VUID messages in `console.log`. A later
  source-tree run also verified the five-image requirement stayed active. A
  fresh 36-second no-replay-wait run emitted no validation error after the
  dynamic-view-ring correction; its active desktop capture is
  `/home/fireburn/Screenshot_FSR3_FG_NoWait_20260819.png`. It needs RenderDoc,
  resize, menu, and low-FPS testing.

Live FSR4 command (the doubled `++` is required to pass literal Quake key
commands through command-line parsing):

```sh
env MESA_VK_DEVICE_SELECT=1002:73df! DRI_PRIME=1 VK_LOADER_DEBUG=error \
./q2rtx \
  +set basedir /home/fireburn/Q2RTX \
  +set libdir /home/fireburn/Q2RTX \
  +set vid_fullscreen 0 +set vid_geometry 2560x1440+1920+0 \
  +set vid_vsync 0 +set viewsize 50 +set drs_enable 0 \
  +set vk_validation 1 +set flt_upscaler 2 \
  +set flt_fsr4_sharpening 0.5 \
  +set scr_fps 2 +set developer 1 +map base1 ++forward ++left
```

The run rebuilt from its initial 320x240 -> 640x480 1080-tier context to a
coherent 2160-tier 1280x720 -> 2560x1440 context.  It then rendered and moved
for tens of seconds without an FSR-specific validation message.  Two distinct
motion screenshots were visually coherent.  Proof screenshot:
`/home/fireburn/Screenshot_FSR4_working_20260813.jpeg`.

GPU-assisted validation was then run at 640x360 -> 1280x720.  It reported no
FSR4-pass descriptor/bounds error, but exposed three independent renderer
issues: PT view-depth images were R16F while declared `r32f` in GLSL (fixed to
`r16f`); tiny negative ray `Tmax` values in direct/indirect lighting; and a
tone-mapping shared-memory data race.  All three were fixed.  Repeat GPU-AV
runs are clean at both 640x360 -> 1280x720 and 1280x720 -> 2560x1440, including
sustained target-resolution frames with no FSR-pass descriptor/bounds finding.

The added device-depth image and descriptor-table shift were then exercised in
both fallback and active-FSR4 GPU-AV runs without a validation finding.  An
explicit `ray_tracing_api query` run is also GPU-AV clean after correcting the
producer stage and guarding rounded-up query launches.  Active FSR4 with
`flt_enable 0` is clean after moving FSR selection ahead of the denoiser early
return.

After this proof run, the independent startup swapchain error was fixed by
removing create-time transitions of unacquired images and tracking each image's
first acquired use.  A separate 1280x720 fallback run with validation enabled
then rendered/presented for ten seconds with zero validation messages.  In-game
screenshot readback still accesses the last presented swapchain image outside
the acquired-frame lifecycle and needs its own offscreen/readback redesign.

## Immediate next actions

1. Link `extern/ffx-vulkan`'s public C upscaler into Q2RTX, map the temporal
   scene color/device depth/motion/camera contract, dispatch one two-frame live
   sequence, and preserve the existing fallback on any capability/error path.
2. Add a central requested-versus-active resolver for Q2RTX/FSR3/experimental
   FSR4, then replace the temporary FSR4 toggle with provider/quality controls
   whose labels expose the actual active implementation and fallback reason.
3. Run GPU-assisted validation and a RenderDoc frame; prove every FSR4 pass's
   descriptors, initializer upload, activation ranges, barriers, and bounds.
4. Expand live FSR4 coverage to resize/tier transitions, reset/map transition,
   weapon and emissive motion, disocclusion, camera cuts, and long runs.  Add
   debug views for color, motion, view-Z, history, and reconstructed output.
5. Extend the corrected acquire/present lifecycle into an offscreen scene/UI
   target, safe screenshot readback, and generated-frame presentation scheduler.
6. Add discrete model-backed quality presets and validate SPD exposure across
   abrupt lighting changes before treating the experimental control as mature.
7. Build the portable analytical-FSR3 single-graphics-queue presenter/UI/pacing
   system and integrate its already-validated compute API into Q2RTX.
8. Continue measured 4.1.1/RR/ML-FG provider capture research in parallel; do
   not relabel the current v07 model as 4.1.1.

## Known risks and cautions

- `ffx_fsr4_vk.c` remains a prototype backend with fixed limits and still needs
  an explicit in-flight retirement API, reflected per-pipeline layouts,
  resource state tracking, and memory accounting.
- Conventional device depth is available only for single-device rendering and
  has no debug view yet.  Device-group input gathering remains unimplemented.
- First-person weapon motion still needs visual tuning and broader coverage;
  it now supplies a conservative reactive-mask signal.
- FSR4 v07 supports five fixed quality models and its separate DRS model.
  Arbitrary old FSR1 scale controls must not select a static model; select the
  explicit DRS model to use Q2RTX's bounded controller.
- Official SDK 2.3 supports its ML effects only through signed DX12 binaries.
  RDNA2 community paths are experimental and are not official AMD support.
- The old source-bearing FSR4 payload has ambiguous history despite repository
  MIT text and per-file “all rights reserved”/opaque weights.  Do not publish
  model data or derived payloads without provenance/legal review.
