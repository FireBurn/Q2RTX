# FidelityFX Vulkan implementation TODO

Last updated: 2026-09-02 (Europe/London)

Status labels: `[x]` verified complete, `[-]` in progress/partially complete,
`[ ]` not started, `[R]` research/unknown viability.

## Immediate visual-correctness follow-up

- [x] Reproduce and repair generated/real strobing from the generated target
  itself, rather than inferring correctness from a real-slot screenshot.
  `flt_frame_generation_debug_capture` proved that SDK-3.1.6 FI/OF produced a
  black target on RX 6800M. The repaired bridge accepts zero-size remaining
  buffer ranges, performs image copies, propagates SDK job failures, supports
  inactive tail SPD mips, and uses regenerated exact Vulkan storage-image
  formats. The direct target, held weapon/emissive, camera cut, map transition,
  and sustained 30-FPS-gated runs are full-colour and validation-clean. Backend
  `1` is the default; `0` remains 1.1.4 compatibility.

## P0 — make the current FSR4 INT8 path correct and observable

- [x] Correlate screenshot corruption with exact render-resolution boundary.
- [x] Audit the hand-written provider and identify missing initializer,
  pre-pass weights, wrong dispatches, wrong scratch size, descriptor collision,
  invalid exposure path, constants, and history-clear workaround.
- [x] Add and unit-test exact renderer-independent DOT4 dispatch/scratch/preset
  helpers (`ffx_fsr4_schedule.*`).
- [x] Replace guessed provider dispatches and resource extents with exact logic.
- [x] Require/upload the 89,216-byte model initializer and stage the 1,024-byte
  pre-pass weights.
- [x] Rebuild shaders with a unique sampler binding and exact Q2 permutation;
  validate every module's SPIR-V ABI.
- [x] Correct backend staging lifetime, reflected 104-byte UBO ABI,
  descriptors, feature/format checks, pass-handle encoding, error propagation,
  job rollback, partial-allocation unwind, and idempotent destruction.
- [x] Replace FSR4 v07's renderer-assumed descriptor-pool/constant-buffer
  rotation with explicit `BeginFrame(frameId) -> RetireFrame(completedFrameId)`
  ownership. Q2RTX connects it to its already-waited per-slot fence; the
  portable RX 6800M test fills all three unretired slots, rejects unsafe fourth
  reuse, and proves reuse only after retirement. A live FSR4 Quality run stays
  active with Vulkan validation clean.
- [x] Fail closed when a supplied v07 SPIR-V module's reflected descriptor
  set/binding/type declaration differs from the portable provider ABI. The
  reusable checker validates all 288 generated modules across six presets and
  three tensor tiers, plus a deliberate pass mismatch; context creation runs
  it before pipeline creation.
- [x] Derive each v07 Vulkan pipeline's compact descriptor-set layout from its
  validated SPIR-V declarations rather than retaining a generic 44-binding
  superset. Descriptor writes now fail before dispatch if any declared
  non-sampler slot is absent; the RX 6800M live path and all 36 standalone
  tests remain clean.
- [x] Make FSR4 external-image state ownership explicit. The host registers
  image/view plus current and restored Vulkan layout/stage/access before each
  dispatch; the provider transitions to GENERAL, validates FFX read/UAV role,
  and restores imports at unregister. Q2RTX supplies all four temporal images.
  The RX 6800M test exercises a real `SHADER_READ_ONLY_OPTIMAL -> GENERAL ->
  SHADER_READ_ONLY_OPTIMAL` round trip.
- [x] Honor FFX float clear values in the Vulkan backend and initialize the
  sampled explicit-exposure fallback to 1.0 rather than zero; a subsequent
  RCAS-on live validation run passed on RX 6800M.
- [x] Remove the per-frame history/reprojection clear; retain one-time
  deterministic initialization only.
- [x] Feed dense pre-TAA, pre-bloom, pre-tone-map linear HDR inputs and bypass
  Q2RTX TAA when FSR is active.
- [x] Keep requested/active FSR selection independent of `flt_enable`, so
  disabling the denoiser does not silently disable or desynchronize FSR.
- [x] Mark temporal history valid after a successful FSR dispatch even with
  ASVGF disabled; a 28-second FSR3 Performance + RCAS live run with
  `flt_enable 0` stayed active with Vulkan validation clean.
- [x] Copy reconstructed linear HDR into the existing display-resolution post
  chain, then bloom/tone map and draw UI.
- [x] Run with Vulkan validation and GPU-assisted validation at fixed
  1280x720 -> 2560x1440 Performance, sharpening off, explicit exposure.
  Standard validation, live visual testing, and GPU-assisted validation pass at
  both 640x360 -> 1280x720 and the target 1440p output.
- [-] Capture every pass in RenderDoc/RGP and prove descriptor/resource bounds.
  The available RenderDoc 1.39 Vulkan layer supports X11/XCB but not Wayland;
  Q2RTX's current SDL build is Wayland-only, so an injected capture fails at
  `VID_Init` before Vulkan work. Do not count this as an FSR4 failure. A live
  960x540 FSR4 v07 validation run is clean; use an X11-enabled SDL build or a
  Wayland-capable capture tool to complete the per-pass evidence.
- [-] Take before/after screenshots and verify static detail, camera motion,
  weapon motion, emissives, disocclusions, resize, reset, and map transitions.
  Static output plus sustained forward/rotation motion are coherent and the old
  extent rectangle/history train is gone; a new RCAS-on capture is coherent.
  A fresh Quality 858x482 -> 1280x720 run with RCAS 0.50 and SPD auto exposure
  crossed `base1 -> base2` with no validation/dispatch error and a coherent
  post-transition viewmodel scene:
  `/home/fireburn/Screenshot_FSR4_v07_map_transition_20260820.png`.
  The menu-to-game 640x480 -> 960x540 extent recreation also selected a new
  644x361 -> 960x540 context with no validation error and coherent output:
  `/home/fireburn/Screenshot_FSR4_v07_resize_20260820.png`. Arbitrary live
  resize, reset, emissive, and disocclusion coverage remains outstanding. The
  console screenshot path itself now freshly acquires a local WSI image, copies
  it, and presents it again instead of transitioning the last presented image;
  the 960x540 Quality + RCAS capture is coherent and emitted no VUID/error:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_state_contract_20260825_validated.png`.
  A 120-frame held-blaster script under active SDK-3.1.6 FI/OF now captures
  moving weapon/projectile emissions at 125 logical FPS, with no VUID, FSR,
  dispatch, or presenter error and a coherent full-colour frame:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_weapon_emissive_fiog.png`.
  A new validation-enabled odd-size 960x540 -> 1133x717 -> 960x540 sequence
  rebuilt Quality at exact 760x480 -> 1133x717 and 644x361 -> 960x540 extents,
  restored active SDK-3.1.6 FI/OF after both expected reset fallbacks, and
  wrote coherent full-colour captures at all three sizes with no VUID, FSR,
  dispatch, or presenter error:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_odd_resize_1133x717.png`
  and `FSR4_odd_resize_restored.png`. Arbitrary reset and disocclusion coverage
  remains outstanding.
- [x] Implement and validate RCAS as a real separate pass; expose the separate
  `flt_fsr4_sharpening` [0,1] control only after it works.  A 0.50 live
  640x360 -> 1280x720 Performance run passed Vulkan validation on RX 6800M.
- [x] Add the exact SPD auto-exposure resources and dispatch behind the
  disabled-by-default `flt_fsr4_auto_exposure` cvar. Its 32-byte cbuffer,
  2×1 current/previous exposure image, R32_UINT atomic counter, R32F mip-5,
  and 64×64 SPD dispatch were live-validated on RX 6800M; disabling it restores
  the 1.0 explicit-exposure identity. A fresh 1280x720 Performance + RCAS 0.50
  + auto-exposure smoke after the public-SDK FSR3 global-image additions was
  validation-clean and visually coherent:
  `/home/fireburn/Screenshot_FSR4_v07_FIFO_table_smoke_20260820.png`.
- [x] Add and ABI-validate Native AA, Quality, Balanced, Performance, and Ultra
  Performance model assets/preset switching with history reset. Each selection
  loads its complete v07 INT8 graph plus matching 89,216-byte initializer and
  1,024-byte pre-pass weights; a live Quality 858x482 -> 1280x720 RX 6800M run
  reported the selected model active with Vulkan validation clean.
  Native AA and Balanced also received fresh 1280x720 RX 6800M live runs and
  reported their exact model labels with no validation errors.
  A later live Performance→Quality change while SDK-3.1.6 FI/OF was active
  exposed a resolver ordering bug: it reported a permanent pending model
  switch before dispatch could rebuild. The resolver now calls the safe,
  device-idle recreation before testing availability; the repeat selected the
  Quality graph at 644x361 -> 960x540 and resumed active FI/OF with no
  VUID/error and a coherent capture.
- [x] Add the v07 DRS model to Q2RTX's existing bounded profiler-driven
  controller. It has its own validated graph/assets and never runs a static
  model at an arbitrary ratio; a 50%-bounded live RX 6800M run reported the DRS
  model active with Vulkan validation clean. Its sole render-size-change reset
  reason now preserves the DRS model's display-sized recurrent history; all
  other reset reasons remain conservative. The DRS input and feedback are
  capped at native 100%, including the controller-empty startup case; a
  `viewsize 150` live validation resolved to 1280x720 -> 1280x720 and stayed
  validation-clean.
  The later resolver fix was also verified across a live static
  Performance→DRS switch with SDK-3.1.6 FI/OF active: it rebuilt the DRS graph,
  reached v11/frame 1588 with the 50..100% controller enabled, and emitted no
  VUID/error or fallback.
- [x] Update the in-game video menu for the FSR3/FSR4 controls and deploy its
  loose source-menu override on this development system, where the installed
  `q2rtx_media.pkz` still contained the obsolete FSR 1.0 controls. Linux
  packaging now preserves NVIDIA's release media archive and updates its
  source-controlled menu/config entries, so it does not silently omit the
  released texture/model/audio tree. A fresh 960x540 Vulkan-validation menu
  capture with source-v07 FSR4 plus SDK-3.1.6 FI/OF selected is readable and
  truth-labelled: `FSR4 v07 (experimental)`, separate FSR4 DRS/sharpening,
  analytical frame generation, 3.1.6 scheduler, and the independent FPS gate
  are all visible at `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_video_menu_audit.png`.
- [x] Version and package the complete older source-v07 FSR4 model set: six
  manifest-checked 89,216-byte initializers and six 1,024-byte pre-pass
  weights. The retained `LICENSE-FSR4-v07.txt` records its upstream revision
  and MIT notice. A clean Release package staging install built the client and
  installed exactly those 12 model binaries plus the notice; this does not
  change the separate FSR4.1.1 research boundary. The staged installed client
  also launched on RX 6800M with validation enabled and reported `FSR4 v07
  INT8/DOT4 Quality active` at 644x361 -> 960x540 with no FSR4 VUID, dispatch,
  or asset-fallback message.
  Root CTest now runs `fsr4_v07_assets`, which independently verifies all six
  pairs against their manifest size and SHA-256 before source changes land.

## P1 — provider-neutral Q2RTX inputs and presentation split

- [x] Add a versioned reusable Vulkan temporal-frame contract and a live
  pre-dispatch validator for its ABI, stage/extents, image producer state,
  requested slots, motion/camera metadata, and depth conventions. Quality FSR4
  and FSR3+analytical-FG RX 6800M validation runs accepted the normal contract
  and stayed clean.
- [x] Add a reusable FSR4 v07 asset-set contract that selects matching model,
  tensor tier, initializer, pass-0 weights, RCAS, and SPD names without tying
  another Vulkan application to Q2RTX's filesystem/backend policy.
- [x] Promote the complete source-v07 Vulkan provider into
  `extern/ffx-vulkan` as `ffx-vulkan::fsr4-v07-vulkan`. Q2RTX now links that
  exact library, its public headers/versioned `ffxFsr4V07…` API avoid AMD SDK
  symbol collisions, and an installed-package consumer contract builds/links
  without the renderer. Its explicit per-frame fence-retirement API is part of
  that installed contract. The full FSR3 source closure remains an
  `add_subdirectory` integration until it has a dependency-complete export.
  `ffx-vulkan::effects` and `examples/full-stack` now provide the tested
  vendored FSR3.1.4/3.1.5/3.1.6 + FSR4-v07 starting point for another Vulkan
  renderer; the example builds and runs without Q2RTX.
- [x] Expose dense scene color, current-to-previous motion, stable primary
  view-Z, camera matrices/jitter, reset reasons, frame ID, and pre-UI boundary.
- [x] Preserve primary view-Z before reflection passes overwrite denoiser depth.
- [x] Remove invalid create-time transitions of unacquired swapchain images;
  initialize each image in its first acquired frame submission.
- [x] Correct the legacy R16F PT view-depth storage-image declarations from
  `r32f` to `r16f` after GPU-assisted validation exposed the mismatch.
- [x] Fix GPU-AV findings in shadow ray `Tmax` and tone-mapping shared-memory
  synchronization so FSR validation is not obscured by renderer errors.
- [x] Produce dense R32F conventional zero-to-one device depth beside primary
  view-Z and publish its finite near=0/far=1/sky=1 convention in the contract.
- [x] Use the compute producer stage in ray-query mode and add invocation bounds
  guards to all rounded-up ray-query dispatches; validate query and pipeline
  shader variants.
- [x] Add device-depth and view-Z debug visualization, plus direct
  presentation-only views for scene HDR, motion, both masks, normals, albedo,
  and roughness. The shader uses semantic mappings (logarithmic positive
  view-Z, signed pixel motion, and HDR/albedo debug tone maps) rather than
  presenting raw storage values. A live RX 6800M `vk_validation=1` view-Z run
  was coherent and validation-clean; selecting a view suspends analytical FG.
- [x] Add dense primary-material reactive and transparency/composition masks
  for FSR3. Transparent, water/glass, warped, screen/camera paths feed both
  masks; view-model pixels feed reactive history rejection. A 1280x720 RX
  6800M run recreated the menu context, activated FSR3, and passed Vulkan
  validation with both masks registered.
- [x] Keep the dedicated HUDless offscreen scene target and add a separate
  alpha UI target. `TAA_OUTPUT` is the display-resolution,
  tone-mapped-but-HUDless scene consumed by both FSR3 FI implementations and
  by the v07 FSR4-super-resolution + analytical-FG combination. The v07
  provider copies its reconstructed output there before the shared FI/OF
  recording stage; this is not AMD's binary ML Frame Generation.
  Active FG now renders the queued UI once to a per-frame-slot linear RGBA16F
  premultiplied-alpha texture and composites it over both generated and real
  scenes. The provider-neutral temporal contract publishes this as
  `VKPT_TEMPORAL_UI_SEPARATE_TEXTURE`; ordinary rendering retains direct UI.
  Targets allocate lazily only while FG runs. A RX 6800M `vk_validation=1`
  FSR3.1.6 FIFO run was coherent with no VUID/error:
  `/home/fireburn/Screenshot_FSR3_alpha_ui_lazy_20260825.png`.
- [x] Add per-input debug views in the live renderer. The contract validator is
  already executed immediately before FSR3, FSR4 v07, and analytical FG imports.
- [x] Add FSR reconstructed-output and private FSR4 history/reprojected views
  without exporting mutable provider resources. The diagnostic seam returns
  only borrowed sampled image views after a successful FSR4 dispatch. A
  validation-enabled RX 6800M history capture is coherent and full-frame:
  `/home/fireburn/Screenshot_FSR4_v07_history_20260820.png`.
- [-] Support/gather temporal inputs for device-group rendering. The generated
  present now correctly indexes its render-finished semaphore by swapchain
  image and GPU, but device-group depth/material input gathering itself remains
  unimplemented.

## P2 — reusable native Vulkan FSR3

- [x] Add standalone `extern/ffx-vulkan` ABI, strict validators, capability
  probe, tests, license/provenance, and upstream import manifest.
- [x] Export the complete FSR3 + source-v07 FSR4 target closure as an installed
  CMake package. `ffx-vulkan::effects` now brings in the compiled private
  FSR3.1.4, 3.1.5, and 3.1.6 FI/OF closures alongside FSR4 and presentation
  policy without exposing source-tree include paths. A clean installed C++
  consumer links and runs `examples/installed-full-stack`. The top-level
  CTest suite also installs the package, copies that self-contained consumer
  outside the source tree, then configures/builds/runs the copy against the
  fresh prefix (36/36 without the external v07 asset bundle; 37/37 with the
  Q2RTX-local layout asset check).  On 2026-08-31 the full Q2RTX tree also
  rebuilt successfully after this package test was added; its root CTest
  checks passed 2/2 (`fsr4_schedule`, `fsr4_v07_assets`).
  The 3.1.6 FI public output deliberately uses a formatless storage-image
  declaration, so the same installed API is valid for both RGBA8 and RGBA16F
  presentation images; compact internal storage images remain typed. A fresh
  Q2RTX-local standalone build passes all 37 CTests, including both formats.
  The clean redistributable subtree was split, audited to exclude binary/model
  payloads and Q2RTX data, and published to `FireBurn/FSR-Vulkan` `main` at
  `0411b8d8`; its clean release build passes 36/36 applicable CTests.
  A distributor can now opt into a complete, validated v07 asset installation
  with `FFX_VK_PORTABLE_INSTALL_FSR4_V07_ASSETS` and an explicit asset path.
  The package installs only the notice and six model-prefixed, manifest-backed
  bundles under `share/ffx-vulkan/fsr4-v07`, then exposes that location through
  `FFX_VK_FSR4_V07_ASSET_DIR`; unrelated neighbouring/extracted blobs are not
  copied.
- [x] Confirm RX 6800M prerequisites for FSR3 compute and analytical frame
  interpolation.
- [x] Vendor the pinned MIT-licensed AMD FSR3 1.1.4 host runtime, pristine
  hashes, provenance, and Linux portability fixes; validate with GCC and Clang.
- [x] Import/implement the FSR3 1.1.4 Vulkan compute backend and generated
  shader permutation headers; validate 2,816 references / 67 unique SPIR-V
  modules and create a live RX 6800M upscaler context.
- [x] Dispatch and read back a real 64x64 -> 128x128 FSR3 upscale through the
  low-level backend; 25/25 RX 6800M runs pass full-extent/finite/alpha/poison
  invariants with zero validation warnings/errors.
- [x] Expose opaque FSR3 create/destroy/record-dispatch through the portable
  public C ABI, including owned scratch/backend/shared resources and backend-off
  unsupported stubs.
- [x] Run a public-API-only two-frame 64x64 -> 128x128 dispatch/readback smoke;
  both frames fully overwrite finite RGBA output with zero Vulkan validation
  warnings/errors. The current standalone suite passes 20/20; the original
  portable C API slice passed its 8/8 GCC and Clang coverage before the later
  asset/contract tests were added. The public context also requires the
  caller to attest that `shaderStorageImageWriteWithoutFormat` was enabled on
  its logical device, rather than treating physical-device support as enough;
  the RX 6800M smoke covers both rejection and successful enabled dispatch.
- [-] Port the public SDK 2.3 algorithms to the reusable backend while keeping
  the pinned 1.1.4 runtime as the reproducible reference implementation.
  The exact v2.3.0 public FSR3.1.5 plus FI/OF 3.1.6 source closure is now
  imported with pristine and current SHA-256 manifests (166 source files).
  The FI/OF 3.1.6 sources are provenance-complete and compile as a separate
  private `ffxVk316...` object-only portability target. All 11 FI and 7 OF
  public HLSL entry points generate as a source/output-hashed,
  Vulkan-1.2-validated compact-descriptor SPIR-V bundle. Its actual fixed
  Q2RTX-profile SDK blob accessors now return every embedded module and reject
  unsupported permutations; a dedicated test verifies all 18 returned blobs,
  error paths, and Wave64 queries. Its real host-graph test links the FI/OF
  schedulers with the public SDK core, creates 42 persistent resources and all
  18 embedded-SPIR-V pipelines, then destroys them cleanly. It records the
  reset Optical-Flow → FI Prepare → FI Dispatch graph against a strict mock
  with 15 dynamic registrations, 12 constant buffers, and 30 FI jobs. A real
  RX 6800M Vulkan test creates all 18 FI/OF compute pipelines. A real Vulkan
  scheduler bridge and portable presenter integration remain separate work.
  The existing SDK-2.3 resource/job bridge now accepts direct SPIR-V scheduler
  blobs before its legacy 3.1.5 catalogue fallback; its GPU test proves that
  route with an unnamed FI module. Wire it to the FI/OF contexts next.
  The bridge now handles the measured FI/OF storage-buffer counter bindings
  (`rw_counters`/`r_counters`) end-to-end: reflection, SDK resource tables,
  Vulkan storage-buffer descriptors, and recorded compute jobs. Reflection now
  uses SPIR-V variables, pointer storage classes, and image declarations to
  select descriptor kinds. Only the SDK `r_`/`rw_` convention selects buffer
  direction because Vulkan has one storage-buffer descriptor type. The bridge
  test now queries and enables the two required
  storage-image-without-format feature bits only when supported, and on the
  RX 6800M creates both live FI/OF contexts and submits their initialization
  work with validation clean. A real RX 6800M reset-frame test now imports the
  public SDK's application/shared images and records/submits Optical Flow ->
  FI Prepare -> FI Dispatch through the same reusable bridge with zero Vulkan
  warnings/errors. Its Vulkan source overlay gives the storage images the
  exact public resource formats (R8_UINT luma, R16G16_SINT optical flow, and
  R16G16_FLOAT dilated motion) rather than relying on DX12's looser typed-UAV
  rules. The same test then records/submits a non-reset temporal frame with
  the bridge restoring imported resource layouts between frames. The reusable
  `ffx-vulkan::fsr3-vk-framegeneration-3.1.6` target now exposes a versioned,
  opaque `create -> prepare -> dispatch -> retire -> destroy` C lifecycle;
  its public-only RX 6800M smoke records reset plus temporal frames with zero
  validation warnings/errors in both RGBA8 and Q2RTX-compatible RGBA16F
  modes. Frame-ID retirement makes the caller's fence boundary explicit while
  retaining multiple queue-ordered frames safely. It is now selectable in
  Q2RTX as `flt_frame_generation_backend 1` and is the default for fresh
  configs; the longer-tested 1.1.4 backend remains available as `0`. The
  integration imports Q2RTX's RGBA16F
  HUDless color, R32F device depth, and RGBA16F motion surface (the SDK
  samples its XY channels), then retires imports at the frame-slot fence. A
  30-second RX 6800M `base1` run at 1280x720, with validation enabled,
  recreated the context at startup/resize and reached active FIFO-paired
  presentation with no VUID, prepare, dispatch, or interpolation-skip log.
  HDR and broader lifecycle/visual coverage remain deliberately experimental.
  The active status now identifies the selected 1.1.4 or 3.1.6 FI/OF scheduler
  rather than labelling both generically.
  The reusable API now also passes the public sRGB/PQ/scRGB transfer function
  and luminance range through both OF and FI; Q2RTX selects scRGB in HDR mode.
  A physical RX 6800M HDR-surface run is validation-clean: source-v07 FSR4
  Quality plus SDK-3.1.6 FI/OF was active under `vid_hdr=1`, and
  `screenshothdr` produced `FSR4_hdr_fiog.hdr`. Broader visual/lifecycle
  coverage remains experimental.
  GCC 16 and Clang 22 both build this coexistence gate while the reusable suite
  passes 36/36. The added Q2RTX-profile FI/OF smoke uses RGBA16F color/output
  and the engine's RGBA16F `FLAT_MOTION` input, verifying that the SDK samples
  the normalized vector in RG while Q2RTX preserves derivative metadata in BA.
  A separate scRGB-luminance smoke validates the HDR runtime constants.
  `RecordDispatch` now also accepts the SDK's optional sampled R16G16_SFLOAT
  distortion field (`UV_after - UV_before`) and retains its imported view until
  the caller fence retires it; the RX 6800M reset/default-field then
  temporal/external-field smoke passes in every RGBA8/RGBA16F/Q2-motion/scRGB
  variant with validation clean. Q2RTX intentionally leaves this null until it
  produces a post-processing distortion field of its own.
  The SDK-3.1.6 FI wrapper's optical-flow scale was corrected from `{1,1}` to
  reciprocal display dimensions, matching the public provider. A fresh
  validation-enabled RX 6800M capture now shows a coherent generated scene
  rather than the former black output:
  `/home/fireburn/Screenshot_FSR316_FG_20260820.png`.
  A queued `base1 -> base2` transition then completed with no VUID/dispatch
  error and a coherent post-transition generated capture:
  `/home/fireburn/Screenshot_FSR316_FG_base2_20260820.png`.
  The direct SDK host-graph fixture now uses the same reciprocal scale, so it
  cannot preserve the invalid `{1,1}` example.
  Its upscaler host scheduler compiles
  as a separate Linux object scaffold after documented non-Windows DLL-export,
  watermark, and opaque-context-size fixes. An always-on graph test now proves
  the reset and temporal-RCAS job/resource lifecycle (19 persistent resources,
  four initialization uploads, 11 pipelines, and 27/21 queued jobs). The
  checked generated 3.1.5 modules now create 11/11 real Vulkan compute
  pipelines and exact reflected descriptor sets on RX 6800M. A dedicated
  SDK-2.3 pipeline callback bridge now maps the scheduler's names to the
  embedded modules and reflects its SRV/UAV/CBV slots back into
  `FfxPipelineState`; its 11/11 RX 6800M test passes. The bridge now also
  allocates/exports/destroys SDK-owned mipmapped image resources through real
  Vulkan memory and SRV/UAV views. Its ordered copy-job layer now initializes
  the SDK's Lanczos/default-mask/default-exposure resources from host-visible
  staging buffers; a real 3.1.5 context creates 19 persistent resources,
  records and completes all four uploads, then destroys cleanly on RX 6800M.
  The bridge now stages the three per-frame constants, records clear/copy/
  transition/UAV/discard jobs, creates reflected descriptor sets, and submits
  real 64x64 -> 128x128 reset and temporal-sharpened graphs using bridge-owned
  stand-in frame/shared images. The temporal-RCAS graph now consumes
  deterministic HDR/depth/motion/mask uploads, starts from poisoned RGBA16F
  output, and copies the native output back: all 49,152 RGB components are
  finite/nonzero and overwrite poison with Khronos validation clean. It now
  imports caller-owned 2D VkImages with explicit format/layout/state,
  restores those caller layouts on SDK unregister, and never takes external
  image/memory ownership; a standalone import/transition/restore test is
  validation-clean. The reusable C lifecycle now creates the host/context,
  reports shared-resource descriptions, and records a full reset dispatch with
  nine imported application images (including initialization uploads) under
  Vulkan validation. It is now selected experimentally in Q2RTX as
  `flt_upscaler 3`, with its SDK-3.1.5 source closure privately namespaced to
  avoid the otherwise identical 1.1.4 exported C symbols. Fence-driven
  Per-dispatch descriptor and constant-buffer storage is now explicitly tagged
  with the caller's monotonic frame ID and reclaimed when Q2RTX's matching
  frame-slot fence retires. This fixes the former unbounded 3.1.5 transient
  storage growth that made a long live run spend excessive time in RADV/libdrm
  during context destruction. The reusable suite passes 37/37 after the API
  addition. Broad visual-quality coverage remains. The
  reset and temporal-sharpened
  test enables Khronos validation and is clean after the generated HLSL
  explicitly annotates every storage image with the matching public
  R8/RGBA8/R16/RG16/RGBA16F format. The current source manifest records this
  intentional annotation patch.
- [-] Produce reproducible Vulkan shader permutation builds and manifests.
  The Q2-compatible 3.1.5 linear-HDR/low-res-MV/unjittered/non-inverted SPIR-V
  profile contains ten public HLSL pass wrappers plus its separate
  AccumulateSharpen permutation. It is generated with pinned DXC, per-type
  compact per-pass resource bindings plus the FFX static-sampler range,
  SHA-256, byte-for-byte checked embedded C bundle, SPIR-V reflection,
  collision rejection, and `spirv-val` CTests. A
  checked catalogue explicitly maps the base and sharpened host
  permutations and rejects all others. General-purpose profile, wave64, and
  FP16 permutations remain.
- [x] Implement FSR3 3.1.5 temporal upscale behind the portable ABI and wire
  it into Q2RTX as experimental `flt_upscaler 3`. A validation-enabled
  1280x720 `base1` fixed-case run rendered coherent static and moving 50%
  output on RX 6800M, with no VUID, dispatch-failure, or fallback log entry.
  The coexistence crash with FSR3 1.1.4 was fixed by privately prefixing the
  complete SDK-3.1.5 source closure (`ffxVk315...`); both implementations are
  present in the final executable. GPU-assisted validation initially detected
  SPD workgroup-memory races in the generated generic-wave luma/shading
  pyramid modules; regenerating with AMD's LDS-only
  `FFX_SPD_NO_WAVE_OPERATIONS=1` permutation removed them. The regenerated
  bundle passed the bridge tests and a 30-second RX 6800M live GPU-AV run with
  no FSR race, VUID, dispatch-failure, or fallback output.
  The generic Vulkan pipeline factory now reflects each module's actual
  `OpEntryPoint` name rather than assuming `CS`: the SDK may supply direct
  SPIR-V declaring `main`, while the generated catalogue declares `CS`.
  Both forms are unit-tested. A fresh validation-enabled RX 6800M startup
  creates the FSR3.1.5 contexts at 640x480 and after resize at 1280x720 with
  no VUID, Vulkan error, or crash.
- [x] Integrate it into Q2RTX as a supported fallback to experimental FSR4.
  The fixed-Performance `flt_upscaler 1` path completed a visual and
  Vulkan-validation live test at 640x360 -> 1280x720 on RX 6800M.
  It now enables its internal auto-exposure graph rather than passing a null
  external exposure resource, matching the observed SDK 3.1.5 requirement.
- [-] Expand Q2RTX FSR3 validation to reset, resize, map transitions, camera
  cuts, weapon/emissive motion, and RenderDoc inspection before calling it
  production ready. Fixed-case GPU-assisted validation now covers the 3.1.5
  SPD modules after selecting their LDS-only permutation; lifecycle and image
  quality coverage remain. A direct `gamemap base2` transition from a live
  `base1` 3.1.5 session completed and rendered coherently with no VUID,
  dispatch-failure, or fallback log entry. An in-process 1280x720 -> 960x540
  -> 1280x720 resize recreated the context at each extent with no validation
  or fallback message. The Quality preset's 67% input scale also produced a
  coherent active 3.1.5 frame. A controlled 960x540 FSR3.1.5 + SDK-3.1.6
  FI/OF camera-cut smoke recorded reset `0x800`, returned to a history-valid
  v11/frame-1193 generated presentation without VUID/error, and captured a
  coherent scene at
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_camera_cut_fiog.png`.
  On 2026-08-31, a second `base1 -> base2` run kept public FSR3.1.5 plus
  SDK-3.1.6 FI/OF active after the expected map reset (`0x840`), returned to
  history-valid frame 1532, and recorded no VUID, FSR, dispatch, or presenter
  error; the full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_map_transition_fiog.png`.
  The same held-blaster/emissive script also passed under FSR3.1.5 + SDK-3.1.6
  FI/OF at 66.7 logical FPS, with no VUID, FSR, dispatch, or presenter error
  and a coherent capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_weapon_emissive_fiog.png`.
  A post-load KWin virtual-desktop focus-loss/recovery repeat on 2026-09-02
  had SDK-3.1.6 FI/OF active before the switch and recorded the expected
  focus reset `0x1000` at frame 879 with no Vulkan-validation, FSR, dispatch,
  or presenter error. Its scripted final screenshot is the standard Q2 menu,
  so it verifies lifecycle safety only rather than image quality. Physical
  input and broader image-quality coverage remain.
- [x] Expose shared FSR3/FSR4 Native AA, Quality, Balanced, Performance, and
  Ultra Performance discrete render-ratio presets. FSR4 v07 selects a matching
  separately compiled model, rather than one graph at arbitrary scale.
- [x] Expose the real FSR3 integrated RCAS pass as an independent modern
  `flt_fsr3_sharpening` [0,1] control.  A 0.20 live Performance run passed
  Vulkan validation on RX 6800M; legacy FSR1 sharpness remains inert.
- [x] Port the reproducible FSR3 1.1.4 analytical Optical Flow and Frame
  Interpolation compute passes into the reusable native Vulkan backend.  The
  public record-only API owns separate OF/FI backends and shared resources;
  it validates 352 FI + 56 OF lookups and submits reset plus temporal frames
  on RX 6800M with finite, fully-overwritten RGBA16F output and zero validation
  warnings/errors. A subsequent 45-second 960x540 FIFO-presenter smoke reached
  active generated→real presentation with no validation message.
- [-] Port/reconcile the newer public SDK 2.3 FSR3 3.1.6 analytical frame
  interpolation algorithms behind the reusable API. The fixed Vulkan profile
  and Q2RTX experimental scheduler selection work on SDR/RX 6800M; HDR,
  long-duration visual/lifecycle coverage, and a general-purpose profile
  remain.
- [-] Build a Linux/Windows portable explicit presenter. Q2RTX now has an
  experimental single-graphics-queue two-acquire/two-present path that renders
  the same queued UI on generated and real frames, with a blocking reserved
  second acquire and real-frame fallback. Distinct generated/real final-blit
  descriptors plus one UI upload eliminate the in-flight descriptor/host-write
  hazard, and same-queue order removes the former CPU replay wait. The reusable
  `ffx-vulkan::framegeneration-presenter-policy` target now exports the
  platform-neutral FIFO/image-count/acquired-pair/semaphore-ownership rules;
  Q2RTX uses it and an installed-package consumer plus 36-test standalone
  suite pass. It now also exports callback-based two-image acquisition: a
  failed second acquire leaves the first image explicitly available for the
  correct one-image fallback, and Q2RTX's core/device-group adapter uses that
  exact path. The policy now turns that retained outcome into a concrete
  ordered one- or two-slot presentation plan, including the reset/interpolation
  real-scene guard and the exact acquire semaphore for each slot; Q2RTX uses
  that plan at EndFrame. A fresh RX 6800M `vk_validation=1` FSR3.1.5 plus
  SDK-3.1.6 FI/OF forced-camera-cut smoke reached history-valid frame 1179
  with active FIFO paired presentation and no VUID, FSR, or presenter error;
  its full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_camera_cut_fiog.png`.
  Both installed-package consumers now invoke the plan API: the small C
  FSR4/presenter consumer and the C++ full FSR3/FSR4 stack configured, linked,
  and ran against a fresh prefix. The installed full-stack example no longer
  reaches into its sibling vendored example for its source: a copied standalone
  directory built and ran using only that installed prefix.
  Generic command-recording, queue-submit, and present callbacks remain
  application-owned.
  FFX dynamic-view ring now retains eight effect calls because FI performs both
  Prepare and Dispatch per real frame; a 36-second RX 6800M validation run
  stayed active with no VUIDs. It marks the presenter active only after an
  actual interpolated generated→real WSI pair, while the diagnostic cadence is
  Q2RTX's completed logical-render rate (and a nominal 2x generated rate only
  while active), never CPU submission timing for a WSI pair. When frame
  generation is requested, Q2RTX now recreates the
  swapchain with mandatory FIFO pacing even if `vid_vsync` is off, so Mailbox
  cannot replace or Immediate tear a generated→real pair; the active status
  makes that policy explicit. A fresh `vid_vsync=0`, 1280x720 FSR3.1.5+FG
  validation run selected FIFO, reached active generated→real presentation
  without VUID/error output, and produced a coherent viewmodel/HUD capture:
  `/home/fireburn/Screenshot_FSR315_FG_FIFO_20260820.png`. Replace this correctness-first WSI policy with
  measured timeline/presentation timing and a VRR/latency policy. The presenter now also
  accepts active public-SDK FSR3.1.5 as well as 3.1.4 because its inputs are
  provider-neutral. A 3.1.5 + analytical-FG live run at 1280x720 reported
  active generated→real presentation with validation clean. The combined
  1280x720 -> 960x540 -> 1280x720 resize also recreated both contexts and
  resumed active generated presentation with no validation, dispatch, or
  fallback error. A live `base1` -> `base2` map transition likewise retained
  active generated presentation with a coherent base2 frame and no validation
  or dispatch error. A paused Video menu now prevents two-image acquisition,
  publishes an inactive/zero-cadence status, resets history, and resumes a
  clean active presenter on return. A real KWin Wayland virtual-desktop
  focus-loss/recovery test now suspends paired presentation, records reset
  `0x1000`, and returns to active FI/OF with a coherent capture. Loading,
  low-FPS, and VRR/pacing coverage remain.
- [-] Validate frame interpolation at >=60 rendered FPS, including camera
  cuts, alt-tab, loading, and low-FPS hysteresis.
  The presenter's HDR mode now matches its post-tone-map HUDless source, and
  all temporal/acquire/present fallbacks explicitly reset FI/OF history before
  generation resumes. `flt_frame_generation_min_rendered_fps` defaults to 30,
  pauses after four low logical frames, then requires eight high frames above
  an adaptive re-enable floor (threshold +2 FPS plus measured FI cost; 0 is an
  explicit test override). The telemetry no longer treats CPU submission of a
  WSI pair as completed presentation rate. A 2026-08-25 960x540 SDK-3.1.6 run
  stayed active at a 30-FPS floor (76.9 logical FPS), while the same scene at a
  60-FPS floor settled in safe fallback after one probe rather than oscillating
  (166.7 logical FPS versus a learned 181.6-FPS re-enable floor). Both runs
  used validation with coherent screenshots and no VUID. On 2026-09-01, the
  current SDK-3.1.6 implementation repeated the 60-FPS-floor 3,900-frame
  check after the black-target repair: it made one guarded attempt, then held
  fallback at the conservative 240-FPS recovery requirement rather than
  toggling presentation while learning. The capture is
  `baseq2/screenshots/FSR315_perf_fg_60fps_sustained.png`. The legacy FSR3 1.1.4
  FI backend also stayed active through the shared 30-FPS gate (83.3 logical
  FPS) with a coherent validation-clean capture. A controlled camera-cut
  regression now proves that FSR4 itself recovers cleanly and that the paired
  presenter suppresses only the reset dispatch's undefined generated slot,
  presenting the real image twice before FI/OF resumes. A live forced
  240-FPS gate fallback→0-FPS-threshold recovery confirms the same guard when
  the provider alone requests reset (`v11/frame 910`, no VUID/error).
  A verified `pushmenu main` → `popmenu` lifecycle then exposed a binary WSI
  semaphore re-signal on resume; the reusable presenter policy now quiesces
  exactly once on either ordinary↔paired transition. The repeat resumed cleanly
  at `v11/frame 1985`, with no VUID/error and a coherent capture. The native
  FSR3.1.4 FI backend passed the same menu resume at v11/frame 1984, and
  public FSR3.1.5 + SDK-3.1.6 FI/OF passed at v11/frame 1982. A live `gamemap
  base2` scene transition retained FSR4-v07 + SDK-3.1.6 FI/OF through reset
  `0x840`, reaching active/eligible v11/frame 1579 with no VUID/error. A live
  960x540 -> 800x600 -> 960x540 windowed transition rebuilt FSR4 Quality and
  resumed FI/OF after both resets with no VUID/error. Remaining work is a
  genuinely sustained >=60-FPS scene plus latency/pacing, real focused-input
  cuts, alt-tab/loading, and broader lifecycle coverage. The same controlled
  cut also passed with public FSR3.1.5 plus SDK-3.1.6 FI/OF, reaching
  history-valid frame 1193 without VUID/error and a coherent capture at
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_camera_cut_fiog.png`.

## P3 — FSR 4.1.1 binary-provider research

- [x] Build deterministic DXBC/DXIL scanner, validator, optional extractor, and
  vkd3d capture manifest tooling in `tools/ffx_dxil` (9 tests passing).
- [x] Verify SDK 2.3 containers: upscaler 1028/885, denoiser 630/630,
  framegeneration 487/486 occurrences/unique, with zero malformed containers.
- [x] Verify Proton `amdxcffx64.dll`: 1294/1150, zero malformed containers.
- [x] Build a minimal AMD SDK 2.3 DX12 harness under Proton and dump only
  actually selected DXIL/SPIR-V with VKD3D_SHADER_DUMP_PATH. The source-only
  tools/ffx_dxil/reference_harness/fsr_provider_probe.cpp cross-compiles with
  MinGW, independently enumerates upscaler, frame-generation, and public
  denoiser/Ray-Regeneration providers when compiled against the full SDK,
  creates a 4.1.1 API context, and records one command-list dispatch without
  checking provider payloads into the tree. Its reduced-closure build reports
  a missing denoiser header rather than guessing a private ABI value. A fresh
  full-SDK 2.3.0/Wine-vkd3d run on selected RDNA2 adapter `1002:73df` listed
  3.1.5/2.3.4 upscalers, 3.1.6 frame generation, and no denoiser/RR provider;
  the 4.1.1 context selected analytical 3.1.5 and its auto-exposure-enabled
  metadata dispatch completed without an FFX warning, producing 11 paired
  DXIL/SPIR-V captures outside the tree. The explicit Frame Generation 4.0.1
  context likewise selected analytical 3.1.6, while the Ray Regeneration 1.2
  context returned `FFX_API_RETURN_NO_PROVIDER` (4). The probe now emits one
  stable `FFX_PROVIDER_PROBE_RESULT` line per effect, recording attempted,
  create/query return values, and selected name/ID. A fresh full-SDK run
  recorded `3.1.5` for upscaling, `3.1.6` for frame generation, and no
  queryable RR provider after return `4`; use this record format whenever a
  compatible adapter/driver becomes available.
- [-] Capture PSO/root signatures, resources/views, constants, uploads, pass
  order, dispatch dimensions, barriers, provider version, and feature queries.
  The controlled RX 6800M capture has the selected provider, a successful
  context/dispatch, 11 paired DXIL/SPIR-V shaders, and 25 provider-owned D3D12
  resource allocations recorded through AMD's public allocation callbacks.
  Root parameters, descriptors, uploads, barriers, and per-pass dimensions
  still require a RenderDoc/d3d12-replayer trace or explicit D3D12
  interception.
- [x] Determine the current official-provider result on RDNA2: SDK 2.3's
  4.1.1 API enumerated only analytical 3.1.5 and 2.3.4 providers and selected
  3.1.5 for a successful 640x360 -> 1280x720 dispatch. No FSR4 neural
  provider was offered by this RX 6800M/RADV/vkd3d-proton configuration.
  Re-test after relevant driver/provider changes; do not infer an INT8 path
  from embedded CS6.4 containers.
- [x] Make the measured official-provider boundary visible in Q2RTX rather
  than leaving source-v07 FSR4 or analytical FI/OF ambiguous. The read-only
  temporal diagnostics page now labels official FSR4.1.1, Ray Regeneration,
  and ML Frame Generation separately and reports concise signed-DX12 reasons;
  the 960x540 Vulkan-validation capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR_provider_diagnostics_audit.png`.
- [ ] Reproduce one fixed 4.1.1 upscale frame through the portable Vulkan ABI.
- [ ] Obtain legal/provenance review before redistributing any extracted model
  or shader payload; retain all required notices.

## P4 — Ray Regeneration and neural Frame Generation

- [R] Capture Ray Regeneration 1.2 selected shaders/provider schedule. Official
  support is RX9000+/DX12; the RX 6800M/RDNA2 DX12 probe does not select a
  neural provider, and the current official SDK marks Vulkan unsupported.
- [-] Add RR-compatible normals/roughness/material, diffuse/specular albedo,
  separate noisy direct/indirect diffuse/specular signals, hit distances,
  view-Z delta, and camera basis inputs while retaining ASVGF fallback.
  Contract v5 now publishes the documented compact material resources from
  checkerboard interleave (oct-normal/linear roughness/category 0 and sqrt
  diffuse/specular albedos), in addition to the dense generic material views.
  Contract v6 also publishes dense linear direct/indirect diffuse and
  direct/indirect specular partitions. Direct diffuse is Q2RTX's direct
  high-frequency channel, indirect diffuse its low-frequency SH coefficient,
  and direct specular is preserved before indirect accumulation. It still
  lacked per-lobe hit distance and dominant-visibility signals needed by an
  actual RR adapter. Contract v7 now carries the physically traced first
  indirect-lobe segment distance in the alpha channel of each indirect
  partition (finite 10,000-unit sky misses, negative for an untraced lobe).
  Direct alpha remains the documented non-negative undefined value. The
  primary direct-sun trace now also preserves its real blocker distance in a
  dense `R16_SFLOAT` image (FP16_MAX exposed, negative untraced), without a
  duplicate ray trace. Contract v10 pairs it with direction/radius and the
  exact resolved `sun_color_ubo` emission via the existing fence-retired
  primary-ray readback ring; it suppresses availability while that ring is
  stale after a sky update. The bridge exports real camera-position delta and
  maps all complete inputs into the reusable validator; a live RX 6800M run
  returned `issues=0x0` with dominant light included. It also explicitly
  converts Q2RTX's surface-to-sun shadow vector to the provider's
  light-to-target direction. The official neural
  provider remains outstanding. The reusable
  `ffx-vulkan::rayregeneration-contract` target now validates the equivalent
  provider-neutral image/alpha/camera metadata ABI (with an installed-package
  consumer test); contract v2 models all seven independently selectable RR
  signals. AO and specular occlusion are valid optional additions, while one
  of the four radiance partitions or dominant light remains required by a real
  provider. Q2RTX currently exports the four radiance partitions plus dominant
  light, not separate AO/specular occlusion. It does not certify pixel contents
  or implement a provider. Contract v4 adds concrete per-signal output and
  checkerboard validation, including output storage state/format and legal
  in-place aliases, so a provider has an explicit dispatch boundary. Contract
  v3 corrected the provider-facing motion
  conventions: a three-component UV/depth scale and previous-minus-current
  camera delta. Q2RTX contract v11 exports dense primary-surface
  `TEMPORAL_RR_MOTION` (PreviousUV-CurrentUV plus previous-minus-current
  signed-linear view-Z) rather than reusing `FLAT_MOTION`'s incompatible
  radial/reflection-denoiser Z channel. A provider dispatch remains outstanding.
- [R] Capture ML Frame Generation 4.0.1 provider schedule/model. Official
  support is RX9000+/Windows 11/DX12; the current official SDK has no Vulkan
  route and this RX 6800M/RDNA2 configuration cannot select the ML provider.
- [ ] Reuse the analytical-FG presentation system if an ML kernel becomes
  runnable; never intermingle UI with interpolated scene color.

## P5 — settings, UI, diagnostics, and documentation

- [x] Migrate stale loose shader cache safely on descriptor-layout upgrades.
  The launcher backs up `shader_vkpt` instead of deleting it, then permits the
  packaged matching `shaders.pkz` to load. This prevents old SPIR-V from being
  paired with the new RR-motion descriptor layout after a package upgrade.

- [x] Make the Gentoo live ebuild package the usable Vulkan feature set:
  system Vulkan/SDL/OpenAL/curl/zlib dependencies, external 7-Zip and shader
  tools, release media/shareware import, `/usr/bin` launcher/server,
  `/usr/share/quake2rtx` data, and current menu/config archive updates.  A
  package-style staged install passed. The release's all-rights-reserved media
  is correctly `bindist`/`mirror` restricted. The complete v07 FSR4 SPIR-V
  graph, six model initializer/pre-weight pairs, manifests, and MIT notice are
  tracked; both the ebuild and CMake's Linux-packaging configuration reject a
  missing model payload before an incomplete install can be created.  The
  ebuild independently asserts each per-model manifest and at least one
  model-prefixed SPIR-V module before and after installation. The
  runtime still publishes a specific missing v07 asset through
  `flt_upscaler_reason` instead of an ambiguous context failure. FSR3 and
  analytical FSR3 frame generation remain built in. On 2026-08-31, the exact
  ebuild system-dependency CMake flags cleanly configured and built the client;
  its install staged the client, menu/shader archives, and full FSR4-v07
  directory before correctly stopping at the raw checkout's intentionally
  absent `blue_noise.pkz`. `src_prepare` copies that required release asset
  before the ebuild invokes the install step.

- [-] Replace legacy FSR1 controls with independent settings:
  `Denoiser`, `Upscaler`, `Quality`, `Sharpening`, `Frame generation`, `Pacing`.
  The versioned loose menu migration now prevents an old user media archive
  from hiding new upscaler enum values (the verified FSR3.1.5 menu formerly
  rendered as `???`). A generic read-only `static` menu item can now bind a
  cvar safely without making ROM diagnostics editable; the Video menu exposes
  a compact `temporal diagnostics...` page for resolved upscaler/frame-
  generation reasons and generated/rendered cadence. It also shows separate,
  concise unavailable statuses for official FSR4.1.1, RR, and MLFG so those
  features cannot be mistaken for the runnable experimental Vulkan paths.
  Root build/CTest and the reusable Vulkan suite pass. A fresh 960x540
  Vulkan-validation run selected source-v07 FSR4 plus SDK-3.1.6 FI/OF before
  opening the page; it showed the active paths, all three official boundaries,
  and the expected menu-frame suspension without a VUID, parser, FSR,
  dispatch, or presenter error:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR_provider_diagnostics_audit.png`.
  Root CTest also verifies all three menu rows, their order, and their concise
  read-only cvar defaults (`temporal_provider_status`).
  There is deliberately no selectable pacing mode: active analytical FG always
  recreates the swapchain for FIFO so every generated->real pair is presented
  in order; Mailbox can replace a generated image and Immediate can tear it.
  The independent FPS safety floor is the relevant user-facing pacing knob.
- [-] Resolve requested versus active implementation through a central
  capability/fallback resolver and display one precise fallback reason.
  `flt_upscaler_active` and `flt_upscaler_reason` now publish the exact live
  outcome and are validated for both active FSR3 and nonlinear-projection
  fallback. The menu now has a read-only diagnostics page for both provider
  reasons and frame-generation cadence; its 2560x1440 Vulkan-validation
  capture is `/home/fireburn/Screenshot_temporal_diagnostics_window_20260820.png`.
- [x] Use per-swapchain-image render-complete semaphores for generated/real
  ownership, and quiesce the present queue exactly once when switching between
  ordinary and paired presentation. Per-image ownership alone did not prove
  that WSI had consumed a previous mode's binary wait; the live menu-resume
  VUID exposed that gap. The reusable policy test and the repeated RX 6800M
  menu smoke now prove the complete ownership rule.
- [x] Remove the inert old 0..2 “lower is sharper” menu control and truth-label
  the source-v07 model family as experimental and not FSR 4.1.1. The menu now
  records the actual native-Vulkan/RDNA2 boundary and exposes all RR substrate
  views through dominant-light blocker view 22.
- [x] Implement real FSR4 v07 RCAS and expose its independent [0,1] amount;
  retain the legacy FSR1 sharpness cvar solely as a no-op migration alias.
- [-] Reset histories on provider/preset/size/HDR/projection/camera-cut changes.
  Provider/preset/size/HDR/projection paths already reset. Contract v4 now
  emits `VKPT_TEMPORAL_RESET_CAMERA_CUT` for a >256-unit single-frame
  teleport, a >90-degree transform discontinuity, or a >0.35-radian lens
  jump; it stores only the last successfully presented camera, so normal
  motion remains reprojectable. A validation-enabled 1280x720 FSR3.1.5
  gameplay smoke stayed active and coherent with no VUID/error:
  `/home/fireburn/Screenshot_FSR315_camera_tracking_smoke_20260820.png`.
  The classifier is now factored into the installed reusable
  `ffx-vulkan::temporal-lifecycle` target, and its boundary test proves normal
  movement, exact 256-unit/90-degree/0.35-radian boundaries, discontinuities,
  and non-finite inputs. Its presentation-availability helper likewise makes
  both edges of a host focus/visibility/WSI availability change a shared
  reset decision. Q2RTX calls those helpers before provider dispatch and
  paired presentation. A rebuilt RX 6800M FSR4-v07 DRS + SDK-3.1.6 FI/OF smoke reached
  v11/history-valid frame 243 with no VUID/error after linking this target.
  An XTest free-camera lens-jump attempt did not reach the Wayland SDL window;
  the retained diagnostic correctly showed only the ordinary startup/map reset.
  `temporal_test_camera_cut` now supplies the controlled engine route: a live
  RX 6800M FSR4-v07 + SDK-3.1.6 FI/OF run recorded its `0x800` reset and then
  captured a coherent history-valid scene with no VUID/error. Continue to use
  a Wayland-native focused-input tool for a physical camera-motion route and
  threshold tuning.
- [x] Reset FSR history when temporal settings, including `flt_fsr_enable`,
  change so disable/re-enable cannot reuse stale recurrent state.
- [x] Log provider/effect version, backend, model, active permutation, input,
  and memory use. `fsr_diagnostics` is now a read-only in-game snapshot of the
  requested/resolved provider, temporal contract and image metadata, FSR3.1.4/
  3.1.5/FI backend state, source-v07 model/tier/permutation, current/effective
  bounded DRS scale, retained last temporal-reset frame/reason bits, and active
  FSR4 allocation accounting. Both reusable FSR3
  upscaler APIs now expose
  SDK-reported effect-owned allocation totals; the SDK-3.1.5 bridge counts its
  exact Vulkan allocation sizes and reports zero aliasable bytes because it
  does not alias heaps. The 2026-08-25 RX 6800M Quality capture reported a
  coherent 644x361 -> 960x540 dispatch, 28.27 MiB for FSR3.1.4 (6.16 MiB
  aliasable), 28.16 MiB for FSR3.1.5 (0 aliasable), and 40.01 MiB for FSR4
  (19.91 MiB activation scratch), and 30.58 MiB for consolidated FSR3.1.6
  FI/OF resident effect-owned resources (0 aliasable), with no VUID/error.
  The FI/OF total includes the shared bridge plus five lifecycle-owned images,
  and never double-counts its overlapping optical-flow/interpolation queries.
- [-] Update in-game help, `doc/client.md`, notices, licenses, and screenshots.
  The source menu and client cvar documentation now describe the discrete v07
  model family, independent FSR3/FSR4 RCAS controls, deprecated inert cvars,
  fallback behavior, and the three read-only official-provider boundary cvars.
  The reusable Vulkan subtree now has explicit upstream
  notices, a standalone-CI workflow, and a subtree-split publishing checklist;
  an isolated source copy passed 37/37 redistributable tests plus its installed
  external consumer. Release packaging and final screenshots remain.
- [x] Add `CLAUDE.md`, `TODO.md`, and a live `HANDOVER.md` maintenance rule.
