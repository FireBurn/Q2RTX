# FidelityFX Vulkan implementation TODO

Last updated: 2026-08-20 (Europe/London)

Status labels: `[x]` verified complete, `[-]` in progress/partially complete,
`[ ]` not started, `[R]` research/unknown viability.

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
- [ ] Capture every pass in RenderDoc/RGP and prove descriptor/resource bounds.
- [-] Take before/after screenshots and verify static detail, camera motion,
  weapon motion, emissives, disocclusions, resize, reset, and map transitions.
  Static output plus sustained forward/rotation motion are coherent and the old
  extent rectangle/history train is gone; a new RCAS-on capture is coherent.
  Resize, reset, map-transition, emissive, and disocclusion coverage remains
  outstanding.
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
- [x] Add the v07 DRS model to Q2RTX's existing bounded profiler-driven
  controller. It has its own validated graph/assets and never runs a static
  model at an arbitrary ratio; a 50%-bounded live RX 6800M run reported the DRS
  model active with Vulkan validation clean. Its sole render-size-change reset
  reason now preserves the DRS model's display-sized recurrent history; all
  other reset reasons remain conservative. The DRS input and feedback are
  capped at native 100%, including the controller-empty startup case; a
  `viewsize 150` live validation resolved to 1280x720 -> 1280x720 and stayed
  validation-clean.
- [x] Update the in-game video menu for the FSR3/FSR4 controls and deploy its
  loose source-menu override on this development system, where the installed
  `q2rtx_media.pkz` still contained the obsolete FSR 1.0 controls. Linux
  packaging now preserves NVIDIA's release media archive and updates its
  source-controlled menu/config entries, so it does not silently omit the
  released texture/model/audio tree.

## P1 — provider-neutral Q2RTX inputs and presentation split

- [x] Add a versioned reusable Vulkan temporal-frame contract and a live
  pre-dispatch validator for its ABI, stage/extents, image producer state,
  requested slots, motion/camera metadata, and depth conventions. Quality FSR4
  and FSR3+analytical-FG RX 6800M validation runs accepted the normal contract
  and stayed clean.
- [x] Add a reusable FSR4 v07 asset-set contract that selects matching model,
  tensor tier, initializer, pass-0 weights, RCAS, and SPD names without tying
  another Vulkan application to Q2RTX's filesystem/backend policy.
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
- [ ] Add device-depth and view-Z debug visualization.
- [x] Add dense primary-material reactive and transparency/composition masks
  for FSR3. Transparent, water/glass, warped, screen/camera paths feed both
  masks; view-model pixels feed reactive history rejection. A 1280x720 RX
  6800M run recreated the menu context, activated FSR3, and passed Vulkan
  validation with both masks registered.
- [ ] Add dedicated HUDless offscreen scene target and separate alpha UI target.
- [ ] Add per-input debug views in the live renderer. The contract validator is
  already executed immediately before FSR3, FSR4 v07, and analytical FG imports.
- [ ] Support/gather temporal inputs for device-group rendering.

## P2 — reusable native Vulkan FSR3

- [x] Add standalone `extern/ffx-vulkan` ABI, strict validators, capability
  probe, tests, license/provenance, and upstream import manifest.
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
  asset/contract tests were added.
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
  retaining multiple queue-ordered frames safely. Portable presenter and
  Q2RTX integration remain.
  GCC 16 and Clang 22 both build this coexistence gate while the reusable suite
  passes 27/27.
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
  descriptor/constant recycling and broad visual-quality coverage remain. The
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
  coherent active 3.1.5 frame; camera-cut and broader image-quality coverage
  remain.
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
  warnings/errors.
- [ ] Port/reconcile the newer public SDK 2.3 FSR3 3.1.6 analytical frame
  interpolation algorithms behind that reusable API.
- [-] Build a Linux/Windows portable explicit presenter. Q2RTX now has an
  experimental single-graphics-queue two-acquire/two-present path that renders
  the same queued UI on generated and real frames, with a blocking reserved
  second acquire and real-frame fallback. Distinct generated/real final-blit
  descriptors plus one UI upload eliminate the in-flight descriptor/host-write
  hazard, and same-queue order removes the former CPU replay wait. The reusable
  FFX dynamic-view ring now retains eight effect calls because FI performs both
  Prepare and Dispatch per real frame; a 36-second RX 6800M validation run
  stayed active with no VUIDs. It now marks the presenter active and reports
  rolling generated/rendered cadence only after an actual interpolated
  generated→real WSI pair; a two-real-frame fallback cannot inflate the
  diagnostics. When frame generation is requested, Q2RTX now recreates the
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
  clean active presenter on return. Camera-cut, alt-tab/loading, low-FPS, and
  pacing coverage remain.
- [ ] Validate frame interpolation at >=60 rendered FPS, including camera
  cuts, alt-tab, loading, and low-FPS hysteresis.
  The presenter's HDR mode now matches its post-tone-map HUDless source, and
  all temporal/acquire/present fallbacks explicitly reset FI/OF history before
  generation resumes. `flt_frame_generation_min_rendered_fps` defaults to 30,
  pauses after four low completed frames, and resumes after eight frames at
  threshold +2 FPS (0 is an explicit test override). Both open and forced-240
  FPS fallback branches were live-validated at 1280x720 without VUIDs.
  Latency/pacing and broad lifecycle coverage remain.

## P3 — FSR 4.1.1 binary-provider research

- [x] Build deterministic DXBC/DXIL scanner, validator, optional extractor, and
  vkd3d capture manifest tooling in `tools/ffx_dxil` (9 tests passing).
- [x] Verify SDK 2.3 containers: upscaler 1028/885, denoiser 630/630,
  framegeneration 487/486 occurrences/unique, with zero malformed containers.
- [x] Verify Proton `amdxcffx64.dll`: 1294/1150, zero malformed containers.
- [x] Build a minimal AMD SDK 2.3 DX12 harness under Proton and dump only
  actually selected DXIL/SPIR-V with VKD3D_SHADER_DUMP_PATH. The source-only
  tools/ffx_dxil/reference_harness/fsr_provider_probe.cpp cross-compiles with
  MinGW, enumerates providers, creates a 4.1.1 API context, and records one
  command-list dispatch without checking provider payloads into the tree.
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
- [ ] Reproduce one fixed 4.1.1 upscale frame through the portable Vulkan ABI.
- [ ] Obtain legal/provenance review before redistributing any extracted model
  or shader payload; retain all required notices.

## P4 — Ray Regeneration and neural Frame Generation

- [R] Capture Ray Regeneration 1.2 selected shaders/provider schedule.  Official
  support is RX9000+/DX12; native RDNA2 viability is unknown.
- [-] Add RR-compatible normals/roughness/material, diffuse/specular albedo,
  separate noisy direct/indirect diffuse/specular signals, hit distances,
  view-Z delta, and camera basis inputs while retaining ASVGF fallback.
  Contract v3 now publishes live-validated dense geometric normals, primary
  albedo, and roughness from checkerboard interleave; RR still needs its exact
  oct-normal/material convention and the separate noisy radiance/hit-distance
  signals.
- [R] Capture ML Frame Generation 4.0.1 provider schedule/model.  Official
  support is RX9000+/Windows 11/DX12; RDNA2 viability is unknown.
- [ ] Reuse the analytical-FG presentation system if an ML kernel becomes
  runnable; never intermingle UI with interpolated scene color.

## P5 — settings, UI, diagnostics, and documentation

- [x] Make the Gentoo live ebuild package the usable Vulkan feature set:
  system Vulkan/SDL/OpenAL/curl/zlib dependencies, external 7-Zip and shader
  tools, release media/shareware import, `/usr/bin` launcher/server,
  `/usr/share/quake2rtx` data, and current menu/config archive updates.  A
  package-style staged install passed.  The release's all-rights-reserved media
  is correctly `bindist`/`mirror` restricted.  It deliberately excludes
  untracked, provenance-sensitive v07 FSR4 model assets; FSR3 and analytical
  FSR3 frame generation remain built in.

- [-] Replace legacy FSR1 controls with independent settings:
  `Denoiser`, `Upscaler`, `Quality`, `Sharpening`, `Frame generation`, `Pacing`.
  The versioned loose menu migration now prevents an old user media archive
  from hiding new upscaler enum values (the verified FSR3.1.5 menu formerly
  rendered as `???`). A generic read-only `static` menu item can now bind a
  cvar safely without making ROM diagnostics editable; the Video menu exposes
  a compact `temporal diagnostics...` page for resolved upscaler/frame-
  generation reasons and generated/rendered cadence. Root build/CTest and the
  reusable Vulkan suite pass. A 2560x1440 Vulkan-validation run opened the
  page and showed the live startup/fallback state, zero cadence, and safe
  reason truncation without a VUID or parser error; capture:
  `/home/fireburn/Screenshot_temporal_diagnostics_window_20260820.png`.
  Pacing controls still need design and implementation.
- [-] Resolve requested versus active implementation through a central
  capability/fallback resolver and display one precise fallback reason.
  `flt_upscaler_active` and `flt_upscaler_reason` now publish the exact live
  outcome and are validated for both active FSR3 and nonlinear-projection
  fallback. The menu now has a read-only diagnostics page for both provider
  reasons and frame-generation cadence; its 2560x1440 Vulkan-validation
  capture is `/home/fireburn/Screenshot_temporal_diagnostics_window_20260820.png`.
- [x] Replace per-frame-slot present semaphores with per-swapchain-image
  render-complete semaphores, eliminating presentation-engine semaphore reuse
  validation errors and establishing the required ownership model for FG.
- [x] Remove the inert old 0..2 “lower is sharper” menu control and truth-label
  the source-v07 model family as experimental and not FSR 4.1.1.
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
  A deliberate camera-cut stimulus is still needed to validate the reset
  branch and tune thresholds.
- [x] Reset FSR history when temporal settings, including `flt_fsr_enable`,
  change so disable/re-enable cannot reuse stale recurrent state.
- [ ] Log provider/effect version, backend, model, active permutation, input,
  and memory use. The resolver already logs the fallback reason, and read-only
  generated/rendered rolling presentation cadence is published only for
  successful generated→real pairs through frame-generation FPS cvars.
- [-] Update in-game help, `doc/client.md`, notices, licenses, and screenshots.
  The source menu and client cvar documentation now describe the discrete v07
  model family, independent FSR3/FSR4 RCAS controls, deprecated inert cvars,
  and fallback behavior; release packaging, notices, and final screenshots
  remain.
- [x] Add `CLAUDE.md`, `TODO.md`, and a live `HANDOVER.md` maintenance rule.
