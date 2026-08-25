# FidelityFX Vulkan handover

Last updated: 2026-08-25, Europe/London.  Update this file at every meaningful
milestone and immediately before ending or transferring the session.

## Objective and truth status

The user asked for FSR3 and FSR4 plus all related features in Q2RTX, with
reusable native-Vulkan components and a demonstrable Vulkan implementation.

Current truth:

- Q2RTX has a read-only `fsr_diagnostics` console command for live evidence.
  It reports requested/resolved provider and reason, temporal contract/image
  metadata, FSR3.1.4/3.1.5/FI status, source-v07 FSR4 model/tier/permutation,
  and effect memory accounting without changing a context or recording work.
  Both reusable FSR3 upscaler APIs now publish their SDK-reported memory totals:
  the 3.1.5 bridge reports actual Vulkan allocation sizes and zero aliasable
  bytes because it deliberately uses independent allocations. In the
  2026-08-25 RX 6800M Quality run it reported an eligible 644x361 -> 960x540
  v07 dispatch, all required scene/motion/view-Z inputs, 28.27 MiB FSR3.1.4
  (6.16 MiB aliasable), 28.16 MiB FSR3.1.5 (0 aliasable), 30.58 MiB
  consolidated FSR3.1.6 FI/OF (0 aliasable), and 40.01 MiB provider-owned
  FSR4 allocation (19.91 MiB activation scratch). The log had no VUID/error
  and the captured output was coherent:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_fiof_diagnostics_memory_20260825.png`.

- Console screenshot readback now respects WSI ownership.  It no longer
  transitions `current_swap_chain_image_index` after its normal present;
  `IMG_ReadPixels[_HDR]_RTX` locally acquires an image, initializes it if
  needed, copies it to the existing host-readable target, and presents it with
  dedicated semaphores without mutating the renderer's current image index.
  The same 960x540 RX 6800M FSR4 Quality + RCAS invocation that previously
  reported an unacquired-present-image validation error now produced a
  full-frame coherent capture without VUID/error output:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_state_contract_20260825_validated.png`.

- The reusable FSR4-v07 provider now owns an explicit external-image state
  contract instead of silently assuming `GENERAL`. Before dispatch, the host
  registers each view's `VkImage`, current layout/stage/access, and requested
  restored state with `ffxFsr4VkSetExternalImageState`; FSR4 enters GENERAL for
  compute and restores imports at unregister, while rejecting an abstract FFX
  read/UAV role mismatch. Q2RTX registers color, motion, view-Z, and output.
  The RX 6800M lifetime test records a real
  `SHADER_READ_ONLY_OPTIMAL -> GENERAL -> SHADER_READ_ONLY_OPTIMAL` round trip;
  the timed FSR4 Q2RTX launch completed after the change.

- The source-v07 FSR4 provider now reflects each validated SPIR-V module into
  its own compact, binding-sorted descriptor-set layout. It uses an immutable
  sampler only where declared and rejects a dispatch before recording if a
  declared non-sampler descriptor lacks a write. This replaces the former
  permissive generic 44-binding layout. The Q2RTX client build, all 34
  standalone tests, and a fresh installed-package consumer build passed on
  2026-08-25; the timed RX 6800M Quality launch also completed without a
  reported validation error.

- Analytical frame generation now has a reusable separate alpha-UI path.
  `VKPT_IMG_TAA_OUTPUT` remains the HUDless display-resolution scene; when FG
  is active, Q2RTX renders the queued UI once into a lazily allocated per-frame
  slot `R16G16B16A16_SFLOAT` texture with premultiplied RGB/source alpha, then
  composites `ui.rgb + scene.rgb * (1-ui.a)` in the generated and real final
  blits. `VkptTemporalFrame.ui` now publishes
  `VKPT_TEMPORAL_UI_SEPARATE_TEXTURE` and full resource metadata. Allocation
  happens before command recording because the existing lazy-image helper has
  a one-time submission; if it fails, the known-good direct replay path is
  retained. A fresh 960x540 RX 6800M `vk_validation=1` FSR3.1.6 FIFO run
  allocated both slots lazily, reached active presentation, logged no VUID or
  validation warning/error, and captured a coherent scene/HUD at
  `/home/fireburn/Screenshot_FSR3_alpha_ui_lazy_20260825.png`.

- `extern/ffx-vulkan` is now a consumable Vulkan project rather than merely a
  Q2RTX-adjacent source tree. The source-v07 FSR4 provider, types, schedule,
  and asset helper are its canonical sources; Q2RTX links the same
  `ffx-vulkan::fsr4-v07-vulkan` static library. Its versioned
  `ffxFsr4V07…` API receives a caller-installed `FfxInterface` only during
  context creation, avoiding both Q2RTX globals and AMD SDK symbol collisions.
  `ffx-vulkan::effects` is the matching convenience integration target for the
  complete FSR3.1.4/3.1.5/3.1.6 + FSR4-v07 set. The target closure is now also
  installable: its private SDK sources compile into the package, which exports
  only versioned public targets and `include/`, so a clean C++ consumer has no
  Q2RTX/source-tree include dependency. `examples/full-stack` and the new
  `examples/installed-full-stack` both passed on 2026-08-25. This remains
  experimental v07, never FSR4.1.1/RR/MLFG.

- `ffx-vulkan::framegeneration-presenter-policy` now exports the reusable
  window-system-neutral part of the FSR3 presentation contract: FIFO selection
  for generated→real pairs, image-count requirements, pair validation, and
  per-image/per-GPU render-complete semaphore indexing. Q2RTX consumes it;
  its standalone test, an installed-package consumer, and a 45-second
  RX 6800M FSR3.1.6 FIFO run all pass without a VUID/error. Hosts still own
  acquire, submit, present, fences, and platform windowing.

- Q2RTX now provides `flt_temporal_debug_view`, a presentation-only selector
  for all dense temporal inputs: pre-tone-map scene HDR, current-to-previous
  motion, conventional device depth, positive-forward view-Z, reactive and
  composition masks, normals, albedo, and roughness. It reuses the final-blit
  descriptor with a semantic debug fragment shader and never mutates provider
  inputs or temporal history. View-Z is log-scaled to Q2RTX's 10,000-unit sky;
  motion displays signed pixel direction in RG and magnitude in B. Analytical
  frame generation is suspended while active, and debug mode removes its FIFO
  swapchain request. A `vk_validation=1` RX 6800M FSR4 Quality run at
  644x361 -> 960x540 captured a coherent view-Z image with no VUID/error:
  `/home/fireburn/Screenshot_temporal_view_z_20260820.png`.

- The same selector now also inspects current reconstructed FSR output and,
  for the v07 provider, its display-resolution previous history and
  reprojected pre/post bridge. `ffxFsr4GetDebugResource` is intentionally a
  narrow private diagnostic seam: it returns only a borrowed sampled resource
  description; it cannot schedule, clear, destroy, or mutate provider state.
  A validation-enabled RX 6800M FSR4 Quality capture after temporal warm-up
  shows coherent full-frame history with no VUID/error:
  `/home/fireburn/Screenshot_FSR4_v07_history_20260820.png`. These views are
  pre-tone-map debug images and therefore should not be brightness-compared to
  final presentation captures.

- After adding those presentation branches, a fresh RX 6800M
  `vk_validation=1` FSR3.1.6 generated→real FIFO run (debug view Off) again
  reached active presentation with no VUID/error and a coherent HUDless scene
  plus replayed UI: `/home/fireburn/Screenshot_FSR3_FG_post_debug_20260820.png`.

- The generated-present signal now uses the same `(swapchain image *
  device_count + GPU)` ownership index as the real present. Previously the
  generated path indexed by image alone: harmless on the RX 6800M single-GPU
  run, but an illegal cross-GPU semaphore alias on device-group builds. The
  corrected path passed a 45-second RX 6800M FSR3.1.6 FIFO run at 960x540,
  reached active presentation, and logged no VUID, validation warning, or
  error. Device-group temporal input gathering remains separately unsupported.
  The shared presenter was then exercised with its alternate reusable FSR3
  1.1.4 Optical Flow/Frame Interpolation backend for 45 seconds at 960x540;
  it also reached active FIFO presentation without a VUID, validation warning,
  or error.

- Presentation audit: `VKPT_IMG_TAA_OUTPUT` is already the dedicated
  display-resolution, tone-mapped-but-HUDless offscreen scene for both FI
  backends. The current presenter composes the same uploaded stretch-pic queue
  once into a reusable alpha texture and composites it over each generated and
  real swapchain image. The remaining architectural work is extending that
  contract to externally presented/non-Q2 UI—not creating another HUDless scene
  target.  Screenshot readback now has its own safe acquire/copy/present cycle.

- The reusable public FSR3.1.6 FI/OF dispatch API now exposes the SDK's
  optional external distortion field as a sampled `R16G16_SFLOAT` image whose
  values are `UV_after - UV_before`.  It preserves the existing neutral SDK
  1x1 field when null, validates/imports an application field when present,
  and retains that view through `RetireFrame` just like its color/output
  imports.  All four RX 6800M validation smokes cover a reset frame without
  the field followed by a temporal frame using it.  Q2RTX leaves it null until
  it has a real post-process distortion producer, so this changes no current
  renderer output while making the reusable API complete for that FI input.

- Existing renderer fallback builds and runs.
- The reusable tree now contains a separate, pinned public SDK v2.3.0 FSR3.1.5
  source closure (`upstream/ffx-2.3.0`) with both pristine and current
  SHA-256 manifests (166 source files). This now includes the official public
  FSR3 Frame Interpolation 3.1.6 and Optical Flow source closure. It is
  provenance-checked and compiles in an isolated object-only target under the
  private `ffxVk316...` namespace. All 18 public HLSL compute entry points
  (11 FI, 7 OF) regenerate as a source/output-hashed and Vulkan-1.2-validated
  portable SPIR-V bundle with compact per-pass descriptor ranges. The real
  fixed-profile SDK blob accessors return the embedded modules by pass and
  reject unsupported permutations; a dedicated test verifies all 18 SPIR-V
  blobs plus error and Wave64-query paths. A real 3.1.6 FI/OF host-graph test
  now links the public SDK core, creates 42 persistent resources and all 18
  embedded-SPIR-V pipelines, then destroys them cleanly. It also records the
  reset Optical-Flow → FI Prepare → FI Dispatch graph against a strict mock:
  15 dynamic registrations, 12 staged constant buffers, and 30 FI jobs. The
  18 modules also create actual Vulkan compute pipelines on the RX 6800M. The
  host is still not connected to a Vulkan scheduler/backend bridge or portable
  presenter. The existing SDK-2.3 resource/job bridge now accepts direct
  SPIR-V scheduler blobs before its 3.1.5 catalogue fallback, the reusable
  seam required to attach FI/OF without duplicating its allocation/barrier/
  descriptor executor. A RX 6800M bridge test supplies an unnamed FI blob and
  proves this direct-SPIR-V route rather than the fallback. The first real
  FI/OF context-creation attempt exposed its `rw_counters`/`r_counters`
  storage buffers. Commit `1c51d094` now binds those measured counter resources
  end-to-end through reflection, scheduler buffer tables, Vulkan storage-buffer
  descriptors, and compute-job recording. Reflection now parses SPIR-V
  variables, pointer storage classes, and image declarations to distinguish
  images, buffers, samplers, and CBVs; only the SDK `r_`/`rw_` convention
  selects buffer direction because Vulkan has one storage-buffer descriptor
  type. The bridge test now queries/enables storage-image
  read/write-without-format only when the logical device supports both bits;
  it creates the real FI and OF contexts and submits their initialization work
  on RX 6800M with validation clean. That proves context/pipeline/resource
  creation and now a real reset FI/OF frame: the RX 6800M test imports all
  application/shared images and records/submits Optical Flow -> FI Prepare ->
  FI Dispatch through the reusable bridge with zero Vulkan warnings/errors.
  Its source-derived SPIR-V now explicitly matches the SDK's R8_UINT luma,
  R16G16_SINT flow-vector, and R16G16_FLOAT dilated-motion storage formats;
  DX12's looser typed-UAV declaration rule was not portable to Vulkan. The
  same test then records/submits a non-reset temporal frame, proving that the
  bridge restores imported layouts between logical frames. Commit `05161ed6`
  made the reusable `ffx-vulkan::fsr3-vk-framegeneration-3.1.6` target provide
  the versioned opaque `create -> prepare -> dispatch -> retire -> destroy`
  lifecycle. Its public-only RX 6800M smoke runs both reset and temporal
  frames with zero validation warnings/errors in both RGBA8 and Q2RTX's
  RGBA16F presentation format. `RetireFrame(completedFrameId)` is deliberately
  required after the application's submission fence, retaining multiple
  queue-ordered imported-view sets safely until GPU completion. Q2RTX now
  selects this provider through `flt_frame_generation_backend 1` (default 0
  retains the 1.1.4 implementation), maps the RGBA16F HUDless color, R32F
  device depth, and RGBA16F motion input, and retires imports at the frame-slot
  fence. A validation-enabled 30-second RX 6800M 1280x720 `base1` run recreated
  its contexts then reached active FIFO paired presentation with no VUID,
  prepare, dispatch, or interpolation-skip messages. It now forwards the
  public sRGB/PQ/scRGB transfer mode and luminance range to both OF and FI;
  Q2RTX selects scRGB for HDR presentation. This remains experimental pending
  a physical-HDR run and broad visual/lifecycle coverage. The reusable
  suite additionally has a validation-enabled exact-Q2RTX fixture: RGBA16F
  scene/output and `FLAT_MOTION`-style RGBA16F motion with the vector in RG.
  The read-only active reason distinguishes the selected 1.1.4 and 3.1.6 FI/OF
  schedulers rather than reporting an ambiguous generic FSR3 label.
  The 3.1.6 wrapper must pass reciprocal display dimensions as `opticalFlowScale`;
  `{1,1}` produces a collapsed field extent and black generated frames. After
  the correction, a validation-enabled RX 6800M capture shows a coherent
  generated scene: `/home/fireburn/Screenshot_FSR316_FG_20260820.png`. A
  queued `base1 -> base2` transition likewise completed without validation or
  dispatch errors and captured a coherent generated post-transition scene:
  `/home/fireburn/Screenshot_FSR316_FG_base2_20260820.png`.
  Its direct SDK host-graph fixture was corrected to the same reciprocal scale.
  The existing
  upscaler host source compiles as an
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
  per-dispatch descriptors/constants until context destruction. It is now
  integrated experimentally into Q2RTX, but uses a private SDK-3.1.5 symbol
  namespace so it can coexist safely with the working 1.1.4 backend.
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
  modules are connected through the experimental resource/job bridge and
  Q2RTX path; generic profile/wave/FP16 permutations remain unfinished.
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
  run. The bridge now imports caller-owned 2D images with explicit Vulkan
  format/layout/state and creates only its image views; unregister restores
  the caller's layout/state and release never destroys caller image memory.
  The import/transition/restore test and the full scheduler test are both
  validation-clean. A reusable C upscaler lifecycle now owns that bridge and
  scheduler, exposes the three shared-resource descriptions, and records a
  first reset dispatch using nine caller-owned imported images; it emits the
  SDK initialization copies before frame work in the same caller command
  buffer. That full public-API dispatch is validation-clean and is now used by
  Q2RTX's experimental `flt_upscaler 3` path.
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
  After the SDK-3.1.5 additions shifted the renderer's global-image table, a
  fresh 1280x720 Performance + RCAS 0.50 + SPD-auto-exposure `base1` smoke
  again reported the exact active FSR4 v07 model with no VUID/error and a
  coherent viewmodel/HUD frame:
  `/home/fireburn/Screenshot_FSR4_v07_FIFO_table_smoke_20260820.png`.
  A subsequent Quality 858x482 -> 1280x720 `base1 -> base2` transition with
  RCAS 0.50 and SPD auto exposure also stayed validation-clean and produced a
  coherent post-transition viewmodel capture:
  `/home/fireburn/Screenshot_FSR4_v07_map_transition_20260820.png`.
  Its normal menu-to-game extent recreation from 640x480 to 960x540 selected
  a fresh 644x361 -> 960x540 graph and remained coherent under validation:
  `/home/fireburn/Screenshot_FSR4_v07_resize_20260820.png`.
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
  interpolated-generated→real pairs and resets on fallback. Frame-generation
  requests now recreate the swapchain in FIFO mode—even with `vid_vsync 0`—so
  Mailbox/Immediate cannot discard or tear the pair; the active status exposes
  `FIFO pacing`. This is correctness-first WSI policy, not a finished
  low-latency/VRR scheduler. A fresh 1280x720 `vk_validation=1`,
  `vid_vsync=0`, FSR3.1.5+FG `base1` run logged FIFO selection at both
  startup and resize, then `FSR3 analytical frame generation active (FIFO
  pacing)` with no VUID/error output. The active gameplay frame is coherent,
  including the viewmodel/HUD: `/home/fireburn/Screenshot_FSR315_FG_FIFO_20260820.png`.
  Ray Regeneration and neural frame generation remain incomplete.
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
  (`libdxcompiler.so 1.10`, build id `5322-c64a5165`).  The converter checkout
  is clean at `3b80347af` (`Fix Objective-C rewriter build with current LLVM`),
  a focused committed pair of Objective-C rewriter compatibility fixes on top
  of `c64a5165`; it is currently one local commit ahead of its `FireBurn`
  remote. Preserve that commit before rebuilding DXC.
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
- The 2026-08-20 fresh live startup also fixed an SDK-direct-SPIR-V edge case:
  the reusable pipeline factory reflects the sole GLCompute `OpEntryPoint`
  rather than hard-coding `CS`. The SDK-supplied runtime blob declared `main`,
  whereas the generated catalogue declares `CS`. Both are unit-tested; the
  RX 6800M now creates FSR3.1.5 contexts at startup and resize with validation
  clean instead of failing pipeline creation.

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
- All six model-specific 89,216-byte initializer and 1,024-byte pre-pass
  weight files are versioned with their graphs. Their SHA-256 checks are in
  each model manifest and their source MIT notice/provenance is retained in
  `baseq2/fsr4_shaders/LICENSE-FSR4-v07.txt`.

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
  It is gated to active FSR3 3.1.4 or public-SDK 3.1.5, one GPU, rectilinear projection,
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
  menu, and low-FPS testing.

## 2026-08-20 packaging and FSR3.1.5 checkpoint

- The public SDK-2.3/FSR3.1.5 Vulkan bridge is now linked into the Q2RTX
  client alongside the stable 1.1.4 path.  Its public header deliberately uses
  opaque bridge types, avoiding a collision with the bespoke v07 FSR4 types.
  Both SDK closures also exported the same unversioned C symbols; the initial
  mixed executable therefore crashed by invoking the 3.1.5 implementation
  with a 1.1.4 context ABI. The complete 3.1.5 source closure is now compiled
  with a private `ffxVk315...` symbol prefix, leaving the stable 1.1.4 ABI
  untouched; `nm` confirms both symbol families in the final executable.
  A full root build plus all standalone bridge/backend tests pass. A
  validation-enabled 1280x720 `base1` live run at 50% rendered coherent
  static and moving output with `flt_upscaler 3`; no VUID, dispatch failure,
  or fallback was logged. Proof captures are
  `/home/fireburn/Screenshot_FSR315_20260820.png` and
  `/home/fireburn/Screenshot_FSR315_motion_20260820.png`. A separate
  `flt_upscaler 1` regression launch also activated FSR3 1.1.4 cleanly.
  GPU-assisted validation then exposed shared-memory races in the generated
  generic-wave SPD luma and shading-change pyramid passes. The source
  generator now selects AMD's LDS-only SPD permutation
  (`FFX_SPD_NO_WAVE_OPERATIONS=1`), which uses explicit workgroup barriers.
  The regenerated 11-module bundle passed SPIR-V/pipeline/bridge tests, and a
  30-second `VK_LAYER_GPUAV_ENABLE=1` live run at 1280x720/50% had no FSR
  data-race, VUID, dispatch-failure, or fallback output. A post-change visual
  capture is `/home/fireburn/Screenshot_FSR315_GPUAV_clean_20260820.png`.
  A direct live `gamemap base2` transition from an active `base1` FSR3.1.5
  session then loaded the second map and rendered coherently without a VUID,
  dispatch-failure, or fallback log entry; capture:
  `/home/fireburn/Screenshot_FSR315_map_transition_20260820.png`.
  An in-process window resize from 1280x720 to 960x540 and back recreated the
  3.1.5 context at its physical 960x544 allocation extent and then 1280x720,
  with no validation or fallback message; post-resize capture:
  `/home/fireburn/Screenshot_FSR315_resize_20260820.png`.
  The shared Quality preset was also verified in the 3.1.5 path at 67% input
  scale (`viewsize 67`) with a coherent active frame; capture:
  `/home/fireburn/Screenshot_FSR315_quality_20260820.png`.
  Analytical FSR3 Optical Flow/Frame Interpolation no longer needlessly
  rejects this 3.1.5 selection: they consume the provider-neutral temporal
  contract and presentation color, not either upscaler's private history. A
  3.1.5 + `flt_frame_generation 1` 1280x720 live run created both contexts,
  reported `active=1` and a generated→real cadence (214.7/429.3 FPS), and
  stayed validation-clean; capture:
  `/home/fireburn/Screenshot_FSR315_FG_active_20260820.png`.
  The combined path has now also completed an in-process 1280x720 -> 960x540
  -> 1280x720 resize under `vk_validation=1`: both upscaler and FI/OF contexts
  recreated at each extent, resumed `active=1`, and reported 218.9/437.9 FPS
  after returning to 1280x720, with no VUID, dispatch failure, or fallback
  error in the full console log. Captures are
  `/home/fireburn/Screenshot_FSR315_FG_resize_960_20260820.png` and
  `/home/fireburn/Screenshot_FSR315_FG_resize_return_20260820.png`.
  A live `gamemap base2` transition from that active 1280x720 configuration
  also reconnected into base2, retained `flt_upscaler_active=3` and
  `flt_frame_generation_active=1`, and reported 174.0/348.0 FPS with no VUID,
  dispatch failure, or fallback error; capture:
  `/home/fireburn/Screenshot_FSR315_FG_map_transition_20260820.png`.
  Paused Video-menu frames are now excluded before the two-acquire presenter
  reserves a generated image. A live pause reported `active=0`, the explicit
  `paused/menu frame` reason, and zeroed cadence; closing the menu reset and
  resumed `active=1` at 57.5/115.0 FPS without a VUID or dispatch error.
  `/home/fireburn/Screenshot_FSR315_FG_menu_resume_20260820.png` captures the
  returned gameplay frame.
  This is still experimental lifecycle and visual-quality coverage, not a
  production readiness claim.
- `/home/fireburn/Overlay/games-fps/q2rtx/q2rtx-9999.ebuild` has been made a
  complete package recipe.  It imports NVIDIA's 1.8 release media/shareware
  assets, preserves the release media archive while updating Q2RTX's current
  menu/config entries, builds native FSR3 plus analytical FSR3 frame
  generation, and installs the wrapper/server in `/usr/bin` with data under
  `/usr/share/quake2rtx`.  It is restricted from binary distribution/mirroring
  because the release includes all-rights-reserved media.  The CMake install
  prefix now respects the package's `/usr` setting.  A clean package-style
  staging install verified every path, media/shader archive, and current menu
  entry.
- The ebuild now installs the complete FSR4 v07 graph and its six matching
  model initializer/weight pairs. A package-style staging test must continue
  to verify every model asset and the retained MIT notice. A fresh detached
  `46144a8d` Release staging build (with the ebuild's declared submodules and
  release data) installed exactly the 12 tracked `.bin` model assets and
  `LICENSE-FSR4-v07.txt` under `usr/share/quake2rtx/baseq2/fsr4_shaders`.
  The installed client then ran on RX 6800M with `vk_validation=1`, FSR4
  Quality, RCAS 0.50, and SPD auto exposure; it recreated from 640x480 to
  960x540, reported `FSR4 v07 INT8/DOT4 Quality active`, and logged no FSR4
  VUID, dispatch failure, or asset fallback. The active staged-package scene
  capture is `/home/fireburn/Screenshot_FSR4_v07_packaged_20260820.png`; it
  shows coherent scene/viewmodel/HUD output without the old render-extent
  rectangle or history trail. This still does not package official FSR4.1.1,
  RR, or ML-FG; their DX12/hardware limitations remain unchanged.
- Root CTest now includes `fsr4_v07_assets`: the dependency-free verifier
  confirms each of the six source-v07 initializer/pre-pass-weight pairs is
  present and matches the size and SHA-256 in its own model manifest. It passed
  alongside `fsr4_schedule` (2/2) and the separate reusable suite (31/31).
- The Video-menu test uncovered an upgrade-path issue: an older user-local
  `q2rtx_media.pkz` (or loose menu) can shadow the installed archive and make
  the new `flt_upscaler=3` value display as `???`. The package now installs a
  revisioned loose `q2rtx.menu`, and `q2rtx.sh` migrates it into the user
  `baseq2` override directory before launch, making a one-time
  `.pre-fsr-menu-update` backup. `Q2RTX_SKIP_MENU_UPDATE=1` opts out for
  manual menu management. A fixture proved the old menu was backed up
  byte-for-byte, and a live migrated-menu capture renders
  `FSR3 3.1.5 (experimental)` correctly:
  `/home/fireburn/Screenshot_FSR315_FG_video_menu_migrated_20260820.png`.
- The menu parser now supports a generic read-only `static --width <chars>
  "label" cvar` item. Unlike reusing an editable pair/spinner for ROM
  diagnostics, it displays a live cvar value, truncates it safely, owns its
  script allocations, and is never selectable. Video now links to a compact
  `temporal diagnostics...` page showing the resolved upscaler and
  frame-generation reasons plus rendered/generated cadence. Root build/CTest
  and the 20-test reusable Vulkan suite pass. A fresh 2560x1440
  `vk_validation=1` run opened the page after UI initialization and rendered
  the startup upscaler state, the expected no-world FG fallback, zero cadence,
  and an ellipsized long reason. Its log has no VUID, validation, parser, or
  FSR error. Capture:
  `/home/fireburn/Screenshot_temporal_diagnostics_window_20260820.png`.
- The temporal contract is now version 4. It stores the last successfully
  presented camera and sets `VKPT_TEMPORAL_RESET_CAMERA_CUT` on a conservative
  discontinuity (>256 Q2 units in one logical frame, >90° forward-vector
  change, or >0.35-radian vertical-FOV jump). Ordinary motion remains
  motion-vector reprojectable; providers discard history only for the marked
  frame. A fresh 1280x720 FSR3.1.5 `base1` gameplay run stayed active and
  coherent under `vk_validation=1` with no VUID/FSR error, capture:
  `/home/fireburn/Screenshot_FSR315_camera_tracking_smoke_20260820.png`.
  The actual cut branch still needs an explicit live stimulus/threshold test.
- The reusable FSR3 upscaler now rejects a device contract that merely has
  physical `shaderStorageImageWriteWithoutFormat` support but did not enable
  that feature on its logical device. The checked Accumulate modules require
  it. Q2RTX passes its enabled feature bit, while the RX 6800M public-API
  smoke explicitly verifies rejection with the bit cleared and successful
  two-frame dispatch with it enabled.

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
then rendered/presented for ten seconds with zero validation messages.  The
separate screenshot readback issue was subsequently fixed with a local WSI
acquire/copy/present cycle rather than a stale last-presented-image transition.

## Immediate next actions

1. Obtain a Wayland-capable capture tool (or X11-enabled SDL build) and record
   the live FSR4 pass graph. RenderDoc's available layer cannot attach to the
   current Wayland-only SDL build; do not interpret that tooling limit as an
   FSR4 dispatch failure.
2. Expand FSR4 visual evidence across camera/weapon/emissive motion,
   disocclusions, arbitrary resize, camera-cut reset, and long-duration runs.
   The fresh WSI-safe screenshot path is available for this work.
3. Validate frame interpolation at sustained >=60 rendered FPS, then exercise
   camera cuts, alt-tab/loading, low-FPS hysteresis, HDR, and pacing/latency.
4. Implement temporal input gathering for device-group rendering; the WSI
   semaphore ownership is fixed, but depth/material temporal input gathering
   remains single-GPU.
5. Continue measured official 4.1.1/RR/ML-FG provider research. On this
   RX 6800M/RADV/vkd3d setup the official API selects analytical FSR3 only;
   do not relabel source-v07 FSR4 as 4.1.1 or claim RR/MLFG support.

## Latest diagnostic evidence (2026-08-25)

- The source-v07 FSR4 backend now reports the provider's actual Vulkan memory
  requirements after context creation. A validation-enabled RX 6800M run at
  960x540 reported 40.01 MiB provider-owned allocations, including 19.91 MiB
  activation scratch, with no VUID, validation warning, or error. This count
  includes Vulkan allocation alignment and excludes opaque descriptor-pool
  driver overhead; it is not a heuristic estimate. The same run verified
  FSR3.1.4's 28.27 MiB (6.16 MiB aliasable), the SDK-3.1.5 bridge's exact
  28.16 MiB (0 aliasable), and the consolidated FSR3.1.6 FI/OF lifecycle's
  30.58 MiB (0 aliasable) through newly public reusable queries. FI/OF counts
  the shared backend once plus its five owned shared images; caller frames are
  imported and excluded.
- The reusable FSR4 v07 backend no longer recycles descriptor pools and
  host-visible constant-buffer bytes on a Q2RTX-specific frame cadence.
  `ffxFsr4VkBeginFrame` reserves one of three partitions and
  `ffxFsr4VkRetireFrame` releases it only through a host-completed frame ID.
  Q2RTX calls retirement immediately after its existing per-slot fence wait.
  The new RX 6800M lifetime test fills all three slots, rejects the unsafe
  fourth record, retires one completed ID, and then safely reuses it; a live
  Quality run remained active without VUID, validation warning, or error.
- Every FSR4 v07 module now has a dependency-free reflected ABI check before
  pipeline creation. It fails closed on an unexpected descriptor set, binding,
  or descriptor type rather than pairing an updated shader bundle with the
  old layout silently. The portable test covers all 288 generated
  preset/tier/pass modules and a deliberate pre-pass-as-model-pass mismatch;
  Q2RTX's menu-size and gameplay-size Quality contexts both passed it on the
  RX 6800M with no validation output.
- The same current build ran the separate Ultra Performance asset graph with
  RCAS 0.50 and SPD auto exposure, then the separate DRS asset graph at a
  fixed 50% controller range (with RCAS/SPD enabled). Both 960x540 RX 6800M
  runs selected the expected model in the live resolver and had no VUID,
  validation warning, or error.
- A clean standalone `extern/ffx-vulkan` Debug configure/build then passed all
  34 CTest cases on the RX 6800M. Coverage includes the pinned FSR3 1.1.4
  backend, public 3.1.5 bridge, 3.1.6 FI/OF API variants, generated SPIR-V and
  source hashes, FSR4 v07 asset selection, Vulkan-validation smokes, and the
  reusable generated-frame presenter policy. The
  first attempt correctly exposed a duplicate RenderDoc layer warning left by
  this session's failed capture registration; after removing only that
  session-created user manifest, the same unmodified build was fully clean.
- A temporary RenderDoc 1.40 command-line build succeeded, but its Vulkan
  layer is X11/XCB-only. The locally registered 1.39 layer is also X11/XCB-only.
  Q2RTX's SDL build reports `x11 not available`, while Wayland startup reports
  that RenderDoc lacks `VK_KHR_wayland_surface`; both capture attempts stop at
  `VID_Init` before FSR4 dispatch. No `.rdc`, proprietary payload, or capture
  artifact was retained or committed. Complete the deferred pass audit with an
  X11-enabled SDL build or a Wayland-capable capture tool.
- The April `Screenshot_20260417_022508.jpeg` and
  `Screenshot_20260417_022520.jpeg` artifacts were rechecked. The latter's
  large translucent old-scene rectangle is the former stale
  history/reprojection resource-lifetime failure, rather than a normal
  low-quality-upscale result. Fresh 2026-08-25 Quality, RCAS 0.50, and Native
  AA captures at `/home/fireburn/Screenshot_FSR4_{current,RCAS_current,native_current}_20260825.png`
  have coherent HUD/viewmodel and no old-frame rectangle or history trail.
  The dark corridor is similarly soft in Native AA and Quality, so that view
  does not evidence a remaining FSR4 temporal corruption or an RCAS failure.
  These are local diagnostic artifacts and deliberately are not versioned.

## Known risks and cautions

- The reusable source-v07 FSR4 backend remains experimental and has fixed
  limits; it now has explicit in-flight retirement, reflected per-pipeline
  layouts, and explicit imported-image state tracking. Its provider-owned Vulkan
  allocation accounting is live and verified at 960x540.
- Conventional device depth is available only for single-device rendering and
  is now inspectable through the temporal-input diagnostic. Device-group input
  gathering remains unimplemented.
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
