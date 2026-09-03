# FidelityFX Vulkan handover

Last updated: 2026-09-02, Europe/London.  Update this file at every meaningful
milestone and immediately before ending or transferring the session.

## Objective and truth status

The user asked for FSR3 and FSR4 plus all related features in Q2RTX, with
reusable native-Vulkan components and a demonstrable Vulkan implementation.

Current truth:

- AMD's current official SDK 2.3 page was rechecked after the native runtime
  validation. It lists FSR Upscaling 4.1.1, Frame Generation 4.0.1, Ray
  Regeneration 1.2.0, and Radiance Caching 0.9.0; only the analytical FSR3
  paths cover RDNA2. The first three new effects remain signed DX12 binaries,
  and Radiance Caching also requires an RX 9000-class GPU. Q2RTX now exposes a
  fourth read-only temporal-diagnostics row for official Radiance Caching,
  alongside FSR4.1.1, RR, and MLFG, so the menu covers the full current SDK
  feature list without implying an unsupported native Vulkan implementation.
  The controlled full-SDK DX12/Wine probe verifies the local behavior:
  Radiance Caching provider 0.9.0 enumerates on device 1002:73df but context
  creation returns 6 after vkd3d reports missing WMMA support. RR still returns
  no-provider 4, while 4.1.1/4.0.1 select analytical 3.1.5/3.1.6. This makes
  the fourth diagnostic row evidence-backed rather than a documentation-only
  claim.

- The community RDNA2 FSR4 route has now been isolated and tested without
  touching the game, SDK, or Proton cache. A new source-only legacy API probe
  is required because that source-v07 SDK is ABI-incompatible with the SDK
  2.3/FSR4.1.1 probe. On device 1002:73df, the historical signed loader
  normally listed only 3.1.5 and 2.3.4; cached amdxcffx64 v4.0.2 alone did not
  change this. With PROTON_FSR4_RDNA3_UPGRADE=1 and FSR4_UPGRADE=1, it listed
  and created 4.0.2. A synthetic FSR4 dispatch returned 0, but vkd3d reported
  missing WMMA support and command-list close failed 0x80070057, so this
  configuration has not executed a completed FSR4 GPU frame. It is a
  historical DX12/Wine provider-selection workaround, not official FSR4.1.1
  and not a portable Vulkan replacement. The native Vulkan source-v07 path in
  Q2RTX remains the runnable experimental FSR4 route on this machine.

- A fresh native-Q2RTX 960x540 validation-enabled source-v07 FSR4 Quality
  revalidation selected the exact 644x361 -> 960x540 INT8/DOT4 16-pass graph,
  automatic exposure, and the SDK-3.1.6 FI/OF presentation path. Diagnostics
  reported FSR4 active and dispatch eligible; no VUID, FSR, dispatch, or
  presenter error appeared. The new in-world motion/emissive capture is
  /home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_native_fiog_revalidation.png;
  visual inspection found coherent full-colour output with no strobe,
  monochrome image, or extent corruption. The initial inactive-window
  suspension was intentional and cleared before the active diagnostic state.

- A new FSR4 + SDK-3.1.6 FI/OF visual regression was found and fixed on the
  RX 6800M. The initial generated capture showed large green/white temporal
  blobs while the FSR4-only control capture was clean. FI/OF had incorrectly
  written its generated presentation into `VKPT_IMG_FSR_RCAS_OUTPUT`, which
  source-v07 FSR4 reserves for recurrent state. It now writes the dedicated
  `VKPT_IMG_FSR_FRAMEGEN_OUTPUT` image; final blit and debug capture use that
  same target. The replacement live capture
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_framegen_output_isolation_fix.png` is
  coherent and full-colour with FSR4 plus FI/OF active. It also records the
  RR complete-binding preflight as valid (`issues=0x0`). Root CTest now has a
  target-isolation gate and passed 4/4; no VUID, FSR, dispatch, or presenter
  error was logged.
  The same output target was then exercised with the public-SDK FSR3.1.5
  Vulkan upscaler plus SDK-3.1.6 FI/OF. Both paths became active, the RR
  binding preflight stayed valid, and the independent clean capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_framegen_target_isolation_control.png`.
  A local KWin one-shot activation then exercised the focused-to-inactive
  reset boundary on source-v07 FSR4; the resulting clean capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_framegen_focus_transition_clean.png`.
  The headless test terminal immediately reclaimed focus, so this is evidence
  for the safe transition only, not a replacement for an interactive sustained
  focused-present soak.

- The reusable `ffx-vulkan::fsr3-vk-framegeneration-3.1.6` wrapper now makes
  the same ownership rule fail closed for every host. `RecordDispatch` rejects
  an output image that aliases the prepared colour, depth, motion, or optional
  distortion field, and its API-smoke test asserts that rejection. The local
  standalone suite passed 38/38 and the Q2RTX root suite passed 4/4 before
  publication. It was published to `FireBurn/FSR-Vulkan` at
  `0c3e9fe9bd8e83c3d4b03c5d7a6545df8ccc7c5a`; GitHub Actions run `33667286306`
  completed configure, build, and test successfully on Ubuntu 24.04.

- The Gentoo `q2rtx-9999.ebuild` was re-audited on 2026-09-02. Its release
  archive root and imported `blue_noise.pkz`, media archive, shareware PAK,
  and player tree agree with the live ebuild's paths; it enables and checks all
  six source-v07 FSR4 model initializer/pre-weight/manifest/SPIR-V sets before
  and after installation. `bash -n` passed and `pkgcheck` reported only the
  expected `VisibleVcsPkg` notice for an unkeyworded live ebuild. No ebuild
  change was needed. A fresh ebuild-equivalent system-dependency CMake
  configure/build/install then staged the client, dedicated launcher/server,
  menu/media/shader archives, and the complete FSR4-v07 asset directory. The
  staged `shaders.pkz` contains the rebuilt `checkerboard_interleave` and
  `final_blit` modules; the loose asset directory contains all six model
  manifests, initializers, pre-pass weights, SPIR-V graphs, and the MIT notice.

- The staged-package validation also uncovered a real upgrade hazard: adding
  global FSR/RR images moves every sampled framebuffer descriptor binding, but
  an incremental shader build/archive could retain an old module. A fresh
  ebuild-equivalent build/install now removes the generated shader directory,
  recompiles it as one ABI unit, reflects each module before packaging, and
  recreates `shaders.pkz`. The installed archive contains exactly 47 modules
  and passed all 6,696 global-binding checks. The launcher marker is now
  `2026-09-framegen-output-v3`, preserving an old loose cache as a recoverable
  backup rather than letting it override the matching archive. This prevents
  the observed sampler-vs-storage-image validation mismatch at startup.

- The Video menu’s all-in-one RTX page was visibly vertically crowded at
  960x540. It now keeps only display basics and five navigation actions, with
  bounded Image Tuning, Temporal Upscaling, and Ray Tracing Features pages.
  `video_menu_layout` enforces this split (at most 15 compact controls per
  child page). The fresh staged-package capture at
  `/home/fireburn/Q2RTX/build/menu-package.DaC2CA/xdg/quake2rtx/baseq2/screenshots/FSR_video_menu_layout_960x540_fixed.png`
  is readable without the old squeezed rows and its validation-enabled launch
  logged no VUID or descriptor ABI error. `Q2RTX_DATA_DIR` is now honored by
  the binary itself before `/usr/share/quake2rtx`, completing the launcher's
  documented staged/portable override rather than accidentally mixing a test
  binary with system-installed data.

- The standalone `FireBurn/FSR-Vulkan` `main` branch is now at
  `0725754c39b9e7187d07b9620e4131e2d9bad3d5` (parent Q2RTX commit
  `1b80d558`). It adds `ffx-vulkan::radiancecache-contract`: a deliberately
  provider-neutral validator for application-owned inference, training, and
  two-counter buffers. It neither generates samples nor supplies neural
  inference/training. The source-tree and installed-consumer CTest suite passed
  38/38 immediately before publication. GitHub Actions run `33665262969` then
  independently completed configure, build, and test successfully on Ubuntu
  24.04.

- A new validation-enabled odd-size FSR4-v07 Quality + RCAS/SPD-auto-exposure
  + SDK-3.1.6 FI/OF lifecycle capture completed 960x540 -> 1133x717 ->
  960x540. The middle recreation selected the exact 760x480 -> 1133x717 graph;
  the final recreation restored 644x361 -> 960x540. FI/OF reactivated after
  both expected reset fallbacks, no VUID/FSR/dispatch/presenter error was
  recorded, and the full-colour captures are
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_odd_resize_1133x717.png`
  and `FSR4_odd_resize_restored.png`. This closes odd-size live-resize coverage;
  arbitrary reset/disocclusion coverage remains separate work.

- The source-only official-provider probe now ends every run with stable
  `FFX_PROVIDER_PROBE_RESULT` records for upscaling, frame generation, and
  Ray Regeneration. A freshly cross-compiled full-SDK 2.3 run on RDNA2 recorded
  successful 4.1.1/4.0.1 context creation but selected analytical `3.1.5` and
  `3.1.6`; RR creation returned `4` with no queryable provider. Future
  compatible-hardware revalidation must preserve these records and compare the
  selected name/ID, rather than equating 4.x API creation with neural support.
  Both full-SDK and deliberately reduced source-closure MinGW builds pass with
  warnings promoted to errors.

- Official sources were rechecked on 2026-09-02: FSR SDK 2.3 remains the
  current release, still marks SDK Vulkan support as unsupported, lists FSR
  Upscaling 4.1.1 for RX 7000-series discrete/RX 9000+ GPUs, and documents
  analytical FSR Frame Generation 3.1.6 as the automatic fallback on other
  hardware. There is therefore no newer official native-Vulkan/RDNA2 neural
  provider to integrate. Sources:
  https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK and
  https://gpuopen.com/amd-fsr-sdk/.

- Vulkan device-group enumeration now correctly accepts `VK_INCOMPLETE` when
  Q2RTX asks for only its selected first group. A fresh SLI-enabled startup
  reported `using device group 0 with 1 device(s)` without the former false
  Vulkan error. This is enumeration correctness only; device-group temporal
  input gathering remains an explicitly unsupported follow-up.

- The user-reported generated/real strobing has been reproduced and repaired.
  The direct FI-target probe proved that SDK-3.1.6 was alternating a black
  generated image with a real image on RX 6800M. The bridge had silently
  discarded GPU-job failures: it rejected SDK zero-size-as-remaining counter
  buffers, could not copy its previous interpolation source, used DX12-style
  storage-image formats in Vulkan SPIR-V, and rejected inactive static SPD
  mip declarations beyond a small image's real mip count. The bridge now
  handles those contracts, propagates failures, and the pinned shader overlay
  regenerates exact Vulkan formats. The current direct capture is full colour:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_generated_output_probe.png`.
  A weapon/emissive run, camera cut, map transition, and sustained 30-FPS-gated
  run all kept SDK-3.1.6 FI/OF active with no FI/OF or presentation VUID.
  Backend `1` is therefore the default; revision 2 migrates only the archived
  quarantine value once, while backend `0` remains an explicit 1.1.4
  compatibility choice. A fresh 2026-09-02 FSR3.1.4 + backend-0 regression
  also reached history-valid contract-v11 frame 308 with active 1.1.4 FI/OF,
  no VUID/FSR/dispatch/presenter error, and a coherent full-colour capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR3_FG_colour_after_rr.png`.

- Frame generation now consumes real SDL/Wayland focus changes rather than
  retaining a compositor-deferred generated/real pair. It suspends paired
  presentation while inactive and requests the new temporal reset bit
  `VKPT_TEMPORAL_RESET_FOCUS_CHANGED` (`0x1000`) on both focus edges. A
  2026-09-02 KWin virtual-desktop round-trip on RX 6800M logged the inactive
  suspension, reset at frame 1080, then active history-valid SDK-3.1.6 FI/OF
  at frame 1209 with no VUID, FSR, dispatch, or presenter error. The recovered
  capture is full-colour and coherent:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_focus_loss_recovery.png`.
  The same real KWin round-trip was repeated with source-v07 FSR4 Quality on
  the same RX 6800M: reset `0x1000` occurred at frame 1086, then FSR4 +
  SDK-3.1.6 FI/OF was history-valid and active at frame 1208 (31.2 logical
  FPS), also with no VUID, FSR, dispatch, or presenter error. Its coherent
  full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_focus_loss_recovery.png`.
  This edge decision is now exported by the reusable
  `ffx-vulkan::temporal-lifecycle` API as
  `ffxVkTemporalPresentationAvailabilityChanged(previous, current)`, alongside
  its camera-cut classifier. The standalone direct test and the isolated
  installed full-stack consumer both pass, so non-Q2 Vulkan hosts can reset
  all temporal providers on focus/visibility/WSI availability changes without
  importing SDL or KWin details.
  A second post-load KWin virtual-desktop round-trip was performed after the
  current standalone and Q2RTX suites passed (2026-09-02): SDK-3.1.6 was
  active before the switch, the inactive/active edges produced reset `0x1000`
  at frame 879, and no Vulkan-validation, FSR, dispatch, or presenter error
  was logged. Its scripted final screenshot is the ordinary Q2 menu, so retain
  it only as lifecycle evidence; it is not a new in-world image-quality
  comparison.

- The packaged Video menu was re-rendered at 960x540 with source-v07 FSR4 and
  SDK-3.1.6 FI/OF selected. It visibly labels the active upscaler as `FSR4 v07
  (experimental)`, keeps DRS and sharpening as distinct FSR4 controls, and
  identifies the presentation path as analytical frame generation with the
  3.1.6 scheduler and independent FPS gate. The capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_video_menu_audit.png`.
  Before opening the menu the log recorded FSR4 at 644x361 -> 960x540 and
  active FI/OF FIFO presentation; opening the menu correctly suspended paired
  presentation, with no VUID, FSR, dispatch, or presenter error.

- The standalone `FSR-Vulkan` README now opens with a feature-availability
  matrix: runnable FSR3 upscaling/FI-OF, experimental source-v07 FSR4,
  provider-neutral RR validation, and the explicit absence of official
  FSR4.1.1/MLFG/RR DX12 binaries. This makes the reusable project's boundary
  clear before integration. The current standalone build and full CTest suite
  pass 37/37 before publication.

- The published standalone project initially failed on GitHub's Ubuntu 24.04
  headers because `libvulkan-dev` 1.3.275 predates the public
  `VK_KHR_compute_shader_derivatives` declarations. The isolated compatibility
  header backports only those published data declarations and continues to
  probe the real runtime extension. The portability commit is parent
  `26b3cbd1` / published split `7afc7ee7`; the remote `portable-vulkan`
  workflow then completed configure, build, and tests successfully. The
  workflow subsequently moved from the deprecated Node-20 `checkout@v4` to
  `checkout@v7` (parent `e69c5ce0` / published split `6868a9cf`); its follow-up
  configure/build/test run also completed successfully.

- The Q2RTX Temporal Diagnostics page now makes the provider boundary visible
  without truncation at 960x540. It separates active `FSR4 v07 INT8/DOT4` and
  analytical FSR3.1.6 FI/OF from read-only `official FSR4.1.1`, `official Ray
  Regeneration`, and `official ML Frame Generation` rows. The concise values
  identify the signed-DX12/no-native-Vulkan boundary (and RX 9000+ for RR and
  MLFG). The fresh validation-enabled run logged active source-v07 FSR4 and
  FI/OF before entering the expected menu-frame suspension, with no VUID, FSR,
  dispatch, or presenter error; capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR_provider_diagnostics_audit.png`.
  `doc/client.md` documents the corresponding read-only cvars and explicitly
  distinguishes those statuses from runnable native-Vulkan selections. The
  root `temporal_provider_status` CTest locks the three menu rows, ordering,
  cvar names, and concise defaults against regression; the root suite now
  passes 3/3.

- The FPS safety gate no longer visibly oscillates while learning FI cost. A
  fresh RX 6800M `vk_validation=1` 60-FPS-floor / 3,900-frame run made one
  guarded SDK-3.1.6 attempt, entered the correct fallback, then stayed blocked
  at its conservative 240-FPS recovery requirement with no VUID/FSR error.
  The full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_perf_fg_60fps_sustained.png`.

- Source-v07 FSR4 Quality plus SDK-3.1.6 FI/OF also survived a fresh HDR
  context recreation on RX 6800M with `vk_validation=1`. Diagnostics reported
  both providers active and the HDR capture was written to
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_hdr_fiog.hdr`;
  there was no VUID, FSR, dispatch, or presenter error. The HDR test helper now
  correctly uses `screenshothdr` rather than the unsupported PNG command.

- A fresh FSR4 Performance -> bounded-DRS transition also rebuilt the v07
  graph at 960x540, selected the DRS model at 100%, and resumed SDK-3.1.6
  FI/OF FIFO presentation without an error. Its full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_drs_switch_fiog.png`.

- A validation-enabled in-process 960x540 -> 800x600 -> 960x540 resize rebuilt
  FSR4 Quality at each exact extent and resumed SDK-3.1.6 FI/OF after each
  expected temporal reset. No VUID, FSR, dispatch, or presenter error was
  logged; the post-resize full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_resize_fiog.png`.

- FSR3.1.5 now has the same explicit transient-resource lifetime discipline as
  the newer FI/OF path. Its reusable dispatch API accepts a monotonic frame ID
  and exposes `RetireFrame`; Q2RTX calls it only after the existing frame-slot
  fence has completed. This reclaims the bridge's per-dispatch descriptor sets
  and uniform buffers instead of retaining an unbounded run until shutdown. A
  3,900-frame live gate test exposed the old teardown path spinning in
  RADV/libdrm from `ffxVkFsr3_3_1_5DestroyBridge`; the corrected package builds
  and passes all 37 standalone tests.

- The reusable SDK-3.1.6 FI/OF API is now format-correct for both supported
  public presentation outputs: its output UAV is intentionally formatless,
  while compact internal images retain their exact Vulkan formats. This avoids
  invalidly imposing Q2RTX's RGBA16F target on an RGBA8 host. A clean standalone
  build passes all 37 CTests (including both RGBA8 and RGBA16F GPU smokes), and
  the Q2RTX RX 6800M `vk_validation=1` live run remains full-colour with active
  FIFO frame generation and no VUID or FSR errors.

- A frame-generation map transition exposed a separate Q2RTX trace-semaphore
  ownership bug: a skipped logical render could leave the current slot's trace
  signal pending, then re-signal it. The transfer handoff now consumes both
  the ordinary previous slot and any stale current slot before reuse. The
  repeat 3.1.5 + SDK-3.1.6 `base1 -> base2` run was validation-clean and its
  full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_map_transition_fiog.png`.
  A repeat on 2026-09-02, after the non-oscillating rate-gate repair, again
  reached active FI/OF on `base2`: the contract recorded reset `0x844`, then
  history-valid v11/frame 1489, logical cadence 55.6 FPS with the explicit
  zero-FPS test threshold, and no VUID, FSR, dispatch, or presenter error.
  The same capture is visibly full-colour and coherent at 62 FPS.

- The reusable presenter now carries acquisition through to an immutable
  ordered present plan. `ffxVkFrameGenerationBuildPresentPlan` produces either
  the normal one-slot fallback or generated-then-real two-slot order, retaining
  each slot's acquire semaphore and selecting the real scene for a reset or
  rejected interpolation slot. Q2RTX preserves the exact acquired pair through
  `EndFrame` and consumes that plan for both final-blit selection and the real
  slot's WSI wait. The presenter-policy unit coverage exercises paired,
  reset, fallback, and malformed-pair cases; the root build and standalone
  suite are clean (36/36). A fresh RX 6800M `vk_validation=1` FSR3.1.5 plus
  SDK-3.1.6 FI/OF smoke exercised the wiring through the controlled
  camera-cut reset: it reached active/history-valid contract-v11 frame 1179,
  wrote no VUID, FSR, or presentation-plan error, and exited normally. The
  full-colour capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_camera_cut_fiog.png`.
  A post-rate-gate repeat on 2026-09-02 consumed the simulated camera-cut reset
  `0x800` at frame 402 and returned to active, history-valid FI/OF by frame
  1167 at 71.4 logical FPS.  Vulkan validation remained clean and its 74-FPS
  capture is likewise full-colour and coherent.
  Both redistributable consumers also now execute this API: the small C
  FSR4/presenter consumer and C++ full FSR3/FSR4 stack configured, linked, and
  ran against an isolated freshly installed prefix. The latter is now actually
  self-contained: copying only `examples/installed-full-stack` to a temporary
  directory and building it against the prefix succeeds, rather than compiling
  a source file from its sibling vendored example as it previously did.
  A repeat of the FSR3.1.5 + SDK-3.1.6 menu-resume smoke on 2026-09-02 also
  logged the deliberate `paused/menu frame` suspension, reset at frame 1194,
  and resumed active FI/OF through history-valid frame 1958 at 71.4 logical
  FPS.  It emitted no VUID, FSR, dispatch, or presenter error and captured a
  full-colour coherent scene at
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_framegen_menu_resume.png`.

- `extern/ffx-vulkan` is now publication-ready as a standalone Git subtree.
  Its top-level CTest suite now installs the just-built package, copies the
  self-contained `installed-full-stack` example outside the source tree, then
  configures, links, and runs that copy against only the fresh prefix. An
  isolated copy with no Q2RTX parent configured, built, and passed all 36
  redistributable CTests, including that package test. The source-v07 FSR4
  shader/model bundle is deliberately absent from that archive. Its one
  payload-dependent layout test runs only when an explicit compatible external
  asset directory is supplied (the Q2RTX-local source build passes 37/37).
  The subtree now has standalone CI, notices, and `PUBLISHING.md` with the exact
  `git subtree split --prefix=extern/ffx-vulkan` procedure. Review the
  provenance boundary before any public push: it must never include v07 model
  payloads, AMD DLLs, extracted binary content, screenshots, or Q2RTX data.
  The reviewed release split was published to
  `https://github.com/FireBurn/FSR-Vulkan` `main` at `0411b8d8` on
  2026-09-01; it built cleanly and passed all 36 redistributable CTests.
  Subsequent reviewed 2026-09-02 splits added the platform-neutral
  presentation-availability lifecycle helper, the feature-availability
  matrix, Ubuntu-24.04 Vulkan-header compatibility, and CI maintenance without
  adding payloads. The current remote `main` is `6868a9cf`; see the current
  verification record near the top of this handover.
  The parent Q2RTX tree was rebuilt after the installed-consumer test was
  added, and its root CTest checks passed 2/2 (`fsr4_schedule` and
  `fsr4_v07_assets`) on 2026-08-31. A fresh 2026-09-02 verification of the
  current tree passed the same Q2RTX 2/2 suite and all 37 local reusable
  CTests, including the source-v07 asset-layout test that is intentionally
  omitted from the clean redistributed package.

- The development config's `flt_temporal_debug_view` is reset to `0` (Off).
  It had been left at view 22 while capturing the grayscale dominant-light
  blocker signal, which intentionally replaces normal colour output. A fresh
  960x540 FSR3 run with the selector Off rendered a full-colour `base1` scene;
  its scripted diagnostic console overlay does not affect the underlying scene:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/quake001.png`.

- SDK-3.1.6 FI/OF reaches real generated→real WSI presentation on RX 6800M
  with Vulkan validation and coherent 960x540 captures. The rate telemetry
  and gate now use Q2RTX's completed logical-render cadence—not CPU submission
  time for a generated/real WSI pair, which had misleadingly reported 84.2
  rendered / 168.4 generated FPS. The `flt_frame_generation_rendered_fps`
  cvar is logical input FPS; `generated_fps` is its nominal 2x rate only while
  a generated pair was active. At a 30-FPS floor, SDK-3.1.6 remained active
  (76.9 logical FPS) with no VUID/error:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR316_FG_30fps_active_gate_20260825.png`.
  The same shared gate also remained active with the older FSR3 1.1.4 FI
  backend at the 30-FPS floor (83.3 logical / 166.7 nominal FPS), validation
  clean and visually coherent:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR314_FG_30fps_active_gate_20260825.png`.
  At a 60-FPS floor this scene falls below the floor with FI enabled. The
  adaptive recovery gate observed its enabled cost, briefly probed once, then
  remained safely in real-frame fallback (logical 166.7 FPS, re-enable floor
  181.6 FPS) rather than continuously oscillating; capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR316_FG_60fps_steady_gate_20260825.png`.

- A camera-cut regression exposed a real FI/OF visual bug: presenting the
  generated image from the reset dispatch showed bright/unstable blobs, while
  the exact same FSR4-v07 run with FI/OF disabled was clean. The paired
  presenter now uses the real scene for that one reset slot while still
  recording the dispatch to seed optical-flow history. The next
  history-valid frame resumes interpolation. The controlled
  `temporal_test_camera_cut` run retained reset `0x800`, remained validation
  clean, and produced a coherent FSR4 + SDK-3.1.6 FI/OF capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_camera_cut_fg_guard.png`.
  A separate live 240-FPS rate-gate fallback→recovery run exercised the
  provider-reset path that has valid engine camera history; it resumed active
  FI/OF at contract-v11/frame-910 with no VUID/error and a coherent capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_framegen_rate_recovery.png`.
  The same controlled reset was repeated with public FSR3.1.5 upscaling plus
  SDK-3.1.6 FI/OF: it retained reset `0x800`, returned to history-valid
  contract-v11/frame-1193, reported active FIFO generation with no VUID/error,
  and captured a coherent scene:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_camera_cut_fiog.png`.

- The verified `pushmenu main` → `popmenu` lifecycle exposed a second real
  bug: resuming paired presentation could re-signal a binary render-finished
  semaphore before WSI consumed the normal/menu present wait. The reusable
  presenter policy now asks for a one-time graphics-queue quiescence whenever
  either side of ordinary↔generated presentation changes (not during either
  steady path). The repeat paused FI/OF for the menu, resumed it at
  contract-v11/frame-1985, and captured a coherent scene with no VUID/error:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_framegen_menu_resume_verified.png`.
  The native FSR3.1.4 FI path passed the same transition at frame 1984:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR314_framegen_menu_resume.png`.
  Public FSR3.1.5 plus SDK-3.1.6 FI/OF likewise resumed at v11/frame 1982,
  validation clean and visually coherent:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_framegen_menu_resume.png`.

- The reusable presenter policy now exports the actual two-image-acquisition
  outcome contract, not just arithmetic/policy helpers. Its host callback
  supports core or device-group acquire calls, returns a valid generated/real
  pair only for two distinct successful images, and preserves the first image
  for normal presentation if the second acquisition fails. The standalone
  policy test covers success, suboptimal, second-acquire fallback,
  first-acquire failure, and duplicate-image rejection. Q2RTX's adapter uses
  it; a fresh RX 6800M FSR3.1.5 + SDK-3.1.6 FI/OF camera-cut smoke passed with
  no VUID/error and a coherent capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_camera_cut_fiog.png`.

- A real `gamemap base2` scene replacement (not a full server restart) kept
  source-v07 FSR4 plus SDK-3.1.6 FI/OF enabled. It retained the expected
  scene/menu reset `0x840`, then reached an eligible/history-valid v11 frame
  1579 with no VUID/error and a coherent `base2` capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_framegen_map_transition.png`.

- The equivalent public FSR3.1.5 plus SDK-3.1.6 FI/OF `base1 -> base2` run is
  now covered too. It retained the expected map reset `0x840`, resumed at
  temporal-contract v11/history-valid frame 1532 with analytical generated→real
  presentation active, emitted no VUID, FSR, dispatch, or presenter error, and
  produced the coherent full-colour capture
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_map_transition_fiog.png`.

- Weapon and emissive motion now have comparable live evidence for both native
  upscalers. A 120-frame held-blaster input exercised projectile/muzzle
  movement while SDK-3.1.6 FI/OF was actively presenting generated→real pairs:
  source-v07 FSR4 was clean at 125 logical FPS and public FSR3.1.5 was clean
  at 66.7. Both runs had no VUID, FSR, dispatch, or presenter error; the
  full-colour captures are
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_weapon_emissive_fiog.png`
  and
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_weapon_emissive_fiog.png`.

- A live windowed `vid_geometry` transition from 960x540 to 800x600 and back
  rebuilt the FSR4 Quality context at 536x402 -> 800x600 and again at
  644x361 -> 960x540. SDK-3.1.6 FI/OF performed its expected fallback/reseed
  and resumed after each transition, with no VUID/error. The 800x600 capture is
  coherent and the script restored the original geometry before exit:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_resize_fiog.png`.

- Physical HDR presentation is now covered on the RX 6800M. With `vid_hdr=1`,
  source-v07 FSR4 Quality and SDK-3.1.6 FI/OF were active at 960x540, and
  `screenshothdr` saved a coherent linear capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_hdr_fiog.hdr`.
  Validation initially exposed HDR-only readback barriers that paired
  `HOST_READ` access with `ALL_COMMANDS`; `IMG_ReadPixelsHDR_RTX` now uses the
  valid HOST -> ALL_COMMANDS and ALL_COMMANDS -> HOST transitions already used
  by SDR capture. The repeat run had no VUID/error, rebuilt cleanly when HDR
  was disabled again, and exited normally.

- The updated reusable `ffx-vulkan::effects` package was reinstalled into a
  fresh temporary prefix after the reset-slot and mode-transition policy API
  changes. An independent `examples/installed-full-stack` CMake consumer then
  configured, linked, and ran using only that prefix; it proves the public
  FSR3.1.4/3.1.5/3.1.6 FI/OF, FSR4-v07, temporal-lifecycle, RR-contract, and
  presenter-policy closure remains installable rather than merely in-tree.

- A live source-v07 FSR4 Performance→Quality change while SDK-3.1.6 FI/OF was
  active revealed that availability was tested before the graph-rebuild code
  in dispatch, causing a permanent `quality model switch pending` fallback.
  The resolver now safely invokes the existing device-idle rebuild first. The
  corrected RX 6800M run rebuilt to 644x361 -> 960x540 Quality, resumed FI/OF,
  reached history-valid v11/frame 1585 without VUID/error, and captured a
  coherent scene:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_quality_switch_fiog.png`.
  The same path was then exercised from static Performance to the dedicated
  DRS graph: controller enabled at 50..100%, DRS model active at
  history-valid v11/frame 1588, SDK FI/OF resumed, and no VUID/error occurred:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_drs_switch_fiog.png`.

- Q2RTX has a read-only `fsr_diagnostics` console command for live evidence.
  It reports requested/resolved provider and reason, temporal contract/image
  metadata, the most recent retained temporal reset frame/reason bits,
  FSR3.1.4/3.1.5/FI status, source-v07 FSR4 model/tier/permutation, and effect
  memory accounting without changing a context or recording work.
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

- Camera-discontinuity handling is now a tested reusable package target rather
  than an untested Q2RTX-local heuristic. `ffx-vulkan::temporal-lifecycle`
  classifies a >256-unit teleport, >90-degree turn, >0.35-radian lens jump, or
  non-finite input as a cut; Q2RTX adapts its resolved camera into that API
  before setting `VKPT_TEMPORAL_RESET_CAMERA_CUT`. Exact boundaries remain
  ordinary reprojectable motion, including a float-safe FOV comparison. The
  standalone test covers each case and the Q2RTX client links the same target.
  A live free-camera/teleport capture is still needed to prove the end-to-end
  game-input route. An attempted XTest mouse-lens jump did not reach the
  Wayland SDL window (the retained reset remained the ordinary startup/map
  reset at frame 11, `0xf3`); use a Wayland-native focused-input tool or a
  controlled engine camera command rather than counting it as a cut test.

- The post-link RX 6800M live regression is clean: the rebuilt Q2RTX client
  selected source-v07 FSR4 DRS plus active SDK-3.1.6 FI/OF, reached temporal
  contract v11/history-valid frame 243, and logged no VUID/error. The current
  full-colour coherent capture is
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_v07_colour_after_rr.png`.

- Official-source feasibility was rechecked on 2026-08-31 without altering the
  user's SDK trees. AMD's current FSR SDK 2.3.0 lists FSR4.1.1, ML Frame
  Generation 4.0.1, and Ray Regeneration 1.2.0; it also adds official FSR4
  support for RDNA3/RX 7000 discrete GPUs. However, the SDK's own known-issue
  table still says Vulkan is unsupported, while its FSR4 page requires the
  signed HLSL/CS_6_6 provider and RX 7000/RX 9000 or later. AMD's RR sample
  requires RX 9000+, Windows 11, and DX12. The locally retained FSR4 fork is
  SDK 2.0.0 and contains only DX12 signed DLLs (`upscaler`, `framegeneration`,
  and loader), no RR module. Combined with the RX 6800M/RDNA2 DX12 probe
  selecting only analytical FSR3.1.5/2.3.4, there is no honest native-Vulkan
  neural FSR4/RR/MLFG provider to attach on this machine. Keep the working
  native FSR3 FI/OF and source-v07 FSR4 paths distinct from that unavailable
  binary feature set. A fresh full-SDK probe made the distinction direct:
  Frame Generation API 4.0.1 created only after selecting analytical 3.1.6,
  while Ray Regeneration API 1.2.0 creation returned
  `FFX_API_RETURN_NO_PROVIDER` (4). Sources:
  https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK and
  https://gpuopen.com/amd-fsr-sdk/.

- The menu revision now exposes all provider-neutral RR input views through
  view 22 (`RR dominant light visibility`). Its FSR4 text distinguishes the
  working source-v07 Vulkan path from official FSR 4.1.1: AMD documents the
  latter as a signed DX12/Windows provider for RX 7000-series discrete GPUs or
  newer, so it cannot run in native Vulkan Q2RTX on the RX 6800M/RDNA2.

- Console screenshot readback now respects WSI ownership.  It no longer
  transitions `current_swap_chain_image_index` after its normal present;
  `IMG_ReadPixels[_HDR]_RTX` locally acquires an image, initializes it if
  needed, copies it to the existing host-readable target, and presents it with
  dedicated semaphores without mutating the renderer's current image index.
  The same 960x540 RX 6800M FSR4 Quality + RCAS invocation that previously
  reported an unacquired-present-image validation error now produced a
  full-frame coherent capture without VUID/error output:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_state_contract_20260825_validated.png`.

- A post-RR-resource-change visual regression run selected FSR4 source-v07
  Performance at 480x270 -> 960x540, reached history-valid temporal contract
  v11/frame 797, and emitted no validation error. The capture is full colour
  with a coherent HUD/viewmodel and no internal-resolution rectangle:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_v07_colour_after_rr.png`.
  The on-screen label correctly says `FSR4 v07 INT8/DOT4 Performance`; do not
  describe it as official FSR 4.1.1.

- The matching post-RR-resource FSR3 analytical frame-generation run selected
  native FSR3 1.1.4 plus active FIFO generated->real presentation at 90.9
  logical FPS (minimum-render gate explicitly disabled for the smoke). It
  reached temporal contract v11/history-valid frame 401 with no validation
  error and a full-colour coherent HUD/viewmodel capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR3_FG_colour_after_rr.png`.

- The public-SDK FSR3.1.5 Vulkan bridge also completed a post-RR-resource
  visual smoke at Quality, reaching temporal contract v11/history-valid frame
  864 without a VUID/error. Its 960x540 `base1` capture is full colour and
  coherent, including the HUD and viewmodel:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_colour_after_rr.png`.

- The combined FSR3.1.5 Quality + FSR3.1.4 analytical-FG path also passed a
  fresh post-RR-resource RX 6800M smoke. It made the expected startup fallback
  before a temporal contract existed, then reached active FIFO presentation at
  v11/history-valid frame 402 with no VUID/error. The 960x540 result is full
  colour and coherent:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR315_FG_colour_after_rr.png`.

- The newer public FSR3.1.6 FI/OF backend passed the same current-layout
  smoke with FSR3.1.5 Quality. It made the expected initialization fallback,
  then reached active FIFO presentation at v11/history-valid frame 401
  (83.3 logical FPS, gate disabled), with no VUID/error. Its 960x540 capture
  is full colour and coherent:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR316_FG_colour_after_rr.png`.
  A longer current-layout run remained active through frame 2150 at the normal
  30-FPS safety floor (52.6 logical / 100 nominal generated FPS), with no
  VUID/error and another coherent capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR316_FG_60s_after_rr.png`.
  It is therefore the default scheduler for fresh configs; existing archived
  `flt_frame_generation_backend` choices are intentionally not migrated. An
  isolated fresh XDG profile (linked game data but no config) verified that
  omitting the cvar selects SDK 3.1.6 and reaches active v11/frame 401 FIFO
  presentation, full colour and validation-clean:
  `/tmp/q2rtx-fresh-fg.F4n4pa/quake2rtx/baseq2/screenshots/FSR316_fresh_default.png`.

- The source-v07 FSR4 super-resolution path now also feeds both public
  analytical FG schedulers through the existing display-resolution HUDless
  `TAA_OUTPUT` contract. This was previously blocked only by an unnecessary
  FSR3-only availability gate, despite FSR4 v07 already producing the same
  contract. On RX 6800M, FSR4-v07 Performance + SDK-3.1.6 FI/OF reached active
  FIFO presentation at v11/frame 402, then stayed active through frame 2246
  with the 30-FPS gate (58.8 logical / 111.1 nominal generated FPS), no
  VUID/error, and a coherent full-colour capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_FSR316_FG_60s.png`.
  The retained FSR3.1.4 scheduler also reached active FIFO presentation at
  v11/frame 402 with no VUID/error and a coherent capture:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_FSR314_FG_smoke.png`.
  This is source-v07 FSR4 upscaling plus public analytical FSR3 FI/OF, not
  official FSR Frame Generation 4.0.1/MLFG.

- The dynamic-size FSR4-v07 model also composes with SDK-3.1.6 analytical FG.
  A RX 6800M `vk_validation=1` run selected `model=drs`, kept FSR4 DRS on,
  reached active FIFO presentation at v11/history-valid frame 797, and emitted
  no VUID/error. Its 960x540 capture is full colour and coherent:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_DRS_FSR316_FG_smoke.png`.
  `fsr_diagnostics` now exposes the renderer's read-only current/effective DRS
  controller scale plus target and bounds. A forced 240->30-FPS controller
  transition proved the full lifecycle under active FI/OF: it reached
  480x270/50% at frame 401, then recovered to 960x540/100% at frame 1034;
  both states had a history-valid v11 contract and no VUID/error. The recovered
  full-colour capture is:
  `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/FSR4_DRS_FG_transition_telemetry.png`.

- The RR motion resource added one global image descriptor. Because loose
  user `baseq2/shader_vkpt` files override packaged `shaders.pkz`, an old cache
  would otherwise be paired with the new host descriptor layout and produce a
  validation type mismatch at pipeline creation. `setup/q2rtx.sh` now has a
  versioned shader-layout migration: it moves the loose directory to a dated
  `pre-2026-09-framegen-output-v3` backup and lets the matching packaged archive
  load. Advanced users can retain a deliberately matching custom set with
  `Q2RTX_SKIP_SHADER_CACHE_MIGRATION=1`. The Gentoo ebuild verifies that this
  launcher migration is installed.

- Source-tree Linux packaging now fails closed for the complete FSR4-v07 asset
  set, matching the ebuild's checks. With
  `CONFIG_VKPT_INSTALL_FSR4_V07_ASSETS=ON`, CMake requires the retained MIT
  notice plus each of the six model-specific initializer, pre-weight, and
  manifest files before generating install rules; it no longer treats a merely
  present `fsr4_shaders` directory as proof of a runnable installation. A
  fresh ebuild-equivalent system-dependency CMake configuration passed.  On
  2026-08-31, that exact system-dependency configuration also rebuilt the
  client and began staging its installation: client, menu/shader archives, and
  all FSR4-v07 contents were staged. It then correctly stopped only at the
  raw source checkout's deliberately absent `blue_noise.pkz`; the ebuild's
  `src_prepare` copies that release asset before it calls CMake install.

- The standalone reusable `ffx-vulkan` package now offers the corresponding
  explicit opt-in asset delivery path. A package builder supplies
  `FFX_VK_PORTABLE_FSR4_V07_ASSET_DIR` and enables
  `FFX_VK_PORTABLE_INSTALL_FSR4_V07_ASSETS`; CMake verifies the notice and all
  six initializer/pre-weight/manifest sets, installs only those model-prefixed
  SPIR-V bundles under `share/ffx-vulkan/fsr4-v07`, and exposes the path as
  `FFX_VK_FSR4_V07_ASSET_DIR` to `find_package` consumers. A standalone
  reduced-closure build/install verified all six bundles and verified that
  unrelated generic `fsr4_initializers.bin`/`fsr4_pre_weights.bin` files were
  not copied. A fresh minimal C consumer then used `find_package(ffx-vulkan)`,
  verified the exported installed directory and all six manifests, and linked
  `ffx-vulkan::fsr4-v07-vulkan` successfully.

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

- Temporal contract v5 adds three compact, provider-neutral Ray
  Regeneration-compatible material inputs: `R8G8B8A8_UNORM` oct-normal / linear
  roughness / category 0, sqrt diffuse albedo, and sqrt specular albedo.
  `checkerboard_interleave` produces them with the documented BRDF
  approximation and publishes explicit compute visibility barriers. A live
  RX 6800M `fsr_diagnostics` invocation reports all three as valid 644x361
  inputs with octahedral/sqrt/four-category metadata. This is only an RR input
  foundation; it does not make the official neural RR binary available. The
  selector views were live captured with `vk_validation=1` on RX 6800M:
  packed normal/roughness/material, diffuse albedo, and specular albedo are
  coherent in `/home/fireburn/.local/share/quake2rtx/baseq2/screenshots/` as
  `RR_normal_rough_material_20260825.png`, `RR_diffuse_albedo_20260825.png`,
  and `RR_specular_albedo_20260825.png`.

- Temporal contract v6 adds four dense linear noisy-radiance partitions for a
  future RR/decoupled-denoiser adapter: direct diffuse, indirect diffuse,
  direct specular, and indirect specular. Q2RTX's direct-lighting pass now
  preserves direct specular before the indirect pass accumulates into SPEC;
  interleave exports the direct high-frequency channel, low-frequency SH
  coefficient, preserved direct specular, and nonnegative SPEC remainder.
  This deliberately documents the renderer's semantics rather than relabeling
  the SH coefficient as a complete official RR signal. A 2026-08-25 RX 6800M
  `vk_validation=1` FSR3 run reported all four `R16G16B16A16_SFLOAT` images
  valid at 644x361, temporal contract v6/history valid, and no VUID/error:
  `baseq2/logs/RR_radiance_contract_20260825.log`. Per-lobe hit distance and
  dominant visibility remain outstanding, as does any neural provider.
  Values 16-19 in `flt_temporal_debug_view` now render those four channels
  directly for visual inspection. The validation-enabled direct-diffuse
  capture is visibly populated and deliberately noisy, as expected before a
  denoiser: `baseq2/screenshots/RR_direct_diffuse_20260825.png`. This exposes
  the renderer's raw partitions; it is not a claim of official RR support.

- Temporal contract v7 completes the directly available RR radiance-alpha
  data without inventing a signal: the alpha of each indirect partition is the
  physically traced first lobe segment distance. A sky/environment miss uses
  Q2RTX's explicit finite `10000` trace distance; a lobe not traced this frame
  is negative. Direct alpha remains non-negative but otherwise undefined, per
  AMD's current contract. This first-segment association remains correct for
  Q2RTX's later-bounce energy accumulation. Dominant-light visibility is the
  remaining signal-level gap; official RR itself remains DX12/RX9000-only.
  The menu's views 20/21 present the two alpha channels as log-scaled
  grayscale for inspection. A 960x540 RX 6800M `vk_validation=1` v7 run
  reached history-valid frame 23 with no VUID/error and a populated diffuse
  distance capture:
  `baseq2/screenshots/RR_indirect_diffuse_hit_distance_20260825.png`.
  Diagnostics are in `baseq2/logs/RR_hit_distance_debug_20260825.log`.

- Temporal contract v10 completes Q2RTX's provider-neutral dominant-light
  data bridge without tracing a second ray. The direct pass writes its actual
  primary sun blocker distance to `R16_SFLOAT` (FP16_MAX exposed; negative
  untraced), and the host now obtains the exact `sun_color_ubo.sun_color`
  emission through the existing primary-ray fence-retired readback ring—not
  through the pre-atmosphere/cvar `sun_light.color`. The signal stays
  unavailable during the readback ring's stale window after a physical-sky
  update; direction, radius, emission, and the image are published together
  only once they correspond to the resolved sky. View 22 remains a log-scale
  visualizer for the image. A 960x540 RX 6800M `vk_validation=1` run reached
  v9/history-valid frame 874 with a valid 644x361 R16 image and exact metadata
  `surface-to-light=(0.8754 -0.2346 0.4226)`,
  `emission=(0.4834 0.1562 0.0263)`, radius `0.087266` radians; no validation
  error was emitted. Capture:
  `baseq2/screenshots/RR_dominant_sun_blocker_20260825.png`; log:
  `baseq2/logs/RR_dominant_sun_metadata_20260825.log`. This establishes the
  Q2RTX-side signal contract only, not availability of AMD's neural RR
  provider. It now also publishes actual camera-position delta (rather than
  asking a consumer to infer it from matrices) and maps the complete input set
  into the installable `ffx-vulkan::rayregeneration-contract`. A fresh RX
  6800M `vk_validation=1` FSR3 run reached v10/history-valid frame 113; the
  reusable preflight returned `valid (issues=0x0; dominant light included)`
  with no validation error. Evidence:
  `baseq2/logs/RR_reusable_preflight_20260825.log`.
  Q2RTX's shadow ray points surface-to-sun; the reusable provider boundary
  explicitly negates that to the AMD RR light-to-target direction convention.
  A fresh corrected-direction validation run logged
  `light-to-surface=(-0.8754 0.2346 -0.4226)` and retained the clean preflight:
  `baseq2/logs/RR_provider_direction_20260825.log`.

- `extern/ffx-vulkan` now exports the versioned, installable
  `ffx-vulkan::rayregeneration-contract` static target. Its public C ABI
  validates provider-neutral RR-style sampled image metadata: signed linear
  depth, motion, compact material/albedo inputs, camera/motion/jitter/depth
  metadata, radiance-partition alpha rules, optional dominant-light data, and
  the two optional scalar AO/specular-occlusion signals. Contract v4 validates
  those additions while retaining the real-provider rule that one primary
  radiance or dominant-light signal is required; it rejects unknown flag bits.
  Q2RTX maps its four radiance partitions plus dominant light and does not yet
  export separate AO/specular-occlusion images. The contract is
  intentionally only a pre-provider validator: it cannot inspect GPU pixels,
  record a dispatch, or make AMD's signed neural RR provider available. Its
  unit test and both standalone/full-stack installed-consumer contracts pass;
  see this milestone's commit for exact build evidence.

- RR motion correctness was tightened against AMD's current integration
  contract. The reusable v3 ABI has a three-component scale (XY UV, Z
  signed-linear depth delta) and previous-minus-current camera motion. Q2RTX's
  usual `FLAT_MOTION` cannot provide that Z value because reflection/refraction
  handling changes it to radial metadata. Temporal contract v11 therefore
  exports `TEMPORAL_RR_MOTION`, written directly by primary rays with
  `PreviousUV-CurrentUV` XY and previous-minus-current signed-linear view-Z.
  The RR bridge uses it at scale `(1,1,1)` and negates Q2RTX's published
  current-minus-previous camera delta. A fresh RX 6800M `vk_validation=1`
  FSR3 run reached v11/history-valid frame 149 with `issues=0x0` and no
  validation error. This proves the input bridge/barriers, not a neural RR
  dispatch.

- The reusable RR contract now also has a provider-dispatch hand-off: populate
  `FfxVkRayRegenerationOutputs`, then call `ffxVkRayRegenerationValidateOutputs`
  after input validation. It checks active output formats, full render extent,
  storage state/usage, legal in-place aliases, and checkerboard subset/origin
  rules. This is deliberately validation only; it does not imply a provider
  records neural work. Its standalone test exercises valid output bindings,
  wrong output format, and checkerboard origin rejection.

- Console screenshot readback now deterministically renders a requested
  temporal debug view into its freshly acquired WSI image before copying it.
  The prior safe-acquire-only code copied an arbitrary fresh image, producing
  black debug captures. The capture uses a third per-frame final-blit
  descriptor slot, so it never updates either normal or generated-present
  descriptor set while that set is still pending. The RX 6800M validation run
  above emitted no VUID/error.

- Debug presentation now makes one shared swapchain-policy decision: temporal
  input inspection disables both generated-frame presentation and its FIFO
  swapchain request. Previously swapchain creation looked only at the raw FG
  cvar while `R_BeginFrame_RTX` correctly disabled FG for debug, producing a
  recreate/destroy/context-create loop every debug frame. A fresh RX 6800M
  run with both FG and debug requested created contexts only for initial
  640x480 startup and the intended 960x540 resize, then retained them through
  frame 35 with history valid and no VUID/error:
  `baseq2/logs/FSR_debug_context_20260825.log`.

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
  queue-ordered imported-view sets safely until GPU completion. Q2RTX selects
  this provider through `flt_frame_generation_backend 1` (the default for
  fresh configs; `0` retains the 1.1.4 implementation), maps the RGBA16F
  HUDless color, R32F
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
  generated-then-real single-queue presentation path. Presenter activity is
  published only after an interpolated-generated→real pair; diagnostics instead
  report logical rendered cadence and nominal 2x rate while active, avoiding
  false claims from CPU WSI submission timing. Frame-generation
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
  experimental presenter pause after four below-threshold logical frames.
  Eight frames above an adaptive re-enable floor resume it; that floor starts
  at threshold +2 FPS and incorporates the last measured FI cost, preventing
  a borderline scene from repeatedly enabling FI only to fall under the same
  threshold. `0` explicitly disables the gate for R&D and 60 reflects AMD's
  recommendation. Both the open-gate and deliberately impossible 240-FPS
  fallback branches were live validated at 1280x720 without a VUID; the local
  archived test setting was restored to the 30-FPS default.
- DXIL extraction/capture tooling is complete and tested; no extracted payload
  is checked into the repository.
- The source-only official SDK 2.3 DX12 probe now cross-compiles under MinGW
  and runs through Wine/vkd3d-proton. It now independently enumerates public
  frame-generation and, from the full SDK's `ffx_denoiser.h`, denoiser/Ray
  Regeneration providers as well as upscaler providers. The deliberately
  reduced vendored source closure omits that header and reports the omission
  rather than guessing a private ABI. A fresh full-SDK 2.3.0 probe on the
  selected RDNA2 adapter `1002:73df` listed analytical upscaler providers
  3.1.5 and 2.3.4, one 3.1.6 frame-generation provider, and no
  denoiser/Ray-Regeneration provider. Creating a 4.1.1 API context selected
  3.1.5 and one 640x360 -> 1280x720 auto-exposure-enabled metadata dispatch
  succeeded without an FFX warning. It generated 11 paired DXIL/SPIR-V capture
  artifacts and recorded 25 provider-owned D3D12 resource allocations outside
  the repo. This is direct evidence that the current official path falls back
  to analytical FSR3 on RDNA2; it is not neural FSR4. See
  tools/ffx_dxil/reference_harness/.
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
  override (revision `2026-09-fsr3-fsr4-v07-rr-fg-gate`, SHA-256
  `8c50a86b1c277aa3ffbb05f834d5aab6d4aa7d7ef22f2ec485c616580af61279`).
  Its FG safety-floor help now accurately describes the conservative
  no-strobe recovery behavior.
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
  MinGW/Wine probe that creates a DX12 device, enumerates independent upscaler,
  frame-generation, and full-SDK denoiser/Ray-Regeneration provider versions,
  creates a 4.1.1 API context, and optionally records one controlled dispatch.
  It reports absent full-SDK denoiser headers in the reduced vendored closure.
  It registers public resource allocation callbacks. The selected RDNA2 result
  is analytical provider 3.1.5 fallback, one analytical 3.1.6 FG provider,
  no denoiser/RR provider, 11 paired capture shaders, and 25 logged allocations;
  no provider/model/capture binary is tracked.

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
  model initializer/weight pairs. It independently requires each model's
  JSON manifest and at least one model-prefixed SPIR-V module both before
  building and after staging, in addition to CMake's complete-directory
  install contract. A package-style staging test must continue to verify every
  model asset and the retained MIT notice. A fresh detached
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
  alongside `fsr4_schedule` (2/2) and the separate reusable suite (36/36).
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
