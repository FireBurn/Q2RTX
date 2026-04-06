#!/usr/bin/env bash
# compile_shaders_fsr4.sh — Compiles FSR4 INT8 shaders (HLSL→SPIR-V)
#
# Requires DXC with the canUseOpCopyLogical patch for RADV compatibility.
#
# Usage: ./compile_shaders_fsr4.sh <sdk_root> <preset> <output_dir>
#
# Environment variables:
#   DXC_BIN    — path to dxc binary (default: dxc)
#   FSR4_DEBUG — set to 1 to enable debug info and disable optimizations
#
# Register-to-binding mapping
# ===========================
# DX12 HLSL uses separate register spaces (t=SRV, u=UAV, b=CBV, s=sampler)
# that can share the same number.  Vulkan requires unique bindings, so we
# use DXC's -fvk-*-shift flags to separate them:
#
#   MODEL passes (pass 1-12): 4 × ByteAddressBuffer / RWByteAddressBuffer
#     -fvk-t-shift 0 0  → t0=bind0 (input), t1=bind1 (weights)
#     -fvk-u-shift 2 0  → u0=bind2 (output), u1=bind3 (scratch)
#     Layout: 4 × STORAGE_BUFFER at bindings 0-3
#
#   FULL passes (pre, post, rcas, spd): mixed textures/images/buffers/cbuffers
#     -fvk-t-shift 0 0  → t0..t20 = bindings 0..20  (COMBINED_IMAGE_SAMPLER)
#     -fvk-s-shift 0 0  → samplers merged with textures
#     -fvk-u-shift 21 0 → u0..u12 = bindings 21..33 (STORAGE_IMAGE / STORAGE_BUFFER)
#     -fvk-b-shift 43 0 → b0=bind43, b21=bind64     (UNIFORM_BUFFER)
#
#   Note: u11 (binding 32) is STORAGE_BUFFER (ScratchBuffer) in pre/post
#         but STORAGE_IMAGE (rw_rcas_output) in rcas — needs separate layouts.

set -euo pipefail

SDK="${1%/}"
if [ ! -d "$SDK/Kits" ] && [ ! -d "$SDK/internal" ]; then
    _try="$SDK"
    for _ in 1 2 3 4 5; do
        _try="$(dirname "$_try")"
        if [ -d "$_try/Kits" ] || [ -d "$_try/internal" ]; then
            echo "Note: SDK root auto-detected as $_try"; SDK="${_try%/}"; break
        fi
    done
fi
[ -d "$SDK/Kits" ] || [ -d "$SDK/internal" ] || { echo "Error: Cannot find SDK root."; exit 1; }
PRESET="${2:-performance}"; OUT="${3:-./spv}"; mkdir -p "$OUT"

DXC_BIN="${DXC_BIN:-dxc}"
command -v "$DXC_BIN" &>/dev/null || { echo "Error: DXC not found."; exit 1; }
echo "Using DXC: $("$DXC_BIN" --version 2>&1 | head -1)"

FSR4_ROOT="$SDK/Kits/FidelityFX/upscalers/fsr4"
MODEL="fsr4_model_v07_i8_${PRESET}"
SHADER_DIR="$FSR4_ROOT/internal/shaders/${MODEL}"

# Choose optimization level based on debug mode
if [ "${FSR4_DEBUG:-0}" = "1" ]; then
    echo "*** DEBUG MODE: -O0, -Zi, validation enabled ***"
    OPT_FLAGS=(-O0 -Zi)
else
    OPT_FLAGS=(-O3)
fi

# Common flags shared by ALL shaders
BASE_FLAGS=(
    -spirv -T cs_6_4 -enable-16bit-types -HV 2021
    "${OPT_FLAGS[@]}"
    -fspv-target-env=vulkan1.3
    "-D_Static_assert(cond,msg)="
    -DFFX_HLSL=1 -DFFX_GPU=1 -DFFX_HLSL_SM=64 -DFSR4_ENABLE_DOT4=1 -DWMMA_ENABLED=0
    -I "$FSR4_ROOT/dx12" -I "$SDK/Kits/FidelityFX/api/internal/dx12"
    -I "$FSR4_ROOT/include/gpu/fsr4" -I "$FSR4_ROOT/include/gpu"
    -I "$FSR4_ROOT/internal/shaders" -I "$SHADER_DIR"
    -I "$SDK/Kits/FidelityFX/api/internal/gpu"
)

# Register-to-binding shifts for MODEL passes (pass 1-12)
# t0=0(input) t1=1(weights) u0=2(output) u1=3(scratch) → 4 × STORAGE_BUFFER
MODEL_SHIFT_FLAGS=(
    -fvk-t-shift 0 0
    -fvk-u-shift 2 0
)

# Register-to-binding shifts for FULL passes (pre, post, rcas, spd)
# t0..t20→0..20  u0..u12→21..33  b0→43 b21→64  samplers merged with t
FULL_SHIFT_FLAGS=(
    -fvk-t-shift 0 0
    -fvk-s-shift 0 0
    -fvk-u-shift 21 0
    -fvk-b-shift 43 0
)

DATA_FLAGS=(-DDATA_TYPE=float -DDATA_TYPE2=float2 -DDATA_TYPE3=float3 "-DDATA_TYPE_VECTOR=float4")

compile_passes() {
    local RES="$1" DEF="$2"
    echo "Compiling passes for ${RES}p..."
    for P in $(seq 1 12); do
        local SPV="$OUT/${MODEL}_${RES}_pass${P}.spv"
        echo "  pass ${P} -> $(basename $SPV)"
        "$DXC_BIN" "${BASE_FLAGS[@]}" "${MODEL_SHIFT_FLAGS[@]}" \
            "-D${DEF}" "-DMLSR_PASS_${P}=1" \
            -E "fsr4_model_v07_i8_pass${P}" "$SHADER_DIR/passes_${RES}.hlsl" -Fo "$SPV"
    done
}
compile_passes 1080 "FFX_MLSR_RESOLUTION=0"
compile_passes 2160 "FFX_MLSR_RESOLUTION=1"
compile_passes 4320 "FFX_MLSR_RESOLUTION=2"

for PASS in pre post; do
    echo "Compiling ${PASS}-pass..."
    "$DXC_BIN" "${BASE_FLAGS[@]}" "${FULL_SHIFT_FLAGS[@]}" \
        "-DFFX_MLSR_RESOLUTION=0" "${DATA_FLAGS[@]}" \
        -E "main" "$SHADER_DIR/${PASS}.hlsl" -Fo "$OUT/${MODEL}_${PASS}.spv"
done

echo "Compiling RCAS..."
"$DXC_BIN" "${BASE_FLAGS[@]}" "${FULL_SHIFT_FLAGS[@]}" \
    "${DATA_FLAGS[@]}" -E "main" \
    "$FSR4_ROOT/internal/shaders/rcas.hlsl" -Fo "$OUT/rcas.spv"

echo "Compiling SPD..."
"$DXC_BIN" "${BASE_FLAGS[@]}" "${FULL_SHIFT_FLAGS[@]}" \
    "${DATA_FLAGS[@]}" -E "main" \
    "$FSR4_ROOT/internal/shaders/spd_auto_exposure.hlsl" -Fo "$OUT/spd_auto_exposure.spv"

# Validate with spirv-val if available
if command -v spirv-val &>/dev/null; then
    echo ""
    echo "Validating SPIR-V with spirv-val..."
    ERRORS=0
    for spv in "$OUT"/${MODEL}_*.spv "$OUT"/rcas.spv "$OUT"/spd_auto_exposure.spv; do
        [ -f "$spv" ] || continue
        if ! spirv-val --target-env vulkan1.3 "$spv" 2>&1; then
            echo "  FAIL: $(basename $spv)"
            ERRORS=$((ERRORS + 1))
        fi
    done
    if [ $ERRORS -gt 0 ]; then
        echo "*** $ERRORS shader(s) failed validation ***"
    else
        echo "  All shaders passed validation."
    fi
else
    echo ""
    echo "Note: install spirv-tools for SPIR-V validation (spirv-val)"
fi

echo ""; echo "Creating symlinks..."
cd "$OUT"; P="fsr4_model_v07_i8_${PRESET}"
ln -sf "${P}_pre.spv" fsr4_pre.spv; ln -sf "${P}_post.spv" fsr4_post.spv
ln -sf rcas.spv fsr4_rcas.spv; ln -sf spd_auto_exposure.spv fsr4_spd.spv
for R in 1080 2160 4320; do for N in $(seq 1 12); do
    ln -sf "${P}_${R}_pass${N}.spv" "fsr4_${R}_pass${N}.spv"
done; done
echo "Done. SPIR-V files in: $OUT"
echo ""
echo "Binding map (verify with: extract_bindings.sh $OUT):"
echo "  MODEL passes: t0=0 t1=1 u0=2 u1=3  (4×STORAGE_BUFFER)"
echo "  FULL passes:  t0..t20=0..20  u0..u12=21..33  b0=43 b21=64"
