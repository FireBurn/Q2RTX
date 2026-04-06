#!/usr/bin/env bash
# compile_shaders_fsr4.sh — Compiles FSR4 INT8 shaders (HLSL→SPIR-V)
#
# Requires a DXC build with the FlattenResourceStructVisitor patch applied.
# Without the patch, the generated SPIR-V will crash RADV/Mesa.
#
# Usage: ./compile_shaders_fsr4.sh <sdk_root> <preset> <output_dir>
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

COMMON_FLAGS=(
    -spirv -T cs_6_4 -enable-16bit-types -HV 2021 -O3 -no-warnings
    -fspv-target-env=vulkan1.3 -Vd
    "-D_Static_assert(cond,msg)="
    -DFFX_HLSL=1 -DFFX_GPU=1 -DFFX_HLSL_SM=64 -DFSR4_ENABLE_DOT4=1 -DWMMA_ENABLED=0
    -I "$FSR4_ROOT/dx12" -I "$SDK/Kits/FidelityFX/api/internal/dx12"
    -I "$FSR4_ROOT/include/gpu/fsr4" -I "$FSR4_ROOT/include/gpu"
    -I "$FSR4_ROOT/internal/shaders" -I "$SHADER_DIR"
    -I "$SDK/Kits/FidelityFX/api/internal/gpu"
)
DATA_FLAGS=(-DDATA_TYPE=float -DDATA_TYPE2=float2 -DDATA_TYPE3=float3 "-DDATA_TYPE_VECTOR=float4")

compile_passes() {
    local RES="$1" DEF="$2"
    echo "Compiling passes for ${RES}p..."
    for P in $(seq 1 12); do
        local SPV="$OUT/${MODEL}_${RES}_pass${P}.spv"
        echo "  pass ${P} -> $(basename $SPV)"
        "$DXC_BIN" "${COMMON_FLAGS[@]}" "-D${DEF}" "-DMLSR_PASS_${P}=1" \
            -E "fsr4_model_v07_i8_pass${P}" "$SHADER_DIR/passes_${RES}.hlsl" -Fo "$SPV"
    done
}
compile_passes 1080 "FFX_MLSR_RESOLUTION=0"
compile_passes 2160 "FFX_MLSR_RESOLUTION=1"
compile_passes 4320 "FFX_MLSR_RESOLUTION=2"

for PASS in pre post; do
    echo "Compiling ${PASS}-pass..."
    "$DXC_BIN" "${COMMON_FLAGS[@]}" "-DFFX_MLSR_RESOLUTION=0" "${DATA_FLAGS[@]}" \
        -E "main" "$SHADER_DIR/${PASS}.hlsl" -Fo "$OUT/${MODEL}_${PASS}.spv"
done

echo "Compiling RCAS..."
"$DXC_BIN" "${COMMON_FLAGS[@]}" "${DATA_FLAGS[@]}" -E "main" \
    "$FSR4_ROOT/internal/shaders/rcas.hlsl" -Fo "$OUT/rcas.spv"

echo "Compiling SPD..."
"$DXC_BIN" "${COMMON_FLAGS[@]}" "${DATA_FLAGS[@]}" -E "main" \
    "$FSR4_ROOT/internal/shaders/spd_auto_exposure.hlsl" -Fo "$OUT/spd_auto_exposure.spv"

echo ""; echo "Creating symlinks..."
cd "$OUT"; P="fsr4_model_v07_i8_${PRESET}"
ln -sf "${P}_pre.spv" fsr4_pre.spv; ln -sf "${P}_post.spv" fsr4_post.spv
ln -sf rcas.spv fsr4_rcas.spv; ln -sf spd_auto_exposure.spv fsr4_spd.spv
for R in 1080 2160 4320; do for N in $(seq 1 12); do
    ln -sf "${P}_${R}_pass${N}.spv" "fsr4_${R}_pass${N}.spv"
done; done
echo "Done. SPIR-V files in: $OUT"
