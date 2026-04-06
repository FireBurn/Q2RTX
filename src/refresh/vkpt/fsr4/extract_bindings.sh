#!/usr/bin/env bash
# extract_bindings.sh — Extract Vulkan descriptor bindings from compiled FSR4 SPIR-V
#
# Usage: ./extract_bindings.sh <shader_dir>
#
# Requires spirv-cross (from spirv-tools or vulkan-sdk)
# Shows the actual binding layout that ffx_fsr4_vk.c must match.

set -euo pipefail

DIR="${1:-.}"

if ! command -v spirv-cross &>/dev/null; then
    echo "Error: spirv-cross not found. Install spirv-tools or vulkan-sdk."
    exit 1
fi

echo "=== Extracting descriptor bindings from FSR4 SPIR-V ==="
echo ""

for spv in "$DIR"/fsr4_pre.spv "$DIR"/fsr4_post.spv "$DIR"/fsr4_rcas.spv "$DIR"/fsr4_spd.spv "$DIR"/fsr4_1080_pass1.spv; do
    [ -f "$spv" ] || continue
    echo "--- $(basename "$spv") ---"
    spirv-cross "$spv" --reflect 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Collect all bindings
bindings = []
for rtype in ['ubos', 'ssbos', 'images', 'separate_images', 'separate_samplers', 'textures']:
    for r in data.get(rtype, []):
        bindings.append({
            'binding': r.get('binding', '?'),
            'set': r.get('set', 0),
            'name': r.get('name', '?'),
            'type': rtype,
        })
bindings.sort(key=lambda b: (b['set'], b['binding']))
for b in bindings:
    vk_type = {
        'ubos': 'UNIFORM_BUFFER',
        'ssbos': 'STORAGE_BUFFER',
        'images': 'STORAGE_IMAGE',
        'separate_images': 'SAMPLED_IMAGE',
        'separate_samplers': 'SAMPLER',
        'textures': 'COMBINED_IMAGE_SAMPLER',
    }.get(b['type'], b['type'])
    print(f\"  set={b['set']} binding={b['binding']:3d}  {vk_type:30s}  {b['name']}\")
" 2>/dev/null || echo "  (spirv-cross --reflect failed or python3 not available)"
    echo ""
done

echo "=== Model pass layout (4x STORAGE_BUFFER at bindings 0-3) ==="
echo "  set=0 binding=  0  STORAGE_BUFFER                  input (t0 ByteAddressBuffer)"
echo "  set=0 binding=  1  STORAGE_BUFFER                  output (u0 RWByteAddressBuffer)"
echo "  set=0 binding=  2  STORAGE_BUFFER                  weights (t1 ByteAddressBuffer)"
echo "  set=0 binding=  3  STORAGE_BUFFER                  scratch (u1 RWByteAddressBuffer)"
echo ""
echo "Use the full-pass bindings above to fix buildFullLayout() in ffx_fsr4_vk.c."
echo "Each binding's VkDescriptorType must match the shader's expected type."
