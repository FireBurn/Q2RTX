# ffx_fsr4_vk.c Descriptor Layout Fix

## Problem

DXC without `-fvk-*-shift` flags maps DX12 register spaces (t/u/b/s) to the
same Vulkan binding numbers.  `b0` (cbuffer), `t0` (texture), `u0` (UAV), and
`s0` (sampler) all get binding 0 — invalid in Vulkan where each binding has one
descriptor type.

## Solution: Register-to-Binding Shifts

The updated compile script adds these DXC flags:

**Model passes** (pass 1–12):
```
-fvk-t-shift 0 0   → t0=bind0 (input), t1=bind1 (weights)
-fvk-u-shift 2 0   → u0=bind2 (output), u1=bind3 (scratch)
```

**Full passes** (pre, post, rcas, spd):
```
-fvk-t-shift 0 0   → t0..t20 = bindings 0..20
-fvk-s-shift 0 0   → samplers merged with textures (COMBINED_IMAGE_SAMPLER)
-fvk-u-shift 21 0  → u0..u12 = bindings 21..33
-fvk-b-shift 43 0  → b0=bind43, b21=bind64
```

## New SLOT Definitions

Replace the existing SLOT defines:

```c
// ── Binding offsets for FULL layout (matching -fvk-*-shift flags) ────────
// SRV textures (t-registers): bindings 0..20
#define SLOT_TEX_BASE        0    // t0..t20 → COMBINED_IMAGE_SAMPLER

// UAV registers (u-registers shifted by 21): bindings 21..33
#define SLOT_UAV_BASE       21    // u0→21, u1→22, ..., u12→33
// Usage by register:
//   u0  (21): rw_spd_global_atomic (STORAGE_IMAGE)
//   u1  (22): rw_history_color / rw_autoexp_mip_5 (STORAGE_IMAGE)
//   u2  (23): rw_mlsr_output_color / rw_auto_exposure_texture (STORAGE_IMAGE)
//   u3  (24): rw_reprojected_color (STORAGE_IMAGE)
//   u6  (27): rw_recurrent_0 (STORAGE_IMAGE)
//   u11 (32): ScratchBuffer (STORAGE_BUFFER in pre/post)
//             rw_rcas_output (STORAGE_IMAGE in rcas) ← CONFLICT
//   u12 (33): (unused or reserved)

// CBV registers (b-registers shifted by 43): bindings 43+
#define SLOT_CBV_BASE       43    // b0→43 (main constants), b21→64 (pass weights)
#define SLOT_CBUF_MAIN      43    // b0: MLSR_Optimized_Constants / cbRCAS / etc.
#define SLOT_CBUF_WEIGHTS   64    // b21: cbPass_Weights (pre-pass only)

// Total bindings needed: max(64) + 1 = 65
#define FULL_LAYOUT_BINDING_COUNT 65

// Model layout (unchanged): 4 × STORAGE_BUFFER at bindings 0-3
```

## buildFullLayout() — Two Variants

Because `u11` (binding 32) is `STORAGE_BUFFER` in pre/post but `STORAGE_IMAGE`
in rcas, we need two layout variants.  The simplest approach: a helper that
takes the type for binding 32 as a parameter.

```c
static VkResult buildFullLayoutVariant(FfxFsr4VkContext* c,
                                        VkDescriptorType u11_type,
                                        VkDescriptorSetLayout* outLayout)
{
    const uint32_t N = FULL_LAYOUT_BINDING_COUNT;
    VkDescriptorSetLayoutBinding* b =
        (VkDescriptorSetLayoutBinding*)calloc(N, sizeof(*b));

    // Initialize all bindings as COMBINED_IMAGE_SAMPLER (safe default for
    // unused bindings — the shader won't access them).
    for (uint32_t i = 0; i < N; i++) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].pImmutableSamplers = &c->linearSampler;
    }

    // UAV bindings 21..33 (u0..u12): default STORAGE_IMAGE
    for (uint32_t u = 0; u <= 12; u++) {
        uint32_t s = SLOT_UAV_BASE + u;
        b[s].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[s].pImmutableSamplers = NULL;
    }

    // u11 (binding 32): per-pass type (STORAGE_BUFFER or STORAGE_IMAGE)
    b[SLOT_UAV_BASE + 11].descriptorType = u11_type;

    // CBV bindings: UNIFORM_BUFFER
    b[SLOT_CBUF_MAIN].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[SLOT_CBUF_MAIN].pImmutableSamplers = NULL;
    b[SLOT_CBUF_WEIGHTS].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[SLOT_CBUF_WEIGHTS].pImmutableSamplers = NULL;

    VkDescriptorSetLayoutCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = N;
    ci.pBindings = b;
    VkResult r = vkCreateDescriptorSetLayout(c->dev, &ci, c->alloc, outLayout);
    free(b);
    return r;
}

static VkResult buildFullLayout(FfxFsr4VkContext* c)
{
    // Layout A: pre, post, spd — u11 is ScratchBuffer (STORAGE_BUFFER)
    VkResult r = buildFullLayoutVariant(c, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         &c->fullLayout);
    if (r != VK_SUCCESS) return r;

    // Layout B: rcas — u11 is rw_rcas_output (STORAGE_IMAGE)
    r = buildFullLayoutVariant(c, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                &c->fullLayoutRcas);
    return r;
}
```

Add `VkDescriptorSetLayout fullLayoutRcas;` to the `FfxFsr4VkContext` struct.

## mkPipeline() — Choose Layout Per Pass

```c
static VkResult mkPipeline(FfxFsr4VkContext* c, uint32_t passIdx, VkPipe* out)
{
    // ...existing code...

    if (kind == PIPE_MODEL) {
        out->dsLayout = c->modelLayout;
    } else if (passIdx == FFX_FSR4_VK_PASS_RCAS) {
        out->dsLayout = c->fullLayoutRcas;
    } else {
        out->dsLayout = c->fullLayout;
    }

    // ...rest of pipeline creation...
}
```

## Descriptor Writes — Use Shifted Bindings

The `writeFullDescs()` function (or equivalent) that writes descriptor sets for
full passes must use the shifted binding numbers:

```c
// Texture SRVs: binding = t-register number (shift=0)
// e.g., r_history_color is t0 → binding 0
// e.g., r_input_color is t3 → binding 3
ws[wc++] = (VkWriteDescriptorSet){
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = ds,
    .dstBinding = 3,  // t3 → binding 3
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    .pImageInfo = &imgInfo_input_color,
};

// UAV images: binding = u-register + 21
// e.g., rw_reprojected_color is u3 → binding 24
ws[wc++] = (VkWriteDescriptorSet){
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = ds,
    .dstBinding = SLOT_UAV_BASE + 3,  // u3 → binding 24
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    .pImageInfo = &imgInfo_rw_reprojected_color,
};

// ScratchBuffer (u11): binding 32
ws[wc++] = (VkWriteDescriptorSet){
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = ds,
    .dstBinding = SLOT_UAV_BASE + 11,  // u11 → binding 32
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .pBufferInfo = &bufInfo_scratch,
};

// Constants (b0): binding 43
ws[wc++] = (VkWriteDescriptorSet){
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = ds,
    .dstBinding = SLOT_CBUF_MAIN,  // b0 → binding 43
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    .pBufferInfo = &bufInfo_constants,
};

// Pass Weights (b21): binding 64 — only for pre-pass
ws[wc++] = (VkWriteDescriptorSet){
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = ds,
    .dstBinding = SLOT_CBUF_WEIGHTS,  // b21 → binding 64
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    .pBufferInfo = &bufInfo_pass_weights,
};
```

## Binding-to-Register Quick Reference

| Register | Shift | Binding | Descriptor Type          | Resource                    |
|----------|-------|---------|--------------------------|-----------------------------|
| t0       | +0    | 0       | COMBINED_IMAGE_SAMPLER   | r_history_color / r_input_color (spd) |
| t1       | +0    | 1       | COMBINED_IMAGE_SAMPLER   | r_velocity                  |
| t3       | +0    | 3       | COMBINED_IMAGE_SAMPLER   | r_input_color               |
| t4       | +0    | 4       | COMBINED_IMAGE_SAMPLER   | r_recurrent_0               |
| t6       | +0    | 6       | COMBINED_IMAGE_SAMPLER   | r_input_exposure            |
| t9       | +0    | 9       | COMBINED_IMAGE_SAMPLER   | r_reprojected_color         |
| t18      | +0    | 18      | COMBINED_IMAGE_SAMPLER   | r_rcas_input                |
| u0       | +21   | 21      | STORAGE_IMAGE            | rw_spd_global_atomic        |
| u1       | +21   | 22      | STORAGE_IMAGE            | rw_history_color / rw_autoexp_mip_5 |
| u2       | +21   | 23      | STORAGE_IMAGE            | rw_mlsr_output_color / rw_auto_exposure |
| u3       | +21   | 24      | STORAGE_IMAGE            | rw_reprojected_color        |
| u6       | +21   | 27      | STORAGE_IMAGE            | rw_recurrent_0              |
| u11      | +21   | 32      | STORAGE_BUFFER *(pre/post)* | ScratchBuffer            |
| u11      | +21   | 32      | STORAGE_IMAGE *(rcas)*   | rw_rcas_output              |
| b0       | +43   | 43      | UNIFORM_BUFFER           | MLSR_Optimized_Constants / cbRCAS / etc. |
| b21      | +43   | 64      | UNIFORM_BUFFER           | cbPass_Weights              |

## Cleanup

Destroy both layouts:
```c
vkDestroyDescriptorSetLayout(c->dev, c->fullLayout, c->alloc);
vkDestroyDescriptorSetLayout(c->dev, c->fullLayoutRcas, c->alloc);
```

## Descriptor Pool

The pool needs enough descriptors for 65 bindings per set.  Update the pool
sizes to account for the higher binding count (mostly COMBINED_IMAGE_SAMPLER
for unused slots, plus STORAGE_IMAGE, STORAGE_BUFFER, and UNIFORM_BUFFER for
active ones).
