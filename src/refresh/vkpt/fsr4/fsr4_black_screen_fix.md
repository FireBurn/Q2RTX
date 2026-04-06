# FSR4 Black Screen Root Cause & Fix

## Status After Layout Fix

The descriptor layout fix is WORKING — zero type mismatches at pipeline creation.
The shaders compile correctly with -fvk-*-shift flags. The game runs at 62 FPS.

## Root Cause: Incomplete Provider

The black screen is caused by the FSR4 provider (`ffx_provider_fsr4upscaler.c`)
not creating or binding the internal resources that the shaders need.

### What the provider currently does

**Context creation:** Only creates pipelines. No internal resources.

**Dispatch (pre/post passes):**
- Registers 4 external resources: color, depth, motionVectors, output
- Builds compute jobs with only 3 SRV textures + 0-1 UAV textures + 1 cbuffer
- Uses sequential array indices (srvTextures[0], [1], [2]) without bindingIndex

**Dispatch (model passes 1-12):**
- Provides 1 srvBuffer + 1 uavBuffer with sentinel indices (0, 1)
- No weights buffer, no scratch buffer

### What the shaders actually need

**Pre-pass (fsr4_pre.spv):**

| Binding | Register | Type                   | Resource            | Source         |
|---------|----------|------------------------|---------------------|----------------|
| 0       | t0       | COMBINED_IMAGE_SAMPLER | r_history_color     | **Internal**   |
| 1       | t1       | COMBINED_IMAGE_SAMPLER | r_velocity          | Dispatch (mv)  |
| 3       | t3       | COMBINED_IMAGE_SAMPLER | r_input_color       | Dispatch (color)|
| 4       | t4       | COMBINED_IMAGE_SAMPLER | r_recurrent_0       | **Internal**   |
| 6       | t6       | COMBINED_IMAGE_SAMPLER | r_input_exposure    | **Internal**   |
| 24      | u3       | STORAGE_IMAGE          | rw_reprojected_color| **Internal**   |
| 32      | u11      | STORAGE_BUFFER         | ScratchBuffer       | **Internal**   |
| 34      |          | UNIFORM_BUFFER         | cbPass_Weights      | **Internal**   |
| 43      | b0       | UNIFORM_BUFFER         | OptimizedConstants  | Dispatch (cb)  |

**Post-pass (fsr4_post.spv):**

| Binding | Register | Type                   | Resource              | Source         |
|---------|----------|------------------------|-----------------------|----------------|
| 3       | t3       | COMBINED_IMAGE_SAMPLER | r_input_color         | Dispatch (color)|
| 6       | t6       | COMBINED_IMAGE_SAMPLER | r_input_exposure      | **Internal**   |
| 9       | t9       | COMBINED_IMAGE_SAMPLER | r_reprojected_color   | **Internal**   |
| 22      | u1       | STORAGE_IMAGE          | rw_history_color      | **Internal**   |
| 23      | u2       | STORAGE_IMAGE          | rw_mlsr_output_color  | Dispatch (out) |
| 27      | u6       | STORAGE_IMAGE          | rw_recurrent_0        | **Internal**   |
| 32      | u11      | STORAGE_BUFFER         | ScratchBuffer         | **Internal**   |
| 43      | b0       | UNIFORM_BUFFER         | OptimizedConstants    | Dispatch (cb)  |

**Model passes (fsr4_*_passN.spv):**

| Binding | Register | Type           | Resource           | Source       |
|---------|----------|----------------|--------------------|--------------|
| 0       | t0       | STORAGE_BUFFER | Input              | ScratchBuf A |
| 1       | t1       | STORAGE_BUFFER | Weights/Init       | **Internal** |
| 2       | u0       | STORAGE_BUFFER | Output             | ScratchBuf B |
| 3       | u1       | STORAGE_BUFFER | ScratchBuffer      | ScratchBuf C |

## Fix Plan

### 1. Add internal resource tracking to provider context

```c
typedef struct Fsr4Context {
    /* ... existing fields ... */

    /* Internal resources — created once, persist across frames */
    FfxResourceInternal ri_scratch;        /* Large SSBO for neural network */
    FfxResourceInternal ri_history_color;  /* Previous frame output (display res) */
    FfxResourceInternal ri_recurrent[2];   /* Recurrent feature maps (ping-pong) */
    FfxResourceInternal ri_reprojected;    /* Pre-pass output (render res) */
    FfxResourceInternal ri_exposure;       /* Auto-exposure texture (1x1 or small) */
    FfxResourceInternal ri_weights;        /* NN weights buffer (if not embedded) */
    int                 recurrent_idx;     /* 0 or 1 — which recurrent is "current" */
} Fsr4Context;
```

### 2. Create internal resources in ffxCreateContext

After `fpCreateBackendContext`, call `fpCreateResource` for each internal resource:

```c
/* Scratch buffer — large enough for the neural network's intermediate data.
   Size depends on resolution; 64 MB is typical for 1080p→4K. */
{
    FfxCreateResourceDescription desc = {0};
    desc.type = FFX_RESOURCE_TYPE_BUFFER;
    desc.width = 64 * 1024 * 1024;  /* 64 MB scratch */
    desc.format = FFX_SURFACE_FORMAT_UNKNOWN;
    c->iface.fpCreateResource(&c->iface, &desc, c->effectId, &c->ri_scratch);
}

/* History color — stores previous frame's upscaled output */
{
    FfxCreateResourceDescription desc = {0};
    desc.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    desc.width = c->maxUpscaleSize.width;
    desc.height = c->maxUpscaleSize.height;
    desc.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    c->iface.fpCreateResource(&c->iface, &desc, c->effectId, &c->ri_history_color);
}

/* Recurrent feature maps (2 for ping-pong) */
for (int i = 0; i < 2; i++) {
    FfxCreateResourceDescription desc = {0};
    desc.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    desc.width = c->maxRenderSize.width;
    desc.height = c->maxRenderSize.height;
    desc.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    c->iface.fpCreateResource(&c->iface, &desc, c->effectId, &c->ri_recurrent[i]);
}

/* Reprojected color (render res) */
{
    FfxCreateResourceDescription desc = {0};
    desc.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    desc.width = c->maxRenderSize.width;
    desc.height = c->maxRenderSize.height;
    desc.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    c->iface.fpCreateResource(&c->iface, &desc, c->effectId, &c->ri_reprojected);
}

/* Auto-exposure (1x1) */
{
    FfxCreateResourceDescription desc = {0};
    desc.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    desc.width = 1;
    desc.height = 1;
    desc.format = FFX_SURFACE_FORMAT_R32_FLOAT;
    c->iface.fpCreateResource(&c->iface, &desc, c->effectId, &c->ri_exposure);
}
```

### 3. Fix dispatch to bind ALL resources with correct register mappings

Pre-pass compute job:
```c
if (pass == 0) {
    /* SRV textures — use srvTextureBindings to map to t-registers */
    cj->pipeline.srvTextureCount = 6;
    cj->srvTextures[0].resource = c->ri_history_color;   /* → t0  (bind 0) */
    cj->srvTextures[1].resource = ri_mv;                  /* → t1  (bind 1) */
    cj->srvTextures[2].resource = (FfxResourceInternal){UINT32_MAX}; /* t2 unused */
    cj->srvTextures[3].resource = ri_color;               /* → t3  (bind 3) */
    cj->srvTextures[4].resource = c->ri_recurrent[c->recurrent_idx]; /* → t4 */
    cj->srvTextures[5].resource = (FfxResourceInternal){UINT32_MAX}; /* t5 unused */
    /* Note: r_input_exposure at t6 needs srvTextureCount >= 7 */
    cj->pipeline.srvTextureCount = 7;
    cj->srvTextures[6].resource = c->ri_exposure;         /* → t6  (bind 6) */

    /* UAV textures */
    cj->pipeline.uavTextureCount = 4;  /* u0..u3, only u3 used */
    cj->uavTextures[0].resource = (FfxResourceInternal){UINT32_MAX}; /* u0 unused */
    cj->uavTextures[1].resource = (FfxResourceInternal){UINT32_MAX}; /* u1 unused */
    cj->uavTextures[2].resource = (FfxResourceInternal){UINT32_MAX}; /* u2 unused */
    cj->uavTextures[3].resource = c->ri_reprojected;     /* → u3  (bind 24) */

    /* UAV buffers: ScratchBuffer at u11 */
    cj->pipeline.uavBufferCount = 1;
    cj->uavBuffers[0].resource  = c->ri_scratch;
    /* writeFullDs needs to know this goes to u11 (binding 32) */

    /* Constant buffers */
    cj->pipeline.constCount = 1;
    cj->cbs[0] = cbHandle;
    /* TODO: cbPass_Weights as cbs[1] if applicable */
}
```

### 4. Fix writeFullDs to use register-mapped bindings

The current code writes `SLOT_SRV_TEX_BASE + i` which treats array index as
register number. This works IF srvTextures[] is indexed by t-register number
(i.e., srvTextures[3] = t3). If the provider fills them this way, no change
needed for SRV textures.

For UAV textures, same logic: uavTextures[i] at `SLOT_UAV_BASE + i`.

For UAV buffers, the current code assumes uavBuffers[0..1] = u4,u5 and
uavBuffers[2] = u11. This is wrong — the provider puts scratch at
uavBuffers[0]. The fix is to use uavBufferBindings[i].bindingIndex:

```c
for (uint32_t i=0; i<cj->pipeline.uavBufferCount; i++) {
    uint32_t idx=cj->uavBuffers[i].resource.internalIndex;
    if (idx<c->resCount && c->res[idx].kind==RES_BUFFER) {
        bufs[bc]=(VkDescriptorBufferInfo){c->res[idx].buf,0,VK_WHOLE_SIZE};
        /* Use bindingIndex if available, otherwise fall back to u11 */
        uint32_t u_reg = cj->pipeline.uavBufferBindings[i].bindingIndex;
        uint32_t binding = SLOT_UAV_BASE + u_reg;
        wr[wc]=(VkWriteDescriptorSet){...ds,binding,...};
        bc++; wc++;
    }
}
```

### 5. Fix writeModelDs for 4 bindings

Model passes need all 4 buffers. If the provider fills them correctly:
- srvBuffers[0] (t0→bind 0): input scratch
- srvBuffers[1] (t1→bind 1): weights/initializer (may be unused if embedded)
- uavBuffers[0] (u0→bind 2): output scratch
- uavBuffers[1] (u1→bind 3): aux scratch

The current writeModelDs hardcodes the binding numbers, which is fine as long
as the provider provides srvBufferCount >= 2 and uavBufferCount >= 2.

### 6. After each frame, swap recurrent state

```c
c->recurrent_idx ^= 1;  /* Swap ping-pong buffers */
```

And in the post-pass, write to `c->ri_recurrent[c->recurrent_idx ^ 1]`
(the OTHER recurrent texture).

## Key Unknowns

1. **Scratch buffer size** — needs measurement from the DX12 reference.
   The SDK likely has a function like `ffxGetScratchMemorySize()` or
   documents the required size per resolution.

2. **cbPass_Weights** — the weights cbuffer at binding 34.
   Its content and how to populate it depend on the FSR4 model.
   If the weights are fully embedded in the shader SPIR-V (via
   `ConstantBufferStorage<N>`), this cbuffer may not be needed.
   The "never been updated" error at binding 34 would then be
   benign — the shader might not actually access it at runtime.

3. **Auto-exposure** — whether the 1x1 exposure texture needs the
   SPD pass to run first, or if a constant value (1.0) works.
   Since `FFX_UPSCALE_ENABLE_AUTO_EXPOSURE` is set, the SPD pass
   should compute auto-exposure. But the SPD pass (fsr4_spd.spv)
   also needs to be scheduled before the pre-pass.

## Priority Order

1. Create scratch buffer in provider (fixes model passes)
2. Create history/recurrent/reprojected textures (fixes pre/post passes)
3. Bind all resources to correct registers in dispatch
4. Fix writeFullDs/writeModelDs to use correct binding indices
5. Add SPD pass scheduling for auto-exposure
6. Add RCAS pass scheduling for sharpening
