/*
Copyright (C) 2026 Q2RTX contributors.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
*/

#version 450
#extension GL_GOOGLE_include_directive : enable

#define GLOBAL_UBO_DESC_SET_IDX 0
#include "global_ubo.h"

layout(push_constant, std430) uniform PushConstants {
    vec2 input_dimensions;
    uint temporal_debug_view;
} push;

layout(set = 1, binding = 0) uniform sampler2D temporal_input_image;

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec4 outColor;

vec3 tonemap_debug(vec3 color)
{
    color = max(color, vec3(0.0));
    return color / (vec3(1.0) + color);
}

void main()
{
    vec2 uv = tex_coord * push.input_dimensions /
        vec2(global_ubo.taa_image_width, global_ubo.taa_image_height);
    vec4 sample_value = textureLod(temporal_input_image, uv, 0);
    vec3 color;

    switch (push.temporal_debug_view) {
    case 1u: /* Pre-tone-map FLAT_COLOR uses Q2RTX's HDR storage scale. */
        color = tonemap_debug(sample_value.rgb / 128.0);
        break;
    case 2u: { /* Normalized current-to-previous motion in RG. */
        vec2 pixels = sample_value.xy * push.input_dimensions;
        color = vec3(clamp(pixels / 32.0 + 0.5, 0.0, 1.0),
            clamp(length(pixels) / 32.0, 0.0, 1.0));
        break;
    }
    case 3u: /* Conventional finite device depth, [0, 1]. */
    case 6u: /* Composition mask. */
        color = vec3(clamp(sample_value.r, 0.0, 1.0));
        break;
    case 4u: /* Positive view-Z; sky is PRIMARY_RAY_T_MAX (10000). */
        color = vec3(clamp(log2(1.0 + max(sample_value.r, 0.0)) /
            log2(10001.0), 0.0, 1.0));
        break;
    case 5u: /* Reactive mask. */
        color = vec3(clamp(sample_value.r, 0.0, 1.0), 0.0, 0.0);
        break;
    case 7u: /* Linear geometric normal in [-1, 1]. */
        color = clamp(sample_value.rgb * 0.5 + 0.5, 0.0, 1.0);
        break;
    case 8u: /* Linear albedo. */
        color = tonemap_debug(sample_value.rgb);
        break;
    case 9u: /* Perceptual roughness. */
        color = vec3(clamp(sample_value.r, 0.0, 1.0));
        break;
    case 10u: /* Current reconstructed output, before Q2RTX post effects. */
    case 11u: /* FSR4 history: previous display-resolution reconstructed HDR. */
    case 12u: /* FSR4 pre/post bridge: reprojected display-resolution HDR. */
        color = tonemap_debug(sample_value.rgb / 128.0);
        break;
    case 13u: /* RR: oct normal in RG, linear roughness B, material type A. */
        color = vec3(sample_value.rg, sample_value.b);
        break;
    case 14u: /* RR: sqrt-encoded diffuse albedo. */
    case 15u: /* RR: sqrt-encoded specular albedo. */
        color = sample_value.rgb * sample_value.rgb;
        break;
    case 16u: /* RR raw direct diffuse. */
    case 17u: /* RR raw indirect diffuse SH coefficient. */
    case 18u: /* RR raw direct specular. */
    case 19u: /* RR raw indirect specular. */
        color = tonemap_debug(sample_value.rgb);
        break;
    default:
        color = vec3(1.0, 0.0, 1.0);
        break;
    }
    outColor = vec4(color, 1.0);
}
