/*
Copyright (C) 2026 Q2RTX contributors.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef VKPT_TEMPORAL_CONTRACT_H
#define VKPT_TEMPORAL_CONTRACT_H

/*
 * Provider-neutral Vulkan frame contract
 * =======================================
 *
 * This header intentionally has no dependency on Q2RTX renderer types.  A
 * temporal upscaler, denoiser, optical-flow implementation, or frame
 * generator can consume VkptTemporalFrame without knowing which engine made
 * the resources.  Resource ownership remains with the application.
 *
 * The contract describes the producer state of each image.  A consumer is
 * still responsible for inserting an appropriate read-after-write barrier
 * before dispatching and for returning borrowed images in the documented
 * layout.  struct_size and contract_version allow this structure to grow
 * without silently changing the ABI used by an out-of-tree provider.
 */

#include <stdint.h>
#include <vulkan/vulkan.h>

#define VKPT_TEMPORAL_CONTRACT_VERSION 5u

typedef enum VkptTemporalStage_e {
	VKPT_TEMPORAL_STAGE_CLOSED = 0,
	VKPT_TEMPORAL_STAGE_FRAME_BEGIN,
	VKPT_TEMPORAL_STAGE_INPUTS_READY,
	VKPT_TEMPORAL_STAGE_SCENE_PRESENTED,
	VKPT_TEMPORAL_STAGE_UI_COMPOSITION,
	VKPT_TEMPORAL_STAGE_PRESENTED
} VkptTemporalStage;

typedef enum VkptTemporalResetReasonBits_e {
	VKPT_TEMPORAL_RESET_FIRST_FRAME          = 1u << 0,
	VKPT_TEMPORAL_RESET_HISTORY_INVALID      = 1u << 1,
	VKPT_TEMPORAL_RESET_FRAME_DISCONTINUITY  = 1u << 2,
	VKPT_TEMPORAL_RESET_RENDER_SIZE_CHANGED  = 1u << 3,
	VKPT_TEMPORAL_RESET_DISPLAY_SIZE_CHANGED = 1u << 4,
	VKPT_TEMPORAL_RESET_SWAPCHAIN_CHANGED    = 1u << 5,
	VKPT_TEMPORAL_RESET_SCENE_CHANGED        = 1u << 6,
	VKPT_TEMPORAL_RESET_SETTINGS_CHANGED     = 1u << 7,
	VKPT_TEMPORAL_RESET_MENU_TRANSITION      = 1u << 8,
	VKPT_TEMPORAL_RESET_NO_WORLD             = 1u << 9,
	VKPT_TEMPORAL_RESET_PROVIDER_CHANGED     = 1u << 10,
	/* A discontinuous camera transform, rather than ordinary motion-vector
	 * reprojection. Consumers must discard temporal/optical-flow history. */
	VKPT_TEMPORAL_RESET_CAMERA_CUT           = 1u << 11
} VkptTemporalResetReasonBits;

typedef enum VkptTemporalFrameFlagBits_e {
	VKPT_TEMPORAL_FRAME_RENDER_WORLD          = 1u << 0,
	VKPT_TEMPORAL_FRAME_DENOISED              = 1u << 1,
	VKPT_TEMPORAL_FRAME_RECTILINEAR_PROJECTION = 1u << 2,
	VKPT_TEMPORAL_FRAME_LINEAR_HDR_INPUT       = 1u << 3,
	VKPT_TEMPORAL_FRAME_HDR_PRESENTATION       = 1u << 4,
	VKPT_TEMPORAL_FRAME_MENU_MODE              = 1u << 5,
	VKPT_TEMPORAL_FRAME_SINGLE_DEVICE          = 1u << 6
} VkptTemporalFrameFlagBits;

typedef enum VkptTemporalResourceFlagBits_e {
	VKPT_TEMPORAL_RESOURCE_VALID          = 1u << 0,
	VKPT_TEMPORAL_RESOURCE_DENSE          = 1u << 1,
	VKPT_TEMPORAL_RESOURCE_LINEAR         = 1u << 2,
	VKPT_TEMPORAL_RESOURCE_PRE_TONEMAP    = 1u << 3,
	VKPT_TEMPORAL_RESOURCE_PRE_UI         = 1u << 4,
	VKPT_TEMPORAL_RESOURCE_STORAGE_SCALED = 1u << 5,
	VKPT_TEMPORAL_RESOURCE_ALPHA_METADATA = 1u << 6,
	VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE  = 1u << 7
} VkptTemporalResourceFlagBits;

typedef enum VkptTemporalInputFlagBits_e {
	VKPT_TEMPORAL_INPUT_SCENE_COLOR    = 1u << 0,
	VKPT_TEMPORAL_INPUT_MOTION_VECTORS = 1u << 1,
	VKPT_TEMPORAL_INPUT_VIEW_Z         = 1u << 2,
	VKPT_TEMPORAL_INPUT_DEVICE_DEPTH   = 1u << 3,
	VKPT_TEMPORAL_INPUT_NORMALS        = 1u << 4,
	VKPT_TEMPORAL_INPUT_ALBEDO         = 1u << 5,
	VKPT_TEMPORAL_INPUT_ROUGHNESS      = 1u << 6,
	VKPT_TEMPORAL_INPUT_REACTIVE_MASK  = 1u << 7,
	VKPT_TEMPORAL_INPUT_UI             = 1u << 8,
	VKPT_TEMPORAL_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK = 1u << 9,
	/* FSR Ray Regeneration-compatible material features.  These describe the
	 * documented packed resources, not availability of AMD's neural provider. */
	VKPT_TEMPORAL_INPUT_DENOISER_NORMAL_ROUGHNESS_MATERIAL = 1u << 10,
	VKPT_TEMPORAL_INPUT_DENOISER_DIFFUSE_ALBEDO = 1u << 11,
	VKPT_TEMPORAL_INPUT_DENOISER_SPECULAR_ALBEDO = 1u << 12
} VkptTemporalInputFlagBits;

typedef enum VkptTemporalNormalEncoding_e {
	VKPT_TEMPORAL_NORMAL_ENCODING_LINEAR_XYZ = 0,
	VKPT_TEMPORAL_NORMAL_ENCODING_OCTAHEDRAL_UV = 1
} VkptTemporalNormalEncoding;

typedef enum VkptTemporalAlbedoEncoding_e {
	VKPT_TEMPORAL_ALBEDO_ENCODING_LINEAR = 0,
	/* FSR Ray Regeneration's default dispatch encoding.  A future provider can
	 * select its non-gamma flag only after explicitly converting these inputs. */
	VKPT_TEMPORAL_ALBEDO_ENCODING_SQRT = 1
} VkptTemporalAlbedoEncoding;

typedef enum VkptTemporalProjection_e {
	VKPT_TEMPORAL_PROJECTION_RECTILINEAR = 0,
	VKPT_TEMPORAL_PROJECTION_NONLINEAR   = 1
} VkptTemporalProjection;

typedef enum VkptTemporalMotionSpace_e {
	VKPT_TEMPORAL_MOTION_NORMALIZED_UV = 0,
	VKPT_TEMPORAL_MOTION_RENDER_PIXELS = 1,
	VKPT_TEMPORAL_MOTION_DISPLAY_PIXELS = 2
} VkptTemporalMotionSpace;

typedef enum VkptTemporalMotionDirection_e {
	VKPT_TEMPORAL_MOTION_CURRENT_TO_PREVIOUS = 0,
	VKPT_TEMPORAL_MOTION_PREVIOUS_TO_CURRENT = 1
} VkptTemporalMotionDirection;

typedef enum VkptTemporalJitterMode_e {
	VKPT_TEMPORAL_MOTION_JITTER_FREE = 0,
	/* Current sample position includes this frame's jitter.  The previous
	 * endpoint is the unjittered previous projection. */
	VKPT_TEMPORAL_MOTION_CURRENT_JITTER_INCLUDED = 1
} VkptTemporalJitterMode;

typedef enum VkptTemporalDepthConvention_e {
	VKPT_TEMPORAL_DEPTH_UNAVAILABLE = 0,
	VKPT_TEMPORAL_DEPTH_VIEW_Z_POSITIVE_FORWARD,
	VKPT_TEMPORAL_DEPTH_VIEW_Z_NEGATIVE_FORWARD,
	VKPT_TEMPORAL_DEPTH_DEVICE_ZERO_TO_ONE,
	VKPT_TEMPORAL_DEPTH_DEVICE_ONE_TO_ZERO
} VkptTemporalDepthConvention;

typedef enum VkptTemporalUiMode_e {
	VKPT_TEMPORAL_UI_NONE = 0,
	/* UI has no texture of its own, but is drawn after the scene at an explicit
	 * command-buffer boundary.  Frame generation belongs before this boundary. */
	VKPT_TEMPORAL_UI_DIRECT_AFTER_SCENE,
	VKPT_TEMPORAL_UI_SEPARATE_TEXTURE
} VkptTemporalUiMode;

/* Separate UI textures contain premultiplied RGB plus straight alpha. A host
 * composites `ui.rgb + scene.rgb * (1 - ui.a)` after interpolation. */

typedef struct VkptTemporalImage_s {
	uint32_t struct_size;
	uint32_t flags;
	VkImage image;
	VkImageView view;
	VkFormat format;
	VkImageLayout layout;
	VkExtent2D allocation_extent;
	VkExtent2D valid_extent;
	VkPipelineStageFlags producer_stage;
	VkAccessFlags producer_access;
	VkImageUsageFlags usage;
	uint32_t queue_family_index;
	/* Multiply stored RGB/scalar values by value_scale to recover the semantic
	 * value described by the slot.  Usually 1.0. */
	float value_scale;
} VkptTemporalImage;

typedef struct VkptTemporalCamera_s {
	uint32_t struct_size;
	uint32_t projection;
	float view[16];
	float view_inverse[16];
	float projection_matrix[16];
	float projection_inverse[16];
	float previous_view[16];
	float previous_projection[16];
	/* Offset added directly to Q2RTX's primary-ray pixel center.  Consumers
	 * whose API defines projection-matrix jitter (including FSR) must convert
	 * the sign instead of assuming this is already their API convention. */
	float jitter_render_pixels[2];
	float jitter_normalized_uv[2];
	float near_plane;
	float far_plane;
	/* Metres represented by one Q2 world unit.  Q2's player dimensions and
	 * id-family scale convention are consistent with one inch per unit. */
	float view_space_to_meters;
	/* Derived from the magnitude of projection_matrix[5].  Q2RTX flips Y in
	 * its Vulkan projection, so consumers must not treat the signed matrix
	 * coefficient as cot(fov/2). */
	float vertical_fov_radians;
	/* World-space camera basis from the inverse view transform.  Analytical
	 * frame interpolation needs these at high precision for camera-motion
	 * classification; they are intentionally not inferred by a provider. */
	float position[3];
	float up[3];
	float right[3];
	float forward[3];
} VkptTemporalCamera;

typedef struct VkptTemporalMotionDescription_s {
	uint32_t struct_size;
	uint32_t space;
	uint32_t direction;
	uint32_t jitter_mode;
	/* Multiplier from stored XY to render-pixel motion. */
	float to_render_pixels[2];
	/* Z stores previous radial distance minus current radial distance. */
	int32_t radial_depth_delta_channel;
	/* W stores Q2RTX's primary depth footprint/fwidth. */
	int32_t depth_footprint_channel;
} VkptTemporalMotionDescription;

typedef struct VkptTemporalDepthDescription_s {
	uint32_t struct_size;
	uint32_t convention;
	float sky_value;
	uint32_t primary_surface_only;
} VkptTemporalDepthDescription;

/* Semantic encoding of the three packed material resources expected by modern
 * decoupled ray denoisers such as FSR Ray Regeneration. `normal_roughness_
 * material` stores octahedral normal UV in RG, linear roughness in B, and a
 * normalized 0..1 material category in A. `diffuse_albedo` and
 * `specular_albedo` are sqrt encoded. */
typedef struct VkptTemporalDenoiserMaterialDescription_s {
	uint32_t struct_size;
	uint32_t normal_encoding;
	uint32_t albedo_encoding;
	uint32_t material_type_count;
} VkptTemporalDenoiserMaterialDescription;

typedef struct VkptTemporalInputs_s {
	uint32_t struct_size;
	uint32_t available_inputs;
	VkptTemporalImage scene_color;
	VkptTemporalImage motion_vectors;
	VkptTemporalImage view_z;
	VkptTemporalImage device_depth;
	/* Q2RTX currently exports a dense geometric normal in linear XYZ [-1, 1],
	 * a primary-material linear albedo, and scalar perceptual roughness. */
	VkptTemporalImage normals;
	VkptTemporalImage albedo;
	VkptTemporalImage roughness;
	VkptTemporalImage denoiser_normal_roughness_material;
	VkptTemporalImage denoiser_diffuse_albedo;
	VkptTemporalImage denoiser_specular_albedo;
	VkptTemporalImage reactive_mask;
	VkptTemporalImage transparency_and_composition_mask;
	VkptTemporalMotionDescription motion_description;
	VkptTemporalDepthDescription view_z_description;
	VkptTemporalDepthDescription device_depth_description;
	VkptTemporalDenoiserMaterialDescription denoiser_material_description;
} VkptTemporalInputs;

typedef struct VkptTemporalUiDescription_s {
	uint32_t struct_size;
	uint32_t mode;
	VkptTemporalImage scene_target;
	VkptTemporalImage ui_texture;
} VkptTemporalUiDescription;

typedef struct VkptTemporalFrame_s {
	uint32_t struct_size;
	uint32_t contract_version;
	uint32_t stage;
	uint32_t flags;
	uint64_t frame_id;
	float frame_time_ms;
	uint32_t reset_reasons;
	uint32_t history_valid;
	VkExtent2D render_size;
	VkExtent2D previous_render_size;
	VkExtent2D display_size;
	VkptTemporalCamera camera;
	VkptTemporalInputs inputs;
	VkptTemporalUiDescription ui;
} VkptTemporalFrame;

#endif /* VKPT_TEMPORAL_CONTRACT_H */
