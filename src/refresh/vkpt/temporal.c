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

#include "vkpt.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef struct VkptTemporalState_s {
	VkptTemporalFrame frame;
	uint32_t pending_reset_reasons;
	uint64_t previous_frame_id;
	VkExtent2D previous_display_size;
	uint32_t previous_menu_mode;
	VkptTemporalCamera previous_camera;
	uint32_t have_previous_camera;
	uint32_t have_previous_frame;
	uint32_t frame_open;
} VkptTemporalState;

static VkptTemporalState temporal_state = {
	.pending_reset_reasons = VKPT_TEMPORAL_RESET_FIRST_FRAME
};

static bool
extent_is_equal(VkExtent2D a, VkExtent2D b)
{
	return a.width == b.width && a.height == b.height;
}

/*
 * Ordinary camera motion is represented by FLAT_MOTION and must retain
 * temporal history. A teleport, a large instantaneous turn, or a lens jump
 * has no reliable reprojectable predecessor, so make that distinction
 * explicit for every provider. The thresholds are deliberately conservative:
 * Q2's normal maximum movement is a few world units per rendered frame, while
 * 256 units is over six metres under the contract's one-inch unit scale.
 */
static bool
camera_cut_detected(const VkptTemporalCamera *current,
	const VkptTemporalCamera *previous)
{
	const float dx = current->position[0] - previous->position[0];
	const float dy = current->position[1] - previous->position[1];
	const float dz = current->position[2] - previous->position[2];
	const float distance_squared = dx * dx + dy * dy + dz * dz;
	const float forward_dot =
		current->forward[0] * previous->forward[0] +
		current->forward[1] * previous->forward[1] +
		current->forward[2] * previous->forward[2];
	const float fov_delta = fabsf(current->vertical_fov_radians -
		previous->vertical_fov_radians);

	return distance_squared > 256.0f * 256.0f ||
		forward_dot < 0.0f || fov_delta > 0.35f;
}

static bool
temporal_validation_fail(char *reason, size_t reason_size, const char *format, ...)
{
	if (reason && reason_size) {
		va_list args;
		va_start(args, format);
		vsnprintf(reason, reason_size, format, args);
		va_end(args);
	}
	return false;
}

static bool
validate_input_image(const VkptTemporalImage *image, VkExtent2D render_size,
	const char *name, char *reason, size_t reason_size)
{
	if (!image || image->struct_size != sizeof(*image))
		return temporal_validation_fail(reason, reason_size,
			"%s image ABI mismatch", name);
	if (!(image->flags & VKPT_TEMPORAL_RESOURCE_VALID) ||
		!image->image || !image->view)
		return temporal_validation_fail(reason, reason_size,
			"%s image is not valid", name);
	if (!(image->flags & VKPT_TEMPORAL_RESOURCE_DENSE) ||
		!(image->usage & VK_IMAGE_USAGE_SAMPLED_BIT))
		return temporal_validation_fail(reason, reason_size,
			"%s image is not dense/sampled", name);
	if (image->layout != VK_IMAGE_LAYOUT_GENERAL || !image->producer_stage ||
		!image->producer_access)
		return temporal_validation_fail(reason, reason_size,
			"%s image has no readable producer state", name);
	if (!image->allocation_extent.width || !image->allocation_extent.height ||
		!extent_is_equal(image->valid_extent, render_size) ||
		image->valid_extent.width > image->allocation_extent.width ||
		image->valid_extent.height > image->allocation_extent.height)
		return temporal_validation_fail(reason, reason_size,
			"%s image extent is invalid", name);
	if (!(image->value_scale > 0.0f))
		return temporal_validation_fail(reason, reason_size,
			"%s image has an invalid value scale", name);
	return true;
}

static void
initialize_image(VkptTemporalImage *image)
{
	memset(image, 0, sizeof(*image));
	image->struct_size = sizeof(*image);
	image->layout = VK_IMAGE_LAYOUT_UNDEFINED;
	image->queue_family_index = VK_QUEUE_FAMILY_IGNORED;
	image->value_scale = 1.0f;
}

static void
set_image(VkptTemporalImage *image, int image_index, VkFormat format,
	VkExtent2D allocation_extent, VkExtent2D valid_extent, uint32_t flags,
	VkPipelineStageFlags producer_stage, VkAccessFlags producer_access,
	float value_scale)
{
	initialize_image(image);
	image->flags = flags | VKPT_TEMPORAL_RESOURCE_VALID;
	image->image = qvk.images[image_index];
	image->view = qvk.images_views[image_index];
	image->format = format;
	image->layout = VK_IMAGE_LAYOUT_GENERAL;
	image->allocation_extent = allocation_extent;
	image->valid_extent = valid_extent;
	image->producer_stage = producer_stage;
	image->producer_access = producer_access;
	image->usage = VK_IMAGE_USAGE_STORAGE_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
		VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		VK_IMAGE_USAGE_SAMPLED_BIT |
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	image->queue_family_index = (uint32_t)qvk.queue_idx_graphics;
	image->value_scale = value_scale;
}

static void
initialize_frame_structures(VkptTemporalFrame *frame)
{
	frame->struct_size = sizeof(*frame);
	frame->contract_version = VKPT_TEMPORAL_CONTRACT_VERSION;
	frame->camera.struct_size = sizeof(frame->camera);
	frame->inputs.struct_size = sizeof(frame->inputs);
	frame->inputs.motion_description.struct_size =
		sizeof(frame->inputs.motion_description);
	frame->inputs.view_z_description.struct_size =
		sizeof(frame->inputs.view_z_description);
	frame->inputs.device_depth_description.struct_size =
		sizeof(frame->inputs.device_depth_description);
	frame->ui.struct_size = sizeof(frame->ui);
	initialize_image(&frame->inputs.scene_color);
	initialize_image(&frame->inputs.motion_vectors);
	initialize_image(&frame->inputs.view_z);
	initialize_image(&frame->inputs.device_depth);
	initialize_image(&frame->inputs.normals);
	initialize_image(&frame->inputs.albedo);
	initialize_image(&frame->inputs.roughness);
	initialize_image(&frame->inputs.reactive_mask);
	initialize_image(&frame->inputs.transparency_and_composition_mask);
	initialize_image(&frame->ui.scene_target);
	initialize_image(&frame->ui.ui_texture);
}

void
vkpt_temporal_request_reset(uint32_t reasons)
{
	temporal_state.pending_reset_reasons |= reasons;
}

void
vkpt_temporal_begin_frame(float frame_time_seconds, bool q2_history_valid,
	bool render_world, bool denoised)
{
	const QVKUniformBuffer_t *ubo = &vkpt_refdef.uniform_buffer;
	VkptTemporalFrame *frame = &temporal_state.frame;
	uint32_t reset_reasons = temporal_state.pending_reset_reasons;
	if (temporal_state.frame_open)
		reset_reasons |= VKPT_TEMPORAL_RESET_FRAME_DISCONTINUITY;

	memset(frame, 0, sizeof(*frame));
	initialize_frame_structures(frame);

	frame->stage = VKPT_TEMPORAL_STAGE_FRAME_BEGIN;
	frame->frame_id = qvk.frame_counter;
	frame->frame_time_ms = frame_time_seconds * 1000.0f;
	frame->render_size = qvk.extent_render;
	frame->previous_render_size.width = (uint32_t)max(0, ubo->prev_width);
	frame->previous_render_size.height = (uint32_t)max(0, ubo->prev_height);
	frame->display_size = qvk.extent_unscaled;

	if (render_world)
		frame->flags |= VKPT_TEMPORAL_FRAME_RENDER_WORLD;
	else
		reset_reasons |= VKPT_TEMPORAL_RESET_NO_WORLD;
	if (denoised)
		frame->flags |= VKPT_TEMPORAL_FRAME_DENOISED;
	if (ubo->pt_projection == PROJECTION_RECTILINEAR)
		frame->flags |= VKPT_TEMPORAL_FRAME_RECTILINEAR_PROJECTION;
	frame->flags |= VKPT_TEMPORAL_FRAME_LINEAR_HDR_INPUT;
	if (qvk.surf_is_hdr)
		frame->flags |= VKPT_TEMPORAL_FRAME_HDR_PRESENTATION;
	if (qvk.frame_menu_mode)
		frame->flags |= VKPT_TEMPORAL_FRAME_MENU_MODE;
	if (qvk.device_count == 1)
		frame->flags |= VKPT_TEMPORAL_FRAME_SINGLE_DEVICE;

	/* The internal ASVGF history is intentionally disabled in noisy/reference
	 * modes, but an external temporal upscaler still owns valid history across
	 * consecutive presented frames.  Only inherit Q2's invalidation when this
	 * frame actually consumes denoised Q2 history. */
	if (denoised && !q2_history_valid)
		reset_reasons |= VKPT_TEMPORAL_RESET_HISTORY_INVALID;
	if (temporal_state.have_previous_frame &&
		frame->frame_id != temporal_state.previous_frame_id + 1)
		reset_reasons |= VKPT_TEMPORAL_RESET_FRAME_DISCONTINUITY;
	if (frame->previous_render_size.width != 0 &&
		!extent_is_equal(frame->render_size, frame->previous_render_size))
		reset_reasons |= VKPT_TEMPORAL_RESET_RENDER_SIZE_CHANGED;
	if (temporal_state.have_previous_frame &&
		!extent_is_equal(frame->display_size, temporal_state.previous_display_size))
		reset_reasons |= VKPT_TEMPORAL_RESET_DISPLAY_SIZE_CHANGED;
	if (temporal_state.have_previous_frame &&
		(uint32_t)qvk.frame_menu_mode != temporal_state.previous_menu_mode)
		reset_reasons |= VKPT_TEMPORAL_RESET_MENU_TRANSITION;

	frame->reset_reasons = reset_reasons;
	frame->history_valid = temporal_state.have_previous_frame &&
		reset_reasons == 0;

	frame->camera.projection =
		(ubo->pt_projection == PROJECTION_RECTILINEAR)
		? VKPT_TEMPORAL_PROJECTION_RECTILINEAR
		: VKPT_TEMPORAL_PROJECTION_NONLINEAR;
	memcpy(frame->camera.view, ubo->V, sizeof(frame->camera.view));
	memcpy(frame->camera.view_inverse, ubo->invV,
		sizeof(frame->camera.view_inverse));
	memcpy(frame->camera.projection_matrix, ubo->P,
		sizeof(frame->camera.projection_matrix));
	memcpy(frame->camera.projection_inverse, ubo->invP,
		sizeof(frame->camera.projection_inverse));
	memcpy(frame->camera.previous_view, ubo->V_prev,
		sizeof(frame->camera.previous_view));
	memcpy(frame->camera.previous_projection, ubo->P_prev,
		sizeof(frame->camera.previous_projection));
	frame->camera.jitter_render_pixels[0] = ubo->sub_pixel_jitter[0];
	frame->camera.jitter_render_pixels[1] = ubo->sub_pixel_jitter[1];
	frame->camera.jitter_normalized_uv[0] =
		ubo->sub_pixel_jitter[0] / (float)max(1u, frame->render_size.width);
	frame->camera.jitter_normalized_uv[1] =
		ubo->sub_pixel_jitter[1] / (float)max(1u, frame->render_size.height);
	frame->camera.near_plane = vkpt_refdef.z_near;
	frame->camera.far_plane = vkpt_refdef.z_far;
	frame->camera.view_space_to_meters = 0.0254f;
	frame->camera.vertical_fov_radians = 2.0f * atanf(
		1.0f / max(0.001f, fabsf(frame->camera.projection_matrix[5])));
	/* Q2RTX's GLSL primary-ray code uses invV[0..2] as right/up/forward.
	 * Matrix storage is column-major in the uniform block, hence the 0/4/8
	 * strides here and the translation in column 3. */
	frame->camera.right[0] = frame->camera.view_inverse[0];
	frame->camera.right[1] = frame->camera.view_inverse[1];
	frame->camera.right[2] = frame->camera.view_inverse[2];
	frame->camera.up[0] = frame->camera.view_inverse[4];
	frame->camera.up[1] = frame->camera.view_inverse[5];
	frame->camera.up[2] = frame->camera.view_inverse[6];
	frame->camera.forward[0] = frame->camera.view_inverse[8];
	frame->camera.forward[1] = frame->camera.view_inverse[9];
	frame->camera.forward[2] = frame->camera.view_inverse[10];
	frame->camera.position[0] = frame->camera.view_inverse[12];
	frame->camera.position[1] = frame->camera.view_inverse[13];
	frame->camera.position[2] = frame->camera.view_inverse[14];

	if (render_world && temporal_state.have_previous_camera &&
		camera_cut_detected(&frame->camera, &temporal_state.previous_camera)) {
		reset_reasons |= VKPT_TEMPORAL_RESET_CAMERA_CUT;
		/* The initial assignment above predates camera population. Update the
		 * externally visible values after this transform-based reset decision. */
		frame->reset_reasons = reset_reasons;
		frame->history_valid = 0;
	}

	frame->inputs.motion_description.space =
		VKPT_TEMPORAL_MOTION_NORMALIZED_UV;
	frame->inputs.motion_description.direction =
		VKPT_TEMPORAL_MOTION_CURRENT_TO_PREVIOUS;
	/* P and P_prev are unjittered when FLAT_MOTION is generated.  Camera
	 * jitter is applied separately to the primary-ray sample position, so a
	 * static surface produces zero motion and consumers must select their
	 * jitter-free motion-vector permutation. */
	frame->inputs.motion_description.jitter_mode =
		VKPT_TEMPORAL_MOTION_JITTER_FREE;
	frame->inputs.motion_description.to_render_pixels[0] =
		(float)frame->render_size.width;
	frame->inputs.motion_description.to_render_pixels[1] =
		(float)frame->render_size.height;
	frame->inputs.motion_description.radial_depth_delta_channel = 2;
	frame->inputs.motion_description.depth_footprint_channel = 3;

	frame->inputs.view_z_description.convention =
		VKPT_TEMPORAL_DEPTH_VIEW_Z_POSITIVE_FORWARD;
	frame->inputs.view_z_description.sky_value = PRIMARY_RAY_T_MAX;
	frame->inputs.view_z_description.primary_surface_only = 1;
	frame->inputs.device_depth_description.convention =
		VKPT_TEMPORAL_DEPTH_UNAVAILABLE;

	frame->ui.mode = VKPT_TEMPORAL_UI_DIRECT_AFTER_SCENE;

	temporal_state.pending_reset_reasons = 0;
	temporal_state.frame_open = 1;
}

void
vkpt_temporal_mark_inputs_ready(void)
{
	VkptTemporalFrame *frame = &temporal_state.frame;

	if (!temporal_state.frame_open)
		return;

	set_image(&frame->inputs.scene_color, VKPT_IMG_FLAT_COLOR,
		VK_FORMAT_R16G16B16A16_SFLOAT, qvk.extent_screen_images,
		qvk.extent_render,
		VKPT_TEMPORAL_RESOURCE_DENSE |
		VKPT_TEMPORAL_RESOURCE_LINEAR |
		VKPT_TEMPORAL_RESOURCE_PRE_TONEMAP |
		VKPT_TEMPORAL_RESOURCE_PRE_UI |
		VKPT_TEMPORAL_RESOURCE_STORAGE_SCALED |
		VKPT_TEMPORAL_RESOURCE_ALPHA_METADATA,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		1.0f / (float)STORAGE_SCALE_HDR);
	frame->inputs.available_inputs |= VKPT_TEMPORAL_INPUT_SCENE_COLOR;

	set_image(&frame->inputs.motion_vectors, VKPT_IMG_FLAT_MOTION,
		VK_FORMAT_R16G16B16A16_SFLOAT, qvk.extent_screen_images,
		qvk.extent_render,
		VKPT_TEMPORAL_RESOURCE_DENSE |
		VKPT_TEMPORAL_RESOURCE_PRE_UI,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		1.0f);
	frame->inputs.available_inputs |= VKPT_TEMPORAL_INPUT_MOTION_VECTORS;

	/* These masks are primary-surface authored rather than inferred from the
	 * denoised image.  That preserves explicit knowledge of transparent,
	 * animated and view-model material paths for any temporal provider. */
	set_image(&frame->inputs.reactive_mask, VKPT_IMG_TEMPORAL_REACTIVE_MASK,
		VK_FORMAT_R8_UNORM, qvk.extent_screen_images, qvk.extent_render,
		VKPT_TEMPORAL_RESOURCE_DENSE |
		VKPT_TEMPORAL_RESOURCE_PRE_UI |
		VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE,
		qvk.use_ray_query ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		VK_ACCESS_SHADER_WRITE_BIT, 1.0f);
	frame->inputs.available_inputs |= VKPT_TEMPORAL_INPUT_REACTIVE_MASK;

	set_image(&frame->inputs.transparency_and_composition_mask,
		VKPT_IMG_TEMPORAL_COMPOSITION_MASK, VK_FORMAT_R8_UNORM,
		qvk.extent_screen_images, qvk.extent_render,
		VKPT_TEMPORAL_RESOURCE_DENSE |
		VKPT_TEMPORAL_RESOURCE_PRE_UI |
		VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE,
		qvk.use_ray_query ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		VK_ACCESS_SHADER_WRITE_BIT, 1.0f);
	frame->inputs.available_inputs |=
		VKPT_TEMPORAL_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK;

	/* checkerboard_interleave writes dense material attributes after every
	 * reflection/refraction path has selected its primary surface. They are a
	 * reusable substrate for denoising providers; no current FSR3/FSR4 dispatch
	 * consumes them yet. */
	set_image(&frame->inputs.normals, VKPT_IMG_TEMPORAL_NORMALS,
		VK_FORMAT_R16G16B16A16_SFLOAT, qvk.extent_screen_images,
		qvk.extent_render,
		VKPT_TEMPORAL_RESOURCE_DENSE | VKPT_TEMPORAL_RESOURCE_LINEAR |
		VKPT_TEMPORAL_RESOURCE_PRE_UI | VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, 1.0f);
	set_image(&frame->inputs.albedo, VKPT_IMG_TEMPORAL_ALBEDO,
		VK_FORMAT_R16G16B16A16_SFLOAT, qvk.extent_screen_images,
		qvk.extent_render,
		VKPT_TEMPORAL_RESOURCE_DENSE | VKPT_TEMPORAL_RESOURCE_LINEAR |
		VKPT_TEMPORAL_RESOURCE_PRE_UI | VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, 1.0f);
	set_image(&frame->inputs.roughness, VKPT_IMG_TEMPORAL_ROUGHNESS,
		VK_FORMAT_R16_SFLOAT, qvk.extent_screen_images, qvk.extent_render,
		VKPT_TEMPORAL_RESOURCE_DENSE | VKPT_TEMPORAL_RESOURCE_LINEAR |
		VKPT_TEMPORAL_RESOURCE_PRE_UI | VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, 1.0f);
	if (qvk.device_count == 1)
		frame->inputs.available_inputs |= VKPT_TEMPORAL_INPUT_NORMALS |
			VKPT_TEMPORAL_INPUT_ALBEDO | VKPT_TEMPORAL_INPUT_ROUGHNESS;

	/* Primary rays write directly to dense screen coordinates.  That is
	 * correct for Q2RTX's normal single-device path.  Device-group rendering
	 * needs an explicit cross-device gather before this slot can be advertised. */
	if (qvk.device_count == 1) {
		VkPipelineStageFlags primary_depth_stage = qvk.use_ray_query
			? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
			: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

		set_image(&frame->inputs.view_z, VKPT_IMG_TEMPORAL_VIEW_Z,
			VK_FORMAT_R32_SFLOAT, qvk.extent_screen_images,
			qvk.extent_render,
			VKPT_TEMPORAL_RESOURCE_DENSE |
			VKPT_TEMPORAL_RESOURCE_LINEAR |
			VKPT_TEMPORAL_RESOURCE_PRE_UI |
			VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE,
			primary_depth_stage,
			VK_ACCESS_SHADER_WRITE_BIT, 1.0f);
		frame->inputs.available_inputs |= VKPT_TEMPORAL_INPUT_VIEW_Z;

		set_image(&frame->inputs.device_depth,
			VKPT_IMG_TEMPORAL_DEVICE_DEPTH, VK_FORMAT_R32_SFLOAT,
			qvk.extent_screen_images, qvk.extent_render,
			VKPT_TEMPORAL_RESOURCE_DENSE |
			VKPT_TEMPORAL_RESOURCE_PRE_UI |
			VKPT_TEMPORAL_RESOURCE_SINGLE_DEVICE,
			primary_depth_stage,
			VK_ACCESS_SHADER_WRITE_BIT, 1.0f);
		frame->inputs.available_inputs |= VKPT_TEMPORAL_INPUT_DEVICE_DEPTH;
		frame->inputs.device_depth_description.convention =
			VKPT_TEMPORAL_DEPTH_DEVICE_ZERO_TO_ONE;
		frame->inputs.device_depth_description.sky_value = 1.0f;
		frame->inputs.device_depth_description.primary_surface_only = 1;
	}

	frame->stage = VKPT_TEMPORAL_STAGE_INPUTS_READY;
}

void
vkpt_temporal_mark_scene_presented(void)
{
	VkptTemporalFrame *frame = &temporal_state.frame;

	if (!temporal_state.frame_open)
		return;

	initialize_image(&frame->ui.scene_target);
	frame->ui.scene_target.flags =
		VKPT_TEMPORAL_RESOURCE_VALID | VKPT_TEMPORAL_RESOURCE_PRE_UI;
	frame->ui.scene_target.image =
		qvk.swap_chain_images[qvk.current_swap_chain_image_index];
	frame->ui.scene_target.view =
		qvk.swap_chain_image_views[qvk.current_swap_chain_image_index];
	frame->ui.scene_target.format = qvk.surf_format.format;
	frame->ui.scene_target.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	frame->ui.scene_target.allocation_extent = qvk.extent_unscaled;
	frame->ui.scene_target.valid_extent = qvk.extent_unscaled;
	frame->ui.scene_target.producer_stage =
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	frame->ui.scene_target.producer_access =
		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	frame->ui.scene_target.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	frame->ui.scene_target.queue_family_index =
		(uint32_t)qvk.queue_idx_graphics;
	frame->stage = VKPT_TEMPORAL_STAGE_SCENE_PRESENTED;
}

void
vkpt_temporal_begin_ui_composition(void)
{
	if (!temporal_state.frame_open)
		return;

	temporal_state.frame.stage = VKPT_TEMPORAL_STAGE_UI_COMPOSITION;
}

void
vkpt_temporal_end_ui_composition(void)
{
	if (!temporal_state.frame_open)
		return;

	temporal_state.frame.stage = VKPT_TEMPORAL_STAGE_SCENE_PRESENTED;
}

void
vkpt_temporal_end_frame(bool presented)
{
	VkptTemporalFrame *frame = &temporal_state.frame;

	if (!temporal_state.frame_open)
		return;

	if (presented) {
		frame->stage = VKPT_TEMPORAL_STAGE_PRESENTED;
		temporal_state.previous_frame_id = frame->frame_id;
		temporal_state.previous_display_size = frame->display_size;
		temporal_state.previous_menu_mode =
			(frame->flags & VKPT_TEMPORAL_FRAME_MENU_MODE) != 0;
		temporal_state.previous_camera = frame->camera;
		temporal_state.have_previous_camera = 1;
		temporal_state.have_previous_frame = 1;
	} else {
		frame->stage = VKPT_TEMPORAL_STAGE_CLOSED;
		temporal_state.pending_reset_reasons |=
			VKPT_TEMPORAL_RESET_FRAME_DISCONTINUITY;
	}
	temporal_state.frame_open = 0;
}

const VkptTemporalFrame *
vkpt_temporal_get_frame(void)
{
	return &temporal_state.frame;
}

bool
vkpt_temporal_validate_current_frame(uint32_t required_inputs,
	char *reason, size_t reason_size)
{
	const VkptTemporalFrame *frame = &temporal_state.frame;
	const struct {
		uint32_t bit;
		const VkptTemporalImage *image;
		const char *name;
	} inputs[] = {
		{ VKPT_TEMPORAL_INPUT_SCENE_COLOR, &frame->inputs.scene_color, "scene color" },
		{ VKPT_TEMPORAL_INPUT_MOTION_VECTORS, &frame->inputs.motion_vectors, "motion vectors" },
		{ VKPT_TEMPORAL_INPUT_VIEW_Z, &frame->inputs.view_z, "view Z" },
		{ VKPT_TEMPORAL_INPUT_DEVICE_DEPTH, &frame->inputs.device_depth, "device depth" },
		{ VKPT_TEMPORAL_INPUT_NORMALS, &frame->inputs.normals, "normals" },
		{ VKPT_TEMPORAL_INPUT_ALBEDO, &frame->inputs.albedo, "albedo" },
		{ VKPT_TEMPORAL_INPUT_ROUGHNESS, &frame->inputs.roughness, "roughness" },
		{ VKPT_TEMPORAL_INPUT_REACTIVE_MASK, &frame->inputs.reactive_mask, "reactive mask" },
	};

	if (reason && reason_size)
		reason[0] = '\0';
	if (frame->struct_size != sizeof(*frame) ||
		frame->contract_version != VKPT_TEMPORAL_CONTRACT_VERSION ||
		frame->inputs.struct_size != sizeof(frame->inputs) ||
		frame->camera.struct_size != sizeof(frame->camera))
		return temporal_validation_fail(reason, reason_size,
			"temporal contract ABI mismatch");
	if (frame->stage != VKPT_TEMPORAL_STAGE_INPUTS_READY ||
		!frame->render_size.width || !frame->render_size.height ||
		!frame->display_size.width || !frame->display_size.height ||
		frame->render_size.width > frame->display_size.width ||
		frame->render_size.height > frame->display_size.height)
		return temporal_validation_fail(reason, reason_size,
			"temporal frame extents/stage are invalid");
	if (frame->inputs.motion_description.struct_size !=
		sizeof(frame->inputs.motion_description) ||
		frame->inputs.motion_description.direction !=
			VKPT_TEMPORAL_MOTION_CURRENT_TO_PREVIOUS ||
		!(frame->inputs.motion_description.to_render_pixels[0] > 0.0f) ||
		!(frame->inputs.motion_description.to_render_pixels[1] > 0.0f) ||
		!(frame->camera.vertical_fov_radians > 0.0f) ||
		!(frame->camera.view_space_to_meters > 0.0f))
		return temporal_validation_fail(reason, reason_size,
			"temporal motion/camera metadata is invalid");
	if ((required_inputs & VKPT_TEMPORAL_INPUT_DEVICE_DEPTH) &&
		frame->inputs.device_depth_description.convention ==
			VKPT_TEMPORAL_DEPTH_UNAVAILABLE)
		return temporal_validation_fail(reason, reason_size,
			"device-depth convention is unavailable");
	if ((required_inputs & VKPT_TEMPORAL_INPUT_VIEW_Z) &&
		frame->inputs.view_z_description.convention ==
			VKPT_TEMPORAL_DEPTH_UNAVAILABLE)
		return temporal_validation_fail(reason, reason_size,
			"view-Z convention is unavailable");

	for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
		if (!(required_inputs & inputs[i].bit))
			continue;
		if (!(frame->inputs.available_inputs & inputs[i].bit))
			return temporal_validation_fail(reason, reason_size,
				"%s is unavailable", inputs[i].name);
		if (!validate_input_image(inputs[i].image, frame->render_size,
			inputs[i].name, reason, reason_size))
			return false;
	}
	return true;
}
