import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union

# Originally from https://github.com/StevenShaw1999/RnB
class GaussianSmoothing(nn.Module):
    """
    Apply Gaussian smoothing on a 1D, 2D, or 3D tensor. Filtering is performed separately for each channel
    in the input using depthwise convolution.

    Args:
        channels (int): Number of channels in the input tensors. The output will have the same number of channels.
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or sequence): Standard deviation of the Gaussian kernel.
        dim (int, optional): Number of dimensions of the data. Default is 2 (spatial).
    """

    def __init__(self, channels: int = 1, kernel_size: int = 3, sigma: float = 0.5, dim: int = 2):
        super().__init__()

        # Ensure kernel_size and sigma are lists of length `dim`
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # Create the Gaussian kernel
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (1 / (std * math.sqrt(2 * math.pi))) * torch.exp(
                -((mgrid - mean) ** 2) / (2 * std ** 2)
            )

        # Normalize kernel to ensure the sum equals 1
        kernel = kernel / torch.sum(kernel)

        # Reshape kernel for depthwise convolution
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        # Assign the appropriate convolution function based on the dimension
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f"Only 1, 2, and 3 dimensions are supported. Received {dim}.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply the Gaussian smoothing to the input tensor.

        Args:
            input (torch.Tensor): The input tensor to apply smoothing on.

        Returns:
            torch.Tensor: The smoothed tensor.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
    
# Originally from https://github.com/StevenShaw1999/RnB
def edge_loss(attn_map, mask, iou, pipe):
    """
    Compute edge loss.
    """
    mask_clone = mask.clone()[1:-1, 1:-1]

    attn_map_clone = attn_map.unsqueeze(0).unsqueeze(0)        
    attn_map_clone = attn_map_clone / (attn_map_clone.max().detach() + 1e-4)
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode='reflect')
    
    # Smoothing
    attn_map_clone = pipe.smth_3(attn_map_clone)

    sobel_output_x = pipe.sobel_conv_x(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_output_y = pipe.sobel_conv_y(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_sum = torch.sqrt(sobel_output_y ** 2  + sobel_output_x ** 2)

    loss_ = 1 - (sobel_sum * mask_clone).sum() / (sobel_sum.sum() + 1e-4) * (1 - iou)
    
    return loss_

# Originally from https://github.com/StevenShaw1999/RnB
def aggregated_grounding_loss(attn_maps, bbox_list, object_positions, latent_h, latent_w, pipe):
    '''
    Computes aggregated groundiung loss used in Global Update.
    '''
    loss = 0
    object_number = len(bbox_list)
    device = pipe._execution_device

    if object_number == 0:
        return torch.tensor(0).float().to(device) if torch.cuda.is_available() else torch.tensor(0).float()

    # Get the attention maps
    attn_all = attn_maps[1]
    attn_edge = attn_maps[1] 
    attn_edge = attn_edge.reshape(latent_h, latent_w, -1)
    assert attn_all.shape[0] == latent_h * latent_w, f"attn_all.shape[0]: {attn_all.shape[0]}, height * width: {latent_h * latent_w}"
    
    # Reshape attn_all to [height, width, num_tokens]
    attn_all = attn_all.reshape(latent_h, latent_w, -1)
    obj_loss = 0

    for obj_idx in range(object_number):
        for num, obj_pos in enumerate(object_positions[obj_idx]):
            if num == 0:
                att_map_obj_raw = attn_all[:, :, obj_pos]
                att_map_edge = attn_edge[:, :, obj_pos]
            else:
                att_map_obj_raw = att_map_obj_raw + attn_all[:, :, obj_pos]
                att_map_edge = att_map_edge + attn_edge[:, :, obj_pos]

        attn_norm = (att_map_obj_raw - att_map_obj_raw.min()) / (att_map_obj_raw.max() - att_map_obj_raw.min() + 1e-4)

        # Init mask
        mask = torch.zeros(size = (latent_h, latent_w)).to(device) 
        mask_clone = mask.clone()

        for obj_box in bbox_list[obj_idx]:
            x_min = int(obj_box[0] * latent_w)
            y_min = int(obj_box[1] * latent_h)
            x_max = int(obj_box[2] * latent_w)
            y_max = int(obj_box[3] * latent_h)

            # Apply mask for the object
            mask[y_min: y_max, x_min: x_max] = 1
        
        # Background region
        mask_none_cls = (1 - mask)

        # Set threshold
        if mask_none_cls.sum() != 0:
            threshold = (attn_norm * mask).sum() / mask.sum() / 5 * 2
            threshold = threshold + ((attn_norm * mask_none_cls).sum() / mask_none_cls.sum() / 5 * 3)
        else:
            threshold = 0

        thres_image = attn_norm.gt(threshold) * 1.0
        noise_image = F.sigmoid(20 * (attn_norm - threshold))

        rows, cols = torch.where(thres_image > 0.3)
        if len(rows) == 0 or len(cols) == 0:
            continue
        x1, y1 = cols.min(), rows.min()
        x2, y2 = cols.max(), rows.max()

        mask_aug = mask_clone
        mask_aug[y1: y2, x1: x2] = 1    
        mask_aug_in = mask_aug * mask 
        iou = (mask_aug * mask).sum() / torch.max(mask_aug, mask).sum()

        if iou < 0.85:
            this_cls_diff_aug_1 = (mask_aug - attn_norm).detach() + attn_norm
            this_cls_diff_aug_in_1 = (mask_aug_in - attn_norm).detach() + attn_norm
            
            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum() * (1 / this_cls_diff_aug_1.sum().detach())
            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum().detach() * (1 / this_cls_diff_aug_1.sum())
            if object_number > 1 and obj_idx > -1:
                if (att_map_obj_raw * mask).max() < (att_map_obj_raw * (1 - mask)).max():
                    obj_loss += edge_loss(att_map_edge, mask, iou, pipe) * 1 

            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum() * (1 / noise_image.sum().detach())) * 0.5
            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum().detach() * (1 / noise_image.sum())) * 0.5
    
    loss = loss + (obj_loss / object_number)
    
    return loss

# ================================================================================ #

def GroundingUpdateFunc(pipe, latents, timestep, timestep_idx, denoiser_args, loss_args, enabled: bool = True):
    """
    Update latents during the denoising process with grounding loss.

    Args:
        pipe: The pipeline object with scheduling and model functions.
        latents: The latent tensor to update.
        timestep: Current timestep in the denoising schedule.
        timestep_idx: Index of the current timestep.
        denoiser_args: Dictionary containing denoiser parameters.
        loss_args: Dictionary containing loss-related parameters.

    Returns:
        Tuple: Updated latents and the computed loss.
    """
    # Clone and prepare latents
    latents = latents.clone().detach().float().requires_grad_(True)

    # Scale latent model input
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    # Process timestep for transformer
    timestep = pipe._timestep_process(timestep, latent_model_input.device, latent_model_input.shape[0])

    # Forward pass through the transformer to get attention maps
    _, attn_maps, _ = pipe.transformer(
        latent_model_input.half(),
        encoder_hidden_states=denoiser_args["prompt_embeds"],
        encoder_attention_mask=denoiser_args["prompt_attention_mask"],
        timestep=timestep,
        added_cond_kwargs=denoiser_args["added_cond_kwargs"],
        return_dict=False,
        return_attn_maps=True,  # GrounDiT flag
        is_object_branch=False,  # GrounDiT flag
    )

    # Calculate the grounding loss
    loss = aggregated_grounding_loss(
        attn_maps,
        bbox_list=loss_args["bbox_list"],
        object_positions=loss_args["phrases_idx"],
        latent_h=loss_args["height"] // (pipe.vae_scale_factor * 2),
        latent_w=loss_args["width"] // (pipe.vae_scale_factor * 2),
        pipe=pipe,
    )
    loss = loss * loss_args["loss_scale"]

    # Update latents based on gradient if loss is non-zero
    if loss != 0:
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
        grad_cond = torch.nan_to_num(grad_cond)
        latents = latents - grad_cond * loss_args["gradient_weight"]
    elif timestep_idx < 5:
        loss = 10

    # Cleanup
    del attn_maps
    torch.cuda.empty_cache()

    return latents.half(), loss


def ClipGuidanceUpdateFunc(pipe, latents, timestep, timestep_idx, denoiser_args, clip_args, enabled: bool = True):
    """
    Update latents using CLIP image guidance.
    Supports one or multiple reference images and optional per-reference regions.
    """
    if not enabled:
        return latents.half(), None

    # Clone and prepare latents
    latents = latents.clone().detach().float().requires_grad_(True)

    # Prepare latent model input (respect CFG)
    do_cfg = denoiser_args.get("do_cfg", False)
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    # Process timestep for transformer
    original_timestep = timestep
    current_timestep = pipe._timestep_process(timestep, latent_model_input.device, latent_model_input.shape[0])

    # Forward pass (no attention maps needed)
    noise_pred, _, _ = pipe.transformer(
        latent_model_input.half(),
        encoder_hidden_states=denoiser_args["prompt_embeds"],
        encoder_attention_mask=denoiser_args["prompt_attention_mask"],
        timestep=current_timestep,
        added_cond_kwargs=denoiser_args["added_cond_kwargs"],
        return_dict=False,
        return_attn_maps=False,
        is_object_branch=False,
    )

    # CFG and "correction" as in DDIM reference
    correction = None
    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        correction = noise_pred_text - noise_pred_uncond
        noise_pred = noise_pred_uncond + denoiser_args.get("guidance_scale", 1.0) * correction

    # Adjust noise prediction for learned sigma
    latent_channels = latents.shape[1]
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        noise_pred = noise_pred.chunk(2, dim=1)[0]

    clip_loss = None
    clip_guidance_scale = clip_args.get("clip_guidance_scale", 0.0)
    if clip_guidance_scale is not None and clip_guidance_scale > 0.0:
        total_norm = None
        # Resolve encoders: prefer list attribute if available, otherwise fallback to single encoder
        encoders = getattr(pipe, "clip_image_encoders", None)
        if (encoders is None or len(encoders) == 0) and getattr(pipe, "clip_image_encoder", None) is not None:
            encoders = [pipe.clip_image_encoder]

        if encoders is not None and len(encoders) > 0:
            # Predict clean latents x0 (without advancing scheduler state)
            pred_x0_latents = pipe.predict_x0_noadvance(noise_pred, original_timestep, latents)
            # Decode predicted x0 to pixel space in [-1, 1]
            z = (pred_x0_latents.to(dtype=pipe.vae.dtype)) / pipe.vae.config.scaling_factor
            x0_image = pipe.vae.decode(z, return_dict=False)[0].float()

            # Regions: either single or list aligned to encoders
            clip_regions = clip_args.get("regions", None)

            for idx, enc in enumerate(encoders):
                region_i = None
                if clip_regions is not None and isinstance(clip_regions, (list, tuple)):
                    region_i = clip_regions[idx]

                x_for_ref = crop_tensor_by_bbox(x0_image, region_i) if region_i is not None else x0_image
                # Style loss via Gram-matrix residual between predicted x0 and reference image
                residual_i = enc.get_gram_matrix_residual(x_for_ref)  # expects [-1, 1] input
                norm_i = torch.linalg.norm(residual_i)
                total_norm = norm_i if total_norm is None else (total_norm + norm_i)

            if total_norm is not None:
                clip_loss = total_norm * clip_guidance_scale

    # DDIM-aligned gradient scaling:
    # rho = RMS(correction) * guidance_scale / RMS(norm_grad) * 0.2
    # and update: latents = latents - rho * norm_grad.detach()
    if clip_loss is not None and total_norm is not None and (clip_loss.requires_grad or (hasattr(clip_loss, "grad_fn") and clip_loss.grad_fn is not None)):
        # Gradient of total style norm w.r.t latents
        norm_grad = torch.autograd.grad(outputs=total_norm.requires_grad_(True), inputs=latents, retain_graph=False, create_graph=False)[0]
        norm_grad = torch.nan_to_num(norm_grad)

        #import pdb; pdb.set_trace()
        # Compute RMS of correction if available; otherwise fall back to 1.0 to avoid zero scaling
        if correction is not None:
            correction_rms = (correction * correction).mean().sqrt().item()
        else:
            correction_rms = 1.0

        norm_grad_rms = (norm_grad * norm_grad).mean().sqrt().item() + 1e-8
        rho = correction_rms * denoiser_args.get("guidance_scale", 1.0) / norm_grad_rms * 0.2

        # Apply update
        latents = latents - rho * norm_grad.detach()

    print(f"Clip (style) loss: {clip_loss}")

    # Cleanup
    del noise_pred
    if "x0_image" in locals():
        del x0_image
    torch.cuda.empty_cache()

    return latents.half(), clip_loss


def TextGuidanceUpdateFunc(pipe, latents, timestep, timestep_idx, denoiser_args, text_args, enabled: bool = True):
    """
    Update latents using CLIP text guidance via text-image residual.
    This is independent from grounding and image style (Gram) guidance.
    """
    if not enabled:
        return latents.half(), None

    texts = text_args.get("texts", None)
    text = text_args.get("text", None)
    text_guidance_scale = text_args.get("text_guidance_scale", 0.0)
    if (texts is None and text is None) or text_guidance_scale is None or text_guidance_scale <= 0.0:
        return latents.half(), None

    # Require a prepared CLIP text encoder on the pipeline
    if getattr(pipe, "clip_text_encoder", None) is None:
        return latents.half(), None

    # Clone and prepare latents
    latents = latents.clone().detach().float().requires_grad_(True)

    # Prepare latent model input (respect CFG)
    do_cfg = denoiser_args.get("do_cfg", False)
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    # Process timestep for transformer
    original_timestep = timestep
    current_timestep = pipe._timestep_process(timestep, latent_model_input.device, latent_model_input.shape[0])

    # Forward pass (no attention maps needed)
    noise_pred, _, _ = pipe.transformer(
        latent_model_input.half(),
        encoder_hidden_states=denoiser_args["prompt_embeds"],
        encoder_attention_mask=denoiser_args["prompt_attention_mask"],
        timestep=current_timestep,
        added_cond_kwargs=denoiser_args["added_cond_kwargs"],
        return_dict=False,
        return_attn_maps=False,
        is_object_branch=False,
    )

    # CFG and "correction" as in DDIM reference
    correction = None
    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        correction = noise_pred_text - noise_pred_uncond
        noise_pred = noise_pred_uncond + denoiser_args.get("guidance_scale", 1.0) * correction

    # Adjust noise prediction for learned sigma
    latent_channels = latents.shape[1]
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        noise_pred = noise_pred.chunk(2, dim=1)[0]

    text_loss = None
    # Predict clean latents x0 (without advancing scheduler state)
    pred_x0_latents = pipe.predict_x0_noadvance(noise_pred, original_timestep, latents)
    # Decode predicted x0 to pixel space in [-1, 1]
    z = (pred_x0_latents.to(dtype=pipe.vae.dtype)) / pipe.vae.config.scaling_factor
    x0_image = pipe.vae.decode(z, return_dict=False)[0].float()

    # Optional regions for text guidance
    regions = text_args.get("regions", None)

    # Compute residual norm over single or multiple texts/regions
    total_norm = None
    if texts is not None and isinstance(texts, (list, tuple)) and len(texts) > 0:
        for idx, txt in enumerate(texts):
            region_i = None
            if regions is not None and isinstance(regions, (list, tuple)):
                region_i = regions[idx]
            x_for_txt = crop_tensor_by_bbox(x0_image, region_i) if region_i is not None else x0_image
            residual_i = pipe.clip_text_encoder.get_residual(x_for_txt, txt)
            norm_i = torch.linalg.norm(residual_i)
            if total_norm is None:
                total_norm = norm_i
            else:
                total_norm = total_norm + norm_i

    text_loss = total_norm * text_guidance_scale

    # Compute gradient and DDIM-aligned scaling (similar to style guidance)
    if text_loss is not None and (text_loss.requires_grad or (hasattr(text_loss, "grad_fn") and text_loss.grad_fn is not None)):
        norm_grad = torch.autograd.grad(outputs=total_norm.requires_grad_(True), inputs=latents, retain_graph=False, create_graph=False)[0]
        norm_grad = torch.nan_to_num(norm_grad)

        if correction is not None:
            correction_rms = (correction * correction).mean().sqrt().item()
        else:
            correction_rms = 1.0

        norm_grad_rms = (norm_grad * norm_grad).mean().sqrt().item() + 1e-8
        rho = correction_rms * denoiser_args.get("guidance_scale", 1.0) / norm_grad_rms * 0.2

        # Apply update
        latents = latents - rho * norm_grad.detach()

    print(f"Clip (text) loss: {text_loss}")

    # Cleanup
    del noise_pred
    torch.cuda.empty_cache()

    return latents.half(), text_loss


def _compute_sobel_edge_map_2d(x_2d: torch.Tensor, pipe) -> torch.Tensor:
    """
    Compute Sobel edge magnitude map on a single-channel 2D tensor using the pipeline's Sobel and smoothing.
    Expects x_2d of shape [H, W] in float dtype. Returns a tensor of shape [H, W], normalized to [0, 1].
    """
    device = pipe._execution_device
    x = x_2d.to(device=device, dtype=pipe.smth_3.weight.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    x = x / (x.max() + 1e-6) if x.max() > 0 else x
    # reflect pad to keep size after conv and reduce border artifacts
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    # smoothing then sobel
    x = pipe.smth_3(x)
    sobel_x = pipe.sobel_conv_x(x)
    sobel_y = pipe.sobel_conv_y(x)
    mag = torch.sqrt(sobel_x.pow(2) + sobel_y.pow(2))  # [1,1,H,W]
    mag = mag.squeeze(0).squeeze(0)       # [H, W]
    mag = mag / (mag.max() + 1e-6) if mag.max() > 0 else mag
    return mag


def _normalize_and_clip_bbox(bbox: Union[List[float], Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Validate and clip a normalized bbox [x_min, y_min, x_max, y_max] to [0,1].
    Returns a tuple if valid area exists, else None.
    """
    if bbox is None:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    x0 = float(max(0.0, min(1.0, x0)))
    y0 = float(max(0.0, min(1.0, y0)))
    x1 = float(max(0.0, min(1.0, x1)))
    y1 = float(max(0.0, min(1.0, y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def crop_tensor_by_bbox(x: torch.Tensor, bbox: Union[List[float], Tuple[float, float, float, float]]) -> torch.Tensor:
    """
    Crop a BCHW tensor to a normalized bbox [x_min, y_min, x_max, y_max] in [0,1] coordinates.
    Returns the cropped tensor. If bbox invalid, returns the input tensor unchanged.
    """
    bbox_norm = _normalize_and_clip_bbox(bbox)
    if bbox_norm is None:
        return x
    if x.ndim != 4:
        return x
    _, _, H, W = x.shape
    x0, y0, x1, y1 = bbox_norm
    x0i = max(0, min(W - 1, int(round(x0 * W))))
    y0i = max(0, min(H - 1, int(round(y0 * H))))
    x1i = max(0, min(W, int(round(x1 * W))))
    y1i = max(0, min(H, int(round(y1 * H))))
    if x1i <= x0i:
        x1i = min(W, x0i + 1)
    if y1i <= y0i:
        y1i = min(H, y0i + 1)
    return x[:, :, y0i:y1i, x0i:x1i]


def save_attention_edge_maps(
    attn_maps,
    object_positions,
    bbox_list,
    latent_h: int,
    latent_w: int,
    pipe,
    save_dir: str,
    step_idx: int,
) -> None:
    """
    Save per-object attention edge maps (Sobol magnitude) to `save_dir` for the given timestep index.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception:
        return

    num_objects = len(bbox_list) if bbox_list is not None else 0
    if num_objects == 0:
        return

    attn_all = attn_maps[1]
    attn_all = attn_all.reshape(latent_h, latent_w, -1)  # [H, W, tokens]

    for obj_idx in range(num_objects):
        token_indices = object_positions[obj_idx] if object_positions is not None else None
        if token_indices is None or len(token_indices) == 0:
            continue

        attn_obj = None
        for k, tok in enumerate(token_indices):
            if k == 0:
                attn_obj = attn_all[:, :, tok]
            else:
                attn_obj = attn_obj + attn_all[:, :, tok]
        if attn_obj is None:
            continue

        attn_obj = (attn_obj - attn_obj.min()) / (attn_obj.max() - attn_obj.min() + 1e-6)
        attn_edge = _compute_sobel_edge_map_2d(attn_obj, pipe)

        arr = (attn_edge.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
        im = Image.fromarray(arr, mode="L")
        im.save(os.path.join(save_dir, f"step_{step_idx:04d}_obj_{obj_idx:02d}.png"))


def edge_guidance_loss(attn_maps, object_positions, object_edge_maps, latent_h, latent_w, pipe) -> torch.Tensor:
    """
    Compute edge guidance loss as mean L1 distance between attention edge maps and provided per-object target edge maps.
    - attn_maps: attention maps output (expects cross-attn prob tensor at attn_maps[1])
    - object_positions: list[List[int]] token indices for each object phrase
    - object_edge_maps: list[Tensor[latent_h, latent_w]] target edge maps in [0,1]
    - latent_h, latent_w: spatial resolution of attention maps
    """
    device = pipe._execution_device
    num_objects = len(object_edge_maps) if object_edge_maps is not None else 0
    if num_objects == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    attn_all = attn_maps[1]
    # [HW, tokens] -> [H, W, tokens]
    attn_all = attn_all.reshape(latent_h, latent_w, -1)

    total_loss = 0.0
    valid_count = 0

    for obj_idx in range(num_objects):
        token_indices = object_positions[obj_idx]
        if token_indices is None or len(token_indices) == 0:
            continue

        # Aggregate attention over all tokens for this object
        attn_obj = None
        for k, tok in enumerate(token_indices):
            if k == 0:
                attn_obj = attn_all[:, :, tok]
            else:
                attn_obj = attn_obj + attn_all[:, :, tok]

        if attn_obj is None:
            continue

        # Normalize attention map to [0,1]
        attn_obj = (attn_obj - attn_obj.min()) / (attn_obj.max() - attn_obj.min() + 1e-6)

        # Compute attention edge map via Sobel
        attn_edge = _compute_sobel_edge_map_2d(attn_obj, pipe)

        # Retrieve target edge map (already at latent resolution and roughly [0,1])
        target_edge = object_edge_maps[obj_idx].to(device=device, dtype=attn_edge.dtype)
        target_edge = target_edge / (target_edge.max() + 1e-6) if target_edge.max() > 0 else target_edge

        # Ensure shapes match (resize target to attention shape if needed)
        if target_edge.shape != attn_edge.shape:
            target_edge = F.interpolate(
                target_edge.unsqueeze(0).unsqueeze(0),
                size=(attn_edge.shape[0], attn_edge.shape[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        # Align dtype/device
        target_edge = target_edge.to(device=attn_edge.device, dtype=attn_edge.dtype)

        # Unmasked mean L1 loss between edge maps
        obj_loss = F.l1_loss(attn_edge, target_edge)

        total_loss = total_loss + obj_loss
        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    return total_loss / float(valid_count)


def EdgeGuidanceUpdateFunc(pipe, latents, timestep, timestep_idx, denoiser_args, edge_args, enabled: bool = True):
    """
    Update latents using Edge Guidance loss that matches attention edge maps to provided per-object target edge maps.
    """
    if not enabled:
        return latents.half(), None

    object_edge_maps = edge_args.get("object_edge_maps", None)
    bbox_list = edge_args.get("bbox_list", None)
    object_positions = edge_args.get("phrases_idx", None)
    if object_edge_maps is None or bbox_list is None or object_positions is None:
        return latents.half(), None

    # Clone and prepare latents
    latents = latents.clone().detach().float().requires_grad_(True)

    # Prepare latent model input (respect CFG)
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    # Process timestep for transformer
    current_timestep = pipe._timestep_process(timestep, latent_model_input.device, latent_model_input.shape[0])

    # Forward pass to get attention maps
    _, attn_maps, _ = pipe.transformer(
        latent_model_input.half(),
        encoder_hidden_states=denoiser_args["prompt_embeds"],
        encoder_attention_mask=denoiser_args["prompt_attention_mask"],
        timestep=current_timestep,
        added_cond_kwargs=denoiser_args["added_cond_kwargs"],
        return_dict=False,
        return_attn_maps=True,
        is_object_branch=False,
    )

    # Compute loss
    latent_h = edge_args["height"] // (pipe.vae_scale_factor * 2)
    latent_w = edge_args["width"] // (pipe.vae_scale_factor * 2)

    # Optionally save attention edge maps for visualization
    save_dir = edge_args.get("save_edge_maps_dir", None)
    if save_dir is not None and str(save_dir) != "":
        try:
            save_attention_edge_maps(
                attn_maps=attn_maps,
                object_positions=object_positions,
                bbox_list=bbox_list,
                latent_h=latent_h,
                latent_w=latent_w,
                pipe=pipe,
                save_dir=save_dir,
                step_idx=timestep_idx,
            )
        except Exception:
            pass

    edge_loss_val = edge_guidance_loss(
        attn_maps=attn_maps,
        object_positions=object_positions,
        object_edge_maps=object_edge_maps,
        latent_h=latent_h,
        latent_w=latent_w,
        pipe=pipe,
    )

    edge_guidance_scale = edge_args.get("edge_guidance_scale", 1.0)
    loss_scale = edge_args.get("loss_scale", 1.0)
    loss = edge_loss_val * edge_guidance_scale * loss_scale

    # Update latents based on gradient if loss is non-zero
    if loss is not None and (loss.requires_grad or (hasattr(loss, "grad_fn") and loss.grad_fn is not None)):
        grad_cond = torch.autograd.grad(outputs=loss.requires_grad_(True), inputs=latents, retain_graph=False, create_graph=False)[0]
        grad_cond = torch.nan_to_num(grad_cond)
        latents = latents - grad_cond * edge_args.get("gradient_weight", 1.0)
    elif timestep_idx < 5:
        # encourage loop to keep going early on
        loss = torch.tensor(10.0, device=latents.device, dtype=latents.dtype)

    print(f"Edge guidance loss: {edge_loss_val}")

    # Cleanup
    del attn_maps
    torch.cuda.empty_cache()

    return latents.half(), edge_loss_val