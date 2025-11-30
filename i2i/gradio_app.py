import os
import copy
from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd
import gradio as gr
import torch
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from i2i.utils import (
    load_i2i_model,
    seed_everything,
    get_phrases_idx_in_prompt,
    sanity_check,
)
from i2i.pipeline import PixArtAlphaPipeline, MainBranch
from i2i.global_update_functions import (
    GroundingUpdateFunc,
    ClipGuidanceUpdateFunc,
    TextGuidanceUpdateFunc,
    EdgeGuidanceUpdateFunc,
    _normalize_and_clip_bbox,
)
from i2i.guidance.clip.base_clip import CLIPEncoder, SigLIPEncoder


class ModelManager:
    """
    Holds a single pipeline/tokenizer on a chosen device to avoid reloading.
    """

    def __init__(self) -> None:
        self.pipe: Optional[PixArtAlphaPipeline] = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self.device: Optional[torch.device] = None

    def ensure(self, model_version: str, gpu_id: int) -> Tuple[PixArtAlphaPipeline, Any, torch.device]:
        """
        Load model if needed; return (pipe, tokenizer, device).
        """
        if model_version not in ("512", "1024"):
            model_version = "512"
        target_model_id = "PixArt-alpha/PixArt-XL-2-512x512" if model_version == "512" else "PixArt-alpha/PixArt-XL-2-1024-MS"
        device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

        if self.pipe is None or self.model_id != target_model_id or self.device != device:
            pipe, tok = load_i2i_model(target_model_id, device)
            self.pipe = pipe
            self.tokenizer = tok
            self.model_id = target_model_id
            self.device = device
        return self.pipe, self.tokenizer, self.device


mm = ModelManager()


def _df_to_layout(data: Any) -> Tuple[List[str], List[List[List[float]]]]:
    """
    Convert a Gradio Dataframe (or pandas.DataFrame) into (phrases, bbox_list).

    Expected columns / row format:
        phrase, x0, y0, x1, y1

    Returns:
        phrases:   list[str]
        bbox_list: list[list[list[float]]]  # one bbox per phrase: [[x0, y0, x1, y1]]
    """
    phrases: List[str] = []
    bbox_list: List[List[List[float]]] = []

    if data is None:
        return phrases, bbox_list

    # --- Normalize input to list-of-rows: [phrase, x0, y0, x1, y1] ---
    rows = data
    if isinstance(data, pd.DataFrame):
        # Ensure correct column order if they exist
        if all(col in data.columns for col in ["phrase", "x0", "y0", "x1", "y1"]):
            rows = data[["phrase", "x0", "y0", "x1", "y1"]].values.tolist()
        else:
            # Fall back to raw values (assume first 5 columns are in the right order)
            rows = data.values.tolist()


    # --- Parse rows ---
    for row in rows:
        if not row or len(row) < 5:
            continue

        phrase = str(row[0]).strip()
        if phrase == "":
            continue

        # Parse bbox coordinates robustly; skip row if any coord missing/invalid
        try:
            x0 = float(row[1]); y0 = float(row[2]); x1 = float(row[3]); y1 = float(row[4])
        except Exception:
            continue

        # Clip to [0, 1]
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        # Ensure (x0, y0) is top-left and (x1, y1) is bottom-right
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        phrases.append(phrase)
        bbox_list.append([[x0, y0, x1, y1]])

    return phrases, bbox_list


def _prepare_prompt_embeddings(
    pipe: PixArtAlphaPipeline,
    prompt: str,
    device: torch.device,
    do_cfg: bool,
    num_images_per_prompt: int,
    negative_prompt: str,
    clean_caption: bool,
    max_sequence_length: int,
):
    """
    Use pipeline.encode_prompt and return cat'd prompt/negative embeds & masks when CFG is used.
    """
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        clean_caption=clean_caption,
        max_sequence_length=max_sequence_length,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    return prompt_embeds, prompt_attention_mask


def _maybe_prepare_clip_image_encoders(
    pipe: PixArtAlphaPipeline, device: torch.device, ref_image_paths: Optional[List[str]]
) -> None:
    if ref_image_paths is None or len(ref_image_paths) == 0:
        return
    cached = getattr(pipe, "_clip_ref_image_paths", None)
    if hasattr(pipe, "clip_image_encoders") and (cached is not None) and tuple(ref_image_paths) == tuple(cached):
        return
    encoders: List[CLIPEncoder] = []
    for _p in ref_image_paths:
        try:
            enc = CLIPEncoder(need_ref=True, ref_path=_p).to(device)
            if hasattr(enc, "ref"):
                enc.ref = enc.ref.to(device)
            encoders.append(enc)
        except Exception:
            continue
    if len(encoders) > 0:
        pipe.clip_image_encoders = encoders
        pipe._clip_ref_image_paths = tuple(ref_image_paths)
        pipe.clip_image_encoder = pipe.clip_image_encoders[0]


def _ensure_text_encoder(pipe: PixArtAlphaPipeline, device: torch.device) -> None:
    if getattr(pipe, "clip_text_encoder", None) is None:
        pipe.clip_text_encoder = SigLIPEncoder().to(device)


def _compute_object_edge_maps_once(
    pipe: PixArtAlphaPipeline,
    device: torch.device,
    width: int,
    height: int,
    edge_images: Optional[List[Image.Image]],
    edge_preprocessor: str = "hed",
) -> Optional[List[torch.Tensor]]:
    """
    Create per-object edge maps once at attention resolution. Mirrors logic used in pipeline.__call__.
    """
    if edge_images is None or len(edge_images) == 0:
        return None

    # Latent attention resolution
    latent_h_small = height // (pipe.vae_scale_factor * 2)
    latent_w_small = width // (pipe.vae_scale_factor * 2)

    import numpy as np
    from PIL import Image as PILImage

    def _to_pil_rgb_any(obj) -> Optional[PILImage.Image]:
        if isinstance(obj, PILImage.Image):
            return obj.convert("RGB").resize((width, height))
        if isinstance(obj, str):
            try:
                im = PILImage.open(obj).convert("RGB")
                return im.resize((width, height))
            except Exception:
                return None
        return None

    def _pil_to_gray_float_2d(pimg: PILImage.Image) -> torch.Tensor:
        arr = np.array(pimg.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def _preprocess_to_soft_edges(pimg: PILImage.Image, method: str) -> Optional[torch.Tensor]:
        try:
            from controlnet_aux.processor import Processor
            proc_id = "softedge_hed" if method in ("hed", "softedge_hed") else ("canny" if method == "canny" else "softedge_hed")
            processor = Processor(proc_id)
            out = processor(pimg, to_pil=True)
            if isinstance(out, PILImage.Image):
                return _pil_to_gray_float_2d(out)
            import numpy as np 
            if isinstance(out, np.ndarray):
                arr = out.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[..., 0:3].mean(axis=-1)
                arr = arr / (255.0 if arr.max() > 1.0 else (arr.max() + 1e-6))
                return torch.from_numpy(arr)
        except Exception:
            pass
        try:
            from PIL import ImageFilter
            edge = pimg.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.5))
            return _pil_to_gray_float_2d(edge)
        except Exception:
            return None

    def _compute_sobel_edge_2d(x_2d: torch.Tensor) -> torch.Tensor:
        x = x_2d.to(device=device, dtype=pipe.smth_3.weight.dtype)[None, None, :, :]
        x = torch.nn.functional.interpolate(x, size=(latent_h_small, latent_w_small), mode="bilinear", align_corners=False)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        x = pipe.smth_3(x)
        sx = pipe.sobel_conv_x(x)
        sy = pipe.sobel_conv_y(x)
        mag = torch.sqrt(sx.pow(2) + sy.pow(2)).squeeze(0).squeeze(0)
        mag = mag / (mag.max() + 1e-6) if mag.max() > 0 else mag
        return mag

    maps: List[torch.Tensor] = []
    for obj in edge_images:
        pimg = _to_pil_rgb_any(obj)
        if pimg is None:
            return None
        if edge_preprocessor in ("hed", "softedge_hed", "canny"):
            soft = _preprocess_to_soft_edges(pimg, "hed" if edge_preprocessor in ("hed", "softedge_hed") else "canny")
            g = soft if soft is not None else _pil_to_gray_float_2d(pimg)
        else:
            g = _pil_to_gray_float_2d(pimg)
        maps.append(_compute_sobel_edge_2d(g))
    return maps


def start_session(
    prompt: str,
    steps: int,
    seed: int,
    model_version: str,
    gpu_id: int,
    guidance_scale: float,
    enable_layout: bool,
    layout_df: List[List[Any]],
    enable_edge: bool,
    edge_files: List[gr.File],
    edge_phrases_text: str,
) -> Tuple[Dict[str, Any], str, List[Image.Image], Optional[Image.Image], gr.update]:
    """
    Initialize interactive generation:
    - Load model
    - Encode prompt
    - Prepare scheduler timesteps and latents
    - Optionally set layout and edge guidance
    Returns (state, status_text, gallery, current_image)
    """
    pipe, tokenizer, device = mm.ensure(model_version=model_version, gpu_id=gpu_id)
    if model_version == "1024":
        # Strongly recommend 1024 only on large GPUs; still allow if user insists
        pass

    height = 512 if model_version == "512" else 1024
    width = height

    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    do_cfg = guidance_scale > 1.0

    # Validate steps
    steps = int(max(1, min(steps, 100)))

    # Prompt embeddings (CFG concatenated if needed)
    prompt_embeds, prompt_attention_mask = _prepare_prompt_embeddings(
        pipe=pipe,
        prompt=prompt,
        device=device,
        do_cfg=do_cfg,
        num_images_per_prompt=1,
        negative_prompt="",
        clean_caption=True,
        max_sequence_length=120,
    )

    # Set timesteps once for a base scheduler; each branch will hold its own copy
    pipe.scheduler.set_timesteps(steps, device=device)

    # Prepare latents and extra step kwargs once
    latent_channels = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=generator, eta=0.0)

    # Micro-conditions
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    if pipe.transformer.config.sample_size == 128:
        resolution = torch.tensor([height, width], dtype=prompt_embeds.dtype, device=device).unsqueeze(0)
        aspect_ratio = torch.tensor([float(height / width)], dtype=prompt_embeds.dtype, device=device).unsqueeze(0)
        if do_cfg:
            resolution = torch.cat([resolution, resolution], dim=0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    # Optional layout and/or edge phrases
    phrases: List[str] = []
    bbox_list: List[List[List[float]]] = []
    phrases_idx: Optional[List[List[int]]] = None
    if enable_layout or enable_edge:
        phrases, bbox_list = _df_to_layout(layout_df)
        # If layout is enabled, validate bboxes; otherwise we can ignore bbox validity
        if enable_layout:
            if len(phrases) != len(bbox_list):
                phrases = []
                bbox_list = []
            else:
                bbox_list = sanity_check(bbox_list, phrases)
        # Compute token indices for phrases if any phrases provided (used by edge/layout guidance)
        if len(phrases) > 0:
            try:
                phrases_idx = get_phrases_idx_in_prompt(prompt, phrases, tokenizer)
            except Exception:
                phrases_idx = None

    # Optional edge guidance (per-object), independent phrases from layout
    edge_images: Optional[List[Image.Image]] = None
    object_edge_maps = None
    edge_phrases: List[str] = []
    edge_phrases_idx: Optional[List[List[int]]] = None
    if enable_edge:
        try:
            # Parse edge phrases from textbox (comma or newline separated)
            if isinstance(edge_phrases_text, str) and edge_phrases_text.strip():
                raw_tokens = re.split(r"[\n,]+", edge_phrases_text)
                edge_phrases = [tok.strip() for tok in raw_tokens if isinstance(tok, str) and tok.strip()]
            # Load edge images
            edge_images = []
            for f in (edge_files or []):
                if f is None or getattr(f, "name", None) is None:
                    continue
                try:
                    edge_images.append(Image.open(f.name).convert("RGB"))
                except Exception:
                    pass
            # Align counts by min length to be tolerant; require at least 1 pair
            pair_count = min(len(edge_phrases), len(edge_images or []))
            if pair_count > 0:
                # Compute token indices for edge phrases
                try:
                    edge_phrases_idx = get_phrases_idx_in_prompt(prompt, edge_phrases[:pair_count], tokenizer)
                except Exception:
                    edge_phrases_idx = None
                # Compute target edge maps from provided images
                object_edge_maps = _compute_object_edge_maps_once(
                    pipe=pipe,
                    device=device,
                    width=width,
                    height=height,
                    edge_images=(edge_images[:pair_count] if edge_images is not None else None),
                    edge_preprocessor="hed",
                )
                # If either mapping or maps failed, disable edge guidance
                if edge_phrases_idx is None or object_edge_maps is None:
                    enable_edge = False
            else:
                enable_edge = False
        except Exception:
            enable_edge = False

    # Build shared state and initial branch 'B0' with its own scheduler and history
    initial_scheduler = copy.deepcopy(pipe.scheduler)
    branch_id = "B0"
    initial_branch = {
        "branch_id": branch_id,
        "i": 0,
        "latents": latents,
        "scheduler": initial_scheduler,
        "gallery": [],
        "last_preview": None,
        # per-branch guidance activations and regions
        "clip_active_from": None,
        "ref_image_paths": None,
        "clip_regions": None,
        "text_active_from": None,
        "text_list": None,
        "text_regions": None,
        # history snapshots populated below
        "history": [],
    }

    state: Dict[str, Any] = {
        "device": device,
        "pipe": pipe,
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": float(guidance_scale),
        "do_cfg": do_cfg,
        "num_steps": steps,
        "latent_channels": latent_channels,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
        "added_cond_kwargs": added_cond_kwargs,
        "extra_step_kwargs": extra_step_kwargs,
        "phrases": phrases,
        "bbox_list": bbox_list,
        "phrases_idx": phrases_idx,
        "enable_layout": bool(enable_layout and len(phrases) > 0),
        "enable_edge": bool(enable_edge and object_edge_maps is not None and edge_phrases_idx is not None),
        "object_edge_maps": object_edge_maps,
        "edge_phrases": edge_phrases,
        "edge_phrases_idx": edge_phrases_idx,
        "edge_guidance_scale": 2.0,
        "edge_preprocessor": "hed",
        "layout_intervals": [(0, min(steps - 1, 25))],
        "edge_intervals": [(0, min(steps - 1, 30))],
        "loss_scale": 10.0,
        "loss_threshold": 1e-5,
        "gradient_weight": 5.0,
        "global_update_max_iter_per_step": 3,
        "clip_guidance_scale": 5.0,
        "text_guidance_scale": 2.0,
        # Branching
        "branches": {branch_id: initial_branch},
        "active_branch_id": branch_id,
        "branch_counter": 1,
    }

    # Record initial snapshot at i=0
    initial_snapshot = {
        "i": 0,
        "latents": initial_branch["latents"].detach().clone(),
        "scheduler": copy.deepcopy(initial_branch["scheduler"]),
        "clip_active_from": None,
        "ref_image_paths": None,
        "clip_regions": None,
        "text_active_from": None,
        "text_list": None,
        "text_regions": None,
    }
    initial_branch["history"].append(initial_snapshot)

    status = "Ready. Click Next Step to begin; or choose Run To End."
    branches_update = gr.update(choices=[branch_id], value=branch_id)
    return state, status, [], None, branches_update


def apply_intervention(
    state: Dict[str, Any],
    branch_id: str,
    intervene_choice: str,
    text_input: str,
    style_file: Optional[gr.File],
    text_scale: float,
    text_x0: float,
    text_y0: float,
    text_x1: float,
    text_y1: float,
    style_scale: float,
    style_x0: float,
    style_y0: float,
    style_x1: float,
    style_y1: float,
) -> Tuple[Dict[str, Any], str]:
    """
    Apply intervention flags into state starting from current step.
    """
    pipe: PixArtAlphaPipeline = state["pipe"]
    device: torch.device = state["device"]
    branches = state.get("branches", {})
    if branch_id not in branches:
        return state, "No active branch."
    br = branches[branch_id]
    i = int(br["i"])

    def _mk_region(x0, y0, x1, y1):
        try:
            region = _normalize_and_clip_bbox([float(x0), float(y0), float(x1), float(y1)])
        except Exception:
            region = None
        if region is None:
            region = (0.0, 0.0, 1.0, 1.0)
        return [float(region[0]), float(region[1]), float(region[2]), float(region[3])]

    if intervene_choice == "Text Guidance":
        if text_input and text_input.strip():
            br["text_active_from"] = i
            br["text_list"] = [text_input.strip()]
            # scale and region
            try:
                state["text_guidance_scale"] = float(max(0.0, text_scale))
            except Exception:
                pass
            br["text_regions"] = [_mk_region(text_x0, text_y0, text_x1, text_y1)]
            _ensure_text_encoder(pipe, device)
            return state, f"[{branch_id}] Text guidance will be applied from step {i}."
        return state, "No text provided. Skipping."

    if intervene_choice == "Style Guidance":
        ref_paths: List[str] = []
        if style_file is not None and getattr(style_file, "name", None) is not None:
            ref_paths.append(style_file.name)
        if len(ref_paths) == 0:
            return state, "No reference image provided. Skipping."
        br["clip_active_from"] = i
        br["ref_image_paths"] = ref_paths
        # scale and region
        try:
            state["clip_guidance_scale"] = float(max(0.0, style_scale))
        except Exception:
            pass
        region = _mk_region(style_x0, style_y0, style_x1, style_y1)
        br["clip_regions"] = [region for _ in ref_paths]
        _maybe_prepare_clip_image_encoders(pipe, device, ref_paths)
        return state, f"[{branch_id}] Style guidance will be applied from step {i}."

    return state, "Continuing without new guidance."


def _in_intervals(step_idx: int, intervals: Optional[List[Tuple[int, int]]]) -> bool:
    if intervals is None:
        return False
    for s, e in intervals:
        if s <= step_idx <= e:
            return True
    return False


def _decode_x0_preview(pipe: PixArtAlphaPipeline, x0_pred: torch.Tensor, height: int, width: int) -> Image.Image:
    """
    Decode predicted x0 to PIL for preview.
    """
    z_mid = (x0_pred.to(dtype=pipe.vae.dtype)) / pipe.vae.config.scaling_factor
    img_mid = pipe.vae.decode(z_mid, return_dict=False)[0]
    # No binning resize here; output target size directly
    imgs = pipe.image_processor.postprocess(img_mid, output_type="pil")
    return imgs[0]


def _snapshot_branch(br: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "i": int(br["i"]),
        "latents": br["latents"].detach().clone(),
        "scheduler": copy.deepcopy(br["scheduler"]),
        "clip_active_from": br.get("clip_active_from", None),
        "ref_image_paths": br.get("ref_image_paths", None),
        "clip_regions": br.get("clip_regions", None),
        "text_active_from": br.get("text_active_from", None),
        "text_list": br.get("text_list", None),
        "text_regions": br.get("text_regions", None),
    }


def step_once(state: Dict[str, Any], branch_id: str) -> Tuple[Dict[str, Any], Optional[Image.Image], str, List[Image.Image], gr.update]:
    """
    Perform one denoising step with currently enabled guidance settings.
    Returns updated (state, preview_image, status_text, gallery).
    """
    pipe: PixArtAlphaPipeline = state["pipe"]
    device: torch.device = state["device"]
    branches = state.get("branches", {})
    if branch_id not in branches:
        return state, None, "No active branch.", [], gr.update()
    br = branches[branch_id]
    i = int(br["i"])
    num_steps = int(state["num_steps"])

    if i >= num_steps:
        return state, br.get("last_preview", None), f"[{branch_id}] Sampling already finished.", br.get("gallery", []), gr.update()

    # Swap in branch scheduler
    old_sched = pipe.scheduler
    pipe.scheduler = br["scheduler"]
    t = pipe.scheduler.timesteps[i]
    latents = br["latents"]
    height = state["height"]
    width = state["width"]

    # Denoiser args
    denoiser_args = {
        "prompt_embeds": state["prompt_embeds"],
        "prompt_attention_mask": state["prompt_attention_mask"],
        "added_cond_kwargs": state["added_cond_kwargs"],
        "do_cfg": state["do_cfg"],
        "guidance_scale": state["guidance_scale"],
    }

    # Build intervals for mid-run guidance
    clip_intervals = []
    text_intervals = []
    if br.get("clip_active_from", None) is not None:
        clip_intervals = [(int(br["clip_active_from"]), num_steps - 1)]
    if br.get("text_active_from", None) is not None:
        text_intervals = [(int(br["text_active_from"]), num_steps - 1)]

    # Args for each update
    grounding_update_args = {
        "loss_scale": state["loss_scale"],
        "gradient_weight": state["gradient_weight"],
        "bbox_list": state.get("bbox_list", None),
        "phrases_idx": state.get("phrases_idx", None),
        "height": height,
        "width": width,
    }
    clip_guidance_args = {
        "clip_guidance_scale": state["clip_guidance_scale"],
        "gradient_weight": state["gradient_weight"],
        "regions": br.get("clip_regions", None),
        "ref_image_paths": br.get("ref_image_paths", None),
    }
    edge_guidance_args = {
        "edge_guidance_scale": state["edge_guidance_scale"],
        "gradient_weight": state["gradient_weight"],
        "loss_scale": state["loss_scale"],
        "bbox_list": state.get("bbox_list", None),
        # Use edge-specific phrases mapping, independent of layout phrases
        "phrases_idx": state.get("edge_phrases_idx", None),
        "height": height,
        "width": width,
        "object_edge_maps": state.get("object_edge_maps", None),
        "save_edge_maps_dir": None,
    }
    text_guidance_args = {
        "text_guidance_scale": state["text_guidance_scale"],
        "text": " ".join(br.get("text_list", []) or []),
        "texts": br.get("text_list", None),
        "regions": br.get("text_regions", None),
    }

    # Partials via closures (direct function calls)
    def _grounding_update(lat, step_idx):
        return GroundingUpdateFunc(pipe, lat, t, step_idx, denoiser_args, grounding_update_args, enabled=True)

    def _edge_update(lat, step_idx):
        return EdgeGuidanceUpdateFunc(pipe, lat, t, step_idx, denoiser_args, edge_guidance_args, enabled=True)

    def _clip_update(lat, step_idx):
        return ClipGuidanceUpdateFunc(pipe, lat, t, step_idx, denoiser_args, clip_guidance_args, enabled=True)

    def _text_update(lat, step_idx):
        return TextGuidanceUpdateFunc(pipe, lat, t, step_idx, denoiser_args, text_guidance_args, enabled=True)

    # Apply updates as in pipeline, honoring intervals
    with torch.enable_grad():
        # Layout (grounding)
        if state.get("enable_layout", False) and _in_intervals(i, state.get("layout_intervals", [])):
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, loss_val = _grounding_update(latents, i)
                if loss_val is None or (isinstance(loss_val, (int, float)) and loss_val <= state["loss_threshold"]):
                    break
                try:
                    # handle tensor loss
                    if hasattr(loss_val, "item") and loss_val.item() <= state["loss_threshold"]:
                        break
                except Exception:
                    pass
                _iter_cnt += 1

        # Edge
        if state.get("enable_edge", False) and _in_intervals(i, state.get("edge_intervals", [])):
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, e_loss = _edge_update(latents, i)
                if e_loss is None or (hasattr(e_loss, "item") and e_loss.item() <= state["loss_threshold"]):
                    break
                _iter_cnt += 1

        # CLIP (style)
        if _in_intervals(i, clip_intervals) and br.get("ref_image_paths", None) is not None:
            # Make sure encoders are ready
            _maybe_prepare_clip_image_encoders(pipe, state["device"], br.get("ref_image_paths"))
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, _ = _clip_update(latents, i)
                _iter_cnt += 1

        # Text
        if _in_intervals(i, text_intervals) and br.get("text_list", None) is not None:
            _ensure_text_encoder(pipe, state["device"])
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, _ = _text_update(latents, i)
                _iter_cnt += 1

        # Main denoising branch (no grad)
        with torch.no_grad():
            misc_args = {
                "do_cfg": state["do_cfg"],
                "guidance_scale": state["guidance_scale"],
                "t": t,
                "latents": latents,
                "extra_step_kwargs": state["extra_step_kwargs"],
                "latent_channels": state["latent_channels"],
            }
            prompt_args = {
                "prompt_embeds": state["prompt_embeds"],
                "prompt_attention_mask": state["prompt_attention_mask"],
                "added_cond_kwargs": state["added_cond_kwargs"],
            }
            next_latents, x0_pred = MainBranch(prompt_args, misc_args, latents, pipe)
            preview = _decode_x0_preview(pipe, x0_pred, height, width)

    # Update branch state
    br["latents"] = next_latents
    br["i"] = i + 1
    br["last_preview"] = preview
    g_list = br.get("gallery", [])
    g_list = g_list + [preview]
    br["gallery"] = g_list
    # Save snapshot after step
    br["history"].append(_snapshot_branch(br))

    # Restore pipe scheduler reference
    br["scheduler"] = pipe.scheduler
    pipe.scheduler = old_sched

    status = f"[{branch_id}] Step {i + 1}/{num_steps} completed."
    return state, preview, status, g_list, gr.update()


def run_to_end(state: Dict[str, Any], branch_id: str) -> Tuple[Dict[str, Any], Optional[Image.Image], str, List[Image.Image], gr.update]:
    """
    Auto-advance to the end, returning the final preview and gallery.
    """
    branches = state.get("branches", {})
    if branch_id not in branches:
        return state, None, "No active branch.", [], gr.update()
    num_steps = int(state["num_steps"])
    while int(branches[branch_id]["i"]) < num_steps:
        state, preview, _, g, _ = step_once(state, branch_id)
    return state, branches[branch_id].get("last_preview", None), f"[{branch_id}] Finished.", branches[branch_id].get("gallery", []), gr.update()


def reset_session() -> Tuple[Dict[str, Any], str, List[Image.Image], Optional[Image.Image], gr.update]:
    return {}, "Reset.", [], None, gr.update(choices=[], value=None)


def select_branch(state: Dict[str, Any], branch_id: str) -> Tuple[Dict[str, Any], Optional[Image.Image], List[Image.Image], str]:
    branches = state.get("branches", {})
    if branch_id not in branches:
        return state, None, [], "No such branch."
    state["active_branch_id"] = branch_id
    br = branches[branch_id]
    preview = br.get("last_preview", None)
    gallery = br.get("gallery", [])
    return state, preview, gallery, f"Switched to {branch_id} (step {br['i']}/{state['num_steps']})."


def _clone_branch_from_snapshot(state: Dict[str, Any], src_branch_id: str, target_i: int) -> Tuple[str, Dict[str, Any]]:
    branches = state["branches"]
    src = branches[src_branch_id]
    # find snapshot
    snap = None
    for s in src["history"]:
        if int(s["i"]) == int(target_i):
            snap = s
            break
    if snap is None:
        if len(src["history"]) == 0:
            raise ValueError("No history to fork.")
        # Choose nearest lower snapshot
        all_is = sorted([int(s["i"]) for s in src["history"]])
        nearest = 0
        for val in all_is:
            if val <= target_i:
                nearest = val
            else:
                break
        for s in src["history"]:
            if int(s["i"]) == nearest:
                snap = s
                break
    new_id = f"B{int(state['branch_counter'])}"
    state["branch_counter"] = int(state["branch_counter"]) + 1
    new_br = {
        "branch_id": new_id,
        "i": int(snap["i"]),
        "latents": snap["latents"].detach().clone(),
        "scheduler": copy.deepcopy(snap["scheduler"]),
        "gallery": list(src.get("gallery", []))[: int(snap["i"])],
        "last_preview": (list(src.get("gallery", []))[: int(snap["i"])] or [None])[-1],
        "clip_active_from": snap.get("clip_active_from", None),
        "ref_image_paths": snap.get("ref_image_paths", None),
        "clip_regions": snap.get("clip_regions", None),
        "text_active_from": snap.get("text_active_from", None),
        "text_list": snap.get("text_list", None),
        "text_regions": snap.get("text_regions", None),
        "history": [copy.deepcopy(snap)],
    }
    return new_id, new_br


def fork_current(state: Dict[str, Any], active_branch_id: str) -> Tuple[Dict[str, Any], str, gr.update]:
    branches = state.get("branches", {})
    if active_branch_id not in branches:
        return state, "No active branch to fork.", gr.update()
    i = int(branches[active_branch_id]["i"])
    new_id, new_br = _clone_branch_from_snapshot(state, active_branch_id, i)
    branches[new_id] = new_br
    state["active_branch_id"] = new_id
    return state, f"Forked {active_branch_id} -> {new_id} at step {i}.", gr.update(choices=list(branches.keys()), value=new_id)


def fork_at_step(state: Dict[str, Any], active_branch_id: str, step_index: int) -> Tuple[Dict[str, Any], str, gr.update]:
    branches = state.get("branches", {})
    if active_branch_id not in branches:
        return state, "No active branch to fork.", gr.update()
    step_index = int(max(0, min(step_index, state["num_steps"])))
    new_id, new_br = _clone_branch_from_snapshot(state, active_branch_id, step_index)
    branches[new_id] = new_br
    state["active_branch_id"] = new_id
    return state, f"Forked {active_branch_id} -> {new_id} at step {step_index}.", gr.update(choices=list(branches.keys()), value=new_id)


def backtrack_to(state: Dict[str, Any], active_branch_id: str, step_index: int) -> Tuple[Dict[str, Any], str, List[Image.Image], Optional[Image.Image]]:
    branches = state.get("branches", {})
    if active_branch_id not in branches:
        return state, "No active branch to backtrack.", [], None
    br = branches[active_branch_id]
    step_index = int(max(0, min(step_index, state["num_steps"])))
    # find snapshot
    snap = None
    for s in br["history"]:
        if int(s["i"]) == int(step_index):
            snap = s
            break
    if snap is None:
        return state, f"No snapshot at step {step_index}.", br.get("gallery", []), br.get("last_preview", None)
    # restore
    br["i"] = int(snap["i"])
    br["latents"] = snap["latents"].detach().clone()
    br["scheduler"] = copy.deepcopy(snap["scheduler"])
    br["clip_active_from"] = snap.get("clip_active_from", None)
    br["ref_image_paths"] = snap.get("ref_image_paths", None)
    br["clip_regions"] = snap.get("clip_regions", None)
    br["text_active_from"] = snap.get("text_active_from", None)
    br["text_list"] = snap.get("text_list", None)
    br["text_regions"] = snap.get("text_regions", None)
    # trim gallery/preview
    br["gallery"] = list(br.get("gallery", []))[: int(snap["i"])]
    br["last_preview"] = (br["gallery"] or [None])[-1]
    return state, f"Backtracked {active_branch_id} to step {step_index}.", br["gallery"], br["last_preview"]

def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("### Interactive Image Generation with Mid-Run Guidance")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="a dog and a bird.")
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=5, maximum=60, step=1, value=50)
                    seed = gr.Number(label="Seed", value=67, precision=0)
                with gr.Row():
                    model_version = gr.Radio(label="Model Version", choices=["512", "1024"], value="512")
                    gpu_id = gr.Number(label="GPU ID", value=0, precision=0)
                guidance_scale = gr.Slider(label="CFG Guidance Scale", minimum=1.0, maximum=10.0, step=0.5, value=4.5)
                gr.Markdown("### Optional Layout Guidance (per-object)")
                enable_layout = gr.Checkbox(label="Enable layout (bboxes)", value=False)
                layout_df = gr.Dataframe(
                    headers=["phrase", "x0", "y0", "x1", "y1"],
                    row_count=2,
                    col_count=(5, "fixed"),
                    datatype=["str", "number", "number", "number", "number"],
                    value=[["dog", 0.23, 0.58, 0.70, 0.94], ["bird", 0.10, 0.10, 0.30, 0.30]],
                )
                gr.Markdown("### Optional Edge Guidance (per-object images + phrases)")
                enable_edge = gr.Checkbox(label="Enable edge guidance", value=False)
                edge_files = gr.Files(label="Edge images (one per object)", file_types=["image"])
                edge_phrases_text = gr.Textbox(
                    label="Edge phrases (one per line or comma-separated; aligned to uploaded images order)",
                    value="dog\nbird",
                    lines=3,
                )
                gr.Markdown("### Branching and Backtracking")
                branch_select = gr.Dropdown(label="Active Branch", choices=[], value=None)
                with gr.Row():
                    btn_fork_current = gr.Button("Fork (from current step)")
                    fork_step_idx = gr.Number(label="Fork from step", value=0, precision=0)
                    btn_fork_at = gr.Button("Fork (from step)")
                with gr.Row():
                    backtrack_step_idx = gr.Number(label="Backtrack to step", value=0, precision=0)
                    btn_backtrack = gr.Button("Backtrack")
                with gr.Row():
                    btn_start = gr.Button("Start")
                    btn_next = gr.Button("Next Step")
                with gr.Row():
                    btn_run = gr.Button("Run To End")
                    btn_reset = gr.Button("Reset")

                gr.Markdown("### Mid-Run Intervention")
                intervene_choice = gr.Radio(choices=["Continue", "Text Guidance", "Style Guidance"], value="Continue", label="Choose intervention")
                text_input = gr.Textbox(label="Text (if Text Guidance)", value="")
                style_file = gr.File(label="Reference Image (if Style Guidance)", file_types=["image"])
                with gr.Row():
                    text_scale = gr.Slider(label="Text guidance scale", minimum=0.0, maximum=15.0, step=0.1, value=2.0)
                gr.Markdown("Text guidance region [x0, y0, x1, y1] in [0,1]")
                with gr.Row():
                    text_x0 = gr.Number(label="text x0", value=0.0)
                    text_y0 = gr.Number(label="text y0", value=0.0)
                    text_x1 = gr.Number(label="text x1", value=1.0)
                    text_y1 = gr.Number(label="text y1", value=1.0)
                with gr.Row():
                    style_scale = gr.Slider(label="Style guidance scale", minimum=0.0, maximum=15.0, step=0.1, value=5.0)
                gr.Markdown("Style guidance region [x0, y0, x1, y1] in [0,1]")
                with gr.Row():
                    style_x0 = gr.Number(label="style x0", value=0.0)
                    style_y0 = gr.Number(label="style y0", value=0.0)
                    style_x1 = gr.Number(label="style x1", value=1.0)
                    style_y1 = gr.Number(label="style y1", value=1.0)
                btn_apply = gr.Button("Apply Guidance Now")

            with gr.Column(scale=1):
                status = gr.Markdown(value="Idle.")
                preview = gr.Image(label="Current Step Preview", interactive=False)
                gallery = gr.Gallery(label="Trajectory", columns=6, preview=True)

        state = gr.State({})

        # Event wiring
        btn_start.click(
            fn=start_session,
            inputs=[prompt, steps, seed, model_version, gpu_id, guidance_scale, enable_layout, layout_df, enable_edge, edge_files, edge_phrases_text],
            outputs=[state, status, gallery, preview, branch_select],
        )

        def _apply_and_echo(state_in, bsel, choice, txt, sfile, t_scale, tx0, ty0, tx1, ty1, s_scale, sx0, sy0, sx1, sy1):
            st, msg = apply_intervention(state_in, bsel, choice, txt, sfile, t_scale, tx0, ty0, tx1, ty1, s_scale, sx0, sy0, sx1, sy1)
            return st, gr.update(value=f"{msg}")

        btn_apply.click(
            fn=_apply_and_echo,
            inputs=[state, branch_select, intervene_choice, text_input, style_file, text_scale, text_x0, text_y0, text_x1, text_y1, style_scale, style_x0, style_y0, style_x1, style_y1],
            outputs=[state, status],
        )

        btn_next.click(fn=step_once, inputs=[state, branch_select], outputs=[state, preview, status, gallery, branch_select])
        btn_run.click(fn=run_to_end, inputs=[state, branch_select], outputs=[state, preview, status, gallery, branch_select])
        btn_reset.click(fn=reset_session, inputs=None, outputs=[state, status, gallery, preview, branch_select])

        branch_select.change(fn=select_branch, inputs=[state, branch_select], outputs=[state, preview, gallery, status])
        btn_fork_current.click(fn=fork_current, inputs=[state, branch_select], outputs=[state, status, branch_select])
        btn_fork_at.click(fn=fork_at_step, inputs=[state, branch_select, fork_step_idx], outputs=[state, status, branch_select])
        btn_backtrack.click(fn=backtrack_to, inputs=[state, branch_select, backtrack_step_idx], outputs=[state, status, gallery, preview])

    return demo


if __name__ == "__main__":
    # Allow host/port overrides via env
    host = os.environ.get("GRADIO_HOST", "0.0.0.0")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    app = build_ui()
    app.queue().launch(server_name=host, server_port=port)


