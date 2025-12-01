import os
import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from i2i.utils import (  # type: ignore
    load_i2i_model,
    seed_everything,
    get_phrases_idx_in_prompt,
    sanity_check,
)
from i2i.pipeline import PixArtAlphaPipeline, MainBranch  # type: ignore
from i2i.global_update_functions import (  # type: ignore
    GroundingUpdateFunc,
    ClipGuidanceUpdateFunc,
    TextGuidanceUpdateFunc,
    EdgeGuidanceUpdateFunc,
    _normalize_and_clip_bbox,
)
from i2i.guidance.clip.base_clip import CLIPEncoder, SigLIPEncoder  # type: ignore


class ModelManager:
    """
    Singleton-like holder for pipeline and tokenizer on a chosen device.
    Avoids re-loading between requests.
    """
    def __init__(self) -> None:
        self.pipe: Optional[PixArtAlphaPipeline] = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self.device: Optional[torch.device] = None

    def ensure(self, model_version: str, gpu_id: int) -> Tuple[PixArtAlphaPipeline, Any, torch.device]:
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
    Create per-object edge maps once at attention resolution. Mirrors logic used in gradio_app.
    """
    if edge_images is None or len(edge_images) == 0:
        return None

    latent_h_small = height // (pipe.vae_scale_factor * 2)
    latent_w_small = width // (pipe.vae_scale_factor * 2)

    import numpy as np
    from PIL import Image as PILImage

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
        pimg = obj.convert("RGB").resize((width, height))
        if edge_preprocessor in ("hed", "softedge_hed", "canny"):
            soft = _preprocess_to_soft_edges(pimg, "hed" if edge_preprocessor in ("hed", "softedge_hed") else "canny")
            g = soft if soft is not None else _pil_to_gray_float_2d(pimg)
        else:
            g = _pil_to_gray_float_2d(pimg)
        maps.append(_compute_sobel_edge_2d(g))
    return maps


def _decode_x0_preview(pipe: PixArtAlphaPipeline, x0_pred: torch.Tensor) -> Image.Image:
    z_mid = (x0_pred.to(dtype=pipe.vae.dtype)) / pipe.vae.config.scaling_factor
    img_mid = pipe.vae.decode(z_mid, return_dict=False)[0]
    imgs = pipe.image_processor.postprocess(img_mid, output_type="pil")
    return imgs[0]


def _setup_extended_attention_kwargs(
    br: Dict[str, Any],
    step_index: int,
    num_steps: int,
) -> Optional[Dict[str, Any]]:
    """
    Set up cross_attention_kwargs for extended attention if this is a merged branch.
    Returns None if extended attention should not be applied.
    
    The extended attention concatenates K/V from source latents to enable
    information flow between the merged branches during self-attention.
    """
    merge_info = br.get("merge_source_latents")
    if merge_info is None:
        return None
    
    if not merge_info.get("extended_attention_enabled", False):
        return None
    
    merge_step = merge_info.get("merge_step", 0)
    
    # Apply extended attention for a window after the merge
    # Typically most effective in early-to-mid steps
    extended_window = min(num_steps - merge_step, 30)  # Apply for up to 30 steps after merge
    
    if step_index < merge_step or step_index > merge_step + extended_window:
        return None
    
    return {
        "extended_attention": True,
        "extended_scale": merge_info.get("extended_scale", 1.0),
        "extended_steps": (merge_step, merge_step + extended_window),
        "step_index": step_index,
        "extended_mode": "pair",  # Use pairwise mode from GrounDiT
        # Source latents for K/V extension
        "source_latents_1": merge_info.get("latents_1"),
        "source_latents_2": merge_info.get("latents_2"),
    }


def _snapshot_branch(br: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "i": int(br["i"]),
        "latents": br["latents"].detach().clone(),
        "scheduler": copy.deepcopy(br["scheduler"]),
        "clip_active_from": br.get("clip_active_from", None),
        "ref_image_paths": copy.deepcopy(br.get("ref_image_paths", None)),
        "clip_regions": copy.deepcopy(br.get("clip_regions", None)),
        "clip_guidance_scale": br.get("clip_guidance_scale", 5.0),
        "text_active_from": br.get("text_active_from", None),
        "text_list": copy.deepcopy(br.get("text_list", None)),
        "text_regions": copy.deepcopy(br.get("text_regions", None)),
        "text_guidance_scale": br.get("text_guidance_scale", 2.0),
    }


def start_session_engine(
    prompt: str,
    steps: int,
    seed: int,
    model_version: str,
    gpu_id: int,
    guidance_scale: float,
    enable_layout: bool,
    layout_items: Optional[List[Dict[str, Union[str, float]]]],  # [{phrase,x0,y0,x1,y1}]
    enable_edge: bool,
    edge_images: Optional[List[Image.Image]],
    edge_phrases_text: Optional[str],
) -> Tuple[Dict[str, Any], str]:
    pipe, tokenizer, device = mm.ensure(model_version=model_version, gpu_id=gpu_id)

    height = 512 if model_version == "512" else 1024
    width = height

    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    do_cfg = guidance_scale > 1.0
    steps = int(max(1, min(steps, 100)))

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

    pipe.scheduler.set_timesteps(steps, device=device)

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

    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    if pipe.transformer.config.sample_size == 128:
        resolution = torch.tensor([height, width], dtype=prompt_embeds.dtype, device=device).unsqueeze(0)
        aspect_ratio = torch.tensor([float(height / width)], dtype=prompt_embeds.dtype, device=device).unsqueeze(0)
        if do_cfg:
            resolution = torch.cat([resolution, resolution], dim=0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    # Layout
    phrases: List[str] = []
    bbox_list: List[List[List[float]]] = []
    phrases_idx: Optional[List[List[int]]] = None
    if enable_layout and layout_items:
        for row in layout_items:
            try:
                phrase = str(row.get("phrase", "")).strip()
                x0 = float(row.get("x0", 0.0))
                y0 = float(row.get("y0", 0.0))
                x1 = float(row.get("x1", 1.0))
                y1 = float(row.get("y1", 1.0))
            except Exception:
                continue
            if phrase == "":
                continue
            # normalize box
            x0 = max(0.0, min(1.0, x0))
            y0 = max(0.0, min(1.0, y0))
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            phrases.append(phrase)
            bbox_list.append([[x0, y0, x1, y1]])
        if len(phrases) == len(bbox_list) and len(phrases) > 0:
            bbox_list = sanity_check(bbox_list, phrases)  # type: ignore
            try:
                phrases_idx = get_phrases_idx_in_prompt(prompt, phrases, tokenizer)  # type: ignore
            except Exception:
                phrases_idx = None
        else:
            phrases = []
            bbox_list = []

    # Edge
    edge_phrases: List[str] = []
    edge_phrases_idx: Optional[List[List[int]]] = None
    object_edge_maps = None
    enable_edge_final = bool(enable_edge)
    if enable_edge and edge_images:
        if isinstance(edge_phrases_text, str) and edge_phrases_text.strip():
            import re as _re
            tokens = _re.split(r"[\n,]+", edge_phrases_text)
            edge_phrases = [t.strip() for t in tokens if isinstance(t, str) and t.strip()]
        pair_count = min(len(edge_phrases), len(edge_images))
        if pair_count > 0:
            try:
                edge_phrases_idx = get_phrases_idx_in_prompt(prompt, edge_phrases[:pair_count], tokenizer)  # type: ignore
            except Exception:
                edge_phrases_idx = None
            object_edge_maps = _compute_object_edge_maps_once(
                pipe=pipe,
                device=device,
                width=width,
                height=height,
                edge_images=edge_images[:pair_count],
                edge_preprocessor="hed",
            )
            if edge_phrases_idx is None or object_edge_maps is None:
                enable_edge_final = False
        else:
            enable_edge_final = False

    initial_scheduler = copy.deepcopy(pipe.scheduler)
    branch_id = "B0"
    initial_branch = {
        "branch_id": branch_id,
        "i": 0,
        "latents": latents,
        "scheduler": initial_scheduler,
        "gallery": [],
        "last_preview": None,
        "clip_active_from": None,
        "ref_image_paths": None,
        "clip_regions": None,
        "clip_guidance_scale": 5.0,  # per-branch guidance scale
        "text_active_from": None,
        "text_list": None,
        "text_regions": None,
        "text_guidance_scale": 2.0,  # per-branch guidance scale
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
        "enable_edge": bool(enable_edge_final and object_edge_maps is not None and edge_phrases_idx is not None),
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
        # Default guidance scales (used when branch doesn't have its own)
        "default_clip_guidance_scale": 5.0,
        "default_text_guidance_scale": 2.0,
        "branches": {branch_id: initial_branch},
        "active_branch_id": branch_id,
        "branch_counter": 1,
    }
    # initial snapshot
    initial_snapshot = {
        "i": 0,
        "latents": initial_branch["latents"].detach().clone(),
        "scheduler": copy.deepcopy(initial_branch["scheduler"]),
        "clip_active_from": None,
        "ref_image_paths": None,
        "clip_regions": None,
        "clip_guidance_scale": 5.0,
        "text_active_from": None,
        "text_list": None,
        "text_regions": None,
        "text_guidance_scale": 2.0,
    }
    initial_branch["history"].append(initial_snapshot)
    status = "Ready. Call step to begin; or run_to_end."
    return state, status


def apply_intervention_engine(
    state: Dict[str, Any],
    branch_id: str,
    intervene_choice: str,
    text_input: Optional[str],
    style_file_path: Optional[str],
    text_scale: Optional[float],
    text_region: Optional[Tuple[float, float, float, float]],
    style_scale: Optional[float],
    style_region: Optional[Tuple[float, float, float, float]],
) -> Tuple[Dict[str, Any], str]:
    pipe: PixArtAlphaPipeline = state["pipe"]
    device: torch.device = state["device"]
    branches = state.get("branches", {})
    if branch_id not in branches:
        return state, "No active branch."
    br = branches[branch_id]
    i = int(br["i"])

    def _mk_region_tuple(r: Optional[Tuple[float, float, float, float]]):
        if r is None:
            return [0.0, 0.0, 1.0, 1.0]
        region = _normalize_and_clip_bbox([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        if region is None:
            region = (0.0, 0.0, 1.0, 1.0)
        return [float(region[0]), float(region[1]), float(region[2]), float(region[3])]

    if intervene_choice == "Text Guidance":
        if text_input and text_input.strip():
            br["text_active_from"] = i
            br["text_list"] = [text_input.strip()]
            if isinstance(text_scale, (int, float)):
                br["text_guidance_scale"] = float(max(0.0, text_scale))
            br["text_regions"] = [_mk_region_tuple(text_region)]
            _ensure_text_encoder(pipe, device)
            print(f"[{branch_id}] Text guidance will be applied from step {i}.")
            return state, f"[{branch_id}] Text guidance will be applied from step {i}."
        return state, "No text provided. Skipping."

    if intervene_choice == "Style Guidance":
        ref_paths: List[str] = []
        if style_file_path is not None and isinstance(style_file_path, str) and len(style_file_path) > 0:
            ref_paths.append(style_file_path)
        if len(ref_paths) == 0:
            return state, "No reference image provided. Skipping."
        br["clip_active_from"] = i
        br["ref_image_paths"] = ref_paths
        if isinstance(style_scale, (int, float)):
            br["clip_guidance_scale"] = float(max(0.0, style_scale))
        br["clip_regions"] = [_mk_region_tuple(style_region) for _ in ref_paths]
        _maybe_prepare_clip_image_encoders(pipe, device, ref_paths)
        print(f"[{branch_id}] Style guidance will be applied from step {i}.")
        return state, f"[{branch_id}] Style guidance will be applied from step {i}."

    return state, "Continuing without new guidance."


def step_once_engine(state: Dict[str, Any], branch_id: str) -> Tuple[Dict[str, Any], Optional[Image.Image], str, List[Image.Image]]:
    pipe: PixArtAlphaPipeline = state["pipe"]
    device: torch.device = state["device"]
    branches = state.get("branches", {})
    if branch_id not in branches:
        return state, None, "No active branch.", []
    br = branches[branch_id]
    i = int(br["i"])
    num_steps = int(state["num_steps"])
    
    # Debug logging
    print(f"[step_once_engine] branch_id={branch_id}, step={i}, num_steps={num_steps}")
    print(f"[step_once_engine] text_active_from={br.get('text_active_from')}, text_list={br.get('text_list')}")
    print(f"[step_once_engine] clip_active_from={br.get('clip_active_from')}, ref_image_paths={br.get('ref_image_paths')}")
    
    if i >= num_steps:
        return state, br.get("last_preview", None), f"[{branch_id}] Sampling already finished.", br.get("gallery", [])

    old_sched = pipe.scheduler
    pipe.scheduler = br["scheduler"]
    t = pipe.scheduler.timesteps[i]
    latents = br["latents"]
    height = state["height"]
    width = state["width"]

    denoiser_args = {
        "prompt_embeds": state["prompt_embeds"],
        "prompt_attention_mask": state["prompt_attention_mask"],
        "added_cond_kwargs": state["added_cond_kwargs"],
        "do_cfg": state["do_cfg"],
        "guidance_scale": state["guidance_scale"],
    }

    def _in_intervals(step_idx: int, intervals: Optional[List[Tuple[int, int]]]) -> bool:
        if intervals is None:
            return False
        for s, e in intervals:
            if s <= step_idx <= e:
                return True
        return False

    clip_intervals = []
    text_intervals = []
    if br.get("clip_active_from", None) is not None:
        clip_intervals = [(int(br["clip_active_from"]), num_steps - 1)]
    if br.get("text_active_from", None) is not None:
        text_intervals = [(int(br["text_active_from"]), num_steps - 1)]
    
    print(f"[step_once_engine] clip_intervals={clip_intervals}, text_intervals={text_intervals}")

    grounding_update_args = {
        "loss_scale": state["loss_scale"],
        "gradient_weight": state["gradient_weight"],
        "bbox_list": state.get("bbox_list", None),
        "phrases_idx": state.get("phrases_idx", None),
        "height": height,
        "width": width,
    }
    clip_guidance_args = {
        "clip_guidance_scale": br.get("clip_guidance_scale", state.get("default_clip_guidance_scale", 5.0)),
        "gradient_weight": state["gradient_weight"],
        "regions": br.get("clip_regions", None),
        "ref_image_paths": br.get("ref_image_paths", None),
    }
    edge_guidance_args = {
        "edge_guidance_scale": state["edge_guidance_scale"],
        "gradient_weight": state["gradient_weight"],
        "loss_scale": state["loss_scale"],
        "bbox_list": state.get("bbox_list", None),
        "phrases_idx": state.get("edge_phrases_idx", None),
        "height": height,
        "width": width,
        "object_edge_maps": state.get("object_edge_maps", None),
        "save_edge_maps_dir": None,
    }
    text_guidance_args = {
        "text_guidance_scale": br.get("text_guidance_scale", state.get("default_text_guidance_scale", 2.0)),
        "text": " ".join(br.get("text_list", []) or []),
        "texts": br.get("text_list", None),
        "regions": br.get("text_regions", None),
    }

    def _grounding_update(lat, step_idx):
        return GroundingUpdateFunc(pipe, lat, t, step_idx, denoiser_args, grounding_update_args, enabled=True)

    def _edge_update(lat, step_idx):
        return EdgeGuidanceUpdateFunc(pipe, lat, t, step_idx, denoiser_args, edge_guidance_args, enabled=True)

    def _clip_update(lat, step_idx):
        return ClipGuidanceUpdateFunc(pipe, lat, t, step_idx, denoiser_args, clip_guidance_args, enabled=True)

    def _text_update(lat, step_idx):
        return TextGuidanceUpdateFunc(pipe, lat, t, step_idx, denoiser_args, text_guidance_args, enabled=True)

    with torch.enable_grad():
        if state.get("enable_layout", False) and _in_intervals(i, state.get("layout_intervals", [])):
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, loss_val = _grounding_update(latents, i)
                try:
                    if loss_val is None or (hasattr(loss_val, "item") and loss_val.item() <= state["loss_threshold"]):
                        break
                except Exception:
                    pass
                _iter_cnt += 1

        if state.get("enable_edge", False) and _in_intervals(i, state.get("edge_intervals", [])):
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, e_loss = _edge_update(latents, i)
                if e_loss is None or (hasattr(e_loss, "item") and e_loss.item() <= state["loss_threshold"]):
                    break
                _iter_cnt += 1

        if _in_intervals(i, clip_intervals) and br.get("ref_image_paths", None) is not None:
            print(f"[step_once_engine] Applying CLIP/style guidance at step {i}")
            _maybe_prepare_clip_image_encoders(pipe, state["device"], br.get("ref_image_paths"))
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, _ = _clip_update(latents, i)
                _iter_cnt += 1
        else:
            print(f"[step_once_engine] NOT applying clip guidance: in_intervals={_in_intervals(i, clip_intervals)}, ref_image_paths={br.get('ref_image_paths')}")

        if _in_intervals(i, text_intervals) and br.get("text_list", None) is not None:
            print(f"[step_once_engine] Applying TEXT guidance at step {i}")
            _ensure_text_encoder(pipe, state["device"])
            _iter_cnt = 0
            while _iter_cnt < int(state["global_update_max_iter_per_step"]):
                latents, _ = _text_update(latents, i)
                _iter_cnt += 1
        else:
            print(f"[step_once_engine] NOT applying text guidance: in_intervals={_in_intervals(i, text_intervals)}, text_list={br.get('text_list')}")

        with torch.no_grad():
            # Check if this is a merged branch that needs extended attention
            extended_kwargs = _setup_extended_attention_kwargs(br, i, num_steps)
            
            misc_args = {
                "do_cfg": state["do_cfg"],
                "guidance_scale": state["guidance_scale"],
                "t": t,
                "latents": latents,
                "extra_step_kwargs": state["extra_step_kwargs"],
                "latent_channels": state["latent_channels"],
            }
            
            # Add cross_attention_kwargs for extended attention if applicable
            if extended_kwargs is not None:
                print(f"[step_once_engine] Applying extended attention for merged branch at step {i}")
                misc_args["cross_attention_kwargs"] = extended_kwargs
            
            prompt_args = {
                "prompt_embeds": state["prompt_embeds"],
                "prompt_attention_mask": state["prompt_attention_mask"],
                "added_cond_kwargs": state["added_cond_kwargs"],
            }
            next_latents, x0_pred = MainBranch(prompt_args, misc_args, latents, pipe)
            preview = _decode_x0_preview(pipe, x0_pred)

    br["latents"] = next_latents
    br["i"] = i + 1
    br["last_preview"] = preview
    g_list = br.get("gallery", [])
    g_list = g_list + [preview]
    br["gallery"] = g_list
    br["history"].append(_snapshot_branch(br))
    br["scheduler"] = pipe.scheduler
    pipe.scheduler = old_sched
    status = f"[{branch_id}] Step {i + 1}/{num_steps} completed."
    return state, preview, status, g_list


def run_to_end_engine(state: Dict[str, Any], branch_id: str) -> Tuple[Dict[str, Any], Optional[Image.Image], str, List[Image.Image]]:
    branches = state.get("branches", {})
    if branch_id not in branches:
        return state, None, "No active branch.", []
    num_steps = int(state["num_steps"])
    while int(branches[branch_id]["i"]) < num_steps:
        state, _, _, _ = step_once_engine(state, branch_id)
    return state, branches[branch_id].get("last_preview", None), f"[{branch_id}] Finished.", branches[branch_id].get("gallery", [])


def select_branch_engine(state: Dict[str, Any], branch_id: str) -> Tuple[Dict[str, Any], Optional[Image.Image], List[Image.Image], str]:
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
    snap = None
    for s in src["history"]:
        if int(s["i"]) == int(target_i):
            snap = s
            break
    if snap is None:
        if len(src["history"]) == 0:
            raise ValueError("No history to fork.")
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
    
    print(f"[_clone_branch_from_snapshot] src_branch_id={src_branch_id}, target_i={target_i}")
    print(f"[_clone_branch_from_snapshot] snap found: i={snap['i']}, text_active_from={snap.get('text_active_from')}, text_list={snap.get('text_list')}")
    print(f"[_clone_branch_from_snapshot] snap found: clip_active_from={snap.get('clip_active_from')}, ref_image_paths={snap.get('ref_image_paths')}")
    
    # When forking at the current step of the source branch, use the source branch's 
    # current guidance settings instead of the snapshot's, because guidance may have been
    # applied after the snapshot was taken but before forking.
    use_current_guidance = (int(src["i"]) == int(snap["i"]))
    
    if use_current_guidance:
        print(f"[_clone_branch_from_snapshot] Using CURRENT branch guidance (fork at current step)")
        guidance_source = src
    else:
        print(f"[_clone_branch_from_snapshot] Using SNAPSHOT guidance (fork at earlier step)")
        guidance_source = snap
    
    new_id = f"B{int(state['branch_counter'])}"
    state["branch_counter"] = int(state["branch_counter"]) + 1
    new_br = {
        "branch_id": new_id,
        "i": int(snap["i"]),
        "latents": snap["latents"].detach().clone(),
        "scheduler": copy.deepcopy(snap["scheduler"]),
        "gallery": list(src.get("gallery", []))[: int(snap["i"])],
        "last_preview": (list(src.get("gallery", []))[: int(snap["i"])] or [None])[-1],
        "clip_active_from": guidance_source.get("clip_active_from", None),
        "ref_image_paths": copy.deepcopy(guidance_source.get("ref_image_paths", None)),
        "clip_regions": copy.deepcopy(guidance_source.get("clip_regions", None)),
        "clip_guidance_scale": guidance_source.get("clip_guidance_scale", 5.0),
        "text_active_from": guidance_source.get("text_active_from", None),
        "text_list": copy.deepcopy(guidance_source.get("text_list", None)),
        "text_regions": copy.deepcopy(guidance_source.get("text_regions", None)),
        "text_guidance_scale": guidance_source.get("text_guidance_scale", 2.0),
        "history": [copy.deepcopy(snap)],
    }
    
    # Update the history snapshot with the current guidance settings if we're forking at current step
    if use_current_guidance and len(new_br["history"]) > 0:
        new_br["history"][0]["clip_active_from"] = guidance_source.get("clip_active_from", None)
        new_br["history"][0]["ref_image_paths"] = copy.deepcopy(guidance_source.get("ref_image_paths", None))
        new_br["history"][0]["clip_regions"] = copy.deepcopy(guidance_source.get("clip_regions", None))
        new_br["history"][0]["clip_guidance_scale"] = guidance_source.get("clip_guidance_scale", 5.0)
        new_br["history"][0]["text_active_from"] = guidance_source.get("text_active_from", None)
        new_br["history"][0]["text_list"] = copy.deepcopy(guidance_source.get("text_list", None))
        new_br["history"][0]["text_regions"] = copy.deepcopy(guidance_source.get("text_regions", None))
        new_br["history"][0]["text_guidance_scale"] = guidance_source.get("text_guidance_scale", 2.0)
    
    print(f"[_clone_branch_from_snapshot] new_br created: id={new_id}, i={new_br['i']}, text_active_from={new_br['text_active_from']}, text_list={new_br['text_list']}")
    
    return new_id, new_br


def fork_current_engine(state: Dict[str, Any], active_branch_id: str) -> Tuple[Dict[str, Any], str]:
    branches = state.get("branches", {})
    if active_branch_id not in branches:
        return state, "No active branch to fork."
    i = int(branches[active_branch_id]["i"])
    new_id, new_br = _clone_branch_from_snapshot(state, active_branch_id, i)
    branches[new_id] = new_br
    state["active_branch_id"] = new_id
    return state, f"Forked {active_branch_id} -> {new_id} at step {i}."


def fork_at_step_engine(state: Dict[str, Any], active_branch_id: str, step_index: int) -> Tuple[Dict[str, Any], str]:
    branches = state.get("branches", {})
    if active_branch_id not in branches:
        return state, "No active branch to fork."
    step_index = int(max(0, min(step_index, state["num_steps"])))
    new_id, new_br = _clone_branch_from_snapshot(state, active_branch_id, step_index)
    branches[new_id] = new_br
    state["active_branch_id"] = new_id
    return state, f"Forked {active_branch_id} -> {new_id} at step {step_index}."


def backtrack_to_engine(state: Dict[str, Any], active_branch_id: str, step_index: int) -> Tuple[Dict[str, Any], str, List[Image.Image], Optional[Image.Image]]:
    branches = state.get("branches", {})
    if active_branch_id not in branches:
        return state, "No active branch to backtrack.", [], None
    br = branches[active_branch_id]
    step_index = int(max(0, min(step_index, state["num_steps"])))
    snap = None
    for s in br["history"]:
        if int(s["i"]) == int(step_index):
            snap = s
            break
    if snap is None:
        return state, f"No snapshot at step {step_index}.", br.get("gallery", []), br.get("last_preview", None)
    br["i"] = int(snap["i"])
    br["latents"] = snap["latents"].detach().clone()
    br["scheduler"] = copy.deepcopy(snap["scheduler"])
    br["clip_active_from"] = snap.get("clip_active_from", None)
    br["ref_image_paths"] = copy.deepcopy(snap.get("ref_image_paths", None))
    br["clip_regions"] = copy.deepcopy(snap.get("clip_regions", None))
    br["clip_guidance_scale"] = snap.get("clip_guidance_scale", 5.0)
    br["text_active_from"] = snap.get("text_active_from", None)
    br["text_list"] = copy.deepcopy(snap.get("text_list", None))
    br["text_regions"] = copy.deepcopy(snap.get("text_regions", None))
    br["text_guidance_scale"] = snap.get("text_guidance_scale", 2.0)
    br["gallery"] = list(br.get("gallery", []))[: int(snap["i"])]
    br["last_preview"] = (br["gallery"] or [None])[-1]
    return state, f"Backtracked {active_branch_id} to step {step_index}.", br["gallery"], br["last_preview"]


def _merge_guidance_settings(br1: Dict[str, Any], br2: Dict[str, Any], step_index: int) -> Dict[str, Any]:
    """
    Merge guidance settings from two branches.
    Combines text lists, style refs, and regions from both branches.
    """
    merged = {}
    
    # Merge text guidance
    text_list_1 = br1.get("text_list") or []
    text_list_2 = br2.get("text_list") or []
    text_regions_1 = br1.get("text_regions") or []
    text_regions_2 = br2.get("text_regions") or []
    
    merged_text_list = list(text_list_1) + [t for t in text_list_2 if t not in text_list_1]
    merged_text_regions = list(text_regions_1) + list(text_regions_2)[:len(text_list_2)]
    
    if len(merged_text_list) > 0:
        merged["text_active_from"] = step_index
        merged["text_list"] = merged_text_list
        merged["text_regions"] = merged_text_regions[:len(merged_text_list)]
        # Use max of the two scales
        merged["text_guidance_scale"] = max(
            br1.get("text_guidance_scale", 2.0),
            br2.get("text_guidance_scale", 2.0)
        )
    else:
        merged["text_active_from"] = None
        merged["text_list"] = None
        merged["text_regions"] = None
        merged["text_guidance_scale"] = 2.0
    
    # Merge style/CLIP guidance
    ref_paths_1 = br1.get("ref_image_paths") or []
    ref_paths_2 = br2.get("ref_image_paths") or []
    clip_regions_1 = br1.get("clip_regions") or []
    clip_regions_2 = br2.get("clip_regions") or []
    
    merged_ref_paths = list(ref_paths_1) + [p for p in ref_paths_2 if p not in ref_paths_1]
    merged_clip_regions = list(clip_regions_1) + list(clip_regions_2)[:len(ref_paths_2)]
    
    if len(merged_ref_paths) > 0:
        merged["clip_active_from"] = step_index
        merged["ref_image_paths"] = merged_ref_paths
        merged["clip_regions"] = merged_clip_regions[:len(merged_ref_paths)]
        merged["clip_guidance_scale"] = max(
            br1.get("clip_guidance_scale", 5.0),
            br2.get("clip_guidance_scale", 5.0)
        )
    else:
        merged["clip_active_from"] = None
        merged["ref_image_paths"] = None
        merged["clip_regions"] = None
        merged["clip_guidance_scale"] = 5.0
    
    return merged


def merge_branches_engine(
    state: Dict[str, Any],
    branch_id_1: str,
    branch_id_2: str,
    step_index_1: int,
    step_index_2: Optional[int] = None,  # If None, uses step_index_1 for both
    merge_weight: float = 0.5,  # Weight for branch_1's latent (0.5 = equal blend)
) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """
    Merge two branches, allowing different steps for each branch.
    
    1. Takes latent from branch_1 at step_index_1
    2. Takes latent from branch_2 at step_index_2 (or step_index_1 if not specified)
    3. Creates a weighted average of the latents
    4. Starts the new branch from the later of the two steps
    5. Stores both source latents for extended attention during denoising
    
    Args:
        state: Session state
        branch_id_1: First branch to merge
        branch_id_2: Second branch to merge  
        step_index_1: The timestep to use from branch_1
        step_index_2: The timestep to use from branch_2 (defaults to step_index_1)
        merge_weight: Weight for branch_1's latent in the blend (default 0.5)
    
    Returns:
        (state, status_message, new_branch_id or None if failed)
    """
    branches = state.get("branches", {})
    
    if branch_id_1 not in branches:
        return state, f"Branch {branch_id_1} not found.", None
    if branch_id_2 not in branches:
        return state, f"Branch {branch_id_2} not found.", None
    if branch_id_1 == branch_id_2:
        return state, "Cannot merge a branch with itself.", None
    
    br1 = branches[branch_id_1]
    br2 = branches[branch_id_2]
    
    # Default step_index_2 to step_index_1 if not provided
    if step_index_2 is None:
        step_index_2 = step_index_1
    
    step_index_1 = int(max(0, min(step_index_1, state["num_steps"])))
    step_index_2 = int(max(0, min(step_index_2, state["num_steps"])))
    
    # Find snapshot for branch 1 at step_index_1
    snap1 = None
    for s in br1["history"]:
        if int(s["i"]) == step_index_1:
            snap1 = s
            break
    
    # If exact step not found, use nearest available step <= target
    if snap1 is None:
        all_is = sorted([int(s["i"]) for s in br1["history"]])
        nearest = 0
        for val in all_is:
            if val <= step_index_1:
                nearest = val
            else:
                break
        for s in br1["history"]:
            if int(s["i"]) == nearest:
                snap1 = s
                break
    
    # Find snapshot for branch 2 at step_index_2
    snap2 = None
    for s in br2["history"]:
        if int(s["i"]) == step_index_2:
            snap2 = s
            break
    
    # If exact step not found, use nearest available step <= target
    if snap2 is None:
        all_is = sorted([int(s["i"]) for s in br2["history"]])
        nearest = 0
        for val in all_is:
            if val <= step_index_2:
                nearest = val
            else:
                break
        for s in br2["history"]:
            if int(s["i"]) == nearest:
                snap2 = s
                break
    
    if snap1 is None:
        return state, f"Could not find valid snapshot for {branch_id_1} at or before step {step_index_1}.", None
    if snap2 is None:
        return state, f"Could not find valid snapshot for {branch_id_2} at or before step {step_index_2}.", None
    
    actual_step_1 = int(snap1["i"])
    actual_step_2 = int(snap2["i"])
    
    latents1 = snap1["latents"].detach().clone()
    latents2 = snap2["latents"].detach().clone()
    
    # Create merged latent as weighted average for initial state
    merge_weight = float(max(0.0, min(1.0, merge_weight)))
    merged_latents = merge_weight * latents1 + (1.0 - merge_weight) * latents2
    
    # The merged branch starts from the later of the two steps
    # This ensures we continue from a valid point in the denoising process
    merge_start_step = max(actual_step_1, actual_step_2)
    
    # Merge guidance settings using the later step
    merged_guidance = _merge_guidance_settings(br1, br2, merge_start_step)
    
    # Create new branch
    new_id = f"B{int(state['branch_counter'])}"
    state["branch_counter"] = int(state["branch_counter"]) + 1
    
    # Use scheduler from the branch at the later step
    scheduler_source = snap1 if actual_step_1 >= actual_step_2 else snap2
    
    new_br = {
        "branch_id": new_id,
        "i": merge_start_step,
        "latents": merged_latents,
        "scheduler": copy.deepcopy(scheduler_source["scheduler"]),
        "gallery": [],  # Start fresh gallery for merged branch
        "last_preview": None,
        # Merged guidance settings
        "clip_active_from": merged_guidance["clip_active_from"],
        "ref_image_paths": merged_guidance["ref_image_paths"],
        "clip_regions": merged_guidance["clip_regions"],
        "clip_guidance_scale": merged_guidance["clip_guidance_scale"],
        "text_active_from": merged_guidance["text_active_from"],
        "text_list": merged_guidance["text_list"],
        "text_regions": merged_guidance["text_regions"],
        "text_guidance_scale": merged_guidance["text_guidance_scale"],
        "history": [],
        # Extended attention: store source latents for K/V concatenation during denoising
        "merge_source_latents": {
            "latents_1": latents1,
            "latents_2": latents2,
            "branch_id_1": branch_id_1,
            "branch_id_2": branch_id_2,
            "step_1": actual_step_1,
            "step_2": actual_step_2,
            "merge_start_step": merge_start_step,
            "extended_attention_enabled": True,
            "extended_scale": 1.0,  # Scale factor for extended attention
        },
    }
    
    # Create initial snapshot
    initial_snap = {
        "i": merge_start_step,
        "latents": merged_latents.detach().clone(),
        "scheduler": copy.deepcopy(scheduler_source["scheduler"]),
        "clip_active_from": merged_guidance["clip_active_from"],
        "ref_image_paths": copy.deepcopy(merged_guidance["ref_image_paths"]),
        "clip_regions": copy.deepcopy(merged_guidance["clip_regions"]),
        "clip_guidance_scale": merged_guidance["clip_guidance_scale"],
        "text_active_from": merged_guidance["text_active_from"],
        "text_list": copy.deepcopy(merged_guidance["text_list"]),
        "text_regions": copy.deepcopy(merged_guidance["text_regions"]),
        "text_guidance_scale": merged_guidance["text_guidance_scale"],
    }
    new_br["history"].append(initial_snap)
    
    branches[new_id] = new_br
    state["active_branch_id"] = new_id
    
    print(f"[merge_branches_engine] Created merged branch {new_id} from {branch_id_1}@step{actual_step_1} and {branch_id_2}@step{actual_step_2}")
    print(f"[merge_branches_engine] Merged branch starts at step {merge_start_step}")
    print(f"[merge_branches_engine] Merged guidance: text_list={merged_guidance['text_list']}, ref_paths={merged_guidance['ref_image_paths']}")
    
    return state, f"Merged {branch_id_1}@{actual_step_1} + {branch_id_2}@{actual_step_2} -> {new_id} at step {merge_start_step}.", new_id


