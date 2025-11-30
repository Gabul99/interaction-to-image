"""
Pipeline Manager - Wraps i2i pipeline for image generation and streaming
"""
import os
import sys
import torch
import random
from typing import Optional, Callable, List, Dict, Tuple
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from i2i.utils import load_i2i_model, sanity_check, get_phrases_idx_in_prompt
from i2i.pipeline import ASPECT_RATIO_512_BIN
from diffusers.utils.torch_utils import randn_tensor

try:
    from .models import ObjectChip, BoundingBox, FeedbackRecord, FeedbackType, FeedbackArea
except ImportError:
    from models import ObjectChip, BoundingBox, FeedbackRecord, FeedbackType, FeedbackArea


class PipelineManager:
    """Pipeline manager for i2i model"""
    
    def __init__(self, model_id: str = "PixArt-alpha/PixArt-XL-2-512x512", device: str = "cuda"):
        self._model_id = model_id
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._tokenizer = None
        self._model_loaded = False
    
    def load_model(self):
        """Load i2i model"""
        if self._model_loaded:
            return
        
        print(f"[PipelineManager] Loading model: {self._model_id}")
        self._pipe, self._tokenizer = load_i2i_model(self._model_id, self._device)
        self._model_loaded = True
        print(f"[PipelineManager] Model loaded")
    
    def _convert_bboxes_to_i2i_format(
        self,
        bboxes: List[BoundingBox],
        objects: List[ObjectChip],
        height: int,
        width: int
    ) -> tuple[Optional[List[str]], Optional[List[List[List[float]]]]]:
        """
        Convert frontend bbox format to i2i format.
        Frontend: {x, y, width, height} (0~1 relative coordinates)
        i2i: [[ul_x, ul_y, lr_x, lr_y]] (0~1 relative coordinates)
        """
        if not bboxes or not objects:
            return None, None
        
        object_map = {obj.id: obj for obj in objects}
        
        phrases = []
        bbox_list = []
        
        for bbox in bboxes:
            obj = object_map.get(bbox.objectId)
            if not obj:
                continue
            
            ul_x = bbox.x
            ul_y = bbox.y
            lr_x = bbox.x + bbox.width
            lr_y = bbox.y + bbox.height
            
            lr_x = min(lr_x, 1.0 - 1e-9)
            lr_y = min(lr_y, 1.0 - 1e-9)
            
            phrases.append(obj.label)
            bbox_list.append([[ul_x, ul_y, lr_x, lr_y]])
        
        return phrases if phrases else None, bbox_list if bbox_list else None
    
    def _parse_intervals(self, interval_str: str) -> List[Tuple[int, int]]:
        """Parse interval string from config.json format. Example: "0-25" -> [(0, 25)]"""
        if not interval_str or not isinstance(interval_str, str):
            return []
        
        intervals = []
        for part in interval_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                intervals.append((int(start.strip()), int(end.strip())))
            else:
                num = int(part.strip())
                intervals.append((num, num))
        return intervals
    
    def _parse_feedback_to_guidance(
        self,
        feedbacks: List[FeedbackRecord],
        total_steps: int = 50
    ) -> Dict:
        """
        Convert feedback to guidance parameters (matching backbone model config.json structure).
        
        Guidance rules (for 50 steps):
        - Layout: "0-25" (fixed 0~25 steps if provided)
        - Edge: "0-30" (fixed 0~30 steps if provided)
        - Style, Text: "20-50" (from step 20 until end if provided)
        """
        guidance_params = {
            "enable_grounding_guidance": False,
            "enable_clip_guidance": False,
            "enable_text_guidance": False,
            "enable_edge_guidance": False,
            "layout_guidance_intervals": [],
            "clip_guidance_intervals": [],
            "text_guidance_intervals": [],
            "edge_guidance_intervals": [],
            "text_guidance_text": None,
            "text_guidance_regions": [],
            "clip_guidance_regions": [],
            "ref_image_paths": None,
            "text_guidance_scale": 0.0,
            "clip_guidance_scale": 0.0,
            "edge_guidance_scale": 0.0,
            "loss_scale": 10.0,
            "loss_threshold": 0.00001,
            "gradient_weight": 5.0,
            "global_update_max_iter_per_step": 3,
            "edge_preprocessor": "hed",
        }
        
        if not feedbacks:
            return guidance_params
        
        layout_feedbacks = []
        edge_feedbacks = []
        style_feedbacks = []
        text_feedbacks = []
        
        for feedback in feedbacks:
            if feedback.type == FeedbackType.text:
                if feedback.area == FeedbackArea.bbox or feedback.area == FeedbackArea.point:
                    layout_feedbacks.append(feedback)
                else:
                    text_feedbacks.append(feedback)
            elif feedback.type == FeedbackType.image:
                style_feedbacks.append(feedback)
        
        if layout_feedbacks:
            guidance_params["enable_grounding_guidance"] = True
            guidance_params["layout_guidance_intervals"] = [(0, 25)]
        
        if edge_feedbacks:
            guidance_params["enable_edge_guidance"] = True
            guidance_params["edge_guidance_intervals"] = [(0, 30)]
            guidance_params["edge_guidance_scale"] = 2.0
            guidance_params["edge_preprocessor"] = "hed"
        
        if style_feedbacks:
            guidance_params["enable_clip_guidance"] = True
            guidance_params["clip_guidance_intervals"] = [(20, total_steps - 1)]
            guidance_params["clip_guidance_scale"] = 5.0
            guidance_params["clip_guidance_regions"] = [[0.0, 0.0, 1.0, 1.0]]
            
            ref_image_paths = []
            for feedback in style_feedbacks:
                if feedback.imageUrl:
                    ref_image_paths.append(feedback.imageUrl)
            
            if ref_image_paths:
                guidance_params["ref_image_paths"] = ref_image_paths
        
        if text_feedbacks:
            guidance_params["enable_text_guidance"] = True
            guidance_params["text_guidance_intervals"] = [(20, total_steps - 1)]
            guidance_params["text_guidance_scale"] = 2.0
            
            text_list = [f.text for f in text_feedbacks if f.text]
            if text_list:
                guidance_params["text_guidance_text"] = text_list
            
            regions = []
            for feedback in text_feedbacks:
                if feedback.bbox:
                    regions.append([
                        feedback.bbox["x"],
                        feedback.bbox["y"],
                        feedback.bbox["x"] + feedback.bbox["width"],
                        feedback.bbox["y"] + feedback.bbox["height"]
                    ])
                elif feedback.point:
                    regions.append([
                        max(0, feedback.point["x"] - 0.1),
                        max(0, feedback.point["y"] - 0.1),
                        min(1, feedback.point["x"] + 0.1),
                        min(1, feedback.point["y"] + 0.1)
                    ])
                else:
                    regions.append([0.0, 0.0, 1.0, 1.0])
            
            guidance_params["text_guidance_regions"] = regions if regions else [[0.0, 0.0, 1.0, 1.0]]
        
        return guidance_params
    
    async def generate_image(
        self,
        prompt: str,
        objects: List[ObjectChip],
        bboxes: List[BoundingBox],
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        callback: Optional[Callable[[int, int, Image.Image], None]] = None,
        feedback_callback: Optional[Callable[[int], Optional[List[FeedbackRecord]]]] = None,
    ) -> Image.Image:
        """
        Generate image using i2i pipeline.
        
        Args:
            prompt: Text prompt
            objects: Object list
            bboxes: Bounding box list
            width: Image width
            height: Image height
            num_inference_steps: Number of inference steps
            callback: Called at each step (step_idx, timestep, image)
            feedback_callback: Called to get feedback (step_idx) -> Optional[List[FeedbackRecord]]
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"[PipelineManager] Generating image: prompt='{prompt}', steps={num_inference_steps}")
        
        if bboxes and len(bboxes) > 0:
            phrases, bbox_list = self._convert_bboxes_to_i2i_format(bboxes, objects, height, width)
            if bbox_list:
                bbox_list = sanity_check(bbox_list, phrases)
                phrases_idx = get_phrases_idx_in_prompt(prompt, phrases, self._tokenizer)
            else:
                phrases = None
                phrases_idx = None
        else:
            phrases = None
            bbox_list = None
            phrases_idx = None
        
        target_height, target_width = self._pipe.classify_height_width_bin(
            height, width, ratios=ASPECT_RATIO_512_BIN
        )
        
        latent_height = target_height // self._pipe.vae_scale_factor
        latent_width = target_width // self._pipe.vae_scale_factor
        latent_shape = (1, 4, latent_height, latent_width)
        
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        latents = randn_tensor(
            latent_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float16
        ) * self._pipe.scheduler.init_noise_sigma
        
        initial_feedbacks: List[FeedbackRecord] = []
        if feedback_callback:
            initial_feedbacks = feedback_callback(0) or []
        
        guidance_params = self._parse_feedback_to_guidance(initial_feedbacks, num_inference_steps)
        
        if bbox_list:
            guidance_params["enable_grounding_guidance"] = True
            guidance_params["layout_guidance_intervals"] = [(0, 25)]
        
        def _callback_wrapper(step_idx, timestep, latents_cb):
            try:
                with torch.no_grad():
                    z = latents_cb / self._pipe.vae.config.scaling_factor
                    img_tensor = self._pipe.vae.decode(z, return_dict=False)[0]
                    img_tensor = self._pipe.resize_and_crop_tensor(img_tensor, width, height)
                    imgs = self._pipe.image_processor.postprocess(img_tensor, output_type="pil")
                    
                    if imgs and len(imgs) > 0:
                        image = imgs[0]
                        if callback:
                            callback(step_idx, timestep, image)
                        
                        if feedback_callback:
                            new_feedbacks = feedback_callback(step_idx)
            except Exception as e:
                print(f"[PipelineManager] Callback error: {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        if phrases is None:
            phrases = []
        if phrases_idx is None:
            phrases_idx = []
        if bbox_list is None:
            bbox_list = []
        
        result = self._pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=target_height,
            width=target_width,
            generator=generator,
            latents=latents,
            callback=_callback_wrapper,
            callback_steps=1,
            output_type="pil",
            bbox_list=bbox_list,
            phrases=phrases,
            phrases_idx=phrases_idx,
            enable_grounding_guidance=guidance_params["enable_grounding_guidance"],
            layout_guidance_intervals=guidance_params["layout_guidance_intervals"],
            loss_scale=guidance_params["loss_scale"],
            loss_threshold=guidance_params["loss_threshold"],
            gradient_weight=guidance_params["gradient_weight"],
            global_update_max_iter_per_step=guidance_params["global_update_max_iter_per_step"],
            enable_text_guidance=guidance_params["enable_text_guidance"],
            text_guidance_intervals=guidance_params["text_guidance_intervals"],
            text_guidance_text=guidance_params["text_guidance_text"],
            text_guidance_regions=guidance_params["text_guidance_regions"],
            text_guidance_scale=guidance_params["text_guidance_scale"],
            enable_clip_guidance=guidance_params["enable_clip_guidance"],
            clip_guidance_intervals=guidance_params["clip_guidance_intervals"],
            ref_image_paths=guidance_params["ref_image_paths"],
            clip_guidance_regions=guidance_params["clip_guidance_regions"],
            clip_guidance_scale=guidance_params["clip_guidance_scale"],
            enable_edge_guidance=guidance_params["enable_edge_guidance"],
            edge_guidance_intervals=guidance_params["edge_guidance_intervals"],
            edge_guidance_scale=guidance_params["edge_guidance_scale"],
            edge_preprocessor=guidance_params["edge_preprocessor"],
        )
        
        final_image = result.images[0]
        
        if final_image.size != (width, height):
            final_image = final_image.resize((width, height), Image.LANCZOS)
        
        return final_image
