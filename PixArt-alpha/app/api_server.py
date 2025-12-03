#!/usr/bin/env python
"""
Simple REST API Server for PixArt-Alpha + ControlNet Image Generation

Usage:
    python app/api_server.py configs/pixart_app_config/PixArt_xl2_img1024_controlHed.py \
        --model_path /path/to/model.pth --image_size 1024 --port 8000

API Endpoints:
    POST /generate - Generate images from text prompt and optional control image
    GET /health - Health check endpoint
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

# Add parent directory to path for imports
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from torchvision.utils import make_grid
from fastapi.middleware.cors import CORSMiddleware

from diffusers import PixArtAlphaPipeline
from diffusion import DPMS, SASolverSampler
from diffusion.data.datasets import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
from diffusion.model.hed import HEDdetector
from diffusion.model.nets import PixArt_XL_2, PixArtMS_XL_2, ControlPixArtHalf, ControlPixArtMSHalf
from diffusion.model.utils import resize_and_crop_tensor
from diffusion.utils.misc import read_config
from tools.download import find_model


# ============================================================================
# Configuration & Model Loading
# ============================================================================

app = FastAPI(
    title="PixArt-Alpha ControlNet API",
    description="REST API for image generation using PixArt-Alpha with ControlNet",
    version="1.0.0",
)

# CORS for browser frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model holders
models = {
    "pipe": None,
    "model": None,
    "vae": None,
    "hed": None,
    "config": None,
    "base_ratios": None,
    "weight_dtype": torch.float16,
    "device": None,
}


def load_models(config_path: str, model_path: str, image_size: int = 1024):
    """Load all required models for image generation."""
    global models
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models["device"] = device
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for image generation")
    
    config = read_config(config_path)
    models["config"] = config
    
    weight_dtype = torch.float16
    models["weight_dtype"] = weight_dtype
    
    print(f"Loading models for image size: {image_size}...")
    
    # Load HED detector
    models["hed"] = HEDdetector(False).to(device)
    
    # Load PixArt pipeline (for text encoding)
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        transformer=None,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    models["pipe"] = pipe
    models["vae"] = pipe.vae
    
    # Load ControlNet model
    latent_size = image_size // 8
    lewei_scale = {512: 1, 1024: 2}
    
    if image_size == 512:
        base_model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[image_size])
        model = ControlPixArtHalf(base_model).to(device)
        models["base_ratios"] = ASPECT_RATIO_512_TEST
    else:  # 1024
        base_model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[image_size])
        model = ControlPixArtMSHalf(base_model).to(device)
        models["base_ratios"] = ASPECT_RATIO_1024_TEST
    
    # Load checkpoint
    state_dict = find_model(model_path)['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    elif 'base_model.pos_embed' in state_dict:
        del state_dict['base_model.pos_embed']
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys (missing pos_embed is normal): {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    model.eval()
    model.to(weight_dtype)
    models["model"] = model
    
    print("All models loaded successfully!")


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field("", description="Negative prompt to guide generation")
    num_images: int = Field(4, ge=1, le=8, description="Number of images to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    width: int = Field(1024, ge=256, le=2048, description="Output image width")
    height: int = Field(1024, ge=256, le=2048, description="Output image height")
    guidance_scale: float = Field(4.5, ge=1.0, le=15.0, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(14, ge=5, le=50, description="Number of denoising steps")
    sampler: str = Field("dpm", description="Sampler to use: 'dpm' or 'sa'")


class GenerationResponse(BaseModel):
    """Response model for image generation."""
    images: List[str] = Field(..., description="List of base64-encoded PNG images")
    seed: int = Field(..., description="Seed used for generation")
    control_image: Optional[str] = Field(None, description="Base64-encoded control image (HED edge)")


# ============================================================================
# Utility Functions
# ============================================================================

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a tensor image to base64-encoded PNG string."""
    # Normalize and convert to numpy
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2 * 255
    tensor = tensor.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    
    # Convert to PIL and encode
    img = Image.fromarray(tensor)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL image to base64-encoded PNG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@torch.no_grad()
def _generate_single(
    prompt: str,
    control_image: Optional[Image.Image] = None,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 14,
    sampler: str = "dpm",
) -> tuple[str, int, Optional[str]]:
    """
    Generate a single image using PixArt-Alpha + ControlNet.
    
    Returns:
        Tuple of (base64 image, seed used, control image base64 or None)
    """
    # Always generate one image in this core function for stability
    num_images = 1
    # Normalize sampler input to avoid None/bytes and case issues
    sampler = sampler or "dpm"
    if isinstance(sampler, bytes):
        sampler = sampler.decode("utf-8")
    #print(sampler)
    sampler = sampler.lower()

    device = models["device"]
    config = models["config"]
    pipe = models["pipe"]
    vae = models["vae"]
    model = models["model"]
    hed = models["hed"]
    base_ratios = models["base_ratios"]
    weight_dtype = models["weight_dtype"]
    
    # Set seed
    if seed is None:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    
    torch.cuda.empty_cache()
    
    # Encode prompts
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
        pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt or "")
    
    # Repeat for batch size
    prompt_embeds = prompt_embeds.repeat(num_images, 1, 1)[:, None]
    negative_prompt_embeds = negative_prompt_embeds.repeat(num_images, 1, 1)[:, None]
    prompt_attention_mask = prompt_attention_mask.repeat(num_images, 1)
    
    torch.cuda.empty_cache()
    
    control_image_b64 = None
    
    # Process control image if provided
    if control_image is not None:
        ar = torch.tensor([control_image.size[1] / control_image.size[0]], device=device)[None]
        custom_hw = torch.tensor([control_image.size[1], control_image.size[0]], device=device)[None]
        # For 512 model (non-MS), force square 512x512 to match model latent size
        if getattr(models.get("config"), "image_size", 512) == 512:
            hw = torch.tensor([[models["config"].image_size, models["config"].image_size]], device=device)
            closest_hw = (models["config"].image_size, models["config"].image_size)
        else:
            closest_hw = base_ratios[min(base_ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))]
            hw = torch.tensor(closest_hw, device=device)[None]
        
        condition_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(int(min(closest_hw))),
            T.CenterCrop([int(closest_hw[0]), int(closest_hw[1])]),
            T.ToTensor(),
        ])
        control_tensor = condition_transform(control_image).unsqueeze(0).to(device)
        hed_edge = hed(control_tensor)
        hed_edge = TF.normalize(hed_edge, [.5], [.5])
        hed_edge = hed_edge.repeat(1, 3, 1, 1).to(weight_dtype)
        
        posterior = vae.encode(hed_edge).latent_dist
        condition = posterior.sample()
        # Repeat condition to batch size if needed
        if num_images > 1 and condition.shape[0] == 1:
            condition = condition.repeat(num_images, 1, 1, 1)
        c = condition * config.scale_factor
        
        # Get control visualization from first element
        vis_condition = condition[0:1]
        c_vis = vae.decode(vis_condition)['sample']
        c_vis = torch.clamp(127.5 * c_vis + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        control_image_b64 = pil_to_base64(Image.fromarray(c_vis))
    else:
        c = None
        # For 512 model (non-MS), force square 512x512
        if getattr(models.get("config"), "image_size", 512) == 512:
            ar = torch.tensor([1.0], device=device)[None]
            custom_hw = torch.tensor([config.image_size, config.image_size], device=device)[None]
            hw = torch.tensor([[config.image_size, config.image_size]], device=device)
        else:
            ar = torch.tensor([height / width], device=device)[None]
            custom_hw = torch.tensor([height, width], device=device)[None]
            closest_hw = base_ratios[min(base_ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))]
            hw = torch.tensor(closest_hw, device=device)[None]
    
    latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
    
    # Repeat hw and ar for batch
    hw = hw.repeat(num_images, 1)
    ar = ar.repeat(num_images, 1)
    
    # Generate images
    if sampler == "dpm":
        z = torch.randn(num_images, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=prompt_attention_mask, c=c)
        
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=prompt_embeds,
            uncondition=negative_prompt_embeds,
            cfg_scale=guidance_scale,
            model_kwargs=model_kwargs
        )
        print(f"Generating {num_images} images with seed {seed}...")
        samples = dpm_solver.sample(
            z,
            steps=num_inference_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        ).to(weight_dtype)
        
    elif sampler == "sa":
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=prompt_attention_mask, c=c)
        
        sas_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sas_solver.sample(
            S=num_inference_steps,
            batch_size=num_images,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=prompt_embeds,
            unconditional_conditioning=negative_prompt_embeds,
            unconditional_guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
        )[0].to(weight_dtype)
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Use 'dpm' or 'sa'.")
    
    # Decode latents
    samples = vae.decode(samples / config.scale_factor).sample
    torch.cuda.empty_cache()
    
    # Resize to target dimensions
    samples = resize_and_crop_tensor(samples, custom_hw[0, 1].item(), custom_hw[0, 0].item())
    
    # Convert to base64 image (single)
    img_b64 = tensor_to_base64(samples[0])
    
    return img_b64, seed, control_image_b64


@torch.no_grad()
def generate_images(
    prompt: str,
    control_image: Optional[Image.Image] = None,
    negative_prompt: str = "",
    num_images: int = 4,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 14,
    sampler: str = "dpm",
) -> tuple[List[str], int, Optional[str]]:
    """
    Generate multiple images by calling the single-image core function repeatedly.
    This avoids batch-dimension mismatches inside the DPM solver when using ControlNet.
    """
    images_b64: List[str] = []
    control_image_b64: Optional[str] = None
    used_seed: Optional[int] = seed

    for i in range(num_images):
        # For reproducibility, reuse the same seed if provided; otherwise let the core pick a random one.
        current_seed = seed
        img_b64, current_seed, c_b64 = _generate_single(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            seed=current_seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            sampler=sampler,
        )
        images_b64.append(img_b64)
        if control_image_b64 is None:
            control_image_b64 = c_b64
        if used_seed is None:
            used_seed = current_seed

    if used_seed is None:
        used_seed = seed if seed is not None else 0

    return images_b64, used_seed, control_image_b64


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cuda_available = torch.cuda.is_available()
    models_loaded = models["model"] is not None
    return {
        "status": "healthy" if (cuda_available and models_loaded) else "degraded",
        "cuda_available": cuda_available,
        "models_loaded": models_loaded,
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(
    prompt: str = Form(..., description="Text prompt for image generation"),
    negative_prompt: str = Form("", description="Negative prompt"),
    num_images: int = Form(4, ge=1, le=8, description="Number of images to generate"),
    seed: Optional[int] = Form(None, description="Random seed"),
    width: int = Form(1024, ge=256, le=2048, description="Output width"),
    height: int = Form(1024, ge=256, le=2048, description="Output height"),
    guidance_scale: float = Form(4.5, ge=1.0, le=15.0, description="Guidance scale"),
    num_inference_steps: int = Form(14, ge=5, le=50, description="Inference steps"),
    sampler: str = Form("dpm", description="Sampler: 'dpm' or 'sa'"),
    image: Optional[UploadFile] = File(None, description="Optional control image"),
):
    """
    Generate images from text prompt and optional control image.
    
    - **prompt**: Text description of the desired image
    - **negative_prompt**: What to avoid in the generation
    - **num_images**: Number of images to generate (1-8, default 4)
    - **seed**: Random seed for reproducibility
    - **width/height**: Output image dimensions
    - **guidance_scale**: How strongly to follow the prompt (higher = stronger)
    - **num_inference_steps**: Number of denoising steps
    - **sampler**: 'dpm' (faster) or 'sa' (different quality)
    - **image**: Optional control image for edge-guided generation
    
    Returns base64-encoded PNG images.
    """
    if models["model"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Server is not ready.")
    
    try:
        # Load control image if provided
        control_image = None
        if image is not None:
            image_bytes = await image.read()
            control_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Generate images
        images_b64, used_seed, control_b64 = generate_images(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_images=num_images,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            sampler=sampler,
        )
        
        return GenerationResponse(
            images=images_b64,
            seed=used_seed,
            control_image=control_b64,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/json", response_model=GenerationResponse)
async def generate_json_endpoint(request: GenerationRequest):
    """
    Generate images from JSON request body (no control image support).
    
    For control image support, use the /generate endpoint with form data.
    """
    if models["model"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Server is not ready.")
    
    try:
        images_b64, used_seed, _ = generate_images(
            prompt=request.prompt,
            control_image=None,
            negative_prompt=request.negative_prompt,
            num_images=request.num_images,
            seed=request.seed,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            sampler=request.sampler,
        )
        
        return GenerationResponse(
            images=images_b64,
            seed=used_seed,
            control_image=None,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ============================================================================
# Binary image endpoints (PNG)
# ============================================================================

def _b64_to_bytes(b64_str: str) -> bytes:
    try:
        return base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decode image: {str(e)}")


@app.post("/generate/image")
async def generate_image_endpoint(
    prompt: str = Form(..., description="Text prompt for image generation"),
    negative_prompt: str = Form("", description="Negative prompt"),
    num_images: int = Form(4, ge=1, le=8, description="Number of images to generate"),
    index: int = Form(0, ge=0, description="Index of image to return (0-based)"),
    seed: Optional[int] = Form(None, description="Random seed"),
    width: int = Form(1024, ge=256, le=2048, description="Output width"),
    height: int = Form(1024, ge=256, le=2048, description="Output height"),
    guidance_scale: float = Form(4.5, ge=1.0, le=15.0, description="Guidance scale"),
    num_inference_steps: int = Form(14, ge=5, le=50, description="Inference steps"),
    sampler: str = Form("dpm", description="Sampler: 'dpm' or 'sa'"),
    image: Optional[UploadFile] = File(None, description="Optional control image"),
):
    """
    Generate images and return a single PNG image (bytes) for direct browser consumption.
    Use 'index' to select which image to return (default 0).
    """
    if models["model"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Server is not ready.")

    try:
        control_image = None
        if image is not None:
            image_bytes = await image.read()
            control_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        images_b64, used_seed, _ = generate_images(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_images=num_images,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            sampler=sampler,
        )

        if not images_b64:
            raise HTTPException(status_code=500, detail="No images generated")
        if index < 0 or index >= len(images_b64):
            index = 0

        img_bytes = _b64_to_bytes(images_b64[index])
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/grid")
async def generate_grid_endpoint(
    prompt: str = Form(..., description="Text prompt for image generation"),
    negative_prompt: str = Form("", description="Negative prompt"),
    num_images: int = Form(4, ge=1, le=8, description="Number of images to generate"),
    seed: Optional[int] = Form(None, description="Random seed"),
    width: int = Form(1024, ge=256, le=2048, description="Output width"),
    height: int = Form(1024, ge=256, le=2048, description="Output height"),
    guidance_scale: float = Form(4.5, ge=1.0, le=15.0, description="Guidance scale"),
    num_inference_steps: int = Form(14, ge=5, le=50, description="Inference steps"),
    sampler: str = Form("dpm", description="Sampler: 'dpm' or 'sa'"),
    image: Optional[UploadFile] = File(None, description="Optional control image"),
):
    """
    Generate images and return a 2x2 PNG grid (or 1xN/2xN grid based on count).
    """
    if models["model"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Server is not ready.")

    try:
        control_image = None
        if image is not None:
            image_bytes = await image.read()
            control_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        images_b64, used_seed, _ = generate_images(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_images=num_images,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            sampler=sampler,
        )

        if not images_b64:
            raise HTTPException(status_code=500, detail="No images generated")

        # Decode to PIL images
        pil_images = []
        for b64 in images_b64:
            img = Image.open(io.BytesIO(_b64_to_bytes(b64))).convert("RGB")
            pil_images.append(img)

        # Build grid
        cols = 2 if len(pil_images) > 1 else 1
        rows = (len(pil_images) + cols - 1) // cols
        w, h = pil_images[0].size
        grid = Image.new("RGB", (cols * w, rows * h))
        for i, img in enumerate(pil_images):
            x = (i % cols) * w
            y = (i // cols) * h
            grid.paste(img, (x, y))

        buf = io.BytesIO()
        grid.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PixArt-Alpha ControlNet REST API Server")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, default="/home/jaesang/interaction-to-image/PixArt-alpha/model/PixArt-XL-2-512-ControlNet.pth", help="Path to model checkpoint")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 1024],
                        help="Image size (512 or 1024)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn
    
    args = parse_args()
    
    # Load models before starting server
    load_models(args.config, args.model_path, args.image_size)
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port)

