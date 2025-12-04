#!/usr/bin/env python
"""
Simple REST API Server for plain PixArt-Alpha 512px image generation.

This server is intentionally minimal and focuses only on:

    input:  text prompt
    output: 4 images (internally as PIL.Image, exposed as base64 PNG over HTTP)

It reuses the lightweight diffusers pipeline setup from `app_512.py` without
ControlNet or additional schedulers/samplers.

Usage (from repo root):

    uvicorn app.simple_api_server:app --host 0.0.0.0 --port 8001

Then call:

    POST /generate
    JSON body: { "prompt": "a cute cat" }
"""
from __future__ import annotations

import base64
import io
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, PixArtAlphaPipeline
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from openai import OpenAI  # type: ignore


# Ensure project root is on sys.path for consistency with other app modules
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


# =============================================================================
# Configuration
# =============================================================================

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_NUM_IMAGES = 4
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_STEPS = 50

USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global pipeline & OpenAI client holders
pipe: Optional[PixArtAlphaPipeline] = None
_openai_client: Optional["OpenAI"] = None


def load_pipeline() -> PixArtAlphaPipeline:
    """
    Lazily load and configure the PixArt-Alpha 512x512 diffusers pipeline.

    This mirrors the logic in `app_512.py` but is kept minimal:
      - 512px PixArt-XL checkpoint
      - DPM-Solver scheduler
      - Text encoder BetterTransformer speedup
    """
    global pipe
    if pipe is not None:
        return pipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PixArt-Alpha generation (no CPU support).")

    print("Loading PixArt-Alpha 512x512 pipeline for simple REST API...")
    local_rank = 0  # kept simple, single GPU

    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Use DPM-Solver scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        print(f"Loaded PixArt pipeline on device: {device}")

    # Speed up T5 text encoder
    pipe.text_encoder.to_bettertransformer()

    # Optional compilation for the transformer
    if USE_TORCH_COMPILE:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
        print("PixArt transformer compiled with torch.compile")

    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    print("PixArt-Alpha 512x512 pipeline is ready.")
    return pipe


# =============================================================================
# Pydantic Models
# =============================================================================


class GenerateRequest(BaseModel):
    """Minimal request model: text to images."""

    prompt: str = Field(..., description="Text prompt for image generation")
    num_images: int = Field(
        DEFAULT_NUM_IMAGES,
        ge=1,
        le=DEFAULT_NUM_IMAGES,
        description="Number of images to generate (fixed to 4 by default).",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional random seed for reproducibility. If omitted, a random seed is used.",
    )
    width: int = Field(
        DEFAULT_WIDTH,
        ge=256,
        le=1024,
        description="Image width. For best results keep at 512.",
    )
    height: int = Field(
        DEFAULT_HEIGHT,
        ge=256,
        le=1024,
        description="Image height. For best results keep at 512.",
    )
    guidance_scale: float = Field(
        DEFAULT_GUIDANCE_SCALE,
        ge=1.0,
        le=15.0,
        description="Classifier-free guidance scale.",
    )
    num_inference_steps: int = Field(
        DEFAULT_STEPS,
        ge=5,
        le=50,
        description="Number of denoising steps.",
    )
    previous_prompt: Optional[str] = Field(
        None,
        description=(
            "Optional previous turn prompt. "
            "If provided and GPT prompting is enabled, the final prompt is composed "
            "from previous + current prompt using a GPT model."
        ),
    )


class GenerateResponse(BaseModel):
    """
    Response model.

    - `images` contains 4 base64-encoded PNG images.
    - `seed` is the seed actually used for generation.
    """

    images: List[str] = Field(
        ..., description="List of base64-encoded PNG images (length == num_images)."
    )
    seed: int = Field(..., description="Seed used for generation.")
    full_prompt: Optional[str] = Field(
        None,
        description="Final composed prompt actually used for generation (may be GPT-refined).",
    )
    image_caption: Optional[str] = Field(
        None,
        description="If one or more images were provided, the GPT-generated caption(s) used for prompt composition.",
    )


# =============================================================================
# Utility Functions
# =============================================================================


def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL.Image to base64-encoded PNG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def randomize_seed(seed: Optional[int]) -> int:
    """Return a valid seed, randomizing if None."""
    if seed is None:
        return random.randint(0, MAX_SEED)
    return int(seed)


def _get_openai_client() -> "OpenAI":
    """
    Lazily construct an OpenAI client.

    Uses the official `openai` Python package if installed and relies on
    standard environment configuration (e.g., OPENAI_API_KEY).
    """
    global _openai_client
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Install `openai>=1.0.0` and set OPENAI_API_KEY."
        )
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _compose_full_prompt_with_gpt(
    previous_prompt: Optional[str],
    current_prompt: str,
    *,
    model_env_var: str = "PROMPT_MODEL",
    default_model: str = "gpt-4.1-mini",
) -> str:
    """
    Use a GPT model to compose a single, high-quality image generation prompt
    from previous + current prompt.
    """
    previous = (previous_prompt or "").strip()
    current = current_prompt.strip()
    if not previous:
        return current

    client = _get_openai_client()
    model = os.getenv(model_env_var, default_model)

    system_msg = (
        "You are an assistant that generates a single, concise image generation prompt by integrating \
        feedback to be applied on the previously generated image, based on the \
        caption of the previously generated image and the current user prompt."
        
        # integrates feedback to be applied on the previous generated image.

        # into the image prompt and 
        
        # merges multiple user prompts into a single, "
        # "concise, image generation prompt suitable for a diffusion model. "
        # "Output only the final prompt, in English."
    )
    user_msg = (
        "Previously generated image caption:\n"
        f"{previous}\n\n"
        "Current prompt(feedback):\n"
        f"{current}\n\n"
        "Compose a single, combined prompt that modifies the previously generated image based on the feedback."
    )

    # Using chat.completions API (OpenAI Python client >=1.0.0)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=256,
        temperature=0.7,
    )
    text = completion.choices[0].message.content or ""
    return text.strip()


def _caption_image_with_gpt(
    image_bytes: bytes,
    hint_prompt: Optional[str] = None,
    *,
    model_env_var: str = "VISION_MODEL",
    default_model: str = "gpt-4.1-mini",
) -> str:
    """
    Use a GPT vision-capable model to produce a rich caption / prompt-like description
    for the given image.
    """
    client = _get_openai_client()
    model = os.getenv(model_env_var, default_model)

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    base_instruction = (
        "You are an advanced AI assistant specializing in image analysis, providing a detailed description of a image."
        "Focus on visual content, style, lighting, composition, and mood. "
        "Do not mention that you are describing an image."
    )

    if hint_prompt:
        base_instruction += (
            "\n\nThe image was synthesized using this text prompt. You may subtly incorporate its intent, "
            "but stay faithful to what is visible in the image:\n"
            f"{hint_prompt}"
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": base_instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )
    text = completion.choices[0].message.content or ""
    return text.strip()


def _caption_images_with_gpt(
    image_bytes_list: List[bytes],
    hint_prompt: Optional[str] = None,
) -> str:
    """
    Use a GPT vision-capable model to produce a combined caption / prompt-like description
    for multiple images.

    This is a simple wrapper that calls `_caption_image_with_gpt` for each image and
    concatenates the captions into a single description.
    """
    captions: List[str] = []
    for image_bytes in image_bytes_list:
        captions.append(
            _caption_image_with_gpt(
                image_bytes=image_bytes,
                hint_prompt=hint_prompt,
            )
        )
    return "\n\n".join(captions).strip()


def _build_final_prompt(
    current_prompt: str,
    previous_prompt: Optional[str],
    image_bytes_list: Optional[List[bytes]],
) -> Tuple[str, Optional[str]]:
    """
    Build the final generation prompt according to the rules:

    - If one or more images are provided:
        1) Generate image caption(s) with GPT.
        2) Compose a full prompt from caption + (previous + current) using GPT.
    - Else if previous prompt is provided:
        1) Compose full prompt from previous + current using GPT.
    - Else:
        1) Use current prompt as-is.

    Returns:
        (final_prompt, image_caption_or_none)
    """
    # With GPT
    try:
        if image_bytes_list:
            # Caption the image(s) first
            caption = _caption_images_with_gpt(image_bytes_list, hint_prompt=None)

            # Compose full prompt from caption + text context
            text_context_parts = []
            if previous_prompt:
                text_context_parts.append(previous_prompt.strip())
            if current_prompt.strip():
                text_context_parts.append(current_prompt.strip())
            text_context = "\n\n".join(text_context_parts) if text_context_parts else ""

            # Treat caption as the "previous" and text context as "current"
            if text_context:
                full_prompt = _compose_full_prompt_with_gpt(
                    previous_prompt=caption,
                    current_prompt=text_context,
                )
            else:
                full_prompt = caption

            return full_prompt, caption

        # Text-only case with previous + current
        if previous_prompt:
            full_prompt = _compose_full_prompt_with_gpt(previous_prompt, current_prompt)
            return full_prompt, None

        return current_prompt.strip(), None
    except Exception as e:  # pragma: no cover - best-effort GPT integration
        # On any GPT failure, fall back gracefully
        print(f"[simple_api_server] GPT prompt building failed: {e}")
        if previous_prompt:
            return f"{previous_prompt.strip()} {current_prompt.strip()}".strip(), None
        return current_prompt.strip(), None


def generate_pil_images(
    prompt: str,
    num_images: int = DEFAULT_NUM_IMAGES,
    seed: Optional[int] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_STEPS,
) -> tuple[List[Image.Image], int]:
    """
    Core generation function.

    This is a plain Python helper you can import and use directly:

        from app.simple_api_server import generate_pil_images
        images, used_seed = generate_pil_images("a cute cat")

    Returns:
        (images, used_seed)
        - images: list of PIL.Image.Image
        - used_seed: int
    """
    pipeline = load_pipeline()

    used_seed = randomize_seed(seed)
    generator = torch.Generator(device=pipeline.device).manual_seed(used_seed)

    print(f"num_inference_steps: {num_inference_steps}")
    result = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images,
        output_type="pil",
        max_sequence_length=512,
    )

    # diffusers returns images as a list of PIL.Image.Image
    images: List[Image.Image] = result.images
    return images, used_seed


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="PixArt-Alpha Simple API",
    description="Minimal REST API for plain PixArt-Alpha 512px text-to-image generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """
    Eagerly load the pipeline on startup so the first request is not slow.
    If CUDA is not available, we still start the server but mark status as degraded.
    """
    try:
        if torch.cuda.is_available():
            load_pipeline()
        else:
            print("WARNING: CUDA is not available. PixArt-Alpha generation will not work.")
    except Exception as e:
        # Log but don't crash; /health will show that the model is not ready.
        print(f"Failed to initialize PixArt-Alpha pipeline on startup: {e}")


@app.get("/health")
def health_check():
    """Simple health endpoint."""
    return {
        "status": "healthy" if (torch.cuda.is_available() and pipe is not None) else "degraded",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": pipe is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    """
    Generate images from a text prompt.

    - **prompt**: Text description of the desired image.
    - **num_images**: Number of images to generate (default 4, max 4).
    - **seed**: Optional seed for reproducibility.
    - **width/height**: Output dimensions (512x512 recommended).
    - **guidance_scale**: How strongly to follow the prompt.
    - **num_inference_steps**: Number of denoising steps.

    Returns:
        Base64-encoded PNG images and the seed used.
    """
    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=503, detail="CUDA is required for PixArt-Alpha generation."
        )

    try:
        # Build final prompt (text-only path; no image_bytes)
        final_prompt, _ = _build_final_prompt(
            current_prompt=request.prompt,
            previous_prompt=request.previous_prompt,
            image_bytes_list=None,
        )

        images, used_seed = generate_pil_images(
            prompt=final_prompt,
            num_images=request.num_images,
            seed=request.seed,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
        )

        images_b64 = [pil_to_base64(img) for img in images]
        return GenerateResponse(
            images=images_b64,
            seed=used_seed,
            full_prompt=final_prompt,
            image_caption=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate_with_image", response_model=GenerateResponse)
async def generate_with_image_endpoint(
    current_prompt: str = Form(""),
    previous_prompt: Optional[str] = Form(
        None,
        description="Optional previous turn prompt for GPT-based composition.",
    ),
    num_images: int = Form(DEFAULT_NUM_IMAGES),
    seed: Optional[int] = Form(None),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE),
    num_inference_steps: int = Form(DEFAULT_STEPS),
    image: Optional[UploadFile] = File(
        None, description="Optional reference image used to derive a caption/prompt."
    ),
    images: Optional[List[UploadFile]] = File(
        None,
        description="Optional list of reference images used to derive a caption/prompt.",
    ),
):
    """
    Extended generation endpoint that optionally accepts an image.

    Prompt determination rules:
    - If `image` is provided:
        1. Generate an image caption using a GPT vision model.
        2. Compose a full prompt from caption + (previous + current) using GPT.
    - Else if `previous_prompt` is provided:
        1. Compose full prompt from previous + current using GPT.
    - Else:
        1. Use `current_prompt` as-is.
    """
    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=503, detail="CUDA is required for PixArt-Alpha generation."
        )

    try:
        image_bytes_list: List[bytes] = []
        if image is not None:
            image_bytes_list.append(await image.read())
        if images is not None:
            for upload_file in images:
                if upload_file is not None:
                    image_bytes_list.append(await upload_file.read())

        final_prompt, caption = _build_final_prompt(
            current_prompt=current_prompt,
            previous_prompt=previous_prompt,
            image_bytes_list=image_bytes_list or None,
        )
        print(f"Caption: {caption}")
        print(f"Final prompt: {final_prompt}")
        

        images, used_seed = generate_pil_images(
            prompt=final_prompt,
            num_images=num_images,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        images_b64 = [pil_to_base64(img) for img in images]
        return GenerateResponse(
            images=images_b64,
            seed=used_seed,
            full_prompt=final_prompt,
            image_caption=caption,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # When run as a script: python app/simple_api_server.py
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SIMPLE_API_PORT", "8001")))


