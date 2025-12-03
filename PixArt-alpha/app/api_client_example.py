#!/usr/bin/env python
"""
Example client for PixArt-Alpha ControlNet REST API

Usage:
    # Text-only generation (4 images)
    python app/api_client_example.py --prompt "a beautiful sunset over mountains"
    
    # With control image
    python app/api_client_example.py --prompt "anime style landscape" --image path/to/image.png
    
    # Custom settings
    python app/api_client_example.py --prompt "cyberpunk city" --num_images 2 --seed 42
"""
import argparse
import base64
import json
import os
from pathlib import Path

import requests


def decode_and_save_images(images_b64: list, output_dir: str, prefix: str = "output"):
    """Decode base64 images and save to files."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, img_b64 in enumerate(images_b64):
        img_bytes = base64.b64decode(img_b64)
        output_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")
    
    return saved_paths


def generate_images_text_only(
    api_url: str,
    prompt: str,
    negative_prompt: str = "",
    num_images: int = 4,
    seed: int = None,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 14,
    sampler: str = "dpm",
):
    """Generate images using JSON endpoint (text-only, no control image)."""
    endpoint = f"{api_url}/generate/json"
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_images": num_images,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "sampler": sampler,
    }
    
    if seed is not None:
        payload["seed"] = seed
    
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    
    return response.json()


def generate_images_with_control(
    api_url: str,
    prompt: str,
    image_path: str = None,
    negative_prompt: str = "",
    num_images: int = 4,
    seed: int = None,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 14,
    sampler: str = "dpm",
):
    """Generate images using form endpoint (supports control image)."""
    endpoint = f"{api_url}/generate"
    
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_images": num_images,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "sampler": sampler,
    }
    
    if seed is not None:
        data["seed"] = seed
    
    files = {}
    if image_path:
        files["image"] = open(image_path, "rb")
    
    try:
        response = requests.post(endpoint, data=data, files=files if files else None)
        response.raise_for_status()
        return response.json()
    finally:
        if files:
            files["image"].close()


def main():
    parser = argparse.ArgumentParser(description="PixArt-Alpha ControlNet API Client")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000",
                        help="API server URL")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to control image (optional)")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--width", type=int, default=1024,
                        help="Output width")
    parser.add_argument("--height", type=int, default=1024,
                        help="Output height")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=14,
                        help="Number of inference steps")
    parser.add_argument("--sampler", type=str, default="dpm", choices=["dpm", "sa"],
                        help="Sampler type")
    parser.add_argument("--output_dir", type=str, default="output/api_results",
                        help="Output directory for generated images")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_images} images...")
    print(f"Prompt: {args.prompt}")
    
    if args.image:
        print(f"Control image: {args.image}")
        result = generate_images_with_control(
            api_url=args.api_url,
            prompt=args.prompt,
            image_path=args.image,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            seed=args.seed,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            sampler=args.sampler,
        )
    else:
        result = generate_images_text_only(
            api_url=args.api_url,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            seed=args.seed,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            sampler=args.sampler,
        )
    
    print(f"Seed used: {result['seed']}")
    
    # Save generated images
    decode_and_save_images(result["images"], args.output_dir, prefix="generated")
    
    # Save control image if present
    if result.get("control_image"):
        decode_and_save_images([result["control_image"]], args.output_dir, prefix="control")
    
    print("Done!")


if __name__ == "__main__":
    main()


