import { API_BASE_URL } from "../config/api";

export interface SimpleGenerateRequest {
  prompt: string;
  previous_prompt?: string | null;
  num_images?: number;
  seed?: number | null;
  width?: number;
  height?: number;
  guidance_scale?: number;
  num_inference_steps?: number;
}

export interface SimpleGenerateResponse {
  images: string[];
  seed: number;
  full_prompt?: string;
  image_caption?: string | null;
}

// Allow overriding the simple PixArt API separately from the main backend
const SIMPLE_PIXART_API_BASE_URL: string =
  // Vite env var (recommended)
  // @ts-ignore
  (import.meta.env?.VITE_SIMPLE_PIXART_API_BASE_URL as string | undefined) ??
  API_BASE_URL;

export async function generateSimpleImages(
  params: SimpleGenerateRequest
): Promise<SimpleGenerateResponse> {
  const body: SimpleGenerateRequest = {
    prompt: params.prompt,
    previous_prompt: params.previous_prompt ?? null,
    num_images: params.num_images ?? 4,
    seed: params.seed ?? null,
    width: params.width ?? 512,
    height: params.height ?? 512,
    guidance_scale: params.guidance_scale ?? 4.5,
    num_inference_steps: params.num_inference_steps ?? 50,
  };

  const res = await fetch(`${SIMPLE_PIXART_API_BASE_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`generateSimpleImages failed: ${res.status}`);
  }

  const data = (await res.json()) as SimpleGenerateResponse;
  if (!Array.isArray(data.images)) {
    throw new Error("Invalid response from simple PixArt API: images missing");
  }

  return data;
}

export interface GenerateWithImageParams {
  current_prompt: string;
  previous_prompt?: string | null;
  num_images?: number;
  seed?: number | null;
  width?: number;
  height?: number;
  guidance_scale?: number;
  num_inference_steps?: number;
  imageDataUrl: string; // data URL (e.g., data:image/png;base64,...)
}

export async function generateWithImage(
  params: GenerateWithImageParams
): Promise<SimpleGenerateResponse> {
  const form = new FormData();
  form.append("current_prompt", params.current_prompt);
  if (params.previous_prompt) {
    form.append("previous_prompt", params.previous_prompt);
  }
  form.append("num_images", String(params.num_images ?? 4));
  if (typeof params.seed === "number") {
    form.append("seed", String(params.seed));
  }
  form.append("width", String(params.width ?? 512));
  form.append("height", String(params.height ?? 512));
  form.append("guidance_scale", String(params.guidance_scale ?? 4.5));
  form.append(
    "num_inference_steps",
    String(params.num_inference_steps ?? 50)
  );

  // Convert data URL to Blob via fetch (works in browser)
  const blob = await fetch(params.imageDataUrl).then((r) => r.blob());
  const file = new File([blob], "input.png", { type: blob.type || "image/png" });
  form.append("image", file);

  const res = await fetch(`${SIMPLE_PIXART_API_BASE_URL}/generate_with_image`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(`generateWithImage failed: ${res.status}`);
  }

  const data = (await res.json()) as SimpleGenerateResponse;
  if (!Array.isArray(data.images)) {
    throw new Error("Invalid response from simple PixArt API: images missing");
  }

  return data;
}

