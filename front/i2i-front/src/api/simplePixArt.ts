import { API_BASE_URL } from "../config/api";
import type { GraphSession } from "../types";

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
  // Single-image path (legacy)
  imageDataUrl?: string; // data URL (e.g., data:image/png;base64,...)
  // Multi-image path
  imageDataUrls?: string[]; // array of data URLs
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

  // Normalize to an array of data URLs (support single and multiple images)
  const urls: string[] =
    params.imageDataUrls && params.imageDataUrls.length
      ? params.imageDataUrls
      : params.imageDataUrl
      ? [params.imageDataUrl]
      : [];

  if (!urls.length) {
    throw new Error("generateWithImage requires at least one imageDataUrl");
  }

  // Convert data URLs to Blobs via fetch (works in browser)
  const blobs = await Promise.all(urls.map((u) => fetch(u).then((r) => r.blob())));

  // First image sent under \"image\" for backward compatibility
  const primaryBlob = blobs[0];
  const primaryFile = new File([primaryBlob], "input-0.png", {
    type: primaryBlob.type || "image/png",
  });
  form.append("image", primaryFile);

  // Remaining images (if any) sent under \"images\"
  for (let i = 1; i < blobs.length; i++) {
    const blob = blobs[i];
    const file = new File([blob], `input-${i}.png`, {
      type: blob.type || "image/png",
    });
    form.append("images", file);
  }

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

/**
 * Save a graph session to the server (for prompt mode)
 * @param mode - Mode string ("step" or "prompt")
 * @param participant - Participant number
 * @param graphSession - GraphSession to save
 * @param bookmarkedNodeIds - Array of bookmarked node IDs
 */
export async function saveSession(
  mode: string,
  participant: number,
  graphSession: GraphSession,
  bookmarkedNodeIds?: string[]
): Promise<{ status: string; message?: string }> {
  const res = await fetch(`${SIMPLE_PIXART_API_BASE_URL}/api/session/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode,
      participant,
      graphSession,
      bookmarkedNodeIds: bookmarkedNodeIds || [],
    }),
  });
  if (!res.ok) {
    throw new Error(`saveSession failed: ${res.status}`);
  }
  return await res.json();
}

/**
 * Load the latest graph session from the server (for prompt mode)
 * @param mode - Mode string ("step" or "prompt")
 * @param participant - Participant number
 * @returns GraphSession data with lastUpdated timestamp and bookmarkedNodeIds, or null if not found
 */
export async function loadSession(
  mode: string,
  participant: number
): Promise<{ graphSession: GraphSession; lastUpdated: string; bookmarkedNodeIds?: string[] } | null> {
  const url = `${SIMPLE_PIXART_API_BASE_URL}/api/session/load?mode=${encodeURIComponent(mode)}&p=${encodeURIComponent(participant)}`;
  const res = await fetch(url, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    if (res.status === 404) {
      return null; // No session found
    }
    throw new Error(`loadSession failed: ${res.status}`);
  }
  return await res.json();
}

