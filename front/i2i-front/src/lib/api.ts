import { API_BASE_URL } from "../config/api";
import type { GraphSession } from "../types";

export type LayoutItem = {
  phrase: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

export type StartSessionResponse = {
  session_id: string;
  status: string;
  branches: string[];
  active_branch_id: string;
  i: number;
  num_steps: number;
};

export async function startSession(params: {
  prompt: string;
  steps?: number;
  seed?: number;
  model_version?: "512" | "1024";
  gpu_id?: number;
  guidance_scale?: number;
  enable_layout?: boolean;
  layout_items?: LayoutItem[];
  enable_edge?: boolean;
  edge_phrases_text?: string;
  edge_files?: File[];
}): Promise<StartSessionResponse> {
  const form = new FormData();
  form.append("prompt", params.prompt);
  form.append("steps", String(params.steps ?? 50));
  form.append("seed", String(params.seed ?? 67));
  form.append("model_version", String(params.model_version ?? "512"));
  form.append("gpu_id", String(params.gpu_id ?? 0));
  form.append("guidance_scale", String(params.guidance_scale ?? 4.5));
  form.append("enable_layout", String(!!params.enable_layout));
  if (params.layout_items && params.layout_items.length > 0) {
    form.append("layout_json", JSON.stringify(params.layout_items));
  }
  form.append("enable_edge", String(!!params.enable_edge));
  if (params.edge_phrases_text) {
    form.append("edge_phrases_text", params.edge_phrases_text);
  }
  (params.edge_files ?? []).forEach((f) => form.append("edge_files", f));

  const res = await fetch(`${API_BASE_URL}/api/session/start`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error(`startSession failed: ${res.status}`);
  }
  return await res.json();
}

export async function stepOnce(params: {
  session_id: string;
  branch_id: string;
}): Promise<{
  branch_id: string;
  i: number;
  num_steps: number;
  preview_png_base64?: string | null;
  gallery_len: number;
  status: string;
}> {
  const res = await fetch(`${API_BASE_URL}/api/session/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`stepOnce failed: ${res.status}`);
  }
  return await res.json();
}

export async function runToEnd(params: {
  session_id: string;
  branch_id: string;
}): Promise<{
  branch_id: string;
  i: number;
  num_steps: number;
  preview_png_base64?: string | null;
  gallery_len: number;
  status: string;
}> {
  const res = await fetch(`${API_BASE_URL}/api/session/run_to_end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`runToEnd failed: ${res.status}`);
  }
  return await res.json();
}

/**
 * SSE event types for run_to_end_stream
 */
export type RunToEndStreamEvent = 
  | { type: "start"; current_step: number; num_steps: number; branch_id: string }
  | { type: "step"; i: number; num_steps: number; branch_id: string; preview_png_base64: string | null; status: string }
  | { type: "complete"; branch_id: string; num_steps: number }
  | { type: "error"; error: string };

/**
 * Run to end with streaming - returns each step's image via SSE
 * @param params session_id and branch_id
 * @param onStep callback called for each step with the step data
 * @param onComplete callback called when generation is complete
 * @param onError callback called on error
 * @returns AbortController to cancel the stream
 */
export function runToEndStream(
  params: { session_id: string; branch_id: string },
  onStep: (data: { i: number; num_steps: number; branch_id: string; preview_png_base64: string | null; status: string }) => void,
  onComplete: (data: { branch_id: string; num_steps: number }) => void,
  onError: (error: string) => void
): AbortController {
  const controller = new AbortController();
  const url = `${API_BASE_URL}/api/session/run_to_end_stream?session_id=${encodeURIComponent(params.session_id)}&branch_id=${encodeURIComponent(params.branch_id)}`;
  
  fetch(url, { signal: controller.signal })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`runToEndStream failed: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }
      
      const decoder = new TextDecoder();
      let buffer = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete SSE events
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6)) as RunToEndStreamEvent;
              
              if ("error" in data) {
                onError(data.error);
                return;
              }
              
              if (data.type === "step") {
                onStep({
                  i: data.i,
                  num_steps: data.num_steps,
                  branch_id: data.branch_id,
                  preview_png_base64: data.preview_png_base64,
                  status: data.status,
                });
              } else if (data.type === "complete") {
                onComplete({
                  branch_id: data.branch_id,
                  num_steps: data.num_steps,
                });
              }
              // Ignore "start" events for now
            } catch (e) {
              console.warn("[runToEndStream] Failed to parse SSE event:", line, e);
            }
          }
        }
      }
    })
    .catch((error) => {
      if (error.name !== "AbortError") {
        console.error("[runToEndStream] Error:", error);
        onError(error.message || String(error));
      }
    });
  
  return controller;
}

export async function applyGuidance(params: {
  session_id: string;
  branch_id: string;
  intervene_choice: "Continue" | "Text Guidance" | "Style Guidance";
  text_input?: string;
  text_scale?: number;
  text_region?: { x0: number; y0: number; x1: number; y1: number };
  style_scale?: number;
  style_region?: { x0: number; y0: number; x1: number; y1: number };
  style_file?: File;
}): Promise<{ status: string }> {
  const form = new FormData();
  form.append("session_id", params.session_id);
  form.append("branch_id", params.branch_id);
  form.append("intervene_choice", params.intervene_choice);
  if (params.text_input) form.append("text_input", params.text_input);
  if (typeof params.text_scale === "number") {
    form.append("text_scale", String(params.text_scale));
  }
  if (params.text_region) {
    form.append("text_x0", String(params.text_region.x0));
    form.append("text_y0", String(params.text_region.y0));
    form.append("text_x1", String(params.text_region.x1));
    form.append("text_y1", String(params.text_region.y1));
  }
  if (typeof params.style_scale === "number") {
    form.append("style_scale", String(params.style_scale));
  }
  if (params.style_region) {
    form.append("style_x0", String(params.style_region.x0));
    form.append("style_y0", String(params.style_region.y0));
    form.append("style_x1", String(params.style_region.x1));
    form.append("style_y1", String(params.style_region.y1));
  }
  if (params.style_file) form.append("style_file", params.style_file);

  const res = await fetch(`${API_BASE_URL}/api/session/apply`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error(`applyGuidance failed: ${res.status}`);
  }
  return await res.json();
}

export async function forkCurrent(params: {
  session_id: string;
  branch_id: string;
}): Promise<{ status: string; branches: string[]; active_branch_id: string }> {
  const res = await fetch(`${API_BASE_URL}/api/session/fork-current`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`forkCurrent failed: ${res.status}`);
  }
  return await res.json();
}

export async function forkAtStep(params: {
  session_id: string;
  branch_id: string;
  step_index: number;
}): Promise<{ status: string; branches: string[]; active_branch_id: string; new_branch_id?: string | null }> {
  const res = await fetch(`${API_BASE_URL}/api/session/fork-at-step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`forkAtStep failed: ${res.status}`);
  }
  return await res.json();
}

export async function backtrackTo(params: {
  session_id: string;
  branch_id: string;
  step_index: number;
}): Promise<{
  status: string;
  branches: string[];
  active_branch_id: string;
  gallery_len: number;
  preview_png_base64?: string | null;
}> {
  const res = await fetch(`${API_BASE_URL}/api/session/backtrack-to`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`backtrackTo failed: ${res.status}`);
  }
  return await res.json();
}

export async function selectBranch(params: {
  session_id: string;
  branch_id: string;
}): Promise<{
  status: string;
  active_branch_id: string;
  i: number;
  num_steps: number;
  gallery_len: number;
  preview_png_base64?: string | null;
}> {
  const res = await fetch(`${API_BASE_URL}/api/session/select-branch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`selectBranch failed: ${res.status}`);
  }
  return await res.json();
}

/**
 * Merge two branches, allowing different steps for each branch.
 * 
 * 1. Takes latent from branch_1 at step_index_1
 * 2. Takes latent from branch_2 at step_index_2 (or step_index_1 if not specified)
 * 3. Creates a weighted average of the latents
 * 4. Starts the new branch from the later of the two steps
 * 5. Stores both source latents for extended attention during denoising
 * 
 * This is triggered when user drags an intermediate node from one branch
 * to another node (can be at a different step).
 */
export type MergeBranchesResponse = {
  status: string;
  branches: string[];
  active_branch_id: string;
  new_branch_id: string | null;
  i?: number;
  num_steps?: number;
  merged_from?: [string, string];
  merge_steps?: {
    branch_1: number;
    branch_2: number;
    start_step: number;
  };
  error?: string;
};

export async function mergeBranches(params: {
  session_id: string;
  branch_id_1: string;
  branch_id_2: string;
  step_index_1: number;  // Step to use from branch_1
  step_index_2?: number; // Step to use from branch_2 (defaults to step_index_1)
  merge_weight?: number; // Weight for branch_1's latent (0.5 = equal blend)
  source_session_id?: string; // Optional: when provided, branch_2 is taken from this session
}): Promise<MergeBranchesResponse> {
  const res = await fetch(`${API_BASE_URL}/api/session/merge-branches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: params.session_id,
      branch_id_1: params.branch_id_1,
      branch_id_2: params.branch_id_2,
      step_index_1: params.step_index_1,
      step_index_2: params.step_index_2 ?? null,
      merge_weight: params.merge_weight ?? 0.5,
      source_session_id: params.source_session_id ?? null,
    }),
  });
  const data = await res.json();
  if (!res.ok && !data.new_branch_id) {
    throw new Error(data.error || `mergeBranches failed: ${res.status}`);
  }
  return data;
}

/**
 * Save a graph session to the server
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
  const res = await fetch(`${API_BASE_URL}/api/session/save`, {
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
 * Load the latest graph session from the server
 * @param mode - Mode string ("step" or "prompt")
 * @param participant - Participant number
 * @returns GraphSession data with lastUpdated timestamp and bookmarkedNodeIds, or null if not found
 */
export async function loadSession(
  mode: string,
  participant: number
): Promise<{ graphSession: GraphSession; lastUpdated: string; bookmarkedNodeIds?: string[] } | null> {
  const url = `${API_BASE_URL}/api/session/load?mode=${encodeURIComponent(mode)}&p=${encodeURIComponent(participant)}`;
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
