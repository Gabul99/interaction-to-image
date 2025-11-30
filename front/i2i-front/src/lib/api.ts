import { API_BASE_URL } from "../config/api";

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


