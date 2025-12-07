import os
import io
import uuid
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from i2i.server.engine import (  # type: ignore
    start_session_engine,
    apply_intervention_engine,
    step_once_engine,
    run_to_end_engine,
    select_branch_engine,
    fork_current_engine,
    fork_at_step_engine,
    backtrack_to_engine,
    merge_branches_engine,
)


# ============ Helpers ============ #
def _img_to_base64_png(img: Optional[Image.Image]) -> Optional[str]:
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _parse_bool(x: Optional[str]) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "on")


# In-memory session store: session_id -> state dict
SESSIONS: Dict[str, Dict[str, Any]] = {}


# ============ FastAPI App ============ #
app = FastAPI(title="I2I Interactive API")

# CORS (adjust origins via env if needed)
origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models for JSON endpoints ---------- #
class StepReq(BaseModel):
    session_id: str
    branch_id: str


class RunReq(BaseModel):
    session_id: str
    branch_id: str


class SelectBranchReq(BaseModel):
    session_id: str
    branch_id: str


class ForkAtReq(BaseModel):
    session_id: str
    branch_id: str
    step_index: int


class BacktrackReq(BaseModel):
    session_id: str
    branch_id: str
    step_index: int


class MergeBranchesReq(BaseModel):
    session_id: str
    branch_id_1: str
    branch_id_2: str
    step_index_1: int  # Step to use from branch_1
    step_index_2: Optional[int] = None  # Step to use from branch_2 (defaults to step_index_1)
    merge_weight: float = 0.5  # Weight for branch_1's latent (0.5 = equal blend)


class SaveSessionReq(BaseModel):
    mode: str
    participant: int
    graphSession: Dict[str, Any]
    bookmarkedNodeIds: Optional[List[str]] = []


# ---------- Compatibility: health ---------- #
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running"}


# ---------- Compatibility: object list ---------- #
@app.post("/api/composition/objects")
async def create_object_list(payload: Dict[str, Any]):
    """
    Extract objects from prompt.
    Returns: {"objects": [{"id": str, "label": str, "color": str}, ...]}
    Mirrors the old backend contract for the React frontend.
    """
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"objects": []})

    # simple keyword split, limit to top N
    words = [w for w in prompt.lower().split() if len(w) > 2][:5]
    colors = [
        "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#ef4444",
        "#f59e0b", "#eab308", "#84cc16", "#22c55e", "#10b981",
    ]
    import time
    objects: List[Dict[str, str]] = []
    if len(words) == 0:
        objects.append({
            "id": f"obj_{int(time.time() * 1000)}_0",
            "label": "Object",
            "color": colors[0],
        })
    else:
        for i, w in enumerate(words):
            objects.append({
                "id": f"obj_{int(time.time() * 1000)}_{i}",
                "label": w.capitalize(),
                "color": colors[i % len(colors)],
            })
    return {"objects": objects}


# ---------- Session Start (multipart) ---------- #
@app.post("/api/session/start")
async def start_session(
    prompt: str = Form(...),
    steps: int = Form(50),
    seed: int = Form(67),
    model_version: str = Form("512"),
    gpu_id: int = Form(0),
    guidance_scale: float = Form(4.5),
    enable_layout: str = Form("false"),
    layout_json: Optional[str] = Form(None),
    enable_edge: str = Form("false"),
    edge_phrases_text: Optional[str] = Form(None),
    edge_files: Optional[List[UploadFile]] = File(None),
):
    try:
        layout_items: Optional[List[Dict[str, Any]]] = None
        if layout_json:
            import json as _json
            try:
                parsed = _json.loads(layout_json)
                if isinstance(parsed, list):
                    layout_items = parsed
            except Exception:
                layout_items = None
        print(edge_files)
        # Load edge images (if any)
        edge_images: Optional[List[Image.Image]] = None
        if edge_files:
            imgs: List[Image.Image] = []
            for uf in edge_files:
                try:
                    data = await uf.read()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    imgs.append(img)
                except Exception:
                    continue
            if len(imgs) > 0:
                edge_images = imgs

        state, status = start_session_engine(
            prompt=prompt,
            steps=int(steps),
            seed=int(seed),
            model_version=model_version,
            gpu_id=int(gpu_id),
            guidance_scale=float(guidance_scale),
            enable_layout=_parse_bool(enable_layout),
            layout_items=layout_items,
            enable_edge=_parse_bool(enable_edge),
            edge_images=edge_images,
            edge_phrases_text=edge_phrases_text,
        )
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = state
        active_branch_id = state.get("active_branch_id", "B0")
        i = int(state["branches"][active_branch_id]["i"])
        num_steps = int(state["num_steps"])
        return JSONResponse(
            {
                "session_id": session_id,
                "status": status,
                "branches": list(state.get("branches", {}).keys()),
                "active_branch_id": active_branch_id,
                "i": i,
                "num_steps": num_steps,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Compatibility: composition start ---------- #
@app.post("/api/composition/start")
async def start_image_generation(request: Request, sketch: UploadFile | None = File(None)):
    """
    Compatibility endpoint for the existing frontend.
    Accepts JSON or multipart just like the legacy server, then internally
    initializes a session using start_session_engine and returns:
      {"sessionId": str, "rootNodeId": str, "websocketUrl": str}
    """
    content_type = request.headers.get("content-type", "")
    prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    layout_items: Optional[List[Dict[str, Any]]] = None
    edge_images: Optional[List[Image.Image]] = None
    enable_edge: bool = False

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            prompt = (form.get("prompt") or "").strip()
            raw_objects = form.get("objects")
            raw_bboxes = form.get("bboxes")
            width = int(form.get("width") or 512)
            height = int(form.get("height") or 512)
            num_inference_steps = int(form.get("num_inference_steps") or 50)

            import json as _json
            # Build layout_items from bboxes (normalize to x0,y0,x1,y1)
            # and map objectId -> phrase label when available
            labels_by_id: Dict[str, str] = {}
            try:
                if raw_objects:
                    objs = _json.loads(raw_objects)
                    if isinstance(objs, list):
                        for o in objs:
                            if isinstance(o, dict) and "id" in o and "label" in o:
                                labels_by_id[str(o["id"])] = str(o["label"])
            except Exception:
                pass
            try:
                if raw_bboxes:
                    bboxes = _json.loads(raw_bboxes)
                    if isinstance(bboxes, list) and len(bboxes) > 0:
                        layout_items = []
                        for bb in bboxes:
                            if not isinstance(bb, dict):
                                continue
                            obj_id = str(bb.get("objectId", ""))
                            phrase = labels_by_id.get(obj_id, "object")
                            x = float(bb.get("x", 0.0))
                            y = float(bb.get("y", 0.0))
                            w = float(bb.get("width", 1.0))
                            h = float(bb.get("height", 1.0))
                            layout_items.append({
                                "phrase": phrase,
                                "x0": x,
                                "y0": y,
                                "x1": x + w,
                                "y1": y + h,
                            })
            except Exception:
                layout_items = None
            # Handle optional sketch file as edge image
            if sketch is not None:
                try:
                    data = await sketch.read()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    edge_images = [img]
                    enable_edge = True
                except Exception:
                    edge_images = None
                    enable_edge = False
        else:
            body = await request.json()
            prompt = str(body.get("prompt") or "").strip()
            width = int(body.get("width") or 512)
            height = int(body.get("height") or 512)
            num_inference_steps = int(body.get("num_inference_steps") or 50)
            # objects/bboxes optional
            objs = body.get("objects") or []
            bbs = body.get("bboxes") or []
            labels_by_id = {}
            try:
                for o in objs:
                    if isinstance(o, dict) and "id" in o and "label" in o:
                        labels_by_id[str(o["id"])] = str(o["label"])
            except Exception:
                pass
            try:
                if isinstance(bbs, list) and len(bbs) > 0:
                    layout_items = []
                    for bb in bbs:
                        if not isinstance(bb, dict):
                            continue
                        obj_id = str(bb.get("objectId", ""))
                        phrase = labels_by_id.get(obj_id, "object")
                        x = float(bb.get("x", 0.0))
                        y = float(bb.get("y", 0.0))
                        w = float(bb.get("width", 1.0))
                        h = float(bb.get("height", 1.0))
                        layout_items.append({
                            "phrase": phrase,
                            "x0": x,
                            "y0": y,
                            "x1": x + w,
                            "y1": y + h,
                        })
            except Exception:
                layout_items = None
    except Exception:
        return JSONResponse({"error": "Invalid request payload"}, status_code=400)

    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)

    # Initialize via internal engine
    try:
        state, _ = start_session_engine(
            prompt=prompt,
            steps=num_inference_steps,
            seed=67,
            model_version="512",
            gpu_id=0,
            guidance_scale=4.5,
            enable_layout=bool(layout_items and len(layout_items) > 0),
            layout_items=layout_items,
            enable_edge=enable_edge,
            edge_images=edge_images,
            # For now, use the whole prompt as a single phrase for edge guidance when sketch is provided
            edge_phrases_text=prompt if enable_edge else None,
        )
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = state
        # Synthesize a prompt node id for frontend graph
        root_node_id = f"node_prompt_{session_id}"
        # For legacy compatibility we include a websocketUrl placeholder
        websocket_url = ""
        return {
            "sessionId": session_id,
            "rootNodeId": root_node_id,
            "websocketUrl": websocket_url,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Step once (JSON) ---------- #
@app.post("/api/session/step")
def step_once(req: StepReq):
    try:
        state = SESSIONS.get(req.session_id)
        if state is None:
            return JSONResponse({"error": "Invalid session_id"}, status_code=404)
        state, preview, status, gallery = step_once_engine(state, req.branch_id)
        SESSIONS[req.session_id] = state
        preview_b64 = _img_to_base64_png(preview)
        return JSONResponse(
            {
                "branch_id": req.branch_id,
                "i": int(state["branches"][req.branch_id]["i"]),
                "num_steps": int(state["num_steps"]),
                "preview_png_base64": preview_b64,
                "gallery_len": len(gallery),
                "status": status,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Apply intervention (multipart) ---------- #
@app.post("/api/session/apply")
async def apply_intervention(
    session_id: str = Form(...),
    branch_id: str = Form(""),
    intervene_choice: str = Form(...),  # "Continue" | "Text Guidance" | "Style Guidance"
    text_input: Optional[str] = Form(None),
    text_scale: Optional[float] = Form(None),
    text_x0: Optional[float] = Form(None),
    text_y0: Optional[float] = Form(None),
    text_x1: Optional[float] = Form(None),
    text_y1: Optional[float] = Form(None),
    style_scale: Optional[float] = Form(None),
    style_x0: Optional[float] = Form(None),
    style_y0: Optional[float] = Form(None),
    style_x1: Optional[float] = Form(None),
    style_y1: Optional[float] = Form(None),
    style_file: Optional[UploadFile] = File(None),
):
    state = SESSIONS.get(session_id)
    if state is None:
        return JSONResponse({"error": "Invalid session_id"}, status_code=404)

    # Default to active branch when not specified
    if not branch_id:
        branch_id = str(state.get("active_branch_id", "B0"))

    # Persist reference image to a temp path if provided
    style_path: Optional[str] = None
    if style_file is not None:
        data = await style_file.read()
        tmp_dir = os.environ.get("I2I_TMP_DIR", "/tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        style_path = os.path.join(tmp_dir, f"ref_{uuid.uuid4().hex}.png")
        with open(style_path, "wb") as f:
            f.write(data)

    t_region: Optional[Tuple[float, float, float, float]] = None
    s_region: Optional[Tuple[float, float, float, float]] = None
    if all(v is not None for v in (text_x0, text_y0, text_x1, text_y1)):
        t_region = (float(text_x0), float(text_y0), float(text_x1), float(text_y1))  # type: ignore
    if all(v is not None for v in (style_x0, style_y0, style_x1, style_y1)):
        s_region = (float(style_x0), float(style_y0), float(style_x1), float(style_y1))  # type: ignore

    print(f"t_region: {t_region}, s_region: {s_region}")
    print(f"text_input: {text_input}, style_file_path: {style_path}")
    print(f"text_scale: {text_scale}, style_scale: {style_scale}")
    print(f"intervene_choice: {intervene_choice}")
    state, msg = apply_intervention_engine(
        state=state,
        branch_id=branch_id,
        intervene_choice=intervene_choice,
        text_input=text_input,
        style_file_path=style_path,
        text_scale=text_scale,
        text_region=t_region,
        style_scale=style_scale,
        style_region=s_region,
    )
    SESSIONS[session_id] = state
    return JSONResponse({"status": msg})


# ---------- Run to end (JSON) ---------- #
@app.post("/api/session/run_to_end")
def run_to_end(req: RunReq):
    try:
        state = SESSIONS.get(req.session_id)
        if state is None:
            return JSONResponse({"error": "Invalid session_id"}, status_code=404)
        state, preview, status, gallery = run_to_end_engine(state, req.branch_id)
        SESSIONS[req.session_id] = state
        preview_b64 = _img_to_base64_png(preview)
        return JSONResponse(
            {
                "branch_id": req.branch_id,
                "i": int(state["branches"][req.branch_id]["i"]),
                "num_steps": int(state["num_steps"]),
                "preview_png_base64": preview_b64,
                "gallery_len": len(gallery),
                "status": status,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Branching & navigation ---------- #
@app.post("/api/session/fork-current")
def fork_current(req: RunReq):
    try:
        state = SESSIONS.get(req.session_id)
        if state is None:
            return JSONResponse({"error": "Invalid session_id"}, status_code=404)
        before_ids = set((state.get("branches") or {}).keys())
        state, msg = fork_current_engine(state, req.branch_id)
        SESSIONS[req.session_id] = state
        after_ids = set((state.get("branches") or {}).keys())
        new_ids = list(after_ids - before_ids)
        new_branch_id = new_ids[0] if len(new_ids) == 1 else None
        return JSONResponse(
            {
                "status": msg,
                "branches": list(state.get("branches", {}).keys()),
                "active_branch_id": state.get("active_branch_id"),
                "new_branch_id": new_branch_id,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/session/fork-at-step")
def fork_at_step(req: ForkAtReq):
    try:
        state = SESSIONS.get(req.session_id)
        if state is None:
            return JSONResponse({"error": "Invalid session_id"}, status_code=404)
        before_ids = set((state.get("branches") or {}).keys())
        state, msg = fork_at_step_engine(state, req.branch_id, int(req.step_index))
        SESSIONS[req.session_id] = state
        after_ids = set((state.get("branches") or {}).keys())
        new_ids = list(after_ids - before_ids)
        new_branch_id = new_ids[0] if len(new_ids) == 1 else None
        return JSONResponse(
            {
                "status": msg,
                "branches": list(state.get("branches", {}).keys()),
                "active_branch_id": state.get("active_branch_id"),
                "new_branch_id": new_branch_id,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/session/backtrack-to")
def backtrack_to(req: BacktrackReq):
    try:
        state = SESSIONS.get(req.session_id)
        if state is None:
            return JSONResponse({"error": "Invalid session_id"}, status_code=404)
        state, msg, gallery, preview = backtrack_to_engine(state, req.branch_id, int(req.step_index))
        SESSIONS[req.session_id] = state
        return JSONResponse(
            {
                "status": msg,
                "branches": list(state.get("branches", {}).keys()),
                "active_branch_id": state.get("active_branch_id"),
                "gallery_len": len(gallery),
                "preview_png_base64": _img_to_base64_png(preview),
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/session/select-branch")
def select_branch(req: SelectBranchReq):
    try:
        state = SESSIONS.get(req.session_id)
        if state is None:
            return JSONResponse({"error": "Invalid session_id"}, status_code=404)
        state, preview, gallery, msg = select_branch_engine(state, req.branch_id)
        SESSIONS[req.session_id] = state
        return JSONResponse(
            {
                "status": msg,
                "active_branch_id": req.branch_id,
                "i": int(state["branches"][req.branch_id]["i"]),
                "num_steps": int(state["num_steps"]),
                "gallery_len": len(gallery),
                "preview_png_base64": _img_to_base64_png(preview),
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/session/merge-branches")
def merge_branches(req: MergeBranchesReq):
    """
    Merge two branches, allowing different steps for each branch.
    
    1. Takes latent from branch_1 at step_index_1
    2. Takes latent from branch_2 at step_index_2 (or step_index_1 if not specified)
    3. Creates a weighted average of the latents
    4. Starts the new branch from the later of the two steps
    5. Stores both source latents for extended attention during denoising
    
    This is triggered when user drags an intermediate node from one branch
    to another node (can be at a different step).
    """
    try:
        state = SESSIONS.get(req.session_id)
        if state is None:
            return JSONResponse({"error": "Invalid session_id"}, status_code=404)
        
        state, msg, new_branch_id = merge_branches_engine(
            state=state,
            branch_id_1=req.branch_id_1,
            branch_id_2=req.branch_id_2,
            step_index_1=int(req.step_index_1),
            step_index_2=int(req.step_index_2) if req.step_index_2 is not None else None,
            merge_weight=float(req.merge_weight),
        )
        SESSIONS[req.session_id] = state
        
        if new_branch_id is None:
            return JSONResponse(
                {
                    "status": msg,
                    "error": msg,
                    "branches": list(state.get("branches", {}).keys()),
                    "active_branch_id": state.get("active_branch_id"),
                    "new_branch_id": None,
                },
                status_code=400,
            )
        
        # Get info about the new merged branch
        br = state["branches"].get(new_branch_id, {})
        merge_info = br.get("merge_source_latents", {})
        
        return JSONResponse(
            {
                "status": msg,
                "branches": list(state.get("branches", {}).keys()),
                "active_branch_id": state.get("active_branch_id"),
                "new_branch_id": new_branch_id,
                "i": int(br.get("i", 0)),
                "num_steps": int(state["num_steps"]),
                "merged_from": [req.branch_id_1, req.branch_id_2],
                "merge_steps": {
                    "branch_1": merge_info.get("step_1", req.step_index_1),
                    "branch_2": merge_info.get("step_2", req.step_index_2 or req.step_index_1),
                    "start_step": merge_info.get("merge_start_step", br.get("i", 0)),
                },
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Session save/load ---------- #
def _get_logs_dir() -> Path:
    """Get the logs directory path."""
    logs_dir = Path(os.environ.get("I2I_LOGS_DIR", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


@app.post("/api/session/save")
async def save_session(req: SaveSessionReq):
    """
    Save a graph session to disk.
    Saves to: logs/{mode}/p{participant}/session_{timestamp}.json
    """
    try:
        logs_dir = _get_logs_dir()
        mode_dir = logs_dir / req.mode
        participant_dir = mode_dir / f"p{req.participant}"
        participant_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"session_{timestamp}.json"
        filepath = participant_dir / filename
        
        # Prepare data with lastUpdated timestamp
        data = {
            "graphSession": req.graphSession,
            "lastUpdated": datetime.now().isoformat() + "Z",
            "bookmarkedNodeIds": req.bookmarkedNodeIds or [],
        }
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return JSONResponse({
            "status": "ok",
            "message": f"Session saved to {filepath}",
            "filepath": str(filepath),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/session/load")
async def load_session(mode: str, p: int):
    """
    Load the latest graph session from disk.
    Returns the most recent session file for the given mode and participant.
    """
    try:
        logs_dir = _get_logs_dir()
        participant_dir = logs_dir / mode / f"p{p}"
        
        if not participant_dir.exists():
            return JSONResponse({"error": "No sessions found"}, status_code=404)
        
        # Find all session files and get the most recent one
        session_files = list(participant_dir.glob("session_*.json"))
        if not session_files:
            return JSONResponse({"error": "No sessions found"}, status_code=404)
        
        # Sort by modification time (most recent first)
        session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_file = session_files[0]
        
        # Read and return the session data
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Mount existing Gradio UI under /gradio ---------- #
try:
    import gradio as gr  # type: ignore
    from i2i import gradio_app as _grapp  # type: ignore
    demo = _grapp.build_ui()
    app = gr.mount_gradio_app(app, demo, path="/gradio")
except Exception:
    # If Gradio isn't available, we keep only REST API
    pass

