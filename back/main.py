"""
FastAPI Backend Server
"""
import os
import sys
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image

try:
    from .models import (
        ImageGenerationRequest,
        ObjectChip,
        BranchCreateRequest,
        WebSocketMessage,
        WebSocketMessageType,
    )
    from .session_manager import SessionManager, SessionStatus
    from .pipeline_manager import PipelineManager
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from models import (
        ImageGenerationRequest,
        ObjectChip,
        BranchCreateRequest,
        WebSocketMessage,
        WebSocketMessageType,
    )
    from session_manager import SessionManager, SessionStatus
    from pipeline_manager import PipelineManager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = FastAPI(title="Interaction-to-Image Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    import sys
    sys.stdout.flush()
    print("=" * 80, flush=True)
    print(f"[Middleware] {request.method} {request.url}", flush=True)
    print(f"[Middleware] Client: {request.client}", flush=True)
    print("=" * 80, flush=True)
    try:
        response = await call_next(request)
        print(f"[Middleware] Status: {response.status_code}", flush=True)
        return response
    except Exception as e:
        print(f"[Middleware] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

session_manager = SessionManager()
pipeline_manager = PipelineManager()

images_dir = os.path.join(os.path.dirname(__file__), "generated_images")
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")
print(f"[Server] Static files mounted: {images_dir} -> /images", flush=True)

@app.on_event("startup")
async def startup_event():
    import sys
    sys.stdout.flush()
    print("[Server] Starting server...", flush=True)
    try:
        pipeline_manager.load_model()
        print("[Server] Model loaded", flush=True)
    except Exception as e:
        print(f"[Server] Model load failed: {e}", flush=True)
        import traceback
        traceback.print_exc()


@app.get("/")
async def root():
    return {"message": "Interaction-to-Image Backend API"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running"}


@app.post("/api/composition/objects")
async def create_object_list(request: dict):
    """
    Extract objects from prompt.
    Returns: {"objects": [{"id": str, "label": str, "color": str}, ...]}
    """
    prompt = request.get("prompt", "")
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    print(f"[API] Object list request: '{prompt}'")
    
    keywords = prompt.lower().split()
    keywords = [w for w in keywords if len(w) > 2][:5]
    
    colors = [
        "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#ef4444",
        "#f59e0b", "#eab308", "#84cc16", "#22c55e", "#10b981",
    ]
    
    import time
    objects = []
    for i, keyword in enumerate(keywords):
        objects.append({
            "id": f"obj_{int(time.time() * 1000)}_{i}",
            "label": keyword.capitalize(),
            "color": colors[i % len(colors)],
        })
    
    if not objects:
        objects = [
            {"id": f"obj_{int(time.time() * 1000)}_0", "label": "Object", "color": colors[0]}
        ]
    
    print(f"[API] Objects created: {len(objects)}")
    return {"objects": objects}


@app.post("/api/composition/start")
async def start_image_generation(request: ImageGenerationRequest):
    """
    Start image generation session.
    Returns: {"sessionId": str, "rootNodeId": str, "websocketUrl": str}
    """
    print(f"[API] Start generation: prompt='{request.prompt}', objects={len(request.objects) if request.objects else 0}, bboxes={len(request.bboxes) if request.bboxes else 0}")
    
    session = session_manager.create_session(
        prompt=request.prompt,
        objects=request.objects or [],
        bboxes=request.bboxes or [],
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
    )
    
    websocket_url = f"ws://localhost:8001/ws/image-stream/{session.id}"
    
    print(f"[API] Session created: {session.id}, rootNodeId: {session.root_node_id}")
    
    return {
        "sessionId": session.id,
        "rootNodeId": session.root_node_id,
        "websocketUrl": websocket_url,
    }


@app.websocket("/ws/image-stream/{session_id}")
async def websocket_image_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for image streaming (deprecated - use websocket_server.py instead).
    This endpoint is kept for compatibility but not actively used.
    """
    await websocket.accept()
    await websocket.send_json({
        "type": "error",
        "message": "This endpoint is deprecated. Use websocket_server on port 8001.",
    })
    await websocket.close()


@app.post("/api/branch/create")
async def create_branch(request: BranchCreateRequest):
    """
    Create a new branch from a source node with feedback.
    Returns: {"branchId": str, "websocketUrl": str}
    """
    print(f"[API] Branch create: session_id={request.sessionId}, source_node_id={request.sourceNodeId}")
    
    session = session_manager.get_session(request.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    branch = session_manager.create_branch(
        session_id=request.sessionId,
        source_node_id=request.sourceNodeId,
        feedback=request.feedback,
    )
    
    websocket_url = f"ws://localhost:8001/ws/image-stream/{request.sessionId}/{branch.id}"
    
    print(f"[API] Branch created: {branch.id}")
    
    return {
        "branchId": branch.id,
        "websocketUrl": websocket_url,
    }


if __name__ == "__main__":
    import sys
    import threading
    import asyncio
    
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    print("[Main] Starting server...", flush=True)
    print("[Main] HTTP port: 8000, WebSocket port: 8001", flush=True)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from main import startup_event
        loop.run_until_complete(startup_event())
    except Exception as e:
        print(f"[Main] Startup failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    loop.close()
    
    def run_websocket_server():
        from websocket_server import start_websocket_server
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(start_websocket_server(host="0.0.0.0", port=8001))
    
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()
    print("[Main] WebSocket server started", flush=True)
    
    import time
    time.sleep(0.5)
    
    from http_fastapi_server import FastAPIHTTPHandler
    from http.server import HTTPServer
    
    PORT = 8000
    
    print(f"[Main] HTTP server running on port {PORT}", flush=True)
    
    with HTTPServer(("0.0.0.0", PORT), FastAPIHTTPHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Main] Shutting down...", flush=True)
            httpd.shutdown()
