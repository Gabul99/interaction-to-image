"""
WebSocket Server - Handles WebSocket connections for image streaming
"""
import asyncio
import sys
import json
import os
import queue
from PIL import Image
from websockets.server import serve as websocket_serve
from websockets.exceptions import ConnectionClosed

try:
    from main import session_manager, pipeline_manager
    from session_manager import SessionStatus
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import session_manager, pipeline_manager
    from session_manager import SessionStatus


async def handle_websocket(websocket, path):
    """
    Handle WebSocket connection.
    Path format: /ws/image-stream/{session_id} or /ws/image-stream/{session_id}/{branch_id}
    """
    print(f"[WebSocket Server] Connection request: {path}", flush=True)
    
    path_parts = path.strip('/').split('/')
    if len(path_parts) < 3 or path_parts[0] != 'ws' or path_parts[1] != 'image-stream':
        await websocket.close(code=1008, reason="Invalid path")
        return
    
    session_id = path_parts[2]
    branch_id = path_parts[3] if len(path_parts) > 3 else None
    
    if branch_id:
        branch = session_manager.get_branch(branch_id)
        if not branch:
            await websocket.send(json.dumps({"type": "error", "message": "Branch not found"}))
            await websocket.close(code=1008, reason="Branch not found")
            return
        
        session = session_manager.get_session(branch.session_id)
        if not session:
            await websocket.send(json.dumps({"type": "error", "message": "Session not found"}))
            await websocket.close(code=1008, reason="Session not found")
            return
        
        session_manager.set_branch_websocket(branch_id, websocket)
    else:
        session = session_manager.get_session(session_id)
        if not session:
            await websocket.send(json.dumps({"type": "error", "message": "Session not found"}))
            await websocket.close(code=1008, reason="Session not found")
            return
        
        session_manager.set_websocket(session_id, websocket)
    
    await websocket.send(json.dumps({
        "type": "connected",
        "sessionId": session_id,
        "branchId": branch_id,
    }))
    
    if not branch_id:
        session_manager.update_session_status(session_id, SessionStatus.GENERATING)
    
    send_queue = None
    send_task = None
    generation_task = None
    
    try:
        send_queue = queue.Queue()
        
        async def send_messages():
            """Send messages from queue to WebSocket"""
            while True:
                try:
                    loop = asyncio.get_event_loop()
                    message = await loop.run_in_executor(None, send_queue.get)
                    
                    if message is None:
                        break
                    
                    message_str, step_idx, node_id = message
                    await websocket.send(message_str)
                    send_queue.task_done()
                except Exception as e:
                    print(f"[WebSocket Server] Send error: {e}", flush=True)
                    break
        
        send_task = asyncio.create_task(send_messages())
        
        images_dir = os.path.join(os.path.dirname(__file__), "generated_images")
        os.makedirs(images_dir, exist_ok=True)
        
        def image_callback(step_idx: int, timestep: int, image: Image.Image):
            """Save image locally and send file URL"""
            try:
                image_filename = f"{session_id}_step_{step_idx}.png"
                if branch_id:
                    image_filename = f"{session_id}_branch_{branch_id}_step_{step_idx}.png"
                
                image_path = os.path.join(images_dir, image_filename)
                image.save(image_path, format="PNG")
                
                image_url = f"/images/{image_filename}"
                
                node_id = f"node_image_{session_id}_{step_idx}"
                if step_idx == 0:
                    parent_node_id = session.root_node_id
                else:
                    parent_node_id = f"node_image_{session_id}_{step_idx - 1}"
                
                if hasattr(timestep, 'item'):
                    timestamp_value = int(timestep.item())
                elif isinstance(timestep, (int, float)):
                    timestamp_value = int(timestep)
                else:
                    timestamp_value = int(timestep) if timestep is not None else 0
                
                message = {
                    "type": "image_step",
                    "sessionId": session_id,
                    "nodeId": node_id,
                    "parentNodeId": parent_node_id,
                    "branchId": branch_id,
                    "step": step_idx,
                    "imageUrl": image_url,
                    "timestamp": timestamp_value,
                }
                
                message_str = json.dumps(message)
                send_queue.put((message_str, step_idx, node_id))
                    
            except Exception as e:
                print(f"[WebSocket Server] Image callback error: {e}", flush=True)
        
        async def start_generation():
            try:
                if branch_id:
                    branch = session_manager.get_branch(branch_id)
                    
                    def feedback_callback(step_idx: int):
                        return branch.feedback if branch else []
                    
                    await pipeline_manager.generate_image(
                        prompt=session.prompt,
                        objects=session.objects,
                        bboxes=session.bboxes,
                        width=session.width,
                        height=session.height,
                        num_inference_steps=session.num_inference_steps,
                        callback=image_callback,
                        feedback_callback=feedback_callback
                    )
                else:
                    await pipeline_manager.generate_image(
                        prompt=session.prompt,
                        objects=session.objects,
                        bboxes=session.bboxes,
                        width=session.width,
                        height=session.height,
                        num_inference_steps=session.num_inference_steps,
                        callback=image_callback
                    )
                
                while send_queue.qsize() > 0:
                    await asyncio.sleep(0.1)
                
                complete_message = {
                    "type": "complete",
                    "sessionId": session_id,
                    "branchId": branch_id,
                }
                await websocket.send(json.dumps(complete_message))
                
                send_queue.put(None)
                await send_task
            except Exception as e:
                print(f"[WebSocket Server] Generation error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                error_message = {
                    "type": "error",
                    "sessionId": session_id,
                    "branchId": branch_id,
                    "message": str(e),
                }
                await websocket.send(json.dumps(error_message))
        
        generation_task = asyncio.create_task(start_generation())
        
        try:
            async for message in websocket:
                data = json.loads(message)
        except ConnectionClosed:
            pass
        finally:
            if send_queue is not None and send_task is not None:
                try:
                    send_queue.put(None)
                    await send_task
                except Exception as e:
                    print(f"[WebSocket Server] Task cleanup error: {e}", flush=True)
            
            if generation_task is not None and not generation_task.done():
                generation_task.cancel()
                try:
                    await generation_task
                except asyncio.CancelledError:
                    pass
            
            if branch_id:
                session_manager.set_branch_websocket(branch_id, None)
            else:
                session_manager.update_session_status(session_id, SessionStatus.COMPLETED)
                session_manager.set_websocket(session_id, None)
    
    except Exception as e:
        print(f"[WebSocket Server] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


async def start_websocket_server(host="0.0.0.0", port=8000):
    """Start WebSocket server"""
    print(f"[WebSocket Server] Starting server on {host}:{port}", flush=True)
    
    async with websocket_serve(handle_websocket, host, port):
        print(f"[WebSocket Server] Server running on port {port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(start_websocket_server())
