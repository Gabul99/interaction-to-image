"""
HTTP Server - Wraps FastAPI app using Python's built-in HTTP server
"""
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import asyncio

_app = None

def get_app():
    """Get FastAPI app (avoid circular import)"""
    global _app
    if _app is None:
        try:
            from main import app as main_app
            _app = main_app
        except ImportError:
            import os
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from main import app as main_app
            _app = main_app
    return _app

class FastAPIHTTPHandler(BaseHTTPRequestHandler):
    """Wrap FastAPI app with Python's built-in HTTP server"""
    
    def log_message(self, format, *args):
        print(f"[HTTP Server] {format % args}", flush=True)
    
    def do_GET(self):
        self._handle_request('GET')
    
    def do_POST(self):
        self._handle_request('POST')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _handle_request(self, method):
        try:
            if self.headers.get('Upgrade', '').lower() == 'websocket':
                self.send_error(426, "Upgrade Required - WebSocket connection should use ws:// protocol")
                return
            
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            path_parts = self.path.split('?', 1)
            path = path_parts[0]
            query_string = path_parts[1].encode() if len(path_parts) > 1 else b''
            
            scope = {
                'type': 'http',
                'method': method,
                'path': path,
                'query_string': query_string,
                'headers': [(k.lower().encode(), v.encode()) for k, v in self.headers.items()],
                'client': self.client_address,
                'server': ('0.0.0.0', 8000),
            }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            body_sent = False
            
            async def receive():
                if not body_sent:
                    return {'type': 'http.request', 'body': body}
                return {'type': 'http.request', 'body': b''}
            
            async def send(message):
                nonlocal body_sent
                if message['type'] == 'http.response.start':
                    self.send_response(message['status'])
                    for key, value in message['headers']:
                        self.send_header(key.decode(), value.decode())
                    self.end_headers()
                elif message['type'] == 'http.response.body':
                    if 'body' in message:
                        self.wfile.write(message['body'])
                    if not message.get('more_body', False):
                        body_sent = True
            
            app = get_app()
            coro = app(scope, receive, send)
            loop.run_until_complete(coro)
            loop.close()
            
        except Exception as e:
            print(f"[HTTP Server] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            try:
                self.send_error(500, str(e))
            except:
                pass

if __name__ == "__main__":
    PORT = 8000
    
    print(f"[HTTP Server] Starting server on port {PORT}", flush=True)
    
    app = get_app()
    
    print(f"[HTTP Server] Calling FastAPI startup event...", flush=True)
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from main import startup_event
        loop.run_until_complete(startup_event())
    except Exception as e:
        print(f"[HTTP Server] Startup event failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    loop.close()
    print(f"[HTTP Server] Startup event completed", flush=True)
    
    with HTTPServer(("0.0.0.0", PORT), FastAPIHTTPHandler) as httpd:
        print(f"[HTTP Server] Server running on port {PORT}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[HTTP Server] Shutting down...", flush=True)
            httpd.shutdown()
