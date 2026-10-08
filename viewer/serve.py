"""Local server for the RoomFill viewer.

`python -m http.server` keeps a listen backlog of 5; browsers open ~15 connections at once (scripts,
fonts, icons, splat data) and Windows resets the overflow with ERR_CONNECTION_RESET. This server is the
same static handler with a deep backlog, threads, and correct MIME types for the splat files.

    python viewer/serve.py            # http://localhost:8766/?scene=room
"""
import http.server
import socketserver
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    # HTTP/1.1 keep-alive: with the default HTTP/1.0 the server closes right after a large body, and
    # Windows answers any unread client bytes with RST, which can truncate the tail of the transfer.
    protocol_version = "HTTP/1.1"
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
        ".woff2": "font/woff2", ".glb": "model/gltf-binary",
        ".ply": "application/octet-stream", ".splat": "application/octet-stream",
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):                     # always serve the latest export / page
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):        # quiet; errors still surface in the browser
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


if __name__ == "__main__":
    with Server(("", PORT), Handler) as httpd:
        print(f"RoomFill viewer on http://localhost:{PORT}/?scene=room", flush=True)
        httpd.serve_forever()
