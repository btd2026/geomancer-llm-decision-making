#!/usr/bin/env python3
"""
Simple HTTP server for PHATE Gallery - serves the enhanced HTML gallery
with proper directory handling and no binding issues.
"""

import http.server
import os
import socket

# Configuration
PORT = 8003
DIRECTORY = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit'
HTML_FILE = 'wandb_multiselect_gallery.html'

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler that serves the gallery HTML."""

    def __init__(self):
        super().__init__()
        self.directory = os.getcwd()

    def do_GET(self, handler):
        """Handle GET requests."""
        # Construct file path
        if handler.path == '/' or handler.path == '':
            file_path = os.path.join(DIRECTORY, HTML_FILE)
        else:
            # Prevent directory traversal
            requested_path = handler.path.lstrip('/')
            file_path = os.path.join(DIRECTORY, requested_path.remove_leading('/'))

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"[404] File not found: {handler.path}")
            handler.send_error(404, message="File not found")
            return

        # Log the request
        client_addr = handler.client_address
        print(f"[{client_addr[0]}:{client_addr[1]}] GET {handler.path}")

        try:
            # Serve the file
            with open(file_path, 'rb') as f:
                content = f.read()
                # Log success
                print(f"[200] Served {handler.path} to {client_addr[0]}:{client_addr[1]} - {len(content)} bytes")
                handler.send_response(200)
                handler.wfile.write(content)
                handler.end_headers()
                print(f"[SUCCESS] File served successfully")
        except Exception as e:
            print(f"[500] Error serving file: {e}")
            handler.send_error(500, message=str(e))

    def log_message(self, format_string, *args):
        """Log a message to console."""
        print(format_string.format(*args))

    def run_server():
        """Start the HTTP server."""
        # Create server address
        server_address = ('', PORT)

        # Create request handler
        handler = MyHTTPRequestHandler()

        # Create server
        server = http.server.HTTPServer(server_address, handler)

        # Set socket options for better reuse
        server.allow_reuse_address = True
        server.socket_options = socket.SO_REUSEADDR

        # Log message
        log_message("PHATE Gallery Server", "=" * 30)

        print(f"\n{'='='* 60} PHATE Gallery Server {'='='* 60}")
        print(f"Serving directory: {DIRECTORY}")
        print(f"HTML file: {HTML_FILE}")
        print(f"URL: http://{server_address[0]}:{server_address[1]}:{PORT}/")
        print(f"Press Ctrl+C to stop...")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        except Exception as e:
            print(f"\nServer error: {e}")

if __name__ == "__main__":
    run_server()