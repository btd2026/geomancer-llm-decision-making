#!/usr/bin/env python3
"""Standalone HTTP server - doesn't rely on python3 -m http.server module"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
import os

# Configuration
PORT = 8003
DIRECTORY = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit'
HTML_FILE = 'wandb_multiselect_gallery.html'

class GalleryHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Request handler that serves the enhanced HTML gallery."""

    def __init__(self):
        super().__init__()
        self.directory = os.getcwd()

    def do_GET(self, handler):
        """Handle GET requests."""
        # Serve the HTML file for root path
        if handler.path == '/' or handler.path == '':
            file_path = os.path.join(DIRECTORY, HTML_FILE)
        elif handler.path.endswith('.html') or handler.path.endswith('.js') or handler.path.endswith('.css') or handler.path.endswith('.json'):
            file_path = os.path.join(DIRECTORY, handler.path.lstrip('/'))

        # Check if file exists
        if not os.path.exists(file_path):
            handler.send_error(404, message="File not found")
            return

        # Log and serve the file
        with open(file_path, 'rb') as f:
            content = f.read()
            handler.send_response(200)
            # Log success
            client_addr = handler.client_address[0]
            print(f"[200] Served {handler.path} to {client_addr[0]} - {len(content)} bytes")

        def log_message(self, message):
        """Log a message to console."""
        print(message)

    def run_server():
        """Start the HTTP server."""
        server_address = ('', PORT)

        # Create request handler
        handler = GalleryHTTPRequestHandler()

        # Create server
        server = HTTPServer(server_address, handler)

        # Allow address reuse
        server.allow_reuse_address = True

        # Log startup
        log_message("PHATE Gallery Server Started", "=" * 30)
        print(f"Server address: {server_address}")
        print(f"Working directory: {DIRECTORY}")
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
