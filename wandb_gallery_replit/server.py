#!/usr/bin/env python3
"""Simple HTTP server for PHATE Gallery."""

import http.server
import socketserver
import os
import sys

# Configuration
PORT = 8003
DIRECTORY = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit'
HTML_FILE = 'wandb_multiselect_gallery.html'

class GalleryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Request handler that serves the enhanced HTML gallery."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        # Serve the HTML file for root path
        if self.path == '/' or self.path == '':
            self.path = '/' + HTML_FILE

        # Log the request
        client_addr = self.client_address[0]
        print(f"[{client_addr}] GET {self.path}")

        # Call parent do_GET
        super().do_GET()

    def log_message(self, format, *args):
        """Log to console."""
        print(f"[LOG] {format % args}")

def main():
    """Start the HTTP server."""
    # Change to the directory containing the HTML file
    os.chdir(DIRECTORY)

    # Create server with reusable address
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), GalleryHTTPRequestHandler) as httpd:
        print("=" * 60)
        print(f"PHATE Gallery Server Started")
        print("=" * 60)
        print(f"Serving directory: {DIRECTORY}")
        print(f"Main HTML file: {HTML_FILE}")
        print(f"URL: http://localhost:{PORT}/")
        print("=" * 60)
        print("Press Ctrl+C to stop...")
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
