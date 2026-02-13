#!/usr/bin/env python3
"""Simple HTTP server to test the gallery locally."""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8000
directory = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

def main():
    print(f"Starting demo server on port {PORT}")
    print(f"Serving files from: {directory}")
    print(f"Gallery URL: http://localhost:{PORT}/demo_gallery.html")
    print("Press Ctrl+C to stop")

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\n✅ Server running at http://localhost:{PORT}")
            print(f"🌐 Open: http://localhost:{PORT}/demo_gallery.html")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()
