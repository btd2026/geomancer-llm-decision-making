#!/usr/bin/env python3
"""
Test script to verify the Replit W&B gallery setup locally before deployment.
This creates a small demo gallery to test the HTML generation and styling.
"""

import json
import base64
from pathlib import Path
from datetime import datetime
import os

def create_sample_data():
    """Create sample metadata for testing the gallery."""
    sample_data = {}

    # Create sample run data that matches your W&B structure
    for i in range(5):
        run_id = f"sample_run_{i:03d}"
        sample_data[run_id] = {
            'run_id': run_id,
            'dataset_name': f'pbmc_{i+1}k',
            'label_key': 'cell_type' if i % 2 == 0 else 'leiden_clusters',
            'subtitle': f'pbmc_{i+1}k - {"cell_type" if i % 2 == 0 else "leiden_clusters"}',
            'wandb_run_id': run_id,
            'wandb_name': f'phate_subdataset_pbmc_{i+1}k__{"cell_type" if i % 2 == 0 else "leiden_clusters"}',
            'wandb_state': 'finished',
            'config': {'n_components': 2, 'knn': 5, 'n_jobs': 1},
            'summary': {'runtime': i * 0.5 + 2.3, 'memory_usage': 512 + i * 128},
            'image_path': f'sample_image_{i}.png',
            'created_at': datetime.now().isoformat(),
            'url': f'https://wandb.ai/cesar-valdez-mcgill-university/geomancer-phate-labeled/runs/{run_id}'
        }

    return sample_data

def create_sample_image():
    """Create a simple test image as base64."""
    # This is a tiny 1x1 PNG in base64 (transparent pixel)
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

def create_demo_html():
    """Create a demo HTML gallery with sample data."""

    # Get sample data
    metadata = create_sample_data()
    sample_image_b64 = create_sample_image()

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>W&B PHATE Demo Gallery</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 16px; }}
        .demo-notice {{
            background: #fff3cd;
            color: #856404;
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #ffeeba;
        }}
        .demo-notice strong {{ color: #721c24; }}
        .stats {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid #dee2e6;
        }}
        .stats span {{
            display: inline-block;
            margin: 0 20px;
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border-radius: 25px;
            font-weight: bold;
        }}
        .controls {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .search-box {{
            width: 100%;
            max-width: 400px;
            padding: 12px 20px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            font-size: 14px;
            transition: border-color 0.3s;
        }}
        .search-box:focus {{
            outline: none;
            border-color: #007bff;
        }}
        .gallery {{
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-color: #007bff;
        }}
        .card img {{
            width: 100%;
            height: 200px;
            background: linear-gradient(45deg, #f0f0f0 25%, transparent 25%, transparent 75%, #f0f0f0 75%),
                        linear-gradient(45deg, #f0f0f0 25%, transparent 25%, transparent 75%, #f0f0f0 75%);
            background-size: 20px 20px;
            background-position: 0 0, 10px 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 14px;
            cursor: pointer;
            transition: opacity 0.3s;
        }}
        .card img:hover {{ opacity: 0.9; }}
        .card-body {{ padding: 20px; }}
        .card-title {{
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            margin: 0 0 8px 0;
            line-height: 1.3;
        }}
        .card-subtitle {{
            font-size: 14px;
            color: #6c757d;
            margin: 0 0 12px 0;
        }}
        .card-meta {{
            font-size: 12px;
            color: #868e96;
            margin-bottom: 8px;
        }}
        .card-link {{
            display: inline-block;
            padding: 8px 16px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 20px;
            font-size: 12px;
            transition: background 0.3s;
        }}
        .card-link:hover {{
            background: #0056b3;
            text-decoration: none;
            color: white;
        }}
        .instructions {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 20px;
            margin: 20px;
            border-radius: 8px;
        }}
        .instructions h3 {{ margin-top: 0; }}
        .instructions code {{
            background: #e2e3e5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 PHATE Visualizations Demo</h1>
            <p>Test gallery for W&B runs from cesar-valdez-mcgill-university/geomancer-phate-labeled</p>
        </div>

        <div class="demo-notice">
            <strong>⚠️ Demo Mode</strong> - This is a test gallery with sample data.
            Run the full script with W&B authentication to see your actual visualizations.
        </div>

        <div class="stats">
            <span>📊 {len(metadata)} Demo Visualizations</span>
            <span>🔬 {len(set(item['dataset_name'] for item in metadata.values()))} Datasets</span>
            <span>🏷️ {len(set(item['label_key'] for item in metadata.values()))} Label Types</span>
        </div>

        <div class="controls">
            <input
                type="text"
                id="searchBox"
                class="search-box"
                placeholder="🔍 Search by dataset name, label, or run ID..."
                onkeyup="filterCards()"
            >
        </div>

        <div class="instructions">
            <h3>🚀 Next Steps to Deploy on Replit:</h3>
            <ol>
                <li><strong>Login to W&B:</strong> <code>wandb login</code></li>
                <li><strong>Run the full script:</strong> <code>python create_replit_wandb_gallery.py</code></li>
                <li><strong>Upload to Replit:</strong> Copy all files to your Replit project</li>
                <li><strong>Start server:</strong> <code>python -m http.server 8000 --directory wandb_gallery_replit</code></li>
                <li><strong>View gallery:</strong> Open Replit web preview</li>
            </ol>
            <p><strong>📖 For detailed instructions, see:</strong> <code>REPLIT_DEPLOYMENT_GUIDE.md</code></p>
        </div>

        <div class="gallery" id="gallery">'''

    # Add demo cards
    for run_id, item in metadata.items():
        html_content += f'''
            <div class="card" data-name="{item['dataset_name'].lower()}" data-label="{item['label_key'].lower()}" data-id="{item['run_id']}">
                <div style="width: 100%; height: 200px; background: linear-gradient(45deg, #f0f0f0 25%, transparent 25%, transparent 75%, #f0f0f0 75%), linear-gradient(45deg, #f0f0f0 25%, transparent 25%, transparent 75%, #f0f0f0 75%); background-size: 20px 20px; background-position: 0 0, 10px 10px; display: flex; align-items: center; justify-content: center; color: #666; font-size: 14px;">
                    📊 PHATE Visualization<br>
                    <small>(Demo placeholder)</small>
                </div>
                <div class="card-body">
                    <div class="card-title">{item['dataset_name']}</div>
                    <div class="card-subtitle">Label: {item['label_key']}</div>
                    <div class="card-meta">
                        Run ID: {item['run_id'][:16]}...<br>
                        Demo Created: {item['created_at'][:16]}
                    </div>
                    <a href="{item['url']}" target="_blank" class="card-link">
                        View in W&B →
                    </a>
                </div>
            </div>'''

    html_content += '''
        </div>
    </div>

    <script>
        function filterCards() {
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const cards = document.querySelectorAll('.card');

            cards.forEach(card => {
                const name = card.dataset.name;
                const label = card.dataset.label;
                const id = card.dataset.id.toLowerCase();

                const matches = name.includes(searchTerm) ||
                               label.includes(searchTerm) ||
                               id.includes(searchTerm);

                card.style.display = matches ? 'block' : 'none';
            });
        }

        // Demo functionality notification
        document.addEventListener('DOMContentLoaded', function() {
            console.log('W&B Gallery Demo loaded successfully!');
            console.log('To create a gallery with real W&B data:');
            console.log('1. Run: wandb login');
            console.log('2. Run: python create_replit_wandb_gallery.py');
        });
    </script>
</body>
</html>'''

    return html_content

def main():
    """Create a demo gallery for testing."""
    print("🧪 Creating Demo W&B Gallery")
    print("=" * 40)

    # Create demo directory
    demo_dir = Path("./wandb_gallery_demo")
    demo_dir.mkdir(exist_ok=True)

    # Generate demo HTML
    html_content = create_demo_html()

    # Write demo file
    demo_file = demo_dir / "demo_gallery.html"
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Demo gallery created: {demo_file}")
    print(f"📁 Demo directory: {demo_dir}")

    # Create simple server script
    server_script = demo_dir / "start_server.py"
    with open(server_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
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
            print(f"\\n✅ Server running at http://localhost:{PORT}")
            print(f"🌐 Open: http://localhost:{PORT}/demo_gallery.html")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped")

if __name__ == "__main__":
    main()
''')

    # Make server script executable
    os.chmod(server_script, 0o755)

    print("\n" + "=" * 40)
    print("✅ Demo Setup Complete!")
    print(f"\n📋 To test locally:")
    print(f"1. cd {demo_dir}")
    print(f"2. python start_server.py")
    print(f"3. Open: http://localhost:8000/demo_gallery.html")
    print(f"\n🚀 To create real gallery with W&B data:")
    print("1. wandb login")
    print("2. python ../create_replit_wandb_gallery.py")

    return demo_file

if __name__ == "__main__":
    main()