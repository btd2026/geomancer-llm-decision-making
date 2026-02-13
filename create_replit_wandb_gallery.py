#!/usr/bin/env python3
"""
Create a Replit-compatible W&B gallery by downloading data and generating HTML.
This script combines the W&B data fetching and HTML generation for Replit deployment.
"""

import wandb
import json
import os
import requests
import base64
from pathlib import Path
from tqdm import tqdm
import argparse

# Configuration for Replit deployment
REPLIT_OUTPUT_DIR = Path("./wandb_gallery_replit")
IMAGES_DIR = REPLIT_OUTPUT_DIR / "images"
METADATA_FILE = REPLIT_OUTPUT_DIR / "wandb_metadata.json"
OUTPUT_HTML = REPLIT_OUTPUT_DIR / "index.html"

# W&B project configuration
ENTITY = "cesar-valdez-mcgill-university"
PROJECT = "geomancer-phate-labeled"

def setup_directories():
    """Create necessary directories for Replit."""
    REPLIT_OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    print(f"✓ Created output directory: {REPLIT_OUTPUT_DIR}")

def download_image(url, output_path):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  ⚠️ Error downloading {url}: {e}")
        return False

def image_to_base64(image_path):
    """Convert image to base64 string for embedding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def download_wandb_data(max_runs=None):
    """Download images and metadata from W&B project."""
    print(f"🔗 Connecting to W&B project: {ENTITY}/{PROJECT}")

    try:
        api = wandb.Api()
        runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
        print(f"📊 Found {len(runs)} runs")

        if max_runs:
            runs = runs[:max_runs]
            print(f"📊 Limited to first {len(runs)} runs for processing")
    except Exception as e:
        print(f"❌ Error connecting to W&B: {e}")
        print("💡 Make sure you have wandb installed and are logged in: wandb login")
        return None, 0

    metadata = {}
    downloaded_count = 0

    for i, run in enumerate(tqdm(runs, desc="Processing runs")):
        try:
            # Extract run metadata - handle different types
            config = {}
            if run.config:
                if isinstance(run.config, str):
                    try:
                        import json
                        config = json.loads(run.config)
                    except (json.JSONDecodeError, TypeError):
                        config = {'raw_config': run.config}
                elif hasattr(run.config, 'items'):
                    config = {k: v for k, v in run.config.items()}
                elif isinstance(run.config, dict):
                    config = run.config

            summary = {}
            if run.summary:
                try:
                    # Try to convert summary to dict
                    if hasattr(run.summary, '_json_dict'):
                        summary = getattr(run.summary, '_json_dict', {})
                    elif hasattr(run.summary, 'keys'):
                        summary = {k: getattr(run.summary, k, None) for k in run.summary.keys()}
                    else:
                        summary = {'summary_str': str(run.summary)}
                except Exception:
                    summary = {'summary_error': 'Could not parse summary'}

            # Extract dataset information from run name
            name_parts = run.name.split('__')
            dataset_name = 'Unknown'
            label_key = 'cell_type'

            if len(name_parts) >= 1:
                first_part = name_parts[0]
                if first_part.startswith('phate_subdataset_'):
                    dataset_name = first_part.replace('phate_subdataset_', '')
                else:
                    dataset_name = first_part

            if len(name_parts) >= 2:
                label_key = name_parts[-1] if len(name_parts) > 1 else 'cell_type'

            # Try to download image
            image_downloaded = False
            output_path = IMAGES_DIR / f"{run.id}.png"

            # Check if already downloaded
            if output_path.exists():
                image_downloaded = True
            else:
                # Try downloading from run files
                try:
                    files = run.files()
                    for f in files:
                        if f.name.endswith(('.png', '.jpg', '.jpeg')) and 'phate' in f.name.lower():
                            if download_image(f.url, output_path):
                                image_downloaded = True
                            break
                except Exception:
                    pass

                # Try downloading from logged images in summary
                if not image_downloaded:
                    try:
                        for key, value in run.summary.items():
                            if isinstance(value, dict) and '_type' in value and value['_type'] == 'image-file':
                                if 'path' in value:
                                    img_url = f"https://api.wandb.ai/files/{ENTITY}/{PROJECT}/{run.id}/{value['path']}"
                                    if download_image(img_url, output_path):
                                        image_downloaded = True
                                    break
                    except Exception:
                        pass

                # Try media/images directory
                if not image_downloaded:
                    try:
                        for f in run.files():
                            if 'media/images' in f.name or f.name.endswith('.png'):
                                if download_image(f.url, output_path):
                                    image_downloaded = True
                                break
                    except:
                        pass

            if image_downloaded:
                downloaded_count += 1

                # Create metadata entry
                metadata[run.id] = {
                    'run_id': run.id,
                    'dataset_name': dataset_name,
                    'label_key': label_key,
                    'subtitle': f"{dataset_name} - {label_key}",
                    'wandb_run_id': run.id,
                    'wandb_name': run.name,
                    'wandb_state': run.state,
                    'config': config,
                    'summary': summary,
                    'image_path': str(output_path),
                    'created_at': run.created_at.isoformat() if run.created_at and hasattr(run.created_at, 'isoformat') else str(run.created_at) if run.created_at else None,
                    'url': run.url
                }

        except Exception as e:
            print(f"❌ Error processing run {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  📈 Processed {i + 1}/{len(runs)} runs, {downloaded_count} with images")

    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Download complete!")
    print(f"  📸 Downloaded images: {downloaded_count}")
    print(f"  💾 Metadata saved to: {METADATA_FILE}")

    return metadata, downloaded_count

def create_replit_html(metadata):
    """Create a self-contained HTML file for Replit with embedded images."""
    print("🎨 Creating HTML gallery...")

    # Build items with embedded images
    items = []
    for run_id, data in metadata.items():
        image_path = Path(data['image_path'])
        if not image_path.exists():
            print(f"⚠️ Image not found: {image_path}")
            continue

        try:
            base64_image = image_to_base64(image_path)
            items.append({
                'run_id': run_id,
                'dataset_name': data.get('dataset_name', 'Unknown'),
                'label_key': data.get('label_key', 'Unknown'),
                'subtitle': data.get('subtitle', ''),
                'wandb_name': data.get('wandb_name', ''),
                'base64_image': base64_image,
                'url': data.get('url', ''),
                'created_at': data.get('created_at', ''),
                'config': data.get('config', {}),
                'summary': data.get('summary', {})
            })
        except Exception as e:
            print(f"⚠️ Error processing image {image_path}: {e}")

    print(f"📊 Processed {len(items)} items for gallery")

    # HTML template
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>W&B PHATE Visualizations - {ENTITY}/{PROJECT}</title>
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
            height: 250px;
            object-fit: cover;
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
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }}
        .modal img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .hidden {{ display: none; }}
        @media (max-width: 768px) {{
            body {{ padding: 10px; }}
            .header {{ padding: 20px; }}
            .header h1 {{ font-size: 24px; }}
            .gallery {{ padding: 10px; grid-template-columns: 1fr; }}
            .card {{ margin: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 PHATE Visualizations</h1>
            <p>Interactive gallery of W&B runs from {ENTITY}/{PROJECT}</p>
        </div>

        <div class="stats">
            <span>📊 {len(items)} Visualizations</span>
            <span>🔬 {len(set(item['dataset_name'] for item in items))} Datasets</span>
            <span>🏷️ {len(set(item['label_key'] for item in items))} Label Types</span>
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

        <div class="gallery" id="gallery">'''

    # Add cards for each item
    for item in items:
        created_date = ""
        if item['created_at']:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                created_date = dt.strftime('%Y-%m-%d %H:%M')
            except:
                created_date = item['created_at'][:10]

        # Truncate long names for display
        display_name = item['dataset_name']
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."

        html_content += f'''
            <div class="card" data-name="{item['dataset_name'].lower()}" data-label="{item['label_key'].lower()}" data-id="{item['run_id']}">
                <img src="data:image/png;base64,{item['base64_image']}"
                     alt="{item['subtitle']}"
                     onclick="openModal(this)">
                <div class="card-body">
                    <div class="card-title">{display_name}</div>
                    <div class="card-subtitle">Label: {item['label_key']}</div>
                    <div class="card-meta">
                        Run ID: {item['run_id'][:8]}...<br>
                        {'Created: ' + created_date if created_date else ''}
                    </div>
                    <a href="{item['url']}" target="_blank" class="card-link">
                        View in W&B →
                    </a>
                </div>
            </div>'''

    html_content += '''
        </div>
    </div>

    <!-- Modal for image viewing -->
    <div id="modal" class="modal" onclick="closeModal()">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <div class="modal-content">
            <img id="modalImg">
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

        function openModal(img) {
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modalImg');
            modal.style.display = 'block';
            modalImg.src = img.src;
        }

        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>'''

    # Write HTML file
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML gallery created: {OUTPUT_HTML}")
    return OUTPUT_HTML

def create_replit_config():
    """Create a simple replit.nix and .replit file for easy deployment."""

    # Create .replit file
    replit_config = """{
  "language": "python3",
  "run": "python3 -m http.server 8000 --directory wandb_gallery_replit",
  "onBoot": "echo 'W&B Gallery Server Starting...'",
  "entrypoint": "create_replit_wandb_gallery.py"
}"""

    with open(".replit", "w") as f:
        f.write(replit_config)

    # Create replit.nix for dependencies
    nix_config = """{ pkgs }: {
    deps = [
        pkgs.python3
        pkgs.python3Packages.requests
        pkgs.python3Packages.tqdm
        pkgs.python3Packages.pip
    ];
}"""

    with open("replit.nix", "w") as f:
        f.write(nix_config)

    # Create requirements.txt for W&B
    requirements = """wandb>=0.15.0
requests>=2.28.0
tqdm>=4.64.0"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)

    print("✅ Created Replit configuration files")

def main():
    """Main function to create Replit W&B gallery."""
    parser = argparse.ArgumentParser(description='Create Replit-compatible W&B gallery')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip W&B download and just create HTML from existing data')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='Maximum number of runs to process')

    args = parser.parse_args()

    print("🚀 Creating Replit W&B Gallery")
    print("=" * 50)

    # Setup directories
    setup_directories()

    # Download or load W&B data
    if args.skip_download and METADATA_FILE.exists():
        print("📂 Loading existing metadata...")
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"📊 Loaded {len(metadata)} runs from cache")
    else:
        print("🔄 Downloading fresh W&B data...")
        result = download_wandb_data(args.max_runs)
        if result is None:
            print("❌ Failed to download W&B data")
            return
        metadata, count = result
        print(f"✅ Downloaded {count} runs with images")

    # Max runs limitation is now handled in download_wandb_data()

    # Create HTML gallery
    html_file = create_replit_html(metadata)

    # Create Replit configuration
    create_replit_config()

    print("\n" + "=" * 50)
    print("✅ Replit W&B Gallery Ready!")
    print(f"📁 Output directory: {REPLIT_OUTPUT_DIR}")
    print(f"🌐 Main file: {OUTPUT_HTML}")
    print("\n📋 To deploy on Replit:")
    print("1. Upload all files to Replit")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the server: python3 -m http.server 8000 --directory wandb_gallery_replit")
    print("4. Open the Replit web preview to view your gallery!")
    print(f"\n🔗 Your gallery will be available at: https://<replit-url>/index.html")

if __name__ == "__main__":
    main()