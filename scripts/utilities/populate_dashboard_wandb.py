#!/usr/bin/env python3
"""
Download wandb images and populate the dashboard's local wandb_data directory.
"""

import wandb
import json
import os
import requests
from pathlib import Path
from tqdm import tqdm

# Dashboard paths
DASHBOARD_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/geomancer_dashboard")
WANDB_DATA_DIR = DASHBOARD_DIR / "wandb_data"
IMAGES_DIR = WANDB_DATA_DIR / "images"
METADATA_FILE = WANDB_DATA_DIR / "wandb_metadata.json"

# Wandb project configuration
ENTITY = "cesar-valdez-mcgill-university"
PROJECT = "geomancer-phate-labeled"

def setup_directories():
    """Create necessary directories."""
    WANDB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directories:")
    print(f"  Data: {WANDB_DATA_DIR}")
    print(f"  Images: {IMAGES_DIR}")

def download_image(url, output_path):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False

def download_wandb_data():
    """Download images and metadata from wandb project."""
    print(f"\nConnecting to W&B project: {ENTITY}/{PROJECT}")

    try:
        api = wandb.Api()
        runs = api.runs(f"{ENTITY}/{PROJECT}")
        print(f"Found {len(runs)} runs")
    except Exception as e:
        print(f"Error connecting to W&B: {e}")
        print("Make sure you have wandb installed and are logged in: wandb login")
        return None

    metadata = {}
    downloaded_count = 0

    for i, run in enumerate(tqdm(runs, desc="Processing runs")):
        try:
            # Extract run metadata
            config = {k: v for k, v in run.config.items()} if run.config else {}
            summary = {k: v for k, v in run.summary.items()} if run.summary else {}

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
                except Exception as e:
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
                    except Exception as e:
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

                # Create metadata entry in dashboard format
                metadata[run.id] = {
                    'dataset_name': dataset_name,
                    'label_key': label_key,
                    'subtitle': f"{dataset_name} - {label_key}",
                    'wandb_run_id': run.id,
                    'wandb_name': run.name,
                    'wandb_state': run.state,
                    'config': config,
                    'summary': summary,
                    'primary_structure': '',  # For manual classification
                    'color_legend': None  # Will be populated if available
                }

        except Exception as e:
            print(f"Error processing run {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(runs)} runs, {downloaded_count} with images")

    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDownload complete!")
    print(f"  Downloaded images: {downloaded_count}")
    print(f"  Metadata saved to: {METADATA_FILE}")
    print(f"  Images saved to: {IMAGES_DIR}")

    return metadata, downloaded_count

def verify_integration():
    """Verify dashboard integration."""
    print(f"\nVerifying dashboard integration...")

    # Check directories exist
    if not IMAGES_DIR.exists():
        print("✗ Images directory missing")
        return False

    # Check metadata file
    if not METADATA_FILE.exists():
        print("✗ Metadata file missing")
        return False

    # Count files
    image_count = len(list(IMAGES_DIR.glob("*.png")))

    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    print(f"✓ Images: {image_count}")
    print(f"✓ Metadata entries: {len(metadata)}")
    print(f"✓ Dashboard paths configured")
    print(f"\nDashboard wandb files location:")
    print(f"  {WANDB_DATA_DIR}")
    print(f"\nRestart your dashboard and visit /gallery to see the wandb plots!")

    return True

def main():
    """Main integration pipeline."""
    print("Dashboard Wandb Integration")
    print("=" * 50)

    # Setup directories
    setup_directories()

    # Download wandb data
    metadata, count = download_wandb_data()
    if not metadata:
        print("Failed to download wandb data")
        return

    # Verify integration
    verify_integration()

    print(f"\n{'=' * 50}")
    print(f"✓ Integration complete!")
    print(f"✓ {count} wandb images ready for dashboard")
    print(f"✓ Restart dashboard and visit /gallery")

if __name__ == "__main__":
    main()