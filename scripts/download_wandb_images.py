#!/usr/bin/env python3
"""Download PHATE images and metadata from W&B project."""

import wandb
import json
import os
from pathlib import Path
import requests
from tqdm import tqdm

# Configuration
ENTITY = "cesar-valdez-mcgill-university"
PROJECT = "geomancer-phate-labeled"
OUTPUT_DIR = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery")
IMAGES_DIR = OUTPUT_DIR / "images_new"
METADATA_FILE = OUTPUT_DIR / "runs_metadata_new.json"

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

def main():
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to W&B project: {ENTITY}/{PROJECT}")

    # Initialize wandb API
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    print(f"Found {len(runs)} runs")

    metadata = []

    for i, run in enumerate(tqdm(runs, desc="Processing runs")):
        # Safely extract config and summary
        try:
            config = {k: v for k, v in run.config.items()} if run.config else {}
        except:
            config = {}

        try:
            summary = {k: v for k, v in run.summary.items()} if run.summary else {}
        except:
            summary = {}

        # Extract label_key from config (nested under data.value.label_key)
        label_key_from_config = ''
        try:
            import json
            if isinstance(run.config, str):
                config_data = json.loads(run.config)
            else:
                config_data = run.config
            label_key_from_config = config_data.get('data', {}).get('value', {}).get('label_key', '')
        except:
            pass

        run_data = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "config": config,
            "summary": summary,
            "label_key_from_config": label_key_from_config,
        }

        # Extract dataset_name and subset_name from run name or config
        # Format: phate_subdataset_<dataset>__<subset>__...__<label_key>
        name_parts = run.name.split('__')
        if len(name_parts) >= 1:
            # Extract dataset from first part after "phate_subdataset_"
            first_part = name_parts[0]
            if first_part.startswith('phate_subdataset_'):
                run_data['dataset_name'] = first_part.replace('phate_subdataset_', '')
            else:
                run_data['dataset_name'] = first_part

        if len(name_parts) >= 2:
            run_data['subset_name'] = name_parts[1] if len(name_parts) > 2 else ''
            run_data['label_key'] = name_parts[-1]
        else:
            run_data['subset_name'] = ''
            run_data['label_key'] = ''

        # Download images from run files
        image_downloaded = False
        try:
            files = run.files()
            for f in files:
                if f.name.endswith(('.png', '.jpg', '.jpeg')) and 'phate' in f.name.lower():
                    # Download this image
                    output_path = IMAGES_DIR / f"{run.id}.png"
                    if not output_path.exists():
                        url = f.url
                        if download_image(url, output_path):
                            image_downloaded = True
                            run_data['image_path'] = str(output_path)
                    else:
                        image_downloaded = True
                        run_data['image_path'] = str(output_path)
                    break
        except Exception as e:
            print(f"  Error accessing files for run {run.id}: {e}")

        # Also check for logged images in summary
        if not image_downloaded:
            try:
                for key, value in run.summary.items():
                    if isinstance(value, dict) and '_type' in value and value['_type'] == 'image-file':
                        # This is a logged image
                        if 'path' in value:
                            img_url = f"https://api.wandb.ai/files/{ENTITY}/{PROJECT}/{run.id}/{value['path']}"
                            output_path = IMAGES_DIR / f"{run.id}.png"
                            if not output_path.exists():
                                if download_image(img_url, output_path):
                                    image_downloaded = True
                                    run_data['image_path'] = str(output_path)
                            else:
                                image_downloaded = True
                                run_data['image_path'] = str(output_path)
                            break
            except Exception as e:
                pass

        # Try to get image from media directory
        if not image_downloaded:
            try:
                for f in run.files():
                    if 'media/images' in f.name or f.name.endswith('.png'):
                        output_path = IMAGES_DIR / f"{run.id}.png"
                        if not output_path.exists():
                            if download_image(f.url, output_path):
                                image_downloaded = True
                                run_data['image_path'] = str(output_path)
                        else:
                            image_downloaded = True
                            run_data['image_path'] = str(output_path)
                        break
            except:
                pass

        if image_downloaded:
            metadata.append(run_data)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(runs)} runs, {len(metadata)} with images")

    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDownload complete!")
    print(f"  Runs with images: {len(metadata)}")
    print(f"  Metadata saved to: {METADATA_FILE}")
    print(f"  Images saved to: {IMAGES_DIR}")

if __name__ == "__main__":
    main()
