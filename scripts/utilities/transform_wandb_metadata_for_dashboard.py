#!/usr/bin/env python3
"""
Transform wandb metadata from this repository's format to dashboard-expected format.
"""

import json
import os
from pathlib import Path
import shutil

# Paths
CURRENT_REPO_WANDB_DIR = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery")
DASHBOARD_WANDB_DIR = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery")

# Input files (from download scripts)
IMAGES_SOURCE_DIR = CURRENT_REPO_WANDB_DIR / "images_new"
METADATA_SOURCE_FILE = CURRENT_REPO_WANDB_DIR / "runs_metadata_with_legends.json"

# Output paths (expected by dashboard)
IMAGES_TARGET_DIR = DASHBOARD_WANDB_DIR / "labeled_images"
METADATA_TARGET_FILE = DASHBOARD_WANDB_DIR / "wandb_metadata.json"

def transform_metadata():
    """Transform metadata from repository format to dashboard format."""
    print("Transforming wandb metadata for dashboard...")

    # Check if source files exist
    if not METADATA_SOURCE_FILE.exists():
        print(f"Source metadata not found: {METADATA_SOURCE_FILE}")
        print("Please run the wandb download scripts first:")
        print("1. python scripts/data_collection/download_wandb_images.py")
        print("2. python scripts/llm_processing/extract_color_legends.py")
        return

    # Load source metadata
    with open(METADATA_SOURCE_FILE, 'r') as f:
        source_metadata = json.load(f)

    print(f"Loaded {len(source_metadata)} runs from source metadata")

    # Transform to dashboard format
    dashboard_metadata = {}

    for run in source_metadata:
        run_id = run['id']

        # Extract dataset name and label key
        dataset_name = run.get('dataset_name', 'Unknown')
        label_key = run.get('label_key_from_config', run.get('label_key', 'cell_type'))

        # Create dashboard format entry
        dashboard_entry = {
            'dataset_name': dataset_name,
            'label_key': label_key,
            'color_legend': run.get('color_legend'),
            'primary_structure': run.get('primary_structure', ''),  # Classification if available
            'subtitle': f"{dataset_name} - {label_key}",
            'wandb_run_id': run_id,
            'wandb_name': run.get('name', ''),
            'wandb_state': run.get('state', 'finished')
        }

        # Add any additional metadata from config/summary
        if 'config' in run and run['config']:
            dashboard_entry['config'] = run['config']
        if 'summary' in run and run['summary']:
            dashboard_entry['summary'] = run['summary']

        dashboard_metadata[run_id] = dashboard_entry

    # Save transformed metadata
    DASHBOARD_WANDB_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_TARGET_FILE, 'w') as f:
        json.dump(dashboard_metadata, f, indent=2)

    print(f"Transformed metadata saved to: {METADATA_TARGET_FILE}")
    print(f"Dashboard format entries: {len(dashboard_metadata)}")

    return dashboard_metadata

def copy_images():
    """Copy/link wandb images to dashboard expected location."""
    print("\nCopying wandb images for dashboard...")

    if not IMAGES_SOURCE_DIR.exists():
        print(f"Source images directory not found: {IMAGES_SOURCE_DIR}")
        print("Please run: python scripts/data_collection/download_wandb_images.py")
        return

    # Create target directory
    IMAGES_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all PNG files
    image_files = list(IMAGES_SOURCE_DIR.glob("*.png"))
    copied_count = 0

    for img_file in image_files:
        target_file = IMAGES_TARGET_DIR / img_file.name

        if not target_file.exists():
            try:
                # Use symlink to save space, fallback to copy if symlink fails
                try:
                    target_file.symlink_to(img_file.absolute())
                    copied_count += 1
                except OSError:
                    # Symlink failed, copy instead
                    shutil.copy2(img_file, target_file)
                    copied_count += 1
            except Exception as e:
                print(f"Error copying {img_file.name}: {e}")

    print(f"Copied/linked {copied_count} images to: {IMAGES_TARGET_DIR}")

    return copied_count

def verify_dashboard_integration():
    """Verify that files are in expected locations for dashboard."""
    print("\nVerifying dashboard integration...")

    # Check metadata file
    if METADATA_TARGET_FILE.exists():
        with open(METADATA_TARGET_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata file exists: {len(metadata)} entries")
    else:
        print("✗ Metadata file missing")
        return False

    # Check images directory
    if IMAGES_TARGET_DIR.exists():
        image_count = len(list(IMAGES_TARGET_DIR.glob("*.png")))
        print(f"✓ Images directory exists: {image_count} images")
    else:
        print("✗ Images directory missing")
        return False

    # Check if dashboard can access files
    print(f"\nDashboard paths:")
    print(f"  Images: {IMAGES_TARGET_DIR}")
    print(f"  Metadata: {METADATA_TARGET_FILE}")
    print(f"\nTo view in dashboard, go to: http://your-dashboard-url/gallery")

    return True

def main():
    """Main integration pipeline."""
    print("Wandb Dashboard Integration")
    print("=" * 50)

    # Transform metadata
    metadata = transform_metadata()
    if not metadata:
        return

    # Copy images
    image_count = copy_images()
    if image_count == 0:
        return

    # Verify integration
    success = verify_dashboard_integration()

    if success:
        print("\n" + "=" * 50)
        print("✓ Integration complete!")
        print(f"✓ {len(metadata)} wandb runs ready for dashboard")
        print(f"✓ {image_count} images available")
        print("\nNext steps:")
        print("1. Restart your dashboard application")
        print("2. Visit /gallery to see all wandb plots")
        print("3. Use the classification interface to label plots")
    else:
        print("\n✗ Integration failed - check errors above")

if __name__ == "__main__":
    main()