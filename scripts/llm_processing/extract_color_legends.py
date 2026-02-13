#!/usr/bin/env python3
"""Extract color legends from h5ad files for W&B PHATE runs."""

import json
import h5py
import wandb
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

METADATA_FILE = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery/runs_metadata_new.json")
OUTPUT_FILE = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery/runs_metadata_with_legends.json")

def get_color_for_index(idx, n_categories):
    """Get viridis colormap color for a category index."""
    # PHATE plots use viridis colormap
    cmap = plt.cm.viridis

    # Sample evenly across the colormap
    if n_categories == 1:
        position = 0.5
    else:
        position = idx / (n_categories - 1)

    color = cmap(position)
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(color[0]*255),
        int(color[1]*255),
        int(color[2]*255)
    )
    return hex_color

def extract_categories_from_h5ad(adata_path, label_key):
    """Extract category names from h5ad file without loading full data."""
    try:
        with h5py.File(adata_path, 'r') as f:
            if 'obs' not in f:
                return None

            obs = f['obs']
            if label_key not in obs:
                return None

            label_data = obs[label_key]
            if 'categories' not in label_data:
                return None

            cats = label_data['categories'][:]
            if hasattr(cats[0], 'decode'):
                cats = [c.decode() for c in cats]

            return list(cats)
    except Exception as e:
        print(f"  Error reading {adata_path}: {e}")
        return None

def get_adata_paths_from_wandb():
    """Fetch adata_path and label_key for each run from W&B."""
    print("Fetching adata paths from W&B...")
    api = wandb.Api()
    runs = api.runs('cesar-valdez-mcgill-university/geomancer-phate-labeled')

    run_info = {}
    for run in tqdm(runs, desc="Fetching W&B configs"):
        try:
            config = run.config
            if isinstance(config, str):
                config = json.loads(config) if config else {}

            data_config = config.get('data', {})
            if isinstance(data_config, dict):
                value = data_config.get('value', {})
                if isinstance(value, dict):
                    adata_path = value.get('adata_path', '')
                    label_key = value.get('label_key', '')
                    if adata_path and label_key:
                        run_info[run.id] = {
                            'adata_path': adata_path,
                            'label_key': label_key
                        }
        except Exception as e:
            continue

    return run_info

def main():
    # Load existing metadata
    print(f"Loading metadata from {METADATA_FILE}...")
    with open(METADATA_FILE) as f:
        runs = json.load(f)

    print(f"Found {len(runs)} runs with images")

    # Get adata paths from W&B
    wandb_info = get_adata_paths_from_wandb()
    print(f"Got adata paths for {len(wandb_info)} runs")

    # Cache for h5ad category lookups (same file might be used multiple times)
    category_cache = {}

    # Process each run
    print("\nExtracting color legends...")
    success_count = 0

    for run in tqdm(runs):
        run_id = run['id']
        label_key = run.get('label_key_from_config', '')

        if run_id not in wandb_info:
            run['color_legend'] = None
            continue

        info = wandb_info[run_id]
        adata_path = info['adata_path']

        # Use label_key from config if available
        if not label_key:
            label_key = info['label_key']

        # Check cache
        cache_key = f"{adata_path}:{label_key}"
        if cache_key in category_cache:
            categories = category_cache[cache_key]
        else:
            # Check if file exists
            if not Path(adata_path).exists():
                run['color_legend'] = None
                continue

            categories = extract_categories_from_h5ad(adata_path, label_key)
            category_cache[cache_key] = categories

        if categories:
            # Build color legend
            n_cats = len(categories)
            legend = []
            for i, cat in enumerate(categories):
                color = get_color_for_index(i, n_cats)
                legend.append({'category': cat, 'color': color})

            run['color_legend'] = legend
            run['adata_path'] = adata_path
            success_count += 1
        else:
            run['color_legend'] = None

    # Save updated metadata (convert numpy types to Python types)
    print(f"\nSaving metadata with color legends...")

    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    runs_clean = convert_numpy(runs)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(runs_clean, f, indent=2)

    print(f"\nDone!")
    print(f"  Extracted legends for {success_count}/{len(runs)} runs")
    print(f"  Saved to: {OUTPUT_FILE}")

    # Show sample
    if success_count > 0:
        sample = next(r for r in runs if r.get('color_legend'))
        print(f"\nSample legend for {sample['name'][:50]}...")
        print(f"  Label key: {sample.get('label_key_from_config')}")
        for item in sample['color_legend'][:5]:
            print(f"    {item['category']}: {item['color']}")
        if len(sample['color_legend']) > 5:
            print(f"    ... and {len(sample['color_legend']) - 5} more")

if __name__ == "__main__":
    main()
