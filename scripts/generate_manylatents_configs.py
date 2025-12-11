#!/usr/bin/env python3
"""
Generate manyLatents experiment configs for all CELLxGENE datasets.

This script creates experiment configuration files for running PHATE
dimensionality reduction on all available H5AD datasets.
"""

import os
from pathlib import Path
import yaml
import argparse

# Default paths
DEFAULT_DATA_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/subsampled")
MANYLATENTS_DIR = Path("/home/btd8/manylatents")
EXPERIMENT_DIR = MANYLATENTS_DIR / "manylatents/configs/experiment/cellxgene"
OUTPUT_BASE = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/manylatents_outputs")

def clean_name(filename):
    """Convert filename to a clean experiment name."""
    name = filename.replace('.h5ad', '').replace(' ', '_').replace('-', '_')
    # Remove special characters
    name = ''.join(c for c in name if c.isalnum() or c == '_')
    return name

def create_experiment_config(h5ad_path, output_dir):
    """Create an experiment config for a single dataset."""
    filename = h5ad_path.name
    exp_name = clean_name(filename)
    dataset_id = filename.replace('.h5ad', '')

    config = {
        "name": f"phate_{exp_name}",
        "seed": 42,
        "project": "cellxgene_phate",
        "data": {
            "adata_path": str(h5ad_path),
            "label_key": None  # Will use all available obs columns
        },
        "algorithms": {
            "latent": {
                "n_components": 2
            }
        }
    }

    return config, exp_name

def main():
    parser = argparse.ArgumentParser(description="Generate manyLatents experiment configs")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                       help=f"Directory with H5AD files (default: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)

    # Create directories
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Find all H5AD files
    h5ad_files = list(DATA_DIR.glob("*.h5ad"))
    print(f"Data directory: {DATA_DIR}")
    print(f"Found {len(h5ad_files)} H5AD files")

    # Generate experiment configs
    config_list = []
    for h5ad_path in sorted(h5ad_files):
        config, exp_name = create_experiment_config(h5ad_path, OUTPUT_BASE)

        # Write config file
        config_path = EXPERIMENT_DIR / f"{exp_name}.yaml"
        with open(config_path, 'w') as f:
            # Write the @package directive as a comment
            f.write("# @package _global_\n")
            # Write name
            f.write(f"name: {config['name']}\n\n")
            # Write defaults
            f.write("defaults:\n")
            f.write("  - override /algorithms/latent: phate\n")
            f.write("  - override /data: cellxgene_dataset\n")
            f.write("  - override /callbacks/embedding: default\n")
            f.write("  - override /metrics: test_metric\n\n")
            # Write the rest
            f.write(f"seed: {config['seed']}\n")
            f.write(f"project: {config['project']}\n\n")
            # Write data section
            f.write("data:\n")
            f.write(f"  adata_path: {config['data']['adata_path']}\n")
            f.write(f"  label_key: {config['data']['label_key']}\n\n")
            # Write algorithms section
            f.write("algorithms:\n")
            f.write("  latent:\n")
            f.write(f"    n_components: {config['algorithms']['latent']['n_components']}\n")

        config_list.append({
            "name": exp_name,
            "h5ad_file": h5ad_path.name,
            "config_path": str(config_path)
        })

        print(f"Created config: {exp_name}")

    # Create master list file
    master_list_path = MANYLATENTS_DIR / "cellxgene_experiments.yaml"
    with open(master_list_path, 'w') as f:
        yaml.dump({"experiments": config_list}, f, default_flow_style=False)

    print(f"\n✓ Created {len(config_list)} experiment configs")
    print(f"✓ Master list saved to: {master_list_path}")
    print(f"✓ Outputs will be saved to: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
