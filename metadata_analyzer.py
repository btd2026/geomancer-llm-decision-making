#!/usr/bin/env python3
"""
Metadata Analyzer for PHATE Gallery

Dynamically extracts actual categories and colors from W&B metadata and run_colors.json
instead of using hardcoded assumptions. Each run can have different categories even with the
same label_key.
"""

import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple


class MetadataAnalyzer:
    """Analyzes W&B metadata and extracted colors to build dynamic dataset info."""

    def __init__(self, wandb_metadata_path: str, run_colors_path: str):
        """
        Args:
            wandb_metadata_path: Path to wandb_metadata.json
            run_colors_path: Path to run_colors.json
        """
        self.wandb_metadata_path = wandb_metadata_path
        self.run_colors_path = run_colors_path
        self.wandb_data = {}
        self.color_data = {}

    def load_all_data(self) -> Tuple[dict, dict]:
        """Load both metadata sources."""
        # Load W&B metadata
        if os.path.exists(self.wandb_metadata_path):
            with open(self.wandb_metadata_path, 'r') as f:
                self.wandb_data = json.load(f)

        # Load extracted colors
        if os.path.exists(self.run_colors_path):
            with open(self.run_colors_path, 'r') as f:
                self.color_data = json.load(f)

        return self.wandb_data, self.color_data

    def get_actual_label_key(self, run_data: dict) -> str:
        """Extract the actual label_key used for coloring."""
        # Check config data.value.label_key (what was actually used)
        config = run_data.get('config', {})
        data_value = config.get('data', {}).get('value', {})

        # The actual label_key is in config.data.value.label_key
        actual_label_key = data_value.get('label_key', '')

        if not actual_label_key:
            # Fallback to top-level label_key
            actual_label_key = run_data.get('label_key', '')

        return actual_label_key

    def extract_dataset_info(self) -> Dict[str, dict]:
        """Extract dataset information grouped by label_key with real categories."""
        dataset_info = {}
        run_specific_info = {}

        print("Analyzing W&B metadata...")

        for run_id, run_data in self.wandb_data.items():
            label_key = self.get_actual_label_key(run_data)

            # Skip runs without proper data
            if not label_key:
                continue

            color_info = self.color_data.get(run_id, {})
            num_categories = color_info.get('num_categories', 0)
            colors = color_info.get('colors', [])

            # Store per-run information
            run_specific_info[run_id] = {
                'label_key': label_key,
                'num_categories': num_categories,
                'colors': colors,
                'dataset_name': run_data.get('dataset_name', ''),
                'subtitle': run_data.get('subtitle', '')
            }

            # Initialize label_key entry if needed
            if label_key not in dataset_info:
                dataset_info[label_key] = {
                    'runs': [],
                    'category_counts': [],
                    'example_colors': []
                }

            # Add this run to the label_key's run list
            dataset_info[label_key]['runs'].append(run_id)
            dataset_info[label_key]['category_counts'].append(num_categories)

            # Store example colors (first 8 or num_categories if fewer)
            if colors:
                example_colors = colors[:min(8, num_categories)]
                dataset_info[label_key]['example_colors'].extend(example_colors)

        return self.normalize_dataset_info(dataset_info), run_specific_info

    def normalize_dataset_info(self, dataset_info: dict) -> dict:
        """Normalize and clean up dataset info."""
        normalized = {}

        for label_key, info in dataset_info.items():
            # Get unique category counts
            unique_counts = sorted(set(info['category_counts']), reverse=True)

            normalized[label_key] = {
                'runs': info['runs'],
                'category_counts': unique_counts,
                'min_categories': min(unique_counts) if unique_counts else 0,
                'max_categories': max(unique_counts) if unique_counts else 0,
                'example_colors': info.get('example_colors', [])[:8]  # Limit to 8 examples
            }

        return normalized

    def get_run_info(self, run_id: str) -> dict:
        """Get specific information for a single run."""
        if run_id in self.wandb_data:
            run_data = self.wandb_data[run_id]
            label_key = self.get_actual_label_key(run_data)
            color_info = self.color_data.get(run_id, {})

            return {
                'run_id': run_id,
                'dataset_name': run_data.get('dataset_name', ''),
                'label_key': label_key,
                'num_categories': color_info.get('num_categories', 0),
                'colors': color_info.get('colors', []),
                'subtitle': run_data.get('subtitle', ''),
                'image_path': f"wandb_gallery_replit/images/{run_id}.png"
            }
        return {}

    def get_summary_stats(self) -> dict:
        """Get summary statistics about the datasets."""
        if not self.wandb_data:
            return {}

        dataset_info, _ = self.extract_dataset_info()

        return {
            'total_runs': len(self.wandb_data),
            'total_label_keys': len(dataset_info),
            'runs_per_label_key': {
                key: len(info['runs']) for key, info in dataset_info.items()
            },
            'category_variation': {
                key: {
                    'min': info['min_categories'],
                    'max': info['max_categories']
                }
                for key, info in dataset_info.items()
            },
            'runs_with_color_data': len(self.color_data)
        }

    def export_for_gallery(self, output_path: str) -> None:
        """Export analyzed data in format suitable for gallery generation."""
        dataset_info, run_specific_info = self.extract_dataset_info()

        output_data = {
            'dataset_info': dataset_info,
            'run_specific_info': run_specific_info,
            'summary_stats': self.get_summary_stats()
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Exported gallery data to {output_path}")
        print(f"  - {len(dataset_info)} label_keys")
        print(f"  - {len(run_specific_info)} runs with specific info")
        print(f"  - Summary: {output_data['summary_stats']}")


def main():
    """Main execution function."""
    wandb_metadata = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/wandb_metadata.json'
    run_colors = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/run_colors.json'
    output_path = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/gallery_data.json'

    analyzer = MetadataAnalyzer(wandb_metadata, run_colors)

    # Load and analyze data
    analyzer.load_all_data()

    # Print summary
    stats = analyzer.get_summary_stats()
    print("\n=== Metadata Analysis Summary ===")
    print(f"Total runs: {stats['total_runs']}")
    print(f"Label keys found: {stats['total_label_keys']}")
    print(f"\nRuns per label key:")
    for key, count in stats.get('runs_per_label_key', {}).items():
        print(f"  {key}: {count}")

    print(f"\nCategory variation by label_key:")
    for key, variation in stats.get('category_variation', {}).items():
        print(f"  {key}: {variation['min']}-{variation['max']} categories")

    # Export for gallery
    analyzer.export_for_gallery(output_path)

    print(f"\nData exported to: {output_path}")


if __name__ == "__main__":
    main()
