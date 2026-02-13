#!/usr/bin/env python3
"""
Extract actual colors used in PHATE plots for each W&B run.
This will create a mapping of run_id -> actual colors used in the plot.
"""

import json
import os
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.colors as mcolors

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    """Convert RGB tuple to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))

def extract_colors_from_image(image_path, num_colors=10):
    """Extract the most common colors from an image, excluding background"""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')

        # Get pixel data
        pixels = np.array(img)
        pixels = pixels.reshape(-1, 3)

        # Remove white/light background colors (common in matplotlib)
        # Remove pixels that are too close to white/gray
        mask = ~((pixels[:, 0] > 240) & (pixels[:, 1] > 240) & (pixels[:, 2] > 240))  # Remove white
        mask = mask & ~((pixels[:, 0] < 50) & (pixels[:, 1] < 50) & (pixels[:, 2] < 50))  # Remove black text

        filtered_pixels = pixels[mask]

        if len(filtered_pixels) == 0:
            return []

        # Count color frequency
        colors_tuples = [tuple(pixel) for pixel in filtered_pixels]
        color_counts = Counter(colors_tuples)

        # Get most common colors, excluding very similar ones
        unique_colors = []
        for color, count in color_counts.most_common(num_colors * 3):  # Get more to filter similar ones
            rgb = np.array(color)

            # Check if this color is similar to any we already have
            is_similar = False
            for existing_color in unique_colors:
                existing_rgb = np.array(hex_to_rgb(existing_color))
                # Calculate color distance
                distance = np.linalg.norm(rgb - existing_rgb)
                if distance < 30:  # Threshold for similar colors
                    is_similar = True
                    break

            if not is_similar and len(unique_colors) < num_colors:
                unique_colors.append(rgb_to_hex(rgb))

        return unique_colors[:num_colors]

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

def main():
    # Load W&B metadata
    metadata_path = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/wandb_metadata.json'
    images_dir = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/images'

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    color_mappings = {}

    print("Extracting colors from PHATE plots...")

    for run_id, run_data in metadata.items():
        image_path = os.path.join(images_dir, f"{run_id}.png")

        if os.path.exists(image_path):
            print(f"Processing {run_id}...")

            # Get number of categories for this run
            num_categories = 5  # default
            try:
                if 'config' in run_data and run_data['config']:
                    args = run_data['config'].get('_wandb', {}).get('value', {}).get('e', {})
                    for env_data in args.values():
                        if isinstance(env_data, dict) and 'args' in env_data:
                            for arg in env_data['args']:
                                if '+label_categories=' in arg:
                                    num_categories = int(arg.split('=')[1])
                                    break
            except Exception as e:
                print(f"Could not extract categories for {run_id}: {e}")

            # Extract colors from image
            colors = extract_colors_from_image(image_path, num_categories)

            if colors:
                color_mappings[run_id] = {
                    'colors': colors,
                    'num_categories': num_categories,
                    'label_key': run_data.get('config', {}).get('data', {}).get('value', {}).get('label_key', ''),
                    'dataset_name': run_data.get('dataset_name', '')
                }
                print(f"  Found {len(colors)} colors: {colors[:3]}...")
            else:
                print(f"  No colors found for {run_id}")
        else:
            print(f"Image not found for {run_id}: {image_path}")

    # Save color mappings
    output_path = '/home/btd8/geomancer-llm-decision-making/run_colors.json'
    with open(output_path, 'w') as f:
        json.dump(color_mappings, f, indent=2)

    print(f"\nColor mappings saved to {output_path}")
    print(f"Processed {len(color_mappings)} runs")

if __name__ == "__main__":
    main()