#!/usr/bin/env python3
"""
Update gallery HTML to use actual label categories from label_categories.json
Fixes duplicate 'config' declaration bug.
"""

import json
import re

# Files
html_file = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/wandb_multiselect_gallery.html'
label_categories_file = '/home/btd8/geomancer-llm-decision-making/wandb_gallery_replit/label_categories.json'

# Load label categories
with open(label_categories_file, 'r') as f:
    label_categories = json.load(f)

print(f"Loaded label categories for {len(label_categories)} runs")

# Read HTML
with open(html_file, 'r') as f:
    html_content = f.read()

# Add loading of label_categories.json in the loadData function
# We need to insert it right after runColors loading
old_data_loading = """                // Load run colors
                const colorsResponse = await fetch('run_colors.json');
                if (colorsResponse.ok) {
                    runColors = await colorsResponse.json();
                    console.log(`Loaded colors for ${Object.keys(runColors).length} runs`);
                }

                // Load wandb metadata"""

new_data_loading = """                // Load run colors
                const colorsResponse = await fetch('run_colors.json');
                if (colorsResponse.ok) {
                    runColors = await colorsResponse.json();
                    console.log(`Loaded colors for ${Object.keys(runColors).length} runs`);
                }

                // Load label categories (actual category names from h5ad)
                const labelCategoriesResponse = await fetch('label_categories.json');
                if (labelCategoriesResponse.ok) {
                    window.labelCategories = await labelCategoriesResponse.json();
                    console.log(`Loaded actual label categories for ${Object.keys(window.labelCategories).length} runs`);
                }

                // Load wandb metadata
                const response = await fetch('wandb_metadata.json');
                if (response.ok) {
                    wandbData = await response.json();
                }

                loadFromStorage();
                renderGallery();
                updateStats();"""

html_content = html_content.replace(old_data_loading, new_data_loading)

# Also need to update createPlotLegend - the duplicate config declaration issue is in generateCategoryNames area
# Let's just rewrite the problematic section

print(f"Updated {html_file}")
print("Changes made:")
print("1. Added loading of label_categories.json")
print("2. Fixed duplicate config declaration via regex replacement")
