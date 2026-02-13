#!/usr/bin/env python3
"""
Update gallery HTML to use actual label categories from label_categories.json
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
# Find the loadData function and add label_categories loading
old_data_loading = """async function loadData() {
            try {
                // Load run colors
                const colorsResponse = await fetch('run_colors.json');
                if (colorsResponse.ok) {
                    runColors = await colorsResponse.json();
                    console.log(`Loaded colors for ${Object.keys(runColors).length} runs`);
                }

                // Load wandb metadata
                const response = await fetch('wandb_metadata.json');
                if (response.ok) {
                    wandbData = await response.json();
                }

                loadFromStorage();
                renderGallery();
                updateStats();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('gallery').innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; padding: 50px; color: #e74c3c;">
                        Error loading gallery data. Please refresh the page.
                    </div>`;
            }
        }"""

new_data_loading = """async function loadData() {
            try {
                // Load run colors
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
                updateStats();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('gallery').innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; padding: 50px; color: #e74c3c;">
                        Error loading gallery data. Please refresh the page.
                    </div>`;
            }
        }"""

# Replace the old loadData function
html_content = html_content.replace(old_data_loading, new_data_loading)

# Now update the createPlotLegend function to use actual categories
# Find the generateCategoryNames function and update it
old_generate = """function generateCategoryNames(labelKey, count) {
            const key = labelKey.toLowerCase();
            const names = [];"""

new_generate = """function generateCategoryNames(labelKey, count, runId) {
            const key = labelKey.toLowerCase();
            const names = [];

            // Check if we have actual categories for this run
            if (window.labelCategories && window.labelCategories[runId]) {
                const actualCategories = window.labelCategories[runId].categories || [];
                const numActual = Math.min(count, actualCategories.length);
                for (let i = 0; i < numActual; i++) {
                    names.push(actualCategories[i]);
                }
                // Pad with generic names if needed
                for (let i = numActual; i < count; i++) {
                    names.push(`Category ${i + 1}`);
                }
                return names;
            }

            // Fall back to generic names if no actual categories
            if (key.includes('development_stage') || key.includes('stage')) {"""

html_content = html_content.replace(old_generate, new_generate)

# Update calls to generateCategoryNames to pass runId
# Find where generateCategoryNames is called and update
old_call_pattern = r'generateCategoryNames\(actualLabelKey, numCategories\)'
new_call_pattern = 'generateCategoryNames(actualLabelKey, numCategories, runId)'

html_content = re.sub(old_call_pattern, new_call_pattern, html_content)

# Also need to pass runId to createPlotLegend
old_plot_legend = 'function createPlotLegend(runData, runInfo) {'
new_plot_legend = '''function createPlotLegend(runData, runInfo) {
            const config = runData.config || {};
            const runId = runData.run_id || runData.wandb_run_id;

            // Get actual colors and categories from label_categories.json
            const actualRunColors = runColors[runId] || runInfo;
            const hasActualColors = actualRunColors && actualRunColors.colors && actualRunColors.colors.length > 0;

            // Get actual categories from label_categories.json
            let actualCategories = null;
            if (window.labelCategories && window.labelCategories[runId]) {
                actualCategories = window.labelCategories[runId].categories || [];
            }

            const actualLabelKey = config?.data?.value?.label_key || runData.label_key || 'N/A';
            const numCategories = actualCategories ? actualCategories.length :
                               (actualRunColors?.num_categories || (hasActualColors ? actualRunColors.colors.length : 5));'''

html_content = html_content.replace(old_plot_legend, new_plot_legend)

# Write updated HTML
with open(html_file, 'w') as f:
    f.write(html_content)

print(f"Updated {html_file}")
print("Changes made:")
print("1. Added loading of label_categories.json")
print("2. Updated generateCategoryNames to use actual categories from label_categories.json")
print("3. Updated createPlotLegend to use actual categories")
