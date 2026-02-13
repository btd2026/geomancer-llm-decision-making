#!/usr/bin/env python3
"""Create an enhanced HTML gallery for W&B PHATE visualizations with color info and labeling."""

from pathlib import Path
import base64
import json
import argparse

IMAGES_DIR = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery/images_new")
METADATA_FILE = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery/runs_metadata_with_legends.json")
OUTPUT_FILE = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery/wandb_labeling_gallery.html")
LABELS_CSV = Path("/home/btd8/llm-paper-analyze/data/wandb_gallery/wandb_labels.csv")

def image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def load_existing_labels(labels_path):
    """Load existing labels from CSV file."""
    import pandas as pd
    if not labels_path.exists():
        print(f"No existing labels file found at {labels_path}")
        return {}

    df = pd.read_csv(labels_path)
    labels = {}

    for _, row in df.iterrows():
        run_id = row.get('run_id', '')
        if not run_id:
            continue
        labels[run_id] = {
            'primary': row.get('primary_structure', ''),
            'density': row.get('density_pattern', ''),
            'branching': row.get('branch_quality', ''),
            'quality': row.get('overall_quality', ''),
            'n_components': str(row.get('n_components', '')) if pd.notna(row.get('n_components', '')) else '',
            'n_clusters': str(row.get('n_clusters', '')) if pd.notna(row.get('n_clusters', '')) else '',
            'flagged': row.get('flagged', False) == True or str(row.get('flagged', '')).lower() == 'true',
            'notes': str(row.get('notes', '')) if pd.notna(row.get('notes', '')) else '',
            'color_info': str(row.get('color_info', '')) if pd.notna(row.get('color_info', '')) else ''
        }
        # Clean up NaN values
        for key in labels[run_id]:
            if key != 'flagged' and isinstance(labels[run_id][key], float):
                labels[run_id][key] = ''
            if labels[run_id][key] == 'nan':
                labels[run_id][key] = ''

    print(f"Loaded {len(labels)} existing labels from {labels_path}")
    return labels

def extract_label_key(run_name):
    """Extract the label_key from run name (last segment after __)."""
    parts = run_name.split('__')
    if len(parts) >= 2:
        return parts[-1]
    return ''

def hex_to_color_name(hex_color):
    """Convert hex color to English color name based on viridis colormap."""
    import colorsys

    # Parse hex color
    hex_clean = hex_color.lstrip('#').lower()
    if len(hex_clean) != 6:
        return hex_color

    r = int(hex_clean[0:2], 16) / 255
    g = int(hex_clean[2:4], 16) / 255
    b = int(hex_clean[4:6], 16) / 255

    # Convert to HSV for better color naming
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360

    # Determine color name based on hue and value
    # Viridis goes: dark purple -> blue -> teal -> green -> yellow
    if v < 0.3:
        return 'dark purple'
    elif h_deg < 60:  # Yellow range
        if s < 0.5:
            return 'light yellow'
        return 'yellow'
    elif h_deg < 90:  # Yellow-green
        return 'yellow-green'
    elif h_deg < 150:  # Green range
        if v > 0.7:
            return 'bright green'
        return 'green'
    elif h_deg < 200:  # Teal/cyan range
        return 'teal'
    elif h_deg < 260:  # Blue range
        if v < 0.5:
            return 'dark blue'
        return 'blue'
    elif h_deg < 300:  # Purple range
        if v < 0.4:
            return 'dark purple'
        return 'purple'
    else:  # Magenta/pink
        return 'magenta'


def create_gallery(existing_labels=None):
    if existing_labels is None:
        existing_labels = {}

    print("Loading W&B run metadata...")

    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        runs = json.load(f)

    print(f"Found {len(runs)} W&B runs")

    # Filter to only runs with existing images
    valid_runs = []
    for run in runs:
        img_path = Path(run['image_path'])
        if img_path.exists():
            valid_runs.append(run)

    print(f"Found {len(valid_runs)} runs with images")

    # Build initial data from existing labels
    initial_data = {
        'primary': {},
        'density': {},
        'branching': {},
        'quality': {},
        'n_components': {},
        'n_clusters': {},
        'flagged': {},
        'notes': {}
    }

    # HTML template with enhanced annotation system
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>W&B PHATE Structure Labeling</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .header h1 { margin: 0 0 5px 0; }
        .header p { margin: 0; opacity: 0.8; font-size: 14px; }
        .progress { background: #ecf0f1; border-radius: 10px; height: 20px; margin: 15px 0 10px 0; }
        .progress-bar { background: #27ae60; height: 100%; border-radius: 10px; transition: width 0.3s; }
        .stats { display: flex; gap: 15px; margin: 10px 0; flex-wrap: wrap; }
        .stat { background: #34495e; padding: 8px 15px; border-radius: 5px; font-size: 13px; }
        .stat-group { background: #1a252f; padding: 10px 15px; border-radius: 5px; }
        .stat-group-title { font-size: 11px; opacity: 0.7; margin-bottom: 5px; }
        .stat-group-items { display: flex; gap: 10px; flex-wrap: wrap; }
        .stat-item { font-size: 12px; }
        .controls { background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .controls button { padding: 10px 20px; margin-right: 10px; margin-bottom: 5px; cursor: pointer; border: none; border-radius: 5px; font-size: 14px; }
        .controls button:hover { opacity: 0.9; }
        .btn-export { background: #27ae60; color: white; }
        .btn-save-html { background: #9b59b6; color: white; }
        .btn-filter { background: #3498db; color: white; }
        .btn-filter.active { background: #2980b9; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2); }
        .btn-clear { background: #e74c3c; color: white; }
        .btn-flagged { background: #f39c12; color: white; }
        .btn-legend { background: #8e44ad; color: white; }
        .search-box { padding: 10px 15px; border: 1px solid #ddd; border-radius: 5px; width: 250px; margin-right: 15px; }
        .color-controls { margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
        .color-controls label { font-size: 13px; font-weight: 500; color: #555; }
        .color-controls select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 13px; min-width: 180px; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); position: relative; transition: all 0.2s; }
        .card.labeled { border-left: 5px solid #27ae60; }
        .card.complete { border-left: 5px solid #9b59b6; }
        .card.flagged { border: 3px solid #f39c12 !important; }
        .card.flagged::before { content: "FLAG"; position: absolute; top: 10px; right: 10px; background: #f39c12; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; z-index: 10; }
        .card img { width: 100%; height: auto; display: block; cursor: pointer; }
        .card img:hover { opacity: 0.95; }
        .card-body { padding: 15px; }

        /* Label indicator badge */
        .label-badge { position: absolute; top: 10px; left: 10px; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; color: white; z-index: 10; display: none; }
        .card.labeled .label-badge { display: block; }

        /* Color indicator bar at top of card */
        .color-bar { height: 8px; margin: -15px -15px 15px -15px; background: transparent; transition: background 0.3s; }

        .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px; gap: 10px; }
        .card-title { font-size: 14px; font-weight: bold; color: #2c3e50; flex: 1; }
        .card-subtitle { font-size: 12px; color: #7f8c8d; margin-bottom: 4px; }
        .card-label-key { font-size: 11px; color: #8e44ad; margin-bottom: 8px; background: #f5eef8; padding: 3px 8px; border-radius: 3px; display: inline-block; font-weight: 600; }
        .flag-btn { background: none; border: 2px solid #f39c12; color: #f39c12; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; transition: all 0.2s; white-space: nowrap; }
        .flag-btn:hover { background: #fef5e7; }
        .flag-btn.flagged { background: #f39c12; color: white; }
        .card-id { font-size: 11px; color: #95a5a6; word-break: break-all; margin-bottom: 8px; font-family: monospace; }

        /* Color legend section */
        .color-legend-section { background: #fdf6e3; border: 1px solid #f0e6d2; border-radius: 5px; margin-bottom: 12px; }
        .color-legend-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; cursor: pointer; }
        .color-legend-header:hover { background: #f5edd5; }
        .color-legend-title { font-size: 11px; font-weight: 600; color: #8e6f3a; display: flex; align-items: center; gap: 5px; }
        .color-legend-label { font-size: 10px; color: #8e44ad; background: #f5eef8; padding: 2px 6px; border-radius: 3px; margin-left: 5px; }
        .color-legend-toggle { font-size: 10px; color: #8e6f3a; }
        .color-legend-body { padding: 8px 10px; border-top: 1px solid #f0e6d2; display: none; }
        .color-legend-body.open { display: block; }
        .color-legend-list { margin: 0; padding: 0; list-style: none; }
        .color-legend-item { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 11px; color: #5d4e37; }
        .color-legend-swatch { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; border: 1px solid rgba(0,0,0,0.1); }
        .color-legend-cat { flex: 1; }
        .color-legend-name { font-weight: 600; color: #8e6f3a; }

        /* Primary structure buttons */
        .primary-label { margin-bottom: 12px; }
        .primary-label-title { font-size: 11px; color: #7f8c8d; margin-bottom: 6px; font-weight: 600; }
        .label-buttons { display: flex; gap: 5px; flex-wrap: wrap; }
        .label-btn { padding: 6px 10px; border: 2px solid #ddd; background: white; cursor: pointer; border-radius: 5px; font-size: 11px; transition: all 0.2s; }
        .label-btn:hover { border-color: #3498db; background: #f8f9fa; }
        .label-btn.selected { color: white; }
        .label-btn[data-label="clusters"].selected { background: #e74c3c; border-color: #e74c3c; }
        .label-btn[data-label="simple_trajectory"].selected { background: #f39c12; border-color: #f39c12; }
        .label-btn[data-label="horseshoe"].selected { background: #d35400; border-color: #d35400; }
        .label-btn[data-label="bifurcation"].selected { background: #27ae60; border-color: #27ae60; }
        .label-btn[data-label="multi_branch"].selected { background: #3498db; border-color: #3498db; }
        .label-btn[data-label="complex_tree"].selected { background: #9b59b6; border-color: #9b59b6; }
        .label-btn[data-label="cyclic"].selected { background: #1abc9c; border-color: #1abc9c; }
        .label-btn[data-label="diffuse"].selected { background: #95a5a6; border-color: #95a5a6; }

        /* Secondary annotation dropdowns */
        .secondary-labels { border-top: 1px solid #eee; padding-top: 12px; margin-top: 8px; }
        .annotation-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .annotation-item { }
        .annotation-item label { font-size: 11px; color: #7f8c8d; display: block; margin-bottom: 3px; font-weight: 500; }
        .annotation-item select { width: 100%; padding: 6px 8px; font-size: 11px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer; }
        .annotation-item select:focus { border-color: #3498db; outline: none; }
        .annotation-item select.filled { border-color: #27ae60; background: #f0fff4; }

        /* Notes field */
        .notes-section { margin-top: 10px; }
        .notes { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px; font-size: 12px; resize: vertical; min-height: 40px; }
        .notes:focus { border-color: #3498db; outline: none; }

        .hidden { display: none !important; }

        /* Legend modal */
        .legend-modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); }
        .legend-modal-content { background: white; max-width: 600px; margin: 50px auto; border-radius: 10px; overflow: hidden; }
        .legend-modal-header { background: #2c3e50; color: white; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }
        .legend-modal-header h2 { margin: 0; font-size: 18px; }
        .legend-modal-close { font-size: 24px; cursor: pointer; opacity: 0.8; }
        .legend-modal-close:hover { opacity: 1; }
        .legend-modal-body { padding: 20px; max-height: 70vh; overflow-y: auto; }

        .legend-section { margin-bottom: 20px; }
        .legend-section h3 { font-size: 14px; color: #2c3e50; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        .legend-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
        .legend-item { display: flex; align-items: center; gap: 8px; font-size: 12px; }
        .legend-color { width: 20px; height: 20px; border-radius: 4px; flex-shrink: 0; }
        .legend-text { color: #333; }

        /* Header legend mini */
        .header-legend { margin: 15px 0; padding: 15px; background: #1a252f; border-radius: 8px; }
        .header-legend-title { font-size: 12px; font-weight: 600; margin-bottom: 10px; opacity: 0.9; }
        .header-legend-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
        .header-legend-item { display: flex; align-items: center; gap: 8px; font-size: 11px; }
        .header-legend-color { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }

        /* Image modal */
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); }
        .modal-content { max-width: 90%; max-height: 90%; margin: auto; display: block; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
        .modal-close { position: absolute; top: 20px; right: 30px; color: white; font-size: 40px; cursor: pointer; }
        .modal-nav { position: absolute; top: 50%; transform: translateY(-50%); color: white; font-size: 50px; cursor: pointer; user-select: none; padding: 20px; opacity: 0.7; }
        .modal-nav:hover { opacity: 1; }
        .modal-prev { left: 10px; }
        .modal-next { right: 10px; }
        .modal-info { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); color: white; background: rgba(0,0,0,0.7); padding: 15px 25px; border-radius: 8px; text-align: center; }
        .modal-info-title { font-weight: bold; margin-bottom: 5px; }
        .modal-info-subtitle { font-size: 12px; opacity: 0.8; }
        .modal-info-label-key { font-size: 11px; background: rgba(52, 152, 219, 0.8); padding: 2px 8px; border-radius: 3px; margin-top: 5px; display: inline-block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>W&B PHATE Structure Labeling</h1>
        <p>Geomancer PHATE visualizations from CELLxGENE datasets</p>
        <div class="progress"><div class="progress-bar" id="progressBar" style="width: 0%"></div></div>
        <div class="stats">
            <div class="stat"><strong><span id="labeledCount">0</span></strong> / ''' + str(len(valid_runs)) + ''' primary labeled</div>
            <div class="stat"><strong><span id="completeCount">0</span></strong> fully annotated</div>
            <div class="stat" style="background:#f39c12"><strong><span id="flaggedCount">0</span></strong> flagged</div>
        </div>
        <div class="stats">
            <div class="stat-group">
                <div class="stat-group-title">PRIMARY STRUCTURE</div>
                <div class="stat-group-items">
                    <span class="stat-item">Clusters: <strong id="count_clusters">0</strong></span>
                    <span class="stat-item">Simple: <strong id="count_simple_trajectory">0</strong></span>
                    <span class="stat-item">Horseshoe: <strong id="count_horseshoe">0</strong></span>
                    <span class="stat-item">Bifurc: <strong id="count_bifurcation">0</strong></span>
                    <span class="stat-item">Multi: <strong id="count_multi_branch">0</strong></span>
                    <span class="stat-item">Complex: <strong id="count_complex_tree">0</strong></span>
                    <span class="stat-item">Cyclic: <strong id="count_cyclic">0</strong></span>
                    <span class="stat-item">Diffuse: <strong id="count_diffuse">0</strong></span>
                </div>
            </div>
        </div>
        <div class="header-legend">
            <div class="header-legend-title">STRUCTURE TYPES & COLORS</div>
            <div class="header-legend-grid">
                <div class="header-legend-item"><div class="header-legend-color" style="background:#e74c3c"></div> Clusters</div>
                <div class="header-legend-item"><div class="header-legend-color" style="background:#f39c12"></div> Simple Trajectory</div>
                <div class="header-legend-item"><div class="header-legend-color" style="background:#d35400"></div> Horseshoe</div>
                <div class="header-legend-item"><div class="header-legend-color" style="background:#27ae60"></div> Bifurcation</div>
                <div class="header-legend-item"><div class="header-legend-color" style="background:#3498db"></div> Multi-branch</div>
                <div class="header-legend-item"><div class="header-legend-color" style="background:#9b59b6"></div> Complex Tree</div>
                <div class="header-legend-item"><div class="header-legend-color" style="background:#1abc9c"></div> Cyclic</div>
                <div class="header-legend-item"><div class="header-legend-color" style="background:#95a5a6"></div> Diffuse</div>
            </div>
        </div>
    </div>

    <div class="controls">
        <input type="text" class="search-box" id="searchBox" placeholder="Search datasets..." onkeyup="filterBySearch()">
        <button class="btn-export" onclick="exportCSV()">Export Labels CSV</button>
        <button class="btn-save-html" onclick="saveToHTML()">Save to HTML</button>
        <button class="btn-legend" onclick="showLegendModal()">Color Legend</button>
        <button class="btn-filter" onclick="filterCards('all')">Show All</button>
        <button class="btn-filter" onclick="filterCards('unlabeled')">Unlabeled Only</button>
        <button class="btn-filter" onclick="filterCards('labeled')">Labeled Only</button>
        <button class="btn-flagged" onclick="filterCards('flagged')">Flagged Only</button>
        <button class="btn-clear" onclick="if(confirm('Clear all labels?')) clearAll()">Clear All</button>

        <div class="color-controls">
            <label>Color cards by:</label>
            <select id="colorBySelect" onchange="applyColorBy(this.value)">
                <option value="none">None (default borders)</option>
                <option value="primary">Primary Structure</option>
                <option value="density">Density Pattern</option>
                <option value="branching">Branch Quality</option>
                <option value="quality">Overall Quality</option>
            </select>
        </div>
    </div>

    <div class="gallery" id="gallery">
'''

    # Add cards for each run
    for idx, run in enumerate(valid_runs):
        run_id = run['id']
        run_name = run['name']
        dataset_name = run['dataset_name']
        subset_name = run.get('subset_name', '')
        # Use label_key from config if available, otherwise extract from run name
        label_key = run.get('label_key_from_config') or extract_label_key(run_name)
        image_path = Path(run['image_path'])

        # Get existing labels
        labels = existing_labels.get(run_id, {})
        primary = labels.get('primary', '')
        density = labels.get('density', '')
        branching = labels.get('branching', '')
        quality = labels.get('quality', '')
        n_components = labels.get('n_components', '')
        n_clusters = labels.get('n_clusters', '')
        is_flagged = labels.get('flagged', False)
        notes = labels.get('notes', '')
        color_info = labels.get('color_info', '')

        # Build color legend from extracted data
        color_legend_items = []
        if run.get('color_legend'):
            for item in run['color_legend']:
                cat = item['category']
                hex_color = item['color']
                color_name = hex_to_color_name(hex_color)
                color_legend_items.append({'category': cat, 'color': color_name, 'hex': hex_color})

        # Populate initial data
        if primary:
            initial_data['primary'][str(idx)] = primary
        if density:
            initial_data['density'][str(idx)] = density
        if branching:
            initial_data['branching'][str(idx)] = branching
        if quality:
            initial_data['quality'][str(idx)] = quality
        if n_components:
            initial_data['n_components'][str(idx)] = n_components
        if n_clusters:
            initial_data['n_clusters'][str(idx)] = n_clusters
        if is_flagged:
            initial_data['flagged'][str(idx)] = True
        if notes:
            initial_data['notes'][str(idx)] = notes

        # Determine card classes
        card_classes = "card"
        if primary:
            card_classes += " labeled"
        if primary and density and branching and quality:
            card_classes += " complete"
        if is_flagged:
            card_classes += " flagged"

        # Convert image to base64
        img_base64 = image_to_base64(image_path)
        img_src = f"data:image/png;base64,{img_base64}"

        # Escape strings
        dataset_name_escaped = (dataset_name or '').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        subset_escaped = (subset_name or '').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        label_key_escaped = (label_key or '').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        color_info_escaped = (color_info or '').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        notes_escaped = (notes or '').replace('<', '&lt;').replace('>', '&gt;')

        # Helper for button selection
        def btn_class(label_type):
            return "label-btn selected" if primary == label_type else "label-btn"

        # Helper for select options
        def option_selected(field_val, opt_val):
            return " selected" if field_val == opt_val else ""

        # Helper for select filled class
        def select_class(field_val):
            return "filled" if field_val else ""

        # Flag button state
        flag_btn_class = "flag-btn flagged" if is_flagged else "flag-btn"
        flag_btn_text = "Flagged" if is_flagged else "Flag"

        # Label badge color
        label_colors = {
            'clusters': '#e74c3c',
            'simple_trajectory': '#f39c12',
            'horseshoe': '#d35400',
            'bifurcation': '#27ae60',
            'multi_branch': '#3498db',
            'complex_tree': '#9b59b6',
            'cyclic': '#1abc9c',
            'diffuse': '#95a5a6'
        }
        badge_color = label_colors.get(primary, '#999')
        badge_text = primary.replace('_', ' ').title() if primary else ''

        # Generate color legend HTML
        if color_legend_items:
            legend_lines = []
            for item in color_legend_items:
                cat_str = str(item['category'])
                cat_escaped = cat_str.replace('<', '&lt;').replace('>', '&gt;')
                legend_lines.append(
                    f'                            <li class="color-legend-item">'
                    f'<span class="color-legend-swatch" style="background:{item["hex"]}"></span>'
                    f'<span class="color-legend-cat">{cat_escaped}</span>'
                    f'<span class="color-legend-name">{item["color"]}</span>'
                    f'</li>'
                )
            color_legend_html = '\n'.join(legend_lines)
        else:
            color_legend_html = '                            <li class="color-legend-item"><span style="color:#999;font-style:italic;">No legend available</span></li>'

        html += f'''
        <div class="{card_classes}" id="card-{idx}" data-id="{run_id}" data-name="{dataset_name_escaped}" data-subset="{subset_escaped}">
            <div class="label-badge" id="badge-{idx}" style="background:{badge_color}">{badge_text}</div>
            <img src="{img_src}" alt="{dataset_name_escaped}" onclick="openModal({idx})">
            <div class="card-body">
                <div class="color-bar" id="colorbar-{idx}"></div>
                <div class="card-header">
                    <div class="card-title">{dataset_name_escaped}</div>
                    <button class="{flag_btn_class}" id="flag-btn-{idx}" onclick="toggleFlag({idx})">{flag_btn_text}</button>
                </div>
                <div class="card-subtitle">{subset_escaped}</div>
                <div class="card-label-key">Label Key: {label_key_escaped}</div>
                <div class="card-id">Run: {run_id}</div>

                <!-- Color Legend Section -->
                <div class="color-legend-section">
                    <div class="color-legend-header" onclick="toggleLegend({idx})">
                        <div class="color-legend-title">
                            <span>&#x1F3A8;</span> Color Legend
                            <span class="color-legend-label">{label_key_escaped}</span>
                        </div>
                        <span class="color-legend-toggle" id="legend-toggle-{idx}">&#9660;</span>
                    </div>
                    <div class="color-legend-body" id="legend-body-{idx}">
                        <ul class="color-legend-list">
{color_legend_html}
                        </ul>
                    </div>
                </div>

                <!-- Primary Structure Type -->
                <div class="primary-label">
                    <div class="primary-label-title">PRIMARY STRUCTURE</div>
                    <div class="label-buttons">
                        <button class="{btn_class('clusters')}" data-label="clusters" onclick="setPrimary({idx}, 'clusters')">Clusters</button>
                        <button class="{btn_class('simple_trajectory')}" data-label="simple_trajectory" onclick="setPrimary({idx}, 'simple_trajectory')">Simple Traj</button>
                        <button class="{btn_class('horseshoe')}" data-label="horseshoe" onclick="setPrimary({idx}, 'horseshoe')">Horseshoe</button>
                        <button class="{btn_class('bifurcation')}" data-label="bifurcation" onclick="setPrimary({idx}, 'bifurcation')">Bifurcation</button>
                        <button class="{btn_class('multi_branch')}" data-label="multi_branch" onclick="setPrimary({idx}, 'multi_branch')">Multi-branch</button>
                        <button class="{btn_class('complex_tree')}" data-label="complex_tree" onclick="setPrimary({idx}, 'complex_tree')">Complex Tree</button>
                        <button class="{btn_class('cyclic')}" data-label="cyclic" onclick="setPrimary({idx}, 'cyclic')">Cyclic</button>
                        <button class="{btn_class('diffuse')}" data-label="diffuse" onclick="setPrimary({idx}, 'diffuse')">Diffuse</button>
                    </div>
                </div>

                <!-- Secondary Annotations -->
                <div class="secondary-labels">
                    <div class="annotation-grid">
                        <div class="annotation-item">
                            <label>Density Pattern</label>
                            <select id="density-{idx}" class="{select_class(density)}" onchange="setAnnotation({idx}, 'density', this.value)">
                                <option value="">-- Select --</option>
                                <option value="uniform"{option_selected(density, 'uniform')}>Uniform</option>
                                <option value="progressive"{option_selected(density, 'progressive')}>Progressive</option>
                                <option value="heterogeneous"{option_selected(density, 'heterogeneous')}>Heterogeneous</option>
                                <option value="dense_core"{option_selected(density, 'dense_core')}>Dense core</option>
                            </select>
                        </div>
                        <div class="annotation-item">
                            <label>Branch Quality</label>
                            <select id="branching-{idx}" class="{select_class(branching)}" onchange="setAnnotation({idx}, 'branching', this.value)">
                                <option value="">-- Select --</option>
                                <option value="none"{option_selected(branching, 'none')}>No branching</option>
                                <option value="clear"{option_selected(branching, 'clear')}>Clear branch</option>
                                <option value="gradual"{option_selected(branching, 'gradual')}>Gradual</option>
                                <option value="multiple"{option_selected(branching, 'multiple')}>Multiple</option>
                            </select>
                        </div>
                        <div class="annotation-item">
                            <label>Overall Quality</label>
                            <select id="quality-{idx}" class="{select_class(quality)}" onchange="setAnnotation({idx}, 'quality', this.value)">
                                <option value="">-- Select --</option>
                                <option value="excellent"{option_selected(quality, 'excellent')}>Excellent</option>
                                <option value="good"{option_selected(quality, 'good')}>Good</option>
                                <option value="fair"{option_selected(quality, 'fair')}>Fair</option>
                                <option value="poor"{option_selected(quality, 'poor')}>Poor</option>
                            </select>
                        </div>
                        <div class="annotation-item">
                            <label>Num Components</label>
                            <select id="n_components-{idx}" class="{select_class(n_components)}" onchange="setAnnotation({idx}, 'n_components', this.value)">
                                <option value="">-- Select --</option>
                                <option value="1"{option_selected(n_components, '1')}>1</option>
                                <option value="2"{option_selected(n_components, '2')}>2</option>
                                <option value="3"{option_selected(n_components, '3')}>3</option>
                                <option value="4"{option_selected(n_components, '4')}>4</option>
                                <option value="5+"{option_selected(n_components, '5+')}>5+</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Notes -->
                <div class="notes-section">
                    <textarea class="notes" id="notes-{idx}" placeholder="Additional notes..." onchange="setNotes({idx}, this.value)">{notes_escaped}</textarea>
                </div>
            </div>
        </div>
'''
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(valid_runs)}")

    # Store metadata for CSV export
    metadata_list = [{"run_id": r['id'], "dataset_name": r['dataset_name'], "subset_name": r.get('subset_name', ''), "label_key": r.get('label_key_from_config') or extract_label_key(r['name']), "idx": i} for i, r in enumerate(valid_runs)]
    metadata_json = json.dumps(metadata_list)
    initial_data_json = json.dumps(initial_data)

    html += '''
    </div>

    <!-- Legend Modal -->
    <div id="legendModal" class="legend-modal" onclick="if(event.target===this)closeLegendModal()">
        <div class="legend-modal-content">
            <div class="legend-modal-header">
                <h2>Color Legend Guide</h2>
                <span class="legend-modal-close" onclick="closeLegendModal()">&times;</span>
            </div>
            <div class="legend-modal-body">
                <div class="legend-section">
                    <h3>Primary Structure Types</h3>
                    <div class="legend-grid">
                        <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div><span class="legend-text">Clusters - Discrete separated groups</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div><span class="legend-text">Simple Trajectory - Single path/arc</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#d35400"></div><span class="legend-text">Horseshoe - U-shaped curve</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div><span class="legend-text">Bifurcation - 2 branches (Y-shape)</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#3498db"></div><span class="legend-text">Multi-branch - 3-4 branches</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div><span class="legend-text">Complex Tree - 5+ branches</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#1abc9c"></div><span class="legend-text">Cyclic - Circular/loop structure</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div><span class="legend-text">Diffuse - No clear structure</span></div>
                    </div>
                </div>
                <div class="legend-section">
                    <h3>Density Patterns</h3>
                    <div class="legend-grid">
                        <div class="legend-item"><div class="legend-color" style="background:#3498db"></div><span class="legend-text">Uniform - Even distribution</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div><span class="legend-text">Progressive - Sparse to dense gradient</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div><span class="legend-text">Heterogeneous - Patchy distribution</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div><span class="legend-text">Dense Core - Core with extensions</span></div>
                    </div>
                </div>
                <div class="legend-section">
                    <h3>Branch Quality</h3>
                    <div class="legend-grid">
                        <div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div><span class="legend-text">None - No visible branching</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div><span class="legend-text">Clear - Distinct branch points</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div><span class="legend-text">Gradual - Smooth divergence</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div><span class="legend-text">Multiple - Many branch points</span></div>
                    </div>
                </div>
                <div class="legend-section">
                    <h3>Overall Quality</h3>
                    <div class="legend-grid">
                        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div><span class="legend-text">Excellent - Clear, interpretable</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#3498db"></div><span class="legend-text">Good - Minor noise</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div><span class="legend-text">Fair - Noisy but structured</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div><span class="legend-text">Poor - Hard to interpret</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Image Modal -->
    <div id="imageModal" class="modal" onclick="if(event.target===this)closeModal()">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <span class="modal-nav modal-prev" onclick="event.stopPropagation();navigateModal(-1)">&#10094;</span>
        <span class="modal-nav modal-next" onclick="event.stopPropagation();navigateModal(1)">&#10095;</span>
        <img class="modal-content" id="modalImg">
        <div class="modal-info" id="modalInfo">
            <div class="modal-info-title" id="modalTitle"></div>
            <div class="modal-info-subtitle" id="modalSubtitle"></div>
            <div class="modal-info-label-key" id="modalLabelKey"></div>
        </div>
    </div>

    <script id="dataScript">
        // DATA_START - initialized with pre-loaded labels
        const data = ''' + initial_data_json + ''';
        // DATA_END
    </script>
    <script>
        const metadata = ''' + metadata_json + ''';
        const totalCards = ''' + str(len(valid_runs)) + ''';
        let currentModalIdx = 0;

        const primaryTypes = ['clusters', 'simple_trajectory', 'horseshoe', 'bifurcation', 'multi_branch', 'complex_tree', 'cyclic', 'diffuse'];

        const labelColors = {
            clusters: '#e74c3c',
            simple_trajectory: '#f39c12',
            horseshoe: '#d35400',
            bifurcation: '#27ae60',
            multi_branch: '#3498db',
            complex_tree: '#9b59b6',
            cyclic: '#1abc9c',
            diffuse: '#95a5a6'
        };

        const colorMaps = {
            primary: labelColors,
            density: {
                uniform: '#3498db',
                progressive: '#9b59b6',
                heterogeneous: '#e74c3c',
                dense_core: '#27ae60'
            },
            branching: {
                none: '#95a5a6',
                clear: '#27ae60',
                gradual: '#f39c12',
                multiple: '#9b59b6'
            },
            quality: {
                excellent: '#27ae60',
                good: '#3498db',
                fair: '#f39c12',
                poor: '#e74c3c'
            }
        };

        let currentColorBy = 'none';

        function setPrimary(idx, label) {
            data.primary[idx] = label;

            const card = document.getElementById('card-' + idx);
            card.querySelectorAll('.label-btn').forEach(btn => {
                btn.classList.remove('selected');
                if (btn.dataset.label === label) {
                    btn.classList.add('selected');
                }
            });

            // Update badge
            const badge = document.getElementById('badge-' + idx);
            badge.style.background = labelColors[label];
            badge.textContent = label.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());

            updateCardStatus(idx);
            updateStats();
            applyColorBy(currentColorBy);
            saveToLocalStorage();
        }

        function setAnnotation(idx, field, value) {
            if (value) {
                data[field][idx] = value;
                document.getElementById(field + '-' + idx).classList.add('filled');
            } else {
                delete data[field][idx];
                document.getElementById(field + '-' + idx).classList.remove('filled');
            }

            updateCardStatus(idx);
            updateStats();
            applyColorBy(currentColorBy);
            saveToLocalStorage();
        }

        function setNotes(idx, value) {
            if (value.trim()) {
                data.notes[idx] = value.trim();
            } else {
                delete data.notes[idx];
            }
            saveToLocalStorage();
        }

        function toggleLegend(idx) {
            const body = document.getElementById('legend-body-' + idx);
            const toggle = document.getElementById('legend-toggle-' + idx);
            body.classList.toggle('open');
            toggle.innerHTML = body.classList.contains('open') ? '&#9650;' : '&#9660;';
        }

        function toggleFlag(idx) {
            if (data.flagged[idx]) {
                delete data.flagged[idx];
            } else {
                data.flagged[idx] = true;
            }

            const card = document.getElementById('card-' + idx);
            const btn = document.getElementById('flag-btn-' + idx);

            card.classList.toggle('flagged', !!data.flagged[idx]);
            btn.classList.toggle('flagged', !!data.flagged[idx]);
            btn.textContent = data.flagged[idx] ? 'Flagged' : 'Flag';

            updateStats();
            saveToLocalStorage();
        }

        function updateCardStatus(idx) {
            const card = document.getElementById('card-' + idx);
            const hasPrimary = data.primary.hasOwnProperty(idx);
            const isComplete = hasPrimary &&
                              data.density.hasOwnProperty(idx) &&
                              data.branching.hasOwnProperty(idx) &&
                              data.quality.hasOwnProperty(idx);

            card.classList.remove('labeled', 'complete');
            if (isComplete) {
                card.classList.add('complete');
            } else if (hasPrimary) {
                card.classList.add('labeled');
            }
        }

        function updateStats() {
            const counts = {};
            primaryTypes.forEach(t => counts[t] = 0);
            let labeled = 0;
            let complete = 0;
            let flagged = Object.keys(data.flagged).length;

            for (const idx in data.primary) {
                const type = data.primary[idx];
                if (counts.hasOwnProperty(type)) {
                    counts[type]++;
                }
                labeled++;

                if (data.density.hasOwnProperty(idx) &&
                    data.branching.hasOwnProperty(idx) &&
                    data.quality.hasOwnProperty(idx)) {
                    complete++;
                }
            }

            document.getElementById('labeledCount').textContent = labeled;
            document.getElementById('completeCount').textContent = complete;
            document.getElementById('flaggedCount').textContent = flagged;
            document.getElementById('progressBar').style.width = (labeled / totalCards * 100) + '%';

            primaryTypes.forEach(type => {
                const el = document.getElementById('count_' + type);
                if (el) el.textContent = counts[type];
            });
        }

        function filterCards(filter) {
            document.querySelectorAll('.card').forEach((card, idx) => {
                const hasPrimary = data.primary.hasOwnProperty(idx);
                const isFlagged = data.flagged.hasOwnProperty(idx);

                if (filter === 'all') {
                    card.classList.remove('hidden');
                } else if (filter === 'unlabeled') {
                    card.classList.toggle('hidden', hasPrimary);
                } else if (filter === 'labeled') {
                    card.classList.toggle('hidden', !hasPrimary);
                } else if (filter === 'flagged') {
                    card.classList.toggle('hidden', !isFlagged);
                }
            });
        }

        function filterBySearch() {
            const query = document.getElementById('searchBox').value.toLowerCase();
            document.querySelectorAll('.card').forEach(card => {
                const name = card.dataset.name.toLowerCase();
                const subset = card.dataset.subset.toLowerCase();
                const matches = name.includes(query) || subset.includes(query);
                card.classList.toggle('hidden', !matches);
            });
        }

        function applyColorBy(field) {
            currentColorBy = field;
            document.getElementById('colorBySelect').value = field;

            document.querySelectorAll('.card').forEach((card, idx) => {
                const colorBar = document.getElementById('colorbar-' + idx);

                if (field === 'none' || !data[field] || !data[field].hasOwnProperty(idx)) {
                    colorBar.style.background = 'transparent';
                } else {
                    const value = data[field][idx];
                    const color = colorMaps[field] && colorMaps[field][value];
                    colorBar.style.background = color || 'transparent';
                }
            });
        }

        function exportCSV() {
            let csv = 'run_id,dataset_name,subset_name,label_key,primary_structure,density_pattern,branch_quality,overall_quality,n_components,flagged,color_info,notes\\n';

            metadata.forEach((item, idx) => {
                const primary = data.primary[idx] || '';
                const density = data.density[idx] || '';
                const branching = data.branching[idx] || '';
                const quality = data.quality[idx] || '';
                const n_comp = data.n_components[idx] || '';
                const flagged = data.flagged[idx] ? 'true' : '';
                const colorInfo = (data.color_info[idx] || '').replace(/"/g, '""').replace(/\\n/g, ' ');
                const notes = (data.notes[idx] || '').replace(/"/g, '""').replace(/\\n/g, ' ');

                csv += `${item.run_id},"${item.dataset_name}","${item.subset_name}","${item.label_key}",${primary},${density},${branching},${quality},${n_comp},${flagged},"${colorInfo}","${notes}"\\n`;
            });

            const blob = new Blob([csv], {type: 'text/csv'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'wandb_phate_labels.csv';
            a.click();
        }

        function saveToHTML() {
            let html = document.documentElement.outerHTML;
            const dataStr = JSON.stringify(data);

            const startMarker = '// DATA_START';
            const endMarker = '// DATA_END';
            const startIdx = html.indexOf(startMarker);
            const endIdx = html.indexOf(endMarker);

            if (startIdx !== -1 && endIdx !== -1) {
                const before = html.substring(0, startIdx);
                const after = html.substring(endIdx + endMarker.length);
                const newDataSection = '// DATA_START - saved with labels baked in\\n        const data = ' + dataStr + ';\\n        // DATA_END';
                html = before + newDataSection + after;
            }

            const blob = new Blob(['<!DOCTYPE html>\\n<html>' + html.substring(html.indexOf('<head>'))], {type: 'text/html'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const timestamp = new Date().toISOString().slice(0,10);
            a.download = 'wandb_gallery_saved_' + timestamp + '.html';
            a.click();
            URL.revokeObjectURL(url);

            alert('HTML saved with ' + Object.keys(data.primary).length + ' labels baked in!');
        }

        function clearAll() {
            Object.keys(data).forEach(key => {
                data[key] = {};
            });

            document.querySelectorAll('.card').forEach((card, idx) => {
                card.classList.remove('labeled', 'complete', 'flagged');
                card.querySelectorAll('.label-btn').forEach(btn => btn.classList.remove('selected'));
                card.querySelectorAll('select').forEach(sel => {
                    sel.value = '';
                    sel.classList.remove('filled');
                });
                card.querySelector('.notes').value = '';
                const flagBtn = document.getElementById('flag-btn-' + idx);
                if (flagBtn) {
                    flagBtn.classList.remove('flagged');
                    flagBtn.textContent = 'Flag';
                }
                const badge = document.getElementById('badge-' + idx);
                badge.style.background = '#999';
                badge.textContent = '';
            });

            updateStats();
            applyColorBy('none');
            saveToLocalStorage();
        }

        function openModal(idx) {
            currentModalIdx = idx;
            const card = document.getElementById('card-' + idx);
            const img = card.querySelector('img');
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImg').src = img.src;
            document.getElementById('modalTitle').textContent = metadata[idx].dataset_name;
            document.getElementById('modalSubtitle').textContent = metadata[idx].subset_name + ' (' + (idx + 1) + '/' + totalCards + ')';
            document.getElementById('modalLabelKey').textContent = 'Label Key: ' + metadata[idx].label_key;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        function navigateModal(direction) {
            currentModalIdx += direction;
            if (currentModalIdx < 0) currentModalIdx = totalCards - 1;
            if (currentModalIdx >= totalCards) currentModalIdx = 0;

            const card = document.getElementById('card-' + currentModalIdx);
            const img = card.querySelector('img');
            document.getElementById('modalImg').src = img.src;
            document.getElementById('modalTitle').textContent = metadata[currentModalIdx].dataset_name;
            document.getElementById('modalSubtitle').textContent = metadata[currentModalIdx].subset_name + ' (' + (currentModalIdx + 1) + '/' + totalCards + ')';
            document.getElementById('modalLabelKey').textContent = 'Label Key: ' + metadata[currentModalIdx].label_key;
        }

        function showLegendModal() {
            document.getElementById('legendModal').style.display = 'block';
        }

        function closeLegendModal() {
            document.getElementById('legendModal').style.display = 'none';
        }

        document.addEventListener('keydown', function(e) {
            const modal = document.getElementById('imageModal');
            if (modal.style.display === 'block') {
                if (e.key === 'Escape') closeModal();
                if (e.key === 'ArrowLeft') navigateModal(-1);
                if (e.key === 'ArrowRight') navigateModal(1);
            }
            if (document.getElementById('legendModal').style.display === 'block') {
                if (e.key === 'Escape') closeLegendModal();
            }
        });

        function saveToLocalStorage() {
            localStorage.setItem('wandb_phate_labels_v2', JSON.stringify(data));
            autoSaveToHTML();
        }

        function autoSaveToHTML() {
            // Update the data script tag so saving the page preserves changes
            const dataScript = document.getElementById('dataScript');
            if (dataScript) {
                const dataStr = JSON.stringify(data);
                dataScript.textContent = '// DATA_START - auto-saved\\n        const data = ' + dataStr + ';\\n        // DATA_END';
            }
        }

        function loadFromLocalStorage() {
            const saved = localStorage.getItem('wandb_phate_labels_v2');

            if (saved) {
                const parsed = JSON.parse(saved);
                Object.keys(parsed).forEach(key => {
                    if (data.hasOwnProperty(key)) {
                        data[key] = parsed[key];
                    }
                });

                // Restore UI state
                for (const idx in data.primary) {
                    const card = document.getElementById('card-' + idx);
                    if (card) {
                        card.querySelectorAll('.label-btn').forEach(btn => {
                            if (btn.dataset.label === data.primary[idx]) {
                                btn.classList.add('selected');
                            }
                        });
                        // Update badge
                        const badge = document.getElementById('badge-' + idx);
                        if (badge) {
                            badge.style.background = labelColors[data.primary[idx]];
                            badge.textContent = data.primary[idx].replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                        }
                    }
                }

                ['density', 'branching', 'quality', 'n_components'].forEach(field => {
                    for (const idx in data[field]) {
                        const select = document.getElementById(field + '-' + idx);
                        if (select) {
                            select.value = data[field][idx];
                            select.classList.add('filled');
                        }
                    }
                });

                for (const idx in data.flagged) {
                    const card = document.getElementById('card-' + idx);
                    const btn = document.getElementById('flag-btn-' + idx);
                    if (card && btn) {
                        card.classList.add('flagged');
                        btn.classList.add('flagged');
                        btn.textContent = 'Flagged';
                    }
                }

                for (const idx in data.notes) {
                    const textarea = document.getElementById('notes-' + idx);
                    if (textarea) {
                        textarea.value = data.notes[idx];
                    }
                }

                document.querySelectorAll('.card').forEach((card, idx) => {
                    updateCardStatus(idx);
                });
            }

            updateStats();
        }

        // Initialize
        loadFromLocalStorage();
    </script>
</body>
</html>
'''

    print(f"\nWriting HTML file...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)

    n_preloaded = len([k for k in initial_data['primary'].keys()])

    print(f"\nGallery created: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nFeatures:")
    print(f"  - {len(valid_runs)} W&B PHATE visualizations")
    print(f"  - Label badge indicator on each image")
    print(f"  - Color Legend input (per label_key categories)")
    print(f"  - Structure types reference modal")
    print(f"  - Search by dataset/subset name")
    print(f"  - Save to HTML button")
    print(f"  - Primary structure: 8 categories")
    print(f"  - Secondary: density, branching, quality, components")
    if n_preloaded > 0:
        print(f"\n  PRE-LOADED: {n_preloaded} labels baked into HTML")

if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description='Create W&B PHATE labeling gallery HTML')
    parser.add_argument('--labels-csv', type=str, default=str(LABELS_CSV),
                        help=f'Path to existing labels CSV file (default: {LABELS_CSV})')
    parser.add_argument('--no-labels', action='store_true',
                        help='Create gallery without pre-loading existing labels')
    args = parser.parse_args()

    if args.no_labels:
        existing_labels = {}
        print("Creating gallery without pre-loaded labels")
    else:
        labels_path = Path(args.labels_csv)
        existing_labels = load_existing_labels(labels_path)

    create_gallery(existing_labels)
