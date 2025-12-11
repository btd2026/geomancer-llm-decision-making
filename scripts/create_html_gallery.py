#!/usr/bin/env python3
"""Create a self-contained HTML gallery for PHATE image labeling with rich multi-dimensional annotations."""

import pandas as pd
import sqlite3
from pathlib import Path
import base64
import argparse
import json

# Configuration (relative to script location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMPLATE_FILE = PROJECT_ROOT / "data" / "manylatents_benchmark" / "labels" / "label_template.csv"
DB_PATH = PROJECT_ROOT / "data" / "manylatents_benchmark" / "manylatents_datasets.db"
OUTPUT_FILE = PROJECT_ROOT / "data" / "manylatents_benchmark" / "labels" / "labeling_gallery.html"
LABELS_CSV = PROJECT_ROOT / "data" / "phate_labels_rich.csv"  # Note: moved from Documents


def load_existing_labels(labels_path):
    """Load existing labels from CSV file and return as dict keyed by dataset_id."""
    if not labels_path.exists():
        print(f"No existing labels file found at {labels_path}")
        return {}

    df = pd.read_csv(labels_path)
    labels = {}

    for _, row in df.iterrows():
        dataset_id = row['dataset_id']
        labels[dataset_id] = {
            'primary': row.get('primary_structure', ''),
            'density': row.get('density_pattern', ''),
            'branching': row.get('branch_quality', ''),
            'quality': row.get('overall_quality', ''),
            'n_components': str(row.get('n_components', '')) if pd.notna(row.get('n_components', '')) else '',
            'n_clusters': str(row.get('n_clusters', '')) if pd.notna(row.get('n_clusters', '')) else '',
            'flagged': row.get('flagged', False) == True or str(row.get('flagged', '')).lower() == 'true',
            'notes': str(row.get('notes', '')) if pd.notna(row.get('notes', '')) else ''
        }
        # Clean up NaN values
        for key in labels[dataset_id]:
            if key != 'flagged' and pd.isna(labels[dataset_id][key]):
                labels[dataset_id][key] = ''
            if labels[dataset_id][key] == 'nan':
                labels[dataset_id][key] = ''

    print(f"Loaded {len(labels)} existing labels from {labels_path}")
    return labels

def image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_gallery(existing_labels=None):
    """Create the gallery HTML, optionally with pre-populated labels."""
    if existing_labels is None:
        existing_labels = {}

    print("Loading datasets from database...")

    # Load metadata from database
    conn = sqlite3.connect(DB_PATH)
    df_meta = pd.read_sql_query("""
        SELECT
            dataset_id,
            dataset_name,
            description,
            collection_name,
            ROUND(file_size_mb, 2) as size_mb,
            n_cells,
            n_features,
            phate_plot_path
        FROM datasets
        WHERE downloaded = 1 AND phate_plot_path IS NOT NULL
        ORDER BY file_size_mb ASC
    """, conn)
    conn.close()

    print(f"Processing {len(df_meta)} images with metadata...")

    # HTML template with rich annotation system
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>PHATE Structure Labeling - Multi-Dimensional</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
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
        .btn-filter { background: #3498db; color: white; }
        .btn-filter.active { background: #2980b9; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2); }
        .btn-clear { background: #e74c3c; color: white; }
        .btn-flagged { background: #f39c12; color: white; }
        .color-controls { margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
        .color-controls label { font-size: 13px; font-weight: 500; color: #555; }
        .color-controls select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 13px; min-width: 180px; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); position: relative; transition: all 0.2s; }
        .card.labeled { border-left: 5px solid #27ae60; }
        .card.complete { border-left: 5px solid #9b59b6; }
        .card.flagged { border: 3px solid #f39c12 !important; }
        .card.flagged::before { content: "FLAG"; position: absolute; top: 10px; right: 10px; background: #f39c12; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; z-index: 10; }
        .card img { width: 100%; height: auto; display: block; cursor: pointer; }
        .card img:hover { opacity: 0.95; }
        .card-body { padding: 15px; }

        /* Color indicator bar at top of card */
        .color-bar { height: 8px; margin: -15px -15px 15px -15px; background: transparent; transition: background 0.3s; }

        .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px; gap: 10px; }
        .card-title { font-size: 14px; font-weight: bold; color: #2c3e50; flex: 1; }
        .flag-btn { background: none; border: 2px solid #f39c12; color: #f39c12; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; transition: all 0.2s; white-space: nowrap; }
        .flag-btn:hover { background: #fef5e7; }
        .flag-btn.flagged { background: #f39c12; color: white; }
        .card-id { font-size: 11px; color: #95a5a6; word-break: break-all; margin-bottom: 10px; font-family: monospace; }
        .card-meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; font-size: 12px; }
        .meta-item { background: #ecf0f1; padding: 5px 8px; border-radius: 4px; }
        .meta-label { color: #7f8c8d; font-size: 10px; display: block; }
        .meta-value { color: #2c3e50; font-weight: 500; }
        .card-desc { font-size: 12px; color: #7f8c8d; margin-bottom: 12px; max-height: 40px; overflow: hidden; text-overflow: ellipsis; }

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

        /* Legend */
        .legend { margin: 15px 0; padding: 15px; background: #1a252f; border-radius: 8px; }
        .legend-title { font-size: 12px; font-weight: 600; margin-bottom: 10px; opacity: 0.9; }
        .legend-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
        .legend-item { display: flex; align-items: center; gap: 8px; font-size: 12px; }
        .legend-color { width: 16px; height: 16px; border-radius: 3px; flex-shrink: 0; }

        /* Image modal */
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); }
        .modal-content { max-width: 90%; max-height: 90%; margin: auto; display: block; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
        .modal-close { position: absolute; top: 20px; right: 30px; color: white; font-size: 40px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PHATE Structure Labeling</h1>
        <p>Multi-dimensional annotation for ML classifier training</p>
        <div class="progress"><div class="progress-bar" id="progressBar" style="width: 0%"></div></div>
        <div class="stats">
            <div class="stat"><strong><span id="labeledCount">0</span></strong> / ''' + str(len(df_meta)) + ''' primary labeled</div>
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
        <div class="legend">
            <div class="legend-title">PRIMARY STRUCTURE TYPES</div>
            <div class="legend-grid">
                <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div> Discrete Clusters (separated groups)</div>
                <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div> Simple Trajectory (single path/arc)</div>
                <div class="legend-item"><div class="legend-color" style="background:#d35400"></div> Horseshoe (U-shape curve)</div>
                <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div> Bifurcation (2 branches, Y-shape)</div>
                <div class="legend-item"><div class="legend-color" style="background:#3498db"></div> Multi-branch (3-4 branches)</div>
                <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div> Complex Tree (5+ branches)</div>
                <div class="legend-item"><div class="legend-color" style="background:#1abc9c"></div> Cyclic/Loop (circular structure)</div>
                <div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div> Diffuse/Unclear (no clear structure)</div>
            </div>
        </div>
    </div>

    <div class="controls">
        <button class="btn-export" onclick="exportCSV()">Export Labels CSV</button>
        <button class="btn-export" onclick="saveToHTML()" style="background:#9b59b6">Save to HTML</button>
        <button class="btn-filter" onclick="filterCards('all')">Show All</button>
        <button class="btn-filter" onclick="filterCards('unlabeled')">Unlabeled Only</button>
        <button class="btn-filter" onclick="filterCards('labeled')">Labeled Only</button>
        <button class="btn-filter" onclick="filterCards('incomplete')">Incomplete Only</button>
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
                <option value="n_components">Num Components</option>
                <option value="n_clusters">Num Clusters</option>
            </select>
        </div>
    </div>

    <div class="gallery" id="gallery">
'''

    # Add cards for each image
    for idx, row in df_meta.iterrows():
        dataset_id = row['dataset_id']
        dataset_name = row['dataset_name'] or 'Unknown'
        description = row['description'] or ''
        collection = row['collection_name'] or ''
        size_mb = row['size_mb'] or 0
        n_cells = row['n_cells'] or 0
        n_features = row['n_features'] or 0
        image_path = Path(row['phate_plot_path'])

        if image_path.exists():
            img_base64 = image_to_base64(image_path)
            img_src = f"data:image/png;base64,{img_base64}"
        else:
            img_src = ""

        # Escape HTML special characters
        dataset_name_escaped = dataset_name.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        description_escaped = description.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')[:200]
        collection_escaped = collection.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')

        # Get existing labels for this dataset
        labels = existing_labels.get(dataset_id, {})
        primary = labels.get('primary', '')
        density = labels.get('density', '')
        branching = labels.get('branching', '')
        quality = labels.get('quality', '')
        n_components = labels.get('n_components', '')
        n_clusters = labels.get('n_clusters', '')
        is_flagged = labels.get('flagged', False)
        notes = labels.get('notes', '')

        # Determine card classes
        card_classes = "card"
        if primary:
            card_classes += " labeled"
        if primary and density and branching and quality and n_components and n_clusters:
            card_classes += " complete"
        if is_flagged:
            card_classes += " flagged"

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

        html += f'''
        <div class="{card_classes}" id="card-{idx}" data-id="{dataset_id}">
            <img src="{img_src}" alt="{dataset_name_escaped}" onclick="openModal(this.src)">
            <div class="card-body">
                <div class="color-bar" id="colorbar-{idx}"></div>
                <div class="card-header">
                    <div class="card-title">{dataset_name_escaped}</div>
                    <button class="{flag_btn_class}" id="flag-btn-{idx}" onclick="toggleFlag({idx})">{flag_btn_text}</button>
                </div>
                <div class="card-id">{dataset_id}</div>
                <div class="card-meta">
                    <div class="meta-item">
                        <span class="meta-label">Size</span>
                        <span class="meta-value">{size_mb:.1f} MB</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Cells</span>
                        <span class="meta-value">{n_cells:,}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Features</span>
                        <span class="meta-value">{n_features:,}</span>
                    </div>
                </div>
                <div class="card-desc" title="{description_escaped}">{description_escaped if description_escaped else collection_escaped}</div>

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
                                <option value="progressive"{option_selected(density, 'progressive')}>Progressive (sparse-dense)</option>
                                <option value="heterogeneous"{option_selected(density, 'heterogeneous')}>Heterogeneous patches</option>
                                <option value="dense_core"{option_selected(density, 'dense_core')}>Dense core with extensions</option>
                            </select>
                        </div>
                        <div class="annotation-item">
                            <label>Branch Quality</label>
                            <select id="branching-{idx}" class="{select_class(branching)}" onchange="setAnnotation({idx}, 'branching', this.value)">
                                <option value="">-- Select --</option>
                                <option value="none"{option_selected(branching, 'none')}>No branching</option>
                                <option value="clear"{option_selected(branching, 'clear')}>Clear branch point</option>
                                <option value="gradual"{option_selected(branching, 'gradual')}>Gradual divergence</option>
                                <option value="multiple"{option_selected(branching, 'multiple')}>Multiple branch points</option>
                            </select>
                        </div>
                        <div class="annotation-item">
                            <label>Overall Quality</label>
                            <select id="quality-{idx}" class="{select_class(quality)}" onchange="setAnnotation({idx}, 'quality', this.value)">
                                <option value="">-- Select --</option>
                                <option value="excellent"{option_selected(quality, 'excellent')}>Excellent (clear structure)</option>
                                <option value="good"{option_selected(quality, 'good')}>Good (some noise)</option>
                                <option value="fair"{option_selected(quality, 'fair')}>Fair (noisy but structured)</option>
                                <option value="poor"{option_selected(quality, 'poor')}>Poor (uninterpretable)</option>
                            </select>
                        </div>
                        <div class="annotation-item">
                            <label>Num Components</label>
                            <select id="n_components-{idx}" class="{select_class(n_components)}" onchange="setAnnotation({idx}, 'n_components', this.value)">
                                <option value="">-- Select --</option>
                                <option value="1"{option_selected(n_components, '1')}>1 (single connected)</option>
                                <option value="2"{option_selected(n_components, '2')}>2</option>
                                <option value="3"{option_selected(n_components, '3')}>3</option>
                                <option value="4"{option_selected(n_components, '4')}>4</option>
                                <option value="5+"{option_selected(n_components, '5+')}>5+</option>
                            </select>
                        </div>
                        <div class="annotation-item">
                            <label>Num Clusters</label>
                            <select id="n_clusters-{idx}" class="{select_class(n_clusters)}" onchange="setAnnotation({idx}, 'n_clusters', this.value)">
                                <option value="">-- Select --</option>
                                <option value="0"{option_selected(n_clusters, '0')}>0 (continuous)</option>
                                <option value="2"{option_selected(n_clusters, '2')}>2</option>
                                <option value="3"{option_selected(n_clusters, '3')}>3</option>
                                <option value="4-5"{option_selected(n_clusters, '4-5')}>4-5</option>
                                <option value="6-10"{option_selected(n_clusters, '6-10')}>6-10</option>
                                <option value="10+"{option_selected(n_clusters, '10+')}>10+</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Notes -->
                <div class="notes-section">
                    <textarea class="notes" id="notes-{idx}" placeholder="Additional notes..." onchange="setNotes({idx}, this.value)">{notes}</textarea>
                </div>
            </div>
        </div>
'''
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df_meta)}")

    # Store metadata for CSV export
    metadata_json = df_meta[['dataset_id', 'dataset_name', 'size_mb', 'n_cells', 'n_features']].to_json(orient='records')

    # Build initial data object from existing labels
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

    for idx, row in df_meta.iterrows():
        dataset_id = row['dataset_id']
        if dataset_id in existing_labels:
            labels = existing_labels[dataset_id]
            if labels.get('primary'):
                initial_data['primary'][str(idx)] = labels['primary']
            if labels.get('density'):
                initial_data['density'][str(idx)] = labels['density']
            if labels.get('branching'):
                initial_data['branching'][str(idx)] = labels['branching']
            if labels.get('quality'):
                initial_data['quality'][str(idx)] = labels['quality']
            if labels.get('n_components'):
                initial_data['n_components'][str(idx)] = labels['n_components']
            if labels.get('n_clusters'):
                initial_data['n_clusters'][str(idx)] = labels['n_clusters']
            if labels.get('flagged'):
                initial_data['flagged'][str(idx)] = True
            if labels.get('notes'):
                initial_data['notes'][str(idx)] = labels['notes']

    initial_data_json = json.dumps(initial_data)

    html += '''
    </div>

    <!-- Image Modal -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>

    <script>
        // DATA_START - initialized with pre-loaded labels from CSV
        const data = ''' + initial_data_json + ''';
        // DATA_END

        const metadata = ''' + metadata_json + ''';
        const totalCards = ''' + str(len(df_meta)) + ''';

        const primaryTypes = ['clusters', 'simple_trajectory', 'horseshoe', 'bifurcation', 'multi_branch', 'complex_tree', 'cyclic', 'diffuse'];

        // Color maps for different fields
        const colorMaps = {
            primary: {
                clusters: '#e74c3c',
                simple_trajectory: '#f39c12',
                horseshoe: '#d35400',
                bifurcation: '#27ae60',
                multi_branch: '#3498db',
                complex_tree: '#9b59b6',
                cyclic: '#1abc9c',
                diffuse: '#95a5a6'
            },
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
            },
            n_components: {
                '1': '#27ae60',
                '2': '#3498db',
                '3': '#9b59b6',
                '4': '#f39c12',
                '5+': '#e74c3c'
            },
            n_clusters: {
                '0': '#27ae60',
                '2': '#3498db',
                '3': '#9b59b6',
                '4-5': '#f39c12',
                '6-10': '#e67e22',
                '10+': '#e74c3c'
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
                              data.quality.hasOwnProperty(idx) &&
                              data.n_components.hasOwnProperty(idx) &&
                              data.n_clusters.hasOwnProperty(idx);

            card.classList.remove('labeled', 'complete');
            if (isComplete) {
                card.classList.add('complete');
            } else if (hasPrimary) {
                card.classList.add('labeled');
            }
        }

        function updateStats() {
            // Count primary labels
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

                // Check if complete
                if (data.density.hasOwnProperty(idx) &&
                    data.branching.hasOwnProperty(idx) &&
                    data.quality.hasOwnProperty(idx) &&
                    data.n_components.hasOwnProperty(idx) &&
                    data.n_clusters.hasOwnProperty(idx)) {
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
                const isComplete = hasPrimary &&
                                  data.density.hasOwnProperty(idx) &&
                                  data.branching.hasOwnProperty(idx) &&
                                  data.quality.hasOwnProperty(idx) &&
                                  data.n_components.hasOwnProperty(idx) &&
                                  data.n_clusters.hasOwnProperty(idx);

                if (filter === 'all') {
                    card.classList.remove('hidden');
                } else if (filter === 'unlabeled') {
                    card.classList.toggle('hidden', hasPrimary);
                } else if (filter === 'labeled') {
                    card.classList.toggle('hidden', !hasPrimary);
                } else if (filter === 'incomplete') {
                    card.classList.toggle('hidden', !hasPrimary || isComplete);
                } else if (filter === 'flagged') {
                    card.classList.toggle('hidden', !isFlagged);
                }
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
            let csv = 'dataset_id,dataset_name,size_mb,n_cells,n_features,primary_structure,density_pattern,branch_quality,overall_quality,n_components,n_clusters,flagged,notes\\n';

            metadata.forEach((item, idx) => {
                const primary = data.primary[idx] || '';
                const density = data.density[idx] || '';
                const branching = data.branching[idx] || '';
                const quality = data.quality[idx] || '';
                const n_comp = data.n_components[idx] || '';
                const n_clust = data.n_clusters[idx] || '';
                const flagged = data.flagged[idx] ? 'true' : '';
                const notes = (data.notes[idx] || '').replace(/"/g, '""').replace(/\\n/g, ' ');
                const name = (item.dataset_name || '').replace(/"/g, '""');

                csv += `${item.dataset_id},"${name}",${item.size_mb || ''},${item.n_cells || ''},${item.n_features || ''},${primary},${density},${branching},${quality},${n_comp},${n_clust},${flagged},"${notes}"\\n`;
            });

            const blob = new Blob([csv], {type: 'text/csv'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'phate_labels_rich.csv';
            a.click();
        }

        function saveToHTML() {
            // Get the current document HTML and update the data initialization
            let html = document.documentElement.outerHTML;

            // Create the new data object JSON string
            const dataStr = JSON.stringify(data);

            // Replace the data section between markers
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

            // Download the modified HTML
            const blob = new Blob(['<!DOCTYPE html>\\n<html>' + html.substring(html.indexOf('<head>'))], {type: 'text/html'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const timestamp = new Date().toISOString().slice(0,10);
            a.download = 'labeling_gallery_saved_' + timestamp + '.html';
            a.click();
            URL.revokeObjectURL(url);

            alert('HTML saved with ' + Object.keys(data.primary).length + ' labels baked in!\\n\\nNote: Open the saved file to continue labeling with your progress preserved.');
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
            });

            updateStats();
            applyColorBy('none');
            saveToLocalStorage();
        }

        function openModal(src) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImg').src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeModal();
        });

        function saveToLocalStorage() {
            localStorage.setItem('phate_labels_v2', JSON.stringify(data));
        }

        function loadFromLocalStorage() {
            const saved = localStorage.getItem('phate_labels_v2');

            if (saved) {
                const parsed = JSON.parse(saved);
                Object.keys(parsed).forEach(key => {
                    if (data.hasOwnProperty(key)) {
                        data[key] = parsed[key];
                    }
                });

                // Restore UI state - primary labels
                for (const idx in data.primary) {
                    const card = document.getElementById('card-' + idx);
                    if (card) {
                        card.querySelectorAll('.label-btn').forEach(btn => {
                            if (btn.dataset.label === data.primary[idx]) {
                                btn.classList.add('selected');
                            }
                        });
                    }
                }

                // Restore dropdowns
                ['density', 'branching', 'quality', 'n_components', 'n_clusters'].forEach(field => {
                    for (const idx in data[field]) {
                        const select = document.getElementById(field + '-' + idx);
                        if (select) {
                            select.value = data[field][idx];
                            select.classList.add('filled');
                        }
                    }
                });

                // Restore flags
                for (const idx in data.flagged) {
                    const card = document.getElementById('card-' + idx);
                    const btn = document.getElementById('flag-btn-' + idx);
                    if (card && btn) {
                        card.classList.add('flagged');
                        btn.classList.add('flagged');
                        btn.textContent = 'Flagged';
                    }
                }

                // Restore notes
                for (const idx in data.notes) {
                    const textarea = document.getElementById('notes-' + idx);
                    if (textarea) {
                        textarea.value = data.notes[idx];
                    }
                }

                // Update card statuses
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

    print(f"\\nWriting HTML file...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)

    print(f"\\n Gallery created: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\\nFeatures:")
    print(f"  - Primary structure: 8 categories (includes Horseshoe)")
    print(f"  - Density pattern: 4 options")
    print(f"  - Branch quality: 4 options")
    print(f"  - Overall quality: 4 options")
    print(f"  - Num Components: 5 options (1, 2, 3, 4, 5+)")
    print(f"  - Num Clusters: 6 options (0-continuous to 10+)")
    print(f"  - Flag button on each card")
    print(f"  - Color by field dropdown (colors card top bars)")
    print(f"  - Notes field for each image")
    print(f"  - Click images to enlarge")
    print(f"  - LocalStorage persistence (new key: phate_labels_v2)")
    print(f"  - Rich CSV export with all annotations")
    # Count pre-loaded labels
    n_preloaded = len([k for k in initial_data['primary'].keys()])
    if n_preloaded > 0:
        print(f"\\n  PRE-LOADED: {n_preloaded} labels baked into HTML")
    print(f"\\nDownload this file and open in your browser to start labeling!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create PHATE labeling gallery HTML')
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
