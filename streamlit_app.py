#!/usr/bin/env python3
"""
Multi-page Streamlit App for Geomancer LLM Decision Making
Includes W&B Gallery and other analysis tools
"""

import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
from urllib.parse import quote
from datetime import datetime
import re

# Configure page
st.set_page_config(
    page_title="Geomancer LLM Decision Making",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def format_config_value(key, value):
    """Format configuration values for better display"""
    if isinstance(value, (int, float)):
        if key.lower() in ['lr', 'learning_rate', 'alpha', 'beta', 'gamma']:
            return f"{value:.6f}" if value < 0.01 else f"{value:.4f}"
        elif key.lower() in ['epochs', 'steps', 'iterations', 'batch_size']:
            return f"{value:,}"
        else:
            return str(value)
    elif isinstance(value, bool):
        return "✅ Yes" if value else "❌ No"
    elif isinstance(value, str) and len(value) > 50:
        return value[:47] + "..."
    else:
        return str(value)

def get_color_legend_info(run_data):
    """Extract color legend information from run metadata"""
    legend_info = []

    # Check for label key information
    if 'label_key' in run_data:
        legend_info.append(f"🎨 **Color Legend:** {run_data['label_key']}")

    # Check for class information in config
    config = run_data.get('config', {})
    if 'num_classes' in config:
        legend_info.append(f"📊 **Classes:** {config['num_classes']}")

    # Check for specific label information
    if 'labels' in run_data:
        labels = run_data['labels']
        if isinstance(labels, dict):
            unique_labels = list(labels.keys())[:10]  # Show first 10
            legend_info.append(f"🏷️ **Labels:** {', '.join(unique_labels)}")

    return legend_info

def load_manual_labels():
    """Load manual labels from file"""
    labels_file = Path("manual_labels.json")
    if labels_file.exists():
        try:
            with open(labels_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_manual_labels(labels):
    """Save manual labels to file"""
    try:
        with open("manual_labels.json", 'w') as f:
            json.dump(labels, f, indent=2)
        return True
    except:
        return False

def get_config_sections(config):
    """Organize config into logical sections"""
    sections = {
        "🧬 Algorithm": {},
        "📊 Data": {},
        "🔧 Training": {},
        "📈 Optimization": {},
        "🎛️ Other": {}
    }

    # Define key mappings
    algo_keys = ['algorithm', 'method', 'model', 'embedding_method']
    data_keys = ['dataset', 'batch_size', 'num_samples', 'input_dim', 'output_dim', 'n_components']
    training_keys = ['epochs', 'steps', 'iterations', 'validation_split', 'early_stopping']
    opt_keys = ['lr', 'learning_rate', 'optimizer', 'alpha', 'beta', 'gamma', 'momentum']

    for key, value in config.items():
        key_lower = key.lower()

        if any(k in key_lower for k in algo_keys):
            sections["🧬 Algorithm"][key] = value
        elif any(k in key_lower for k in data_keys):
            sections["📊 Data"][key] = value
        elif any(k in key_lower for k in training_keys):
            sections["🔧 Training"][key] = value
        elif any(k in key_lower for k in opt_keys):
            sections["📈 Optimization"][key] = value
        else:
            sections["🎛️ Other"][key] = value

    # Remove empty sections
    return {k: v for k, v in sections.items() if v}

# Initialize session state for manual labels
if 'manual_labels' not in st.session_state:
    st.session_state.manual_labels = load_manual_labels()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .gallery-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        text-align: center;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔬 Geomancer LLM Decision Making</h1>
    <p>Advanced Analysis Dashboard & W&B Gallery</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎨 W&B Gallery", "📊 Analytics", "⚙️ Settings"])

# W&B Gallery Tab
with tab1:
    st.header("🎨 W&B PHATE Visualizations Gallery")

    # Load metadata
    metadata_path = Path("wandb_gallery_replit/wandb_metadata.json")

    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            st.success(f"✅ Loaded {len(metadata)} W&B runs with visualizations")

            # Create filter sidebar
            st.sidebar.markdown("### 🔍 Gallery Filters")

            # Extract unique values for filtering
            datasets = set()
            label_keys = set()
            algorithms = set()

            for run_id, run_data in metadata.items():
                datasets.add(run_data.get('dataset_name', 'unknown'))
                label_keys.add(run_data.get('label_key', 'unknown'))
                if 'config' in run_data:
                    algorithms.add(run_data['config'].get('algorithm', 'unknown'))

            # Filter controls
            selected_datasets = st.sidebar.multiselect(
                "📊 Datasets",
                sorted(datasets),
                default=sorted(datasets)[:3] if len(datasets) > 3 else sorted(datasets)
            )

            selected_labels = st.sidebar.multiselect(
                "🏷️ Label Keys",
                sorted(label_keys),
                default=sorted(label_keys)[:3] if len(label_keys) > 3 else sorted(label_keys)
            )

            if algorithms:
                selected_algorithms = st.sidebar.multiselect(
                    "🧬 Algorithms",
                    sorted(algorithms),
                    default=sorted(algorithms)[:3] if len(algorithms) > 3 else sorted(algorithms)
                )

            # Search box
            search_term = st.sidebar.text_input("🔍 Search runs", "")

            # Filter metadata
            filtered_runs = {}
            for run_id, run_data in metadata.items():
                # Apply filters
                if selected_datasets and run_data.get('dataset_name') not in selected_datasets:
                    continue
                if selected_labels and run_data.get('label_key') not in selected_labels:
                    continue
                if algorithms and selected_algorithms:
                    run_algo = run_data.get('config', {}).get('algorithm', 'unknown')
                    if run_algo not in selected_algorithms:
                        continue
                if search_term and search_term.lower() not in str(run_data).lower():
                    continue

                filtered_runs[run_id] = run_data

            st.sidebar.markdown(f"**Showing {len(filtered_runs)} of {len(metadata)} runs**")

            # Display gallery
            if filtered_runs:
                cols_per_row = 2
                run_items = list(filtered_runs.items())

                for i in range(0, len(run_items), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for j in range(cols_per_row):
                        if i + j < len(run_items):
                            run_id, run_data = run_items[i + j]

                            with cols[j]:
                                # Check if image exists
                                image_path = Path(f"wandb_gallery_replit/images/{run_id}.png")
                                if image_path.exists():
                                    # Check for manual label
                                    manual_label = st.session_state.manual_labels.get(run_id, "")

                                    # Create caption with label info
                                    caption = f"Run: {run_id}"
                                    if manual_label:
                                        caption += f"\n🏷️ {manual_label}"

                                    # Display image with enhanced caption
                                    st.image(
                                        str(image_path),
                                        caption=caption,
                                        use_container_width=True
                                    )

                                    # Show manual label badge if exists
                                    if manual_label:
                                        st.success(f"🏷️ **Label:** {manual_label}")
                                    else:
                                        st.info("💡 *Click Details → Labels to add manual label*")

                                    # Display metadata in expandable section
                                    with st.expander(f"📋 Details: {run_data.get('dataset_name', 'Unknown')}"):
                                        # Create tabs for organized view
                                        info_tab, config_tab, labels_tab, raw_tab = st.tabs([
                                            "ℹ️ Info", "⚙️ Config", "🏷️ Labels", "📊 Raw Data"
                                        ])

                                        with info_tab:
                                            col1, col2 = st.columns(2)

                                            with col1:
                                                st.markdown("**📋 Basic Information**")
                                                st.info(f"**Run ID:** `{run_id}`")
                                                st.info(f"**Dataset:** {run_data.get('dataset_name', 'N/A')}")
                                                st.info(f"**Label Key:** {run_data.get('label_key', 'N/A')}")

                                                # Display timing info if available
                                                if 'created_at' in run_data:
                                                    st.info(f"**Created:** {run_data['created_at']}")

                                            with col2:
                                                # Color legend information
                                                st.markdown("**🎨 Visualization Legend**")
                                                legend_info = get_color_legend_info(run_data)

                                                if legend_info:
                                                    for info in legend_info:
                                                        st.markdown(info)
                                                else:
                                                    st.markdown("🎯 PHATE embedding visualization")

                                                # W&B link if available
                                                if 'url' in run_data:
                                                    st.markdown(f"[🔗 View in W&B]({run_data['url']})")

                                        with config_tab:
                                            if 'config' in run_data:
                                                config_sections = get_config_sections(run_data['config'])

                                                for section_name, section_config in config_sections.items():
                                                    if section_config:
                                                        st.markdown(f"**{section_name}**")

                                                        # Create a nice table layout
                                                        config_rows = []
                                                        for key, value in section_config.items():
                                                            formatted_value = format_config_value(key, value)
                                                            config_rows.append({
                                                                "Parameter": key,
                                                                "Value": formatted_value
                                                            })

                                                        if config_rows:
                                                            config_df = pd.DataFrame(config_rows)
                                                            st.table(config_df)

                                                        st.markdown("---")
                                            else:
                                                st.info("No configuration data available for this run.")

                                        with labels_tab:
                                            # Manual labeling interface
                                            st.markdown("**🏷️ Manual Labeling System**")

                                            # Get current manual label
                                            current_label = st.session_state.manual_labels.get(run_id, "")

                                            # Label categories
                                            label_categories = [
                                                "🧬 Cell Type Analysis",
                                                "📊 Clustering Quality",
                                                "🎯 Embedding Quality",
                                                "🔬 Experimental",
                                                "✅ Validated",
                                                "❌ Failed",
                                                "🔄 Needs Review",
                                                "⭐ Highlighted"
                                            ]

                                            # Quick label buttons
                                            st.markdown("**Quick Labels:**")
                                            label_cols = st.columns(4)

                                            for idx, category in enumerate(label_categories):
                                                col_idx = idx % 4
                                                with label_cols[col_idx]:
                                                    if st.button(category, key=f"quick_{run_id}_{idx}"):
                                                        st.session_state.manual_labels[run_id] = category
                                                        if save_manual_labels(st.session_state.manual_labels):
                                                            st.success(f"Labeled as: {category}")
                                                            st.rerun()

                                            # Custom label input
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                custom_label = st.text_input(
                                                    "Custom Label:",
                                                    value=current_label,
                                                    key=f"custom_label_{run_id}"
                                                )

                                            with col2:
                                                if st.button("💾 Save", key=f"save_{run_id}"):
                                                    st.session_state.manual_labels[run_id] = custom_label
                                                    if save_manual_labels(st.session_state.manual_labels):
                                                        st.success("✅ Saved!")
                                                        st.rerun()

                                            # Show current label
                                            if current_label:
                                                st.success(f"**Current Label:** {current_label}")

                                            # Label statistics
                                            st.markdown("**📊 Label Statistics:**")
                                            if st.session_state.manual_labels:
                                                label_counts = pd.Series(list(st.session_state.manual_labels.values())).value_counts()
                                                st.bar_chart(label_counts)
                                            else:
                                                st.info("No manual labels created yet.")

                                        with raw_tab:
                                            # Full metadata as JSON
                                            st.markdown("**📊 Complete Metadata**")
                                            st.json(run_data)
            else:
                st.warning("No runs match the current filters. Try adjusting your selection.")

        except Exception as e:
            st.error(f"Error loading gallery metadata: {e}")
            st.info("Make sure the W&B gallery has been generated in the wandb_gallery_replit/ directory.")

    else:
        st.warning("⚠️ Gallery metadata not found.")
        st.info("Please run the gallery generation script first to create the W&B gallery.")

        if st.button("🔄 Refresh Page"):
            st.rerun()

# Analytics Tab
with tab2:
    st.header("📊 Analytics Dashboard")

    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Summary metrics
            st.subheader("📈 Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Total Runs</p>
                </div>
                """.format(len(metadata)), unsafe_allow_html=True)

            # Count datasets
            datasets = [run.get('dataset_name', 'unknown') for run in metadata.values()]
            unique_datasets = len(set(datasets))

            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Unique Datasets</p>
                </div>
                """.format(unique_datasets), unsafe_allow_html=True)

            # Count label keys
            label_keys = [run.get('label_key', 'unknown') for run in metadata.values()]
            unique_labels = len(set(label_keys))

            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Label Types</p>
                </div>
                """.format(unique_labels), unsafe_allow_html=True)

            # Count algorithms
            algorithms = [run.get('config', {}).get('algorithm', 'unknown') for run in metadata.values()]
            unique_algos = len(set(algorithms))

            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Algorithms</p>
                </div>
                """.format(unique_algos), unsafe_allow_html=True)

            # Charts
            st.subheader("📊 Distribution Charts")

            col1, col2 = st.columns(2)

            with col1:
                # Dataset distribution
                dataset_counts = pd.Series(datasets).value_counts()
                st.bar_chart(dataset_counts)
                st.caption("Runs per Dataset")

            with col2:
                # Label key distribution
                label_counts = pd.Series(label_keys).value_counts()
                st.bar_chart(label_counts)
                st.caption("Runs per Label Key")

        except Exception as e:
            st.error(f"Error loading analytics data: {e}")
    else:
        st.info("Analytics will be available once the W&B gallery is generated.")

# Settings Tab
with tab3:
    st.header("⚙️ Application Settings")

    st.subheader("📁 File Management")

    # Gallery status
    if metadata_path.exists():
        st.success("✅ W&B Gallery: Ready")
        file_size = metadata_path.stat().st_size / 1024 / 1024  # MB
        st.info(f"Metadata file size: {file_size:.2f} MB")
    else:
        st.warning("⚠️ W&B Gallery: Not found")

    # Directory info
    gallery_dir = Path("wandb_gallery_replit")
    if gallery_dir.exists():
        image_files = list(gallery_dir.glob("images/*.png"))
        st.info(f"Gallery images: {len(image_files)} files")

    st.subheader("🏷️ Manual Labels Management")

    # Manual labels statistics
    if st.session_state.manual_labels:
        st.success(f"✅ **{len(st.session_state.manual_labels)}** runs have manual labels")

        # Show label distribution
        label_values = list(st.session_state.manual_labels.values())
        if label_values:
            label_counts = pd.Series(label_values).value_counts()
            st.bar_chart(label_counts)

        # Export/import options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export Labels"):
                labels_json = json.dumps(st.session_state.manual_labels, indent=2)
                st.download_button(
                    "💾 Download Labels JSON",
                    labels_json,
                    "manual_labels_export.json",
                    "application/json"
                )

        with col2:
            if st.button("🗑️ Clear All Labels"):
                if st.checkbox("⚠️ Confirm deletion"):
                    st.session_state.manual_labels = {}
                    save_manual_labels({})
                    st.success("All labels cleared!")
                    st.rerun()

        # Detailed label management
        with st.expander("🔍 Detailed Label Management"):
            st.markdown("**All Manual Labels:**")
            for run_id, label in st.session_state.manual_labels.items():
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    st.text(f"Run: {run_id[:8]}...")
                with col2:
                    st.text(label)
                with col3:
                    if st.button("🗑️", key=f"delete_{run_id}"):
                        del st.session_state.manual_labels[run_id]
                        save_manual_labels(st.session_state.manual_labels)
                        st.rerun()

    else:
        st.info("No manual labels created yet. Use the Labels tab in the gallery to start labeling.")

    st.subheader("🔄 Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Gallery Data"):
            st.rerun()

    with col2:
        if st.button("📋 Show System Info"):
            st.info("Running on Streamlit with W&B integration")
            st.text(f"Working directory: {os.getcwd()}")

    # Label import functionality
    st.subheader("📥 Import Labels")
    uploaded_file = st.file_uploader("Upload Labels JSON", type=['json'])
    if uploaded_file is not None:
        try:
            imported_labels = json.load(uploaded_file)
            if st.button("🔄 Import Labels"):
                st.session_state.manual_labels.update(imported_labels)
                save_manual_labels(st.session_state.manual_labels)
                st.success(f"Imported {len(imported_labels)} labels!")
                st.rerun()
        except Exception as e:
            st.error(f"Error importing labels: {e}")

    st.subheader("ℹ️ About")
    st.markdown("""
    **Geomancer LLM Decision Making Dashboard**

    This application provides:
    - 🎨 Interactive W&B Gallery with 119+ PHATE visualizations
    - 📊 Analytics and insights from your experiments
    - ⚙️ Configuration and management tools

    Built with Streamlit for the Geomancer project.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🔬 Geomancer LLM Decision Making Dashboard | Built with Streamlit & W&B
</div>
""", unsafe_allow_html=True)