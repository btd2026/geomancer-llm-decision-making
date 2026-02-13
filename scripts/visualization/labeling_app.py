#!/usr/bin/env python3
"""
Streamlit app for collaborative PHATE structure labeling.
Run with: streamlit run labeling_app.py --server.port 8501
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

# Configuration
LABELS_DIR = Path("/home/btd8/llm-paper-analyze/data/manylatents_benchmark/labels")
TEMPLATE_FILE = LABELS_DIR / "label_template.csv"
LABELS_FILE = LABELS_DIR / "labels.csv"
BACKUP_DIR = LABELS_DIR / "backups"

# Structure types
STRUCTURE_TYPES = {
    0: ("Clusters", "Well-separated groups (distinct cell types)"),
    1: ("Trajectories", "Linear/branching paths (differentiation)"),
    2: ("Continuous", "Smooth manifolds (gradual transitions)"),
    3: ("Noisy/Mixed", "Poor structure preservation (scattered)")
}

# Page config
st.set_page_config(
    page_title="PHATE Structure Labeling",
    page_icon="🔬",
    layout="wide"
)

def load_data():
    """Load the labeling data."""
    if LABELS_FILE.exists():
        df = pd.read_csv(LABELS_FILE)
    elif TEMPLATE_FILE.exists():
        df = pd.read_csv(TEMPLATE_FILE)
    else:
        st.error(f"No label file found at {TEMPLATE_FILE}")
        return None

    # Ensure label column exists and is numeric
    if 'label' not in df.columns:
        df['label'] = None

    return df

def save_data(df):
    """Save the labeling data with backup."""
    # Create backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if LABELS_FILE.exists():
        backup_path = BACKUP_DIR / f"labels_backup_{timestamp}.csv"
        pd.read_csv(LABELS_FILE).to_csv(backup_path, index=False)

    # Save current labels
    df.to_csv(LABELS_FILE, index=False)
    return True

def get_progress(df):
    """Calculate labeling progress."""
    total = len(df)
    labeled = df['label'].notna().sum()
    return labeled, total

def main():
    st.title("🔬 PHATE Structure Labeling Tool")

    # Load data
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
        st.session_state.current_idx = 0

    df = st.session_state.df

    if df is None:
        return

    # Sidebar - Progress and Navigation
    with st.sidebar:
        st.header("Progress")
        labeled, total = get_progress(df)
        st.progress(labeled / total if total > 0 else 0)
        st.write(f"**{labeled} / {total}** labeled ({100*labeled/total:.1f}%)")

        # Label distribution
        st.subheader("Label Distribution")
        for idx, (name, desc) in STRUCTURE_TYPES.items():
            count = (df['label'] == idx).sum()
            st.write(f"{name}: **{count}**")

        st.divider()

        # Navigation
        st.header("Navigation")

        # Jump to specific index
        new_idx = st.number_input(
            "Go to dataset #",
            min_value=1,
            max_value=total,
            value=st.session_state.current_idx + 1
        ) - 1

        if new_idx != st.session_state.current_idx:
            st.session_state.current_idx = new_idx
            st.rerun()

        # Filter options
        st.subheader("Filter")
        filter_option = st.radio(
            "Show:",
            ["All", "Unlabeled only", "Labeled only"]
        )

        st.divider()

        # Save button
        if st.button("💾 Save Labels", type="primary", use_container_width=True):
            save_data(df)
            st.success("Labels saved!")

        # Export button
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            "phate_labels.csv",
            "text/csv",
            use_container_width=True
        )

    # Filter data based on selection
    if filter_option == "Unlabeled only":
        filtered_indices = df[df['label'].isna()].index.tolist()
    elif filter_option == "Labeled only":
        filtered_indices = df[df['label'].notna()].index.tolist()
    else:
        filtered_indices = df.index.tolist()

    if not filtered_indices:
        st.info("No datasets match the current filter.")
        return

    # Ensure current index is valid for filter
    if st.session_state.current_idx not in filtered_indices:
        st.session_state.current_idx = filtered_indices[0]

    current_idx = st.session_state.current_idx
    row = df.iloc[current_idx]

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display image
        st.subheader(f"Dataset {current_idx + 1} of {total}")
        st.caption(f"ID: `{row['dataset_id']}`")

        image_path = Path(row['image_path'])
        if image_path.exists():
            image = Image.open(image_path)
            st.image(image, use_container_width=True)
        else:
            st.error(f"Image not found: {image_path}")

    with col2:
        # Labeling interface
        st.subheader("Classification")

        # Current label
        current_label = row['label']
        if pd.notna(current_label):
            current_label = int(current_label)
            st.success(f"Current: **{STRUCTURE_TYPES[current_label][0]}**")
        else:
            st.warning("Not yet labeled")

        st.divider()

        # Label buttons
        st.write("**Select structure type:**")

        for idx, (name, desc) in STRUCTURE_TYPES.items():
            is_selected = current_label == idx
            button_type = "primary" if is_selected else "secondary"

            if st.button(
                f"{idx}: {name}",
                key=f"btn_{idx}",
                type=button_type,
                use_container_width=True,
                help=desc
            ):
                df.at[current_idx, 'label'] = idx
                st.session_state.df = df

                # Auto-advance to next unlabeled
                next_unlabeled = df[df['label'].isna()].index.tolist()
                if next_unlabeled:
                    st.session_state.current_idx = next_unlabeled[0]
                elif current_idx < total - 1:
                    st.session_state.current_idx = current_idx + 1

                st.rerun()

            st.caption(desc)

        st.divider()

        # Notes field
        notes = st.text_area(
            "Notes (optional)",
            value=row.get('notes', '') or '',
            key=f"notes_{current_idx}"
        )
        if notes != (row.get('notes', '') or ''):
            df.at[current_idx, 'notes'] = notes
            st.session_state.df = df

    # Navigation buttons at bottom
    st.divider()
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

    with nav_col1:
        if st.button("⬅️ Previous", use_container_width=True, disabled=current_idx == 0):
            st.session_state.current_idx = max(0, current_idx - 1)
            st.rerun()

    with nav_col2:
        # Quick jump to next unlabeled
        unlabeled = df[df['label'].isna()].index.tolist()
        if unlabeled and st.button(f"🎯 Next Unlabeled ({len(unlabeled)} left)", use_container_width=True):
            st.session_state.current_idx = unlabeled[0]
            st.rerun()

    with nav_col3:
        if st.button("Next ➡️", use_container_width=True, disabled=current_idx >= total - 1):
            st.session_state.current_idx = min(total - 1, current_idx + 1)
            st.rerun()

    # Instructions
    with st.expander("📖 Labeling Instructions"):
        st.markdown("""
        ### Structure Types

        **0 - Clusters**: Well-separated, distinct groups of cells. Clear boundaries between groups.
        - Indicates discrete cell types or states
        - High between-cluster distance

        **1 - Trajectories**: Linear or branching paths connecting cells.
        - Indicates developmental processes or differentiation
        - May have branching points

        **2 - Continuous**: Smooth gradients without clear boundaries.
        - Indicates continuous variation in cell states
        - High local intrinsic dimensionality

        **3 - Noisy/Mixed**: Scattered points, poor structure.
        - May indicate technical noise or failed embedding
        - Low trustworthiness/continuity metrics

        ### Tips
        - Use keyboard shortcuts: Press 0-3 to label quickly
        - Click "Next Unlabeled" to jump to datasets that need labeling
        - Save frequently to avoid losing work
        - Add notes for ambiguous cases
        """)

if __name__ == "__main__":
    main()
