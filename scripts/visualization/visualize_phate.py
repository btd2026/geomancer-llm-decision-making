#!/usr/bin/env python3
"""
Visualize PHATE embedding from Mouse Adipose Tissue dataset.
Creates publication-quality plots saved as PNG files.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("=" * 80)
print("PHATE Visualization: Mouse White Adipose Tissue")
print("=" * 80)

# Paths
data_path = '/home/btd8/llm-paper-analyze/phate_results/Mouse_Adipose_PHATE.h5ad'
output_dir = Path('/home/btd8/llm-paper-analyze/phate_visualizations')
output_dir.mkdir(exist_ok=True)

print(f"\n1. Loading PHATE results...")
print(f"   Path: {data_path}")

try:
    adata = sc.read_h5ad(data_path)
    print(f"   ✅ Data loaded successfully!")
    print(f"   Shape: {adata.shape}")
    print(f"   PHATE embedding: {adata.obsm['X_phate'].shape}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import sys
    sys.exit(1)

# Check available annotations
print(f"\n2. Available metadata columns:")
for i, col in enumerate(adata.obs.columns, 1):
    n_unique = adata.obs[col].nunique()
    dtype = adata.obs[col].dtype
    print(f"   {i}. {col}: {n_unique} unique values ({dtype})")

print(f"\n3. Generating visualizations...")

# Set plotting style
sc.set_figure_params(dpi=150, facecolor='white', figsize=(8, 6))

# Plot 1: Basic PHATE embedding (colored by density)
print(f"\n   Creating Plot 1: Basic PHATE embedding...")
fig, ax = plt.subplots(figsize=(10, 8))

phate_coords = adata.obsm['X_phate']
scatter = ax.scatter(
    phate_coords[:, 0],
    phate_coords[:, 1],
    c=np.arange(len(phate_coords)),  # Color by index for gradient effect
    s=5,
    alpha=0.6,
    cmap='viridis'
)

ax.set_xlabel('PHATE 1', fontsize=14)
ax.set_ylabel('PHATE 2', fontsize=14)
ax.set_title('PHATE Embedding - Mouse White Adipose Tissue\n(10,000 cells)',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f8f8')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cell Index', fontsize=12)

plt.tight_layout()
output_file = output_dir / 'phate_basic.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_file}")
plt.close()

# Plot 2: PHATE embedding colored by metadata (if available)
# Try to find categorical annotations
categorical_cols = []
for col in adata.obs.columns:
    if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
        n_unique = adata.obs[col].nunique()
        if 2 <= n_unique <= 50:  # Reasonable number of categories
            categorical_cols.append((col, n_unique))

if categorical_cols:
    # Sort by number of categories
    categorical_cols.sort(key=lambda x: x[1])

    print(f"\n   Creating plots colored by annotations...")

    for col, n_cats in categorical_cols[:3]:  # Plot top 3
        print(f"   Creating Plot: PHATE colored by '{col}' ({n_cats} categories)...")

        fig, ax = plt.subplots(figsize=(12, 9))

        # Create color map for categories
        categories = adata.obs[col].unique()
        n_colors = len(categories)

        if n_colors <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
        else:
            colors = plt.cm.tab20b(np.linspace(0, 1, n_colors))

        # Plot each category
        for i, cat in enumerate(categories):
            mask = adata.obs[col] == cat
            coords = phate_coords[mask]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=[colors[i]],
                label=str(cat)[:30],  # Truncate long labels
                s=10,
                alpha=0.6
            )

        ax.set_xlabel('PHATE 1', fontsize=14)
        ax.set_ylabel('PHATE 2', fontsize=14)
        ax.set_title(f'PHATE Embedding - Colored by {col}\n({n_cats} categories)',
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')

        # Legend
        if n_cats <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     frameon=True, fontsize=9)

        plt.tight_layout()
        safe_col = col.replace(' ', '_').replace('/', '_')
        output_file = output_dir / f'phate_by_{safe_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_file}")
        plt.close()

# Plot 3: Density plot
print(f"\n   Creating Plot: Density heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

# Create 2D histogram for density
hist, xedges, yedges = np.histogram2d(
    phate_coords[:, 0],
    phate_coords[:, 1],
    bins=50
)

# Plot heatmap
im = ax.imshow(
    hist.T,
    origin='lower',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap='hot',
    aspect='auto',
    interpolation='gaussian'
)

ax.set_xlabel('PHATE 1', fontsize=14)
ax.set_ylabel('PHATE 2', fontsize=14)
ax.set_title('PHATE Embedding - Cell Density Heatmap',
             fontsize=16, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Cell Density', fontsize=12)

plt.tight_layout()
output_file = output_dir / 'phate_density.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_file}")
plt.close()

# Plot 4: Summary panel with multiple views
print(f"\n   Creating Plot: Summary panel...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Subplot 1: Basic scatter
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(phate_coords[:, 0], phate_coords[:, 1],
            c='#3498db', s=2, alpha=0.4)
ax1.set_xlabel('PHATE 1')
ax1.set_ylabel('PHATE 2')
ax1.set_title('A. Standard View')
ax1.grid(True, alpha=0.3)

# Subplot 2: Density
ax2 = fig.add_subplot(gs[0, 1])
ax2.hexbin(phate_coords[:, 0], phate_coords[:, 1],
           gridsize=40, cmap='YlOrRd', alpha=0.8)
ax2.set_xlabel('PHATE 1')
ax2.set_ylabel('PHATE 2')
ax2.set_title('B. Density (Hexbin)')
ax2.grid(True, alpha=0.3)

# Subplot 3: Contour
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(phate_coords[:, 0], phate_coords[:, 1],
            c='lightgray', s=1, alpha=0.3)
# Add contour
h, x, y = np.histogram2d(phate_coords[:, 0], phate_coords[:, 1], bins=30)
ax3.contour(h.T, extent=[x[0], x[-1], y[0], y[-1]],
            colors='black', linewidths=1.5, alpha=0.6)
ax3.set_xlabel('PHATE 1')
ax3.set_ylabel('PHATE 2')
ax3.set_title('C. With Contours')
ax3.grid(True, alpha=0.3)

# Subplot 4: Stats
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
stats_text = f"""
PHATE Embedding Statistics

Dataset: Mouse White Adipose Tissue
Total cells in original: {adata.n_obs:,}
Total genes: {adata.n_vars:,}

PHATE Parameters:
• Dimensions: 2
• KNN: 5
• Diffusion time (t): 49 (auto)

Embedding Range:
• PHATE 1: [{phate_coords[:, 0].min():.4f}, {phate_coords[:, 0].max():.4f}]
• PHATE 2: [{phate_coords[:, 1].min():.4f}, {phate_coords[:, 1].max():.4f}]

Statistics:
• Mean PHATE 1: {phate_coords[:, 0].mean():.4f}
• Mean PHATE 2: {phate_coords[:, 1].mean():.4f}
• Std PHATE 1: {phate_coords[:, 0].std():.4f}
• Std PHATE 2: {phate_coords[:, 1].std():.4f}
"""
ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
         verticalalignment='center')

fig.suptitle('PHATE Embedding - Mouse White Adipose Tissue Atlas',
             fontsize=18, fontweight='bold', y=0.98)

output_file = output_dir / 'phate_summary_panel.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_file}")
plt.close()

print(f"\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated visualizations:")
print(f"  Location: {output_dir}")
print(f"\n  Files created:")

for img_file in sorted(output_dir.glob('*.png')):
    size_mb = img_file.stat().st_size / 1e6
    print(f"    ✓ {img_file.name} ({size_mb:.1f} MB)")

print(f"\nTo view these images:")
print(f"  1. Download files from: {output_dir}")
print(f"  2. Or use: scp btd8@<cluster>:{output_dir}/*.png .")
print(f"  3. Open PNG files with any image viewer")
print("=" * 80)
