#!/usr/bin/env python3
"""
Calculate quality metrics for the PHATE embedding we already created.
Uses ManyLatents metric functions directly.
"""

import sys
sys.path.insert(0, '/home/btd8/manylatents')

import scanpy as sc
import numpy as np
import torch
from pathlib import Path

print("=" * 80)
print("Calculate Quality Metrics for PHATE Embedding")
print("=" * 80)

# Paths
phate_file = '/home/btd8/llm-paper-analyze/phate_results/Mouse_Adipose_PHATE.h5ad'
output_dir = Path('/home/btd8/llm-paper-analyze/manylatents_results')
output_dir.mkdir(exist_ok=True)

print(f"\n1. Loading PHATE results...")
print(f"   Path: {phate_file}")

try:
    adata = sc.read_h5ad(phate_file)
    print(f"   ✅ Data loaded!")
    print(f"   Cells: {adata.n_obs:,}")
    print(f"   Genes: {adata.n_vars:,}")
    print(f"   PHATE embedding: {adata.obsm['X_phate'].shape}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Get original data and embedding
X_original = adata.X
if hasattr(X_original, 'toarray'):
    X_original = X_original.toarray()

embedding = adata.obsm['X_phate']

print(f"\n2. Data prepared:")
print(f"   Original data shape: {X_original.shape}")
print(f"   Embedding shape: {embedding.shape}")

print(f"\n3. Importing ManyLatents metrics...")
try:
    from manylatents.metrics.trustworthiness import Trustworthiness
    from manylatents.metrics.continuity import Continuity
    from manylatents.metrics.lid import LocalIntrinsicDimensionality
    from manylatents.metrics.knn_preservation import KNNPreservation
    from manylatents.metrics.tangent_space import TangentSpaceApproximation
    print(f"   ✅ Metrics imported successfully")
except Exception as e:
    print(f"   ❌ Error importing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create simple dataset object for metrics that need it
class SimpleDataset:
    def __init__(self, data):
        self.data = data
        self.original_data = data

dataset = SimpleDataset(X_original)

print(f"\n4. Calculating metrics...")
metrics = {}

# Trustworthiness
print(f"\n   Computing Trustworthiness...")
try:
    for k in [10, 25, 50]:
        tsa = Trustworthiness(embedding, dataset, n_neighbors=k)
        metrics[f'trustworthiness_k{k}'] = float(tsa)
        print(f"   ✅ Trustworthiness@{k}: {tsa:.4f}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Continuity
print(f"\n   Computing Continuity...")
try:
    for k in [10, 25, 50]:
        cont = Continuity(embedding, dataset, n_neighbors=k)
        metrics[f'continuity_k{k}'] = float(cont)
        print(f"   ✅ Continuity@{k}: {cont:.4f}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Local Intrinsic Dimensionality
print(f"\n   Computing Local Intrinsic Dimensionality...")
try:
    for k in [10, 20, 30]:
        lid = LocalIntrinsicDimensionality(embedding, k=k)
        metrics[f'lid_k{k}'] = float(lid)
        print(f"   ✅ LID@{k}: {lid:.4f}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# KNN Preservation
print(f"\n   Computing KNN Preservation...")
try:
    knn_pres = KNNPreservation(embedding, dataset, k=10)
    metrics['knn_preservation_k10'] = float(knn_pres)
    print(f"   ✅ KNN Preservation@10: {knn_pres:.4f}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Tangent Space Approximation
print(f"\n   Computing Tangent Space Approximation...")
try:
    tsa = TangentSpaceApproximation(embedding, dataset, k=10)
    metrics['tangent_space_approx_k10'] = float(tsa)
    print(f"   ✅ Tangent Space Approximation@10: {tsa:.4f}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Add dataset info
metrics.update({
    'n_cells': adata.n_obs,
    'n_genes': adata.n_vars,
    'embedding_dims': embedding.shape[1],
    'dataset': 'Mouse_White_Adipose_Tissue'
})

print(f"\n5. Saving metrics...")
try:
    # Save to JSON
    import json
    with open(output_dir / 'phate_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Saved JSON: {output_dir}/phate_metrics.json")

    # Save to CSV
    import pandas as pd
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'phate_metrics.csv', index=False)
    print(f"   ✅ Saved CSV: {output_dir}/phate_metrics.csv")

    # Create formatted report
    with open(output_dir / 'phate_metrics_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHATE Quality Metrics Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: Mouse White Adipose Tissue\n")
        f.write(f"Cells: {adata.n_obs:,}\n")
        f.write(f"Genes: {adata.n_vars:,}\n")
        f.write(f"Embedding dimensions: {embedding.shape[1]}\n\n")

        f.write("=" * 80 + "\n")
        f.write("METRICS\n")
        f.write("=" * 80 + "\n\n")

        # Group metrics
        tsa_metrics = {k: v for k, v in metrics.items() if 'trustworthiness' in k}
        cont_metrics = {k: v for k, v in metrics.items() if 'continuity' in k}
        lid_metrics = {k: v for k, v in metrics.items() if 'lid' in k}
        other_metrics = {k: v for k, v in metrics.items() if k not in tsa_metrics and k not in cont_metrics and k not in lid_metrics and k not in ['n_cells', 'n_genes', 'embedding_dims', 'dataset']}

        if tsa_metrics:
            f.write("Trustworthiness (neighborhood preservation):\n")
            for k, v in sorted(tsa_metrics.items()):
                f.write(f"  {k:.<45} {v:.4f}\n")
            f.write("\n")

        if cont_metrics:
            f.write("Continuity (manifold structure preservation):\n")
            for k, v in sorted(cont_metrics.items()):
                f.write(f"  {k:.<45} {v:.4f}\n")
            f.write("\n")

        if lid_metrics:
            f.write("Local Intrinsic Dimensionality:\n")
            for k, v in sorted(lid_metrics.items()):
                f.write(f"  {k:.<45} {v:.4f}\n")
            f.write("\n")

        if other_metrics:
            f.write("Other Metrics:\n")
            for k, v in sorted(other_metrics.items()):
                f.write(f"  {k:.<45} {v:.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Trustworthiness (0-1):\n")
        f.write("  > 0.9: Excellent - Local neighborhoods very well preserved\n")
        f.write("  > 0.7: Good - Reasonable preservation\n")
        f.write("  < 0.5: Poor - Local structure not well preserved\n\n")

        f.write("Continuity (0-1):\n")
        f.write("  > 0.9: Excellent - Manifold structure well preserved\n")
        f.write("  > 0.7: Good - Adequate structure preservation\n\n")

        f.write("Local Intrinsic Dimensionality:\n")
        f.write("  Lower values: Better dimension reduction\n")
        f.write("  Close to embedding dims (2): Ideal compression\n\n")

        f.write("KNN Preservation:\n")
        f.write("  Fraction of k-nearest neighbors preserved in embedding\n")
        f.write("  Higher is better (0-1 scale)\n\n")

    print(f"   ✅ Saved report: {output_dir}/phate_metrics_report.txt")

except Exception as e:
    print(f"   ⚠️  Error saving: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("METRICS CALCULATION COMPLETE")
print("=" * 80)

print(f"\n📊 SUMMARY OF METRICS:")
print(f"   " + "-" * 76)
for metric_name, value in sorted(metrics.items()):
    if isinstance(value, (int, float)) and metric_name not in ['n_cells', 'n_genes', 'embedding_dims']:
        print(f"   {metric_name:.<50} {value:.4f}")
    elif metric_name in ['n_cells', 'n_genes', 'embedding_dims']:
        print(f"   {metric_name:.<50} {value}")
print(f"   " + "-" * 76)

print(f"\n💾 Output Files:")
print(f"   - JSON: {output_dir}/phate_metrics.json")
print(f"   - CSV: {output_dir}/phate_metrics.csv")
print(f"   - Report: {output_dir}/phate_metrics_report.txt")

print(f"\n📖 To view the report:")
print(f"   cat {output_dir}/phate_metrics_report.txt")

print("=" * 80)
