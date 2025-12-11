#!/usr/bin/env python3
"""
Compute embedding metrics using manylatents framework.
# Setup paths relative to script location
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


This script loads existing PHATE embedding CSVs and computes structural metrics
using the manylatents metrics module. These metrics can be used for ML classification
of embedding structure types.
"""

import sys
from pathlib import Path

# Add manylatents to path
MANYLATENTS_PATH = Path(PROJECT_ROOT / "manylatents")
sys.path.insert(0, str(MANYLATENTS_PATH))

import pandas as pd
import numpy as np
from glob import glob
import logging
import warnings
import gc
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Paths
PHATE_RESULTS_DIR = Path("/nfs/roberts/project/pi_sk2433/shared/Geomancer_2025_Data/phate_results")
OUTPUT_DIR = Path(PROJECT_ROOT / "data" / "manylatents_benchmark")
CHECKPOINT_FILE = OUTPUT_DIR / "manylatents_metrics_checkpoint.csv"
CLASSIFICATION_CSV = Path(PROJECT_ROOT / "data" / "manylatents_benchmark" / "datasets_for_classification.csv")

# Maximum points for metrics computation (to avoid memory issues)
MAX_POINTS = 2000

# Skip computationally expensive metrics
SKIP_REEB_GRAPH = True  # Reeb graph is very memory-intensive


def load_embedding(csv_path: Path) -> Optional[np.ndarray]:
    """Load embedding from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'dim_1' in df.columns and 'dim_2' in df.columns:
            embedding = df[['dim_1', 'dim_2']].values
            return embedding
        else:
            logger.warning(f"CSV {csv_path} missing dim_1/dim_2 columns")
            return None
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None


def subsample_embedding(embedding: np.ndarray, max_points: int = MAX_POINTS) -> np.ndarray:
    """Subsample embedding if too large."""
    if len(embedding) <= max_points:
        return embedding

    indices = np.random.choice(len(embedding), max_points, replace=False)
    return embedding[indices]


def compute_manylatents_metrics(embedding: np.ndarray) -> Dict[str, Any]:
    """Compute metrics using manylatents framework."""
    metrics = {}

    # Import manylatents metrics
    try:
        from manylatents.metrics.fractal_dimension import FractalDimension
        from manylatents.metrics.participation_ratio import ParticipationRatio
        from manylatents.metrics.persistent_homology import PersistentHomology
        from manylatents.metrics.reeb_graph import ReebGraphNodesEdges
        from manylatents.metrics.anisotropy import Anisotropy
        from manylatents.metrics.continuity import Continuity
        from manylatents.metrics.correlation import PearsonCorrelation
    except ImportError as e:
        logger.error(f"Failed to import manylatents metrics: {e}")
        return metrics

    n_points = len(embedding)
    metrics['n_points'] = n_points

    # 1. Fractal Dimension
    try:
        fd = FractalDimension(None, embedding, n_box_sizes=10)
        metrics['fractal_dimension'] = fd
    except Exception as e:
        logger.warning(f"FractalDimension failed: {e}")
        metrics['fractal_dimension'] = np.nan

    # 2. Participation Ratio (local effective dimensionality)
    try:
        k = min(25, n_points - 1)
        pr = ParticipationRatio(embedding, n_neighbors=k)
        metrics['participation_ratio'] = pr
    except Exception as e:
        logger.warning(f"ParticipationRatio failed: {e}")
        metrics['participation_ratio'] = np.nan

    # 3. Persistent Homology - dimension 0 (connected components)
    # Use subsampled data if too large (ripser is O(n^3))
    ph_embedding = embedding if len(embedding) <= 1000 else embedding[np.random.choice(len(embedding), 1000, replace=False)]
    try:
        ph0 = PersistentHomology(ph_embedding, homology_dim=0, persistence_threshold=0.05)
        metrics['persistent_h0'] = ph0
    except Exception as e:
        logger.warning(f"PersistentHomology H0 failed: {e}")
        metrics['persistent_h0'] = np.nan

    # 4. Persistent Homology - dimension 1 (loops/cycles)
    try:
        ph1 = PersistentHomology(ph_embedding, homology_dim=1, persistence_threshold=0.05)
        metrics['persistent_h1'] = ph1
    except Exception as e:
        logger.warning(f"PersistentHomology H1 failed: {e}")
        metrics['persistent_h1'] = np.nan

    del ph_embedding
    gc.collect()

    # 5. Reeb Graph (branching structure) - Skip if flag is set
    if not SKIP_REEB_GRAPH:
        try:
            reeb_result = ReebGraphNodesEdges(embedding, n_bins=10)
            metrics['reeb_n_nodes'] = len(reeb_result['nodes'])
            metrics['reeb_n_edges'] = len(reeb_result['edges'])
            # Compute branching coefficient: edges/nodes ratio
            if len(reeb_result['nodes']) > 0:
                metrics['reeb_branching_coeff'] = len(reeb_result['edges']) / len(reeb_result['nodes'])
            else:
                metrics['reeb_branching_coeff'] = 0
        except Exception as e:
            logger.warning(f"ReebGraph failed: {e}")
            metrics['reeb_n_nodes'] = np.nan
            metrics['reeb_n_edges'] = np.nan
            metrics['reeb_branching_coeff'] = np.nan

    # 6. Anisotropy (how non-uniform the embedding is)
    try:
        aniso = Anisotropy(embedding)
        metrics['anisotropy'] = aniso
    except Exception as e:
        logger.warning(f"Anisotropy failed: {e}")
        metrics['anisotropy'] = np.nan

    # Additional geometric metrics from our original script that complement manylatents
    # These don't exist in manylatents but are useful for classification

    # 7. Basic statistics
    metrics['x_mean'] = np.mean(embedding[:, 0])
    metrics['y_mean'] = np.mean(embedding[:, 1])
    metrics['x_std'] = np.std(embedding[:, 0])
    metrics['y_std'] = np.std(embedding[:, 1])
    metrics['x_range'] = np.ptp(embedding[:, 0])
    metrics['y_range'] = np.ptp(embedding[:, 1])
    metrics['aspect_ratio'] = metrics['x_range'] / max(metrics['y_range'], 1e-10)

    # 8. PCA-based shape metrics
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(embedding)
    metrics['pca_var_ratio'] = pca.explained_variance_ratio_[0]
    if pca.explained_variance_[1] > 0:
        metrics['pca_elongation'] = pca.explained_variance_[0] / pca.explained_variance_[1]
    else:
        metrics['pca_elongation'] = 1.0

    # 9. Convex hull metrics
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(embedding)
        metrics['hull_area'] = hull.volume  # In 2D, volume is area
        metrics['hull_perimeter'] = hull.area  # In 2D, area is perimeter
        metrics['hull_compactness'] = (4 * np.pi * metrics['hull_area']) / (metrics['hull_perimeter'] ** 2)
        metrics['point_density_in_hull'] = n_points / max(metrics['hull_area'], 1e-10)
    except Exception as e:
        logger.warning(f"ConvexHull failed: {e}")
        metrics['hull_area'] = np.nan
        metrics['hull_perimeter'] = np.nan
        metrics['hull_compactness'] = np.nan
        metrics['point_density_in_hull'] = np.nan

    # 10. Density-based metrics using DBSCAN
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors

        # Compute distance percentiles for eps
        nn = NearestNeighbors(n_neighbors=min(5, n_points-1))
        nn.fit(embedding)
        distances, _ = nn.kneighbors(embedding)
        median_dist = np.median(distances[:, -1])

        for pctl in [5, 10, 25]:
            eps = np.percentile(distances[:, -1], pctl)
            db = DBSCAN(eps=eps, min_samples=5)
            labels = db.fit_predict(embedding)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / len(labels)

            metrics[f'dbscan_eps{pctl}_n_clusters'] = n_clusters
            metrics[f'dbscan_eps{pctl}_noise_ratio'] = noise_ratio

            if n_clusters > 1 and len(set(labels)) > 1:
                try:
                    sil = silhouette_score(embedding, labels)
                    metrics[f'dbscan_eps{pctl}_silhouette'] = sil
                except:
                    metrics[f'dbscan_eps{pctl}_silhouette'] = np.nan
            else:
                metrics[f'dbscan_eps{pctl}_silhouette'] = np.nan
    except Exception as e:
        logger.warning(f"DBSCAN metrics failed: {e}")
        for pctl in [5, 10, 25]:
            metrics[f'dbscan_eps{pctl}_n_clusters'] = np.nan
            metrics[f'dbscan_eps{pctl}_noise_ratio'] = np.nan
            metrics[f'dbscan_eps{pctl}_silhouette'] = np.nan

    # 11. Local density metrics
    try:
        from sklearn.neighbors import NearestNeighbors
        k = min(20, n_points - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(embedding)
        distances, _ = nn.kneighbors(embedding)
        local_density = 1.0 / (np.mean(distances, axis=1) + 1e-10)

        metrics['density_mean'] = np.mean(local_density)
        metrics['density_std'] = np.std(local_density)
        metrics['density_cv'] = metrics['density_std'] / max(metrics['density_mean'], 1e-10)
        metrics['density_skew'] = float(pd.Series(local_density).skew())
    except Exception as e:
        logger.warning(f"Density metrics failed: {e}")
        metrics['density_mean'] = np.nan
        metrics['density_std'] = np.nan
        metrics['density_cv'] = np.nan
        metrics['density_skew'] = np.nan

    # 12. Spatial entropy
    try:
        from scipy.stats import entropy
        n_bins = 20
        H, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=n_bins)
        H_flat = H.flatten()
        H_norm = H_flat / H_flat.sum()
        H_norm = H_norm[H_norm > 0]
        metrics['spatial_entropy'] = entropy(H_norm)
        metrics['spatial_entropy_normalized'] = metrics['spatial_entropy'] / np.log(n_bins * n_bins)
    except Exception as e:
        logger.warning(f"Spatial entropy failed: {e}")
        metrics['spatial_entropy'] = np.nan
        metrics['spatial_entropy_normalized'] = np.nan

    return metrics


def find_embedding_csv(dataset_dir: Path) -> Optional[Path]:
    """Find the most recent embedding CSV in a dataset directory."""
    csvs = list(dataset_dir.glob("embeddings_*.csv"))
    if not csvs:
        return None
    # Return most recent
    return max(csvs, key=lambda x: x.stat().st_mtime)


def load_checkpoint() -> set:
    """Load already processed dataset IDs from checkpoint."""
    if CHECKPOINT_FILE.exists():
        df = pd.read_csv(CHECKPOINT_FILE)
        return set(df['dataset_id'].tolist())
    return set()


def save_checkpoint(results: list):
    """Save results to checkpoint file."""
    if results:
        df = pd.DataFrame(results)
        if CHECKPOINT_FILE.exists():
            existing = pd.read_csv(CHECKPOINT_FILE)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(CHECKPOINT_FILE, index=False)


def extract_phate_dir_from_path(path_str: str) -> Optional[str]:
    """Extract PHATE directory ID from image path."""
    import re
    match = re.search(r'/phate_results/([a-f0-9-]+)/', str(path_str))
    if match:
        return match.group(1)
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load classification CSV to get dataset mapping
    if CLASSIFICATION_CSV.exists():
        class_df = pd.read_csv(CLASSIFICATION_CSV)
        logger.info(f"Loaded {len(class_df)} datasets from classification CSV")

        # Build mapping: dataset_id -> phate_dir_id
        dataset_to_phate = {}
        for _, row in class_df.iterrows():
            dataset_id = row['dataset_id']
            phate_path = row.get('phate_plot_image', '')
            phate_dir = extract_phate_dir_from_path(phate_path)
            if phate_dir:
                dataset_to_phate[dataset_id] = phate_dir

        logger.info(f"Mapped {len(dataset_to_phate)} datasets to PHATE results")
    else:
        logger.warning("Classification CSV not found, using direct directory listing")
        dataset_to_phate = {}

    # Load checkpoint
    processed = load_checkpoint()
    logger.info(f"Already processed: {len(processed)} datasets")

    # Process datasets from classification CSV (these have labels)
    if dataset_to_phate:
        datasets = list(dataset_to_phate.items())
    else:
        # Fallback to direct directory listing
        datasets = [(d.name, d.name) for d in PHATE_RESULTS_DIR.iterdir() if d.is_dir()]

    logger.info(f"Processing {len(datasets)} datasets")

    for i, (dataset_id, phate_dir_id) in enumerate(datasets):
        if dataset_id in processed:
            continue

        logger.info(f"[{i+1}/{len(datasets)}] Processing {dataset_id}")

        # Find embedding CSV in PHATE results directory
        phate_dir = PHATE_RESULTS_DIR / phate_dir_id
        if not phate_dir.exists():
            logger.warning(f"  PHATE directory not found: {phate_dir}")
            continue

        csv_path = find_embedding_csv(phate_dir)
        if csv_path is None:
            logger.warning(f"  No embedding CSV found")
            continue

        # Load embedding
        embedding = load_embedding(csv_path)
        if embedding is None:
            continue

        logger.info(f"  Loaded embedding: {embedding.shape}")

        # Subsample if needed
        original_size = len(embedding)
        embedding = subsample_embedding(embedding, MAX_POINTS)
        if len(embedding) < original_size:
            logger.info(f"  Subsampled to {len(embedding)} points")

        # Compute metrics
        try:
            metrics = compute_manylatents_metrics(embedding)
            metrics['dataset_id'] = dataset_id  # Use CELLxGENE dataset ID, not PHATE dir ID
            metrics['original_n_points'] = original_size

            # Save immediately to checkpoint
            save_checkpoint([metrics])
            processed.add(dataset_id)
            logger.info(f"  Computed {len(metrics)} metrics (saved)")
        except Exception as e:
            logger.error(f"  Failed to compute metrics: {e}")
            continue
        finally:
            # Clear memory after each dataset
            del embedding
            gc.collect()

    # All results saved incrementally

    # Load final results and save to main output file
    if CHECKPOINT_FILE.exists():
        final_df = pd.read_csv(CHECKPOINT_FILE)
        output_path = OUTPUT_DIR / "manylatents_metrics.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"Final results saved to {output_path}")
        logger.info(f"Total datasets processed: {len(final_df)}")
        logger.info(f"Total metrics per dataset: {len(final_df.columns)}")


if __name__ == "__main__":
    main()
