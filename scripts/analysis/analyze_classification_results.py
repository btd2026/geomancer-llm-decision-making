#!/usr/bin/env python3
"""
Analyze the classification results in detail and create summary reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_and_analyze_results():
    """Load and analyze the classification results."""
    print("="*70)
    print("DETAILED ANALYSIS OF CLASSIFICATION RESULTS")
    print("="*70)

    # Load results
    results_path = PROJECT_ROOT / "data" / "phate_structure_predictions.csv"
    results_df = pd.read_csv(results_path)
    print(f"Loaded results for {len(results_df)} datasets")

    # Basic statistics
    print(f"\n📊 Structure Distribution:")
    structure_counts = results_df['predicted_structure'].value_counts()
    for structure, count in structure_counts.items():
        pct = count / len(results_df) * 100
        print(f"   {structure:15}: {count:3d} ({pct:5.1f}%)")

    # Confidence analysis
    print(f"\n🎯 Confidence Analysis:")
    conf_stats = results_df['confidence'].describe()
    print(f"   Mean confidence: {conf_stats['mean']:.3f}")
    print(f"   Median confidence: {conf_stats['50%']:.3f}")
    print(f"   Min confidence: {conf_stats['min']:.3f}")
    print(f"   Max confidence: {conf_stats['max']:.3f}")

    # Confidence by structure
    print(f"\n🔍 Confidence by Structure Type:")
    conf_by_structure = results_df.groupby('predicted_structure')['confidence'].agg(['mean', 'std', 'count'])
    for structure in conf_by_structure.index:
        mean_conf = conf_by_structure.loc[structure, 'mean']
        std_conf = conf_by_structure.loc[structure, 'std']
        count = conf_by_structure.loc[structure, 'count']
        print(f"   {structure:15}: {mean_conf:.3f} ± {std_conf:.3f} (n={count})")

    # Dataset size analysis
    print(f"\n📈 Dataset Size Analysis:")
    size_stats = results_df['n_points'].describe()
    print(f"   Mean size: {size_stats['mean']:.0f} points")
    print(f"   Median size: {size_stats['50%']:.0f} points")
    print(f"   Size range: {size_stats['min']:.0f} - {size_stats['max']:.0f} points")

    # Size by structure
    print(f"\n📊 Dataset Size by Structure Type:")
    size_by_structure = results_df.groupby('predicted_structure')['n_points'].agg(['mean', 'median'])
    for structure in size_by_structure.index:
        mean_size = size_by_structure.loc[structure, 'mean']
        median_size = size_by_structure.loc[structure, 'median']
        print(f"   {structure:15}: mean={mean_size:.0f}, median={median_size:.0f}")

    return results_df

def create_confidence_analysis(results_df):
    """Create detailed confidence analysis."""
    print(f"\n" + "="*50)
    print("CONFIDENCE LEVEL ANALYSIS")
    print("="*50)

    # Define confidence categories
    high_conf = results_df[results_df['confidence'] > 0.6]
    med_conf = results_df[(results_df['confidence'] > 0.4) & (results_df['confidence'] <= 0.6)]
    low_conf = results_df[results_df['confidence'] <= 0.4]

    print(f"High confidence (>0.6): {len(high_conf)} datasets ({len(high_conf)/len(results_df)*100:.1f}%)")
    print(f"Medium confidence (0.4-0.6): {len(med_conf)} datasets ({len(med_conf)/len(results_df)*100:.1f}%)")
    print(f"Low confidence (≤0.4): {len(low_conf)} datasets ({len(low_conf)/len(results_df)*100:.1f}%)")

    # Show high confidence predictions
    if len(high_conf) > 0:
        print(f"\n🎯 High Confidence Predictions:")
        for _, row in high_conf.head(10).iterrows():
            print(f"   {row['dataset_id'][:36]}: {row['predicted_structure']} ({row['confidence']:.3f})")

    # Show low confidence predictions (potential misclassifications)
    if len(low_conf) > 0:
        print(f"\n⚠️  Low Confidence Predictions (Review Candidates):")
        for _, row in low_conf.head(10).iterrows():
            print(f"   {row['dataset_id'][:36]}: {row['predicted_structure']} ({row['confidence']:.3f})")

def analyze_probability_distributions(results_df):
    """Analyze the probability distributions for each structure type."""
    print(f"\n" + "="*50)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("="*50)

    # Get probability columns
    prob_cols = [c for c in results_df.columns if c.startswith('prob_')]
    structures = [c.replace('prob_', '') for c in prob_cols]

    print(f"Structure types found: {structures}")

    # For each predicted structure, show the probability distribution
    for structure in results_df['predicted_structure'].unique():
        subset = results_df[results_df['predicted_structure'] == structure]
        print(f"\n📊 {structure.upper()} predictions (n={len(subset)}):")

        # Show probability stats for this structure
        prob_col = f'prob_{structure}'
        if prob_col in results_df.columns:
            probs = subset[prob_col]
            print(f"   Own-class probability: {probs.mean():.3f} ± {probs.std():.3f}")
            print(f"   Range: {probs.min():.3f} - {probs.max():.3f}")

            # Show competing probabilities
            other_prob_cols = [c for c in prob_cols if c != prob_col]
            for other_col in other_prob_cols:
                other_struct = other_col.replace('prob_', '')
                other_probs = subset[other_col]
                print(f"   vs {other_struct}: {other_probs.mean():.3f} ± {other_probs.std():.3f}")

def create_summary_report(results_df):
    """Create a summary report for easy interpretation."""
    print(f"\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)

    total_datasets = len(results_df)

    print(f"🔬 **DATASET OVERVIEW**")
    print(f"   • Total datasets classified: {total_datasets}")
    print(f"   • Average dataset size: {results_df['n_points'].mean():.0f} cells")
    print(f"   • Size range: {results_df['n_points'].min():.0f} - {results_df['n_points'].max():.0f} cells")

    print(f"\n🧬 **STRUCTURE PREDICTIONS**")
    structure_counts = results_df['predicted_structure'].value_counts()
    for i, (structure, count) in enumerate(structure_counts.items(), 1):
        pct = count / total_datasets * 100
        print(f"   {i}. {structure.replace('_', ' ').title()}: {count} datasets ({pct:.1f}%)")

    print(f"\n🎯 **MODEL CONFIDENCE**")
    high_conf = (results_df['confidence'] > 0.6).sum()
    med_conf = ((results_df['confidence'] > 0.4) & (results_df['confidence'] <= 0.6)).sum()
    low_conf = (results_df['confidence'] <= 0.4).sum()

    print(f"   • High confidence (>60%): {high_conf} datasets")
    print(f"   • Medium confidence (40-60%): {med_conf} datasets")
    print(f"   • Low confidence (<40%): {low_conf} datasets")
    print(f"   • Average confidence: {results_df['confidence'].mean():.1%}")

    print(f"\n📋 **KEY FINDINGS**")
    most_common = structure_counts.index[0]
    most_common_pct = structure_counts.iloc[0] / total_datasets * 100
    print(f"   • Most common structure: {most_common.replace('_', ' ')} ({most_common_pct:.1f}%)")

    # Find structure with highest average confidence
    conf_by_struct = results_df.groupby('predicted_structure')['confidence'].mean()
    most_confident_struct = conf_by_struct.idxmax()
    most_confident_conf = conf_by_struct.max()
    print(f"   • Most confidently predicted: {most_confident_struct.replace('_', ' ')} ({most_confident_conf:.1%} avg confidence)")

    # Find largest datasets by structure
    largest_by_struct = results_df.groupby('predicted_structure')['n_points'].mean()
    largest_struct = largest_by_struct.idxmax()
    largest_size = largest_by_struct.max()
    print(f"   • Largest datasets tend to be: {largest_struct.replace('_', ' ')} ({largest_size:.0f} avg cells)")

def save_detailed_results(results_df):
    """Save additional analysis files."""
    print(f"\n📁 **SAVING DETAILED ANALYSIS**")

    # Create summary by confidence level
    confidence_summary = []
    for conf_level, (min_conf, max_conf) in [('High', (0.6, 1.0)), ('Medium', (0.4, 0.6)), ('Low', (0.0, 0.4))]:
        subset = results_df[(results_df['confidence'] > min_conf) & (results_df['confidence'] <= max_conf)]
        if len(subset) > 0:
            for structure in subset['predicted_structure'].unique():
                struct_subset = subset[subset['predicted_structure'] == structure]
                confidence_summary.append({
                    'confidence_level': conf_level,
                    'structure': structure,
                    'count': len(struct_subset),
                    'avg_confidence': struct_subset['confidence'].mean(),
                    'avg_size': struct_subset['n_points'].mean()
                })

    conf_summary_df = pd.DataFrame(confidence_summary)
    conf_path = PROJECT_ROOT / "data" / "confidence_analysis.csv"
    conf_summary_df.to_csv(conf_path, index=False)
    print(f"   • Confidence analysis saved to: {conf_path}")

    # Create structure-specific reports
    struct_reports_dir = PROJECT_ROOT / "data" / "structure_reports"
    struct_reports_dir.mkdir(exist_ok=True)

    for structure in results_df['predicted_structure'].unique():
        struct_subset = results_df[results_df['predicted_structure'] == structure]
        struct_subset = struct_subset.sort_values('confidence', ascending=False)

        struct_path = struct_reports_dir / f"{structure}_predictions.csv"
        struct_subset.to_csv(struct_path, index=False)
        print(f"   • {structure} predictions saved to: {struct_path}")

def main():
    # Load and analyze results
    results_df = load_and_analyze_results()

    # Detailed confidence analysis
    create_confidence_analysis(results_df)

    # Probability distribution analysis
    analyze_probability_distributions(results_df)

    # Create summary report
    create_summary_report(results_df)

    # Save detailed analysis
    save_detailed_results(results_df)

    print(f"\n✅ **ANALYSIS COMPLETE**")
    print(f"   📊 Results: {PROJECT_ROOT}/data/phate_structure_predictions.csv")
    print(f"   📈 Analysis: {PROJECT_ROOT}/data/confidence_analysis.csv")
    print(f"   📁 Structure reports: {PROJECT_ROOT}/data/structure_reports/")

    return results_df

if __name__ == "__main__":
    results = main()