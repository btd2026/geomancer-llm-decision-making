#!/usr/bin/env python3
"""
Extract algorithms from papers - LOCAL VERSION.
- Dimensionality reduction algorithms (PCA, UMAP, t-SNE, etc.)
- Algorithm parameters
- Pipeline sequence order
- Context (which section mentioned)
"""
import json
import re
import sqlite3
from pathlib import Path

# Use local paths instead of container paths
config_dir = Path(__file__).parent.parent / 'configs'
data_dir = Path(__file__).parent.parent / 'data' / 'papers'

# Load research context
with open(config_dir / 'research_context.json') as f:
    context = json.load(f)

# Database path (use actual path, not container path)
db_path = data_dir / 'metadata' / 'papers.db'

# Algorithm patterns from research context
ALGORITHMS = {
    'PCA': {
        'patterns': [r'\bPCA\b', r'principal component analysis'],
        'category': 'dimensionality_reduction',
        'params': [r'(\d+)\s*(?:principal\s*)?components', r'n_components\s*=\s*(\d+)']
    },
    't-SNE': {
        'patterns': [r'\bt-SNE\b', r'\btsne\b', r't-distributed stochastic neighbor embedding'],
        'category': 'dimensionality_reduction',
        'params': [r'perplexity\s*=?\s*(\d+)', r'learning[_ ]rate\s*=?\s*([\d.]+)']
    },
    'UMAP': {
        'patterns': [r'\bUMAP\b', r'uniform manifold approximation'],
        'category': 'dimensionality_reduction',
        'params': [r'n_neighbors\s*=?\s*(\d+)', r'min_dist\s*=?\s*([\d.]+)', r'n_components\s*=?\s*(\d+)']
    },
    'PHATE': {
        'patterns': [r'\bPHATE\b', r'potential of heat-diffusion'],
        'category': 'dimensionality_reduction',
        'params': [r'k\s*=?\s*(\d+)', r't\s*=?\s*(\d+)']
    },
    'Autoencoder': {
        'patterns': [r'\bautoencoder\b', r'\bAE\b'],
        'category': 'dimensionality_reduction',
        'params': [r'latent[_ ]dim(?:ension)?\s*=?\s*(\d+)', r'(\d+)[- ]dimensional latent']
    },
    'VAE': {
        'patterns': [r'\bVAE\b', r'variational autoencoder'],
        'category': 'dimensionality_reduction',
        'params': [r'latent[_ ]dim(?:ension)?\s*=?\s*(\d+)']
    },
    'scVI': {
        'patterns': [r'\bscVI\b', r'single-cell variational inference'],
        'category': 'dimensionality_reduction',
        'params': [r'n_latent\s*=?\s*(\d+)']
    },
    'DCA': {
        'patterns': [r'\bDCA\b', r'deep count autoencoder'],
        'category': 'dimensionality_reduction',
        'params': []
    },
    'Diffusion Maps': {
        'patterns': [r'diffusion map', r'diffusion component'],
        'category': 'dimensionality_reduction',
        'params': [r'n_components\s*=?\s*(\d+)']
    },
    'ICA': {
        'patterns': [r'\bICA\b', r'independent component analysis'],
        'category': 'dimensionality_reduction',
        'params': [r'n_components\s*=?\s*(\d+)']
    },
    'NMF': {
        'patterns': [r'\bNMF\b', r'non-negative matrix factorization'],
        'category': 'dimensionality_reduction',
        'params': [r'n_components\s*=?\s*(\d+)']
    },
    'ZIFA': {
        'patterns': [r'\bZIFA\b', r'zero-inflated factor analysis'],
        'category': 'dimensionality_reduction',
        'params': []
    },
    'scGAN': {
        'patterns': [r'\bscGAN\b'],
        'category': 'dimensionality_reduction',
        'params': []
    },
    # Normalization methods
    'log-normalization': {
        'patterns': [r'log[- ]normalized', r'log2? transformation', r'log1p'],
        'category': 'normalization',
        'params': []
    },
    'CPM': {
        'patterns': [r'\bCPM\b', r'counts per million'],
        'category': 'normalization',
        'params': []
    },
    'TPM': {
        'patterns': [r'\bTPM\b', r'transcripts per million'],
        'category': 'normalization',
        'params': []
    },
    'FPKM': {
        'patterns': [r'\bFPKM\b', r'fragments per kilobase million'],
        'category': 'normalization',
        'params': []
    },
    'SCTransform': {
        'patterns': [r'SCTransform', r'sctransform'],
        'category': 'normalization',
        'params': []
    },
    'scran': {
        'patterns': [r'\bscran\b'],
        'category': 'normalization',
        'params': []
    },
    # Clustering
    'Louvain': {
        'patterns': [r'louvain clustering', r'louvain algorithm'],
        'category': 'clustering',
        'params': [r'resolution\s*=?\s*([\d.]+)']
    },
    'Leiden': {
        'patterns': [r'leiden clustering', r'leiden algorithm'],
        'category': 'clustering',
        'params': [r'resolution\s*=?\s*([\d.]+)']
    },
    'k-means': {
        'patterns': [r'k-means', r'kmeans'],
        'category': 'clustering',
        'params': [r'k\s*=?\s*(\d+)', r'(\d+)\s*clusters']
    },
}

# Sequence indicator words
SEQUENCE_INDICATORS = [
    'first', 'second', 'third', 'then', 'next', 'after', 'before',
    'followed by', 'subsequently', 'finally', 'initially'
]

def find_algorithms(text, section='abstract'):
    """Find all algorithms mentioned in text."""
    findings = []
    text_lower = text.lower()

    for algo_name, algo_info in ALGORITHMS.items():
        # Check if algorithm is mentioned
        for pattern in algo_info['patterns']:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Get context around each match
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context_text = text[start:end].strip()

                    # Extract parameters if present in context
                    params = {}
                    for param_pattern in algo_info['params']:
                        param_match = re.search(param_pattern, context_text, re.IGNORECASE)
                        if param_match:
                            param_value = param_match.group(1)
                            # Extract parameter name from pattern
                            param_name = re.search(r'(\w+)', param_pattern).group(1)
                            params[param_name] = param_value

                    finding = {
                        'algorithm_name': algo_name,
                        'algorithm_category': algo_info['category'],
                        'parameters': json.dumps(params) if params else None,
                        'mentioned_in_section': section,
                        'context_text': context_text,
                        'extraction_method': 'regex',
                        'confidence_score': 0.8,  # High confidence for exact matches
                        'position': match.start()  # For sequence ordering
                    }
                    findings.append(finding)
                break  # Only need one pattern match per algorithm

    return findings

def determine_sequence_order(findings):
    """Determine pipeline sequence order from algorithm findings."""
    if not findings:
        return findings

    # Sort by position in text (earlier mentions likely come first in pipeline)
    findings.sort(key=lambda x: x['position'])

    # Assign sequence numbers
    sequence = 1
    for finding in findings:
        finding['sequence_order'] = sequence
        sequence += 1
        del finding['position']  # Remove temporary position field

    return findings

def process_paper(paper_id, pmid, title, abstract):
    """Process a single paper to extract algorithm information."""
    all_findings = []

    # Extract from title
    title_findings = find_algorithms(title or '', section='title')
    all_findings.extend(title_findings)

    # Extract from abstract
    abstract_findings = find_algorithms(abstract or '', section='abstract')
    all_findings.extend(abstract_findings)

    # Determine sequence order
    all_findings = determine_sequence_order(all_findings)

    # Add paper_id to each finding
    for finding in all_findings:
        finding['paper_id'] = paper_id

    return all_findings

def main():
    """Main execution."""
    print("=" * 80)
    print("ALGORITHM EXTRACTION (LOCAL)")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all papers
    cursor.execute("""
        SELECT id, pmid, title, abstract
        FROM papers
        WHERE abstract IS NOT NULL AND abstract != ''
    """)

    papers = cursor.fetchall()
    print(f"Processing {len(papers)} papers...")
    print()

    # Clear existing algorithm extractions (for re-processing)
    cursor.execute("DELETE FROM extracted_algorithms")
    conn.commit()

    total_algorithms = 0
    papers_with_algos = 0
    algo_counts = {}

    for paper in papers:
        paper_id = paper['id']
        pmid = paper['pmid']
        title = paper['title'] or ''
        abstract = paper['abstract'] or ''

        # Process paper
        findings = process_paper(paper_id, pmid, title, abstract)

        if findings:
            papers_with_algos += 1
            print(f"[{pmid}] Found {len(findings)} algorithm(s):")

            # Insert findings
            for finding in findings:
                algo_name = finding['algorithm_name']
                print(f"  - {algo_name} ({finding['algorithm_category']})", end='')
                if finding['parameters']:
                    print(f" {finding['parameters']}", end='')
                print(f" [seq: {finding['sequence_order']}]")

                # Track counts
                algo_counts[algo_name] = algo_counts.get(algo_name, 0) + 1

                cursor.execute("""
                    INSERT INTO extracted_algorithms (
                        paper_id, algorithm_name, algorithm_category,
                        parameters, sequence_order, mentioned_in_section,
                        context_text, extraction_method, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    finding['paper_id'], finding['algorithm_name'],
                    finding['algorithm_category'], finding['parameters'],
                    finding['sequence_order'], finding['mentioned_in_section'],
                    finding['context_text'], finding['extraction_method'],
                    finding['confidence_score']
                ))

                total_algorithms += 1

            # Update papers table
            cursor.execute("""
                UPDATE papers SET methods_extracted = 1 WHERE id = ?
            """, (paper_id,))

    conn.commit()
    conn.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Papers processed: {len(papers)}")
    print(f"Papers with algorithms: {papers_with_algos}")
    print(f"Total algorithms extracted: {total_algorithms}")
    print()
    print("Top algorithms:")
    for algo, count in sorted(algo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {algo}: {count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
