#!/usr/bin/env python3
"""
Extract dataset information from papers - LOCAL VERSION.
- GEO accession numbers
- GitHub URLs
- SRA accessions
- ArrayExpress IDs
- Dataset characteristics (organism, tissue, cell counts)
"""
import json
import re
import sqlite3
from pathlib import Path

# Use local paths instead of container paths
config_dir = Path(__file__).parent.parent / 'configs'
data_dir = Path(__file__).parent.parent / 'data' / 'papers'

# Load MCP config
with open(config_dir / 'mcp_config.json') as f:
    mcp_config = json.load(f)

# Database path (use actual path, not container path)
db_path = data_dir / 'metadata' / 'papers.db'

# Regex patterns for dataset identifiers
PATTERNS = {
    'geo': r'GS[EM]\d+',  # GSE (series) or GSM (sample)
    'sra': r'SRP\d+|SRX\d+|SRR\d+',  # SRA accessions
    'arrayexpress': r'E-[A-Z]+-\d+',  # ArrayExpress
    'github': r'https?://(?:www\.)?github\.com/[\w\-]+/[\w\-\.]+',  # GitHub URLs
    'zenodo': r'zenodo\.\d+|10\.5281/zenodo\.\d+',  # Zenodo DOIs
}

# Patterns for extracting numerical data
NUMBER_PATTERNS = {
    'n_cells': [
        r'(\d+(?:,\d+)*)\s*(?:single\s*)?cells',
        r'(\d+(?:,\d+)*)\s*scRNA-seq',
        r'(\d+(?:,\d+)*)\s*nuclei',
        r'total of (\d+(?:,\d+)*)\s*cells',
        r'(\d+)k\s*cells',  # e.g., "10k cells"
    ],
    'n_genes': [
        r'(\d+(?:,\d+)*)\s*genes',
        r'(\d+)k\s*genes',
    ]
}

# Common organisms
ORGANISMS = [
    'human', 'mouse', 'homo sapiens', 'mus musculus',
    'drosophila', 'c. elegans', 'zebrafish', 'rat',
    'arabidopsis', 'yeast', 'e. coli'
]

# Common tissues
TISSUES = [
    'brain', 'blood', 'liver', 'kidney', 'heart', 'lung', 'skin',
    'pancreas', 'spleen', 'thymus', 'bone marrow', 'muscle',
    'retina', 'cortex', 'hippocampus', 'pbmc', 'tumor', 'cancer',
    'embryo', 'fetal', 'adult', 'intestine', 'colon', 'breast'
]

def extract_accessions(text):
    """Extract all dataset accession numbers from text."""
    results = {}
    for acc_type, pattern in PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Deduplicate and clean
            results[acc_type] = list(set([m.strip() for m in matches]))
    return results

def extract_cell_count(text):
    """Extract cell count from text."""
    for pattern in NUMBER_PATTERNS['n_cells']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            count_str = match.group(1).replace(',', '')
            # Handle "k" suffix
            if 'k' in pattern.lower():
                return int(float(count_str) * 1000)
            return int(count_str)
    return None

def extract_gene_count(text):
    """Extract gene count from text."""
    for pattern in NUMBER_PATTERNS['n_genes']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            count_str = match.group(1).replace(',', '')
            if 'k' in pattern.lower():
                return int(float(count_str) * 1000)
            return int(count_str)
    return None

def extract_organism(text):
    """Extract organism from text."""
    text_lower = text.lower()
    for organism in ORGANISMS:
        if organism.lower() in text_lower:
            return organism.title()
    return None

def extract_tissue(text):
    """Extract tissue type from text."""
    text_lower = text.lower()
    found_tissues = []
    for tissue in TISSUES:
        if tissue.lower() in text_lower:
            found_tissues.append(tissue.title())
    return ', '.join(found_tissues[:3]) if found_tissues else None

def extract_platform(text):
    """Extract sequencing platform from text."""
    platforms = {
        '10x': ['10x', '10X Genomics', 'chromium'],
        'Drop-seq': ['drop-seq', 'dropseq'],
        'Smart-seq': ['smart-seq', 'smartseq', 'smart-seq2'],
        'Seq-Well': ['seq-well', 'seqwell'],
        'inDrop': ['indrop'],
        'MARS-seq': ['mars-seq'],
        'CEL-seq': ['cel-seq'],
        'Visium': ['visium', 'spatial transcriptomics'],
    }

    text_lower = text.lower()
    for platform_name, keywords in platforms.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return platform_name
    return None

def process_paper(paper_id, pmid, abstract, title):
    """Process a single paper to extract dataset information."""
    # Combine text for searching
    full_text = f"{title} {abstract}".lower()

    # Extract accessions
    accessions = extract_accessions(full_text)

    # Extract characteristics
    organism = extract_organism(full_text)
    tissue = extract_tissue(full_text)
    n_cells = extract_cell_count(full_text)
    n_genes = extract_gene_count(full_text)
    platform = extract_platform(full_text)

    # Update papers table
    updates = {}
    if 'geo' in accessions:
        updates['has_geo_accession'] = 1
        updates['geo_accession'] = ', '.join(accessions['geo'])
    if 'github' in accessions:
        updates['has_github'] = 1
        updates['github_url'] = accessions['github'][0]  # Take first URL

    # Create dataset records
    datasets = []

    # GEO datasets
    if 'geo' in accessions:
        for geo_id in accessions['geo']:
            dataset = {
                'paper_id': paper_id,
                'accession_type': 'GEO',
                'accession_id': geo_id,
                'organism': organism,
                'tissue_type': tissue,
                'n_cells': n_cells,
                'n_genes': n_genes,
                'sequencing_platform': platform,
            }
            datasets.append(dataset)

    # SRA datasets
    if 'sra' in accessions:
        for sra_id in accessions['sra']:
            dataset = {
                'paper_id': paper_id,
                'accession_type': 'SRA',
                'accession_id': sra_id,
                'organism': organism,
                'tissue_type': tissue,
                'n_cells': n_cells,
                'n_genes': n_genes,
                'sequencing_platform': platform,
            }
            datasets.append(dataset)

    # ArrayExpress datasets
    if 'arrayexpress' in accessions:
        for ae_id in accessions['arrayexpress']:
            dataset = {
                'paper_id': paper_id,
                'accession_type': 'ArrayExpress',
                'accession_id': ae_id,
                'organism': organism,
                'tissue_type': tissue,
                'n_cells': n_cells,
                'n_genes': n_genes,
                'sequencing_platform': platform,
            }
            datasets.append(dataset)

    # Zenodo datasets
    if 'zenodo' in accessions:
        for zenodo_id in accessions['zenodo']:
            dataset = {
                'paper_id': paper_id,
                'accession_type': 'zenodo',
                'accession_id': zenodo_id,
                'organism': organism,
                'tissue_type': tissue,
                'n_cells': n_cells,
                'n_genes': n_genes,
                'sequencing_platform': platform,
            }
            datasets.append(dataset)

    return updates, datasets

def main():
    """Main execution."""
    print("=" * 80)
    print("DATASET EXTRACTION (LOCAL)")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get papers that haven't been processed
    cursor.execute("""
        SELECT id, pmid, title, abstract
        FROM papers
        WHERE abstract IS NOT NULL AND abstract != ''
    """)

    papers = cursor.fetchall()
    print(f"Processing {len(papers)} papers...")
    print()

    papers_updated = 0
    datasets_inserted = 0
    geo_found = 0
    github_found = 0

    for paper in papers:
        paper_id = paper['id']
        pmid = paper['pmid']
        title = paper['title'] or ''
        abstract = paper['abstract'] or ''

        # Process paper
        updates, datasets = process_paper(paper_id, pmid, abstract, title)

        # Update papers table
        if updates:
            update_fields = ', '.join([f"{k} = ?" for k in updates.keys()])
            update_values = list(updates.values()) + [paper_id]
            cursor.execute(f"UPDATE papers SET {update_fields} WHERE id = ?", update_values)
            papers_updated += 1

            if updates.get('has_geo_accession'):
                geo_found += 1
                print(f"[{pmid}] GEO: {updates['geo_accession']}")
            if updates.get('has_github'):
                github_found += 1
                print(f"[{pmid}] GitHub: {updates['github_url']}")

        # Insert datasets
        for dataset in datasets:
            try:
                cursor.execute("""
                    INSERT INTO datasets (
                        paper_id, accession_type, accession_id,
                        organism, tissue_type, n_cells, n_genes,
                        sequencing_platform
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset['paper_id'], dataset['accession_type'], dataset['accession_id'],
                    dataset['organism'], dataset['tissue_type'], dataset['n_cells'],
                    dataset['n_genes'], dataset['sequencing_platform']
                ))
                datasets_inserted += 1
            except sqlite3.IntegrityError:
                pass  # Dataset already exists

    conn.commit()
    conn.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Papers processed: {len(papers)}")
    print(f"Papers updated: {papers_updated}")
    print(f"Papers with GEO: {geo_found}")
    print(f"Papers with GitHub: {github_found}")
    print(f"Datasets inserted: {datasets_inserted}")
    print("=" * 80)

if __name__ == "__main__":
    main()
