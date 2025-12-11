#!/usr/bin/env python3
"""
Extract enhanced dataset information from full-text Methods and Data Availability sections.
Complements abstract-based extraction with detailed information from full papers.
"""
import json
import re
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent.parent / 'data' / 'papers' / 'metadata' / 'papers.db'

# Enhanced patterns for full-text extraction
DATASET_PATTERNS = {
    'geo': [
        r'GEO\s*(?:accession|ID|number)?\s*:?\s*(GS[EM]\d+)',
        r'Gene Expression Omnibus\s*\(?\s*(GSE\d+)',
        r'available\s+(?:at|from|in)\s+GEO\s*:?\s*(GSE\d+)',
        r'deposited\s+(?:at|in)\s+GEO\s*:?\s*(GSE\d+)',
    ],
    'sra': [
        r'SRA\s*(?:accession|ID)?\s*:?\s*(SRP\d+|SRX\d+|SRR\d+)',
        r'Sequence Read Archive\s*:?\s*(SRP\d+)',
    ],
    'arrayexpress': [
        r'ArrayExpress\s*(?:accession|ID)?\s*:?\s*(E-[A-Z]+-\d+)',
    ],
}

# Detailed cell count patterns for Methods sections
CELL_COUNT_PATTERNS = [
    r'(?:sequenced|profiled|analyzed)\s+(\d+(?:,\d+)*)\s+(?:single\s*)?cells',
    r'(?:total|final)\s+(?:of\s+)?(\d+(?:,\d+)*)\s+cells',
    r'(\d+(?:,\d+)*)\s+cells?\s+(?:were|was)\s+(?:sequenced|analyzed|profiled)',
    r'captured\s+(\d+(?:,\d+)*)\s+cells?',
    r'(\d+)k\s+cells',
]

# Platform detection with more specificity
PLATFORM_PATTERNS = {
    '10x Chromium v2': [r'10[xX]\s+Genomics\s+Chromium\s+(?:Single\s+Cell)?\s*v?2', r'10[xX]\s+v2'],
    '10x Chromium v3': [r'10[xX]\s+Genomics\s+Chromium\s+(?:Single\s+Cell)?\s*v?3', r'10[xX]\s+v3'],
    '10x Chromium v3.1': [r'10[xX]\s+Genomics\s+Chromium\s+(?:Single\s+Cell)?\s*v?3\.1'],
    '10x Chromium': [r'10[xX]\s+Genomics\s+Chromium', r'10[xX]\s+Genomics'],
    'Drop-seq': [r'Drop-seq', r'DropSeq'],
    'Smart-seq2': [r'Smart-seq\s*2', r'Smart-seq2', r'Smartseq2'],
    'Smart-seq3': [r'Smart-seq\s*3', r'Smart-seq3'],
    'inDrop': [r'inDrop'],
    'MARS-seq': [r'MARS-seq'],
    'CEL-seq2': [r'CEL-seq\s*2', r'CEL-seq2'],
    'STRT-seq': [r'STRT-seq'],
    'Seq-Well': [r'Seq-Well'],
    'Visium': [r'Visium', r'10[xX]\s+(?:Genomics\s+)?Visium'],
}

# Preprocessing methods
PREPROCESSING_PATTERNS = {
    'CellRanger': r'Cell\s*Ranger\s+(?:v?(\d+\.?\d*))?',
    'STARsolo': r'STARsolo|STAR\s+solo',
    'Alevin': r'Alevin',
    'kallisto|bustools': r'kallisto\s*\|\s*bustools|kb-python',
}

# Normalization details
NORMALIZATION_PATTERNS = {
    'scran': r'scran.*?(?:v(?:ersion)?\s*(\d+\.?\d*))?',
    'SCTransform': r'SCTransform',
    'Seurat': r'Seurat.*?(?:v(?:ersion)?\s*(\d+\.?\d*))?',
    'log-normalization': r'log[- ]normalized|log\s+transformation',
}

def extract_datasets_from_fulltext(methods_text, data_avail_text):
    """Extract dataset accessions from full-text sections."""
    datasets = []
    combined_text = f"{methods_text} {data_avail_text}"

    for dataset_type, patterns in DATASET_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if match and match not in [d['accession_id'] for d in datasets]:
                    datasets.append({
                        'accession_type': dataset_type.upper(),
                        'accession_id': match.strip(),
                        'source': 'fulltext'
                    })

    return datasets

def extract_cell_count(methods_text):
    """Extract cell count with better accuracy from Methods section."""
    for pattern in CELL_COUNT_PATTERNS:
        match = re.search(pattern, methods_text, re.IGNORECASE)
        if match:
            count_str = match.group(1).replace(',', '')
            # Check if it's in 'k' format
            if 'k' in pattern:
                return int(float(count_str) * 1000)
            try:
                return int(count_str)
            except:
                continue
    return None

def extract_platform(methods_text):
    """Extract sequencing platform with version specificity."""
    for platform_name, patterns in PLATFORM_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, methods_text, re.IGNORECASE):
                return platform_name
    return None

def extract_preprocessing(methods_text):
    """Extract preprocessing pipeline information."""
    preprocessing = {}

    for tool_name, pattern in PREPROCESSING_PATTERNS.items():
        match = re.search(pattern, methods_text, re.IGNORECASE)
        if match:
            preprocessing[tool_name] = True

    for method_name, pattern in NORMALIZATION_PATTERNS.items():
        match = re.search(pattern, methods_text, re.IGNORECASE)
        if match:
            preprocessing[method_name] = True

    return preprocessing if preprocessing else None

def main():
    """Main execution."""
    print("=" * 80)
    print("EXTRACTING FROM FULL TEXT")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get papers with full text
    cursor.execute("""
        SELECT id, pmid, title, methods_section, data_availability_section
        FROM papers
        WHERE has_fulltext = 1
          AND (methods_section IS NOT NULL OR data_availability_section IS NOT NULL)
    """)

    papers = cursor.fetchall()
    print(f"Processing {len(papers)} papers with full text...\n")

    datasets_found = 0
    cell_counts_found = 0
    platforms_found = 0
    preprocessing_found = 0

    for paper in papers:
        paper_id = paper['id']
        pmid = paper['pmid']
        methods = paper['methods_section'] or ''
        data_avail = paper['data_availability_section'] or ''

        print(f"[{pmid}] Extracting from full text...")

        # Extract datasets
        datasets = extract_datasets_from_fulltext(methods, data_avail)

        if datasets:
            print(f"  → Datasets: {', '.join([d['accession_id'] for d in datasets])}")

            # Insert or update datasets
            for ds in datasets:
                # Check if already exists
                cursor.execute("""
                    SELECT id FROM datasets
                    WHERE paper_id = ? AND accession_id = ?
                """, (paper_id, ds['accession_id']))

                existing = cursor.fetchone()

                if not existing:
                    # Extract additional characteristics
                    cell_count = extract_cell_count(methods)
                    platform = extract_platform(methods)
                    preprocessing = extract_preprocessing(methods)

                    cursor.execute("""
                        INSERT INTO datasets (
                            paper_id, accession_type, accession_id,
                            n_cells, sequencing_platform
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        paper_id, ds['accession_type'], ds['accession_id'],
                        cell_count, platform
                    ))

                    datasets_found += 1

                    if cell_count:
                        cell_counts_found += 1
                    if platform:
                        platforms_found += 1
                    if preprocessing:
                        preprocessing_found += 1

                else:
                    # Update existing with better info from full text
                    updates = {}

                    cell_count = extract_cell_count(methods)
                    if cell_count:
                        updates['n_cells'] = cell_count
                        cell_counts_found += 1

                    platform = extract_platform(methods)
                    if platform:
                        updates['sequencing_platform'] = platform
                        platforms_found += 1

                    if updates:
                        update_sql = ', '.join([f"{k} = ?" for k in updates.keys()])
                        update_values = list(updates.values()) + [existing['id']]
                        cursor.execute(
                            f"UPDATE datasets SET {update_sql} WHERE id = ?",
                            update_values
                        )

        # Extract and display sample preprocessing info
        preprocessing = extract_preprocessing(methods)
        if preprocessing:
            tools = ', '.join(preprocessing.keys())
            print(f"  → Preprocessing: {tools}")

    conn.commit()
    conn.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Papers with full text: {len(papers)}")
    print(f"New datasets found: {datasets_found}")
    print(f"Cell counts extracted: {cell_counts_found}")
    print(f"Platforms identified: {platforms_found}")
    print(f"Preprocessing pipelines found: {preprocessing_found}")
    print("=" * 80)

if __name__ == "__main__":
    main()
