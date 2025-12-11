#!/usr/bin/env python3
"""
Generate concise dataset descriptions for papers.
Uses Claude API if ANTHROPIC_API_KEY is set, otherwise constructs from extracted data.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

# Use local paths
config_dir = Path(__file__).parent.parent / 'configs'
data_dir = Path(__file__).parent.parent / 'data' / 'papers'
db_path = data_dir / 'metadata' / 'papers.db'

# Check if Anthropic API is available
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
USE_API = ANTHROPIC_API_KEY is not None

if USE_API:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Using Claude API for description generation")
    except ImportError:
        print("anthropic package not installed. Install with: pip install anthropic")
        USE_API = False
else:
    print("ANTHROPIC_API_KEY not set. Using programmatic description generation")


def generate_description_from_api(paper):
    """Generate description using Claude API."""
    title = paper['title']
    abstract = paper['abstract'] or ''
    methods = paper.get('methods_section', '')

    # Combine available text
    context = f"Title: {title}\n\nAbstract: {abstract}"
    if methods:
        context += f"\n\nMethods: {methods[:1000]}"  # Limit methods to 1000 chars

    prompt = """Based on the paper information below, generate a concise 1-2 sentence description of the specific datasets used in this study.

Focus on:
- Dataset accession IDs (GEO, SRA, ArrayExpress, etc.)
- Organism (human, mouse, etc.)
- Tissue type
- Number of cells
- Sequencing platform

Format example: "Human PBMC scRNA-seq dataset (GSE123456) with 25,000 cells sequenced using 10x Chromium v3"

Paper information:
{context}

Generate a concise dataset description (1-2 sentences maximum):"""

    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": prompt.format(context=context)
            }]
        )
        description = message.content[0].text.strip()
        return description
    except Exception as e:
        print(f"API error for PMID {paper['pmid']}: {e}")
        return None


def generate_description_from_data(paper, datasets):
    """Generate description programmatically from extracted data."""
    if not datasets:
        # No specific datasets found, generate from paper text
        text = f"{paper['title']} {paper['abstract'] or ''}"

        # Try to extract basic info
        parts = []

        # Organism
        organisms = ['human', 'mouse', 'rat', 'drosophila']
        for org in organisms:
            if org.lower() in text.lower():
                parts.append(org.title())
                break

        # Tissue
        tissues = ['brain', 'blood', 'pbmc', 'liver', 'kidney', 'pancreas', 'tumor', 'cancer']
        for tissue in tissues:
            if tissue.lower() in text.lower():
                parts.append(tissue)
                break

        if parts:
            return f"{' '.join(parts)} scRNA-seq dataset"
        else:
            return "Single-cell RNA-seq dataset"

    # Build description from dataset records
    parts = []

    # Group datasets by type
    geo_ids = [d['accession_id'] for d in datasets if d['accession_type'] == 'GEO']
    sra_ids = [d['accession_id'] for d in datasets if d['accession_type'] == 'SRA']

    # Get common properties from first dataset
    first = datasets[0]
    organism = first.get('organism')
    tissue = first.get('tissue_type')
    n_cells = first.get('n_cells')
    platform = first.get('sequencing_platform')

    # Build description
    if organism:
        parts.append(organism)
    if tissue:
        parts.append(tissue)

    parts.append("scRNA-seq")

    # Add count
    if len(datasets) > 1:
        parts.append(f"({len(datasets)} datasets")
    else:
        parts.append("dataset")

    # Add accessions
    if geo_ids:
        if len(geo_ids) == 1:
            parts.append(f"- {geo_ids[0]}")
        elif len(geo_ids) <= 3:
            parts.append(f"- {', '.join(geo_ids)}")
        else:
            parts.append(f"- {geo_ids[0]} and {len(geo_ids)-1} others")

    if len(datasets) > 1:
        parts.append(")")

    # Add cell count
    if n_cells:
        parts.append(f"with ~{n_cells:,} cells")

    # Add platform
    if platform:
        parts.append(f"using {platform}")

    return ' '.join(parts)


def main():
    """Main execution."""
    print("=" * 80)
    print("DATASET DESCRIPTION GENERATION")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get papers that need descriptions (or regenerate all if --force flag)
    import sys
    force_regenerate = '--force' in sys.argv

    if force_regenerate:
        print("Force regeneration enabled - processing all papers")
        cursor.execute("""
            SELECT id, pmid, title, abstract, data_availability_statement
            FROM papers
            WHERE abstract IS NOT NULL AND abstract != ''
            LIMIT 100
        """)
    else:
        cursor.execute("""
            SELECT id, pmid, title, abstract, data_availability_statement
            FROM papers
            WHERE (dataset_description IS NULL OR dataset_description = '')
            AND abstract IS NOT NULL AND abstract != ''
            LIMIT 100
        """)

    papers = cursor.fetchall()
    total = len(papers)
    print(f"Generating descriptions for {total} papers...")
    print()

    updated = 0
    skipped = 0

    for i, paper in enumerate(papers, 1):
        paper_id = paper['id']
        pmid = paper['pmid']

        # Get associated datasets
        cursor.execute("""
            SELECT *
            FROM datasets
            WHERE paper_id = ?
        """, (paper_id,))
        datasets = [dict(row) for row in cursor.fetchall()]

        # Generate description
        if USE_API:
            description = generate_description_from_api(dict(paper))
            if description is None:
                # Fallback to programmatic
                description = generate_description_from_data(dict(paper), datasets)
        else:
            description = generate_description_from_data(dict(paper), datasets)

        if description:
            # Update database
            cursor.execute("""
                UPDATE papers
                SET dataset_description = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (description, paper_id))
            updated += 1

            # Print progress
            print(f"[{i}/{total}] {pmid}: {description[:80]}...")
        else:
            skipped += 1
            print(f"[{i}/{total}] {pmid}: SKIPPED")

        # Commit every 10 papers
        if i % 10 == 0:
            conn.commit()

    # Final commit
    conn.commit()
    conn.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total papers: {total}")
    print(f"Descriptions generated: {updated}")
    print(f"Skipped: {skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
