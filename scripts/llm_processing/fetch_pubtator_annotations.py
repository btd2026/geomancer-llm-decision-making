#!/usr/bin/env python3
"""
Fetch PubTator3 annotations for papers.
Stores entity annotations in database.
"""
import json
import sqlite3
import requests
import time
from pathlib import Path
from datetime import datetime

# Database path
db_path = Path(__file__).parent.parent / 'data' / 'papers' / 'metadata' / 'papers.db'

# PubTator3 API
PUBTATOR_API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"

def fetch_pubtator3_annotations(pmid):
    """Fetch PubTator3 annotations for a PMID."""
    url = f"{PUBTATOR_API}?pmids={pmid}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # PubTator3 returns array of documents
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # Get first document
        return data

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching PubTator3 for PMID {pmid}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Error parsing JSON for PMID {pmid}: {e}")
        return None

def parse_annotations(pubtator_data):
    """Parse PubTator3 JSON and extract entity annotations."""
    annotations = []

    if not pubtator_data:
        return annotations

    passages = pubtator_data.get('passages', [])

    for passage in passages:
        # Determine section
        section_info = passage.get('infons', {})
        section = section_info.get('section_type', section_info.get('type', 'abstract'))

        # Parse annotations
        for annotation in passage.get('annotations', []):
            infons = annotation.get('infons', {})

            # Get entity type and identifier
            entity_type = infons.get('type', 'unknown')
            entity_id = infons.get('identifier', '')

            # Get text and location
            entity_text = annotation.get('text', '')
            locations = annotation.get('locations', [])

            if locations:
                start_pos = locations[0].get('offset', 0)
                length = locations[0].get('length', 0)
                end_pos = start_pos + length
            else:
                start_pos = 0
                end_pos = 0

            entity = {
                'entity_type': entity_type,
                'entity_text': entity_text,
                'entity_id': entity_id,
                'entity_name': infons.get('name', entity_text),
                'section': section,
                'start_pos': start_pos,
                'end_pos': end_pos
            }
            annotations.append(entity)

    return annotations

def create_tables(cursor):
    """Create PubTator annotations table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pubtator_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            entity_type TEXT NOT NULL,
            entity_text TEXT NOT NULL,
            entity_id TEXT,
            entity_name TEXT,
            section TEXT,
            start_pos INTEGER,
            end_pos INTEGER,
            confidence_score REAL,
            source TEXT DEFAULT 'pubtator3',
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pubtator_paper
        ON pubtator_annotations(paper_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pubtator_type
        ON pubtator_annotations(entity_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pubtator_id
        ON pubtator_annotations(entity_id)
    """)

def main():
    """Main execution."""
    print("=" * 80)
    print("FETCHING PUBTATOR3 ANNOTATIONS")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Create tables
    create_tables(cursor)
    conn.commit()

    # Get all papers
    cursor.execute("SELECT id, pmid, title FROM papers WHERE pmid IS NOT NULL")
    papers = cursor.fetchall()

    print(f"Processing {len(papers)} papers...\n")

    total_annotations = 0
    papers_processed = 0
    papers_skipped = 0
    papers_failed = 0

    for i, paper in enumerate(papers, 1):
        paper_id = paper['id']
        pmid = paper['pmid']

        # Check if already processed
        cursor.execute(
            "SELECT COUNT(*) FROM pubtator_annotations WHERE paper_id = ?",
            (paper_id,)
        )
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            print(f"[{i}/{len(papers)}] PMID {pmid}: Already processed ({existing_count} annotations), skipping...")
            papers_skipped += 1
            continue

        print(f"[{i}/{len(papers)}] PMID {pmid}: Fetching PubTator3 annotations...")

        # Fetch annotations
        pubtator_data = fetch_pubtator3_annotations(pmid)

        if not pubtator_data:
            papers_failed += 1
            print(f"  → Failed to fetch")
            # Rate limit
            time.sleep(0.5)
            continue

        # Parse annotations
        annotations = parse_annotations(pubtator_data)

        if not annotations:
            print(f"  → No annotations found")
            papers_processed += 1
            time.sleep(0.5)
            continue

        # Count by type
        type_counts = {}
        for ann in annotations:
            entity_type = ann['entity_type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        # Store in database
        for ann in annotations:
            cursor.execute("""
                INSERT INTO pubtator_annotations (
                    paper_id, entity_type, entity_text, entity_id, entity_name,
                    section, start_pos, end_pos
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper_id, ann['entity_type'], ann['entity_text'],
                ann['entity_id'], ann['entity_name'], ann['section'],
                ann['start_pos'], ann['end_pos']
            ))

        total_annotations += len(annotations)
        papers_processed += 1

        # Print summary
        type_summary = ', '.join([f"{t}: {c}" for t, c in sorted(type_counts.items())])
        print(f"  → {len(annotations)} annotations ({type_summary})")

        # Commit every 10 papers
        if papers_processed % 10 == 0:
            conn.commit()
            print(f"\n--- Progress: {papers_processed + papers_skipped}/{len(papers)} papers processed ---\n")

        # Rate limit (avoid overloading NCBI API)
        time.sleep(0.5)

    conn.commit()
    conn.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total papers: {len(papers)}")
    print(f"Papers processed: {papers_processed}")
    print(f"Papers skipped (already done): {papers_skipped}")
    print(f"Papers failed: {papers_failed}")
    print(f"Total annotations extracted: {total_annotations}")

    if total_annotations > 0:
        print(f"Average annotations per paper: {total_annotations / papers_processed:.1f}")

    print("=" * 80)

    # Show entity type breakdown
    print()
    print("Entity type breakdown:")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT entity_type, COUNT(*) as count
        FROM pubtator_annotations
        GROUP BY entity_type
        ORDER BY count DESC
    """)

    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    conn.close()

if __name__ == "__main__":
    main()
