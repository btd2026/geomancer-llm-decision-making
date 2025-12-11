#!/usr/bin/env python3
"""
Fetch full text from PubMed Central for open access articles.
Stores Methods and Data Availability sections for enhanced extraction.

Methods section extraction is focused on dataset-relevant subsections:
- Extracts only subsections with keywords: data, dataset, single-cell, sequencing,
  preprocessing, quality control, normalization, dimensionality reduction, etc.
- Limits each subsection to 2000 characters
- Overall limit of 6000 characters for Methods section
- Falls back to first 3000 chars if no relevant subsections found
"""
import json
import sqlite3
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from pathlib import Path

# Database path
db_path = Path(__file__).parent.parent / 'data' / 'papers' / 'metadata' / 'papers.db'

# NCBI E-utilities settings
EMAIL = "your_email@example.com"  # TODO: Get from config
TOOL = "llm-paper-analyze"
BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

def get_pmc_id(pmid):
    """Convert PMID to PMC ID using id converter."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json&tool={TOOL}&email={EMAIL}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())

        records = data.get('records', [])
        if records and len(records) > 0:
            record = records[0]
            pmcid = record.get('pmcid', None)
            return pmcid
        return None

    except Exception as e:
        print(f"  Error converting PMID {pmid} to PMC ID: {e}")
        return None

def fetch_pmc_fulltext(pmcid):
    """Fetch full text XML from PMC for a given PMC ID."""
    # Remove PMC prefix if present
    if pmcid.startswith('PMC'):
        pmcid_num = pmcid[3:]
    else:
        pmcid_num = pmcid

    url = f"{BASE_URL}efetch.fcgi?db=pmc&id={pmcid_num}&retmode=xml&tool={TOOL}&email={EMAIL}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_data = response.read()

        return xml_data

    except Exception as e:
        print(f"  Error fetching full text for {pmcid}: {e}")
        return None

def extract_sections(xml_data):
    """Extract specific sections from PMC XML, focusing on dataset-relevant information."""
    if not xml_data:
        return {}

    try:
        root = ET.fromstring(xml_data)

        sections = {}

        # Find article body
        body = root.find('.//body')
        if body is None:
            return sections

        # Extract Methods section - focus on dataset-relevant subsections
        methods_section = None
        for sec in body.findall('.//sec'):
            title = sec.find('.//title')
            if title is not None and title.text:
                title_lower = title.text.lower()
                if any(keyword in title_lower for keyword in ['method', 'material', 'experimental']):
                    methods_section = sec
                    break

        if methods_section is not None:
            # Extract only dataset-relevant subsections from Methods
            methods_text = extract_dataset_relevant_subsections(methods_section)
            if methods_text:
                sections['methods'] = methods_text

        # Extract Data Availability section
        back = root.find('.//back')
        if back is not None:
            for sec in back.findall('.//sec'):
                title = sec.find('.//title')
                if title is not None and title.text:
                    title_lower = title.text.lower()
                    if any(keyword in title_lower for keyword in ['data availability', 'data access', 'accession']):
                        data_avail_text = extract_text_from_element(sec)
                        sections['data_availability'] = data_avail_text
                        break

        # Also check for data availability in notes
        for note in root.findall('.//notes'):
            note_type = note.get('notes-type', '').lower()
            if 'data' in note_type or 'availability' in note_type:
                data_note_text = extract_text_from_element(note)
                if 'data_availability' in sections:
                    sections['data_availability'] += '\n\n' + data_note_text
                else:
                    sections['data_availability'] = data_note_text

        return sections

    except Exception as e:
        print(f"  Error parsing XML: {e}")
        return {}

def extract_dataset_relevant_subsections(methods_section):
    """Extract only subsections relevant to datasets from Methods section."""
    relevant_keywords = [
        'data', 'dataset', 'single-cell', 'scrna', 'scRNA-seq',
        'sequencing', 'sample', 'geo', 'accession',
        'preprocessing', 'quality control', 'normalization',
        'dimension', 'reduction', 'umap', 'pca', 't-sne', 'tsne'
    ]

    relevant_parts = []

    # Check all subsections within Methods
    for subsec in methods_section.findall('.//sec'):
        subsec_title = subsec.find('.//title')
        if subsec_title is not None and subsec_title.text:
            title_lower = subsec_title.text.lower()

            # Include subsection if title contains relevant keywords
            if any(keyword in title_lower for keyword in relevant_keywords):
                subsec_text = extract_text_from_element(subsec)
                # Limit each subsection to 2000 characters for conciseness
                if len(subsec_text) > 2000:
                    subsec_text = subsec_text[:2000] + '...[truncated]'
                relevant_parts.append(f"## {subsec_title.text}\n{subsec_text}")

    # If no relevant subsections found, extract first ~3000 chars of entire Methods
    if not relevant_parts:
        full_methods = extract_text_from_element(methods_section)
        if len(full_methods) > 3000:
            full_methods = full_methods[:3000] + '...[truncated]'
        return full_methods

    # Combine relevant subsections
    combined = '\n\n'.join(relevant_parts)

    # Apply overall limit to prevent extremely long extractions
    if len(combined) > 6000:
        combined = combined[:6000] + '...[truncated]'

    return combined

def extract_text_from_element(element):
    """Recursively extract text from an XML element."""
    text_parts = []

    # Get element text
    if element.text:
        text_parts.append(element.text.strip())

    # Process children
    for child in element:
        child_text = extract_text_from_element(child)
        if child_text:
            text_parts.append(child_text)

        # Get tail text after child
        if child.tail:
            text_parts.append(child.tail.strip())

    return ' '.join([t for t in text_parts if t])

def update_database_schema(cursor):
    """Add PMC-related columns to papers table."""
    try:
        cursor.execute("ALTER TABLE papers ADD COLUMN pmcid TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE papers ADD COLUMN has_fulltext INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE papers ADD COLUMN methods_section TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE papers ADD COLUMN data_availability_section TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE papers ADD COLUMN fulltext_fetched_at TIMESTAMP")
    except sqlite3.OperationalError:
        pass

def main():
    """Main execution."""
    print("=" * 80)
    print("FETCHING PMC FULL TEXT")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Update schema
    print("Updating database schema...")
    update_database_schema(cursor)
    conn.commit()
    print()

    # Get all papers
    cursor.execute("""
        SELECT id, pmid, title
        FROM papers
        WHERE pmid IS NOT NULL
    """)

    papers = cursor.fetchall()
    print(f"Processing {len(papers)} papers...\n")

    papers_with_pmc = 0
    papers_with_fulltext = 0
    papers_failed = 0
    papers_skipped = 0

    for i, paper in enumerate(papers, 1):
        paper_id = paper['id']
        pmid = paper['pmid']

        # Check if already processed
        cursor.execute(
            "SELECT pmcid, has_fulltext FROM papers WHERE id = ?",
            (paper_id,)
        )
        result = cursor.fetchone()
        if result and result['has_fulltext']:
            print(f"[{i}/{len(papers)}] PMID {pmid}: Already has full text, skipping...")
            papers_skipped += 1
            continue

        print(f"[{i}/{len(papers)}] PMID {pmid}: Checking PMC availability...")

        # Get PMC ID
        pmcid = get_pmc_id(pmid)

        if not pmcid:
            print(f"  → Not in PMC Open Access")
            papers_failed += 1
            time.sleep(0.34)
            continue

        print(f"  → Found PMC ID: {pmcid}")
        papers_with_pmc += 1

        # Update PMC ID
        cursor.execute(
            "UPDATE papers SET pmcid = ? WHERE id = ?",
            (pmcid, paper_id)
        )

        # Fetch full text
        print(f"  → Fetching full text...")
        xml_data = fetch_pmc_fulltext(pmcid)

        if not xml_data:
            print(f"  → Failed to fetch full text")
            papers_failed += 1
            time.sleep(0.34)
            continue

        # Extract sections
        sections = extract_sections(xml_data)

        if not sections:
            print(f"  → No Methods/Data Availability sections found")
            time.sleep(0.34)
            continue

        # Update database
        methods = sections.get('methods', '')
        data_avail = sections.get('data_availability', '')

        cursor.execute("""
            UPDATE papers
            SET methods_section = ?,
                data_availability_section = ?,
                has_fulltext = 1,
                fulltext_fetched_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (methods, data_avail, paper_id))

        papers_with_fulltext += 1

        # Show what we got
        extracted = []
        if methods:
            extracted.append(f"Methods: {len(methods)} chars")
        if data_avail:
            extracted.append(f"Data Avail: {len(data_avail)} chars")

        print(f"  → Extracted: {', '.join(extracted)}")

        # Commit every 5 papers
        if (i % 5) == 0:
            conn.commit()
            print(f"\n--- Progress: {i}/{len(papers)} papers checked ---\n")

        # Rate limiting
        time.sleep(0.34)

    conn.commit()
    conn.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total papers: {len(papers)}")
    print(f"Papers skipped (already processed): {papers_skipped}")
    print(f"Papers with PMC ID: {papers_with_pmc}")
    print(f"Papers with full text extracted: {papers_with_fulltext}")
    print(f"Papers without PMC access: {papers_failed}")

    if papers_with_pmc > 0:
        pmc_rate = (papers_with_pmc / (len(papers) - papers_skipped)) * 100
        print(f"PMC availability rate: {pmc_rate:.1f}%")

    if papers_with_fulltext > 0:
        success_rate = (papers_with_fulltext / papers_with_pmc) * 100
        print(f"Full text extraction success rate: {success_rate:.1f}%")

    print("=" * 80)

if __name__ == "__main__":
    main()
