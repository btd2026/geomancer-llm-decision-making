#!/usr/bin/env python3
"""
View papers from database with clickable links.
Run this inside container or on host with sqlite3.
"""
import sqlite3
import json
from pathlib import Path
import sys

# Database path (adjust if running on host)
db_path = '/workspace/metadata/papers.db' if Path('/workspace').exists() else Path.home() / 'llm-paper-analyze/data/papers/metadata/papers.db'

def view_papers(limit=10, recent_only=False):
    """Display papers with links."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check if pubmed_url column exists
    cursor.execute("PRAGMA table_info(papers)")
    columns = [col[1] for col in cursor.fetchall()]
    has_urls = 'pubmed_url' in columns

    query = """
        SELECT pmid, doi, title, journal, publication_date, authors
        FROM papers
    """

    if recent_only:
        query += " WHERE publication_date >= '2024'"

    query += " ORDER BY publication_date DESC, id DESC"

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    papers = cursor.fetchall()

    if not papers:
        print("No papers found in database")
        conn.close()
        return

    print("=" * 100)
    print(f"PAPERS IN DATABASE ({len(papers)} shown)")
    print("=" * 100)
    print()

    for i, paper in enumerate(papers, 1):
        pmid, doi, title, journal, pub_date, authors = paper

        # Generate URLs
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "N/A"
        doi_url = f"https://doi.org/{doi}" if doi else "N/A"

        # Truncate long fields
        title_short = title[:90] + "..." if len(title) > 90 else title
        authors_short = authors[:60] + "..." if authors and len(authors) > 60 else (authors or "N/A")

        print(f"{i}. [{pmid}] {title_short}")
        print(f"   Journal: {journal}")
        print(f"   Date: {pub_date}")
        print(f"   Authors: {authors_short}")
        print(f"   PubMed: {pubmed_url}")
        if doi:
            print(f"   DOI: {doi_url}")
        print()

    conn.close()

    print("=" * 100)
    print(f"To open a paper, copy the PubMed URL and paste in your browser")
    print("=" * 100)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    recent = '--recent' in sys.argv or '-r' in sys.argv

    view_papers(limit=limit, recent_only=recent)
