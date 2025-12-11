#!/usr/bin/env python3
"""
Analyze papers in the database.
"""
import sqlite3
from pathlib import Path
from collections import Counter
import json

db_path = Path.home() / 'llm-paper-analyze/data/papers/metadata/papers.db'

def analyze_papers():
    """Generate analysis of papers in database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all papers
    cursor.execute("""
        SELECT pmid, title, journal, publication_date, keywords, mesh_terms, abstract
        FROM papers
        ORDER BY publication_date DESC
    """)
    papers = cursor.fetchall()

    total = len(papers)

    print("=" * 100)
    print(f"PAPER DATABASE ANALYSIS ({total} papers)")
    print("=" * 100)
    print()

    # Papers by year
    years = Counter()
    for paper in papers:
        pub_date = paper[3]
        if pub_date:
            year = pub_date.split('-')[0]
            years[year] += 1

    print("PAPERS BY YEAR:")
    for year in sorted(years.keys(), reverse=True):
        bar = "█" * years[year]
        print(f"  {year}: {bar} ({years[year]})")
    print()

    # Recent papers (2024-2025)
    recent = [p for p in papers if p[3] and p[3].startswith('2024') or p[3].startswith('2025')]
    print(f"RECENT PAPERS (2024-2025): {len(recent)} papers")
    for i, paper in enumerate(recent[:10], 1):
        pmid, title, journal, pub_date = paper[:4]
        title_short = title[:70] + "..." if len(title) > 70 else title
        print(f"  {i}. [{pub_date}] {title_short}")
        print(f"     {journal}")
        print(f"     https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        print()

    if len(recent) > 10:
        print(f"  ... and {len(recent) - 10} more recent papers")
    print()

    # Top journals
    journals = Counter()
    for paper in papers:
        journal = paper[2]
        if journal:
            journals[journal] += 1

    print("TOP JOURNALS:")
    for journal, count in journals.most_common(10):
        journal_short = journal[:60] + "..." if len(journal) > 60 else journal
        print(f"  {count:2d} papers: {journal_short}")
    print()

    # Key topics from titles
    print("COMMON TOPICS IN TITLES:")
    title_words = Counter()
    exclude = {'a', 'an', 'the', 'of', 'in', 'and', 'or', 'for', 'to', 'from', 'with', 'by', 'through'}

    for paper in papers:
        title = paper[1].lower()
        words = title.replace('-', ' ').split()
        for word in words:
            word_clean = word.strip('.:,;()[]{}')
            if len(word_clean) > 3 and word_clean not in exclude:
                title_words[word_clean] += 1

    for word, count in title_words.most_common(20):
        if count > 1:
            print(f"  {word}: {count}")
    print()

    # Algorithms mentioned
    algorithms = {
        'pca': 'PCA',
        't-sne': 't-SNE',
        'tsne': 't-SNE',
        'umap': 'UMAP',
        'phate': 'PHATE',
        'autoencoder': 'Autoencoder',
        'vae': 'VAE',
        'scvi': 'scVI',
        'nmf': 'NMF',
        'ica': 'ICA',
        'diffusion': 'Diffusion Maps'
    }

    algo_counts = Counter()
    for paper in papers:
        title = paper[1].lower()
        abstract = paper[6].lower() if paper[6] else ""
        text = title + " " + abstract

        for key, name in algorithms.items():
            if key in text:
                algo_counts[name] += 1

    if algo_counts:
        print("DIMENSIONALITY REDUCTION ALGORITHMS MENTIONED:")
        for algo, count in algo_counts.most_common():
            bar = "█" * count
            print(f"  {algo:20s}: {bar} ({count})")
        print()

    # Papers with datasets
    cursor.execute("SELECT COUNT(*) FROM papers WHERE has_geo_accession = 1")
    geo_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM papers WHERE has_github = 1")
    github_count = cursor.fetchone()[0]

    print("DATA AVAILABILITY:")
    print(f"  Papers with GEO accession: {geo_count}")
    print(f"  Papers with GitHub links: {github_count}")
    print()

    print("=" * 100)
    print(f"Total Papers: {total}")
    print(f"Date Range: {min(years.keys())} - {max(years.keys())}")
    print(f"Database: {db_path}")
    print("=" * 100)

    conn.close()

if __name__ == "__main__":
    analyze_papers()
