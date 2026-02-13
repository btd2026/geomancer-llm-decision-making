#!/usr/bin/env python3
"""
Search PubMed and store results in database - LOCAL VERSION.
Uses NCBI E-utilities API to search for papers.
Filters papers to only include those with valid GEO datasets.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re

# Use local paths instead of container paths
config_dir = Path(__file__).parent.parent / 'configs'
data_dir = Path(__file__).parent.parent / 'data' / 'papers'

# Load research context
with open(config_dir / 'research_context.json') as f:
    context = json.load(f)

# Load MCP config
with open(config_dir / 'mcp_config.json') as f:
    mcp_config = json.load(f)

# Database path (use actual path, not container path)
db_path = data_dir / 'metadata' / 'papers.db'
db_path.parent.mkdir(parents=True, exist_ok=True)

# NCBI E-utilities settings
EMAIL = mcp_config['servers']['ncbi']['config']['email']
TOOL = mcp_config['servers']['ncbi']['config']['tool']
BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

# GEO accession patterns
GEO_PATTERN = re.compile(r'\b(GSE\d+|GSM\d+|GPL\d+|GDS\d+)\b', re.IGNORECASE)

def extract_geo_accessions(text):
    """Extract GEO accession numbers from text."""
    if not text:
        return []

    matches = GEO_PATTERN.findall(text)
    # Remove duplicates and return uppercase versions
    return list(set([m.upper() for m in matches]))

def validate_geo_accession(accession):
    """Validate a GEO accession number using NCBI E-utilities API."""
    try:
        # Use esearch to check if the accession exists in GEO DataSets
        params = {
            'db': 'gds',
            'term': f'{accession}[ACCN]',
            'retmode': 'xml',
            'email': EMAIL,
            'tool': TOOL
        }

        url = BASE_URL + 'esearch.fcgi?' + urllib.parse.urlencode(params)

        # Rate limiting
        time.sleep(0.34)

        with urllib.request.urlopen(url, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        count = root.find('.//Count')

        if count is not None and int(count.text) > 0:
            return True
        return False

    except Exception as e:
        print(f"  Warning: Could not validate GEO accession {accession}: {e}")
        return False

def validate_geo_accessions(accessions):
    """Validate a list of GEO accessions and return only valid ones."""
    valid_accessions = []

    for acc in accessions:
        # Prioritize GSE (Series) accessions as they represent complete datasets
        if acc.startswith('GSE'):
            if validate_geo_accession(acc):
                valid_accessions.append(acc)

    # If no GSE accessions are valid, check other types
    if not valid_accessions:
        for acc in accessions:
            if not acc.startswith('GSE'):
                if validate_geo_accession(acc):
                    valid_accessions.append(acc)

    return valid_accessions

def build_pubmed_query():
    """Build PubMed search query from research context."""
    criteria = context['search_criteria']

    # Required keywords (any)
    required = ' OR '.join([f'"{kw}"[Title/Abstract]' for kw in criteria['required_keywords_any']])

    # DR keywords and trajectory keywords
    all_categories = criteria['required_keywords_all_from_one_category']

    # Combine dimensionality reduction and trajectory analysis terms
    method_terms = []
    if 'dimensionality_reduction' in all_categories:
        method_terms.extend(all_categories['dimensionality_reduction'])
    if 'trajectory_analysis' in all_categories:
        method_terms.extend(all_categories['trajectory_analysis'])

    method_query = ' OR '.join([f'"{kw}"[Title/Abstract]' for kw in method_terms])

    # GEO dataset requirement - search for common GEO accession patterns
    geo_query = '("GEO"[Title/Abstract] OR "GSE"[Title/Abstract] OR "Gene Expression Omnibus"[Title/Abstract] OR "data availability"[Title/Abstract])'

    # Date range
    date_range = criteria['date_range']
    date_filter = f'("{date_range["start"]}"[Publication Date] : "{date_range["end"]}"[Publication Date])'

    # Combine - require single-cell + methods + GEO + date range
    full_query = f'({required}) AND ({method_query}) AND {geo_query} AND {date_filter}'

    return full_query

def search_pubmed(query, max_results=20):
    """Search PubMed and return list of PMIDs."""
    print(f"Searching PubMed for up to {max_results} papers...")
    print(f"Query: {query[:150]}...")

    # Build URL for esearch
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'xml',
        'email': EMAIL,
        'tool': TOOL,
        'sort': 'relevance'
    }

    url = BASE_URL + 'esearch.fcgi?' + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()

        # Parse XML
        root = ET.fromstring(xml_data)

        # Extract PMIDs
        pmids = [id_elem.text for id_elem in root.findall('.//Id')]

        count = root.find('.//Count')
        total_count = int(count.text) if count is not None else 0

        print(f"Found {total_count} total papers, fetching {len(pmids)} papers")

        return pmids, total_count

    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return [], 0

def fetch_paper_details(pmids):
    """Fetch detailed information for a list of PMIDs."""
    if not pmids:
        return []

    print(f"Fetching details for {len(pmids)} papers...")

    # Build URL for efetch
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml',
        'email': EMAIL,
        'tool': TOOL
    }

    url = BASE_URL + 'efetch.fcgi?' + urllib.parse.urlencode(params)

    try:
        # Rate limiting (3 requests per second without API key)
        time.sleep(0.34)

        with urllib.request.urlopen(url) as response:
            xml_data = response.read()

        # Parse XML
        root = ET.fromstring(xml_data)

        papers = []
        for article in root.findall('.//PubmedArticle'):
            paper = extract_paper_info(article)
            if paper:
                papers.append(paper)

        print(f"Successfully parsed {len(papers)} papers")
        return papers

    except Exception as e:
        print(f"Error fetching paper details: {e}")
        return []

def extract_paper_info(article_elem):
    """Extract paper information from XML element."""
    try:
        # PMID
        pmid_elem = article_elem.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else None

        # Article details
        article = article_elem.find('.//Article')
        if article is None:
            return None

        # Title
        title_elem = article.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else 'No title'

        # Abstract
        abstract_parts = article.findall('.//AbstractText')
        abstract = ' '.join([p.text for p in abstract_parts if p.text]) if abstract_parts else ''

        # Authors
        author_list = article.findall('.//Author')
        authors = []
        for author in author_list[:10]:  # Limit to first 10 authors
            last = author.find('.//LastName')
            first = author.find('.//ForeName')
            if last is not None:
                name = last.text
                if first is not None:
                    name = f"{first.text} {name}"
                authors.append(name)
        authors_str = ', '.join(authors) if authors else ''

        # Journal
        journal_elem = article.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else ''

        # Publication date
        pub_date = article.find('.//PubDate')
        pub_year = pub_date.find('.//Year')
        pub_month = pub_date.find('.//Month')
        pub_day = pub_date.find('.//Day')

        date_str = ''
        if pub_year is not None:
            date_str = pub_year.text
            if pub_month is not None:
                date_str += f"-{pub_month.text}"
            if pub_day is not None:
                date_str += f"-{pub_day.text}"

        # DOI
        doi = None
        article_ids = article_elem.findall('.//ArticleId')
        for aid in article_ids:
            if aid.get('IdType') == 'doi':
                doi = aid.text
                break

        # MeSH terms
        mesh_terms = []
        mesh_headings = article_elem.findall('.//MeshHeading/DescriptorName')
        mesh_terms = [m.text for m in mesh_headings if m.text]

        # Keywords
        keywords = []
        kw_list = article.findall('.//Keyword')
        keywords = [kw.text for kw in kw_list if kw.text]

        # Generate URLs
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        doi_url = f"https://doi.org/{doi}" if doi else None

        # Extract GEO accessions from abstract and title
        geo_text = f"{title} {abstract}"
        geo_accessions = extract_geo_accessions(geo_text)

        return {
            'pmid': pmid,
            'doi': doi,
            'title': title,
            'abstract': abstract,
            'authors': authors_str,
            'journal': journal,
            'publication_date': date_str,
            'keywords': json.dumps(keywords),
            'mesh_terms': json.dumps(mesh_terms),
            'pubmed_url': pubmed_url,
            'doi_url': doi_url,
            'geo_accessions': geo_accessions
        }

    except Exception as e:
        print(f"Error parsing article: {e}")
        return None

def filter_papers_with_geo(papers):
    """Filter papers to only include those with valid GEO datasets."""
    if not papers:
        return []

    print(f"\nValidating GEO datasets for {len(papers)} papers...")
    print("=" * 80)

    filtered_papers = []
    no_geo_count = 0
    invalid_geo_count = 0

    for i, paper in enumerate(papers, 1):
        pmid = paper.get('pmid', 'unknown')
        title = paper.get('title', '')[:60]
        geo_accessions = paper.get('geo_accessions', [])

        print(f"\n[{i}/{len(papers)}] Paper {pmid}: {title}...")

        if not geo_accessions:
            print(f"  ❌ No GEO accessions found")
            no_geo_count += 1
            continue

        print(f"  Found GEO accessions: {', '.join(geo_accessions)}")
        print(f"  Validating...")

        # Validate GEO accessions
        valid_accessions = validate_geo_accessions(geo_accessions)

        if valid_accessions:
            print(f"  ✓ Valid GEO datasets: {', '.join(valid_accessions)}")
            paper['validated_geo_accessions'] = valid_accessions
            filtered_papers.append(paper)
        else:
            print(f"  ❌ No valid GEO datasets found")
            invalid_geo_count += 1

    print("\n" + "=" * 80)
    print(f"GEO Filtering Summary:")
    print(f"  Total papers: {len(papers)}")
    print(f"  Papers without GEO accessions: {no_geo_count}")
    print(f"  Papers with invalid GEO accessions: {invalid_geo_count}")
    print(f"  Papers with valid GEO datasets: {len(filtered_papers)}")
    print("=" * 80)

    return filtered_papers

def store_papers(papers):
    """Store papers in database."""
    if not papers:
        print("No papers to store")
        return

    # Filter out papers before the start date
    date_range = context['search_criteria']['date_range']
    start_year = int(date_range['start'].split('-')[0])

    filtered_papers = []
    excluded = 0
    for paper in papers:
        pub_date = paper.get('publication_date', '')
        if pub_date:
            # Extract year from publication date
            year_str = pub_date.split('-')[0] if '-' in pub_date else pub_date
            try:
                year = int(year_str)
                if year >= start_year:
                    filtered_papers.append(paper)
                else:
                    excluded += 1
                    print(f"Excluding paper {paper['pmid']} (published {pub_date}, before {start_year})")
            except ValueError:
                # If year can't be parsed, include the paper
                filtered_papers.append(paper)
        else:
            # If no date, include the paper
            filtered_papers.append(paper)

    if excluded > 0:
        print(f"Excluded {excluded} papers published before {start_year}")

    papers = filtered_papers

    if not papers:
        print("No papers to store after date filtering")
        return

    print(f"Storing {len(papers)} papers in database...")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    inserted = 0
    updated = 0

    for paper in papers:
        try:
            # Check if paper already exists
            cursor.execute("SELECT id FROM papers WHERE pmid = ?", (paper['pmid'],))
            existing = cursor.fetchone()

            # Get validated GEO accessions as JSON
            geo_accessions = paper.get('validated_geo_accessions', [])
            geo_accessions_json = json.dumps(geo_accessions) if geo_accessions else None

            if existing:
                # Update existing
                cursor.execute("""
                    UPDATE papers SET
                        doi = ?, title = ?, abstract = ?, authors = ?,
                        journal = ?, publication_date = ?, keywords = ?, mesh_terms = ?,
                        pubmed_url = ?, doi_url = ?, geo_accessions = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE pmid = ?
                """, (
                    paper['doi'], paper['title'], paper['abstract'], paper['authors'],
                    paper['journal'], paper['publication_date'], paper['keywords'],
                    paper['mesh_terms'], paper['pubmed_url'], paper['doi_url'],
                    geo_accessions_json, paper['pmid']
                ))
                updated += 1
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO papers (
                        pmid, doi, title, abstract, authors, journal,
                        publication_date, keywords, mesh_terms, pubmed_url, doi_url, geo_accessions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper['pmid'], paper['doi'], paper['title'], paper['abstract'],
                    paper['authors'], paper['journal'], paper['publication_date'],
                    paper['keywords'], paper['mesh_terms'], paper['pubmed_url'], paper['doi_url'],
                    geo_accessions_json
                ))
                inserted += 1

        except Exception as e:
            print(f"Error storing paper {paper.get('pmid', 'unknown')}: {e}")

    conn.commit()
    conn.close()

    print(f"Database updated: {inserted} new papers, {updated} updated")
    return inserted, updated

def main():
    """Main execution."""
    print("=" * 80)
    print("PUBMED PAPER SEARCH (LOCAL)")
    print("=" * 80)
    print(f"Project: {context['project_name']}")
    print(f"Database: {db_path}")
    print()

    # Build query
    query = build_pubmed_query()

    # Show date range
    date_range = context['search_criteria']['date_range']
    print(f"Date range: {date_range['start']} to {date_range['end']}")
    print()

    # Search PubMed
    max_results = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    pmids, total_count = search_pubmed(query, max_results)

    if not pmids:
        print("No papers found or error occurred")
        return

    print()
    print(f"Total papers matching query: {total_count}")
    print(f"Fetching top {len(pmids)} papers")
    print()

    # Fetch details
    papers = fetch_paper_details(pmids)

    if not papers:
        print("Failed to fetch paper details")
        return

    print()

    # Filter papers with valid GEO datasets
    papers = filter_papers_with_geo(papers)

    if not papers:
        print("\n⚠️ No papers with valid GEO datasets found!")
        print("Consider running with more results or adjusting search criteria.")
        return

    print()

    # Display sample
    print("=" * 80)
    print("SAMPLE RESULTS (first 3 papers with valid GEO datasets)")
    print("=" * 80)
    for i, paper in enumerate(papers[:3], 1):
        print(f"\n{i}. [{paper['pmid']}] {paper['title']}")
        print(f"   Journal: {paper['journal']}")
        print(f"   Date: {paper['publication_date']}")
        print(f"   Authors: {paper['authors'][:100]}...")
        print(f"   GEO Accessions: {', '.join(paper.get('validated_geo_accessions', []))}")
        if paper['abstract']:
            print(f"   Abstract: {paper['abstract'][:200]}...")

    print()
    print("=" * 80)

    # Store in database
    inserted, updated = store_papers(papers)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total papers found: {total_count}")
    print(f"Papers fetched: {len(papers)}")
    print(f"Papers inserted: {inserted}")
    print(f"Papers updated: {updated}")
    print(f"Database: {db_path}")
    print()
    print(f"Note: All stored papers have validated GEO datasets available.")
    print("=" * 80)

if __name__ == "__main__":
    main()
