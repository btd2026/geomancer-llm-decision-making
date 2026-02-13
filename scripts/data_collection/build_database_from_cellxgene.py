#!/usr/bin/env python3
"""
Build database from CELLxGENE metadata.

This script processes the CELLxGENE full metadata CSV file and populates
the papers and datasets tables with collection and dataset information.
It fetches additional metadata from PubMed and PMC when available.

Features:
- Resume capability (checks existing records)
- Rate limiting with exponential backoff
- Progress tracking with tqdm
- Batch commits for efficiency
- Comprehensive error handling
- Dry-run mode for testing
- Configurable via YAML and CLI

Author: Claude Code
Created: 2025-11-04
"""

import argparse
import csv
import json
import logging
import logging.handlers
import sqlite3
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import yaml
from tqdm import tqdm


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Collection:
    """Represents a CELLxGENE collection (maps to a paper)."""
    collection_id: str
    collection_name: str
    doi: Optional[str]
    citation: str
    datasets: List[Dict[str, Any]]


@dataclass
class Paper:
    """Paper metadata from PubMed."""
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: Optional[str]
    authors: Optional[str]
    journal: Optional[str]
    publication_date: Optional[str]
    keywords: Optional[str]
    mesh_terms: Optional[str]


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (defaults to config/pipeline_config.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / 'config' / 'pipeline_config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger
    """
    log_config = config['logging']

    # Create logger
    logger = logging.getLogger('cellxgene_pipeline')
    logger.setLevel(getattr(logging, log_config['level']))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    formatter = logging.Formatter(
        log_config['format'],
        datefmt=log_config['date_format']
    )

    # Console handler
    if log_config['console']['enabled']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_config['console']['level']))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_config['file']['enabled']:
        log_path = Path(log_config['file']['path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=log_config['file']['max_bytes'],
            backupCount=log_config['file']['backup_count']
        )
        file_handler.setLevel(getattr(logging, log_config['level']))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# Database Functions
# ============================================================================

def get_db_connection(config: Dict[str, Any]) -> sqlite3.Connection:
    """
    Get database connection.

    Args:
        config: Configuration dictionary

    Returns:
        SQLite connection
    """
    script_dir = Path(__file__).parent
    db_path = script_dir.parent / config['database']['path']

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Please run migrate_schema_for_cellxgene.py first."
        )

    conn = sqlite3.connect(
        str(db_path),
        timeout=config['database']['timeout']
    )

    # Enable Write-Ahead Logging for better concurrency
    if config['performance']['use_wal_mode']:
        conn.execute("PRAGMA journal_mode=WAL")

    # Enable foreign key enforcement for data integrity
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def paper_exists(conn: sqlite3.Connection, doi: str) -> Optional[int]:
    """
    Check if a paper with given DOI already exists.

    Args:
        conn: Database connection
        doi: DOI to check

    Returns:
        Paper ID if exists, None otherwise
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM papers WHERE doi = ?", (doi,))
    row = cursor.fetchone()
    return row[0] if row else None


def dataset_exists(conn: sqlite3.Connection, dataset_id: str) -> bool:
    """
    Check if a dataset already exists.

    Args:
        conn: Database connection
        dataset_id: CELLxGENE dataset ID

    Returns:
        True if exists, False otherwise
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM datasets WHERE dataset_id = ?", (dataset_id,))
    return cursor.fetchone() is not None


def insert_paper(
    conn: sqlite3.Connection,
    collection: Collection,
    paper: Optional[Paper],
    config: Dict[str, Any],
    logger: logging.Logger
) -> int:
    """
    Insert or update a paper in the database.

    Args:
        conn: Database connection
        collection: Collection information
        paper: Paper metadata (None if not found in PubMed)
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Paper ID
    """
    cursor = conn.cursor()

    # Check if paper exists
    existing_id = paper_exists(conn, collection.doi) if collection.doi else None

    if existing_id:
        # Update existing paper
        logger.debug(f"Updating existing paper {existing_id} for DOI {collection.doi}")

        # Get existing collection_ids
        cursor.execute(
            "SELECT all_collection_ids, source FROM papers WHERE id = ?",
            (existing_id,)
        )
        row = cursor.fetchone()
        existing_collection_ids = json.loads(row[0]) if row[0] else []
        existing_source = row[1]

        # Add new collection_id if not already present
        if collection.collection_id not in existing_collection_ids:
            existing_collection_ids.append(collection.collection_id)

        # Update source field
        new_source = existing_source
        if config['processing']['existing_papers']['update_source']:
            if existing_source == 'pubmed_search':
                new_source = 'both'
            elif existing_source is None:
                new_source = 'cellxgene'

        # Update paper
        cursor.execute("""
            UPDATE papers SET
                collection_name = ?,
                all_collection_ids = ?,
                source = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            collection.collection_name,
            json.dumps(existing_collection_ids),
            new_source,
            existing_id
        ))

        return existing_id

    else:
        # Insert new paper
        logger.debug(f"Inserting new paper for DOI {collection.doi}")

        # Prepare collection IDs
        all_collection_ids = json.dumps([collection.collection_id])

        if paper:
            # Have PubMed data
            cursor.execute("""
                INSERT INTO papers (
                    pmid, doi, title, abstract, authors, journal, publication_date,
                    keywords, mesh_terms, collection_id, all_collection_ids,
                    collection_name, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.pmid, paper.doi, paper.title, paper.abstract,
                paper.authors, paper.journal, paper.publication_date,
                paper.keywords, paper.mesh_terms,
                collection.collection_id, all_collection_ids,
                collection.collection_name, 'cellxgene'
            ))
        else:
            # No PubMed data, use collection info
            cursor.execute("""
                INSERT INTO papers (
                    doi, title, collection_id, all_collection_ids,
                    collection_name, source
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                collection.doi, collection.collection_name,
                collection.collection_id, all_collection_ids,
                collection.collection_name, 'cellxgene'
            ))

        return cursor.lastrowid


def insert_dataset(
    conn: sqlite3.Connection,
    paper_id: int,
    dataset: Dict[str, Any],
    collection_id: str,
    logger: logging.Logger
) -> None:
    """
    Insert a dataset into the database.

    Args:
        conn: Database connection
        paper_id: ID of associated paper
        dataset: Dataset information
        collection_id: Collection ID
        logger: Logger instance
    """
    cursor = conn.cursor()

    logger.debug(f"Inserting dataset {dataset['dataset_id']}")

    cursor.execute("""
        INSERT INTO datasets (
            paper_id, dataset_id, collection_id, dataset_title,
            dataset_version_id, dataset_h5ad_path, citation,
            n_cells
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        paper_id,
        dataset['dataset_id'],
        collection_id,
        dataset['dataset_title'],
        dataset['dataset_version_id'],
        dataset['dataset_h5ad_path'],
        dataset['citation'],
        dataset.get('dataset_total_cell_count')
    ))


# ============================================================================
# CSV Processing
# ============================================================================

def parse_cellxgene_csv(
    csv_path: Path,
    logger: logging.Logger,
    limit: Optional[int] = None
) -> Dict[str, Collection]:
    """
    Parse CELLxGENE metadata CSV and group by collection.

    Args:
        csv_path: Path to CSV file
        logger: Logger instance
        limit: Optional limit on number of rows to process

    Returns:
        Dictionary mapping collection_id to Collection objects
    """
    logger.info(f"Parsing CSV file: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    collections = {}
    rows_processed = 0

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if limit and rows_processed >= limit:
                break

            collection_id = row['collection_id']

            # Create or get collection
            if collection_id not in collections:
                collections[collection_id] = Collection(
                    collection_id=collection_id,
                    collection_name=row['collection_name'],
                    doi=row['collection_doi'] if row['collection_doi'] else None,
                    citation=row['citation'],
                    datasets=[]
                )

            # Add dataset to collection
            collections[collection_id].datasets.append({
                'dataset_id': row['dataset_id'],
                'dataset_version_id': row['dataset_version_id'],
                'dataset_title': row['dataset_title'],
                'dataset_h5ad_path': row['dataset_h5ad_path'],
                'citation': row['citation'],
                'dataset_total_cell_count': int(row['dataset_total_cell_count'])
                    if row.get('dataset_total_cell_count') else None
            })

            rows_processed += 1

    logger.info(
        f"Parsed {rows_processed} datasets into "
        f"{len(collections)} unique collections"
    )

    return collections


# ============================================================================
# API Functions
# ============================================================================

class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests_per_second: float):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_second: Maximum requests per second
        """
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        time_since_last = now - self.last_request_time

        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)

        self.last_request_time = time.time()


def doi_to_pmid(
    doi: str,
    config: Dict[str, Any],
    rate_limiter: RateLimiter,
    logger: logging.Logger
) -> Optional[str]:
    """
    Convert DOI to PMID using NCBI E-utilities.

    Args:
        doi: DOI to convert
        config: Configuration dictionary
        rate_limiter: Rate limiter instance
        logger: Logger instance

    Returns:
        PMID if found, None otherwise
    """
    if not doi:
        return None

    logger.debug(f"Converting DOI to PMID: {doi}")

    ncbi_config = config['api']['ncbi']
    retry_config = config['rate_limits']

    params = {
        'db': 'pubmed',
        'term': f'{doi}[DOI]',
        'retmode': 'xml',
        'email': ncbi_config['email'],
        'tool': ncbi_config['tool']
    }

    if ncbi_config.get('api_key'):
        params['api_key'] = ncbi_config['api_key']

    url = ncbi_config['base_url'] + 'esearch.fcgi?' + urllib.parse.urlencode(params)

    for attempt in range(retry_config['max_retries']):
        try:
            rate_limiter.wait()

            with urllib.request.urlopen(url, timeout=retry_config['request_timeout']) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            id_list = root.find('.//IdList')

            if id_list is not None and len(id_list) > 0:
                pmid = id_list[0].text
                logger.debug(f"Found PMID {pmid} for DOI {doi}")
                return pmid

            logger.debug(f"No PMID found for DOI {doi}")
            return None

        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{retry_config['max_retries']} "
                f"failed for DOI {doi}: {e}"
            )

            if attempt < retry_config['max_retries'] - 1:
                delay = min(
                    retry_config['initial_retry_delay'] *
                    (retry_config['exponential_backoff_factor'] ** attempt),
                    retry_config['max_retry_delay']
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to convert DOI {doi} after all retries")
                return None


def fetch_pubmed_metadata(
    pmid: str,
    config: Dict[str, Any],
    rate_limiter: RateLimiter,
    logger: logging.Logger
) -> Optional[Paper]:
    """
    Fetch paper metadata from PubMed.

    Args:
        pmid: PubMed ID
        config: Configuration dictionary
        rate_limiter: Rate limiter instance
        logger: Logger instance

    Returns:
        Paper object if found, None otherwise
    """
    logger.debug(f"Fetching PubMed metadata for PMID {pmid}")

    ncbi_config = config['api']['ncbi']
    retry_config = config['rate_limits']

    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml',
        'email': ncbi_config['email'],
        'tool': ncbi_config['tool']
    }

    if ncbi_config.get('api_key'):
        params['api_key'] = ncbi_config['api_key']

    url = ncbi_config['base_url'] + 'efetch.fcgi?' + urllib.parse.urlencode(params)

    for attempt in range(retry_config['max_retries']):
        try:
            rate_limiter.wait()

            with urllib.request.urlopen(url, timeout=retry_config['request_timeout']) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            article = root.find('.//PubmedArticle')

            if article is None:
                logger.warning(f"No article found for PMID {pmid}")
                return None

            return parse_pubmed_article(
                article,
                logger,
                extract_keywords=config['features']['extract_keywords']
            )

        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{retry_config['max_retries']} "
                f"failed for PMID {pmid}: {e}"
            )

            if attempt < retry_config['max_retries'] - 1:
                delay = min(
                    retry_config['initial_retry_delay'] *
                    (retry_config['exponential_backoff_factor'] ** attempt),
                    retry_config['max_retry_delay']
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to fetch PMID {pmid} after all retries")
                return None


def parse_pubmed_article(
    article_elem: ET.Element,
    logger: logging.Logger,
    extract_keywords: bool = True
) -> Optional[Paper]:
    """
    Parse PubMed article XML element.

    Args:
        article_elem: XML element containing article data
        logger: Logger instance
        extract_keywords: Whether to extract keywords and MeSH terms

    Returns:
        Paper object
    """
    try:
        # PMID
        pmid_elem = article_elem.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else None

        # Article
        article = article_elem.find('.//Article')
        if article is None:
            return None

        # Title
        title_elem = article.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else 'No title'

        # Abstract
        abstract_parts = article.findall('.//AbstractText')
        abstract = ' '.join([p.text for p in abstract_parts if p.text]) if abstract_parts else None

        # Authors
        author_list = article.findall('.//Author')
        authors = []
        for author in author_list[:10]:
            last = author.find('.//LastName')
            first = author.find('.//ForeName')
            if last is not None:
                name = last.text
                if first is not None:
                    name = f"{first.text} {name}"
                authors.append(name)
        authors_str = ', '.join(authors) if authors else None

        # Journal
        journal_elem = article.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else None

        # Publication date
        pub_date = article.find('.//PubDate')
        date_str = None
        if pub_date is not None:
            pub_year = pub_date.find('.//Year')
            pub_month = pub_date.find('.//Month')
            pub_day = pub_date.find('.//Day')

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
        if extract_keywords:
            mesh_headings = article_elem.findall('.//MeshHeading/DescriptorName')
            mesh_terms = [m.text for m in mesh_headings if m.text]

        # Keywords
        keywords = []
        if extract_keywords:
            kw_list = article.findall('.//Keyword')
            keywords = [kw.text for kw in kw_list if kw.text]

        return Paper(
            pmid=pmid,
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors_str,
            journal=journal,
            publication_date=date_str,
            keywords=json.dumps(keywords) if keywords else None,
            mesh_terms=json.dumps(mesh_terms) if mesh_terms else None
        )

    except Exception as e:
        logger.error(f"Error parsing PubMed article: {e}")
        return None


# ============================================================================
# Main Processing
# ============================================================================

def process_collections(
    collections: Dict[str, Collection],
    config: Dict[str, Any],
    logger: logging.Logger,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Process collections and populate database.

    Args:
        collections: Dictionary of Collection objects
        config: Configuration dictionary
        logger: Logger instance
        dry_run: If True, don't modify database

    Returns:
        Statistics dictionary
    """
    logger.info(f"Processing {len(collections)} collections")

    # Initialize statistics
    stats = {
        'collections_processed': 0,
        'papers_inserted': 0,
        'papers_updated': 0,
        'papers_skipped': 0,
        'datasets_inserted': 0,
        'datasets_skipped': 0,
        'errors': 0,
        'api_calls': 0
    }

    # Get database connection
    if not dry_run:
        conn = get_db_connection(config)
    else:
        conn = None
        logger.info("DRY RUN MODE - No database modifications will be made")

    # Initialize rate limiter
    ncbi_config = config['api']['ncbi']
    has_api_key = ncbi_config.get('api_key') is not None
    max_rate = (
        config['rate_limits']['ncbi_with_api_key'] if has_api_key
        else config['rate_limits']['ncbi_without_api_key']
    )
    max_rate *= config['rate_limits']['rate_limit_buffer']
    rate_limiter = RateLimiter(max_rate)

    # Process collections with progress bar
    collection_items = list(collections.items())
    progress_bar = tqdm(
        collection_items,
        desc="Processing collections",
        disable=not config['progress']['show_progress_bars']
    )

    batch_counter = 0
    batch_size = config['database']['batch_commit_size']

    for collection_id, collection in progress_bar:
        try:
            stats['collections_processed'] += 1

            # Skip if no DOI and required
            if not collection.doi and config['processing']['missing_doi']['skip_if_not_found']:
                logger.debug(f"Skipping collection {collection_id} - no DOI")
                stats['papers_skipped'] += 1
                continue

            # Check if paper already exists
            if not dry_run:
                existing_paper_id = paper_exists(conn, collection.doi) if collection.doi else None
            else:
                existing_paper_id = None

            # Fetch PubMed metadata if DOI available
            paper = None
            if collection.doi:
                # Convert DOI to PMID
                pmid = doi_to_pmid(collection.doi, config, rate_limiter, logger)
                stats['api_calls'] += 1

                if pmid:
                    # Fetch metadata
                    paper = fetch_pubmed_metadata(pmid, config, rate_limiter, logger)
                    stats['api_calls'] += 1

            # Insert or update paper
            if not dry_run:
                paper_id = insert_paper(conn, collection, paper, config, logger)

                if existing_paper_id:
                    stats['papers_updated'] += 1
                else:
                    stats['papers_inserted'] += 1
            else:
                paper_id = 1  # Dummy ID for dry run
                stats['papers_inserted'] += 1

            # Insert datasets
            for dataset in collection.datasets:
                # Skip if dataset already exists
                if not dry_run and dataset_exists(conn, dataset['dataset_id']):
                    logger.debug(f"Dataset {dataset['dataset_id']} already exists, skipping")
                    stats['datasets_skipped'] += 1
                    continue

                if not dry_run:
                    insert_dataset(conn, paper_id, dataset, collection_id, logger)

                stats['datasets_inserted'] += 1

            # Batch commit
            batch_counter += 1
            if not dry_run and batch_counter >= batch_size:
                conn.commit()
                logger.debug(f"Batch commit after {batch_counter} collections")
                batch_counter = 0

            # Update progress bar
            progress_bar.set_postfix({
                'papers': stats['papers_inserted'] + stats['papers_updated'],
                'datasets': stats['datasets_inserted'],
                'errors': stats['errors']
            })

        except Exception as e:
            logger.error(f"Error processing collection {collection_id}: {e}")
            stats['errors'] += 1

            if stats['errors'] >= config['error_handling']['max_errors']:
                logger.critical("Maximum error count reached, stopping")
                break

            if not config['error_handling']['continue_on_error']:
                raise

    # Final commit
    if not dry_run and batch_counter > 0:
        conn.commit()
        logger.info("Final batch committed")

    # Close connection
    if not dry_run:
        conn.close()

    return stats


def print_statistics(stats: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Print processing statistics.

    Args:
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    if not config['output']['print_summary']:
        return

    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    print(f"\nCollections processed: {stats['collections_processed']}")

    print(f"\nPapers:")
    print(f"  - Inserted: {stats['papers_inserted']}")
    print(f"  - Updated: {stats['papers_updated']}")
    print(f"  - Skipped: {stats['papers_skipped']}")
    print(f"  - Total: {stats['papers_inserted'] + stats['papers_updated']}")

    print(f"\nDatasets:")
    print(f"  - Inserted: {stats['datasets_inserted']}")
    print(f"  - Skipped: {stats['datasets_skipped']}")
    print(f"  - Total: {stats['datasets_inserted']}")

    print(f"\nAPI Calls: {stats['api_calls']}")
    print(f"Errors: {stats['errors']}")

    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Build database from CELLxGENE metadata CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal run
  python build_database_from_cellxgene.py

  # Dry run with limit
  python build_database_from_cellxgene.py --dry-run --limit 10

  # Custom config and CSV
  python build_database_from_cellxgene.py --config custom_config.yaml --csv data.csv

  # Verbose mode
  python build_database_from_cellxgene.py --verbose --limit 5
        """
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration YAML file (default: config/pipeline_config.yaml)'
    )

    parser.add_argument(
        '--csv',
        type=Path,
        help='Path to CELLxGENE metadata CSV (default: cellxgene_full_metadata.csv)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without modifying database (for testing)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of CSV rows to process (for testing)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug output'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main execution.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Override config with CLI arguments
    if args.dry_run:
        config['testing']['dry_run'] = True
    if args.limit:
        config['testing']['limit'] = args.limit
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
        config['logging']['console']['level'] = 'DEBUG'
    if args.no_progress:
        config['progress']['show_progress_bars'] = False

    # Add logging.handlers module for RotatingFileHandler
    import logging.handlers

    # Set up logging
    logger = setup_logging(config)

    # Print header
    print("=" * 80)
    print("CELLxGENE DATABASE BUILDER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if config['testing']['dry_run']:
        print("\n*** DRY RUN MODE - No database changes will be made ***")

    try:
        # Determine CSV path
        if args.csv:
            csv_path = args.csv
        else:
            script_dir = Path(__file__).parent
            csv_path = script_dir.parent / config['input']['cellxgene_metadata']

        logger.info(f"CSV file: {csv_path}")

        # Parse CSV
        collections = parse_cellxgene_csv(
            csv_path,
            logger,
            limit=config['testing'].get('limit')
        )

        # Process collections
        stats = process_collections(
            collections,
            config,
            logger,
            dry_run=config['testing']['dry_run']
        )

        # Print statistics
        print_statistics(stats, config)

        # Check for errors
        if stats['errors'] > 0:
            logger.warning(f"Completed with {stats['errors']} errors")
            return 1

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
