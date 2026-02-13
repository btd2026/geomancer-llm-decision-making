#!/usr/bin/env python3
"""
Generate LLM-based descriptions for papers and datasets.

This script uses the Claude API (Haiku model for cost efficiency) to generate
concise, informative descriptions for papers and datasets in the database.
It includes cost tracking, resume capability, and robust error handling.

Features:
- Generates 2-3 sentence descriptions for papers and datasets
- Uses Claude Haiku (claude-3-haiku-20240307) for cost efficiency
- Tracks costs: input/output tokens and estimated USD
- Resume capability: skips records with existing descriptions
- Rate limiting and exponential backoff
- Progress tracking with tqdm
- Checkpoint saves every 100 descriptions
- Transaction safety with rollback on errors
- Comprehensive logging

Author: Claude Code
Created: 2025-11-04
"""

import argparse
import json
import logging
import logging.handlers
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm


# ============================================================================
# Configuration and Constants
# ============================================================================

# Claude API Configuration
CLAUDE_API_ENDPOINT = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-haiku-20240307"
CLAUDE_API_VERSION = "2023-06-01"

# Cost tracking (as of Nov 2024, per million tokens)
# Haiku pricing: $0.25 per MTok input, $1.25 per MTok output
HAIKU_INPUT_COST_PER_MTOK = 0.25
HAIKU_OUTPUT_COST_PER_MTOK = 1.25

# Rate limiting
DEFAULT_REQUESTS_PER_MINUTE = 50
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds
DEFAULT_TIMEOUT = 30  # seconds

# Checkpoint frequency
CHECKPOINT_FREQUENCY = 100  # Save progress every N descriptions


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CostTracker:
    """Tracks token usage and costs for Claude API calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    failed_calls: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage from an API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.api_calls += 1

    def record_failure(self) -> None:
        """Record a failed API call."""
        self.failed_calls += 1

    @property
    def total_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = (self.input_tokens / 1_000_000) * HAIKU_INPUT_COST_PER_MTOK
        output_cost = (self.output_tokens / 1_000_000) * HAIKU_OUTPUT_COST_PER_MTOK
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.input_tokens + self.output_tokens,
            'api_calls': self.api_calls,
            'failed_calls': self.failed_calls,
            'total_cost_usd': round(self.total_cost, 4),
            'avg_tokens_per_call': (
                (self.input_tokens + self.output_tokens) // self.api_calls
                if self.api_calls > 0 else 0
            )
        }


@dataclass
class PaperRecord:
    """Represents a paper from the database."""
    paper_id: int
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: Optional[str]
    journal: Optional[str]
    publication_date: Optional[str]
    dataset_description: Optional[str]


@dataclass
class DatasetRecord:
    """Represents a dataset from the database."""
    dataset_id: int
    paper_id: int
    dataset_name: Optional[str]
    organism: Optional[str]
    tissue_type: Optional[str]
    condition: Optional[str]
    n_cells: Optional[int]
    paper_title: str
    dataset_description: Optional[str]


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging with both file and console handlers.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('generate_llm_descriptions')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler with rotation
    log_file = log_dir / f"generate_descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# Database Operations
# ============================================================================

def get_database_connection(db_path: Path) -> sqlite3.Connection:
    """
    Create database connection with appropriate settings.

    Args:
        db_path: Path to SQLite database

    Returns:
        Database connection

    Raises:
        sqlite3.Error: If connection fails
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key enforcement
    return conn


def fetch_papers_without_descriptions(
    conn: sqlite3.Connection,
    limit: Optional[int] = None
) -> List[PaperRecord]:
    """
    Fetch papers that don't have LLM-generated descriptions.

    Args:
        conn: Database connection
        limit: Maximum number of records to fetch (None for all)

    Returns:
        List of PaperRecord objects
    """
    query = """
        SELECT
            id, pmid, doi, title, abstract, journal,
            publication_date, dataset_description
        FROM papers
        WHERE (dataset_description IS NULL OR dataset_description = '')
        AND title IS NOT NULL
        ORDER BY id
    """

    if limit is not None:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)

    papers = []
    for row in cursor:
        papers.append(PaperRecord(
            paper_id=row['id'],
            pmid=row['pmid'],
            doi=row['doi'],
            title=row['title'],
            abstract=row['abstract'],
            journal=row['journal'],
            publication_date=row['publication_date'],
            dataset_description=row['dataset_description']
        ))

    return papers


def fetch_datasets_without_descriptions(
    conn: sqlite3.Connection,
    limit: Optional[int] = None
) -> List[DatasetRecord]:
    """
    Fetch datasets that don't have LLM-generated descriptions.

    Args:
        conn: Database connection
        limit: Maximum number of records to fetch (None for all)

    Returns:
        List of DatasetRecord objects
    """
    query = """
        SELECT
            d.id, d.paper_id,
            d.dataset_title as dataset_name,
            d.organism,
            NULL as tissue_type,
            NULL as condition,
            d.n_cells,
            p.title as paper_title,
            d.llm_description as dataset_description
        FROM datasets d
        JOIN papers p ON d.paper_id = p.id
        WHERE (d.llm_description IS NULL OR d.llm_description = '')
        AND p.title IS NOT NULL
        AND d.dataset_id IS NOT NULL
        ORDER BY d.id
    """

    if limit is not None:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)

    datasets = []
    for row in cursor:
        datasets.append(DatasetRecord(
            dataset_id=row['id'],
            paper_id=row['paper_id'],
            dataset_name=row['dataset_name'],
            organism=row['organism'],
            tissue_type=row['tissue_type'],
            condition=row['condition'],
            n_cells=row['n_cells'],
            paper_title=row['paper_title'],
            dataset_description=row['dataset_description']
        ))

    return datasets


def update_paper_description(
    conn: sqlite3.Connection,
    paper_id: int,
    description: str
) -> None:
    """
    Update paper description in database.

    Args:
        conn: Database connection
        paper_id: Paper ID
        description: Generated description

    Raises:
        sqlite3.Error: If update fails
    """
    conn.execute(
        """
        UPDATE papers
        SET llm_description = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (description, paper_id)
    )


# ============================================================================
# Claude API Integration
# ============================================================================

class ClaudeAPIError(Exception):
    """Custom exception for Claude API errors."""
    pass


class ClaudeRateLimitError(ClaudeAPIError):
    """Raised when rate limit is exceeded."""
    pass


class ClaudeClient:
    """Client for interacting with Claude API."""

    def __init__(
        self,
        api_key: str,
        logger: logging.Logger,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize Claude API client.

        Args:
            api_key: Anthropic API key
            logger: Logger instance
            requests_per_minute: Rate limit
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.logger = logger
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0.0
        self.min_interval = 60.0 / requests_per_minute

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)

    def _make_request(
        self,
        prompt: str,
        max_tokens: int = 300
    ) -> Tuple[str, int, int]:
        """
        Make a request to Claude API with retry logic.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            ClaudeAPIError: If request fails after retries
            ClaudeRateLimitError: If rate limit is exceeded
        """
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": CLAUDE_API_VERSION,
            "content-type": "application/json"
        }

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()

                request = Request(
                    CLAUDE_API_ENDPOINT,
                    data=json.dumps(payload).encode('utf-8'),
                    headers=headers
                )

                self.last_request_time = time.time()

                with urlopen(request, timeout=self.timeout) as response:
                    response_data = json.loads(response.read().decode('utf-8'))

                # Validate response structure
                if 'content' not in response_data:
                    raise ClaudeAPIError("Invalid response structure: missing 'content'")

                if not response_data['content'] or len(response_data['content']) == 0:
                    raise ClaudeAPIError("Empty content in response")

                text = response_data['content'][0]['text']

                # Extract token usage
                usage = response_data.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)

                return text.strip(), input_tokens, output_tokens

            except HTTPError as e:
                if e.code == 429:  # Rate limit
                    retry_delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {retry_delay:.1f}s"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise ClaudeRateLimitError("Rate limit exceeded after retries")

                elif e.code >= 500:  # Server error
                    retry_delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(
                        f"Server error {e.code} (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {retry_delay:.1f}s"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_delay)
                        continue

                # Client error (4xx) - don't retry
                error_body = e.read().decode('utf-8') if e.fp else "No error details"
                raise ClaudeAPIError(f"HTTP {e.code}: {error_body}")

            except URLError as e:
                retry_delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                self.logger.warning(
                    f"Network error (attempt {attempt + 1}/{self.max_retries}): {e.reason}. "
                    f"Waiting {retry_delay:.1f}s"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ClaudeAPIError(f"Network error after retries: {e.reason}")

            except json.JSONDecodeError as e:
                raise ClaudeAPIError(f"Invalid JSON response: {e}")

            except Exception as e:
                raise ClaudeAPIError(f"Unexpected error: {type(e).__name__}: {e}")

        raise ClaudeAPIError(f"Failed after {self.max_retries} attempts")

    def generate_paper_description(self, paper: PaperRecord) -> Tuple[str, int, int]:
        """
        Generate description for a paper.

        Args:
            paper: PaperRecord object

        Returns:
            Tuple of (description, input_tokens, output_tokens)
        """
        # Build context
        abstract = paper.abstract or "No abstract available."
        journal_info = ""
        if paper.journal:
            journal_info = f"Journal: {paper.journal}"
            if paper.publication_date:
                journal_info += f" ({paper.publication_date[:4]})"  # Year only

        prompt = f"""Summarize this single-cell RNA-seq paper in 2-3 sentences:

Title: {paper.title}
Abstract: {abstract}
{journal_info}

Focus on: research question, methodology, key findings, biological significance.
Keep it clear and accessible. Write in third person (e.g., "This paper..."). Do NOT include phrases like "In summary" or "This paper describes"."""

        description, input_tokens, output_tokens = self._make_request(prompt)

        # Post-process: remove common prefixes
        prefixes_to_remove = [
            "This paper ",
            "The paper ",
            "This study ",
            "The study ",
            "In summary, ",
            "Summary: "
        ]

        for prefix in prefixes_to_remove:
            if description.startswith(prefix):
                description = description[len(prefix):]
                description = description[0].upper() + description[1:]
                break

        return description, input_tokens, output_tokens

    def generate_dataset_description(
        self,
        dataset: DatasetRecord
    ) -> Tuple[str, int, int]:
        """
        Generate description for a dataset.

        Args:
            dataset: DatasetRecord object

        Returns:
            Tuple of (description, input_tokens, output_tokens)
        """
        # Build context
        dataset_title = dataset.dataset_name or "Unnamed dataset"
        organism_info = dataset.organism or "Unknown organism"
        tissue_info = dataset.tissue_type or "Unknown tissue"
        cell_count = f"{dataset.n_cells:,}" if dataset.n_cells else "Unknown number of"
        condition_info = f" under {dataset.condition} conditions" if dataset.condition else ""

        prompt = f"""Describe this single-cell dataset in 2-3 sentences:

Dataset: {dataset_title}
From paper: {dataset.paper_title}
Organism: {organism_info}
Tissue: {tissue_info}
Cells: {cell_count}
{f"Condition: {dataset.condition}" if dataset.condition else ""}

Focus on: biological context, what was studied, experimental design.
Keep it clear and accessible. Do NOT include phrases like "This dataset" or "In summary"."""

        description, input_tokens, output_tokens = self._make_request(prompt)

        # Post-process: remove common prefixes
        prefixes_to_remove = [
            "This dataset ",
            "The dataset ",
            "In summary, ",
            "Summary: "
        ]

        for prefix in prefixes_to_remove:
            if description.startswith(prefix):
                description = description[len(prefix):]
                description = description[0].upper() + description[1:]
                break

        return description, input_tokens, output_tokens


# ============================================================================
# Main Processing
# ============================================================================

def process_papers(
    conn: sqlite3.Connection,
    client: ClaudeClient,
    cost_tracker: CostTracker,
    logger: logging.Logger,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> int:
    """
    Process papers and generate descriptions.

    Args:
        conn: Database connection
        client: Claude API client
        cost_tracker: Cost tracking object
        logger: Logger instance
        dry_run: If True, don't update database
        limit: Maximum number of papers to process

    Returns:
        Number of papers processed
    """
    papers = fetch_papers_without_descriptions(conn, limit)

    if not papers:
        logger.info("No papers found needing descriptions")
        return 0

    logger.info(f"Found {len(papers)} papers needing descriptions")

    processed = 0

    with tqdm(total=len(papers), desc="Generating paper descriptions") as pbar:
        for i, paper in enumerate(papers):
            try:
                description, input_tokens, output_tokens = (
                    client.generate_paper_description(paper)
                )

                cost_tracker.add_usage(input_tokens, output_tokens)

                if not dry_run:
                    update_paper_description(conn, paper.paper_id, description)

                    # Checkpoint every N records
                    if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                        conn.commit()
                        logger.debug(f"Checkpoint: committed {i + 1} papers")

                processed += 1

                logger.debug(
                    f"Paper {paper.paper_id}: {input_tokens + output_tokens} tokens, "
                    f"description length: {len(description)} chars"
                )

            except ClaudeRateLimitError as e:
                logger.error(f"Rate limit error for paper {paper.paper_id}: {e}")
                cost_tracker.record_failure()
                # Stop processing to avoid further rate limit issues
                logger.warning("Stopping due to rate limit")
                break

            except ClaudeAPIError as e:
                logger.error(f"API error for paper {paper.paper_id}: {e}")
                cost_tracker.record_failure()
                # Continue with next paper

            except sqlite3.Error as e:
                logger.error(f"Database error for paper {paper.paper_id}: {e}")
                conn.rollback()
                # Continue with next paper

            except Exception as e:
                logger.error(
                    f"Unexpected error for paper {paper.paper_id}: "
                    f"{type(e).__name__}: {e}"
                )
                cost_tracker.record_failure()
                # Continue with next paper

            finally:
                pbar.update(1)

    # Final commit
    if not dry_run and processed > 0:
        conn.commit()
        logger.info(f"Final commit: {processed} paper descriptions saved")

    return processed


def update_dataset_description(
    conn: sqlite3.Connection,
    dataset_id: int,
    description: str
) -> None:
    """
    Update dataset description in database.

    Args:
        conn: Database connection
        dataset_id: Dataset ID
        description: Generated description

    Raises:
        sqlite3.Error: If update fails
    """
    conn.execute(
        """
        UPDATE datasets
        SET llm_description = ?
        WHERE id = ?
        """,
        (description, dataset_id)
    )


def process_datasets(
    conn: sqlite3.Connection,
    client: ClaudeClient,
    cost_tracker: CostTracker,
    logger: logging.Logger,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> int:
    """
    Process datasets and generate descriptions.

    Args:
        conn: Database connection
        client: Claude API client
        cost_tracker: Cost tracking object
        logger: Logger instance
        dry_run: If True, don't update database
        limit: Maximum number of datasets to process

    Returns:
        Number of datasets processed
    """
    datasets = fetch_datasets_without_descriptions(conn, limit)

    if not datasets:
        logger.info("No datasets found needing descriptions")
        return 0

    logger.info(f"Found {len(datasets)} datasets needing descriptions")

    processed = 0

    with tqdm(total=len(datasets), desc="Generating dataset descriptions") as pbar:
        for i, dataset in enumerate(datasets):
            try:
                description, input_tokens, output_tokens = (
                    client.generate_dataset_description(dataset)
                )

                cost_tracker.add_usage(input_tokens, output_tokens)

                if not dry_run:
                    update_dataset_description(conn, dataset.dataset_id, description)

                    # Checkpoint every N records
                    if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                        conn.commit()
                        logger.debug(f"Checkpoint: committed {i + 1} datasets")

                processed += 1

                logger.debug(
                    f"Dataset {dataset.dataset_id}: {input_tokens + output_tokens} tokens, "
                    f"description length: {len(description)} chars"
                )

            except ClaudeRateLimitError as e:
                logger.error(f"Rate limit error for dataset {dataset.dataset_id}: {e}")
                cost_tracker.record_failure()
                logger.warning("Stopping due to rate limit")
                break

            except ClaudeAPIError as e:
                logger.error(f"API error for dataset {dataset.dataset_id}: {e}")
                cost_tracker.record_failure()

            except sqlite3.Error as e:
                logger.error(f"Database error for dataset {dataset.dataset_id}: {e}")
                conn.rollback()

            except Exception as e:
                logger.error(
                    f"Unexpected error for dataset {dataset.dataset_id}: "
                    f"{type(e).__name__}: {e}"
                )
                cost_tracker.record_failure()

            finally:
                pbar.update(1)

    # Final commit
    if not dry_run and processed > 0:
        conn.commit()
        logger.info(f"Final commit: {processed} dataset descriptions saved")

    return processed


def generate_descriptions(
    db_path: Path,
    api_key: str,
    logger: logging.Logger,
    target: str = "papers",
    dry_run: bool = False,
    limit: Optional[int] = None,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
) -> CostTracker:
    """
    Main function to generate descriptions.

    Args:
        db_path: Path to database
        api_key: Claude API key
        logger: Logger instance
        target: What to process ("papers" or "datasets")
        dry_run: If True, don't update database
        limit: Maximum number of records to process
        requests_per_minute: API rate limit

    Returns:
        CostTracker with usage statistics
    """
    conn = get_database_connection(db_path)
    client = ClaudeClient(api_key, logger, requests_per_minute=requests_per_minute)
    cost_tracker = CostTracker()

    try:
        if target == "papers":
            processed = process_papers(
                conn, client, cost_tracker, logger, dry_run, limit
            )
            logger.info(f"Processed {processed} papers")

        elif target == "datasets":
            processed = process_datasets(
                conn, client, cost_tracker, logger, dry_run, limit
            )
            logger.info(f"Processed {processed} datasets")

        else:
            raise ValueError(f"Invalid target: {target}")

    finally:
        conn.close()

    return cost_tracker


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate LLM descriptions for papers and datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate descriptions for all papers
  %(prog)s --db papers.db --api-key $ANTHROPIC_API_KEY

  # Dry run (no database updates)
  %(prog)s --db papers.db --api-key $ANTHROPIC_API_KEY --dry-run

  # Process only 10 papers (for testing)
  %(prog)s --db papers.db --api-key $ANTHROPIC_API_KEY --limit 10

  # Use API key from environment variable
  export ANTHROPIC_API_KEY=sk-ant-...
  %(prog)s --db papers.db

  # Custom rate limit and log level
  %(prog)s --db papers.db --rate-limit 30 --log-level DEBUG
        """
    )

    parser.add_argument(
        '--db',
        type=Path,
        default=Path('papers.db'),
        help='Path to SQLite database (default: papers.db)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )

    parser.add_argument(
        '--target',
        type=str,
        choices=['papers', 'datasets'],
        default='papers',
        help='What to generate descriptions for (default: papers)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of records to process (for testing)'
    )

    parser.add_argument(
        '--rate-limit',
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help=f'API requests per minute (default: {DEFAULT_REQUESTS_PER_MINUTE})'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without updating database'
    )

    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Directory for log files (default: logs/)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir, args.log_level)

    logger.info("=" * 70)
    logger.info("LLM Description Generator")
    logger.info("=" * 70)

    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error(
            "No API key provided. Use --api-key or set ANTHROPIC_API_KEY environment variable"
        )
        return 1

    # Validate database
    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    # Log configuration
    logger.info(f"Database: {args.db}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Model: {CLAUDE_MODEL}")
    logger.info(f"Rate limit: {args.rate_limit} requests/minute")
    if args.limit:
        logger.info(f"Limit: {args.limit} records")
    if args.dry_run:
        logger.warning("DRY RUN MODE - no database updates will be made")
    logger.info("")

    # Process descriptions
    start_time = time.time()

    try:
        cost_tracker = generate_descriptions(
            args.db,
            api_key,
            logger,
            target=args.target,
            dry_run=args.dry_run,
            limit=args.limit,
            requests_per_minute=args.rate_limit
        )

        # Print summary
        elapsed = time.time() - start_time
        summary = cost_tracker.get_summary()

        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total API calls: {summary['api_calls']}")
        logger.info(f"Failed calls: {summary['failed_calls']}")
        logger.info(f"Input tokens: {summary['input_tokens']:,}")
        logger.info(f"Output tokens: {summary['output_tokens']:,}")
        logger.info(f"Total tokens: {summary['total_tokens']:,}")
        logger.info(f"Average tokens/call: {summary['avg_tokens_per_call']}")
        logger.info(f"Estimated cost: ${summary['total_cost_usd']:.4f} USD")
        logger.info(f"Elapsed time: {elapsed:.1f} seconds")
        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {type(e).__name__}: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
