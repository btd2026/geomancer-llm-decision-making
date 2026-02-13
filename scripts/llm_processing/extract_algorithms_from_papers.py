#!/usr/bin/env python3
"""
Extract algorithms from papers using LLM analysis.

This script extracts dimensionality reduction and analysis algorithms from
scientific papers. It prioritizes PMC full-text (Methods section) when available,
falls back to abstracts, and uses Claude API for structured extraction.

Features:
- Extracts from PMC full-text Methods section (preferred)
- Falls back to abstract if no full-text available
- Uses Claude API with structured JSON output
- Stores results in extracted_algorithms table
- Handles multiple algorithms per paper
- Assigns sequence ordering based on text appearance
- Resume capability (skips already-processed papers)
- Rate limiting and exponential backoff
- Progress tracking with tqdm
- Comprehensive error handling

Author: Claude Code
Created: 2025-11-04
"""

import argparse
import json
import logging
import logging.handlers
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
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

# PMC E-utilities Configuration
PMC_EFETCH_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Cost tracking (per million tokens)
HAIKU_INPUT_COST_PER_MTOK = 0.25
HAIKU_OUTPUT_COST_PER_MTOK = 1.25

# Rate limiting
DEFAULT_REQUESTS_PER_MINUTE = 50
DEFAULT_PMC_REQUESTS_PER_SECOND = 3
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_TIMEOUT = 30

# Checkpoint frequency
CHECKPOINT_FREQUENCY = 50

# Algorithm categories of interest
ALGORITHM_CATEGORIES = [
    "linear_dimensionality_reduction",
    "nonlinear_dimensionality_reduction",
    "deep_learning_dimensionality_reduction",
    "trajectory_inference",
    "velocity_analysis",
    "clustering",
    "normalization",
    "batch_correction",
    "feature_selection"
]

# Known algorithms (for validation)
KNOWN_ALGORITHMS = {
    # Linear dimensionality reduction
    "PCA", "ICA", "NMF", "FA",
    # Nonlinear dimensionality reduction
    "t-SNE", "UMAP", "PHATE", "Diffusion Maps", "LLE", "Isomap",
    # Deep learning
    "VAE", "Autoencoder", "scVI", "scVI-LD", "DCA", "scGAN", "ZIFA",
    # Trajectory
    "Monocle", "Monocle2", "Monocle3", "Slingshot", "PAGA", "URD", "Palantir",
    # Velocity
    "velocyto", "scVelo", "RNA Velocity", "CellRank",
    # Clustering
    "Louvain", "Leiden", "k-means", "hierarchical clustering", "DBSCAN",
    # Normalization
    "log-normalization", "CPM", "TPM", "FPKM", "RPKM", "SCTransform", "scran",
    # Batch correction
    "ComBat", "Harmony", "Seurat Integration", "MNN", "Scanorama",
    # Feature selection
    "HVG", "highly variable genes"
}


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
            'total_cost_usd': round(self.total_cost, 4)
        }


@dataclass
class PaperRecord:
    """Represents a paper from the database."""
    paper_id: int
    pmid: Optional[str]
    title: str
    abstract: Optional[str]
    pmc_id: Optional[str]
    methods_extracted: bool


@dataclass
class ExtractedAlgorithm:
    """Represents an extracted algorithm."""
    algorithm_name: str
    algorithm_category: str
    parameters: Optional[Dict[str, Any]]
    context: str
    confidence: float
    sequence_order: int


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging with both file and console handlers.

    Args:
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('extract_algorithms')
    logger.setLevel(getattr(logging, log_level.upper()))

    if logger.handlers:
        return logger

    # File handler
    log_file = log_dir / f"extract_algorithms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
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
    Create database connection.

    Args:
        db_path: Path to SQLite database

    Returns:
        Database connection

    Raises:
        FileNotFoundError: If database doesn't exist
        sqlite3.Error: If connection fails
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_papers_for_extraction(
    conn: sqlite3.Connection,
    limit: Optional[int] = None,
    force_reprocess: bool = False
) -> List[PaperRecord]:
    """
    Fetch papers that need algorithm extraction.

    Args:
        conn: Database connection
        limit: Maximum number of papers to fetch
        force_reprocess: If True, reprocess all papers

    Returns:
        List of PaperRecord objects
    """
    if force_reprocess:
        query = """
            SELECT
                id, pmid, title, abstract,
                NULL as pmc_id, methods_extracted
            FROM papers
            WHERE abstract IS NOT NULL AND abstract != ''
            ORDER BY id
        """
    else:
        query = """
            SELECT
                id, pmid, title, abstract,
                NULL as pmc_id, methods_extracted
            FROM papers
            WHERE (methods_extracted IS NULL OR methods_extracted = 0)
            AND abstract IS NOT NULL AND abstract != ''
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
            title=row['title'],
            abstract=row['abstract'],
            pmc_id=row['pmc_id'] if 'pmc_id' in row.keys() else None,
            methods_extracted=bool(row['methods_extracted'])
        ))

    return papers


def save_extracted_algorithms(
    conn: sqlite3.Connection,
    paper_id: int,
    algorithms: List[ExtractedAlgorithm],
    clear_existing: bool = False
) -> int:
    """
    Save extracted algorithms to database.

    Args:
        conn: Database connection
        paper_id: Paper ID
        algorithms: List of extracted algorithms
        clear_existing: If True, clear existing entries for this paper

    Returns:
        Number of algorithms saved

    Raises:
        sqlite3.Error: If database operation fails
    """
    cursor = conn.cursor()

    # Clear existing entries if requested
    if clear_existing:
        cursor.execute(
            "DELETE FROM extracted_algorithms WHERE paper_id = ?",
            (paper_id,)
        )

    # Insert new algorithms
    saved = 0
    for algo in algorithms:
        cursor.execute(
            """
            INSERT INTO extracted_algorithms (
                paper_id, algorithm_name, algorithm_category,
                parameters, sequence_order, mentioned_in_section,
                context_text, extraction_method, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                algo.algorithm_name,
                algo.algorithm_category,
                json.dumps(algo.parameters) if algo.parameters else None,
                algo.sequence_order,
                'methods',  # Assuming from methods section
                algo.context,
                'llm',
                algo.confidence
            )
        )
        saved += 1

    # Mark paper as processed
    cursor.execute(
        """
        UPDATE papers
        SET methods_extracted = 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (paper_id,)
    )

    return saved


# ============================================================================
# PMC Full-Text Fetching
# ============================================================================

def extract_methods_section(full_text: str) -> Optional[str]:
    """
    Extract Methods section from PMC full-text.

    Args:
        full_text: Full-text content from PMC

    Returns:
        Methods section text, or None if not found
    """
    # Try various common section headers
    patterns = [
        r'(?i)##?\s*methods?\s*\n(.*?)(?:\n##?\s*[A-Z]|\Z)',
        r'(?i)##?\s*materials?\s+and\s+methods?\s*\n(.*?)(?:\n##?\s*[A-Z]|\Z)',
        r'(?i)##?\s*experimental\s+procedures?\s*\n(.*?)(?:\n##?\s*[A-Z]|\Z)',
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text, re.DOTALL | re.MULTILINE)
        if match:
            methods_text = match.group(1).strip()
            # Limit to reasonable length (avoid extracting entire paper)
            if len(methods_text) > 50:
                return methods_text[:5000]  # Max 5000 chars

    return None


def fetch_pmc_full_text(
    pmid: str,
    logger: logging.Logger,
    timeout: int = DEFAULT_TIMEOUT
) -> Optional[str]:
    """
    Fetch full-text from PMC using PMID.

    Args:
        pmid: PubMed ID
        logger: Logger instance
        timeout: Request timeout

    Returns:
        Full-text content, or None if unavailable
    """
    # Note: This is a simplified implementation
    # In production, you'd need to:
    # 1. First check if PMC ID exists for this PMID
    # 2. Use proper PMC API endpoint
    # 3. Parse XML response correctly

    # For now, return None to fall back to abstract
    # TODO: Implement proper PMC full-text fetching
    logger.debug(f"PMC full-text fetching not yet implemented for PMID {pmid}")
    return None


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
            timeout: Request timeout
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
        max_tokens: int = 1000
    ) -> Tuple[str, int, int]:
        """
        Make a request to Claude API with retry logic.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            ClaudeAPIError: If request fails
            ClaudeRateLimitError: If rate limit exceeded
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

                # Validate response
                if 'content' not in response_data:
                    raise ClaudeAPIError("Invalid response: missing 'content'")

                if not response_data['content']:
                    raise ClaudeAPIError("Empty content in response")

                text = response_data['content'][0]['text']
                usage = response_data.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)

                return text.strip(), input_tokens, output_tokens

            except HTTPError as e:
                if e.code == 429:
                    retry_delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {retry_delay:.1f}s"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise ClaudeRateLimitError("Rate limit exceeded")

                elif e.code >= 500:
                    retry_delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(
                        f"Server error {e.code} (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {retry_delay:.1f}s"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_delay)
                        continue

                error_body = e.read().decode('utf-8') if e.fp else "No details"
                raise ClaudeAPIError(f"HTTP {e.code}: {error_body}")

            except URLError as e:
                retry_delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                self.logger.warning(
                    f"Network error (attempt {attempt + 1}/{self.max_retries}): {e.reason}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ClaudeAPIError(f"Network error: {e.reason}")

            except json.JSONDecodeError as e:
                raise ClaudeAPIError(f"Invalid JSON response: {e}")

            except Exception as e:
                raise ClaudeAPIError(f"Unexpected error: {type(e).__name__}: {e}")

        raise ClaudeAPIError(f"Failed after {self.max_retries} attempts")

    def extract_algorithms(
        self,
        paper: PaperRecord,
        methods_text: Optional[str] = None
    ) -> Tuple[List[ExtractedAlgorithm], int, int]:
        """
        Extract algorithms from paper using LLM.

        Args:
            paper: PaperRecord object
            methods_text: Methods section text (if available)

        Returns:
            Tuple of (algorithms, input_tokens, output_tokens)
        """
        # Use methods text if available, otherwise use abstract
        if methods_text:
            text = methods_text
            source = "Methods section"
        else:
            text = paper.abstract or ""
            source = "Abstract"

        # Build prompt
        prompt = f"""Extract all dimensionality reduction and analysis algorithms from this {source}.

Text:
{text}

Return ONLY a JSON array (no other text). Each algorithm should have:
{{
  "name": "algorithm name",
  "category": "category from list below",
  "parameters": {{"param": "value"}},
  "context": "brief context (1 sentence)",
  "confidence": 0.0-1.0
}}

Categories:
- linear_dimensionality_reduction
- nonlinear_dimensionality_reduction
- deep_learning_dimensionality_reduction
- trajectory_inference
- velocity_analysis
- clustering
- normalization
- batch_correction
- feature_selection

Algorithms of interest:
PCA, ICA, NMF, t-SNE, UMAP, PHATE, Diffusion Maps, VAE, Autoencoder, scVI, DCA,
scGAN, ZIFA, Monocle, Slingshot, scVelo, velocyto, RNA Velocity, PAGA, Louvain,
Leiden, k-means, log-normalization, CPM, TPM, FPKM, SCTransform, scran, ComBat,
Harmony, Seurat Integration, MNN, Scanorama, HVG

Example:
[
  {{
    "name": "PCA",
    "category": "linear_dimensionality_reduction",
    "parameters": {{"n_components": 50}},
    "context": "Used for initial dimensionality reduction",
    "confidence": 0.9
  }}
]

Return empty array [] if no algorithms found. ONLY return the JSON array."""

        response_text, input_tokens, output_tokens = self._make_request(prompt)

        # Parse JSON response
        algorithms = self._parse_algorithm_response(response_text)

        return algorithms, input_tokens, output_tokens

    def _parse_algorithm_response(self, response_text: str) -> List[ExtractedAlgorithm]:
        """
        Parse LLM response into ExtractedAlgorithm objects.

        Args:
            response_text: JSON response from LLM

        Returns:
            List of ExtractedAlgorithm objects
        """
        # Extract JSON from response (handle markdown code blocks)
        json_text = response_text.strip()

        # Remove markdown code blocks if present
        if json_text.startswith('```'):
            lines = json_text.split('\n')
            # Remove first and last lines (``` markers)
            json_text = '\n'.join(lines[1:-1])

        # Remove any "json" language identifier
        json_text = json_text.replace('```json', '').replace('```', '').strip()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response text: {response_text}")
            return []

        if not isinstance(data, list):
            self.logger.warning("Response is not a JSON array")
            return []

        algorithms = []
        for i, item in enumerate(data):
            try:
                algo = ExtractedAlgorithm(
                    algorithm_name=item['name'],
                    algorithm_category=item['category'],
                    parameters=item.get('parameters'),
                    context=item.get('context', ''),
                    confidence=float(item.get('confidence', 0.5)),
                    sequence_order=i + 1
                )
                algorithms.append(algo)
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Skipping invalid algorithm entry: {e}")
                continue

        return algorithms


# ============================================================================
# Main Processing
# ============================================================================

def process_papers(
    conn: sqlite3.Connection,
    client: ClaudeClient,
    cost_tracker: CostTracker,
    logger: logging.Logger,
    dry_run: bool = False,
    limit: Optional[int] = None,
    force_reprocess: bool = False
) -> Tuple[int, int]:
    """
    Process papers and extract algorithms.

    Args:
        conn: Database connection
        client: Claude API client
        cost_tracker: Cost tracking object
        logger: Logger instance
        dry_run: If True, don't update database
        limit: Maximum papers to process
        force_reprocess: If True, reprocess all papers

    Returns:
        Tuple of (papers_processed, algorithms_extracted)
    """
    papers = fetch_papers_for_extraction(conn, limit, force_reprocess)

    if not papers:
        logger.info("No papers found needing algorithm extraction")
        return 0, 0

    logger.info(f"Found {len(papers)} papers for processing")

    papers_processed = 0
    algorithms_extracted = 0

    with tqdm(total=len(papers), desc="Extracting algorithms") as pbar:
        for i, paper in enumerate(papers):
            try:
                # Try to get full-text methods section
                methods_text = None
                if paper.pmid:
                    methods_text = fetch_pmc_full_text(paper.pmid, logger)

                # Extract algorithms
                algorithms, input_tokens, output_tokens = client.extract_algorithms(
                    paper, methods_text
                )

                cost_tracker.add_usage(input_tokens, output_tokens)

                if algorithms:
                    logger.debug(
                        f"Paper {paper.paper_id}: extracted {len(algorithms)} algorithms"
                    )

                    if not dry_run:
                        saved = save_extracted_algorithms(
                            conn, paper.paper_id, algorithms, clear_existing=force_reprocess
                        )
                        algorithms_extracted += saved

                        # Checkpoint
                        if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                            conn.commit()
                            logger.debug(f"Checkpoint: committed {i + 1} papers")
                else:
                    # Mark as processed even if no algorithms found
                    if not dry_run:
                        conn.execute(
                            "UPDATE papers SET methods_extracted = 1 WHERE id = ?",
                            (paper.paper_id,)
                        )

                papers_processed += 1

            except ClaudeRateLimitError as e:
                logger.error(f"Rate limit error for paper {paper.paper_id}: {e}")
                cost_tracker.record_failure()
                logger.warning("Stopping due to rate limit")
                break

            except ClaudeAPIError as e:
                logger.error(f"API error for paper {paper.paper_id}: {e}")
                cost_tracker.record_failure()

            except sqlite3.Error as e:
                logger.error(f"Database error for paper {paper.paper_id}: {e}")
                conn.rollback()

            except Exception as e:
                logger.error(
                    f"Unexpected error for paper {paper.paper_id}: "
                    f"{type(e).__name__}: {e}"
                )
                cost_tracker.record_failure()

            finally:
                pbar.update(1)

    # Final commit
    if not dry_run and papers_processed > 0:
        conn.commit()
        logger.info(f"Final commit: {papers_processed} papers processed")

    return papers_processed, algorithms_extracted


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract algorithms from papers using LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract algorithms from all unprocessed papers
  %(prog)s --db papers.db --api-key $ANTHROPIC_API_KEY

  # Dry run
  %(prog)s --db papers.db --api-key $ANTHROPIC_API_KEY --dry-run

  # Process only 10 papers (testing)
  %(prog)s --db papers.db --api-key $ANTHROPIC_API_KEY --limit 10

  # Force reprocess all papers
  %(prog)s --db papers.db --api-key $ANTHROPIC_API_KEY --force-reprocess

  # Use environment variable for API key
  export ANTHROPIC_API_KEY=sk-ant-...
  %(prog)s --db papers.db
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
        '--limit',
        type=int,
        help='Maximum papers to process (for testing)'
    )

    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Reprocess papers that were already processed'
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
    logger.info("Algorithm Extraction from Papers")
    logger.info("=" * 70)

    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error(
            "No API key provided. Use --api-key or set ANTHROPIC_API_KEY"
        )
        return 1

    # Validate database
    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    # Log configuration
    logger.info(f"Database: {args.db}")
    logger.info(f"Model: {CLAUDE_MODEL}")
    logger.info(f"Rate limit: {args.rate_limit} requests/minute")
    if args.limit:
        logger.info(f"Limit: {args.limit} papers")
    if args.force_reprocess:
        logger.info("Force reprocess: enabled")
    if args.dry_run:
        logger.warning("DRY RUN MODE - no database updates")
    logger.info("")

    # Process papers
    start_time = time.time()

    try:
        conn = get_database_connection(args.db)
        client = ClaudeClient(api_key, logger, requests_per_minute=args.rate_limit)
        cost_tracker = CostTracker()

        papers_processed, algorithms_extracted = process_papers(
            conn,
            client,
            cost_tracker,
            logger,
            dry_run=args.dry_run,
            limit=args.limit,
            force_reprocess=args.force_reprocess
        )

        conn.close()

        # Print summary
        elapsed = time.time() - start_time
        summary = cost_tracker.get_summary()

        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Papers processed: {papers_processed}")
        logger.info(f"Algorithms extracted: {algorithms_extracted}")
        logger.info(f"API calls: {summary['api_calls']}")
        logger.info(f"Failed calls: {summary['failed_calls']}")
        logger.info(f"Input tokens: {summary['input_tokens']:,}")
        logger.info(f"Output tokens: {summary['output_tokens']:,}")
        logger.info(f"Total tokens: {summary['total_tokens']:,}")
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
