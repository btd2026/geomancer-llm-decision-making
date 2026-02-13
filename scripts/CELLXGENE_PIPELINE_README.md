# CELLxGENE Database Pipeline

This directory contains scripts for building and enriching the papers database with CELLxGENE collection and dataset metadata.

## Overview

The CELLxGENE pipeline consists of three main scripts:

1. **`database/migrate_schema_for_cellxgene.py`**: Schema migration to add CELLxGENE-specific columns
2. **`data_collection/build_database_from_cellxgene.py`**: Main pipeline to populate database from CELLxGENE metadata
3. **`llm_processing/generate_llm_descriptions.py`**: Generate AI-powered descriptions (to be implemented)
4. **`llm_processing/extract_algorithms_from_papers.py`**: Extract algorithm information from papers (to be implemented)

## Prerequisites

### Python Packages

```bash
pip install pyyaml tqdm
```

### Required Files

- **Database**: `data/papers/metadata/papers.db` (must exist)
- **CELLxGENE Metadata**: `cellxgene_full_metadata.csv` (at project root)
- **Configuration**: `config/pipeline_config.yaml`

### NCBI E-utilities

The pipeline uses NCBI E-utilities API to fetch paper metadata. Make sure:
- Email is configured in `config/pipeline_config.yaml`
- Rate limiting is respected (3 req/s without API key, 10 req/s with API key)

## Quick Start

### 1. Run Schema Migration

**First time setup** - add CELLxGENE columns to database:

```bash
python scripts/database/migrate_schema_for_cellxgene.py
```

This script is **idempotent** - safe to run multiple times. It will:
- Add 7 new columns to `papers` table
- Add 9 new columns to `datasets` table
- Create 7 performance indices
- Display summary of changes

**Output:**
```
================================================================================
CELLxGENE SCHEMA MIGRATION
================================================================================
Started at: 2025-11-04 08:47:56
Database: /path/to/papers.db

Migrating papers table...
  ✓ Added column: collection_id
  ✓ Added column: all_collection_ids
  ...

✓ Migration complete - 23 changes applied
```

### 2. Build Database from CELLxGENE

**Test with small limit first:**

```bash
# Dry run (no database changes)
python scripts/data_collection/build_database_from_cellxgene.py --dry-run --limit 10

# Real run with limit
python scripts/data_collection/build_database_from_cellxgene.py --limit 100
```

**Full production run:**

```bash
python scripts/data_collection/build_database_from_cellxgene.py
```

This will:
- Parse `cellxgene_full_metadata.csv` (1,573 datasets)
- Group datasets by collection (paper)
- Convert DOIs to PMIDs using NCBI API
- Fetch paper metadata from PubMed
- Insert/update papers and datasets in database
- Show progress bar with statistics

**Output:**
```
================================================================================
CELLxGENE DATABASE BUILDER
================================================================================
Started at: 2025-11-04 08:51:30

Processing collections: 100%|██████████| 5/5 [00:03<00:00,  1.42it/s]

================================================================================
PROCESSING SUMMARY
================================================================================

Collections processed: 5

Papers:
  - Inserted: 5
  - Updated: 0
  - Total: 5

Datasets:
  - Inserted: 10
  - Total: 10

API Calls: 10
Errors: 0
```

## Schema Changes

### Papers Table (New Columns)

| Column | Type | Description |
|--------|------|-------------|
| `collection_id` | TEXT | Primary CELLxGENE collection ID |
| `all_collection_ids` | TEXT | JSON array of all collection IDs (for papers with multiple collections) |
| `collection_name` | TEXT | CELLxGENE collection name |
| `source` | TEXT | Data source: 'pubmed_search', 'cellxgene', or 'both' |
| `llm_description` | TEXT | AI-generated description of the paper |
| `full_text` | TEXT | Full text from PMC (if available) |
| `has_full_text` | INTEGER | Boolean flag for full text availability |

### Datasets Table (New Columns)

| Column | Type | Description |
|--------|------|-------------|
| `dataset_id` | TEXT | CELLxGENE dataset ID (unique) |
| `collection_id` | TEXT | Associated collection ID |
| `dataset_title` | TEXT | Dataset title |
| `dataset_version_id` | TEXT | Dataset version ID |
| `dataset_h5ad_path` | TEXT | Filename of h5ad file |
| `llm_description` | TEXT | AI-generated description |
| `citation` | TEXT | Full citation string |
| `downloaded` | INTEGER | Boolean flag for download status |
| `benchmarked` | INTEGER | Boolean flag for benchmark status |

### New Indices

Performance indices created:
- `idx_papers_doi` - Fast DOI lookups
- `idx_papers_collection_id` - Collection filtering
- `idx_papers_source` - Source filtering
- `idx_datasets_dataset_id` - Dataset ID lookups
- `idx_datasets_collection_id` - Collection filtering
- `idx_datasets_downloaded` - Download status filtering
- `idx_datasets_benchmarked` - Benchmark status filtering

## Configuration

Edit `config/pipeline_config.yaml` to customize:

### API Settings

```yaml
api:
  ncbi:
    email: "your_email@example.com"  # REQUIRED
    api_key: null  # Optional: Add for 10 req/s instead of 3
```

### Rate Limiting

```yaml
rate_limits:
  ncbi_without_api_key: 3  # requests per second
  ncbi_with_api_key: 10
  max_retries: 3
  request_timeout: 30  # seconds
```

### Database Settings

```yaml
database:
  path: "data/papers/metadata/papers.db"
  batch_commit_size: 100  # Commit every N records
```

### Processing Behavior

```yaml
processing:
  multiple_collections:
    primary_strategy: "first"  # Which collection_id to use as primary
    store_all: true  # Store all in JSON array

  missing_doi:
    skip_if_not_found: false  # Continue even if DOI missing

  existing_papers:
    merge_strategy: "update"  # Update existing papers
    update_source: true  # Set source to "both"
```

## Command-Line Options

### build_database_from_cellxgene.py

```bash
python scripts/data_collection/build_database_from_cellxgene.py [OPTIONS]

Options:
  --config PATH       Path to config YAML (default: config/pipeline_config.yaml)
  --csv PATH          Path to CELLxGENE CSV (default: cellxgene_full_metadata.csv)
  --dry-run           Run without modifying database
  --limit N           Process only first N CSV rows
  --verbose           Enable debug output
  --no-progress       Disable progress bars
  -h, --help          Show help message
```

### Examples

```bash
# Test with dry run
python scripts/data_collection/build_database_from_cellxgene.py --dry-run --limit 5

# Process first 100 rows with verbose output
python scripts/data_collection/build_database_from_cellxgene.py --limit 100 --verbose

# Use custom configuration
python scripts/data_collection/build_database_from_cellxgene.py --config my_config.yaml

# Full production run
python scripts/data_collection/build_database_from_cellxgene.py
```

## Resume Capability

The pipeline supports **automatic resume** - it checks for existing records:

- **Papers**: If DOI exists, updates the record instead of inserting
- **Datasets**: If dataset_id exists, skips insertion
- **Progress tracking**: Uses database as checkpoint (no separate files needed)

**Example resume behavior:**

```bash
# First run: Insert 10 papers, 25 datasets
python scripts/data_collection/build_database_from_cellxgene.py --limit 50

# Second run (same limit): Update 10 papers, skip 25 datasets
python scripts/data_collection/build_database_from_cellxgene.py --limit 50

# Third run (larger limit): Update 10 existing, insert 15 new papers
python scripts/data_collection/build_database_from_cellxgene.py --limit 100
```

## Error Handling

The pipeline includes robust error handling:

### Retry Logic

- **Exponential backoff** for failed API calls
- **Configurable retries** (default: 3 attempts)
- **Jittered delays** to avoid thundering herd

### Graceful Degradation

- If PubMed lookup fails, paper still inserted with CELLxGENE metadata only
- Missing DOIs don't stop processing (configurable)
- Malformed data logged but doesn't crash pipeline

### Error Limits

```yaml
error_handling:
  continue_on_error: true  # Keep processing after errors
  max_errors: 100  # Stop if errors exceed this
  log_failures: true  # Log failed records
```

## Logs

Logs are written to:
- **Console**: INFO level and above
- **File**: `logs/build_database.log` (all levels)
  - Rotates at 10 MB
  - Keeps 5 backup files

**Example log output:**

```
2025-11-04 08:51:22 - cellxgene_pipeline - INFO - Parsing CSV file: cellxgene_full_metadata.csv
2025-11-04 08:51:22 - cellxgene_pipeline - INFO - Parsed 5 datasets into 2 unique collections
2025-11-04 08:51:22 - cellxgene_pipeline - DEBUG - Converting DOI to PMID: 10.1016/j.isci.2022.104097
2025-11-04 08:51:23 - cellxgene_pipeline - DEBUG - Found PMID 35372810
2025-11-04 08:51:23 - cellxgene_pipeline - DEBUG - Inserting new paper for DOI 10.1016/j.isci.2022.104097
```

## Performance

### Benchmarks (measured)

With **3 req/s** rate limit (no API key):
- ~1.4-1.8 collections/second
- ~2 API calls per collection (DOI→PMID, fetch metadata)
- **Estimated time for full dataset**: ~15-20 minutes for 1,573 datasets (~800 unique collections)

With **10 req/s** rate limit (with API key):
- ~3-4 collections/second
- **Estimated time**: ~5-7 minutes for full dataset

### Optimization Tips

1. **Get NCBI API key** (free) - 3x faster processing
2. **Increase batch commit size** for fewer disk writes
3. **Enable WAL mode** (enabled by default) for better concurrency
4. **Run on fast disk** (SSD recommended)

## Troubleshooting

### "Database not found"

Run schema migration first:
```bash
python scripts/database/migrate_schema_for_cellxgene.py
```

### "Configuration file not found"

Create config file:
```bash
mkdir -p config
cp config/pipeline_config.yaml.example config/pipeline_config.yaml
# Edit with your email
```

### "Rate limit exceeded"

Increase delays in config:
```yaml
rate_limits:
  ncbi_without_api_key: 2  # Reduce from 3
```

Or get an NCBI API key.

### "Too many errors"

Check logs:
```bash
tail -100 logs/build_database.log
```

Increase error tolerance:
```yaml
error_handling:
  max_errors: 200  # Increase from 100
```

## Data Validation

### Check Database State

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data/papers/metadata/papers.db')
cursor = conn.cursor()

# Count records
cursor.execute('SELECT COUNT(*) FROM papers WHERE source=\"cellxgene\"')
print(f'CELLxGENE papers: {cursor.fetchone()[0]}')

cursor.execute('SELECT COUNT(*) FROM datasets WHERE dataset_id IS NOT NULL')
print(f'CELLxGENE datasets: {cursor.fetchone()[0]}')

conn.close()
"
```

### Verify Sample Records

```sql
-- Sample paper with CELLxGENE data
SELECT
    title,
    doi,
    collection_id,
    collection_name,
    source
FROM papers
WHERE source = 'cellxgene'
LIMIT 1;

-- Sample dataset
SELECT
    dataset_id,
    dataset_title,
    collection_id,
    n_cells
FROM datasets
WHERE dataset_id IS NOT NULL
LIMIT 1;
```

## Next Steps

After running the database builder:

1. **Generate LLM Descriptions**
   ```bash
   python scripts/llm_processing/generate_llm_descriptions.py
   ```
   - Uses Claude Haiku for cost efficiency
   - Generates concise paper descriptions
   - Tracks token usage and costs

2. **Extract Algorithm Information**
   ```bash
   python scripts/llm_processing/extract_algorithms_from_papers.py
   ```
   - Extracts algorithm names, parameters, and context
   - Uses PMC full text when available
   - Populates `extracted_algorithms` table

3. **Populate Benchmark Results**
   ```bash
   python scripts/populate_benchmark_results.py
   ```
   - Scans `results/benchmarks/` directory
   - Matches datasets by CELLxGENE dataset_id
   - Populates `manylatents_results` table

## File Structure

```
llm-paper-analyze/
├── scripts/
│   ├── database/
│   │   └── migrate_schema_for_cellxgene.py     # Schema migration
│   ├── data_collection/
│   │   └── build_database_from_cellxgene.py    # Main builder
│   ├── llm_processing/
│   │   ├── generate_llm_descriptions.py        # LLM descriptions (TODO)
│   │   └── extract_algorithms_from_papers.py   # Algorithm extraction (TODO)
│   └── CELLXGENE_PIPELINE_README.md            # This file
├── config/
│   └── pipeline_config.yaml                    # Configuration
├── data/papers/metadata/
│   └── papers.db                               # SQLite database
├── logs/
│   └── build_database.log                     # Log file
└── cellxgene_full_metadata.csv                 # CELLxGENE metadata
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{cellxgene_pipeline,
  title = {CELLxGENE Database Pipeline},
  author = {Claude Code},
  year = {2025},
  url = {https://github.com/yourusername/llm-paper-analyze}
}
```

## Support

For issues or questions:
- Check logs in `logs/build_database.log`
- Review configuration in `config/pipeline_config.yaml`
- Open an issue on GitHub

## License

MIT License - see LICENSE file for details.
