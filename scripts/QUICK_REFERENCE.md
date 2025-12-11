# CELLxGENE Pipeline - Quick Reference

## One-Time Setup

```bash
# 1. Install dependencies
pip install pyyaml tqdm

# 2. Configure email (edit config file)
nano config/pipeline_config.yaml
# Set: api.ncbi.email to your email address

# 3. Run schema migration
python scripts/migrate_schema_for_cellxgene.py
```

## Common Commands

### Testing

```bash
# Dry run (no database changes)
python scripts/build_database_from_cellxgene.py --dry-run --limit 10

# Test with small limit
python scripts/build_database_from_cellxgene.py --limit 50

# Verbose debug output
python scripts/build_database_from_cellxgene.py --limit 10 --verbose
```

### Production

```bash
# Process specific number of rows
python scripts/build_database_from_cellxgene.py --limit 500

# Full run (all 1,573 datasets)
python scripts/build_database_from_cellxgene.py

# With custom config
python scripts/build_database_from_cellxgene.py --config my_config.yaml
```

### Validation

```bash
# Check database state
python3 -c "
import sqlite3
conn = sqlite3.connect('data/papers/metadata/papers.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM papers WHERE source=\"cellxgene\"')
print(f'CELLxGENE papers: {c.fetchone()[0]}')
c.execute('SELECT COUNT(*) FROM datasets WHERE dataset_id IS NOT NULL')
print(f'CELLxGENE datasets: {c.fetchone()[0]}')
conn.close()
"

# View logs
tail -f logs/build_database.log
tail -100 logs/build_database.log  # Last 100 lines

# Check last run summary
grep "PROCESSING SUMMARY" -A 20 logs/build_database.log | tail -20
```

## File Locations

| File | Path |
|------|------|
| Migration script | `scripts/migrate_schema_for_cellxgene.py` |
| Builder script | `scripts/build_database_from_cellxgene.py` |
| Configuration | `config/pipeline_config.yaml` |
| Database | `data/papers/metadata/papers.db` |
| Input CSV | `cellxgene_full_metadata.csv` |
| Logs | `logs/build_database.log` |
| Documentation | `scripts/CELLXGENE_PIPELINE_README.md` |

## Quick SQL Queries

```sql
-- Count CELLxGENE papers
SELECT COUNT(*) FROM papers WHERE source = 'cellxgene';

-- Count CELLxGENE datasets
SELECT COUNT(*) FROM datasets WHERE dataset_id IS NOT NULL;

-- View sample paper
SELECT title, doi, collection_id, pmid, source
FROM papers
WHERE source = 'cellxgene'
LIMIT 1;

-- View sample dataset
SELECT dataset_id, dataset_title, n_cells
FROM datasets
WHERE dataset_id IS NOT NULL
LIMIT 1;

-- Papers with multiple collections
SELECT
    doi,
    title,
    all_collection_ids
FROM papers
WHERE json_array_length(all_collection_ids) > 1;

-- Datasets by collection
SELECT
    p.title,
    COUNT(d.id) as dataset_count
FROM papers p
JOIN datasets d ON p.id = d.paper_id
WHERE p.source = 'cellxgene'
GROUP BY p.id
ORDER BY dataset_count DESC;
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Database not found | Run migration: `python scripts/migrate_schema_for_cellxgene.py` |
| Config not found | Copy template: `cp config/pipeline_config.yaml.example config/pipeline_config.yaml` |
| Rate limit errors | Reduce rate in config or get NCBI API key |
| Too many errors | Check logs: `tail -100 logs/build_database.log` |
| Want to restart | Safe to re-run - resumes automatically |

## Performance Tips

1. **Get NCBI API key** (free) - 3x faster
2. **Run on fast disk** (SSD recommended)
3. **Increase batch size** in config (default: 100)
4. **Enable WAL mode** (default: enabled)
5. **Use limit flag** for testing

## Expected Performance

| Rate Limit | Time for Full Dataset |
|------------|----------------------|
| 3 req/s (no key) | 9-10 minutes |
| 10 req/s (with key) | 3-4 minutes |

Processing rate: ~1.4-1.8 collections/second (with 3 req/s)

## Help

```bash
# Show all options
python scripts/build_database_from_cellxgene.py --help

# Migration help
python scripts/migrate_schema_for_cellxgene.py --help
```

## Common Workflows

### First Time Setup

```bash
python scripts/migrate_schema_for_cellxgene.py
python scripts/build_database_from_cellxgene.py --dry-run --limit 5
python scripts/build_database_from_cellxgene.py --limit 100
python scripts/build_database_from_cellxgene.py  # Full run
```

### Development/Testing

```bash
python scripts/build_database_from_cellxgene.py --dry-run --limit 5 --verbose
# Review output
python scripts/build_database_from_cellxgene.py --limit 5
# Check database
```

### Production Run

```bash
# Backup database first
cp data/papers/metadata/papers.db data/papers/metadata/papers.db.backup

# Run pipeline
python scripts/build_database_from_cellxgene.py

# Verify results
python3 -c "..."  # See validation commands above
```

### Re-running After Changes

```bash
# Safe to re-run - automatically resumes
python scripts/build_database_from_cellxgene.py

# Will update existing papers and skip existing datasets
```

## Status Indicators

During processing, you'll see:

```
Processing collections: 100%|██████████| 5/5 [00:03<00:00, 1.42it/s, papers=5, datasets=10, errors=0]
                                                              │               │           │           │
                                                              │               │           │           └─ Error count
                                                              │               │           └─ Datasets inserted
                                                              │               └─ Papers inserted/updated
                                                              └─ Processing rate (collections/sec)
```

## Configuration Quick Edits

```yaml
# In config/pipeline_config.yaml

# Change email
api.ncbi.email: "your_email@example.com"

# Add API key for faster processing
api.ncbi.api_key: "your_ncbi_api_key"

# Change batch commit size
database.batch_commit_size: 200  # Default: 100

# Change rate limits
rate_limits.ncbi_without_api_key: 2  # Default: 3

# Change log level
logging.level: "DEBUG"  # Default: INFO
```

## Emergency Commands

```bash
# Stop running process
Ctrl+C  # Graceful shutdown (commits current batch)

# Check if process is running
ps aux | grep build_database_from_cellxgene

# Kill hung process
kill -9 <PID>

# Restore from backup
cp data/papers/metadata/papers.db.backup data/papers/metadata/papers.db

# Clear logs
> logs/build_database.log
```

## Next Scripts (Coming Soon)

```bash
# Generate LLM descriptions (Script 3)
python scripts/generate_llm_descriptions.py

# Extract algorithms (Script 4)
python scripts/extract_algorithms_from_papers.py
```
