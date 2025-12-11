# Scripts Directory

This directory contains all executable scripts for the llm-paper-analyze project.

**Last Updated:** November 4, 2025

---

## Current Execution Model

**Local Python Scripts** - All scripts run directly using `python3` without requiring containers.

**Database Location:** `/home/btd8/llm-paper-analyze/data/papers/metadata/papers.db`

---

## Core Pipeline Scripts

### 1. search_pubmed_local.py ✨ **UPDATED**
**Purpose:** Search PubMed and fetch papers with validated GEO datasets

**Usage:**
```bash
python3 scripts/search_pubmed_local.py [N]
```

**Examples:**
```bash
# Fetch 100 papers (only stores those with valid GEO datasets)
python3 scripts/search_pubmed_local.py 100

# Fetch 20 papers (default if no argument)
python3 scripts/search_pubmed_local.py
```

**Features:**
- Uses NCBI E-utilities API directly
- Builds query from `research_context.json`
- **GEO FILTERING:** Only stores papers with validated GEO datasets (mandatory)
- **GEO API Validation:** Validates each accession using NCBI E-utilities (db=gds)
- **Trajectory Support:** Searches for pseudotime, RNA velocity, trajectory inference keywords
- Filters papers to 2024+ only (post-processing filter)
- Rate limited: 3 requests/second (0.34s delay)
- **Hit Rate:** ~27% of papers have valid GEO datasets

**What it extracts:**
- PMID, DOI, title, abstract
- Authors, journal, publication date
- Keywords and MeSH terms
- URLs (PubMed, DOI)
- **GEO accessions** (GSE, GSM, GPL, GDS) with real-time validation
- Stores only validated accessions as JSON array

**Example Output:**
```
[8/100] Paper 39938577: Joint analysis of single-cell RNA sequencing...
  Found GEO accessions: GSE138794
  Validating...
  ✓ Valid GEO datasets: GSE138794

GEO Filtering Summary:
  Total papers: 100
  Papers without GEO accessions: 73
  Papers with valid GEO datasets: 27
```

---

### 2. extract_datasets_local.py
**Purpose:** Extract dataset information from paper abstracts

**Usage:**
```bash
python3 scripts/extract_datasets_local.py
```

**What it extracts:**
- GEO accession numbers (GSE, GSM)
- SRA accessions (SRP, SRX, SRR)
- ArrayExpress IDs
- GitHub URLs
- Cell/gene counts
- Organism (human, mouse, etc.)
- Tissue types
- Sequencing platforms (10x, Smart-seq, etc.)

**Output:** Updates `papers` table and creates records in `datasets` table

---

### 3. extract_algorithms_local.py
**Purpose:** Extract dimensionality reduction algorithms from papers

**Usage:**
```bash
python3 scripts/extract_algorithms_local.py
```

**What it extracts:**
- Algorithm names (PCA, UMAP, t-SNE, PHATE, VAE, etc.)
- Algorithm categories (dimensionality_reduction, normalization, clustering)
- Sequence order in pipeline
- Context text (surrounding text snippet)

**Algorithms detected:**
- **Dimensionality Reduction:** PCA, UMAP, t-SNE, PHATE, Autoencoder, VAE, NMF, ICA, Diffusion Maps, scVI, DCA, scGAN, ZIFA
- **Normalization:** log-normalization, CPM, TPM, SCTransform, scran
- **Clustering:** Louvain, Leiden, k-means

**Output:** Creates records in `extracted_algorithms` table

---

## CELLxGENE Pipeline Scripts

### 4. build_database_from_cellxgene.py ✨ **NEW - Production Grade**
**Purpose:** Build comprehensive database from CELLxGENE metadata with PubMed enrichment

**Usage:**
```bash
# Process CELLxGENE metadata file
python3 scripts/build_database_from_cellxgene.py \
  --input /path/to/cellxgene_full_metadata.csv \
  --db papers.db \
  --log-level INFO

# Dry run (no database updates)
python3 scripts/build_database_from_cellxgene.py \
  --input metadata.csv \
  --db papers.db \
  --dry-run

# Process limited number for testing
python3 scripts/build_database_from_cellxgene.py \
  --input metadata.csv \
  --db papers.db \
  --limit 10
```

**Features:**
- **Type-safe:** Full type hints on all functions
- **Comprehensive error handling:** Specific exception types with exponential backoff
- **Resume capability:** Checks existing records, skips processed papers
- **Rate limiting:** Configurable PubMed API rate limits (3 req/sec default)
- **Progress tracking:** tqdm progress bars with ETA
- **Batch operations:** Commits every 100 records for efficiency
- **Logging:** Rotating file handler + console output
- **Configuration-driven:** Supports YAML config files
- **Database safety:** Parameterized queries, transaction rollback on errors

**What it does:**
1. Parses CELLxGENE CSV metadata
2. Extracts unique papers by DOI
3. Fetches PubMed metadata (PMID, abstract, journal, etc.)
4. Attempts PMC full-text retrieval
5. Stores paper and dataset information in database

**Critical Fix Applied:** Added `import logging.handlers` (was causing crashes)

---

### 5. generate_llm_descriptions.py ✨ **NEW - Production Grade**
**Purpose:** Generate AI-powered descriptions for papers and datasets using Claude API

**Usage:**
```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Generate descriptions for all papers
python3 scripts/generate_llm_descriptions.py --db papers.db

# Dry run (test without database updates)
python3 scripts/generate_llm_descriptions.py --db papers.db --dry-run

# Process limited number with custom rate limit
python3 scripts/generate_llm_descriptions.py \
  --db papers.db \
  --limit 10 \
  --rate-limit 30

# Debug mode
python3 scripts/generate_llm_descriptions.py \
  --db papers.db \
  --log-level DEBUG
```

**Features:**
- **Claude Haiku:** Uses claude-3-haiku-20240307 for cost efficiency
- **Cost tracking:** Counts input/output tokens, calculates USD cost
- **Resume capability:** Automatically skips papers with existing descriptions
- **Rate limiting:** Configurable (50 req/min default)
- **Retry logic:** Exponential backoff with jitter for API failures
- **Checkpoints:** Saves every 100 descriptions
- **Validation:** Validates API responses before processing
- **Transaction safety:** Rolls back on errors
- **Progress tracking:** tqdm with detailed progress

**Cost estimates (Haiku):**
- Input: $0.25 per 1M tokens
- Output: $1.25 per 1M tokens
- Typical paper: ~500 input + 100 output tokens = $0.00025 per description

**Prompt templates:**
- **Papers:** 2-3 sentence summary focusing on research question, methodology, findings
- **Datasets:** 2-3 sentence description with biological context, experimental design

**Output:** Updates `papers.dataset_description` field

---

### 6. extract_algorithms_from_papers.py ✨ **NEW - Production Grade**
**Purpose:** Extract dimensionality reduction and analysis algorithms using LLM analysis

**Usage:**
```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Extract from all unprocessed papers
python3 scripts/extract_algorithms_from_papers.py --db papers.db

# Dry run
python3 scripts/extract_algorithms_from_papers.py --db papers.db --dry-run

# Process limited number
python3 scripts/extract_algorithms_from_papers.py \
  --db papers.db \
  --limit 10

# Force reprocess all papers
python3 scripts/extract_algorithms_from_papers.py \
  --db papers.db \
  --force-reprocess
```

**Features:**
- **PMC full-text priority:** Extracts from Methods section when available
- **Fallback to abstract:** Uses abstract if no full-text
- **Structured extraction:** Returns JSON with name, category, parameters, context, confidence
- **Multiple algorithms:** Handles multiple algorithms per paper
- **Sequence ordering:** Orders algorithms by appearance in text
- **Resume capability:** Skips papers with `methods_extracted = 1`
- **Validation:** Validates extracted data against known algorithms
- **Rate limiting:** Configurable (50 req/min default)
- **Checkpoints:** Saves every 50 papers

**Algorithms extracted:**
- **Linear DR:** PCA, ICA, NMF
- **Nonlinear DR:** t-SNE, UMAP, PHATE, Diffusion Maps
- **Deep Learning DR:** VAE, Autoencoder, scVI, DCA, scGAN, ZIFA
- **Trajectory:** Monocle, Slingshot, PAGA, URD, Palantir
- **Velocity:** velocyto, scVelo, RNA Velocity, CellRank
- **Clustering:** Louvain, Leiden, k-means, hierarchical, DBSCAN
- **Normalization:** log-norm, CPM, TPM, FPKM, SCTransform, scran
- **Batch correction:** ComBat, Harmony, Seurat Integration, MNN, Scanorama
- **Feature selection:** HVG (highly variable genes)

**LLM prompt:** Structured JSON extraction with categories and confidence scores

**Output:** Creates records in `extracted_algorithms` table

---

## Legacy/Alternative Scripts

### generate_dataset_descriptions.py (Legacy)
**Note:** Superseded by `generate_llm_descriptions.py` (more robust, better error handling)

**Purpose:** Generate AI-powered concise dataset descriptions

**Usage:**
```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Generate descriptions for all papers
python3 scripts/generate_dataset_descriptions.py --force

# Generate descriptions for papers without descriptions only
python3 scripts/generate_dataset_descriptions.py
```

**Features:**
- Uses Claude API (claude-3-7-sonnet-20250219)
- Generates 1-2 sentence descriptions
- Fallback to programmatic generation if API unavailable
- Processes abstracts and available methods sections
- Average 276 characters per description

**Example Output:**
```
"Human prostate cancer single-cell RNA-seq (n=14), bulk RNA-seq (n=13),
and whole exome sequencing (n=8) datasets from multiple cohorts including
TCGA, analyzing tumor heterogeneity and treatment response markers."
```

**Output:** Updates `dataset_description` field in `papers` table

**Cost:** ~$0.02 for 100 papers

---

### 5. fetch_pmc_fulltext.py
**Purpose:** Fetch full-text Methods and Data Availability sections from PMC

**Usage:**
```bash
python3 scripts/fetch_pmc_fulltext.py
```

**Features:**
- Converts PMID to PMC ID
- Fetches full-text XML from PMC Open Access
- Extracts Methods section (up to 6000 chars)
- Extracts Data Availability statement
- Rate limited to avoid overload
- Focuses on dataset-relevant subsections

**Output:** Updates `papers` table with methods and data availability text

**Note:** Only works for papers in PMC Open Access collection

---

### 6. extract_from_fulltext.py
**Purpose:** Enhanced extraction from full-text Methods sections

**Usage:**
```bash
python3 scripts/extract_from_fulltext.py
```

**Prerequisites:** Must run `fetch_pmc_fulltext.py` first

**What it extracts (enhanced accuracy):**
- More precise GEO/SRA/ArrayExpress accessions
- Better cell count detection
- Specific platform versions (10x v2 vs v3, Smart-seq2 vs v3)
- Preprocessing tools (CellRanger, STARsolo, Alevin, kallisto|bustools)
- Normalization details (scran versions, SCTransform)

**Output:** Updates `datasets` table with enhanced information

---

### 7. export_database.py
**Purpose:** Export database tables to CSV and generate reports

**Usage:**
```bash
python3 scripts/export_database.py
```

**Output Location:** `exports/export_YYYYMMDD_HHMMSS/`

**Files Generated:**
- `papers.csv` - All paper metadata
- `datasets.csv` - Dataset characteristics
- `extracted_algorithms.csv` - Algorithm mentions
- `papers.db` - Copy of database
- `SUMMARY.md` - Statistics and analysis
- `README.md` - Export documentation

---

## Benchmarking Scripts (Phase 2)

### 8. create_synthetic_scrna.py ✨
**Purpose:** Generate realistic synthetic scRNA-seq datasets for testing

**Usage:**
```bash
# Default parameters (5k cells, 2k genes, 5 cell types, 90% sparse)
python3 scripts/create_synthetic_scrna.py

# Custom parameters
python3 scripts/create_synthetic_scrna.py \
  --n-cells 10000 \
  --n-genes 3000 \
  --n-cell-types 6 \
  --sparsity 0.92 \
  --output data/synthetic/my_dataset.h5ad
```

**Features:**
- Creates realistic scRNA-seq characteristics
- High sparsity (typical: 85-95% zeros)
- Multiple distinct cell type populations
- Marker genes with cell-type-specific expression
- Saved in AnnData (.h5ad) format

**Parameters:**
- `--n-cells`: Number of cells (default: 5000)
- `--n-genes`: Number of genes (default: 2000)
- `--n-cell-types`: Number of cell populations (default: 5)
- `--sparsity`: Target sparsity fraction (default: 0.90)
- `--output`: Output file path (default: data/synthetic/scrna_test.h5ad)

**Output:** AnnData file with:
- Expression matrix (sparse, realistic sparsity)
- Cell metadata (cell_id, cell_type, n_genes_detected)
- Gene metadata (gene_id, n_cells_expressed)
- QC metrics (total_counts, n_genes_by_counts, etc.)

**Example:**
```bash
# Create large realistic dataset
python3 scripts/create_synthetic_scrna.py \
  --n-cells 10000 \
  --n-genes 3000 \
  --n-cell-types 6 \
  --sparsity 0.92 \
  --output data/synthetic/realistic_scrna.h5ad

# Output:
# ✅ SUCCESS - Synthetic Dataset Created!
# Saved to: data/synthetic/realistic_scrna.h5ad
# Shape: (10000, 3000) (cells × genes)
# Sparsity: 91.90%
# Cell types: 6
# File size: 230.7 MB
```

---

### 9. benchmark_with_file.py ✨
**Purpose:** Benchmark algorithms on external .h5ad dataset files

**Usage:**
```bash
# Benchmark a specific dataset
python3 scripts/benchmark_with_file.py data/synthetic/realistic_scrna.h5ad

# Custom database path
python3 scripts/benchmark_with_file.py \
  data/geo/GSE123456.h5ad \
  --db-path custom_path/papers.db
```

**What it does:**
1. Loads .h5ad dataset file
2. Runs PCA, UMAP, t-SNE on the data
3. Measures execution time and performance
4. Stores results in `file_benchmarks` table

**Algorithms tested:**
- **PCA** (2 components)
- **UMAP** (2 components)
- **t-SNE** (2 components)

**Metrics captured:**
- Dataset characteristics (n_cells, n_genes, sparsity)
- Execution time (seconds)
- Embeddings shape
- Success/failure status
- Error messages (if failed)

**Output table:** `file_benchmarks`
```sql
CREATE TABLE file_benchmarks (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    dataset_name TEXT,
    dataset_path TEXT,
    n_cells INTEGER,
    n_genes INTEGER,
    algorithm_name TEXT,
    success BOOLEAN,
    embeddings_shape TEXT,
    execution_time REAL,
    scores TEXT,
    error TEXT
);
```

**Example results:**
```
Algorithm       Time (s)   Cells      Genes
─────────────────────────────────────────
PCA             1.01       10000      3000
UMAP            58.13      10000      3000
t-SNE           54.94      10000      3000
```

---

### 10. simple_benchmark_v2.py ✨
**Purpose:** Benchmark algorithms on built-in test datasets

**Usage:**
```bash
# Run benchmarks on swissroll dataset (default)
python3 scripts/simple_benchmark_v2.py

# Custom database path
python3 scripts/simple_benchmark_v2.py --db-path custom_path/papers.db
```

**What it does:**
1. Generates swissroll manifold test data (5k samples, 200 features)
2. Runs PCA, UMAP, t-SNE
3. Measures performance
4. Stores results in `poc_benchmarks` table

**Use case:** Proof of concept validation, testing pipeline changes

**Output table:** `poc_benchmarks`
```sql
CREATE TABLE poc_benchmarks (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    dataset_name TEXT,
    algorithm_name TEXT,
    success BOOLEAN,
    embeddings_shape TEXT,
    execution_time REAL,
    scores TEXT,
    error TEXT
);
```

**Example results:**
```
Algorithm       Time (s)   Shape
──────────────────────────────
PCA             0.48       [5000, 2]
UMAP            17.90      [5000, 2]
t-SNE           89.24      [5000, 2]
```

---

### 11. download_geo_dataset.py
**Purpose:** Download scRNA-seq datasets from GEO and convert to AnnData

**Usage:**
```bash
# Download a specific GEO dataset
python3 scripts/download_geo_dataset.py GSE152048

# Specify custom output directory
python3 scripts/download_geo_dataset.py GSE152048 --output data/geo/
```

**What it does:**
1. Downloads dataset from GEO using GEOparse
2. Searches for compatible data files (.h5ad, .h5, .mtx)
3. Converts to AnnData format if needed
4. Saves as .h5ad file

**Supported formats:**
- AnnData (.h5ad) - direct copy
- HDF5 (.h5) - convert to AnnData
- Matrix Market (.mtx) - convert to AnnData
- Raw sequencing data - reports as not supported

**Limitations:**
- Many GEO datasets only have raw sequencing reads (not processed matrices)
- Some datasets lack public files
- Use `create_synthetic_scrna.py` as fallback for testing

**Example:**
```bash
python3 scripts/download_geo_dataset.py GSE152048

# If fails with "Could not find compatible data files"
# Use synthetic data instead:
python3 scripts/create_synthetic_scrna.py \
  --n-cells 10000 \
  --n-genes 3000 \
  --output data/synthetic/test_data.h5ad
```

---

## Utility Scripts

### show_query.py
**Purpose:** Display PubMed search query without executing search

**Usage:**
```bash
python3 scripts/show_query.py
```

**Output:** Shows the query built from `research_context.json`

---

### view_papers.py
**Purpose:** View papers from database

**Usage:**
```bash
python3 scripts/view_papers.py
```

**Output:** Displays paper list with basic information

---

### analyze_papers.py
**Purpose:** Generate analysis statistics

**Usage:**
```bash
python3 scripts/analyze_papers.py
```

**Output:** Statistics about papers, algorithms, and datasets

---

## Database Management Scripts

### init_database.sql
**Purpose:** Database schema definition

**Tables:**
- `papers` - Paper metadata and processing status (includes geo_accessions column)
- `extracted_algorithms` - Algorithm mentions with parameters
- `datasets` - Dataset characteristics
- `manylatents_results` - Results from re-analysis

**Usage:**
```bash
sqlite3 data/papers/metadata/papers.db < scripts/init_database.sql
```

---

### migrate_add_geo_accessions.py ✨ **NEW**
**Purpose:** Add geo_accessions column to existing papers table

**Usage:**
```bash
python3 scripts/migrate_add_geo_accessions.py
```

**What it does:**
- Checks if `geo_accessions` column exists
- Adds column if missing (ALTER TABLE)
- Safe to run multiple times
- Displays current schema after migration

**Example Output:**
```
Migrating database: /home/btd8/llm-paper-analyze/data/papers/metadata/papers.db
Adding column 'geo_accessions'...
✓ Column 'geo_accessions' added successfully.
```

---

### add_dataset_description.sql
**Purpose:** Migration to add dataset_description field

**Usage:**
```bash
sqlite3 data/papers/metadata/papers.db < scripts/add_dataset_description.sql
```

**Changes:**
- Adds `dataset_description TEXT` column to `papers` table

---

## PubTator3 Integration Scripts (Ready but Not Yet Run)

### fetch_pubtator_annotations.py
**Purpose:** Fetch entity annotations from PubTator3 API

**Usage:**
```bash
pip install requests  # if not installed
python3 scripts/fetch_pubtator_annotations.py
```

**What it extracts:**
- Species (NCBI Taxonomy IDs)
- Cell Lines (Cellosaurus IDs)
- Genes (NCBI Gene IDs)
- Diseases (MeSH terms)
- Chemicals
- Genetic Variants

---

### enrich_datasets_from_pubtator.py
**Purpose:** Enrich dataset records with PubTator3 entities

**Usage:**
```bash
python3 scripts/enrich_datasets_from_pubtator.py
```

**Prerequisites:** Must run `fetch_pubtator_annotations.py` first

---

### run_pubtator_enrichment.sh
**Purpose:** Run PubTator3 pipeline (fetch + enrich)

**Usage:**
```bash
bash scripts/run_pubtator_enrichment.sh
```

---

## Complete Workflow

### Full Pipeline (From Scratch)

**Phase 1: Paper Mining**
```bash
# 1. Search PubMed (fetch 100 papers)
python3 scripts/search_pubmed_local.py 100

# 2. Extract datasets from abstracts
python3 scripts/extract_datasets_local.py

# 3. Extract algorithms
python3 scripts/extract_algorithms_local.py

# 4. Generate AI descriptions (requires Claude API key)
export ANTHROPIC_API_KEY="your-key-here"
python3 scripts/generate_dataset_descriptions.py --force

# 5. Fetch PMC full-text (optional, for enhanced extraction)
python3 scripts/fetch_pmc_fulltext.py

# 6. Enhanced extraction from full-text (optional)
python3 scripts/extract_from_fulltext.py

# 7. PubTator3 entity extraction (optional)
python3 scripts/fetch_pubtator_annotations.py
python3 scripts/enrich_datasets_from_pubtator.py

# 8. Export results
python3 scripts/export_database.py
```

**Phase 2: Benchmarking** (NEW)
```bash
# 9. Create synthetic scRNA-seq dataset (for testing)
python3 scripts/create_synthetic_scrna.py \
  --n-cells 10000 \
  --n-genes 3000 \
  --n-cell-types 6 \
  --sparsity 0.92 \
  --output data/synthetic/realistic_scrna.h5ad

# 10. Run benchmarks on the dataset
python3 scripts/benchmark_with_file.py data/synthetic/realistic_scrna.h5ad

# 11. (Optional) Run proof of concept benchmark
python3 scripts/simple_benchmark_v2.py

# 12. Query benchmark results
sqlite3 data/papers/metadata/papers.db \
  "SELECT algorithm_name, execution_time, n_cells, n_genes
   FROM file_benchmarks
   ORDER BY execution_time;"
```

---

### Incremental Updates (Adding More Papers)

```bash
# Fetch more papers
python3 scripts/search_pubmed_local.py 150

# Re-run extractions (only processes new papers)
python3 scripts/extract_datasets_local.py
python3 scripts/extract_algorithms_local.py

# Generate descriptions for new papers
python3 scripts/generate_dataset_descriptions.py

# Export updated database
python3 scripts/export_database.py
```

---

### Regenerate Dataset Descriptions

```bash
# Regenerate with AI (overwrites existing)
export ANTHROPIC_API_KEY="your-key-here"
python3 scripts/generate_dataset_descriptions.py --force

# Generate only for papers without descriptions
python3 scripts/generate_dataset_descriptions.py
```

---

## Configuration Files

### research_context.json
Location: `configs/research_context.json`

Defines:
- Research focus and target domain
- Search keywords and date range
- Target algorithms and frameworks
- Data extraction targets
- Quality filters

**Date Range:** 2024-01-01 to 2025-12-31 (enforced by search script)

---

### mcp_config.json
Location: `configs/mcp_config.json`

Defines:
- NCBI E-utilities settings (email, tool name)
- Database paths
- Rate limits
- Server configurations

---

## Database Queries (Quick Reference)

```sql
-- View all papers
SELECT pmid, title, publication_date, dataset_description FROM papers;

-- Papers with GEO accessions
SELECT pmid, title, geo_accession FROM papers WHERE has_geo_accession = 1;

-- Papers with GitHub repos
SELECT pmid, title, github_url FROM papers WHERE has_github = 1;

-- Algorithm distribution
SELECT algorithm_name, COUNT(*) as count
FROM extracted_algorithms
GROUP BY algorithm_name
ORDER BY count DESC;

-- Datasets by organism
SELECT organism, COUNT(*) as count
FROM datasets
GROUP BY organism
ORDER BY count DESC;

-- Papers by year
SELECT SUBSTR(publication_date, 1, 4) as year, COUNT(*)
FROM papers
GROUP BY year
ORDER BY year;

-- Dataset descriptions sample
SELECT pmid, dataset_description
FROM papers
WHERE dataset_description IS NOT NULL
LIMIT 10;

-- Benchmark results from proof of concept
SELECT algorithm_name, execution_time, embeddings_shape
FROM poc_benchmarks
ORDER BY execution_time;

-- Benchmark results from realistic datasets
SELECT dataset_name, algorithm_name, execution_time, n_cells, n_genes
FROM file_benchmarks
ORDER BY dataset_name, execution_time;

-- Compare algorithms on same dataset
SELECT algorithm_name,
       execution_time,
       n_cells,
       n_genes
FROM file_benchmarks
WHERE dataset_name = 'realistic_scrna'
ORDER BY execution_time;
```

---

## Troubleshooting

### API Rate Limits

**PubMed E-utilities:**
- Without API key: 3 requests/second
- With API key: 10 requests/second
- Add to `mcp_config.json` if you have one

**PMC E-utilities:**
- Same rate limits as PubMed
- Automatically rate-limited in scripts

### Database Locked

```bash
# Check for running processes
lsof data/papers/metadata/papers.db

# Kill processes if needed
pkill -f "python.*paper"
```

### Missing API Key

```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set for current session
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Or run without API (uses fallback)
python3 scripts/generate_dataset_descriptions.py
```

### Pre-2024 Papers in Database

The system now automatically filters out pre-2024 papers during search. If you have old papers:

```sql
-- Check for old papers
SELECT pmid, publication_date FROM papers
WHERE SUBSTR(publication_date, 1, 4) < '2024';

-- Delete if found
DELETE FROM papers WHERE SUBSTR(publication_date, 1, 4) < '2024';
```

---

## Performance Metrics

| Operation | Time (100 papers) | Notes |
|-----------|-------------------|-------|
| Search PubMed | ~2-3 minutes | Rate limited |
| Extract datasets | ~10 seconds | Fast regex |
| Extract algorithms | ~15 seconds | Regex + context |
| Generate AI descriptions | ~3-5 minutes | API calls |
| Fetch PMC full-text | ~6-8 minutes | Depends on availability |
| Export database | ~5 seconds | File operations |

---

## Best Practices

1. **Always run in order:** Search → Extract datasets → Extract algorithms → Generate descriptions
2. **Set API key as environment variable** for AI descriptions
3. **Export after each major step** to preserve results
4. **Check date range** in exported papers to verify filtering
5. **Monitor API costs** if using Claude API for large batches

---

## Current System Status

**Phase 1 (Paper Mining):**
- ✅ 98 papers in database (2024-2025)
- ✅ 98 AI-generated descriptions (100% coverage)
- ✅ 203 algorithm extraction records
- ✅ 20 dataset records
- ✅ Date filtering working correctly
- ✅ All core scripts operational

**Phase 2 (Benchmarking):**
- ✅ Proof of concept validated (6 benchmark results)
- ✅ Realistic data testing complete (3 benchmark results)
- ✅ Total benchmark results: 9
- ✅ Algorithms tested: PCA, UMAP, t-SNE
- ✅ Pipeline ready for production datasets (10k-50k cells)

---

## Contact & Support

- **User:** btd8@yale.edu
- **System:** Local Python scripts (no container required)
- **Database:** SQLite (`data/papers/metadata/papers.db`)

---

## Quick Reference

| Task | Command |
|------|---------|
| **Phase 1: Paper Mining** | |
| Search papers | `python3 scripts/search_pubmed_local.py 100` |
| Extract datasets | `python3 scripts/extract_datasets_local.py` |
| Extract algorithms | `python3 scripts/extract_algorithms_local.py` |
| Generate AI descriptions | `python3 scripts/generate_dataset_descriptions.py --force` |
| Fetch full-text | `python3 scripts/fetch_pmc_fulltext.py` |
| Export database | `python3 scripts/export_database.py` |
| View query | `python3 scripts/show_query.py` |
| View papers | `python3 scripts/view_papers.py` |
| **Phase 2: Benchmarking** | |
| Create synthetic data | `python3 scripts/create_synthetic_scrna.py --n-cells 10000 --n-genes 3000` |
| Benchmark dataset file | `python3 scripts/benchmark_with_file.py data/synthetic/realistic_scrna.h5ad` |
| Run POC benchmark | `python3 scripts/simple_benchmark_v2.py` |
| Download GEO dataset | `python3 scripts/download_geo_dataset.py GSE123456` |
