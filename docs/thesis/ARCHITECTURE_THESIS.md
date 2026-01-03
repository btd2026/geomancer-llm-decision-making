# System Architecture

**Document Version:** 1.0
**Last Updated:** December 11, 2025
**Audience:** Technical reviewers, thesis committee, future maintainers

---

## Table of Contents

1. [Overview](#overview)
2. [System Design Principles](#system-design-principles)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [API Integrations](#api-integrations)
7. [Database Design](#database-design)
8. [Computational Architecture](#computational-architecture)
9. [Error Handling &Resilience](#error-handling--resilience)
10. [Performance Optimizations](#performance-optimizations)

---

## Overview

This system implements a three-layer architecture for intelligent algorithm recommendation in single-cell RNA sequencing analysis:

1. **Data Layer:** Collection and integration of datasets, papers, and metadata
2. **Computation Layer:** Benchmarking and metric calculation using ManyLatents
3. **Intelligence Layer:** Machine learning for algorithm recommendation

The architecture prioritizes **reproducibility**, **scalability**, and **maintainability** to support rigorous scientific research.

---

## System Design Principles

### 1. Separation of Concerns

Each component has a single, well-defined responsibility:

- **Data collectors** interact with external APIs
- **Database layer** handles all persistence
- **Benchmarking engine** executes algorithms and calculates metrics
- **ML pipeline** handles feature extraction and model training

### 2. Idempotency

All operations are designed to be safely re-run:

- Database insertions use `INSERT OR IGNORE` for duplicates
- Scripts check processing flags before re-processing
- Resume capability built into all long-running operations

### 3. Explicit Over Implicit

- Configuration files over hardcoded values
- Type hints on all functions
- Comprehensive logging of all operations
- Clear error messages with actionable guidance

### 4. Fail-Fast with Graceful Degradation

- Validate inputs before processing
- Use specific exception types for different failures
- Continue processing on non-critical errors
- Log all errors for debugging

### 5. Reproducibility by Default

- Fixed random seeds for all stochastic operations
- Version tracking for all external dependencies
- Complete audit trail in logs
- Checkpoint saves for resume capability

---

## Component Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  CELLxGENE   │    │   PubMed     │    │   Claude     │             │
│  │   Census     │    │  E-utilities │    │     API      │             │
│  │  (Datasets)  │    │  (Papers)    │    │    (LLM)     │             │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘             │
│         │                   │                   │                      │
└─────────┼───────────────────┼───────────────────┼──────────────────────┘
          │                   │                   │
          v                   v                   v
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA COLLECTION LAYER                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  Python ETL Scripts                                           │     │
│  ├───────────────────────────────────────────────────────────────┤     │
│  │  • build_database_from_cellxgene.py                          │     │
│  │    - Parse CELLxGENE metadata CSV                            │     │
│  │    - Group datasets by collection (paper)                    │     │
│  │    - Fetch paper metadata via DOI → PMID                     │     │
│  │    - Store papers and datasets with linkage                  │     │
│  │                                                               │     │
│  │  • generate_llm_descriptions.py                              │     │
│  │    - Generate paper descriptions (2-3 sentences)             │     │
│  │    - Generate dataset descriptions (2-3 sentences)           │     │
│  │    - Use Claude Haiku for cost efficiency                    │     │
│  │    - Track costs and token usage                             │     │
│  │                                                               │     │
│  │  • extract_algorithms_from_papers.py                         │     │
│  │    - Fetch PMC full-text (Methods section)                   │     │
│  │    - LLM-based algorithm extraction                          │     │
│  │    - Structured JSON output with validation                  │     │
│  │    - Store in extracted_algorithms table                     │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Features:                                                               │
│  • Rate limiting with exponential backoff                                │
│  • Specific exception handling (HTTPError, URLError, JSONDecodeError)   │
│  • Transaction safety with rollback                                      │
│  • Resume capability (skip already processed records)                    │
│  • Progress tracking with tqdm                                           │
│  • Checkpoint saves every N records                                      │
│  • Rotating file logs + console output                                   │
│                                                                          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 v
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATABASE LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SQLite Database (papers.db)                                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │  ┌─────────────┐     ┌─────────────┐     ┌──────────────────┐ │    │
│  │  │   papers    │────▶│  datasets   │────▶│ manylatents_     │ │    │
│  │  │             │     │             │     │    results       │ │    │
│  │  │ • PMID      │     │ • dataset_id│     │ • Metrics        │ │    │
│  │  │ • DOI       │     │ • n_cells   │     │ • Runtime        │ │    │
│  │  │ • Title     │     │ • organism  │     │ • Algorithm      │ │    │
│  │  │ • Abstract  │     │ • tissue    │     └──────────────────┘ │    │
│  │  │ • LLM desc  │     │ • LLM desc  │              ▲           │    │
│  │  └──────┬──────┘     └─────────────┘              │           │    │
│  │         │                                          │           │    │
│  │         │         ┌──────────────────┐             │           │    │
│  │         └────────▶│  extracted_      │─────────────┘           │    │
│  │                   │  algorithms      │                         │    │
│  │                   │                  │                         │    │
│  │                   │ • Algorithm name │                         │    │
│  │                   │ • Category       │                         │    │
│  │                   │ • Parameters     │                         │    │
│  │                   │ • Context        │                         │    │
│  │                   └──────────────────┘                         │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Design Features:                                                        │
│  • Referential integrity via foreign keys                                │
│  • Indexes on frequently queried columns                                 │
│  • JSON fields for flexible metadata                                     │
│  • UNIQUE constraints on natural keys (PMID, DOI, dataset_id)          │
│  • Audit columns (created_at, updated_at)                                │
│                                                                          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 v
┌─────────────────────────────────────────────────────────────────────────┐
│                       COMPUTATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ManyLatents Benchmarking Framework                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │  Input: H5AD file (subsampled to ≤50K cells)                   │    │
│  │         Algorithm config (YAML)                                 │    │
│  │                                                                 │    │
│  │  Algorithms:                                                    │    │
│  │    ┌──────┐  ┌──────┐  ┌──────┐  ┌───────┐                    │    │
│  │    │ PCA  │  │ UMAP │  │t-SNE │  │ PHATE │                    │    │
│  │    └───┬──┘  └───┬──┘  └───┬──┘  └───┬───┘                    │    │
│  │        │         │         │         │                         │    │
│  │        └─────────┴─────────┴─────────┘                         │    │
│  │                    │                                            │    │
│  │                    v                                            │    │
│  │         ┌──────────────────────┐                               │    │
│  │         │  Metric Calculation  │                               │    │
│  │         ├──────────────────────┤                               │    │
│  │         │ • Trustworthiness    │                               │    │
│  │         │ • Continuity         │                               │    │
│  │         │ • LID                │                               │    │
│  │         │ • Participation R    │                               │    │
│  │         │ • Runtime            │                               │    │
│  │         │ • Memory             │                               │    │
│  │         └──────────┬───────────┘                               │    │
│  │                    │                                            │    │
│  │                    v                                            │    │
│  │         Output: CSV (metrics)                                   │    │
│  │                 PNG (visualization)                             │    │
│  │                 YAML (metadata)                                 │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Execution:                                                              │
│  • SLURM job arrays for parallelization                                 │
│  • 101 datasets × 4 algorithms = 404 jobs                               │
│  • Memory: 32GB per job (sufficient for 50K cells)                      │
│  • CPUs: 4 per job                                                       │
│  • Concurrent: 15 jobs                                                   │
│  • Wall time: 4 hours per job                                            │
│                                                                          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 v
┌─────────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENCE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Machine Learning Pipeline                                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │  Step 1: Feature Extraction                                     │    │
│  │  ├─ n_cells (quantitative)                                     │    │
│  │  ├─ n_genes (quantitative)                                     │    │
│  │  ├─ tissue_type (categorical, one-hot encoded)                 │    │
│  │  ├─ organism (categorical, one-hot encoded)                    │    │
│  │  └─ sparsity (computed if available)                           │    │
│  │                                                                 │    │
│  │  Step 2: Label Generation                                       │    │
│  │  ├─ For each dataset, identify best algorithm                  │    │
│  │  ├─ Criteria: highest trustworthiness_k25                      │    │
│  │  └─ Result: dataset_id → algorithm_name mapping                │    │
│  │                                                                 │    │
│  │  Step 3: Model Training                                         │    │
│  │  ├─ Algorithm: Random Forest (n_estimators=100)                │    │
│  │  ├─ Validation: 5-fold cross-validation                        │    │
│  │  ├─ Evaluation: Confusion matrix, per-class metrics            │    │
│  │  └─ Output: Trained model (pickle), feature importances        │    │
│  │                                                                 │    │
│  │  Step 4: Prediction API                                         │    │
│  │  ├─ Input: Dataset features                                    │    │
│  │  ├─ Process: Feature engineering → Model inference             │    │
│  │  └─ Output: Predicted algorithm + confidence                   │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Workflow 1: Database Population

```
1. Read CELLxGENE metadata CSV
   ├─ 1,573 rows (datasets)
   └─ Columns: dataset_id, collection_id, collection_name, DOI, n_cells, etc.

2. Group by collection_id (each collection = 1 paper)
   ├─ ~900 unique collections
   └─ 1-56 datasets per collection

3. For each collection:
   ├─ Extract DOI from citation
   ├─ Call PubMed API: DOI → PMID
   ├─ Call PubMed API: PMID → paper metadata (title, abstract, authors, etc.)
   ├─ INSERT paper into papers table (or UPDATE if exists)
   ├─ Get paper_id from database
   └─ For each dataset in collection:
       ├─ INSERT dataset into datasets table
       └─ Link to paper via paper_id foreign key

4. Generate LLM descriptions
   ├─ For papers: SELECT papers WHERE llm_description IS NULL
   ├─ Call Claude API with paper metadata
   ├─ UPDATE papers SET llm_description = result
   ├─ Repeat for datasets
   └─ Cost: ~$0.0002 per description

5. Extract algorithms
   ├─ For papers with full-text: SELECT papers WHERE has_full_text = 1
   ├─ Extract Methods section
   ├─ Call Claude API for structured extraction
   ├─ INSERT into extracted_algorithms table
   └─ Mark paper as methods_extracted = 1

Result: Fully populated database with papers, datasets, and algorithms
```

### Workflow 2: Benchmarking

```
1. Generate experiment configs
   ├─ For each dataset in database
   ├─ Create YAML config file
   └─ Specify: algorithm, parameters, input path, output path

2. Subsample large datasets (optional)
   ├─ For datasets >50K cells
   ├─ Random sample 50K cells (seed=42)
   └─ Save as new H5AD file

3. Submit SLURM job array
   ├─ One job per (dataset, algorithm) pair
   ├─ 101 datasets × 4 algorithms = 404 jobs
   ├─ Run 15 jobs concurrently
   └─ Wall time: 4 hours per job

4. Each job:
   ├─ Load H5AD file
   ├─ Run algorithm (PCA/UMAP/t-SNE/PHATE)
   ├─ Calculate metrics (trustworthiness, continuity, LID, PR)
   ├─ Save results:
   │   ├─ {algorithm}_metrics.csv
   │   ├─ {algorithm}_embedding.png
   │   └─ metrics.yaml
   └─ INSERT results into manylatents_results table

5. Aggregate results
   ├─ Collect all CSV files
   ├─ Create comparison tables
   └─ Generate summary report

Result: Comprehensive benchmark results for all dataset-algorithm pairs
```

### Workflow 3: ML Training & Prediction

```
1. Feature extraction
   ├─ SELECT dataset_id, n_cells, n_genes, organism, tissue FROM datasets
   ├─ One-hot encode categorical variables
   └─ Normalize quantitative features

2. Label generation
   ├─ SELECT dataset_id, algorithm_name, MAX(trustworthiness_k25)
   │   FROM manylatents_results GROUP BY dataset_id
   └─ Result: dataset_id → best_algorithm

3. Train-test split
   ├─ 80% training, 20% test
   └─ Stratified by best_algorithm to maintain class balance

4. Model training
   ├─ Random Forest with 100 trees
   ├─ 5-fold cross-validation
   └─ Hyperparameter tuning (optional)

5. Evaluation
   ├─ Confusion matrix
   ├─ Per-class precision, recall, F1
   ├─ Feature importance analysis
   └─ Save model to disk (pickle)

6. Prediction (inference)
   ├─ Input: New dataset features
   ├─ Load trained model
   ├─ Feature engineering (same as training)
   ├─ Model.predict(features)
   └─ Output: Recommended algorithm + confidence

Result: Trained model for automated algorithm recommendation
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Database** | SQLite | 3.30+ | Relational data storage |
| **Data Processing** | Python | 3.10+ | ETL, analysis, ML |
| **Numerical Computing** | NumPy | 1.24+ | Array operations |
| **Data Manipulation** | Pandas | 2.0+ | Dataframes |
| **Single-cell Analysis** | Scanpy | 1.9+ | scRNA-seq workflows |
| **Data Formats** | AnnData | 0.9+ | H5AD file handling |
| **Visualization** | Matplotlib, Seaborn | 3.7+, 0.12+ | Plots and heatmaps |
| **Progress Tracking** | tqdm | 4.65+ | Progress bars |
| **LLM Integration** | Anthropic SDK | 0.7+ | Claude API client |
| **Machine Learning** | scikit-learn | 1.3+ | Random Forest, metrics |
| **HPC Scheduling** | SLURM | 20.11+ | Job arrays |

### External APIs

| API | Purpose | Rate Limit | Authentication |
|-----|---------|------------|----------------|
| **CELLxGENE Census** | Dataset metadata | None | None (public) |
| **PubMed E-utilities** | Paper metadata | 3 req/s (10 with key) | Optional API key |
| **PMC API** | Full-text XML | 3 req/s | None |
| **Claude API** | LLM descriptions | 50 req/min (Haiku) | Required API key |

### Custom Frameworks

| Framework | Source | Purpose |
|-----------|--------|---------|
| **ManyLatents** | github.com/zstone12/manylatents | Algorithm benchmarking |
| **Custom metrics** | This project | Extended metric calculations |

**See:** `DATABASE_SCHEMA_THESIS.md` for complete schema documentation
**See:** `METHODOLOGY.md` for research methods and design decisions
**See:** `API_REFERENCE.md` for complete script and API documentation

---

**Document Status:** Complete
**Review Date:** December 11, 2025
**Next Review:** Upon system modifications
