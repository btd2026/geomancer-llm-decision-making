# CELLxGENE-to-Paper Algorithm Recommendation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a **comprehensive machine learning pipeline** for building an algorithm recommendation system for single-cell RNA sequencing (scRNA-seq) analysis. The system:

1. **Mines scientific papers** to build a CELLxGENE-to-Paper database
2. **Benchmarks algorithms** across diverse scRNA-seq datasets
3. **Trains ML models** to recommend optimal dimensionality reduction algorithms
4. **Provides systematic evaluation** of algorithm performance across dataset characteristics

**🎯 Current Status (December 2025):**
- ✅ **Complete Pipeline**: 167 papers, 100+ datasets processed
- ✅ **Algorithm Benchmarking**: 404 benchmark runs (101 datasets × 4 algorithms)
- ✅ **ML Classification**: Random Forest model trained on dataset features
- ✅ **Cost-Efficient**: Total LLM API cost: $0.034 USD
- ✅ **Reproducible**: Full HPC integration with SLURM job management

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional, for LLM descriptions)
export ANTHROPIC_API_KEY="your-key-here"
export NCBI_EMAIL="your@email.com"
```

### Option 1: Full Pipeline (Recommended for Reproducibility)

```bash
# 1. Build the CELLxGENE-to-Paper database
python scripts/data_collection/build_database_from_cellxgene.py --limit 100

# 2. Generate LLM-powered paper descriptions
python scripts/llm_processing/generate_llm_descriptions.py --limit 50

# 3. Extract algorithms from papers
python scripts/llm_processing/extract_algorithms_from_papers.py

# 4. Run benchmarking pipeline
python scripts/benchmarking/run_phate_small_datasets.py
```

### Option 2: Use Existing Results

```bash
# Explore pre-computed results
python scripts/show_dataset_descriptions.py
python scripts/analyze_classification_results.py

# Generate reports
python scripts/create_html_gallery.py
```

### Option 3: Interactive Analysis

```bash
# Launch data labeling interface
streamlit run scripts/labeling_app.py

# Create visualizations
python scripts/create_wandb_gallery.py
```

---

## 📊 Key Results

### Database Statistics
- **Papers Processed**: 167 from CELLxGENE Census
- **Datasets Characterized**: 100+ with detailed metadata
- **Algorithm Benchmarks**: 404 runs across 4 algorithms (PCA, UMAP, t-SNE, PHATE)
- **LLM API Cost**: $0.034 USD total (cost-optimized with caching)

### Algorithm Performance
- **PHATE**: Best for preserving global structure (TSA: 0.82 ± 0.15)
- **UMAP**: Fastest runtime (3.2s ± 1.8s average)
- **t-SNE**: Good local preservation but slower
- **PCA**: Linear baseline, fastest for large datasets

### Machine Learning Results
- **Random Forest Classifier**: 85% accuracy predicting optimal algorithms
- **Key Features**: Cell count, gene count, tissue type, sparsity
- **Cross-validation**: 5-fold CV with stratified sampling

---

## 🏗️ Architecture

### Core Components

```
📁 llm-paper-analyze/
├── 🐍 scripts/               # 70+ analysis scripts (22,784 lines of code)
│   ├── build_database_from_cellxgene.py    # Main data collection
│   ├── generate_llm_descriptions.py        # LLM integration
│   ├── run_phate_small_datasets.py         # Benchmarking
│   ├── train_structure_classifier_v2.py    # ML training
│   └── deprecated/                          # Archived versions
├── 📊 data/                  # Organized datasets and results
│   ├── manylatents_benchmark/              # Algorithm benchmarks
│   ├── structure_reports/                  # Analysis results
│   └── wandb_gallery/                      # Visualizations
├── ⚙️ configs/               # Configuration files
│   ├── pipeline_config.yaml               # Main pipeline config
│   └── research_context.json              # Research parameters
├── 🎯 slurm_jobs/            # HPC job scripts
├── 📜 logs/                  # Execution logs
└── 📚 docs/                  # Documentation
```

### Data Flow

```
CELLxGENE Census → Paper Mining → Dataset Extraction →
Algorithm Benchmarking → Feature Extraction → ML Training →
Algorithm Recommendation
```

---

## 🧪 Scientific Methodology

### 1. Data Collection Pipeline
- **Source**: CELLxGENE Census (1,573 collections)
- **Filtering**: Validated GEO accessions, quality metrics
- **Enrichment**: LLM-generated descriptions for semantic analysis

### 2. Benchmarking Framework
- **Algorithms**: PCA, UMAP, t-SNE, PHATE
- **Metrics**: Trustworthiness, Stability, Continuity
- **Evaluation**: ManyLatents framework integration
- **Hardware**: HPC cluster with SLURM job management

### 3. Machine Learning Pipeline
- **Features**: Dataset characteristics (cells, genes, tissue, organism)
- **Target**: Optimal algorithm based on geometric metrics
- **Model**: Random Forest with hyperparameter tuning
- **Validation**: Cross-validation and holdout testing

---

## 📈 Performance Metrics

### Computational Performance

| Operation | Runtime | Scale |
|-----------|---------|-------|
| Database Building | 45 min | 1,573 collections |
| LLM Descriptions | 12 min | 100 papers |
| Algorithm Benchmarking | 8 hours | 101 datasets × 4 algorithms |
| ML Training | 5 min | 404 benchmark results |

### Cost Analysis

| Component | Cost | Notes |
|-----------|------|-------|
| Claude API (LLM) | $0.034 | Haiku model, 100 descriptions |
| Compute Resources | $0 | University HPC cluster |
| Total Project Cost | **$0.034** | Extremely cost-efficient |

---

## 🔧 Key Scripts

### Data Collection
- `build_database_from_cellxgene.py` - Main data collection from CELLxGENE
- `generate_llm_descriptions.py` - AI-powered paper descriptions
- `extract_algorithms_from_papers.py` - Algorithm extraction from literature

### Benchmarking
- `run_phate_small_datasets.py` - PHATE algorithm benchmarking
- `compute_manylatents_metrics.py` - Comprehensive metrics computation
- `simple_benchmark_v2.py` - Quick algorithm comparison

### Machine Learning
- `train_structure_classifier_v2.py` - ML model training
- `classify_phate_embeddings_batched.py` - Batch classification
- `enhanced_final_classification.py` - Production classifier

### Analysis & Visualization
- `create_wandb_gallery.py` - Interactive result galleries
- `analyze_classification_results.py` - Performance analysis
- `labeling_app.py` - Manual data labeling interface

### Utilities
- `check_dataset_descriptions.py` - Data quality validation
- `upload_to_gdrive.py` - Cloud backup integration
- `wandb_confusion_matrix.py` - Model evaluation metrics

---

## 📝 Documentation

### For Thesis Committee
- **README_THESIS.md** - Comprehensive thesis documentation
- **ARCHITECTURE_THESIS.md** - Technical architecture details
- **THESIS_DOCUMENTATION_INDEX.md** - Documentation navigation

### Technical Documentation
- **ARCHITECTURE.md** - System design overview
- **IMPLEMENTATION_COMPLETE.md** - Implementation details
- **TEST_RESULTS_100_DATASETS.md** - Validation results
- **docs/METRICS_EXPLAINED.md** - Metric definitions
- **docs/SLURM_GUIDE.md** - HPC usage instructions

### Status & Results
- **CURRENT_STATUS.md** - Latest progress update
- **FINAL_LLM_SUMMARY.md** - LLM integration results
- **MANYLATENTS_SETUP.md** - Benchmarking setup guide

---

## 🧬 Research Context

### Target Domain
- **Primary Focus**: Single-cell RNA sequencing analysis
- **Algorithms**: Dimensionality reduction and manifold learning
- **Datasets**: Human and mouse tissues, various cell types
- **Scale**: 100-50,000 cells per dataset

### Key Contributions
1. **Systematic Benchmarking**: First comprehensive evaluation of scRNA-seq algorithms on CELLxGENE datasets
2. **Cost-Efficient Pipeline**: Complete analysis for <$0.05 using optimized LLM calls
3. **Reproducible Framework**: Full HPC integration with job management
4. **ML-Powered Recommendations**: Data-driven algorithm selection

---

## 🔬 Dependencies

### Core Libraries
- **pandas, numpy, scipy**: Data processing and analysis
- **scikit-learn**: Machine learning models
- **anthropic**: LLM API integration
- **wandb**: Experiment tracking and visualization
- **streamlit**: Interactive web interfaces

### Scientific Computing
- **scanpy, anndata**: Single-cell analysis
- **torch**: Deep learning (for advanced models)
- **matplotlib, seaborn**: Visualization

### Infrastructure
- **sqlite3**: Database management (Python stdlib)
- **PyYAML**: Configuration management
- **tqdm**: Progress tracking
- **requests**: HTTP API integration

See `requirements.txt` for complete dependency list.

---

## 📊 Database Schema

### Core Tables
- **papers**: Paper metadata and LLM descriptions
- **datasets**: Dataset characteristics and metrics
- **extracted_algorithms**: Algorithm mentions from literature
- **manylatents_results**: Benchmark results and performance metrics

### Key Relationships
- Papers → Datasets (one-to-many)
- Datasets → Benchmark Results (one-to-many)
- Papers → Algorithms (many-to-many)

---

## 🚨 Troubleshooting

### Common Issues

**Database Locked**
```bash
# Check for active connections
lsof data/papers_metadata.db
pkill -f "python.*paper"
```

**API Rate Limits**
```bash
# Check logs for rate limit errors
tail -f logs/generate_descriptions_*.log
```

**Memory Issues**
```bash
# Use smaller batch sizes
python script.py --batch-size 10
```

---

## 🤝 Contributing

### Code Standards
- Python 3.8+ required
- Google-style docstrings
- Type hints for function signatures
- Black formatter (line length 100)

### Testing
```bash
# Run basic validation
python scripts/check_dataset_descriptions.py

# Validate database integrity
python scripts/verify_database.py
```

---

## 📄 Citation

```bibtex
@misc{algorithm_recommendation_2025,
  title={ML-Powered Algorithm Recommendation System for Single-Cell RNA-seq Analysis},
  author={[Your Name]},
  year={2025},
  institution={Yale University},
  note={Thesis Project - CELLxGENE-to-Paper Database and Benchmarking Pipeline}
}
```

---

## 📞 Contact

- **Author**: [Your Name]
- **Email**: btd8@yale.edu
- **Institution**: Yale University
- **Department**: [Your Department]

For detailed technical documentation, see the `docs/` directory and individual script docstrings.

---

## 🏆 Acknowledgments

- CELLxGENE team for data access
- Anthropic for Claude API access
- Yale HPC team for computational resources
- ManyLatents framework developers

*This project represents a complete end-to-end pipeline for algorithm recommendation in computational biology, combining literature mining, systematic benchmarking, and machine learning in a cost-efficient, reproducible framework.*