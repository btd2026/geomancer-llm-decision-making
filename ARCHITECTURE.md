# Three-Repo Architecture: Building the Algorithm Recommendation System

**Date**: October 27, 2025
**Status**: Data Collection Phase (GEO-validated datasets)

---

## 🎯 The Big Picture

You have **three interconnected repositories** that work together to build a dataset-driven algorithm recommendation system for scRNA-seq analysis:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR RESEARCH PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

   📚 llm-paper-analyze          🔬 manylatents              🤖 manyagents
   (This repo)                   (Algorithm framework)       (Orchestrator)
   ─────────────────             ───────────────────         ─────────────

   1. Mine papers                2. Run algorithms           3. Coordinate
   2. Extract datasets           3. Compute metrics          4. Orchestrate
   3. Generate configs    ────>  4. Return results    <────  5. Store results
   4. Train ML model
   5. Deploy recommender

                              ┌──────────────┐
                              │   WORKFLOW   │
                              └──────────────┘

   Extract Datasets     →    Benchmark Them    →    Train Model
   (paper mining)            (manylatents)          (ML training)
        ↓                         ↓                       ↓
    10 GEO IDs            embeddings + metrics     recommender API
```

---

## 📦 Repository Roles

### **1. llm-paper-analyze** (Knowledge Extraction → Training Data)

**Location**: `/home/btd8/llm-paper-analyze/`
**Purpose**: Mine scientific literature to extract datasets and create benchmark specifications

**Key Responsibilities**:
- ✅ Search PubMed for scRNA-seq papers
- ✅ **Validate GEO datasets** via NCBI API (mandatory)
- ✅ Extract dataset characteristics (GEO IDs, cell counts, tissue types)
- ✅ Extract algorithm mentions (PCA, UMAP, VAE, scVelo, etc.)
- ✅ **Trajectory analysis** support (pseudotime, RNA velocity)
- ✅ Generate AI descriptions of datasets
- ⏭ **Create benchmark configuration files** (what to test)
- ⏭ **Store benchmark results** (from manyagents)
- ⏭ **Train ML recommendation model**

**Database** (`data/papers/metadata/papers.db`):
```sql
papers                     -- 27 papers with validated GEO datasets
  ├─ geo_accessions        -- JSON array of validated accessions (NEW)
extracted_algorithms       -- Algorithm extraction (from previous runs)
datasets                   -- Dataset records (from previous runs)
manylatents_results        -- Benchmark results (TO BE POPULATED)
```

**Key Output**: Benchmark specification files → sent to manyagents

---

### **2. manylatents** (Algorithm Execution Engine)

**Location**: `/home/btd8/manylatents/`
**Purpose**: Unified framework for running dimensionality reduction algorithms

**Technology**:
- PyTorch Lightning (neural networks)
- Hydra (configuration management)
- Python API for programmatic access

**Supported Algorithms**:
- **Traditional DR**: PCA, UMAP, t-SNE, PHATE
- **Neural Networks**: Autoencoders, VAEs
- **Sequential Pipelines**: PCA → Autoencoder, etc.

**Key Function** (`manylatents/api.py`):
```python
from manylatents.api import run

# Single algorithm execution
result = run(
    data='swissroll',  # or input_data=numpy_array
    algorithms={'latent': {'_target_': '...PCA', 'n_components': 50}}
)

# Returns:
# {
#   'embeddings': numpy array,
#   'scores': {metric_name: value},
#   'metadata': {...}
# }
```

**Metrics Computed**:
- Trustworthiness
- Continuity
- Participation Ratio
- Fractal Dimension
- (Custom metrics via config)

**Key Output**: Embeddings + metrics → returned to manyagents

---

### **3. manyagents** (Orchestration & Workflow Management)

**Location**: `/home/btd8/manyagents/`
**Purpose**: Orchestrate complex multi-step workflows and coordinate tool execution

**Technology**:
- Direct Python API integration (no subprocess overhead)
- Async workflow execution
- In-memory data passing between steps
- SLURM job submission support

**Key Components**:

**ManyLatentsAdapter** (`manyagents/adapters/`):
```python
from manyagents.adapters import ManyLatentsAdapter

adapter = ManyLatentsAdapter()

# Run single algorithm
result = await adapter.run(
    task_config={
        "algorithm": "pca",
        "data": "swissroll",
        "n_components": 50
    },
    input_files={}
)

# Sequential workflow: PCA → UMAP
result1 = await adapter.run(
    task_config={"algorithm": "pca", "n_components": 50},
    input_files={}
)

result2 = await adapter.run(
    task_config={"algorithm": "umap", "n_components": 2},
    input_data=result1['output_files']['embeddings']
)
```

**Config Map** (`config_map.yaml`):
Maps high-level names to manylatents configs:
```yaml
algorithms:
  PCA:
    path: algorithms/latent=pca
  UMAP:
    path: algorithms/latent=umap
  TSNE:
    path: algorithms/latent=tsne
```

**Key Output**: Orchestrated results → written to llm-paper-analyze database

---

## 🔄 The Complete Data Flow

### **Phase 1: Paper Mining** (COMPLETE ✅)

```
User Request
    ↓
llm-paper-analyze/scripts/search_pubmed_local.py
    ↓
PubMed API
    ↓
papers.db (98 papers)
```

**Status**: ✅ Complete
**Output**: 98 papers, 10 with GEO accessions

---

### **Phase 2: Dataset Extraction** (IN PROGRESS ⏭)

```
papers.db
    ↓
extract_datasets_local.py
    ↓
datasets table (20 records)
    ↓
[NEW] extract_dataset_features_ai.py
    ↓
Enhanced dataset metadata:
  - n_cells, n_genes, sparsity
  - organism, tissue_type
  - GEO accession, download URLs
```

**Status**: ⏭ Needs enhancement
**Output**: Comprehensive dataset catalog with downloadable GEO accessions

---

### **Phase 3: Benchmark Specification** (TO DO 📝)

```
datasets table
    ↓
[NEW] generate_benchmark_configs.py
    ↓
Creates JSON files for each dataset:

configs/benchmarks/GSE123456.json:
{
  "dataset_id": "GSE123456",
  "geo_accession": "GSE123456",
  "characteristics": {
    "n_cells": 25000,
    "n_genes": 20000,
    "organism": "human",
    "tissue": "PBMC"
  },
  "algorithms_to_test": [
    {"name": "PCA", "params": {"n_components": 50}},
    {"name": "UMAP", "params": {"n_components": 2}},
    {"name": "TSNE", "params": {"n_components": 2}},
    {"name": "VAE", "params": {"latent_dim": 10}},
    {"name": "Autoencoder", "params": {"latent_dim": 10}}
  ],
  "metrics": ["trustworthiness", "continuity", "pr", "lid"]
}
```

**Status**: 📝 To be built
**Output**: Benchmark specification files → consumed by manyagents

---

### **Phase 4: Benchmark Execution** (TO DO 🔬)

This is where **manyagents** orchestrates **manylatents**.

```
Step 1: Download Datasets
─────────────────────────
llm-paper-analyze/scripts/download_geo_datasets.py
    ↓
Downloads from GEO to: data/geo/GSE123456/
    ↓
Converts to AnnData format (.h5ad)
    ↓
Stores in: data/geo/processed/GSE123456.h5ad


Step 2: Run Benchmarks (manyagents orchestrates manylatents)
──────────────────────────────────────────────────────────────
[NEW] llm-paper-analyze/scripts/run_benchmarks.py
    ↓
Reads: configs/benchmarks/*.json
    ↓
For each dataset × algorithm combination:
    ↓
    Calls manyagents API:
    ┌────────────────────────────────────────────────┐
    │  from manyagents.adapters import               │
    │      ManyLatentsAdapter                        │
    │                                                │
    │  adapter = ManyLatentsAdapter()                │
    │                                                │
    │  for each_dataset:                             │
    │      data = load_h5ad(dataset_path)            │
    │                                                │
    │      for each_algorithm:                       │
    │          result = await adapter.run(           │
    │              task_config={                     │
    │                  "algorithm": algo_name,       │
    │                  "n_components": params        │
    │              },                                │
    │              input_data=data                   │
    │          )                                     │
    │                                                │
    │          # Store in database                   │
    │          store_result(dataset, algo, result)   │
    └────────────────────────────────────────────────┘
            ↓
    manyagents calls manylatents.api.run()
            ↓
    manylatents executes algorithm
            ↓
    Returns: embeddings, scores, metadata
            ↓
    Store in papers.db → manylatents_results table


Step 3: Store Results
──────────────────────
INSERT INTO manylatents_results (
    dataset_id,
    algorithm_id,
    tsa,
    lid,
    pr,
    trustworthiness,
    continuity,
    execution_time,
    memory_usage,
    algorithm_parameters
) VALUES (...);
```

**Status**: 📝 To be built
**Output**: Populated `manylatents_results` table with benchmark data

---

### **Phase 5: Model Training** (TO DO 🤖)

```
manylatents_results table
    ↓
[NEW] train_recommender.py
    ↓
Feature Engineering:
  X = [n_cells, n_genes, sparsity, tissue, organism, ...]
  Y = [TSA, LID, PR, runtime, memory]
    ↓
Train ML Models:
  - Random Forest Classifier (best algorithm)
  - Gradient Boosting Regressor (predict TSA per algo)
  - Multi-task Neural Network (predict all metrics)
    ↓
Save trained model:
  models/algorithm_recommender.pkl
```

**Status**: 📝 To be built
**Output**: Trained ML model

---

### **Phase 6: Deployment** (TO DO 🚀)

```
[NEW] recommend_algorithm.py --dataset my_data.h5ad

Analyzes dataset characteristics
    ↓
Loads trained model
    ↓
Predicts best algorithms
    ↓
Returns ranked recommendations:

1. UMAP (predicted TSA: 0.87, runtime: 52s)
2. VAE (predicted TSA: 0.82, runtime: 180s)
3. PCA (predicted TSA: 0.71, runtime: 8s)
```

**Status**: 📝 To be built
**Output**: CLI/API for algorithm recommendations

---

## 🗂️ File Organization

### **llm-paper-analyze/** (Current Repo)

```
llm-paper-analyze/
├── data/
│   ├── papers/
│   │   └── metadata/
│   │       └── papers.db              # Main database
│   └── geo/                           # [NEW] Downloaded datasets
│       ├── raw/                       # Original GEO files
│       └── processed/                 # .h5ad files for manylatents
│
├── configs/
│   └── benchmarks/                    # [NEW] Benchmark specifications
│       ├── GSE123456.json
│       ├── GSE789012.json
│       └── ...
│
├── scripts/
│   ├── search_pubmed_local.py         # ✅ Complete
│   ├── extract_algorithms_local.py    # ✅ Complete
│   ├── extract_datasets_local.py      # ✅ Basic version
│   ├── generate_dataset_descriptions.py # ✅ Complete
│   │
│   ├── extract_dataset_features_ai.py # 📝 TO BUILD - Enhanced extraction
│   ├── download_geo_datasets.py       # 📝 TO BUILD - GEO downloader
│   ├── generate_benchmark_configs.py  # 📝 TO BUILD - Config generator
│   ├── run_benchmarks.py              # 📝 TO BUILD - Orchestrates manyagents
│   ├── train_recommender.py           # 📝 TO BUILD - ML training
│   └── recommend_algorithm.py         # 📝 TO BUILD - CLI tool
│
├── models/                            # [NEW] Trained ML models
│   ├── algorithm_recommender.pkl
│   └── feature_scaler.pkl
│
└── docs/
    ├── ARCHITECTURE.md                # This file
    ├── RECOMMENDATION_SYSTEM_PLAN.md  # ✅ Complete
    └── API_USAGE.md                   # 📝 TO BUILD
```

---

## 🔌 Integration Points

### **1. llm-paper-analyze → manyagents**

**Interface**: Benchmark configuration files (JSON)

**Location**: `llm-paper-analyze/configs/benchmarks/*.json`

**Schema**:
```json
{
  "dataset_id": "GSE123456",
  "geo_accession": "GSE123456",
  "data_path": "/home/btd8/llm-paper-analyze/data/geo/processed/GSE123456.h5ad",
  "characteristics": {
    "n_cells": 25000,
    "n_genes": 20000,
    "sparsity": 0.92,
    "organism": "human",
    "tissue": "PBMC"
  },
  "algorithms_to_test": [
    {
      "name": "PCA",
      "manylatents_config": {"algorithm": "pca", "n_components": 50}
    },
    {
      "name": "UMAP",
      "manylatents_config": {"algorithm": "umap", "n_components": 2}
    }
  ]
}
```

**Consumer**: `llm-paper-analyze/scripts/run_benchmarks.py` reads these files and calls manyagents

---

### **2. manyagents → manylatents**

**Interface**: Python API (direct function calls)

**Code Pattern**:
```python
# In llm-paper-analyze/scripts/run_benchmarks.py

from manyagents.adapters import ManyLatentsAdapter
import anndata as ad

adapter = ManyLatentsAdapter()

# Load dataset
adata = ad.read_h5ad("data/geo/processed/GSE123456.h5ad")

# Run algorithm
result = await adapter.run(
    task_config={
        "algorithm": "pca",
        "n_components": 50
    },
    input_data=adata.X.toarray()  # Pass numpy array
)

# Extract results
embeddings = result['output_files']['embeddings']
scores = result['output_files']['scores']
metadata = result['metadata']
```

---

### **3. manyagents → llm-paper-analyze**

**Interface**: Direct database writes

**Code Pattern**:
```python
# In llm-paper-analyze/scripts/run_benchmarks.py

import sqlite3

conn = sqlite3.connect('data/papers/metadata/papers.db')

conn.execute("""
    INSERT INTO manylatents_results (
        dataset_id, algorithm_id,
        tsa, lid, pr, trustworthiness, continuity,
        execution_time, memory_usage,
        algorithm_parameters
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    dataset_id, algorithm_id,
    scores['tsa'], scores['lid'], scores['pr'],
    scores['trustworthiness'], scores['continuity'],
    metadata['execution_time'], metadata['memory_usage'],
    json.dumps(task_config)
))

conn.commit()
```

---

## 🚀 Implementation Roadmap

### **Week 1: Enhanced Data Extraction**

**Goal**: Extract comprehensive dataset features

**Tasks**:
- [ ] Create `extract_dataset_features_ai.py` using Claude API
- [ ] Extract: n_cells, n_genes, sparsity, organism, tissue, platform
- [ ] Update database with enhanced metadata
- [ ] Verify 10 GEO datasets have complete metadata

**Deliverable**: Enhanced `datasets` table with all features needed for ML

---

### **Week 2: Dataset Download Pipeline**

**Goal**: Download and preprocess datasets from GEO

**Tasks**:
- [ ] Create `download_geo_datasets.py`
- [ ] Use GEOparse/scanpy to download datasets
- [ ] Convert to standardized AnnData format (.h5ad)
- [ ] Store in `data/geo/processed/`
- [ ] Add quality control checks

**Deliverable**: 10 preprocessed datasets ready for benchmarking

---

### **Week 3: Benchmark Configuration**

**Goal**: Create benchmark specifications

**Tasks**:
- [ ] Create `generate_benchmark_configs.py`
- [ ] Generate JSON configs for each dataset
- [ ] Define algorithm suite (PCA, UMAP, t-SNE, VAE, Autoencoder)
- [ ] Define metrics to compute
- [ ] Store in `configs/benchmarks/`

**Deliverable**: 10 benchmark config files

---

### **Week 4-5: Benchmark Execution**

**Goal**: Run all dataset × algorithm combinations

**Tasks**:
- [ ] Create `run_benchmarks.py`
- [ ] Integrate with manyagents adapter
- [ ] Add progress tracking and logging
- [ ] Handle errors and retries
- [ ] Store results in `manylatents_results` table
- [ ] Run on HPC/SLURM if available

**Deliverable**: Populated `manylatents_results` table (10 datasets × 5 algorithms = 50+ benchmark runs)

---

### **Week 6: Model Training**

**Goal**: Train ML recommendation model

**Tasks**:
- [ ] Create `train_recommender.py`
- [ ] Feature engineering (dataset characteristics → ML features)
- [ ] Train multiple model types (RF, GBM, Neural Net)
- [ ] Cross-validation and hyperparameter tuning
- [ ] Save trained models

**Deliverable**: Trained algorithm recommendation model

---

### **Week 7: Deployment & Testing**

**Goal**: Deploy recommendation system

**Tasks**:
- [ ] Create `recommend_algorithm.py` CLI tool
- [ ] Create API endpoint (Flask/FastAPI)
- [ ] Test on held-out datasets
- [ ] Document usage
- [ ] Create Jupyter notebook demo

**Deliverable**: Working recommendation system

---

## 🧪 Testing Strategy

### **Unit Tests**

```bash
# Test individual components
pytest llm-paper-analyze/tests/test_dataset_extraction.py
pytest llm-paper-analyze/tests/test_geo_download.py
pytest llm-paper-analyze/tests/test_benchmark_config.py
```

### **Integration Tests**

```bash
# Test manyagents ↔ manylatents integration
pytest llm-paper-analyze/tests/test_manyagents_integration.py
```

**Example**:
```python
async def test_single_algorithm_execution():
    """Test running PCA via manyagents."""
    adapter = ManyLatentsAdapter()

    result = await adapter.run(
        task_config={"algorithm": "pca", "n_components": 2},
        input_data=np.random.randn(100, 50)
    )

    assert result['success'] == True
    assert result['output_files']['embeddings'].shape == (100, 2)
    assert 'scores' in result['output_files']
```

### **End-to-End Test**

```bash
# Test full pipeline on 1 dataset
python scripts/run_benchmarks.py --dataset GSE123456 --algorithms PCA,UMAP --dry-run
```

---

## 📊 Expected Outputs

### **After Benchmark Execution**

**Database State**:
```sql
SELECT
    d.geo_accession,
    a.algorithm_name,
    r.tsa,
    r.execution_time
FROM manylatents_results r
JOIN datasets d ON r.dataset_id = d.id
JOIN extracted_algorithms a ON r.algorithm_id = a.id
ORDER BY d.geo_accession, r.tsa DESC;
```

**Example Output**:
```
GSE123456 | UMAP          | 0.87 | 52.3s
GSE123456 | VAE           | 0.82 | 180.1s
GSE123456 | PCA           | 0.71 | 8.2s
GSE123456 | t-SNE         | 0.68 | 210.5s
GSE123456 | Autoencoder   | 0.85 | 95.3s
...
```

### **After Model Training**

**Model Performance**:
```
Algorithm Recommendation Accuracy: 73.5%
Top-3 Recommendation Accuracy: 91.2%
TSA Prediction RMSE: 0.08
Runtime Prediction RMSE: 12.3s
```

### **Recommendation Example**

```bash
$ python recommend_algorithm.py --dataset my_data.h5ad

Dataset Characteristics:
  Cells: 35,000
  Genes: 18,500
  Sparsity: 89%
  Organism: Human
  Tissue: Brain

Recommended Algorithms (ranked by predicted TSA):

1. UMAP
   Predicted TSA: 0.87 ± 0.04
   Predicted Runtime: 52s
   Confidence: High
   Best for: Preserving global structure in large datasets

2. VAE
   Predicted TSA: 0.82 ± 0.06
   Predicted Runtime: 180s
   Confidence: Medium
   Best for: Noise robustness, probabilistic modeling

3. PCA
   Predicted TSA: 0.71 ± 0.03
   Predicted Runtime: 8s
   Confidence: High
   Best for: Speed, interpretability, linear relationships
```

---

## 🔧 Development Environment Setup

### **Prerequisites**

```bash
# 1. Ensure all three repos are cloned
ls /home/btd8/
# Should show: llm-paper-analyze/ manylatents/ manyagents/

# 2. Install manylatents
cd /home/btd8/manylatents
uv sync
source .venv/bin/activate
pip install -e .

# 3. Install manyagents
cd /home/btd8/manyagents
uv sync
source .venv/bin/activate
pip install -e .

# 4. Install llm-paper-analyze dependencies
cd /home/btd8/llm-paper-analyze
pip install -r requirements.txt  # Create this if needed
```

### **Python Path Configuration**

Add to `llm-paper-analyze/scripts/run_benchmarks.py`:
```python
import sys
sys.path.append('/home/btd8/manylatents')
sys.path.append('/home/btd8/manyagents')
```

Or set `PYTHONPATH`:
```bash
export PYTHONPATH="/home/btd8/manylatents:/home/btd8/manyagents:$PYTHONPATH"
```

---

## 🎯 Success Metrics

### **Technical Metrics**

- ✅ **Benchmark Coverage**: 10 datasets × 5-8 algorithms = 50-80 runs
- ✅ **Model Accuracy**: >70% algorithm recommendation accuracy
- ✅ **TSA Prediction**: RMSE < 0.1 for metric predictions
- ✅ **Runtime**: Full benchmark suite completes in <24 hours

### **Scientific Metrics**

- ✅ **Discovery Rate**: Model recommends non-PCA algorithms >50% of time
- ✅ **Performance Gain**: Average TSA improvement vs always-PCA baseline
- ✅ **Generalization**: Model works on unseen datasets
- ✅ **Reproducibility**: All results reproducible with fixed seeds

---

## 📚 Key Resources

### **Documentation**

- **manylatents README**: `/home/btd8/manylatents/README.md`
- **manylatents API**: `/home/btd8/manylatents/manylatents/api.py`
- **manyagents README**: `/home/btd8/manyagents/README.md`
- **manyagents Examples**: `/home/btd8/manyagents/example_workflows.py`

### **Code Examples**

**Running manylatents directly**:
```python
from manylatents.api import run

result = run(
    data='swissroll',
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 10}}
)
```

**Running via manyagents**:
```python
from manyagents.adapters import ManyLatentsAdapter

adapter = ManyLatentsAdapter()
result = await adapter.run(
    task_config={"algorithm": "pca", "n_components": 10},
    input_data=data_array
)
```

---

## ❓ FAQ

### **Q: Why three separate repos?**

**A**: Separation of concerns:
- **llm-paper-analyze**: Knowledge extraction + ML training (research pipeline)
- **manylatents**: Reusable algorithm framework (scientific tool)
- **manyagents**: General orchestration framework (infrastructure)

This allows each component to evolve independently and be reused in other projects.

---

### **Q: Can I run benchmarks without manyagents?**

**A**: Yes! You can call manylatents directly:

```python
from manylatents.api import run

result = run(data='swissroll', algorithms={'latent': 'pca'})
```

But manyagents provides:
- Async orchestration (run multiple benchmarks in parallel)
- Progress tracking and logging
- Error handling and retries
- SLURM job submission
- Result aggregation

---

### **Q: How do I add a new algorithm?**

**A**: Three steps:

1. **Add to manylatents** (if not already supported)
2. **Update config_map.yaml** in manyagents:
   ```yaml
   algorithms:
     MY_ALGO:
       path: algorithms/latent=myalgo
       base: algorithms=latent_base
   ```
3. **Add to benchmark configs** in llm-paper-analyze:
   ```json
   {
     "name": "MY_ALGO",
     "manylatents_config": {"algorithm": "myalgo", "param": "value"}
   }
   ```

---

### **Q: What if a benchmark fails?**

**A**: The system should:
1. Log the error
2. Continue with other benchmarks
3. Mark the result as failed in the database
4. Optionally retry with different parameters

Implement retry logic in `run_benchmarks.py`:
```python
try:
    result = await adapter.run(task_config, input_data=data)
except Exception as e:
    logger.error(f"Benchmark failed: {e}")
    # Store failure in database
    # Continue with next benchmark
```

---

## 🎬 Next Steps

**Immediate**:
1. Review this architecture document
2. Confirm it aligns with your vision
3. Choose which phase to start with

**Recommended Starting Point**:
1. **Test integration** - Verify manyagents can call manylatents
2. **Download 1 dataset** - Test GEO download pipeline
3. **Run 1 benchmark** - Test full workflow with 1 dataset × 3 algorithms
4. **Scale up** - Once working, scale to 10 datasets × 5 algorithms

**Quick Integration Test**:
```bash
cd /home/btd8/llm-paper-analyze

# Create simple test script
cat > test_integration.py << 'EOF'
import asyncio
import numpy as np
from manyagents.adapters import ManyLatentsAdapter

async def test():
    adapter = ManyLatentsAdapter()

    # Test data
    data = np.random.randn(100, 50)

    # Test PCA
    result = await adapter.run(
        task_config={"algorithm": "pca", "n_components": 2},
        input_data=data
    )

    print(f"Success: {result['success']}")
    print(f"Embeddings shape: {result['output_files']['embeddings'].shape}")

asyncio.run(test())
EOF

python test_integration.py
```

---

**Ready to build? Let's start with the integration test!**
