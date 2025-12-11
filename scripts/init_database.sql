-- Papers table
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT UNIQUE,  -- PubMed ID
    doi TEXT,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    journal TEXT,
    publication_date TEXT,
    keywords TEXT,  -- JSON array
    mesh_terms TEXT,  -- JSON array

    -- Data availability
    has_geo_accession BOOLEAN DEFAULT 0,
    geo_accession TEXT,
    geo_accessions TEXT,  -- JSON array of validated GEO accessions (GSE, GSM, etc.)
    has_github BOOLEAN DEFAULT 0,
    github_url TEXT,
    data_availability_statement TEXT,
    dataset_description TEXT,  -- Concise description of datasets used in the paper

    -- Processing status
    pdf_downloaded BOOLEAN DEFAULT 0,
    pdf_path TEXT,
    text_extracted BOOLEAN DEFAULT 0,
    text_path TEXT,
    methods_extracted BOOLEAN DEFAULT 0,

    -- Relevance scores
    relevance_score REAL,
    citation_count INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted algorithms table
CREATE TABLE IF NOT EXISTS extracted_algorithms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,

    -- Algorithm details
    algorithm_name TEXT NOT NULL,
    algorithm_category TEXT,  -- 'dimensionality_reduction', 'clustering', 'normalization'
    parameters TEXT,  -- JSON object
    sequence_order INTEGER,  -- Order in pipeline

    -- Context
    mentioned_in_section TEXT,  -- 'methods', 'results', 'discussion'
    context_text TEXT,

    -- Validation
    extraction_method TEXT,  -- 'regex', 'llm', 'manual'
    confidence_score REAL,

    FOREIGN KEY (paper_id) REFERENCES papers (id)
);

-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,

    -- Dataset characteristics
    dataset_name TEXT,
    organism TEXT,
    tissue_type TEXT,
    condition TEXT,
    n_cells INTEGER,
    n_genes INTEGER,
    sequencing_platform TEXT,

    -- Accession info
    accession_type TEXT,  -- 'GEO', 'SRA', 'ArrayExpress', 'zenodo'
    accession_id TEXT,

    -- Processing
    preprocessing_steps TEXT,  -- JSON array
    normalization_method TEXT,

    FOREIGN KEY (paper_id) REFERENCES papers (id)
);

-- ManyLatents results table (for re-analysis)
CREATE TABLE IF NOT EXISTS manylatents_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    algorithm_name TEXT NOT NULL,

    -- Geometric metrics
    tsa REAL,  -- Trust Surface Area
    lid REAL,  -- Local Intrinsic Dimensionality
    pr REAL,   -- Participation Ratio
    trustworthiness REAL,
    continuity REAL,

    -- Runtime metrics
    execution_time_seconds REAL,
    memory_peak_mb REAL,

    -- Parameters used
    parameters TEXT,  -- JSON object
    manylatents_version TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pmid);
CREATE INDEX IF NOT EXISTS idx_papers_relevance ON papers(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_algorithms_name ON extracted_algorithms(algorithm_name);
CREATE INDEX IF NOT EXISTS idx_datasets_accession ON datasets(accession_id);
