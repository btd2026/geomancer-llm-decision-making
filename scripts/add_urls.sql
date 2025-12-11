-- Add URL columns to papers table
ALTER TABLE papers ADD COLUMN pubmed_url TEXT;
ALTER TABLE papers ADD COLUMN doi_url TEXT;

-- Update existing papers with PubMed URLs
UPDATE papers
SET pubmed_url = 'https://pubmed.ncbi.nlm.nih.gov/' || pmid || '/'
WHERE pmid IS NOT NULL;

-- Update papers with DOI URLs
UPDATE papers
SET doi_url = 'https://doi.org/' || doi
WHERE doi IS NOT NULL;
