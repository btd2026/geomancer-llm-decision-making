#!/bin/bash
# Run PubTator3 enrichment pipeline
# 1. Fetch annotations from PubTator3 API
# 2. Enrich dataset records with entity information

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "PUBTATOR3 ENRICHMENT PIPELINE"
echo "================================================================================"
echo ""

# Step 1: Fetch PubTator3 annotations
echo "Step 1: Fetching PubTator3 annotations..."
echo ""
python3 "$SCRIPT_DIR/fetch_pubtator_annotations.py"

if [ $? -ne 0 ]; then
    echo "Error: Failed to fetch PubTator3 annotations"
    exit 1
fi

echo ""
echo "================================================================================"
echo ""

# Step 2: Enrich datasets
echo "Step 2: Enriching dataset records..."
echo ""
python3 "$SCRIPT_DIR/enrich_datasets_from_pubtator.py"

if [ $? -ne 0 ]; then
    echo "Error: Failed to enrich datasets"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  - Review enriched data: python3 scripts/explore_papers.py"
echo "  - Export database: python3 scripts/export_database.py"
echo ""
