#!/usr/bin/env python3
"""
Show the PubMed search query that will be used.
Runs on host (no container needed).
"""
import json
from pathlib import Path

# Load research context
config_dir = Path.home() / 'llm-paper-analyze' / 'configs'
with open(config_dir / 'research_context.json') as f:
    context = json.load(f)

def build_pubmed_query():
    """Build PubMed search query from research context."""
    criteria = context['search_criteria']

    # Required keywords (any)
    required = ' OR '.join([f'"{kw}"[Title/Abstract]' for kw in criteria['required_keywords_any']])

    # DR keywords
    dr_terms = criteria['required_keywords_all_from_one_category']['dimensionality_reduction']
    dr_query = ' OR '.join([f'"{kw}"[Title/Abstract]' for kw in dr_terms])

    # Date range
    date_range = criteria['date_range']
    date_filter = f'("{date_range["start"]}"[Publication Date] : "{date_range["end"]}"[Publication Date])'

    # Combine
    full_query = f'({required}) AND ({dr_query}) AND {date_filter}'

    return full_query

if __name__ == "__main__":
    print("=" * 80)
    print("PUBMED SEARCH QUERY")
    print("=" * 80)
    print()

    query = build_pubmed_query()
    print(query)
    print()

    print("=" * 80)
    print("RESEARCH CONTEXT SUMMARY")
    print("=" * 80)
    print(f"Project: {context['project_name']}")
    print(f"Domain: {context['research_focus']['primary_domain']}")
    print(f"Computational Focus: {context['research_focus']['computational_focus']}")
    print()

    print(f"Target Algorithms ({len(context['target_algorithms'])}):")
    for i, alg in enumerate(context['target_algorithms'], 1):
        print(f"  {i}. {alg}")
    print()

    print(f"Target Frameworks ({len(context['target_frameworks'])}):")
    for i, fw in enumerate(context['target_frameworks'], 1):
        print(f"  {i}. {fw}")
    print()

    print(f"Date Range: {context['search_criteria']['date_range']['start']} to {context['search_criteria']['date_range']['end']}")
    print()

    print("=" * 80)
    print("This query will search for papers about single-cell RNA-seq that use")
    print("dimensionality reduction methods, published between 2024-2025.")
    print("=" * 80)
