#!/usr/bin/env python3
"""
Validate GEO dataset file formats to identify suitable single-cell RNA-seq data.
Checks for 10x CellRanger format (matrix.mtx, barcodes.tsv, features.tsv),
h5ad (AnnData/Scanpy), and h5 (10x HDF5) files.
"""
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime
import re

try:
    import GEOparse
    HAS_GEOPARSE = True
except ImportError:
    HAS_GEOPARSE = False
    print("WARNING: GEOparse not installed. Install with: pip install GEOparse")

# Use local paths
data_dir = Path(__file__).parent.parent / 'data' / 'papers'
db_path = data_dir / 'metadata' / 'papers.db'

# File format patterns
FORMAT_PATTERNS = {
    '10x_filtered_mtx': {
        'required': ['matrix.mtx', 'barcodes.tsv', 'features.tsv'],
        'keywords': ['filtered'],
        'description': '10x CellRanger filtered matrix format'
    },
    '10x_raw_mtx': {
        'required': ['matrix.mtx', 'barcodes.tsv'],
        'keywords': ['raw'],
        'description': '10x CellRanger raw matrix format'
    },
    'h5ad': {
        'required': ['.h5ad'],
        'keywords': [],
        'description': 'AnnData/Scanpy h5ad format'
    },
    'h5': {
        'required': ['.h5'],
        'keywords': ['filtered', 'feature', 'bc', 'matrix'],
        'description': '10x HDF5 format'
    },
    'rds': {
        'required': ['.rds'],
        'keywords': ['seurat', 'singlecell'],
        'description': 'Seurat RDS format'
    }
}

def check_file_format(filenames):
    """
    Analyze a list of supplementary filenames and identify available formats.

    Args:
        filenames: List of supplementary file names

    Returns:
        dict: Format information with keys:
            - formats: List of detected formats (e.g., ['10x_filtered_mtx', 'h5ad'])
            - files: Dict mapping format to list of matching files
            - has_suitable: Boolean indicating if any suitable format found
    """
    filenames_lower = [f.lower() for f in filenames]

    result = {
        'formats': [],
        'files': {},
        'has_suitable': False
    }

    # Check for h5ad files (highest priority)
    h5ad_files = [f for f in filenames if f.lower().endswith('.h5ad')]
    if h5ad_files:
        result['formats'].append('h5ad')
        result['files']['h5ad'] = h5ad_files
        result['has_suitable'] = True

    # Check for h5 files (10x HDF5 format)
    h5_files = [f for f in filenames if f.lower().endswith('.h5') and not f.lower().endswith('.h5ad')]
    if h5_files:
        # Filter for likely 10x files
        filtered_h5 = [f for f in h5_files if any(kw in f.lower() for kw in ['filtered', 'feature', 'bc', 'matrix', '10x'])]
        if filtered_h5:
            result['formats'].append('h5')
            result['files']['h5'] = filtered_h5
            result['has_suitable'] = True

    # Check for 10x MTX format
    has_matrix = any('matrix.mtx' in f.lower() for f in filenames_lower)
    has_barcodes = any('barcode' in f.lower() and '.tsv' in f.lower() for f in filenames_lower)
    has_features = any(('feature' in f.lower() or 'gene' in f.lower()) and '.tsv' in f.lower() for f in filenames_lower)

    if has_matrix and has_barcodes:
        # Check if filtered or raw
        filtered_files = [f for f in filenames if 'filtered' in f.lower()]
        raw_files = [f for f in filenames if 'raw' in f.lower() and 'matrix' in f.lower()]

        if filtered_files and has_features:
            result['formats'].append('10x_filtered_mtx')
            mtx_files = [f for f in filenames if any(kw in f.lower() for kw in ['matrix.mtx', 'barcode', 'feature', 'gene']) and 'filtered' in f.lower()]
            result['files']['10x_filtered_mtx'] = mtx_files
            result['has_suitable'] = True
        elif raw_files:
            result['formats'].append('10x_raw_mtx')
            mtx_files = [f for f in filenames if any(kw in f.lower() for kw in ['matrix.mtx', 'barcode', 'feature', 'gene']) and 'raw' in f.lower()]
            result['files']['10x_raw_mtx'] = mtx_files
            # Raw files are usable but not preferred
            if not result['has_suitable']:
                result['has_suitable'] = True

    # Check for RDS files (Seurat format)
    rds_files = [f for f in filenames if f.lower().endswith('.rds')]
    if rds_files:
        result['formats'].append('rds')
        result['files']['rds'] = rds_files
        # RDS is usable but requires conversion
        if not result['has_suitable']:
            result['has_suitable'] = True

    # Check for TAR archives that might contain 10x files
    tar_files = [f for f in filenames if f.lower().endswith('.tar') and 'raw' in f.lower()]
    if tar_files and not result['formats']:
        # TAR file might contain 10x files, mark as potentially suitable
        result['formats'].append('tar_archive')
        result['files']['tar_archive'] = tar_files
        result['has_suitable'] = True  # Will need to download and inspect

    return result

def get_geo_supplementary_files(geo_accession, use_geoparse=False):
    """
    Get supplementary file information for a GEO accession using NCBI API.

    Args:
        geo_accession: GEO accession ID (e.g., 'GSE220243')
        use_geoparse: If True, use GEOparse library; otherwise use NCBI web scraping

    Returns:
        dict: {
            'files': List of file names,
            'urls': List of download URLs,
            'error': Error message if failed
        }
    """
    import urllib.request
    import urllib.parse

    result = {
        'files': [],
        'urls': [],
        'error': None
    }

    try:
        # Use NCBI web page to get supplementary files (lightweight approach)
        # This avoids downloading the full GSE SOFT file
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_accession}&targ=self&form=text&view=quick"

        print(f"  Fetching metadata from NCBI...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')

        # Parse supplementary file URLs from the text
        # Format: !Series_supplementary_file = ftp://...
        for line in content.split('\n'):
            if '!Series_supplementary_file' in line or '!Sample_supplementary_file' in line:
                # Extract URL
                parts = line.split('=')
                if len(parts) >= 2:
                    file_url = parts[1].strip()
                    # Extract filename from URL
                    filename = file_url.split('/')[-1]
                    result['files'].append(filename)
                    result['urls'].append(file_url)

        # If no files found, try the FTP listing approach
        if not result['files']:
            # Construct FTP directory URL
            series_num = geo_accession.replace('GSE', '')
            series_prefix = 'GSE' + series_num[:-3] + 'nnn'
            ftp_base = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{series_prefix}/{geo_accession}/suppl/"

            # Common filenames for single-cell data
            common_files = [
                f"{geo_accession}_RAW.tar",
                f"{geo_accession}.tar",
            ]

            result['urls'].append(ftp_base)
            result['files'].extend(common_files)
            result['error'] = "Using FTP fallback - filenames are estimated"

    except Exception as e:
        result['error'] = str(e)
        print(f"  Error: {e}")

        # Last resort: provide FTP URL
        series_num = geo_accession.replace('GSE', '')
        series_prefix = 'GSE' + series_num[:-3] + 'nnn'
        ftp_url = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{series_prefix}/{geo_accession}/suppl/"
        result['urls'].append(ftp_url)
        result['files'].append(f"{geo_accession}_RAW.tar")

    return result

def validate_paper_formats(paper_id, pmid, geo_accessions_json):
    """
    Validate file formats for all GEO accessions in a paper.

    Args:
        paper_id: Database paper ID
        pmid: PubMed ID
        geo_accessions_json: JSON string of GEO accessions

    Returns:
        dict: Validation results
    """
    if not geo_accessions_json:
        return None

    geo_accessions = json.loads(geo_accessions_json)

    all_formats = []
    all_files = {}
    all_urls = []
    has_any_suitable = False

    print(f"\n[PMID: {pmid}] Checking {len(geo_accessions)} GEO accession(s): {', '.join(geo_accessions)}")

    for geo_acc in geo_accessions:
        # Only check GSE (Series) accessions - they have supplementary files
        if not geo_acc.startswith('GSE'):
            print(f"  Skipping {geo_acc} (not a GSE series)")
            continue

        print(f"\n  Checking {geo_acc}...")

        # Get supplementary files (using lightweight NCBI API, not GEOparse)
        supp_info = get_geo_supplementary_files(geo_acc, use_geoparse=False)

        if supp_info['error']:
            print(f"    Warning: {supp_info['error']}")

        if not supp_info['files']:
            print(f"    No supplementary files found")
            continue

        print(f"    Found {len(supp_info['files'])} file(s):")
        for f in supp_info['files'][:10]:  # Show first 10
            print(f"      - {f}")
        if len(supp_info['files']) > 10:
            print(f"      ... and {len(supp_info['files']) - 10} more")

        # Analyze formats
        format_info = check_file_format(supp_info['files'])

        if format_info['formats']:
            print(f"    ✓ Detected formats: {', '.join(format_info['formats'])}")
            all_formats.extend(format_info['formats'])
            all_files.update({f"{geo_acc}_{k}": v for k, v in format_info['files'].items()})
            all_urls.extend(supp_info['urls'])

            if format_info['has_suitable']:
                has_any_suitable = True
        else:
            print(f"    ❌ No suitable formats detected")

        # Rate limiting
        time.sleep(0.5)

    # Remove duplicates
    all_formats = list(set(all_formats))

    return {
        'formats': all_formats,
        'files': all_files,
        'urls': all_urls,
        'has_suitable': has_any_suitable
    }

def update_database(paper_id, validation_result):
    """Update database with validation results."""
    if validation_result is None:
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE papers SET
                geo_file_format = ?,
                has_suitable_format = ?,
                download_urls = ?,
                format_validated_at = ?
            WHERE id = ?
        """, (
            json.dumps(validation_result['files']),
            1 if validation_result['has_suitable'] else 0,
            json.dumps(validation_result['urls']),
            datetime.now().isoformat(),
            paper_id
        ))
        conn.commit()
    finally:
        conn.close()

def main():
    """Main execution."""
    print("=" * 80)
    print("GEO DATASET FORMAT VALIDATION")
    print("=" * 80)
    print(f"Database: {db_path}")

    if not HAS_GEOPARSE:
        print("\nWARNING: GEOparse not installed. Using fallback method.")
        print("Install with: pip install GEOparse\n")

    # Get papers with GEO accessions
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, pmid, title, geo_accessions
        FROM papers
        WHERE geo_accessions IS NOT NULL
        ORDER BY id
    """)

    papers = cursor.fetchall()
    conn.close()

    print(f"\nFound {len(papers)} papers with GEO accessions")
    print("=" * 80)

    stats = {
        'total': len(papers),
        'with_suitable': 0,
        'without_suitable': 0,
        'errors': 0,
        'formats_found': {}
    }

    suitable_papers = []

    for i, (paper_id, pmid, title, geo_accessions) in enumerate(papers, 1):
        print(f"\n[{i}/{len(papers)}] Paper {pmid}")
        print(f"  Title: {title[:70]}...")

        try:
            result = validate_paper_formats(paper_id, pmid, geo_accessions)

            if result:
                update_database(paper_id, result)

                if result['has_suitable']:
                    stats['with_suitable'] += 1
                    suitable_papers.append((pmid, title, result['formats']))
                    print(f"  ✓ HAS SUITABLE FORMATS: {', '.join(result['formats'])}")

                    # Count format types
                    for fmt in result['formats']:
                        stats['formats_found'][fmt] = stats['formats_found'].get(fmt, 0) + 1
                else:
                    stats['without_suitable'] += 1
                    print(f"  ❌ No suitable formats found")
            else:
                stats['without_suitable'] += 1
                print(f"  ❌ No GEO accessions to validate")

        except Exception as e:
            stats['errors'] += 1
            print(f"  ERROR: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total papers checked: {stats['total']}")
    print(f"Papers with suitable formats: {stats['with_suitable']} ({stats['with_suitable']/stats['total']*100:.1f}%)")
    print(f"Papers without suitable formats: {stats['without_suitable']}")
    print(f"Errors: {stats['errors']}")

    if stats['formats_found']:
        print("\nFormats detected:")
        for fmt, count in sorted(stats['formats_found'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {fmt}: {count} papers")

    if suitable_papers:
        print(f"\n{len(suitable_papers)} papers with suitable datasets:")
        for pmid, title, formats in suitable_papers:
            print(f"  [{pmid}] {title[:60]}...")
            print(f"    Formats: {', '.join(formats)}")

    print("\n" + "=" * 80)
    print(f"Results saved to database: {db_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
