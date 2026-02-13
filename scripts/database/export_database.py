#!/usr/bin/env python3
"""
Export database to shareable formats.
Creates CSV files, summary report, and optionally copies the database.
"""
import json
import sqlite3
import csv
from pathlib import Path
from datetime import datetime

# Paths
project_dir = Path(__file__).parent.parent
data_dir = project_dir / 'data' / 'papers'
db_path = data_dir / 'metadata' / 'papers.db'

# Create export directory
export_dir = project_dir / 'exports'
export_dir.mkdir(exist_ok=True)

# Timestamp for this export
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
export_subdir = export_dir / f'export_{timestamp}'
export_subdir.mkdir(exist_ok=True)

def export_table_to_csv(cursor, table_name, output_path):
    """Export a table to CSV."""
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    if not rows:
        print(f"  {table_name}: No data to export")
        return 0

    # Get column names
    columns = [description[0] for description in cursor.description]

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"  {table_name}: {len(rows)} rows -> {output_path.name}")
    return len(rows)

def generate_summary_report(cursor, output_path):
    """Generate a human-readable summary report."""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("DATABASE EXPORT SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Database: {db_path}")
    report_lines.append("")

    # Papers summary
    report_lines.append("=" * 80)
    report_lines.append("PAPERS")
    report_lines.append("=" * 80)

    cursor.execute("SELECT COUNT(*) FROM papers")
    total_papers = cursor.fetchone()[0]
    report_lines.append(f"Total papers: {total_papers}")

    cursor.execute("SELECT COUNT(*) FROM papers WHERE has_geo_accession = 1")
    geo_papers = cursor.fetchone()[0]
    report_lines.append(f"Papers with GEO accessions: {geo_papers}")

    cursor.execute("SELECT COUNT(*) FROM papers WHERE has_github = 1")
    github_papers = cursor.fetchone()[0]
    report_lines.append(f"Papers with GitHub repos: {github_papers}")

    cursor.execute("SELECT COUNT(*) FROM papers WHERE methods_extracted = 1")
    methods_papers = cursor.fetchone()[0]
    report_lines.append(f"Papers with extracted algorithms: {methods_papers}")

    # Papers by year
    report_lines.append("")
    report_lines.append("Papers by year:")
    cursor.execute("""
        SELECT substr(publication_date, 1, 4) as year, COUNT(*) as count
        FROM papers
        WHERE year IS NOT NULL AND year != ''
        GROUP BY year
        ORDER BY year DESC
    """)
    for row in cursor.fetchall():
        report_lines.append(f"  {row[0]}: {row[1]}")

    # Top journals
    report_lines.append("")
    report_lines.append("Top journals:")
    cursor.execute("""
        SELECT journal, COUNT(*) as count
        FROM papers
        WHERE journal IS NOT NULL AND journal != ''
        GROUP BY journal
        ORDER BY count DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        report_lines.append(f"  {row[0][:60]}: {row[1]}")

    # Datasets summary
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DATASETS")
    report_lines.append("=" * 80)

    cursor.execute("SELECT COUNT(*) FROM datasets")
    total_datasets = cursor.fetchone()[0]
    report_lines.append(f"Total datasets: {total_datasets}")

    cursor.execute("SELECT accession_type, COUNT(*) FROM datasets GROUP BY accession_type")
    report_lines.append("")
    report_lines.append("By accession type:")
    for row in cursor.fetchall():
        report_lines.append(f"  {row[0]}: {row[1]}")

    cursor.execute("""
        SELECT organism, COUNT(*)
        FROM datasets
        WHERE organism IS NOT NULL
        GROUP BY organism
        ORDER BY COUNT(*) DESC
    """)
    organisms = cursor.fetchall()
    if organisms:
        report_lines.append("")
        report_lines.append("By organism:")
        for row in organisms:
            report_lines.append(f"  {row[0]}: {row[1]}")

    cursor.execute("""
        SELECT sequencing_platform, COUNT(*)
        FROM datasets
        WHERE sequencing_platform IS NOT NULL
        GROUP BY sequencing_platform
        ORDER BY COUNT(*) DESC
    """)
    platforms = cursor.fetchall()
    if platforms:
        report_lines.append("")
        report_lines.append("By sequencing platform:")
        for row in platforms:
            report_lines.append(f"  {row[0]}: {row[1]}")

    # Algorithms summary
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("EXTRACTED ALGORITHMS")
    report_lines.append("=" * 80)

    cursor.execute("SELECT COUNT(*) FROM extracted_algorithms")
    total_algos = cursor.fetchone()[0]
    report_lines.append(f"Total algorithm mentions: {total_algos}")

    cursor.execute("""
        SELECT algorithm_category, COUNT(*)
        FROM extracted_algorithms
        GROUP BY algorithm_category
        ORDER BY COUNT(*) DESC
    """)
    report_lines.append("")
    report_lines.append("By category:")
    for row in cursor.fetchall():
        report_lines.append(f"  {row[0]}: {row[1]}")

    cursor.execute("""
        SELECT algorithm_name, COUNT(*)
        FROM extracted_algorithms
        GROUP BY algorithm_name
        ORDER BY COUNT(*) DESC
        LIMIT 15
    """)
    report_lines.append("")
    report_lines.append("Top algorithms:")
    for row in cursor.fetchall():
        report_lines.append(f"  {row[0]}: {row[1]}")

    # Sample papers with full metadata
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("SAMPLE PAPERS (first 5 with algorithms and datasets)")
    report_lines.append("=" * 80)

    cursor.execute("""
        SELECT DISTINCT p.pmid, p.title, p.journal, p.publication_date,
               p.geo_accession, p.github_url
        FROM papers p
        LEFT JOIN datasets d ON p.id = d.paper_id
        LEFT JOIN extracted_algorithms a ON p.id = a.paper_id
        WHERE (d.id IS NOT NULL OR a.id IS NOT NULL)
        LIMIT 5
    """)

    for i, paper in enumerate(cursor.fetchall(), 1):
        report_lines.append("")
        report_lines.append(f"{i}. PMID: {paper[0]}")
        report_lines.append(f"   Title: {paper[1]}")
        report_lines.append(f"   Journal: {paper[2]} ({paper[3]})")
        if paper[4]:
            report_lines.append(f"   GEO: {paper[4]}")
        if paper[5]:
            report_lines.append(f"   GitHub: {paper[5]}")

        # Get algorithms for this paper
        cursor.execute("""
            SELECT algorithm_name, algorithm_category, parameters
            FROM extracted_algorithms
            WHERE paper_id = (SELECT id FROM papers WHERE pmid = ?)
            ORDER BY sequence_order
        """, (paper[0],))
        algos = cursor.fetchall()
        if algos:
            report_lines.append("   Algorithms:")
            for algo in algos:
                params = f" {algo[2]}" if algo[2] else ""
                report_lines.append(f"     - {algo[0]} ({algo[1]}){params}")

        # Get datasets for this paper
        cursor.execute("""
            SELECT accession_id, organism, tissue_type, sequencing_platform
            FROM datasets
            WHERE paper_id = (SELECT id FROM papers WHERE pmid = ?)
        """, (paper[0],))
        datasets = cursor.fetchall()
        if datasets:
            report_lines.append("   Datasets:")
            for ds in datasets:
                details = []
                if ds[1]: details.append(ds[1])
                if ds[2]: details.append(ds[2])
                if ds[3]: details.append(ds[3])
                detail_str = f" ({', '.join(details)})" if details else ""
                report_lines.append(f"     - {ds[0]}{detail_str}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return '\n'.join(report_lines)

def main():
    """Main execution."""
    print("=" * 80)
    print("DATABASE EXPORT")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Export to: {export_subdir}")
    print()

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Export tables to CSV
    print("Exporting tables to CSV...")
    export_table_to_csv(cursor, 'papers', export_subdir / 'papers.csv')
    export_table_to_csv(cursor, 'datasets', export_subdir / 'datasets.csv')
    export_table_to_csv(cursor, 'extracted_algorithms', export_subdir / 'extracted_algorithms.csv')

    print()
    print("Generating summary report...")
    report_text = generate_summary_report(cursor, export_subdir / 'SUMMARY.txt')

    # Also save as markdown for better viewing
    md_path = export_subdir / 'SUMMARY.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n")
    print(f"  Summary saved to: {export_subdir / 'SUMMARY.txt'}")
    print(f"  Markdown version: {md_path}")

    # Create a README
    readme_path = export_subdir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# Paper Database Export\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Contents\n\n")
        f.write("- `papers.csv` - Paper metadata (title, abstract, authors, journal, etc.)\n")
        f.write("- `datasets.csv` - Extracted dataset information (GEO accessions, organisms, platforms)\n")
        f.write("- `extracted_algorithms.csv` - Algorithm mentions from papers\n")
        f.write("- `SUMMARY.txt` - Human-readable summary report\n")
        f.write("- `papers.db` - SQLite database (can be opened with DB Browser for SQLite)\n\n")
        f.write("## Quick Stats\n\n")

        cursor.execute("SELECT COUNT(*) FROM papers")
        f.write(f"- Papers: {cursor.fetchone()[0]}\n")
        cursor.execute("SELECT COUNT(*) FROM datasets")
        f.write(f"- Datasets: {cursor.fetchone()[0]}\n")
        cursor.execute("SELECT COUNT(*) FROM extracted_algorithms")
        f.write(f"- Algorithm mentions: {cursor.fetchone()[0]}\n\n")

        f.write("## How to View\n\n")
        f.write("**CSV files:** Open with Excel, Google Sheets, or any spreadsheet software\n\n")
        f.write("**Database:** Use [DB Browser for SQLite](https://sqlitebrowser.org/) (free)\n\n")
        f.write("**Command line:**\n```bash\n")
        f.write("# View papers\n")
        f.write("sqlite3 papers.db 'SELECT title, journal FROM papers LIMIT 5;'\n\n")
        f.write("# Count by algorithm\n")
        f.write("sqlite3 papers.db 'SELECT algorithm_name, COUNT(*) FROM extracted_algorithms GROUP BY algorithm_name;'\n")
        f.write("```\n")

    print(f"  README saved to: {readme_path}")

    # Copy database file
    import shutil
    db_copy_path = export_subdir / 'papers.db'
    shutil.copy2(db_path, db_copy_path)
    print(f"\n  Database copied to: {db_copy_path}")

    conn.close()

    print()
    print("=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"Location: {export_subdir}")
    print()
    print("Files created:")
    for file in sorted(export_subdir.iterdir()):
        size = file.stat().st_size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024*1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size/(1024*1024):.1f}MB"
        print(f"  {file.name:40s} {size_str:>10s}")
    print()
    print("To share with your PI:")
    print(f"  zip -r {export_subdir.name}.zip {export_subdir.name}/")
    print()

if __name__ == "__main__":
    main()
