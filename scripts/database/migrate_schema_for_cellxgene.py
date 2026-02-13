#!/usr/bin/env python3
"""
Schema migration script for CELLxGENE integration.

This script adds new columns to the papers and datasets tables to support
CELLxGENE data integration. It is idempotent - can be run multiple times
safely as it checks for column existence before adding.

New columns added:
- papers: collection_id, all_collection_ids, collection_name, source,
          llm_description, full_text, has_full_text
- datasets: dataset_id, collection_id, dataset_title, dataset_version_id,
           dataset_h5ad_path, llm_description, citation, downloaded, benchmarked

Author: Claude Code
Created: 2025-11-04
"""

import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime


def get_db_path() -> Path:
    """Get database path from project structure."""
    script_dir = Path(__file__).parent
    db_path = script_dir.parent / 'data' / 'papers' / 'metadata' / 'papers.db'

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Please run scripts/init_database.sql first."
        )

    return db_path


def get_existing_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    """
    Get list of existing column names for a table.

    Args:
        cursor: SQLite cursor
        table_name: Name of table to inspect

    Returns:
        List of column names
    """
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return columns


def add_column_if_not_exists(
    cursor: sqlite3.Cursor,
    table_name: str,
    column_name: str,
    column_definition: str
) -> bool:
    """
    Add a column to a table if it doesn't already exist.

    Args:
        cursor: SQLite cursor
        table_name: Name of table
        column_name: Name of column to add
        column_definition: Full column definition (e.g., "TEXT DEFAULT 'value'")

    Returns:
        True if column was added, False if it already existed
    """
    existing_columns = get_existing_columns(cursor, table_name)

    if column_name in existing_columns:
        return False

    # Add the column
    sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    cursor.execute(sql)
    return True


def create_index_if_not_exists(
    cursor: sqlite3.Cursor,
    index_name: str,
    table_name: str,
    column_name: str
) -> bool:
    """
    Create an index if it doesn't already exist.

    Args:
        cursor: SQLite cursor
        index_name: Name for the index
        table_name: Table to index
        column_name: Column to index

    Returns:
        True if index was created, False if it already existed
    """
    # Check if index exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,)
    )

    if cursor.fetchone():
        return False

    # Create index
    sql = f"CREATE INDEX {index_name} ON {table_name}({column_name})"
    cursor.execute(sql)
    return True


def migrate_papers_table(cursor: sqlite3.Cursor) -> Dict[str, bool]:
    """
    Add CELLxGENE-related columns to papers table.

    Args:
        cursor: SQLite cursor

    Returns:
        Dictionary mapping column names to whether they were added
    """
    print("\n" + "=" * 80)
    print("Migrating papers table")
    print("=" * 80)

    columns_to_add = [
        ("collection_id", "TEXT"),
        ("all_collection_ids", "TEXT"),  # JSON array
        ("collection_name", "TEXT"),
        ("source", "TEXT DEFAULT 'pubmed_search'"),
        ("llm_description", "TEXT"),
        ("full_text", "TEXT"),
        ("has_full_text", "INTEGER DEFAULT 0"),
    ]

    results = {}

    for column_name, column_def in columns_to_add:
        added = add_column_if_not_exists(cursor, "papers", column_name, column_def)
        results[column_name] = added

        if added:
            print(f"  ✓ Added column: {column_name}")
        else:
            print(f"  - Column already exists: {column_name}")

    return results


def migrate_datasets_table(cursor: sqlite3.Cursor) -> Dict[str, bool]:
    """
    Add CELLxGENE-related columns to datasets table.

    Args:
        cursor: SQLite cursor

    Returns:
        Dictionary mapping column names to whether they were added
    """
    print("\n" + "=" * 80)
    print("Migrating datasets table")
    print("=" * 80)

    columns_to_add = [
        ("dataset_id", "TEXT"),  # CELLxGENE dataset ID
        ("collection_id", "TEXT"),
        ("dataset_title", "TEXT"),
        ("dataset_version_id", "TEXT"),
        ("dataset_h5ad_path", "TEXT"),  # Local path to downloaded h5ad
        ("llm_description", "TEXT"),
        ("citation", "TEXT"),
        ("downloaded", "INTEGER DEFAULT 0"),
        ("benchmarked", "INTEGER DEFAULT 0"),
    ]

    results = {}

    for column_name, column_def in columns_to_add:
        added = add_column_if_not_exists(cursor, "datasets", column_name, column_def)
        results[column_name] = added

        if added:
            print(f"  ✓ Added column: {column_name}")
        else:
            print(f"  - Column already exists: {column_name}")

    return results


def create_indices(cursor: sqlite3.Cursor) -> Dict[str, bool]:
    """
    Create performance indices for new columns.

    Args:
        cursor: SQLite cursor

    Returns:
        Dictionary mapping index names to whether they were created
    """
    print("\n" + "=" * 80)
    print("Creating indices")
    print("=" * 80)

    indices = [
        ("idx_papers_doi", "papers", "doi"),
        ("idx_papers_collection_id", "papers", "collection_id"),
        ("idx_papers_source", "papers", "source"),
        ("idx_datasets_dataset_id", "datasets", "dataset_id"),
        ("idx_datasets_collection_id", "datasets", "collection_id"),
        ("idx_datasets_downloaded", "datasets", "downloaded"),
        ("idx_datasets_benchmarked", "datasets", "benchmarked"),
    ]

    results = {}

    for index_name, table_name, column_name in indices:
        created = create_index_if_not_exists(cursor, index_name, table_name, column_name)
        results[index_name] = created

        if created:
            print(f"  ✓ Created index: {index_name}")
        else:
            print(f"  - Index already exists: {index_name}")

    return results


def verify_migration(cursor: sqlite3.Cursor) -> bool:
    """
    Verify that all expected columns and indices exist.

    Args:
        cursor: SQLite cursor

    Returns:
        True if all migrations successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("Verifying migration")
    print("=" * 80)

    # Check papers columns
    papers_columns = get_existing_columns(cursor, "papers")
    expected_papers_cols = [
        "collection_id", "all_collection_ids", "collection_name",
        "source", "llm_description", "full_text", "has_full_text"
    ]

    papers_ok = all(col in papers_columns for col in expected_papers_cols)

    if papers_ok:
        print(f"  ✓ Papers table: All {len(expected_papers_cols)} new columns present")
    else:
        missing = [col for col in expected_papers_cols if col not in papers_columns]
        print(f"  ✗ Papers table: Missing columns: {missing}")

    # Check datasets columns
    datasets_columns = get_existing_columns(cursor, "datasets")
    expected_datasets_cols = [
        "dataset_id", "collection_id", "dataset_title", "dataset_version_id",
        "dataset_h5ad_path", "llm_description", "citation", "downloaded", "benchmarked"
    ]

    datasets_ok = all(col in datasets_columns for col in expected_datasets_cols)

    if datasets_ok:
        print(f"  ✓ Datasets table: All {len(expected_datasets_cols)} new columns present")
    else:
        missing = [col for col in expected_datasets_cols if col not in datasets_columns]
        print(f"  ✗ Datasets table: Missing columns: {missing}")

    # Check indices
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    existing_indices = [row[0] for row in cursor.fetchall()]

    expected_indices = [
        "idx_papers_doi", "idx_papers_collection_id", "idx_papers_source",
        "idx_datasets_dataset_id", "idx_datasets_collection_id",
        "idx_datasets_downloaded", "idx_datasets_benchmarked"
    ]

    indices_ok = all(idx in existing_indices for idx in expected_indices)

    if indices_ok:
        print(f"  ✓ Indices: All {len(expected_indices)} indices present")
    else:
        missing = [idx for idx in expected_indices if idx not in existing_indices]
        print(f"  ✗ Indices: Missing indices: {missing}")

    return papers_ok and datasets_ok and indices_ok


def print_summary(
    papers_results: Dict[str, bool],
    datasets_results: Dict[str, bool],
    indices_results: Dict[str, bool]
) -> None:
    """
    Print summary of migration changes.

    Args:
        papers_results: Results from papers table migration
        datasets_results: Results from datasets table migration
        indices_results: Results from index creation
    """
    print("\n" + "=" * 80)
    print("MIGRATION SUMMARY")
    print("=" * 80)

    papers_added = sum(1 for added in papers_results.values() if added)
    datasets_added = sum(1 for added in datasets_results.values() if added)
    indices_added = sum(1 for added in indices_results.values() if added)

    print(f"\nPapers table:")
    print(f"  - Columns added: {papers_added}")
    print(f"  - Columns already existed: {len(papers_results) - papers_added}")

    print(f"\nDatasets table:")
    print(f"  - Columns added: {datasets_added}")
    print(f"  - Columns already existed: {len(datasets_results) - datasets_added}")

    print(f"\nIndices:")
    print(f"  - Indices created: {indices_added}")
    print(f"  - Indices already existed: {len(indices_results) - indices_added}")

    total_changes = papers_added + datasets_added + indices_added

    if total_changes == 0:
        print(f"\n✓ No changes needed - schema already up to date!")
    else:
        print(f"\n✓ Migration complete - {total_changes} changes applied")

    print("=" * 80)


def main() -> int:
    """
    Main migration execution.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 80)
    print("CELLxGENE SCHEMA MIGRATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Get database path
        db_path = get_db_path()
        print(f"Database: {db_path}")

        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Run migrations
        papers_results = migrate_papers_table(cursor)
        datasets_results = migrate_datasets_table(cursor)
        indices_results = create_indices(cursor)

        # Commit changes
        conn.commit()
        print("\n✓ Changes committed to database")

        # Verify migration
        success = verify_migration(cursor)

        # Print summary
        print_summary(papers_results, datasets_results, indices_results)

        # Close connection
        conn.close()

        if not success:
            print("\n⚠ Warning: Verification found some issues")
            return 1

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1

    except sqlite3.Error as e:
        print(f"\n✗ Database error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
