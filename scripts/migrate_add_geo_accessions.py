#!/usr/bin/env python3
"""
Migration script to add geo_accessions column to existing papers table.
"""
import sqlite3
from pathlib import Path

# Database path
data_dir = Path(__file__).parent.parent / 'data' / 'papers'
db_path = data_dir / 'metadata' / 'papers.db'

def migrate():
    """Add geo_accessions column if it doesn't exist."""
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("No migration needed - database will be created with new schema.")
        return

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(papers)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'geo_accessions' in columns:
            print("✓ Column 'geo_accessions' already exists. No migration needed.")
        else:
            print("Adding column 'geo_accessions'...")
            cursor.execute("ALTER TABLE papers ADD COLUMN geo_accessions TEXT")
            conn.commit()
            print("✓ Column 'geo_accessions' added successfully.")

        # Show column info
        cursor.execute("PRAGMA table_info(papers)")
        print("\nCurrent papers table schema:")
        for row in cursor.fetchall():
            print(f"  {row[1]} ({row[2]})")

    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

    print("\nMigration complete!")

if __name__ == "__main__":
    migrate()
