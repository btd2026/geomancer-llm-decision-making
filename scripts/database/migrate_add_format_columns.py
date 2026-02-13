#!/usr/bin/env python3
"""
Migration script to add GEO format validation columns to papers table.
Adds columns to track file format availability and download URLs.
"""
import sqlite3
from pathlib import Path

# Database path
data_dir = Path(__file__).parent.parent / 'data' / 'papers'
db_path = data_dir / 'metadata' / 'papers.db'

def migrate():
    """Add format validation columns if they don't exist."""
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("No migration needed - database will be created with new schema.")
        return

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check existing columns
        cursor.execute("PRAGMA table_info(papers)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # Columns to add
        new_columns = {
            'geo_file_format': 'TEXT',  # JSON object with format details
            'has_suitable_format': 'BOOLEAN DEFAULT 0',  # TRUE if 10x/h5ad/h5 found
            'download_urls': 'TEXT',  # JSON array of download URLs
            'format_validated_at': 'TIMESTAMP'  # When format was last validated
        }

        added_count = 0
        for col_name, col_type in new_columns.items():
            if col_name in existing_columns:
                print(f"✓ Column '{col_name}' already exists.")
            else:
                print(f"Adding column '{col_name}'...")
                cursor.execute(f"ALTER TABLE papers ADD COLUMN {col_name} {col_type}")
                added_count += 1
                print(f"✓ Column '{col_name}' added successfully.")

        if added_count > 0:
            conn.commit()
            print(f"\n✓ Added {added_count} new column(s).")
        else:
            print("\n✓ All columns already exist. No changes needed.")

        # Show current schema
        cursor.execute("PRAGMA table_info(papers)")
        print("\nCurrent papers table schema:")
        for row in cursor.fetchall():
            nullable = "NULL" if row[3] == 0 else "NOT NULL"
            default = f" DEFAULT {row[4]}" if row[4] else ""
            print(f"  {row[1]:<30} {row[2]:<15} {nullable}{default}")

    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

    print("\nMigration complete!")

if __name__ == "__main__":
    migrate()
