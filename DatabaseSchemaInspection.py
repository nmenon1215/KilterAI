"""
Inspect the Kilter Board database schema to understand its structure.

Usage:
    python inspect_database.py --database kilter.db
"""

import sqlite3
import argparse
import pandas as pd


def inspect_database(db_path):
    """
    Inspect the structure of the SQLite database.

    Args:
        db_path: Path to the database file
    """
    print(f"Inspecting database at {db_path}...")
    conn = sqlite3.connect(db_path)

    # Get list of tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("\nTables in database:")
    for i, table in enumerate(tables):
        table_name = table[0]
        print(f"{i + 1}. {table_name}")

    print("\nDetailed schema for each table:")
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")

        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        for col in columns:
            col_id, col_name, col_type, not_null, default_val, pk = col
            print(f"  - {col_name} ({col_type}){' PRIMARY KEY' if pk else ''}")

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"  Row count: {row_count}")

        # Show sample data (first 5 rows) if table has data
        if row_count > 0:
            print("  Sample data (first 5 rows):")
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()

            # Get column names
            column_names = [col[1] for col in columns]

            # Create a DataFrame for nicer display
            sample_df = pd.DataFrame(sample_data, columns=column_names)

            # Display with max rows and max columns
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(sample_df)

    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Inspect Kilter Board database schema')
    parser.add_argument('--database', type=str, default='kilter.db', help='Path to database file')

    args = parser.parse_args()
    inspect_database(args.database)


if __name__ == "__main__":
    main()