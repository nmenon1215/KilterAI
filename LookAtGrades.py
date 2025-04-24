"""
View the full difficulty_grades table from the Kilter Board database.

This script displays all rows in the difficulty_grades table to help understand
the mapping between numerical difficulty values and boulder grades.
"""

import sqlite3
import pandas as pd

# Connect to the database
db_path = '../kilter.db'  # Update this path if your database is stored elsewhere
conn = sqlite3.connect(db_path)

# Query the full difficulty_grades table
query = "SELECT * FROM difficulty_grades ORDER BY difficulty"

# Load into a DataFrame for nicer display
difficulty_df = pd.read_sql_query(query, conn)

# Print the entire table
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.width', 100)      # Set display width
print(difficulty_df)

# Save to CSV for future reference
difficulty_df.to_csv('difficulty_grades.csv', index=False)
print(f"\nData also saved to difficulty_grades.csv")

# Close the connection
conn.close()