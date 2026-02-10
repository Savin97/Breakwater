# db_testing_for_stage_1_overhaul.py
import os
import duckdb

from config import DB_PATH

def main():
    connection = duckdb.connect(DB_PATH)

    try:
        result = connection.execute("""
            CREATE TABLE IF NOT EXISTS prices(
                stock TEXT,
                date DATE,
                price DOUBLE,
                ingested_at TIMESTAMP
                );
                                    
            """)
        
        print("prices table ready")

        # verify schema
        schema = connection.execute("DESCRIBE prices").fetchall()
        print("\nprices schema:")
        for row in schema:
            print(row)
    finally:
        connection.close()

if __name__ == "__main__":
    main()
