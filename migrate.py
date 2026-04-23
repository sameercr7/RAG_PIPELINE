from db import get_connection
import pymysql

def migrate():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Check if column exists
            cur.execute("SHOW COLUMNS FROM analyzed_data LIKE 'ingestion_status'")
            if not cur.fetchone():
                print("Adding 'ingestion_status' column...")
                cur.execute("ALTER TABLE analyzed_data ADD COLUMN ingestion_status VARCHAR(20) DEFAULT 'PENDING'")
                conn.commit()
                print("Column added successfully.")
            else:
                print("Column 'ingestion_status' already exists.")
            
            # Index the status column for faster polling
            try:
                cur.execute("CREATE INDEX idx_ingestion_status ON analyzed_data (ingestion_status)")
                conn.commit()
                print("Index created.")
            except Exception as e:
                print(f"Index might already exist: {e}")
                
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
