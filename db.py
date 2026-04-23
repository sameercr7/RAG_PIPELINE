import pymysql
import pymysql.cursors
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB


def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
    )


def fetch_analyzed_data_batch(offset: int, batch_size: int) -> list[dict]:
    """Fetch a batch of rows from analyzed_data for ingestion."""
    sql = """
        SELECT
            id,
            input_text,
            contextual_understanding,
            topic_title,
            incidents,
            events,
            person_names,
            organisation_names,
            location_names,
            district_names,
            thana_names,
            broad_category,
            sub_category,
            sentiment_label,
            post_bank_author_name,
            post_bank_source,
            post_bank_post_timestamp,
            post_bank_post_url
        FROM analyzed_data
        WHERE input_text IS NOT NULL AND input_text != ''
        ORDER BY id
        LIMIT %s OFFSET %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (batch_size, offset))
            return cur.fetchall()


def get_total_analyzed_rows() -> int:
    sql = "SELECT COUNT(*) as cnt FROM analyzed_data WHERE input_text IS NOT NULL AND input_text != ''"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()["cnt"]


def fetch_pending_by_source(source: str, limit: int = 100) -> list[dict]:
    """Fetch PENDING rows for a specific source."""
    sql = """
        SELECT
            id, input_text, contextual_understanding, topic_title,
            incidents, events, person_names, organisation_names,
            location_names, district_names, thana_names,
            broad_category, sub_category, sentiment_label,
            post_bank_author_name, post_bank_source,
            post_bank_post_timestamp, post_bank_post_url
        FROM analyzed_data
        WHERE ingestion_status = 'PENDING'
          AND post_bank_core_source = %s
          AND input_text IS NOT NULL AND input_text != ''
        LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (source, limit))
            return cur.fetchall()


def update_ingestion_status(ids: list[int], status: str = 'INGESTED'):
    """Update status for multiple rows."""
    if not ids:
        return
    sql = f"UPDATE analyzed_data SET ingestion_status = %s WHERE id IN ({','.join(['%s'] * len(ids))})"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [status] + ids)
            conn.commit()


def get_admin_stats() -> list[dict]:
    """Get sync progress per source."""
    sql = """
        SELECT 
            post_bank_core_source as source,
            COUNT(*) as total,
            SUM(CASE WHEN ingestion_status = 'INGESTED' THEN 1 ELSE 0 END) as ingested
        FROM analyzed_data
        WHERE post_bank_core_source IS NOT NULL
        GROUP BY post_bank_core_source
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()