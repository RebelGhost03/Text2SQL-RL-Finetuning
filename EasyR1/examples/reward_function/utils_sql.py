import re
import sqlparse
import sqlite3
import os
from typing import List, Tuple

def remove_comments(sql: str) -> str:
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    sql = re.sub(r'--[^\n]*', '', sql)
    return sql

def format_sql(sql: str) -> str:
    sql = remove_comments(sql)
    sql = sqlparse.format(sql, reindent=True, keyword_case='upper', indent_width=1, wrap_after=999999999)
    # ... [Logic for BIRD SQL formatting truncation/wrapping] ...
    # Simplified for brevity, insert full format_sql logic from original notebook here
    return sql.strip()

def extract_queries(lines, queries=None):
    if queries is None: queries = []
    # ... [Insert full extract_queries logic from original notebook here] ...
    alias = f'[{len(queries)}]'
    queries.append((alias, '')) # Placeholder
    return alias

def forward_convert(sql: str) -> str:
    """Converts standard SQL to the BIRD dataset intermediate representation."""
    # ... [Insert full forward_convert logic from original notebook here] ...
    # This acts as a wrapper for the complex formatting logic
    return format_sql(sql) + ";" 

def backward_convert(sql: str) -> str:
    """Converts BIRD intermediate representation back to standard SQL."""
    # ... [Insert full backward_convert logic from original notebook here] ...
    return sql

def execute_sql(sql_query: str, db_path: str, timeout: int = 3) -> Tuple[bool, list]:
    """Execute SQL query on SQLite database."""
    conn = None
    try:
        # Adjust path relative to workspace if necessary
        if '/workspace/dataset' not in db_path:
             db_path = '/workspace/dataset/' + db_path.split('./')[1] if './' in db_path else db_path
        
        conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA max_page_count = 10000")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchmany(1000)
        return True, results
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()

def normalize_results(results: list) -> list:
    """Normalize results for comparison (sort, strip whitespace, lower case)."""
    if not results:
        return []
    normalized = []
    for row in results:
        normalized_row = tuple(
            str(val).strip().lower() if val is not None else None
            for val in row
        )
        normalized.append(normalized_row)
    return sorted(normalized)