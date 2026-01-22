import json
import sqlite3
import re
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# --- THREAD-LOCAL STORAGE ---
# Each thread gets its own independent database connections to avoid locking contention
thread_local = threading.local()

def get_thread_connection(db_path: str, timeout: int = 5) -> sqlite3.Connection:
    """
    Get or create a connection for the current thread.
    """
    if not hasattr(thread_local, "connections"):
        thread_local.connections = {}
    
    # Return cached connection if available
    if db_path in thread_local.connections:
        return thread_local.connections[db_path]
    
    # Create new connection
    try:
        # check_same_thread=False is needed, but since we use thread_local, 
        # we naturally isolate them anyway.
        conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        
        # --- AGGRESSIVE READ OPTIMIZATIONS ---
        conn.execute("PRAGMA query_only = ON") 
        conn.execute("PRAGMA journal_mode = WAL") 
        conn.execute("PRAGMA synchronous = OFF") 
        conn.execute("PRAGMA temp_store = MEMORY") 
        conn.execute("PRAGMA cache_size = -64000") # Use 64MB of RAM for cache
        conn.execute("PRAGMA mmap_size = 3000000000") # Memory Map ~30GB (Try to map entire DB to RAM)
        conn.execute("PRAGMA locking_mode = NORMAL")
        
        thread_local.connections[db_path] = conn
        return conn
    except Exception as e:
        print(f"Error connecting to {db_path}: {e}")
        return None

def inject_limit(sql: str, limit: int = 10) -> str:
    """
    Injects a LIMIT clause to prevent the DB from calculating full results
    for complex queries when we only need a sample.
    """
    sql_upper = sql.upper().strip()
    
    # Simple check: if LIMIT already exists, don't touch it
    if "LIMIT" in sql_upper:
        return sql
        
    # Remove trailing semicolon if present
    sql = sql.strip().rstrip(';')
    
    # Append LIMIT
    return f"{sql} LIMIT {limit}"

def process_single_item(item: dict) -> dict:
    """
    Worker function to process a single JSON item.
    """
    input_seq = item.get("input_seq")
    original_sql = item.get("sql")
    db_path = item.get("db_path")
    
    # Clean SQL comments
    clean_sql = _remove_comments(original_sql)
    
    # OPTIMIZATION: Push LIMIT to DB engine
    optimized_sql = inject_limit(clean_sql, 10)
    
    conn = get_thread_connection(db_path)
    res_str = "[]"
    
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(optimized_sql)
            # Fetch is already fast because of LIMIT injection
            results = cursor.fetchall() 
            normalized_res = normalize_results(results)
            res_str = str(normalized_res)
        except Exception as e:
            # If optimized SQL fails (rare syntax edge cases), fallback to original
            try:
                cursor = conn.cursor()
                cursor.execute(clean_sql)
                results = cursor.fetchmany(10)
                normalized_res = normalize_results(results)
                res_str = str(normalized_res)
            except Exception as e2:
                # print(f"SQL Fail in {db_path}: {e2}") # Optional: Uncomment for debug
                pass

    formatted_sql = f"<sql>{original_sql}</sql> <db_path>{db_path}</db_path> <answer>{res_str}</answer>"
    
    return {
        "input_seq": input_seq,
        "sql": formatted_sql
    }

def normalize_results(results: list) -> list:
    if not results:
        return []
    # Fast list comprehension normalization
    return sorted([
        tuple(str(val).strip().lower() if val is not None else '' for val in row)
        for row in results
    ])

def _remove_comments(sql_str):
    # (Keep your existing utility function)
    lines = sql_str.split('\n')
    cleaned_lines = [line.split('--')[0] for line in lines]
    sql_str = '\n'.join(cleaned_lines)
    while '/*' in sql_str:
        start = sql_str.find('/*')
        end = sql_str.find('*/', start)
        if end != -1:
            sql_str = sql_str[:start] + ' ' + sql_str[end+2:]
        else:
            break
    return sql_str

def calculate_difficulty(sql: str) -> str:
    """
    Calculates complexity score and classifies SQL into Simple, Moderate, or Challenging.
    Based on specific weights for Joins, Subqueries, Set Ops, and Aggregations.
    """
    # 1. Preprocessing: Sanitize and Uppercase
    clean_sql = _remove_comments(sql).upper()
    
    score = 0.0

    # 2. Structural Complexity
    # JOINs: 1 point for single, scaling up to 3 points for >= 3 joins
    join_count = len(re.findall(r'\bJOIN\b', clean_sql))
    if join_count >= 3:
        score += 3
    elif join_count == 2:
        score += 2
    elif join_count == 1:
        score += 1

    # Subqueries: 2 * (N_select - 1)
    select_count = len(re.findall(r'\bSELECT\b', clean_sql))
    if select_count > 1:
        score += 2 * (select_count - 1)

    # Recursive CTEs (Weighted heavily: 3 points)
    # Checking for "WITH RECURSIVE" or just complex "WITH" logic
    if "WITH RECURSIVE" in clean_sql:
        score += 3

    # 3. Advanced Keywords
    # Set Operations: 2 points each
    set_ops = len(re.findall(r'\b(UNION|INTERSECT|EXCEPT)\b', clean_sql))
    score += set_ops * 2

    # GROUP BY: 2 points
    if re.search(r'\bGROUP\s+BY\b', clean_sql):
        score += 2

    # CASE WHEN: 1 point
    if "CASE WHEN" in clean_sql:
        score += 1

    # 4. Aggregations and Functions
    # Standard Aggregates (0.5 points each to distinguish from heavy logic)
    aggs = len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', clean_sql))
    score += aggs * 0.5

    # Text/Date Functions (0.5 points each)
    funcs = len(re.findall(r'\b(SUBSTRING|EXTRACT|STRFTIME|DATE|YEAR|MONTH)\b', clean_sql))
    score += funcs * 0.5

    # 5. Filtering Logic
    # If WHERE clause has > 2 logical connectives (AND/OR)
    # We roughly estimate this by counting AND/OR in the whole string for speed,
    # or strictly looking after 'WHERE'. Let's look globally for simplicity as logic is often complex.
    logic_count = len(re.findall(r'\b(AND|OR)\b', clean_sql))
    if logic_count > 2:
        score += 1

    # 6. Classification Thresholds
    if score >= 3:
        return "Challenging"
    elif score >= 2:
        return "Moderate"
    else:
        return "Simple"
    
def convert_path_difficulty(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            spider_data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Loaded {len(original_data)} items. Starting parallel processing...")
    
    new_data_s = []
    new_data_m = []
    new_data_c = []
    
    # Tách data theo độ khó
    for data in original_data:
        output_seq = data.get('output_seq')
        
        format_data = {
           "instruction": data.get('input_seq'),
           "input": '',
           "output": data.get('output_seq')
        }
        
        sql_difficulty = calculate_difficulty(output_seq.split('```sql\n')[1].split(';\n```')[0])
        if sql_difficulty == "Simple":
            new_data_s.append(format_data)
        if sql_difficulty != "Challenging":
            new_data_m.append(format_data)
        new_data_c.append(format_data)

    save_path_s = os.path.join(r"D:\\IT\\FPT\\Project\\2025\\Chatbox\\EasyR1\\Data_processing\\dataset", 'sft_' + output_filename + '_simple.json')
    save_path_m = os.path.join(r"D:\\IT\\FPT\\Project\\2025\\Chatbox\\EasyR1\\Data_processing\\dataset", 'sft_' + output_filename + '_simple_moderate.json')
    save_path_c = os.path.join(r"D:\\IT\\FPT\\Project\\2025\\Chatbox\\EasyR1\\Data_processing\\dataset", 'sft_' + output_filename + '_full.json')
    
    with open(save_path_s, 'w', encoding='utf-8') as f:
        json.dump(new_data_s, f, indent=2, ensure_ascii=False)
    with open(save_path_m, 'w', encoding='utf-8') as f:
        json.dump(new_data_m, f, indent=2, ensure_ascii=False)
    with open(save_path_c, 'w', encoding='utf-8') as f:
        json.dump(new_data_c, f, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    start_time = time.time()
    input_file = f"D:\\IT\\FPT\\Project\\2025\\Chatbox\\EasyR1\\Data_processing\\dataset\\train_data.json"
    output_file = "train_data"
    
    convert_path_difficulty(input_file, output_file)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print("="*30)
    print(f"COMPLETED in {int(minutes)}m {seconds:.2f}s")
    print("="*30)