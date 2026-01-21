import re
import sqlite3
import os
import threading
import ast
import sqlglot
import sqlparse
from sqlglot import exp
from typing import Any, List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

# Metadata
REWARD_NAME = "sql_reasoning"
REWARD_TYPE = "batch"

# --- GLOBAL THREAD STATE ---
# Dùng để quản lý connection riêng cho từng thread và đóng chúng an toàn
thread_local = threading.local()
all_connections = [] 
conn_lock = threading.Lock()

# --- UTILITIES ---

def remove_comments(sql: str) -> str:
    """Loại bỏ comment để xử lý SQL sạch."""
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    sql = re.sub(r'--[^\n]*', '', sql)
    return sql

def extract_sql_content(response: str) -> str:
    """Lấy nội dung SQL từ markdown."""
    try:
        if "```sql" in response:
            return response.split("```sql")[1].split("```")[0].strip()
        elif "```" in response:
             return response.split("```")[1].split("```")[0].strip()
        return response.strip()
    except IndexError:
        return response.strip()

def extract_sql_metadata(response: str):
    """
    Lấy SQL, DB Path và Answer định dạng list python từ string.
    """
    # Pattern bắt thêm phần <answer>
    pattern = re.compile(r"<sql>(.*?)</sql>\s*<db_path>(.*?)</db_path>\s*<answer>(.*?)</answer>", re.DOTALL)
    match = re.search(pattern, response)
    
    if match:
        sql = match.group(1).strip()
        db_path = match.group(2).strip()
        answer_raw = match.group(3).strip()
        
        # Parse chuỗi "[('val',), ...]" thành List Python thật
        try:
            # ast.literal_eval an toàn hơn eval() thường
            answer_list = ast.literal_eval(answer_raw)
        except Exception:
            answer_list = []
            
        return sql, db_path, answer_list
        
    return None, None, []

def inject_limit(sql: str, limit: int = 10) -> str:
    """
    OPTIMIZATION: Chèn LIMIT vào query để DB engine dừng sớm,
    giúp execution_reward trả về kết quả cực nhanh.
    """
    if not sql: return sql
    sql_upper = sql.upper().strip()
    if "LIMIT" in sql_upper:
        return sql
    # Xóa dấu chấm phẩy cuối cùng nếu có
    sql = sql.strip().rstrip(';')
    return f"{sql} LIMIT {limit}"

def normalize_results(results: list) -> list:
    """Chuẩn hóa kết quả để so sánh chính xác."""
    if not results:
        return []
    normalized = []
    for row in results:
        normalized_row = tuple(
            str(val).strip().lower() if val is not None else ''
            for val in row
        )
        normalized.append(normalized_row)
    return sorted(normalized)

# --- DATABASE OPTIMIZATIONS ---

def get_thread_connection(db_path: str, timeout: int = 5) -> sqlite3.Connection:
    """
    Tạo kết nối DB tối ưu riêng cho từng thread (Thread-safe).
    """
    if '/workspace/dataset' not in db_path:
         db_path = '/workspace/dataset/' + db_path.split('./')[1] if './' in db_path else db_path

    # Kiểm tra cache của thread hiện tại
    if not hasattr(thread_local, "connections"):
        thread_local.connections = {}
    
    if db_path in thread_local.connections:
        return thread_local.connections[db_path]

    try:
        conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        
        # --- CÁC CẤU HÌNH TĂNG TỐC ĐỘ ĐỌC ---
        conn.execute("PRAGMA query_only = ON")      # Chỉ cho phép đọc
        conn.execute("PRAGMA journal_mode = WAL")   # Tăng tốc độ đồng thời
        conn.execute("PRAGMA synchronous = OFF")    # Giảm ghi đĩa
        conn.execute("PRAGMA temp_store = MEMORY")  # Temp table trong RAM
        conn.execute("PRAGMA mmap_size = 100000000000") # Map DB vào RAM (~100GB)
        conn.execute("PRAGMA cache_size = -1000000") # Cache 1GB
        
        thread_local.connections[db_path] = conn
        
        # Đăng ký vào list chung để dọn dẹp sau này
        with conn_lock:
            all_connections.append(conn)
            
        return conn
    except Exception as e:
        # print(f"Connection failed {db_path}: {e}")
        return None

def execute_sql(sql_query: str, conn: sqlite3.Connection) -> Tuple[bool, list]:
    """Thực thi SQL với kết nối có sẵn."""
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall() # Dùng fetchall vì đã có LIMIT ở query rồi
        return True, results
    except Exception as e:
        return False, str(e)

# --- REWARD FUNCTIONS ---

def get_sql_components(sql_string: str) -> Tuple[Set[str], Set[str]]:
    """Phân tích cú pháp SQL để lấy bảng và cột."""
    try:
        parsed = sqlglot.parse_one(sql_string)
        tables = {t.name.lower() for t in parsed.find_all(exp.Table)}
        columns = {c.name.lower() for c in parsed.find_all(exp.Column)}
        alias = {c.name.lower() for c in parsed.find_all(exp.Alias)}
        columns.update(alias)
        return tables, columns
    except Exception:
        return set(), set()

def structural_reward(response_sql: str, ground_truth_sql: str) -> float:
    """Tính điểm cấu trúc (tối đa 0.4)."""
    gen_tables, gen_cols = get_sql_components(response_sql)
    gt_tables, gt_cols = get_sql_components(ground_truth_sql)
    
    table_score = 0.0
    col_score = 0.0
    
    if gen_tables and gt_tables:
        table_score = len(gen_tables & gt_tables) / len(gen_tables | gt_tables)
        
    if gen_cols and gt_cols:
        col_score = len(gen_cols & gt_cols) / len(gen_cols | gt_cols)
        
    return (table_score * 0.2) + (col_score * 0.2)

def execution_reward(response_sql: str, gt_answer: str, db_path: str) -> float:
    """
    Tính điểm thực thi (0.0, 0.1, 0.6).
    Sử dụng LIMIT injection và Thread-local connection.
    """
    conn = get_thread_connection(db_path)
    if not conn:
        return 0.0
        
    # Inject LIMIT để query chạy nhanh hơn
    opt_gen_sql = inject_limit(response_sql, 10)

    try:
        # Chạy Gen SQL (nhanh nhờ mmap + limit)
        gen_success, gen_results = execute_sql(opt_gen_sql, conn)
        if not gen_success: return 0.0 # GT lỗi thì bỏ qua11
        
        # So sánh kết quả
        if normalize_results(gt_answer) == normalize_results(gen_results):
            return 0.6  # Kết quả đúng
        else:
            return 0.1  # Chạy được nhưng sai kết quả
    except Exception:
        return 0.0

# --- MAIN PARALLEL PROCESSING ---

def process_single_reward(input_data: Dict[str, Any]) -> Dict[str, float]:
    """Hàm worker xử lý 1 dòng dữ liệu."""
    response = input_data["response"]
    
    # Lấy thông tin từ Ground Truth (bao gồm DB path)
    ground_truth, db_path_raw, gt_answer = extract_sql_metadata(input_data["ground_truth"])
    
    # Lấy SQL từ model response
    gen_sql = extract_sql_content(response)

    # 1. Structural Match (Không cần DB)
    struct_score = structural_reward(gen_sql, ground_truth)
    
    # 2. Execution Match (Cần DB - Tối ưu hóa cao)
    if db_path_raw:
        exec_score = execution_reward(gen_sql, gt_answer, db_path_raw)
    else:
        exec_score = 0.0

    raw_quality_score = struct_score + exec_score
    
    return {
        "overall": raw_quality_score,
        "structural": struct_score,
        "execution": exec_score,
    }

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.0) -> List[Dict[str, float]]:
    """
    Hàm chính: Tính điểm song song (Parallel).
    """
    scores = [None] * len(reward_inputs)
    
    # --- CẤU HÌNH CHO SERVER H100 (120 Cores) ---
    cpu_count = os.cpu_count() or 120
    
    # Với server mạnh và RAM dư dả (2TB), ta có thể dùng hệ số * 2
    # Tuy nhiên, để tránh overhead của Python GIL quá lớn, start ở mức bằng số cores là đẹp nhất.
    max_workers = cpu_count  # Set thẳng bằng 120
    
    # Nếu muốn hardcode luôn để chắc chắn:
    # max_workers = 120 
        
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single_reward, item): i 
                for i, item in enumerate(reward_inputs)
            }
            
            for future in future_to_index:
                index = future_to_index[future]
                try:
                    scores[index] = future.result()
                except Exception as e:
                    print(f"Error at index {index}: {e}")
                    scores[index] = {"overall": 0.0, "structural": 0.0, "execution": 0.0}

    finally:
        with conn_lock:
            for conn in all_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            all_connections.clear()
            if hasattr(thread_local, "connections"):
                thread_local.connections.clear()

    return scores