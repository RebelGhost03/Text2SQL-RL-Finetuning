import sqlite3
from typing import Dict, List, Literal

def introspect_db(db_path: str) -> Dict:
    """
    Introspect a SQLite database and extract structured schema information.

    Returns: 
        {
            table_name_value: {
                col_name_value: {
                    "data_type": type,
                    "value_prefixs": [...value_prefixs],
                    "primary_key": bool,
                    "foreign_key": {
                        "ref_table": ref_table_value,
                        "ref_column": ref_column_value
                    }
                }
            }
        }
    """
    result = {}
    conn = None
    cursor = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        tables = cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
                AND name NOT LIKE 'sqlite_%'
        """).fetchall()

        for table_row in tables:
            table_name = table_row["name"]
            result[table_name] = {}

            # Build foreign key mapping for this table.
            # If the referenced column is missing, fall back to the primary key
            # of the referenced table. Skip the foreign key only if it cannot
            # be resolved.
            fk_rows = cursor.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()
            fk_map = {}
            for fk in fk_rows:
                from_col = fk["from"]
                ref_table = fk["table"]
                ref_column = fk["to"]
                if not from_col or not ref_table:
                    continue
                if not ref_column:
                    pk_rows = cursor.execute(f'PRAGMA table_info("{ref_table}")').fetchall()
                    pk_cols = [r["name"] for r in pk_rows if r["pk"] == 1]
                    if pk_cols:
                        ref_column = pk_cols[0]
                    else:
                        continue
                fk_map[from_col] = {
                    "ref_table": ref_table,
                    "ref_column": ref_column
                }

            # Collect column data types, non-null values, and attach foreign key metadata.
            columns = cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            for col in columns:
                col_name = col["name"]
                data_type = col["type"]
                is_primary_key = col["pk"] > 0
                if any(t in data_type.upper() for t in ("CHAR", "TEXT", "CLOB", "VARCHAR")):
                    # Only collect value prefixes for columns with text-like declared types
                    try:
                        rows = cursor.execute(
                            f"""
                            SELECT DISTINCT
                                substr(
                                    CASE
                                        WHEN instr("{col_name}", ' ') = 0
                                        THEN "{col_name}"
                                        ELSE substr("{col_name}", 1, instr("{col_name}", ' ') - 1)
                                    END,
                                    1,
                                    10
                                ) AS prefix
                            FROM "{table_name}"
                            WHERE "{col_name}" IS NOT NULL
                            """
                        ).fetchall()
                        value_prefixs = [r[0].replace('"',' ').replace("'",' ').split()[0] for r in rows]
                        value_prefixs = set(value_prefixs)
                    except:
                        value_prefixs = None
                else:
                    value_prefixs = None
                result[table_name][col_name] = {
                    "data_type": data_type,
                    "value_prefixs": value_prefixs,
                    "primary_key": is_primary_key,
                    "foreign_key": fk_map.get(col_name)
                }
        
        for table_name, table in result.items():
            for column_name, column in table.items():
                if column["foreign_key"]:
                    ref = column["foreign_key"]
                    ref_table = ref["ref_table"]
                    ref_column = ref["ref_column"]
                    for table_name_2 in result.keys():
                        if table_name_2.lower() == ref_table.lower():
                            ref_table = table_name_2
                            break
                    for column_name_2 in result[table_name].keys():
                        if column_name_2.lower() == ref_column.lower():
                            ref_column = column_name_2
                            break
                    ref["ref_table"] = ref_table
                    ref["ref_column"] = ref_column
                            
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return result

def find_values(
    db_path: str, 
    table_name: str,
    column_name: str,
    value_prefix: str | None = None,
    limit: int | None = 2
) -> List[str]:
    # return []
    """
    Find distinct values in a given table column that start with value_prefix.
    """

    conn = None
    cursor = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        sql = f"""
        SELECT DISTINCT "{column_name}"
        FROM "{table_name}"
        WHERE "{column_name}" IS NOT NULL
        """

        params = []

        if value_prefix:
            sql += f'AND "{column_name}" LIKE ?'
            params.append(value_prefix + "%")

        if limit is not None:
            sql += f" LIMIT {limit}"

        rows = cursor.execute(sql, params).fetchall()
        out = [r[0] for r in rows]
        out = [str(v) for v in out if v is not None]
        out = [r[:100] for r in out]
        return out

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def schema_to_string(schema: Dict, mode: Literal["ddl", "markdown", None] = None) -> str:
    schema_description = ""
    
    if mode == "ddl":
        for table_name, table in schema.items():
            schema_description += f"CREATE TABLE {table_name} (\n"
            for column_name, column in table.items():
                schema_description += f"\t`{column_name}` {column['data_type']},"
                # if "meaning" in column:
                #     schema_description += f"\t -- {column['meaning']}"
                if column["values"]:
                    if column["data_type"].upper() in ("TEXT", "CHAR", "VARCHAR", "CLOB"):
                        all_values = ", ".join([f"'{value}'" for value in column["values"]])
                    else:
                        all_values = ", ".join([str(value) for value in column["values"]])
                    schema_description += f"\t-- example: [{all_values}]"
                schema_description += "\n"
            schema_description += ");\n"

    else:
        for table_name, table in schema.items():
            schema_description += f"(Table) {table_name}:\n"
            for column_name, column in table.items():
                schema_description += f"\t(Column) {column_name}"
                if column["foreign_key"]:
                    if column["foreign_key"]["ref_table"] in schema and column["foreign_key"]["ref_column"] in schema[column["foreign_key"]["ref_table"]]:
                        schema_description += f' - refer to - {column["foreign_key"]["ref_table"]}.{column["foreign_key"]["ref_column"]}'
                schema_description += f": {column['data_type']}"
                if "meaning" in column:
                    schema_description += f", {column['eaning']}"
                schema_description += "\n"
    
    return schema_description

def filter_schema(schema: Dict, elements: Dict[str, List[str]]):
    filtered = {}
    for table_name, column_names in elements.items():
        if table_name in schema:
            filtered[table_name] = {}
            for column_name in column_names:
                if column_name in schema[table_name]:
                    filtered[table_name][column_name] = schema[table_name][column_name].copy()
                    if filtered[table_name][column_name]["foreign_key"]:
                        ref = filtered[table_name][column_name]["foreign_key"]
                        if ref["ref_table"] not in elements or ref["ref_column"] not in elements[ref["ref_table"]]:
                            filtered[table_name][column_name]["foreign_key"] = None
    
    return filtered