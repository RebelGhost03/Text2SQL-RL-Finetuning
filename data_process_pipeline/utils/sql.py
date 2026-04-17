from typing import Dict, List, Set
import sqlglot
from sqlglot import expressions as exp

def qualify_columns(expr: exp.Expression) -> None:
    """
    Recursively fill missing table names for columns

    Example:
        SELECT id FROM users; 
        -> SELECT users.id FROM users;
    """

    # Get table name or alias for the current scope
    from_node = expr.find(exp.From)
    current_scope_table = from_node.alias_or_name

    # DFS traversal over nodes
    # Recurse into CTE or Subquery node
    # Fill table for Column nodes without qualifier
    node_stack = [expr]
    while node_stack:
        current_node = node_stack.pop()
        if isinstance(current_node, exp.Column) and not current_node.table:
            current_node.set("table", exp.to_identifier(current_scope_table))
        for node in current_node.iter_expressions():
            if isinstance(node, exp.CTE) or isinstance(node, exp.Subquery):
                qualify_columns(node)
            else:
                node_stack.append(node)

def extract_base_schema(sql: str) -> Dict:
    """
    Extract base (physical) tables and their referenced columns from a SQL query,
    excluding tables introduced by CTEs and subqueris.

    Returns:
        Dict[str, Set[str]]: Mapping from base table name to referenced columns.
        Example:
            {
                "table_name": {"column1", "column2", ...},
                ...
            }
    """

    expr = sqlglot.parse_one(sql, read="sqlite")
    qualify_columns(expr)

    # Collect aliases of runtime tables introduced by CTEs and subqueries
    runtime_tables = set()
    for node in expr.find_all(exp.CTE, exp.Subquery):
        if node.alias:
            runtime_tables.add(node.alias)
    
    # Map table alias to base table name (exclude runtime tables)
    table_alias_to_name = {}
    for node in expr.find_all(exp.Table):
        if node.name not in runtime_tables:
            alias = node.alias_or_name
            name = node.name
            table_alias_to_name[alias] = name
    
    column_alias = set()
    for node in expr.find_all(exp.Alias):
        column_alias.add(node.alias)

    # Build schema: base table -> referenced columns
    schema: Dict[str, Set[str]] = {}
    for node in expr.find_all(exp.Column):
        column = node.name
        table_alias = node.table
        if column in column_alias:
            continue
        if table_alias in table_alias_to_name:
            table = table_alias_to_name[table_alias]
            if table not in schema:
                schema[table] = set()
            schema[table].add(column)
    
    return schema
