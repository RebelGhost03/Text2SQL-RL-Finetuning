from typing import List, Dict
import re

from utils.text import (
    text_to_canonical_form,
    to_snake_case
)

from utils.sqlite import (
    find_values
)

def transform_to_keys(texts: List[str]) -> List[str]:
    """
    Transform a list of names into keys composed of their rarest words.
    Returns a list of comma-separated keys, one per input text.

    text -> key1,key2
    """
    canonical_texts = [text_to_canonical_form(text) for text in texts]
    word_count = {}
    for text in canonical_texts:
        words = text.split()
        for w in words:
            word_count[w] = word_count.get(w, 0) + 1
    
    keys = []
    for text in canonical_texts:
        words = text.split()
        try:
            min_count = min([word_count[w] for w in words])
        except:
            min_count = 0
        words = [w for w in words if word_count[w] == min_count]
        # words = sorted(words, key=lambda w: word_count[w])
        # if any(word_count[w] == 1 for w in words):
        #     words = [w for w in words if word_count[w] == 1]
        # else:
        #     words = words[:1]
        keys.append(",".join(words))
    
    return keys

def _add_schema_item(
    subschema: Dict, 
    schema: Dict, 
    table_name: str,
    column_name: str | None = None,
    column_values: List | None = None
):
    if table_name not in schema:
        raise KeyError(f"Table '{table_name}' not found in schema")
    if column_name is not None:
        if column_name not in schema[table_name]:
            raise KeyError(f"Column '{column_name}' not found in '{table_name}'")

    if table_name not in subschema:
        subschema[table_name] = {}

    if column_name is not None:
        if column_name not in subschema[table_name]:
            subschema[table_name][column_name] = {
                "data_type": schema[table_name][column_name]["data_type"],
                "foreign_key": schema[table_name][column_name]["foreign_key"],
                "values": set(),
                "meaning": schema[table_name][column_name]["meaning"]
            }
        elif column_values is not None:
            subschema[table_name][column_name]["values"].update(column_values)
    else:
        # If adding a table without specifying columns, automatically include its primary key
        for column_name, column in schema[table_name].items():
            if column["primary_key"]:
                subschema[table_name][column_name] = {
                    "data_type": schema[table_name][column_name]["data_type"],
                    "foreign_key": schema[table_name][column_name]["foreign_key"],
                    "values": set(),
                    "meaning": schema[table_name][column_name]["meaning"]
                }
                break

def _extract_exact_columns(
    text: str,
    schema: Dict,
    subschema: Dict | None = None,
    remove_entities: bool = False
):
    if subschema is None:
        subschema = {}

    for table_name, tables in schema.items():
        for column_name in tables.keys():
            if column_name in text.split():
                _add_schema_item(subschema, schema, table_name, column_name)

    if remove_entities:
        for tables in subschema.values():
            for column_name in subschema.keys():
                text = text.replace(column_name, "")
        text = re.sub(r"\s+", " ", text)

    return subschema, text

def _extract_from_data(
    text: str,
    schema: Dict,
    db_path: str | None = None,
    subschema: Dict | None = None,
    remove_entities: bool = False
) -> Dict:
    if not subschema:
        subschema = {}
    
    cleaned_text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    
    entities = set()

    for table_name, table in schema.items():
        for column_name, column in table.items():
            value_prefixs = column["value_prefixs"]
            if not value_prefixs:
                continue
            value_prefixs = [v.lower() for v in value_prefixs]
            for word in cleaned_text.split():
                done = False
                word = word[:10].lower()
                if word in value_prefixs:
                    if db_path:
                        values = find_values(db_path, table_name, column_name, word)
                    else:
                        values = []
                    if values:
                        for val in values:
                            if str(val) in text:
                                entities.update(values)
                                entities.add(table_name)
                                entities.add(column_name)
                            if str(val).lower() in text.lower():
                                _add_schema_item(subschema, schema, table_name, column_name, values)
                                done = True
                            if done:
                                break
                        if done:
                            break

    if remove_entities:
        for entity in entities:
            text = text.replace(str(entity), "")
        text = re.sub(r"\s+", " ", text)

    return subschema, text

def expand_semantics(text: str) -> str:
    text = re.sub(r"\btitle\b", "title, name", text, flags=re.IGNORECASE)
    text = re.sub(r"\bname\b", "title, name", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwho\b", "name, first name, last name, middle name, full name", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhich\b", "name, title", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhat\b", "name, title", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhen\b", "year, date, day, time", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhow many\b", "count, number, quantity", text, flags=re.IGNORECASE)   
    text = re.sub(r"\bpeople\b", "people, person", text, flags=re.IGNORECASE)
    text = re.sub(r"\bperson\b", "people, person", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfree\b", "free price, cost", text, flags=re.IGNORECASE)
    text = re.sub(r"\blist\b", "name, title", text, flags=re.IGNORECASE)
    text = re.sub(r"\bselling\b", "selling, price, cost, bill, quantity", text, flags=re.IGNORECASE)
    text = re.sub(r"USD", " cost, amount, price ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\$€£¥₩₹₽₫₿¢]", " cost, amount, price ", text)
    text = re.sub(r"\bfirst\b", "first, 1, 1st", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsecond\b", "second, 2, 2nd", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthird\b", "third, 3, 3rd", text, flags=re.IGNORECASE)
    text = re.sub(r"\b1st\b", "1st, first", text, flags=re.IGNORECASE)
    text = re.sub(r"\b2nd\b", "2nd, second", text, flags=re.IGNORECASE)
    text = re.sub(r"\b3rd\b", "3rd, third", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", r"date, time, day, datetime, timestamp, birth, born, dob \g<0>", text)
    text = re.sub(r"\b(19\d{2}|20\d{2})\b", r"year, date, datetime, time, timestamp, birth, born, dob \1", text)
    text = re.sub(r"\bborn\b", "dob, birth, born, date, birthday", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbirthday\b", "dob, date, birth", text, flags=re.IGNORECASE)
    
    return text

def generate_schema_keys(schema: Dict, semantic_map: Dict) -> Dict:
    key_map = {}

    table_names = list(schema.keys())
    semantic_table_names = [
        semantic_map.get(table_name, "") 
        for table_name in table_names
    ]
    table_keys = transform_to_keys(table_names)
    semantic_table_keys = transform_to_keys(semantic_table_names)
    for name, key, semantic_key in zip(table_names, table_keys, semantic_table_keys):
        all_parts = key.split(",") + semantic_key.split(",")
        all_parts = [i for i in all_parts if i]
        all_parts = list(set(all_parts))
        key_map[name] = ",".join(all_parts)

    for table_name, table in schema.items():
        column_names = list(table.keys())
        semantic_column_names = [
            semantic_map.get(f"{table_name}|{column_name}", "")
            for column_name in column_names
        ]
        column_keys = transform_to_keys(column_names)
        semantic_column_keys = transform_to_keys(semantic_column_names)
        for column_name, key, semantic_key in zip(column_names, column_keys, semantic_column_keys):
            all_parts = key.split(",") + semantic_key.split(",")
            all_parts = [i for i in all_parts if i]
            all_parts = list(set(all_parts))
            name = f"{table_name}|{column_name}"
            key_map[name] = ",".join(all_parts)

    return key_map

def _update_foreign_keys(subschema: Dict, schema: Dict):
    additional_cols = []
    for table_name, table in subschema.items():
        for column_name, column in schema[table_name].items():
            if column["foreign_key"]:
                # Direct foreign key to another subschema table
                if column["foreign_key"]["ref_table"] in subschema:
                    additional_cols.append({
                        "table": table_name,
                        "column": column_name
                    })
                    additional_cols.append({
                        "table": column["foreign_key"]["ref_table"],
                        "column": column["foreign_key"]["ref_column"]
                    })
                
                # Foreign key through an intermediate table
                else:
                    mid_table_name = column["foreign_key"]["ref_table"]
                    for mid_table_colum_name, mid_table_column in schema[mid_table_name].items():
                        if mid_table_column["foreign_key"]:
                            if mid_table_column["foreign_key"]["ref_table"] in subschema:
                                additional_cols.append({
                                    "table": table_name,
                                    "column": column_name
                                })
                                additional_cols.append({
                                    "table": mid_table_name,
                                    "column": column["foreign_key"]["ref_column"]
                                })
                                additional_cols.append({
                                    "table": mid_table_name,
                                    "column": mid_table_colum_name
                                })
                                additional_cols.append({
                                    "table": mid_table_column["foreign_key"]["ref_table"],
                                    "column": mid_table_column["foreign_key"]["ref_column"]
                                })   
    for item in additional_cols:
        _add_schema_item(subschema, schema, item["table"], item["column"])

def extract_subschema(
    question: str, 
    evidence: str,
    schema: Dict,
    db_path: str | None = None,
    schema_semantic_map: Dict | None = None
) -> Dict:
    """
    Extract the subschema relevant to the question.

    Args:
        question: Natural language question.
        schema: Full database schema.
    Returns:
        A subschema containing only tables and columns relevant to the question.
    """

    subschema, full_question = _extract_from_data(
        text=question + " | " + evidence,
        schema=schema,
        db_path=db_path,
        remove_entities=True
    )

    parts = full_question.split(" | ")
    question = parts[0]
    evidence = parts[1]

    subschema, evidence = _extract_exact_columns(
        text=evidence,
        schema=schema,
        subschema=subschema,
        remove_entities=True
    )
    subschema, _ = _extract_exact_columns(
        text=question,
        schema=schema,
        subschema=subschema,
        remove_entities=False
    )

    full_question = evidence + " " + question

    full_question = expand_semantics(full_question)
    full_question = text_to_canonical_form(full_question)
    question_keys = set(full_question.split())

    key_map = generate_schema_keys(schema, schema_semantic_map)

    # Select tables relevant to the question
    for table_name in schema.keys():
        keys_str = key_map[table_name]
        keys = keys_str.split(",")
        for k in keys:       
            # If key appears in the question     
            if k in question_keys:
                _add_schema_item(subschema, schema, table_name)
    
    # Select columns relevant to the question
    # Initialize table entry in subschema if not already present
    for table_name, table in schema.items():
        for column_name, column in table.items():
            keys_str = key_map[f"{table_name}|{column_name}"]
            keys = keys_str.split(",")
            for k in keys:
                # If key appears in the question
                if k in question_keys:
                    _add_schema_item(subschema, schema, table_name, column_name)

    _update_foreign_keys(subschema, schema)

    # Remove tables with no selected columns
    subschema = {
        table_name: table
        for table_name, table in subschema.items()
        if table
    }

    need_table_names = subschema.keys()
    for table_name in need_table_names:
        for column_name in schema[table_name]:
            if column_name not in subschema[table_name]:
                _add_schema_item(subschema, schema, table_name, column_name)

    for table_name, table in subschema.items():
        for column_name, column in table.items():
            if not column["values"] and any(t in column["data_type"].lower() for t in ("char","text","clob","nchar","nvarchar","int","real","floa","doub","numeric","decimal")):
                column["values"].update(find_values(db_path, table_name, column_name, limit=1))

    if len(subschema) >= 5:
        deleted_tables = []
        for table_name, table in subschema.items():
            is_del = True
            for column_name, column in table.items():
                if column["foreign_key"]:
                    ref_table = column["foreign_key"]["ref_table"]
                    if ref_table in subschema:
                        is_del = False
                        break
            if is_del:
                deleted_tables.append(table_name)
        subschema = {k: v for k, v in subschema.items() if k not in deleted_tables}

    for table in subschema.values():
        for column in table.values():
            column["values"] = list(column["values"])

    return subschema