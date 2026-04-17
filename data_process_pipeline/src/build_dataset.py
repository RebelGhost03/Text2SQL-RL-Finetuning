import json
import re
from typing import List, Dict
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.subschema import extract_subschema
from src.normalize import normalize_question
from utils.sqlite import introspect_db, schema_to_string, filter_schema
from utils.sql import extract_base_schema
from utils.text import fix_unicode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--table_path", type=str)
    parser.add_argument("--thinking_path", type=str)
    parser.add_argument("--reasoning_path", type=str)
    parser.add_argument("--db_dir", type=str)
    parser.add_argument("--column_meaning_path", type=str)
    parser.add_argument("--cached_schemas_path", type=str)
    parser.add_argument("--out_path", type=str)
    
    return parser.parse_args()

def buid_semantic_map(table_path: str) -> Dict:
  with open(table_path, "r", encoding='utf-8') as f:
    semantic_map = {}
    data = json.load(f)
    for item in data:
      db_id = item["db_id"]
      semantic_map[db_id] = {}
      table_names = item["table_names_original"]
      semantic_table_names = item["table_names"]
      for table_name, semantic_name in zip(table_names, semantic_table_names):
        semantic_map[db_id][table_name] = semantic_name
      
      column_names = item["column_names_original"]
      semantic_column_names = item["column_names"]

      for column, semantic_column in zip(column_names, semantic_column_names):
        table_id = column[0]
        if table_id < 0 or table_id >= len(table_names):
          continue
        table_name = table_names[table_id]
        column_name = column[1]
        semantic_name = semantic_column[1]
        semantic_map[db_id][f"{table_name}|{column_name}"] = semantic_name

    return semantic_map

def load_samples(data_path: str, thinking_path: str | None = None) -> List[Dict]:
  if not thinking_path:
    with open(data_path, "r", encoding='utf-8') as f:
      data = json.load(f)
      return data

  with open(data_path, "r", encoding='utf-8') as f:
    filtered_ds = json.load(f)
  with open(thinking_path, "r", encoding='utf-8') as f:
    think_ds = json.load(f)

  for item in think_ds:
    input_seq = item["input_seq"]
    output_seq = item["output_seq"]

    if "Question:" in input_seq:
      input_seq = input_seq.split("Question:")[1]
    if "Instructions:\n" in input_seq:
      input_seq = input_seq.split("Instructions:\n")[0]
    input_seq = re.sub(r"/s+", " ", input_seq).strip()
    
    reasoning = output_seq.split("<think>")[1].split("</think>")[0].strip()
    if "<answer>" in output_seq:
      output_seq = output_seq.split("<answer>")[1]
    if "</answer>" in output_seq:
      output_seq = output_seq.split("</answer>")[0]
    if "```sql" in output_seq:
      output_seq = output_seq.split("```sql")[1]
    if "```" in output_seq:
      output_seq = output_seq.split("```")[0]
    output_seq = re.sub(r"/s+", " ", output_seq).strip()

    item["input_seq"] = input_seq
    item["output_seq"] = output_seq

  for item in filtered_ds:
    question = item["question"]
    question = re.sub(r"/s+", " ", question).strip()
    for i in range(len(think_ds)):
      think_item = think_ds[i]
      if item["question"] in think_item["input_seq"]:
        evidence = think_item["input_seq"].replace(question, "").strip()
        sql = think_item["output_seq"]
        
        item["evidence"] = evidence
        item["SQL"] = sql

        del think_ds[i]
        break

  return filtered_ds

def load_schemas(column_meaning_path: str, cached_schemas_path: str | None = None, db_paths: List[str] | None = None) -> Dict[str, Dict]:
  with open(column_meaning_path, "r", encoding='utf-8') as f:
    column_meaning = json.load(f)

  if cached_schemas_path:
    with open(cached_schemas_path, "r", encoding='utf-8') as f:
      schemas = json.load(f)
  else:
    schemas = {}
    for db_path in db_paths:
      db_id = db_path.split("\\")[-1].split(".sqlite")[0]
      schema = introspect_db(db_path)
      schemas[db_id] = schema
  
  for column_path, meaning in column_meaning.items():
    patterns = [
      r"The .+ column in the .+ table of the .+ database",
      r"In the .+ table of the .+ database, the .+ column",
      r"In the .+ database, within the .+ table, the .+ column",
      r"In the .+ database, the .+ table contains a .+ column",
      r"In the .+ database, the .+ table has an .+ column",
      r"The .+ column in the .+ table records the datetime",
      r"The .+ column in the .+ table .+ records",
      r"In the .+ table of .+ db, the .+ column",
      r"In the .+ table of .+ db, .+ is",
      r"In the .+ table, the .+ column",
      r"The .+ column in the .+ table",
      r"The .+ column",
    ]
    example_patterns = [
      r"\(\s*Example.+\)",
      r"Example.+",

      r"\(\s*with examples.+\)",
      r"with examples.+",

      r"\(\s*such as.+\)",
      r"such as.+",

      r"\(\s*with an example.+\)",
      r"with an example.+",

      r"\(\s*e\.g\..+\)",
      r"e\.g\..+",

      r"\(\s*exemplified.+\)",
      r"exemplified.+",

      r"\(\s*, with .+ as examples.+\)",
      r", with .+ as examples.+",

      r"\(\s*including example.+s\)",
      r"including example.+s",
    ]
    
    meaning = re.sub(r"^\W+|\W+$", "", meaning)
    for pattern in patterns:
      match = re.match(pattern, meaning)
      if match:
        meaning = meaning[match.end():].lstrip(" ,.-")
        break
    
    for pattern in example_patterns:
      meaning = re.sub(pattern, "", meaning, flags=re.IGNORECASE)
    
    meaning = re.sub(r"\s+", " ", meaning).strip(" ,.-")

    parts = column_path.split("|")
    if len(parts) == 3:
      db_id, table_name, column_name = parts
      if db_id in schemas \
        and table_name in schemas[db_id] \
        and column_name in schemas[db_id][table_name]:
        schemas[db_id][table_name][column_name]["meaning"] = meaning
  
  for db_id, schema in schemas.items():
    for table_name, table in schema.items():
      for column_name, column in table.items():
        if "meaning" not in column:
          column["meaning"] = ""

  return schemas

def generate_reasoning_prompt(
  item: Dict,
  example_item: Dict,
  schemas: Dict
) -> str:
  return f"""
You are an expert at generating step-by-step reasoning for a Text-to-SQL task.

Your goal is to analyze the Question, Evidence, Schema, and Ground Truth SQL, then generate a precise reasoning process that explains how the SQL is derived.

You MUST strictly follow the reasoning style, structure, granularity, and distribution of the provided Example.

# Schema:
{schema_to_string(filter_schema(schemas[item["db_id"]], extract_base_schema(item['SQL'])))}

# Question:
{item["question"]}

# Evidence:
{item["evidence"]}

# Groudth truth SQL:
{item['SQL']}

# Example:
## Example's schema:
{schema_to_string(filter_schema(schemas[example_item["db_id"]], extract_base_schema(example_item["SQL"])))}

## Example's question:
{example_item["question"]}

## Example's evidence:
{example_item["evidence"]}

## Example's grouth truth SQL:
{example_item["SQL"]}

## Example's reasoning output:
{example_item["reasoning"]}

# Instructions:

- Carefully analyze Question, Evidence, Schema, and Ground Truth SQL.
- Generate step-by-step reasoning that EXACTLY matches the logic of the Ground Truth SQL.
- Follow the SAME structure, wording style, reasoning depth, and ordering as the Example.
- Keep reasoning distribution consistent with the Example (no extra explanation, no missing steps).

- Clearly identify:
 • Target output (SELECT)
 • Relevant tables
 • Correct columns (no misalignment)
 • Join conditions (if any)
 • Filtering conditions (WHERE)
 • Aggregations (COUNT, SUM, AVG, MAX, MIN, etc.)
 • GROUP BY / ORDER BY / LIMIT (if present)

- Use ONLY schema elements that exist.
- Do NOT confuse similar column names.
- Do NOT invent tables, columns, or relationships.
- Ensure every clause in SQL is justified in reasoning.
- Ensure operators and comparison logic EXACTLY match the SQL.

- Do NOT generate SQL.
- Do NOT add extra commentary.
- Output ONLY the final reasoning.

Now generate the reasoning for the main sample.
"""

def generate_final_prompt(question, evidence, schema_description) -> str:
  
  if normalize_question(evidence) != "":
    question = normalize_question(evidence)+ '; ' + normalize_question(question)
  else: question = normalize_question(question)
  
  return f"""Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question. 
Database Engine: 
SQLite
Database Schema: 
{fix_unicode(schema_description)}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, columns descriptions, examples and any relevant relationships or constraints.
Question: 
{question}
Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
Output format:
Please provide a detailed chain-of-thought reasoning process and include your thought process within ‘<think>‘ tags. Your final answer should be enclosed within ‘<answer>‘ tags.
Ensure that your SQL query follows the correct syntax and is formatted as follows:
```sql
– Your SQL query here
```
Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary.</think>
<answer>
```sql
Correct SQL query here
```
</answer>
"""

def build_dataset(data, schemas, semantic_map, out_path, db_dir, reasoning_items=None):

  def process_item(item):
    question = item["question"]
    evidence = item["evidence"]
    sql = item["SQL"]
    db_id = item["db_id"]
    reasoning = None

    if reasoning_items:
      for i in range(len(reasoning_items)):
        if question == reasoning_items[i]["question"]:
          reasoning = reasoning_items[i]["reasoning"]
          del reasoning_items[i]
          break

    subschema = extract_subschema(
      question,
      evidence,
      schemas[db_id],
      db_path=f"{db_dir}/{db_id}/{db_id}.sqlite",
      schema_semantic_map=semantic_map[db_id]
    )

    input_seq = generate_final_prompt(
      question,
      evidence,
      schema_to_string(subschema, mode="ddl")
    )

    out_seq = sql
    '''if reasoning:
      out_seq = f"<think>{reasoning}</think><answer>{out_seq}</answer>"'''

    return {
      "input_seq": input_seq,
      "output_seq": out_seq,
      "db_desc": fix_unicode(schema_to_string(subschema, mode="ddl")),
      "question": normalize_question(evidence)+ '\n' + normalize_question(question) if normalize_question(evidence) != "" else normalize_question(question),
    }

  final_data = []

  with ThreadPoolExecutor(max_workers=32) as executor:
    for result in tqdm(executor.map(process_item, data), total=len(data)):
      final_data.append(result)

  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
  args = parse_args()
  
  data = load_samples(args.data_path, args.thinking_path)
  db_paths = []

  for root, dirs, files in os.walk(args.db_dir):
    for file in files:
      if file.endswith(".sqlite"):
        db_paths.append(os.path.join(root, file))

  schemas = load_schemas(
    db_paths=db_paths,
    column_meaning_path=args.column_meaning_path,
    cached_schemas_path=args.cached_schemas_path
  )
  semantic_map = buid_semantic_map(table_path=args.table_path)
  
  '''with open("temp.json", "w", encoding='utf-8') as f:
    json.dump(schemas, f, indent=2, ensure_ascii=False)'''

  reasoning_items = None
  if args.reasoning_path:
    with open(args.reasoning_path, "r", encoding='utf-8') as f:
      reasoning_items = json.load(f)

  build_dataset(data, schemas=schemas, semantic_map=semantic_map, out_path=args.out_path, reasoning_items=reasoning_items, db_dir=args.db_dir)