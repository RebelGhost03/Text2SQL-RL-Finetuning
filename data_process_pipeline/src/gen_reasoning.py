import argparse
from typing import Dict, List
import json
import os

from utils.sql import extract_base_schema
from utils.sqlite import schema_to_string, filter_schema

from .build_dataset import load_schemas

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

def generate_reasoning(data: List[Dict], example: Dict, schemas: Dict):
    for item in data:
        prompt = generate_reasoning_prompt(item, example, schemas)
        item["prompt"] = prompt
    
    return data

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--db_dir", type=str)
    parser.add_argument("--column_meaning_path", type=str)
    parser.add_argument("--cached_schemas_path", type=str)
    parser.add_argument("--out_path", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    db_paths = []
    if args.db_dir:
        for root, dirs, files in os.walk(args.db_dir):
            for file in files:
                if file.endswith(".sqlite"):
                    db_paths.append(os.path.join(root, file))

    schemas = load_schemas(
        db_paths=db_paths,
        column_meaning_path=args.column_meaning_path,
        cached_schemas_path=args.cached_schemas_path
    )

    with open(args.data_path, "r") as f:
        data = json.load(f)

    example = data["example"]
    data = data["data"]

    for item in data:
        prompt = generate_reasoning_prompt(item, example, schemas)
        item["prompt"] = prompt
    
    with open(args.out_path, "w") as f:
        json.dump(data, f, indent=2)


