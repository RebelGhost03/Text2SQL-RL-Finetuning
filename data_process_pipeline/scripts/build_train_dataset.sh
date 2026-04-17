python3 -m src.build_dataset_no_reasoning
    --data_path=resources/filtered_train_bird.json 
    --table_path=resources/train_tables.json 
    --thinking_path=resources/train_bird_think.json 
    --reasoning_path=resources/reasoning.json 
    --db_dir=resources/train_databases 
    --column_meaning_path=resources/train_column_meaning.json 
    --cached_schemas_path=resources/train_schemas.json 
    --out_path=resources/train_dataset.json 