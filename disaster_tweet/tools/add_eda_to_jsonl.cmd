SET DATA_PATH=C:\Users\Robert_Popovici\Desktop\licenta\datasets

python add_eda_to_jsonl.py ^
    --input_file "%DATA_PATH%\humanitarian_orig\train_eda.jsonl" ^
    --output_file "%DATA_PATH%\humanitarian\train_eda.jsonl" ^
    --input_file "%DATA_PATH%\humanitarian_orig\test_eda.jsonl" ^
    --output_file "%DATA_PATH%\humanitarian\test_eda.jsonl" ^
    --input_file "%DATA_PATH%\humanitarian_orig\dev_eda.jsonl" ^
    --output_file "%DATA_PATH%\humanitarian\dev_eda.jsonl" ^
    --input_file "%DATA_PATH%\humanitarian_orig\unlabeled_eda.jsonl" ^
    --output_file "%DATA_PATH%\humanitarian\unlabeled_eda.jsonl" ^
    --input_file "%DATA_PATH%\informative_orig\train_eda.jsonl" ^
    --output_file "%DATA_PATH%\informative\train_eda.jsonl" ^
    --input_file "%DATA_PATH%\informative_orig\test_eda.jsonl" ^
    --output_file "%DATA_PATH%\informative\test_eda.jsonl" ^
    --input_file "%DATA_PATH%\informative_orig\dev_eda.jsonl" ^
    --output_file "%DATA_PATH%\informative\dev_eda.jsonl" ^
    --input_file "%DATA_PATH%\informative_orig\unlabeled_eda.jsonl" ^
    --output_file "%DATA_PATH%\informative\unlabeled_eda.jsonl"