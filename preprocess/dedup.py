import pandas as pd
import json
import os
from tqdm import tqdm

input_file = '../../data/wildguardmix_PH_paraphrase/train.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

seen = set()
deduped_data = {}

for key, entry in data.items():
    orig_text = entry['orig'][0]
    if orig_text not in seen:
        seen.add(orig_text)
        deduped_data[key] = entry

output_file = "../../data/wildguardmix_PH_paraphrase_catsplit/train.json"
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(deduped_data, outfile, ensure_ascii=False, indent=2)
