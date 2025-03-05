from datasets import load_dataset
import json
import os
import numpy as np
from eda import eda
from langdetect import detect
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

def check_lang(text):
    try:
        return detect(text) == 'en'
    except Exception:
        return False 

def process_split(ds, label_column, cat_column):
    print(ds.shape)
    ds = ds[
            (ds[cat_column] != 'Needs Caution') &
            (ds[cat_column] != 'Other')]
    print(ds.shape)
    print(ds[cat_column].value_counts())
    return ds

def format_as_json():
    dst_path = './data/aegis2.0_PH'
    text_column = 'prompt'
    cat_column = 'violated_categories'
    label_column = 'prompt_label'
    seed = 1234567
    labels = {
        'safe' : '0',
        'unsafe' : '1'
    }
    os.makedirs(dst_path, exist_ok=True)

    
    splits = {'train': 'train.json', 'validation': 'validation.json', 'test': 'test.json'}
    ds_train = pd.read_json("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/" + splits["train"])
    ds_test = pd.read_json("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/" + splits["test"])
    ds_valid = pd.read_json("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/" + splits["validation"])

    ds_train = process_split(ds_train, label_column, cat_column)
    print(ds_train.shape)

    ds_valid = process_split(ds_valid, label_column, cat_column)
    print(ds_valid.shape)

    ds_test = process_split(ds_test, label_column, cat_column)
    print(ds_test.shape)

    datasets = {
       'test': ds_test,
       'dev' : ds_valid,
       'train_lb': ds_train
    }

    langs = Counter()
    
    for split_name, split_ds in datasets.items():
        data = {}
        cnt = 0
        with open(os.path.join(dst_path, f'{split_name}.json'), 'w') as outfile:
            print(split_ds.shape)
            filtered_ds = split_ds[
                (split_ds[label_column].notna()) &
                (split_ds[text_column].str.len() > 0) &
                (split_ds[text_column].apply(check_lang))
            ]
            print(filtered_ds.shape)

            for idx, (index, elem) in enumerate(tqdm(filtered_ds.iterrows(), total=len(filtered_ds), desc=f'Processing {split_name}')):
                data[str(idx)] = {}
                data[str(idx)]['ori'] = elem[text_column]
                try:
                    data[str(idx)]['label'] = labels[elem[label_column]]
                    # data[str(idx)]['label_binary'] = str(elem[label_column])
                except Exception as e:
                    print(f"Label not in dictionary: {elem[cat_column]}")
                    raise e
                if split_name in ['train', 'train_lb', 'train_ulb']:
                    if len(data[str(idx)]['ori']) == 0:
                        continue
                    probs = [0.1, 0.1, 0.1, 0.1]
                    try: 
                        # print(data[str(idx)]['ori'])
                        # syn = eda(data[str(idx)]['ori'], 0.2, 0.0, 0.0, 0.0, 1)
                        # print(syn[0])
                        # exit()
                        # data[str(idx)]['eda_synonym'] = eda(data[str(idx)]['ori'], 0.0, 0.0, 0.0, 0.0, per_technique=True)
                        data[str(idx)]['eda_synonym'] = [data[str(idx)]['ori']]
                        data[str(idx)]['eda_full'] = eda(data[str(idx)]['ori'], probs[0], probs[1], probs[2], probs[3], num_aug=12)
                    except Exception as e:
                        print("language not supported")
                        raise e
                cnt += 1
            print(cnt)
            json.dump(data, outfile)

    # for lang, count in langs.items():
    #     print(f'language: {lang}, count: {count}')

if __name__ == '__main__':
    format_as_json()