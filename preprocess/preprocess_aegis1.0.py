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
    
def majority_label(row):
    counts = row[['labels_0', 'labels_1', 'labels_2']].value_counts()
    if counts.iloc[0] >= 2:
        return counts.idxmax()
    else:
        return np.nan
    
def format_as_json():
    dst_path = './data/aegis1.0_PH'
    text_column = 'text'
    cat_column = 'prompt_harm_label'
    label_column = 'safety'
    seed = 1234567
    labels = {
        'Safe' : '0',
        'Hate/Identity Hate' : '1',
        'Sexual' : '2',
        'Violence' : '3',
        'Suicide and Self Harm' : '4',
        'Threat' : '5',
        'Sexual Minor' : '6',
        'Guns/Illegal Weapons' : '7',
        'Controlled/Regulated substances' : '8',
        'Criminal Planning/Confessions' : '9',
        'PII' : '10',
        'Harassment' : '11',
        'Profanity' : '12',
        'Other' : '13',
        'Needs Caution' : '14'
    }
    os.makedirs(dst_path, exist_ok=True)

   
    splits = {'train': 'Content Moderation Extracted Annotations 02.08.24_train_release_0418_v1.parquet', 'test': 'Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet'}
    ds_train = pd.read_parquet("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/" + splits["train"])
    ds_test = pd.read_parquet("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/" + splits["test"])

    ds_train = ds_train[ds_train['text_type'] == 'user_message']
    ds_train[cat_column] = ds_train.apply(majority_label, axis=1)
    ds_train = ds_train.dropna(subset=[cat_column])
    ds_train[label_column] = np.where(ds_train[cat_column].eq('Safe'), '0', '1')
    ds_train = ds_train[(ds_train[cat_column].isin(labels))]

    ds_test = ds_test[ds_test['text_type'] == 'user_message']
    ds_test[cat_column] = ds_test.apply(majority_label, axis=1)
    ds_test = ds_test.dropna(subset=[cat_column])
    ds_test[label_column] = np.where(ds_test[cat_column].eq('Safe'), '0', '1')
    ds_test = ds_test[(ds_test[cat_column].isin(labels))]

    ds_train_full, ds_valid = train_test_split(
        ds_train,
        test_size=400,
        stratify=ds_train[label_column],
        random_state=seed
    )

    print("Class distribution for train:\n")
    print(ds_train[cat_column].value_counts())
    print("Class distribution for test:\n")
    print(ds_test[cat_column].value_counts())
    print(ds_train.shape)
    print(ds_test.shape)
    exit()
    datasets = {
       'test': ds_test,
       'dev' : ds_valid,
       'train_lb': ds_train_full
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
                # & (split_ds[cat_column].isin(labels))
            ]
            print(filtered_ds.shape)

            for idx, (index, elem) in enumerate(tqdm(filtered_ds.iterrows(), total=len(filtered_ds), desc=f'Processing {split_name}')):
                data[str(idx)] = {}
                data[str(idx)]['ori'] = elem[text_column]
                try:
                    data[str(idx)]['label'] = elem[label_column]
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