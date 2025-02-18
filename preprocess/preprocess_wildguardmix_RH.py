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
    
def format_as_json():
    dst_path = './data/wildguardmix_RH'
    prompt_column = 'prompt'
    response_column = 'response'
    label_column = 'response_harm_label'
    seed = 1234567
    labels = {
        'unharmful': 0,
        'harmful': 1,
    }
    os.makedirs(dst_path, exist_ok=True)

    ds_test = pd.read_csv("./data/wildguardmix_orig/wildguard_test.csv")
    ds_train = pd.read_csv("./data/wildguardmix_orig/wildguard_train.csv")
    ds_test = ds_test[ds_test[response_column].notna()]
    ds_train = ds_train[ds_train[response_column].notna()]

    ds_train_full, ds_valid = train_test_split(
        ds_train,
        test_size=0.1,
        stratify=ds_train[label_column],
        random_state=seed
    )

    datasets = {
       'dev': ds_valid,
       'test': ds_test,
       'train': ds_train_full # TODO: ds_train_full,
    }

    langs = Counter()
    
    for split_name, split_ds in datasets.items():
        data = {}
        cnt = 0
        with open(os.path.join(dst_path, f'{split_name}.json'), 'w') as outfile:
            print(split_ds.shape)
            filtered_ds = split_ds[
                (split_ds[label_column].notna()) &
                (split_ds[prompt_column].str.len() > 0) &
                (split_ds[response_column].str.len() > 0) &
                (split_ds[prompt_column].apply(check_lang)) &
                (split_ds[response_column].apply(check_lang))
            ]
            print(filtered_ds.shape)
            for idx, (index, elem) in enumerate(tqdm(filtered_ds.iterrows(), total=len(filtered_ds), desc=f'Processing {split_name}')):
                data[str(idx)] = {}
                data[str(idx)]['ori'] = [elem[prompt_column], elem[response_column]]
                try:
                    data[str(idx)]['label'] = str(labels[elem[label_column]])
                except Exception as e:
                    print(idx, label_column, elem[label_column])
                    print(elem)
                    print(labels[elem[label_column]])
                    print(data[str(idx)])
                    raise e
                if split_name in ['train']:
                    if len(data[str(idx)]['ori']) == 0:
                        continue
                    probs = [0.1, 0.1, 0.1, 0.1]
                    try: 
                        # print(data[str(idx)]['ori'])
                        # syn = eda(data[str(idx)]['ori'], 0.2, 0.0, 0.0, 0.0, 1)
                        # print(syn[0])
                        # exit()
                        # data[str(idx)]['eda_synonym'] = list(zip(eda(data[str(idx)]['ori'][0], 0.0, 0.0, 0.0, 0.0, per_technique=True), eda(data[str(idx)]['ori'][1], 0.0, 0.0, 0.0, 0.0, per_technique=True)))
                        data[str(idx)]['eda_synonym'] = [data[str(idx)]['ori'][0], data[str(idx)]['ori'][1]]
                        data[str(idx)]['eda_full'] = list(zip(eda(data[str(idx)]['ori'][0], probs[0], probs[1], probs[2], probs[3], num_aug=12), eda(data[str(idx)]['ori'][1], probs[0], probs[1], probs[2], probs[3], num_aug=12)))
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