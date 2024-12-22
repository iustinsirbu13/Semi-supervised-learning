from datasets import load_dataset
import json
import os
import numpy as np
from eda import eda
from langdetect import detect
from collections import Counter

def format_as_json():
    dst_path = './data/XSTest_RH'
    prompt_column = 'prompt'
    response_column = 'response'
    label_column = 'label'
    seed = 1234567
    labels = {
        'unharmful': 0,
        'harmful': 1,
    }
    os.makedirs(dst_path, exist_ok=True)

    ds = load_dataset("allenai/xstest-response")
    datasets = {
       'dev': None,
       'test': ds['response_harmfulness'],
       'train': None # TODO: ds_train_full,
    }

    
    for split_name, split_ds in datasets.items():
        data = {}
        cnt = 0
        
        with open(os.path.join(dst_path, f'{split_name}.json'), 'w') as outfile:
            if split_ds is not None:
                for idx, elem in enumerate(split_ds.filter(lambda example: example[label_column] is not None
                                                        and len(example[prompt_column]) > 0 and len(example[response_column]) > 0)) :
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
                        probs = [0.0, 0.0, 0.0, 0.0]
                        probs[np.random.randint(0, 3)] = 0.2
                        try: 
                            # print(data[str(idx)]['ori'])
                            # syn = eda(data[str(idx)]['ori'], 0.2, 0.0, 0.0, 0.0, 1)
                            # print(syn[0])
                            # exit()
                            data[str(idx)]['aug_0'] = [eda(data[str(idx)]['ori'][0], 0.2, 0.0, 0.0, 0.0, 1)[0], eda(data[str(idx)]['ori'][1], 0.2, 0.0, 0.0, 0.0, 1)[0]]
                            data[str(idx)]['aug_1'] = [eda(data[str(idx)]['ori'][0], probs[0], probs[1], probs[2], probs[3], 1)[0], eda(data[str(idx)]['ori'][1], probs[0], probs[1], probs[2], probs[3], 1)[0]]
                        except Exception as e:
                            print("language not supported")
                            raise e
            json.dump(data, outfile)

    # for lang, count in langs.items():
    #     print(f'language: {lang}, count: {count}')

if __name__ == '__main__':
    format_as_json()