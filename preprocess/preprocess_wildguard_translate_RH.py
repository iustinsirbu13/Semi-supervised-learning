# from datasets import load_dataset
import json
import os
import numpy as np
from eda import eda
# from langdetect import detect
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from tqdm import tqdm

MODE = 'dev'
# Load translation model

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe', no_progress_bar=False, skip_invalid_size_inputs_valid_test=True)
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe', no_progress_bar=False, skip_invalid_size_inputs_valid_test=True)

en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe', no_progress_bar=False, skip_invalid_size_inputs_valid_test=True)
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe', no_progress_bar=False, skip_invalid_size_inputs_valid_test=True)

en2de.cuda()
de2en.cuda()

en2ru.cuda()
ru2en.cuda()

def augment(data, batch_prompts, batch_responses, batch_keys):
    probs = [0.1, 0.1, 0.1, 0.1]
    try: 
        # print(data[str(idx)]['ori'])
        # syn = eda(data[str(idx)]['ori'], 0.2, 0.0, 0.0, 0.0, 1)
        # print(syn[0])
        # exit()
        # data[str(idx)]['eda_synonym'] = list(zip(eda(data[str(idx)]['ori'][0], 0.0, 0.0, 0.0, 0.0, per_technique=True), eda(data[str(idx)]['ori'][1], 0.0, 0.0, 0.0, 0.0, per_technique=True)))
        for key, prompt, response in zip(batch_keys, batch_prompts, batch_responses):
            try:
                data[key]['eda_full'] = list(zip(eda(prompt, probs[0], probs[1], probs[2], probs[3], num_aug=12), eda(response, probs[0], probs[1], probs[2], probs[3], num_aug=12)))
            except:
                data[key]['eda_full'] = [prompt, response]
        
        try:
            de_translate_prompt = de2en.translate(en2de.translate(batch_prompts,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
            ru_translate_prompt = ru2en.translate(en2ru.translate(batch_prompts,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
            de_translate_response = de2en.translate(en2de.translate(batch_responses,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
            ru_translate_response = ru2en.translate(en2ru.translate(batch_responses,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
            de_translate = list(zip(de_translate_prompt, de_translate_response))
            ru_translate = list(zip(ru_translate_prompt, ru_translate_response))
        except Exception as e:
            de_translate = list(zip(batch_prompts, batch_responses))
            ru_translate = list(zip(batch_prompts, batch_responses))
        
        for key, de_text, ru_text in zip(batch_keys, de_translate, ru_translate):
            data[key]['orig'] = [data[key]['ori']]
            data[key]['eda_full'] = data[key]['eda_full'] + [de_text, ru_text]
    except Exception as e:
        print("language not supported")
        raise e
    
def format_as_json():
    dst_path = './data/wildguardmix_RH_translate+eda'
    prompt_column = 'prompt'
    response_column = 'response'
    label_column = 'response_harm_label'
    seed = 1234567
    labels = {
        'unharmful': 0,
        'harmful': 1,
    }
    os.makedirs(dst_path, exist_ok=True)

    ds_test = pd.read_csv("/data/wildguard_test.csv")
    ds_train = pd.read_csv("/data/wildguard_train.csv")

    ds_train = ds_train[ds_train[response_column].notna()]
    ds_test = ds_test[ds_test[response_column].notna()]

    ds_train_full, ds_valid = train_test_split(
        ds_train,
        test_size=0.1,
        stratify=ds_train[label_column],
        random_state=seed
    )


    print(f'Size of train set is {len(ds_train_full)}\n')
    print("Training set class counts:")
    print(ds_train_full[label_column].value_counts())

    print(f'\nSize of valid set is {len(ds_valid)}\n')
    print("Valid set class counts:")
    print(ds_valid[label_column].value_counts())

    datasets = {
       'dev': ds_valid,
       'test': ds_test,
       'train': ds_train_full # TODO: ds_train_full,
    }

    langs = Counter()
    batch_prompts = []
    batch_responses = []
    batch_keys = []

    for split_name, split_ds in datasets.items():
        data = {}
        cnt = 0
        with open(os.path.join(dst_path, f'{split_name}.json'), 'w') as outfile:
            print(split_ds.shape)
            filtered_ds = split_ds[
                (split_ds[label_column].notna()) &
                (split_ds[prompt_column].str.len() > 0) &
                (split_ds[response_column].str.len() > 0)
            ]

            if (split_name == 'train'):
                filtered_ds = filtered_ds[filtered_ds[response_column].apply(lambda x: (len(en2de.encode(x)) < 1024))]
                filtered_ds = filtered_ds[filtered_ds[prompt_column].apply(lambda x: (len(en2de.encode(x)) < 1024))]
            batchsize = 1024

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
                    batch_prompts.append(elem[prompt_column])
                    batch_responses.append(elem[response_column])
                    batch_keys.append(str(idx))
                    
                    if len(batch_prompts) >= batchsize:
                        augment(data, batch_prompts, batch_responses, batch_keys)
                        batch_prompts = []
                        batch_responses = []
                        batch_keys = []

                cnt += 1
            if len(batch_prompts):
                augment(data, batch_prompts, batch_responses, batch_keys)

            print(cnt)
            json.dump(data, outfile)

    # for lang, count in langs.items():
    #     print(f'language: {lang}, count: {count}')

if __name__ == '__main__':
    format_as_json()