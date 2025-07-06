# from datasets import load_dataset
import json
import os
import numpy as np

# from langdetect import detect
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cache = {}

def load_model_tokenizer(src_lang, tgt_lang):
    key = f"{src_lang}-{tgt_lang}"
    if key in model_cache:
        return model_cache[key]

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    if src_lang == 'ro':
        src_lang = 'roa'
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    
    model_cache[key] = (model, tokenizer)
    return model, tokenizer

def backtranslate(batch_texts, curr_lang, next_lang):
    model_forward, tokenizer_forward = load_model_tokenizer(curr_lang, next_lang)
    model_backward, tokenizer_backward = load_model_tokenizer(next_lang, curr_lang)

    with torch.no_grad():
        batch_forward = tokenizer_forward(batch_texts, return_tensors="pt",
                                          padding=True, truncation=True).to(device)
        gen_forward = model_forward.generate(**batch_forward, max_length=512)
        texts_forward = tokenizer_forward.batch_decode(gen_forward.cpu(), skip_special_tokens=True)

        batch_backward = tokenizer_backward(texts_forward, return_tensors="pt",
                                            padding=True, truncation=True).to(device)
        gen_backward = model_backward.generate(**batch_backward, max_length=512)
        texts_backward = tokenizer_backward.batch_decode(gen_backward.cpu(), skip_special_tokens=True)

    del batch_forward, gen_forward, batch_backward, gen_backward
    torch.cuda.empty_cache()

    return texts_backward, texts_forward

def augment(data, batch_prompts, batch_responses, batch_keys):
    for lang in ['ru', 'de', 'fr', 'ro']:
        try:
            translated_prompts, intermediate_prompts = backtranslate(batch_prompts, 'en', lang)
            translated_responses, intermediate_responses = backtranslate(batch_responses, 'en', lang)
            for key, translated_prompt, intermediate_prompt, translated_response, intermediate_response in zip(batch_keys, translated_prompts, intermediate_prompts, translated_responses, intermediate_responses):
                if 'orig' not in data[key]:
                    data[key]['orig'] = [data[key]['ori']]
                if 'intermediate' not in data[key]:
                    data[key]['intermediate'] = []
                if 'translated' not in data[key]:
                    data[key]['translated'] = []
                data[key]['intermediate'].append([(intermediate_prompt, intermediate_response)])
                data[key]['translated'].append([(translated_prompt, translated_response)])
        except Exception as e:
            print(f"Error translating to/from {lang}: {e}", flush=True)
            continue
 
def format_as_json():
    dst_path = './data/wildguardmix_RH_translate_MarianMT'
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
    ds_valid = pd.read_csv("/data/wildguard_dev.csv")
    ds_train_full = pd.read_csv("/data/wildguard_train_part_4.csv")


    print(f'Size of train set is {len(ds_train_full)}\n')
    print("Training set class counts:")
    print(ds_train_full[label_column].value_counts())

    print(f'\nSize of valid set is {len(ds_valid)}\n')
    print("Valid set class counts:")
    print(ds_valid[label_column].value_counts())

    datasets = {
       'train_4': ds_train_full # TODO: ds_train_full,
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

            batchsize = 128

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
                if split_name in ['train_4']:
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