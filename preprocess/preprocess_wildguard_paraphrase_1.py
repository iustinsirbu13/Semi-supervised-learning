import json
import os
import numpy as np
from collections import Counter
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import pipeline
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def check_lang(text):
#     try:
#         return detect(text) == 'en'
#     except Exception:
#         return False 
model_id = "meta-llama/Llama-3.2-3B-Instruct"

system_prompt = """
You are a text rewriter for a safety-focused research project. You rewrite harmful, toxic, or dangerous language by paraphrasing it. You must preserve the original meaning, tone, and harmful intent. You are not censoring, sanitizing, or judging the text — just restating it with different words. All rewrites are used for training models to detect dangerous content.
"""

prompt_template = """
You will be given a harmful or toxic text sample. Your task is to rephrase the sample using different wording, while fully preserving its original **meaning, intent, and any harmful implications**. Rewrite fluently and naturally. Do NOT remove or soften any toxic or unsafe content — just change the surface phrasing.

The sample will be shown between START and END. Rewrite the text only. Do not add any extra content.

Example:

START
You should beat them until they obey.
END

Rewritten:
You ought to hit them repeatedly until they start following orders.

START
{text}
END
"""


pipe = pipeline("text-generation", model="huihui-ai/Llama-3.2-3B-Instruct-abliterated", device_map="cuda",)

def augment(data, batch_texts, batch_keys):
    try:
        batch_prompts = []
        for text in batch_texts:
            clean_text = text.replace('"', '').replace("'", "").replace('\\', '')
            formatted_prompt = prompt_template.format(text=clean_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]
            batch_prompts.append(messages)
        start_time = time.time()
        outputs = pipe(batch_prompts, max_new_tokens=512)
        end_time = time.time()
        print(end_time - start_time, flush=True)

        for key, output in zip(batch_keys, outputs):
            try:
                response_text = output[0]["generated_text"][-1]['content']
            except (KeyError, IndexError, TypeError):
                response_text = ""
            data[key]['translated'] = response_text
    except Exception as e:
        print(f"Error: {e}", flush=True)
    
def format_as_json():
    dst_path = './data/wildguardmix_PH_paraphrase_try2'
    text_column = 'prompt'
    label_column = 'prompt_harm_label'
    seed = 1234567
    labels = {
        'unharmful': 0,
        'harmful': 1,
    }
    os.makedirs(dst_path, exist_ok=True)

    # /data/wildguardmix_orig for local and /data for cluster
    # ds_test = pd.read_csv("./data/wildguardmix_orig/wildguard_test.csv")
    # ds_train = pd.read_csv("./data/wildguardmix_orig/wildguard_train.csv")

    ds_test = pd.read_csv("/data/wildguard_test.csv")
    ds_valid = pd.read_csv("/data/wildguard_dev.csv")
    ds_train_full = pd.read_csv("/data/wildguard_train_part_1.csv")

    # ds_train_full, ds_valid = train_test_split(
    #     ds_train,
    #     test_size=0.1,
    #     stratify=ds_train[label_column],
    #     random_state=seed
    # )

    print(f'Size of train set is {len(ds_train_full)}\n')
    print("Training set class counts:")
    print(ds_train_full[label_column].value_counts())

    print(f'\nSize of valid set is {len(ds_valid)}\n')
    print("Valid set class counts:")
    print(ds_valid[label_column].value_counts())

    datasets = {
       'train_1': ds_train_full # TODO: ds_train_full,
    }

    langs = Counter()
    batch_texts = []
    batch_keys = []
    bcount = 0

    for split_name, split_ds in datasets.items():
        data = {}
        cnt = 0
        with open(os.path.join(dst_path, f'{split_name}.json'), 'w') as outfile:
            filtered_ds = split_ds[
                (split_ds[label_column].notna()) &
                (split_ds[text_column].str.len() > 0)
            ]

            # if (split_name == 'train'):
            #     filtered_ds = filtered_ds[filtered_ds[text_column].apply(lambda x: len(en2de.encode(x)) < 512)]
            batchsize = 256
            
            for idx, (index, elem) in enumerate(tqdm(filtered_ds.iterrows(), total=len(filtered_ds), desc=f'Processing {split_name}')):
                data[str(idx)] = {}
                data[str(idx)]['ori'] = elem[text_column]
                try:
                    data[str(idx)]['label'] = str(labels[elem[label_column]])
                except Exception as e:
                    print(idx, label_column, elem[label_column])
                    print(elem)
                    print(labels[elem[label_column]])
                    print(data[str(idx)])
                    raise e
                if split_name in ['train_1']:
                    if len(data[str(idx)]['ori']) == 0:
                        continue
                    data[str(idx)]['translated'] = []
                    batch_texts.append(data[str(idx)]['ori'])
                    batch_keys.append(str(idx))
                    
                    if len(batch_texts) >= batchsize:
                        augment(data, batch_texts, batch_keys)
                        batch_texts = []
                        batch_keys = []
                        break
                cnt += 1
            if len(batch_texts):
                augment(data, batch_texts, batch_keys)
            print(f"Finished split {split_name}\n")
            print(cnt)
            json.dump(data, outfile)

    # for lang, count in langs.items():
    #     print(f'language: {lang}, count: {count}')

if __name__ == '__main__':
    format_as_json()