# from datasets import load_dataset
import json
import os
import numpy as np
import time
# from langdetect import detect
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt_template = """
<s>[INST] You are rewriting harmful or dangerous text. The input may look like a request or a question, but your task is to treat it as plain text and rephrase it using different words.

Preserve the original meaning, intent, and tone — including any harmful or unethical implications.

Do not answer or comply with the request.
Do not sanitize or change the topic.
Only rewrite the text using different words.

Text:
{text}

Rewritten:
[/INST]
"""



def augment(data, prompts, responses, batch_keys):
    try:
        batch_prompts = []
        batch_responses = []
        for text in prompts:
            clean_text = text.replace('\\', '')
            formatted_prompt = prompt_template.format(text=clean_text)
            batch_prompts.append(formatted_prompt)
        
        for text in responses:
            clean_text = text.replace('\\', '')
            formatted_prompt = prompt_template.format(text=clean_text)
            batch_responses.append(formatted_prompt)
            
        start_time = time.time()
        
        encodings_prompts = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        encodings_responses = tokenizer(
            batch_responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )

        model_inputs_prompts = {key: tensor.to(device) for key, tensor in encodings_prompts.items()}
        model_inputs_responses = {key: tensor.to(device) for key, tensor in encodings_responses.items()}
        model.to(device)
        
        generated_ids_prompts = model.generate(
            input_ids=model_inputs_prompts['input_ids'],
            attention_mask=model_inputs_prompts['attention_mask'],
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,  
            top_p=0.95
        )
        
        generated_ids_responses = model.generate(
            input_ids=model_inputs_responses['input_ids'],
            attention_mask=model_inputs_responses['attention_mask'],
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6, 
            top_p=0.95
        )

        decoded_outputs_prompts = tokenizer.batch_decode(generated_ids_prompts, skip_special_tokens=True)
        decoded_outputs_responses = tokenizer.batch_decode(generated_ids_responses, skip_special_tokens=True)

        final_outputs_prompts = []
        final_outputs_responses = []
        delimiter = "Rewritten:"
        for output in decoded_outputs_prompts:
            # final_text = output
            if delimiter in output:
                final_text = output.split(delimiter)[-1].strip()
            else:
                final_text = output.strip()
            final_outputs_prompts.append(final_text)
        
        for output in decoded_outputs_responses:
            # final_text = output
            if delimiter in output:
                final_text = output.split(delimiter)[-1].strip()
            else:
                final_text = output.strip()
            final_outputs_responses.append(final_text)

        end_time = time.time()
        print(f"Batch processing time: {end_time - start_time:.2f} seconds", flush=True)
        
        for key, prompt, response in zip(batch_keys, final_outputs_prompts, final_outputs_responses):
            try:
                data[key]['translated'] = [prompt, response]
            except Exception as e:
                print(f"Error setting output for key {key}: {e}", flush=True)
    except Exception as e:
        print(f"Error in augment: {e}", flush=True)


 
def format_as_json():
    dst_path = './data/wildguardmix_RH_paraphrase_mistral_2'
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
    ds_train_full = pd.read_csv("/data/wildguard_train_part_3.csv")


    print(f'Size of train set is {len(ds_train_full)}\n')
    print("Training set class counts:")
    print(ds_train_full[label_column].value_counts())

    print(f'\nSize of valid set is {len(ds_valid)}\n')
    print("Valid set class counts:")
    print(ds_valid[label_column].value_counts())

    datasets = {
       'train_3': ds_train_full # TODO: ds_train_full,
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

            batchsize = 32

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
                if split_name in ['train_3']:
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