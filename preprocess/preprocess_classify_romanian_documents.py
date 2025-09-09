import json
import jsonlines
import os
import pandas as pd
import time
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--text_column", type=str, default='text')
    parser.add_argument("--new_column", type=str, default='generated')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--system_prompt", type=str, default='sp1')
    parser.add_argument("--model_id", type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--api", type=str, default='pipeline')
    args = parser.parse_args()
    return args


system_prompt2 = """Ești un asistent specializat în drept administrativ. Sarcina ta este să clasifici documente preluate de pe site-urile primăriilor.

Definiții:
* Legislativ: Documente cu caracter de lege, hotărâre sau reglementare generală și obligatorie. Acestea stabilesc norme, drepturi și obligații pentru cetățeni sau pentru anumite grupuri (ex: regulamente, hotărâri de consiliu local, dispoziții cu caracter normativ). În această categorie intră și documentele care discută sau rezumă hotărârile luate în documente legislative, fără a fi ele însăși scrise sub forma de document legislativ.
* Ne-legislativ: Documente care nu au caracter de reglementare și nu stabilesc norme generale. Acestea se referă la situații specifice, la un singur caz sau la un număr limitat de persoane (ex: certificate, autorizații, adeverințe, avize, dispoziții cu caracter individual). Include în aceasta categorie și documentele financiare sau alte tipuri de rapoarte generale, care nu au caracter de reglementare. Include aici și documente care se referă la știri despre subiecte de interes general, dar care nu au vreo legătură cu o hotărâre sau reglementare generală.

Instrucțiuni stricte:
1. Răspunde doar cu exact unul dintre cele două cuvinte:
   - Legislativ
   - Ne-legislativ
2. Nu adăuga explicații, nu reformula, nu insera alt text.
3. Orice alt răspuns este invalid.
"""

system_prompt1 = """Ești un asistent specializat în drept administrativ. Sarcina ta este să clasifici documente preluate de pe site-urile primăriilor.

Instrucțiuni stricte:
1. Răspunde doar cu exact unul dintre cele două cuvinte:
   - Legislativ
   - Ne-legislativ
2. Nu adăuga explicații, nu reformula, nu insera alt text.
3. Orice alt răspuns este invalid.
"""


def augment(batch_texts, pipe, args):
    system_prompt = system_prompt2 if args.system_prompt == 'sp2' else system_prompt1
    batch_messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        for user_text in batch_texts
    ]
    outputs = pipe(batch_messages, max_new_tokens=args.max_new_tokens)
    return [output[0]["generated_text"][-1]['content'] for output in outputs]

def augment_with_type(batch_texts, pipe, args):
    system_prompt = system_prompt2 if args.system_prompt == 'sp2' else system_prompt1
    batch_messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": system_prompt},]},
            {"role": "user", "content": [{"type": "text", "text": user_text},]}
        ]
        for user_text in batch_texts
    ]
    outputs = pipe(batch_messages, max_new_tokens=args.max_new_tokens)
    return [output[0]["generated_text"][-1]['content'] for output in outputs]

def augment_user_only(batch_texts, pipe, args):
    batch_messages = [
        [
            {"role": "user", "content": user_only_prompt.format(text=user_text)}
        ]
        for user_text in batch_texts
    ]
    outputs = pipe(batch_messages, max_new_tokens=args.max_new_tokens)
    return [output[0]["generated_text"][-1]['content'] for output in outputs]


def augment_generate(batch_texts, model, tokenizer, args):
    system_prompt = system_prompt2 if args.system_prompt == 'sp2' else system_prompt1
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        for user_text in batch_texts
    ]

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True)

    try:
        return [tokenizer.decode(output[input.shape[0]:], skip_special_tokens=True) for output, input in zip(outputs, inputs['input_ids'])]
    except Exception as e:
        for output, input in zip(outputs, inputs['input_ids']):
            print('input shape', input.shape)
            print('output shape', output.shape)
            raise e

if __name__ == '__main__':
    args = parse_args()
    print('args', args)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    if args.api == 'pipeline':
        if args.model_id in ['google/gemma-2-2b-it', 'google/gemma-2-2b-it-pytorch']:
            raise NotImplementedError()
            pipe = pipeline("text-generation", model=args.model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda")
            augment_fn = augment_user_only
        elif args.model_id in ['google/gemma-3-1b-it']:
            pipe = pipeline("text-generation", model=args.model_id, torch_dtype=torch.bfloat16, device_map="cuda")
            augment_fn = augment_with_type
        else:
            pipe = pipeline("text-generation", model=args.model_id, device_map="cuda")
            augment_fn = augment
    elif args.api == 'generate':
        # tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
        print('Loading model and tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", local_files_only=True)
        print('Loaded model and tokenizer.')
        augment_fn = augment_generate
    else:
        raise NotImplementedError()

    # df = pd.read_csv(args.input_csv)
    # df[args.text_column] = df[args.text_column].fillna('')

    df = pd.read_csv(args.input_csv, usecols=['id', 'ori', 'label', 'decoded'])
    # df[args.text_column] = df[args.text_column].map(lambda x: x[:2500] if isinstance(x, str) else '\n')

    # Check existing output
    if os.path.exists(args.output_csv):
        df_done = pd.read_csv(args.output_csv)
        start_idx = len(df_done)
        print(f"Resuming from index {start_idx}")
    else:
        df_done = pd.DataFrame(columns=df.columns.tolist() + [args.new_column])
        start_idx = 0

    # Process in batches
    for i in tqdm(range(start_idx, len(df), args.batch_size)):
        batch = df.iloc[i:i+args.batch_size].copy()
        if args.api == 'pipeline':
            batch[args.new_column] = augment_fn(batch[args.text_column], pipe, args)
        elif args.api == 'generate':
            batch[args.new_column] = augment_fn(batch[args.text_column], model, tokenizer, args)
        else:
            raise NotImplementedError()
        # df_done = pd.concat([df_done, batch], ignore_index=True)
        # Save progress
        # df_done.to_csv(args.output_csv, index=False)
        header = not os.path.exists(args.output_csv)
        batch.to_csv(args.output_csv, mode='a', header=header, index=False)
        print(f"Processed up to index {i + args.batch_size - 1}")


