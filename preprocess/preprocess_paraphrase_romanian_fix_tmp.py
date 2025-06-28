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
    parser.add_argument("--new_column", type=str, default='augmented')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--system_prompt", type=str, default='sp1')
    parser.add_argument("--model_id", type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--api", type=str, default='pipeline')
    args = parser.parse_args()
    return args

# system_prompt = """Ești un asistent folositor, capabil să reformulezi un text primit astfel încât să modifici forma sa, fără a îi schimba sensul.
# De exemplu, poți să înlocuiești unele cuvinte cu sinonime, să corectezi greșeli de scriere, sau să elimini caractere nedorite. Pentru fiecare prompt primit, vei returna o astfel de reformulare, fără explicații sau alte informații adiționale."""

system_prompt1 = (
            "Ești un asistent util care reformulează texte în limba română. "
            "Pentru fiecare mesaj primit, returnezi o reformulare a textului care păstrează sensul original, dar modifică forma. "
            "De exemplu, poți înlocui cuvinte cu sinonime, corecta greșeli de scriere și elimina caractere nedorite. "
            "Nu oferi explicații sau informații adiționale, doar reformularea."
        )


system_prompt2 = (
            "Ești un asistent util care reformulează texte în limba română. "
            "Pentru fiecare mesaj primit, returnezi o reformulare a textului care păstrează sensul original, dar modifică forma. "
            "De exemplu, poți înlocui cuvinte cu sinonime, corecta greșeli de scriere și elimina caractere nedorite. "
            "Mai mult chiar, poți face și reformulări ample sau sumarizări, atâta timp cat înțelesul textului nu este afectat. "
            "Nu oferi explicații sau informații adiționale, doar reformularea."
        )

user_only_prompt = (
            "Ești un asistent util care reformulează texte în limba română. "
            "Pentru fiecare mesaj primit, returnezi o reformulare a textului care păstrează sensul original, dar modifică forma. "
            "De exemplu, poți înlocui cuvinte cu sinonime, corecta greșeli de scriere și elimina caractere nedorite. "
            "Mai mult chiar, poți face și reformulări ample sau sumarizări, atâta timp cat înțelesul textului nu este afectat. "
            "Mesajul care trebuie reformulat este textul cuprins intre tokenii <START> și <STOP>. "
            "Nu oferi explicații sau informații adiționale, doar reformularea textului, fără toenii <START> și <STOP>. "
            "<START> {text} <STOP>"
)


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

    df = pd.read_csv(args.input_csv)
    print('read input file')

    df_done = pd.read_csv(args.output_csv)
    print('read done file')
    df_cleaned = df_done.drop_duplicates(subset="id", keep="first")
    print('cleaned df')

    df_expected = df.iloc[:df_cleaned.shape[0]]

    assert df_cleaned['id'].reset_index(drop=True).equals(df_expected['id'].reset_index(drop=True))
    assert df_cleaned['text'].reset_index(drop=True).equals(df_expected['text'].reset_index(drop=True).fillna('\n'))

    print('asserts passed')
    df_cleaned.to_csv(args.output_csv, index=False)
    print('done')