from datasets import load_dataset
import json
import os
ds_train = load_dataset("allenai/wildguardmix", "wildguardtrain", split='train')
ds_train = ds_train.filter(lambda example: example['response'] is not None and example['response'].strip() != "")

from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', truncation_side='left')

from tqdm import tqdm
lens = [len(tokenizer(ds_train['response'][idx])['input_ids']) for idx in tqdm(range(len(ds_train['response'])))]

import pandas as pd
lens_ds = pd.Series(lens)

# (lens_ds < 512).value_counts(normalize=True)

percentage_below = (lens_ds < 1024).value_counts(normalize=True).get(True, 0) * 100
print(f"Percentage of sequences with length < 512: {percentage_below:.2f}%")

lens_ds.describe(percentiles=[0.1, 0.9, 0.95, 0.99])