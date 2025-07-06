import pandas as pd
from langdetect import detect
from sklearn.model_selection import train_test_split

def check_lang(text):
    try:
        return detect(text) == 'en'
    except Exception:
        return False
        
# ds_test = pd.read_parquet("hf://datasets/allenai/wildguardmix/test/wildguard_test.parquet")
# ds_train = pd.read_parquet("hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet")
ds_train = pd.read_csv("./data/wildguardmix_orig/wildguard_train.csv")
# ds_test = ds_test[ds_test['response'].notna()]
# ds_train = ds_train[ds_train['response'].notna()]
# ds_test = ds_test[(ds_test['prompt'].apply(check_lang))]
# ds_train = ds_train[(ds_train['prompt'].apply(check_lang))] 

# ds_train['token_count'] = ds_train['prompt'].apply(lambda x: len(x.split()))
# count_less_than_1024 = (ds_train['token_count'] < 1024).sum()
# print("Count of prompts with less than 1024 tokens:", count_less_than_1024)
# print(ds_test['response_harm_label'].value_counts())
# print(ds_test.shape)
# print(ds_train.shape)
label_column = 'prompt_harm_label'
seed = 1234567
ds_train_full, ds_valid = train_test_split(
    ds_train,
    test_size=0.1,
    stratify=ds_train[label_column],
    random_state=seed
)

# ds_test.to_csv("./data/wildguardmix_orig_test/wildguard_test.csv", index=False)
# print("Test set downloaded\n")
ds_train_full.to_csv("./data/wildguardmix_orig_test/wildguard_train.csv", index=False)
print("Train set downloaded\n")
ds_valid.to_csv("./data/wildguardmix_orig_test/wildguard_dev.csv", index=False)
print("Valid set downloaded\n")