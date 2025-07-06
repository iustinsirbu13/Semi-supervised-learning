import pandas as pd
import numpy as np

df = pd.read_csv('./data/wildguardmix_orig_test/wildguard_train_part_4.csv')
# df = df[df['response'].notna()]
# ds_train = ds_train[ds_train['response'].notna()]
print(f'Size of train set is {len(df)}\n')
print("Training set class counts:")
print(df['prompt_harm_label'].value_counts())

dfs = np.array_split(df, 2)

for i, split_df in enumerate(dfs, start=1):
    split_df.to_csv(f'./data/wildguardmix_orig_test/wildguard_train_part_4_{i}.csv', index=False)

print("CSV file has been split into 4 parts successfully.")