import ijson
import os
import json
from calculate_similarity import get_similarity
import matplotlib.pyplot as plt

dst_path = '../../data/wildguardmix_RH_paraphrase_mistral_2'
files = ['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'train_6', 'train_7', 'train_8']

entries = {}
cnt = 0
bad = 0

for file in files:
    file_path = os.path.join(dst_path, f'{file}.json')
    with open(file_path, 'r') as datafile:
        parser = ijson.kvitems(datafile, "")
        while True:
            try:
                key, value = next(parser)
            except StopIteration:
                break
            print(key)
            value['orig'] = [value['ori']]
            if 'translated' not in value or len(value['translated']) == 0:
                value['translated'] = [value['ori']]
                bad += 1
            else:
                value['translated'] = [value['translated']]
            entries[str(cnt)] = value   
            cnt += 1 

print(f'{bad} bad entries\n')
print(f'merged files to a total of {cnt} entries\n')

with open(f"{dst_path}/train.json", 'w') as outfile:
    json.dump(entries, outfile)

# plt.figure(figsize=(10, 6))
# plt.scatter(token_counts, similarities, alpha=0.7)
# plt.xlabel('Number of Tokens')
# plt.ylabel('Similarity')
# plt.title('Similarity vs. Number of Tokens')
# plt.grid(True)
# plt.show()