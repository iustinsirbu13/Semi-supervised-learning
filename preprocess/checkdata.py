import ijson
import os
import json
from calculate_similarity import get_similarity
import matplotlib.pyplot as plt

dst_path = '../../data/wildguardmix_RH_paraphrase_mistral_2aug'
split = 'train'

file_path = os.path.join(dst_path, f'{split}.json')

chosen_key = 0
chosen_entry = ""
maxlen = 0
entries = {}
similarities = []
token_counts = []

with open(file_path, 'r') as datafile:
    parser = ijson.kvitems(datafile, "")
    cnt = 0
    for _ in range(10):
        key, value = next(parser)

    print(cnt)
    entries[key] = value
    while maxlen < 500:
        break
        key, value = next(parser)
        token_count, similarity = get_similarity(value)
        # print(value)
        print(token_count)
        similarities.append(similarity)
        token_counts.append(token_count)
        print(maxlen)
        entries[maxlen] = value
        maxlen += 1


with open("test.json", 'w') as outfile:
    json.dump(entries, outfile)

# plt.figure(figsize=(10, 6))
# plt.scatter(token_counts, similarities, alpha=0.7)
# plt.xlabel('Number of Tokens')
# plt.ylabel('Similarity')
# plt.title('Similarity vs. Number of Tokens')
# plt.grid(True)
# plt.show()