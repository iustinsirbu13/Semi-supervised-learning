import ijson
import os
import json
from calculate_similarity import get_similarity
import matplotlib.pyplot as plt

dst_path1 = '../../data/wildguardmix_PH_translate_MarianMT_test'
dst_path2 = '../../data/wildguardmix_PH_translate'
file = 'train'

entries = {}
cnt = 0

file_path1 = os.path.join(dst_path1, f'{file}.json')
file_path2 = os.path.join(dst_path2, f'{file}.json')

with open(file_path1, 'r') as datafile:
    data1 = json.load(datafile)

with open(file_path2, 'r') as datafile:
    data2 = json.load(datafile)

for key in data1:
    print(key)
    if data1[key]['ori'] != data2[key]['ori']:
        print(f'different on pos {key}\n')
        print(data1[key]['ori'])
        print('\n')
        print(data1[key]['ori'])
        break


# plt.figure(figsize=(10, 6))
# plt.scatter(token_counts, similarities, alpha=0.7)
# plt.xlabel('Number of Tokens')
# plt.ylabel('Similarity')
# plt.title('Similarity vs. Number of Tokens')
# plt.grid(True)
# plt.show()