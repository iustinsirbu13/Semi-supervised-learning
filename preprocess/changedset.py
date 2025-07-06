import json

input_file1 = '../../data/wildguardmix_RH_paraphrase_mistral/train.json'
input_file2 = '../../data/wildguardmix_RH_paraphrase_mistral_2/train.json'
output_file = '../../data/wildguardmix_RH_paraphrase_mistral_2aug/train_complete.json'

with open(input_file1, 'r', encoding='utf-8') as f:
    data_llm1 = json.load(f)

with open(input_file2, 'r', encoding='utf-8') as f:
    data_llm2 = json.load(f)

cnt = 0
for key in data_llm2:
    cnt += 1
    if 'translated' in data_llm1[key]:
        aug_llm1 = data_llm1[key]['translated']
    else:
        aug_llm1 = data_llm1[key]['orig']
    if 'translated' in data_llm2[key]:
        aug_llm2 = data_llm2[key]['translated']
    else:
        aug_llm2 = data_llm2[key]['orig']
    data_llm2[key]['translated'] = [aug_llm1[0], aug_llm2[0]]

# max_key = max(int(key) for key in data_aegis)
# cnt = 0
# for key in data_wg:
#     max_key = max_key + 1
#     entry_wg = data_wg[key]
#     data_aegis[str(max_key)] = entry_wg

print(f'Number of entries: {cnt}\n')

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_llm2, f, ensure_ascii=False, indent=4)

print(f"Modified JSON has been saved to {output_file}")
