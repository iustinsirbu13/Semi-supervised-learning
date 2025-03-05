import json

input_file1 = '../../data/wildguardmix_PH_eda_test/train_lb.json'
input_file2 = '../../data/aegis1.0_PH_eda_2/train_lb.json'
output_file = '../../data/aegis1.0_PH_eda_2/train_lb1.json'

with open(input_file1, 'r', encoding='utf-8') as f:
    data_wg = json.load(f)

with open(input_file2, 'r', encoding='utf-8') as f:
    data_aegis = json.load(f)

max_key = max(int(key) for key in data_aegis)
cnt = 0
for key in data_wg:
    max_key = max_key + 1
    entry_wg = data_wg[key]
    data_aegis[str(max_key)] = entry_wg

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_aegis, f, ensure_ascii=False, indent=4)

print(f"Modified JSON has been saved to {output_file}")
