import json
import random

with open('../../data/wildguardmix_PH_eda/train.json', 'r') as infile:
    data = json.load(infile)

entries_by_label = {'0': [], '1': []}
cnt = 0
for key, entry in data.items():
    cnt += 1
    label = entry.get('label')
    if label in entries_by_label:
        entries_by_label[label].append(key)

print(f"Size of dataset: {cnt}\n")
num_labeled = 10000
num_labeled_per_class = num_labeled // 2

if len(entries_by_label['0']) < num_labeled_per_class or len(entries_by_label['1']) < num_labeled_per_class:
    raise ValueError("Not enough examples for one or both labels.")

random.seed(1234567)
sampled_keys_0 = random.sample(entries_by_label['0'], num_labeled_per_class)
sampled_keys_1 = random.sample(entries_by_label['1'], num_labeled_per_class)
selected_keys = set(sampled_keys_0 + sampled_keys_1)

train_lb = {key: data[key] for key in selected_keys}

train_ulb = {key: data[key] for key in data if key not in selected_keys}

print(f"Size of lb dataset: {len(train_lb)}\n")
print(f"Size of ulb dataset: {len(train_ulb)}\n")
with open('../../data/wildguardmix_PH_eda_test/train_lb.json', 'w') as lb_file:
    json.dump(train_lb, lb_file, indent=4)

with open('../../data/wildguardmix_PH_eda_test/train_ulb.json', 'w') as ulb_file:
    json.dump(train_ulb, ulb_file, indent=4)

print("Splitting complete. Created train_lb.json and train_ulb.json.")
