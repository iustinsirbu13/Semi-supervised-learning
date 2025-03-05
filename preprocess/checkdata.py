import ijson
import os
import json

dst_path = '../../data/aegis2.0_RH_eda'
split = 'train_lb'

file_path = os.path.join(dst_path, f'{split}.json')

chosen_key = 0
chosen_entry = ""
maxlen = 0

with open(file_path, 'r') as datafile:
    parser = ijson.kvitems(datafile, "")
    for i in range(7):
        first_key, first_entry = next(parser)

    with open("check.json", 'w') as outfile:
        json.dump(first_entry, outfile)

print(f"First entry (key: {first_key}) successfully written to 'check.json'.")
