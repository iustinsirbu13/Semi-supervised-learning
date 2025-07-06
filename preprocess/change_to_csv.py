import json
import csv

def json_to_csv_simple(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(csv_path, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['label', 'content'])
        for key, val in data.items():
            writer.writerow([
                val.get('label', '').replace('\n', ' '),
                val.get('ori', '').replace('\n', ' ')
            ])

def json_to_csv_train(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(csv_path, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['label', 'content', 'synonym_aug', 'back_translation'])
        for key, val in data.items():
            label = val.get('label', '').replace('\n', ' ')
            ori = val.get('ori', '').replace('\n', ' ')
            back = ''
            if isinstance(val.get('translated'), list) and val['translated']:
                back = val['translated'][0].replace('\n', ' ')
            writer.writerow([label, ori, ori, back])

if __name__ == '__main__':
    json_to_csv_simple('../../data/OAIMod_PH/dev.json',  '../../data/OAIMod_PH/dev.csv')
    json_to_csv_simple('../../data/OAIMod_PH/test.json', '../../data/OAIMod_PH/test.csv')
    json_to_csv_train('../../data/OAIMod_PH/train.json', '../../data/OAIMod_PH/train.csv')
    print("All CSVs written: dev.csv, test.csv, train.csv")
