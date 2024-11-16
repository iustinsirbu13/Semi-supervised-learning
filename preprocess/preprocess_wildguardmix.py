from datasets import load_dataset
import json
import os


def format_as_json():
    dst_path = './data/wildguardmix_PH'
    text_column = 'prompt'
    label_column = 'prompt_harm_label'
    seed = 1234567
    labels = {
        'unharmful': 0,
        'harmful': 1,
    }
    os.makedirs(dst_path, exist_ok=True)

    ds_test = load_dataset("allenai/wildguardmix", "wildguardtest", split='test')
    ds_train = load_dataset("allenai/wildguardmix", "wildguardtrain", split='train')
    train_size = ds_train.num_rows

    # ds_train = ds_train.train_test_split(test_size=train_size//10, seed=seed, stratify_by_column=label_column)
    ds_train = ds_train.train_test_split(test_size=train_size//10, seed=seed)
    ds_valid = ds_train['test']
    # ds_train = ds_train['train'].train_test_split(test_size=train_size//10, seed=seed, stratify_by_column=label_column)
    ds_train_full = ds_train['train']
    ds_train = ds_train_full.train_test_split(test_size=train_size//10, seed=seed)
    ds_labeled = ds_train['test']
    ds_unlabeled = ds_train['train']
    assert ds_labeled.num_rows + ds_unlabeled.num_rows + ds_valid.num_rows == train_size, f'{ds_labeled.num_rows, ds_unlabeled.num_rows, ds_valid.num_rows} does not add up to {train_size}'

    datasets = {
       'labeled': ds_labeled,
       'unlabeled': ds_unlabeled,
       'dev': ds_valid,
       'test': ds_test,
       'train': ds_labeled # TODO: ds_train_full,
    }

    for split_name, split_ds in datasets.items():
        data = {}
        with open(os.path.join(dst_path, f'{split_name}.json'), 'w') as outfile:
            for idx, elem in enumerate(split_ds.filter(lambda example: example[label_column] is not None)):
                data[str(idx)] = {}
                data[str(idx)]['ori'] = elem[text_column]
                try:
                    data[str(idx)]['label'] = str(labels[elem[label_column]])
                except Exception as e:
                    print(idx, label_column, elem[label_column])
                    print(elem)
                    print(labels[elem[label_column]])
                    print(data[str(idx)])
                    raise e
                if split_name in ['labeled', 'unlabeled', 'train']:
                    data[str(idx)]['aug_0'] = data[str(idx)]['ori']
                    data[str(idx)]['aug_1'] = data[str(idx)]['ori']
            json.dump(data, outfile)


if __name__ == '__main__':
    format_as_json()