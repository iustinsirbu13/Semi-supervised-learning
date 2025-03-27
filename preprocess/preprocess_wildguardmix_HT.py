
SUBCATEGORIES = [
 'sensitive_information_organization_government',
 'private_information_individual',
 'copyright_violations',

 'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns',
 'causing_material_harm_by_disseminating_misinformation',

 'social_stereotypes_and_unfair_discrimination',
 'violence_and_physical_harm',
 'toxic_language_hate_speech',
 'sexual_content',

 'cyberattack',
 'fraud_assisting_illegal_activities',
 'defamation_encouraging_unethical_or_unsafe_actions',
 'mental_health_over-reliance_crisis',

 'others',
 'benign',

]

CATEGORIES = [
    'PRIVACY',
    'MISINFORMATION',
    'HARMFUL_LANGUAGE',
    'MALICIOUS_USES',
    'OTHER_HARMS',
    'BENIGN',
]

TOP_CATEGORIES = [
    'unharmful',
    'harmful'
]

SUBCAT2CAT = {
 'sensitive_information_organization_government': 'PRIVACY',
 'private_information_individual': 'PRIVACY',
 'copyright_violations': 'PRIVACY',

 'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns': 'MISINFORMATION',
 'causing_material_harm_by_disseminating_misinformation': 'MISINFORMATION',

 'social_stereotypes_and_unfair_discrimination': 'HARMFUL_LANGUAGE',
 'violence_and_physical_harm': 'HARMFUL_LANGUAGE',
 'toxic_language_hate_speech': 'HARMFUL_LANGUAGE',
 'sexual_content': 'HARMFUL_LANGUAGE',

 'cyberattack': 'MALICIOUS_USES',
 'fraud_assisting_illegal_activities': 'MALICIOUS_USES',
 'defamation_encouraging_unethical_or_unsafe_actions': 'MALICIOUS_USES',
 'mental_health_over-reliance_crisis': 'MALICIOUS_USES',

 'others': 'OTHER_HARMS',
 'benign': 'BENIGN',

}

def split_wildguard_subcategories(output_path=None):
    from datasets import load_dataset
    import os
    from sklearn.model_selection import train_test_split

    ds_test = load_dataset("allenai/wildguardmix", "wildguardtest", split='test')
    ds_train = load_dataset("allenai/wildguardmix", "wildguardtrain", split='train')
    df_train = ds_train.to_pandas()
    df_test = ds_test.to_pandas()

    df_test.dropna(subset=['prompt_harm_label'], inplace=True)
    df_train = df_train[df_train.prompt != '']
    df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train.subcategory)

    for df_aux in [df_train, df_valid, df_test]:
        df_aux['TOP_CATEGORY_LABEL'] = df_aux['prompt_harm_label'].map(TOP_CATEGORIES.index)
        df_aux['CATEGORY'] = df_aux['subcategory'].map(SUBCAT2CAT)
        df_aux['CATEGORY_LABEL'] = df_aux['CATEGORY'].map(CATEGORIES.index)
        df_aux['SUBCATEGORY_LABEL'] = df_aux['subcategory'].map(SUBCATEGORIES.index)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        df_train.to_csv(os.path.join(output_path, 'train.csv'))
        df_valid.to_csv(os.path.join(output_path, 'dev.csv'))
        df_test.to_csv(os.path.join(output_path, 'test.csv'))
    else:
        return {
            'train': df_train,
            'dev': df_valid,
            'test': df_test,
        }


def read_wildguard_for_bt(input_path, data_type='train'):
    import os
    import pandas as pd
    df = pd.read_csv(os.path.join(input_path, f'{data_type}.csv'))
    ori_sen = df.prompt.tolist()
    label = [-1] * len(ori_sen)
    print(f'WildGuard {data_type} has {len(ori_sen)} samples.')
    return ori_sen, label

def split_wildguard_into_tasks(data_path, csv_dataset, json_dataset, final_name, label_col):
    import json
    import os
    import pandas as pd
    csv_path = os.path.join(data_path, csv_dataset)
    json_path = os.path.join(data_path, json_dataset)
    # output_path = os.path.join(data_path, final_name)
    # os.makedirs(output_path, exist_ok=True)

    for data_split in ['test', 'dev', 'train']:
        with open(os.path.join(json_path, f'{data_split}.json'), 'r') as infile:
            data = json.load(infile)
            df = pd.read_csv(os.path.join(csv_path, f'{data_split}.csv'))
            
            labels = df[label_col].tolist()
            for idx, label in enumerate(labels):
                data[str(idx)]['label'] = label

            output_path = os.path.join(data_path, final_name)
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f'{data_split}.json'), 'w') as outfile:
                json.dump(data, outfile, indent=4)

        



if __name__ == '__main__':
    # split_wildguard_subcategories('./data/wildguardmix_HT_init')

    # split_wildguard_into_tasks('./data', 'wildguardmix_HT_init', 'wildguardmix_HT', 'wildguardmix_HT_subcat', 'SUBCATEGORY_LABEL')
    # split_wildguard_into_tasks('./data', 'wildguardmix_HT_init', 'wildguardmix_HT', 'wildguardmix_HT_topcat', 'TOP_CATEGORY_LABEL')
    # split_wildguard_into_tasks('./data', 'wildguardmix_HT_init', 'wildguardmix_HT', 'wildguardmix_HT_cat', 'CATEGORY_LABEL')
    pass
