import pandas as pd
import json
import os
from tqdm import tqdm

categories = {
    'benign' : 0,                                                                                
    'others' : 1,                                                                                 
    'social_stereotypes_and_unfair_discrimination' : 2,                                           
    'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns' : 3,     
    'sensitive_information_organization_government' : 4,                                        
    'toxic_language_hate_speech' : 5,                                                             
    'violence_and_physical_harm' : 6,                                                             
    'private_information_individual' : 7,                                                         
    'defamation_encouraging_unethical_or_unsafe_actions' : 8,                                     
    'fraud_assisting_illegal_activities' : 9,                                                     
    'sexual_content' : 10,                                                                         
    'mental_health_over-reliance_crisis' : 11,                                                     
    'copyright_violations' : 12,                                                                   
    'cyberattack' : 13,                                                                            
    'causing_material_harm_by_disseminating_misinformation' : 14                                  
}

dst_path = './data/wildguardmix_orig_test/'

csv_paths = [os.path.join(dst_path, f'wildguard_train_part_{i}.csv') for i in range(1, 5)]

df = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)

input_file = '../../data/wildguardmix_PH_paraphrase/train.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

mapping = df.set_index('prompt')['subcategory'].to_dict()

for key in tqdm(data):
    data[key]['cat'] = str(categories[mapping.get(data[key]['orig'][0], 'others')])

output_file = "../../data/wildguardmix_PH_paraphrase_catsplit/train.json"

with open(output_file, 'w') as outfile:
    json.dump(data, outfile)
