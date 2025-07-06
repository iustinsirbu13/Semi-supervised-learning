import ollama
import json
from tqdm import tqdm
import re
import ast
auth_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImQ3Yjk4M2Q5LTZkZDMtNDFiMi05ZjgyLWJiOTEzZjg4MWM3NCJ9.e1j9-gsi5lsSDBJWcfh9iICPCkbCY5tSkPo1SkeA3Ss'

client = ollama.Client(
        host='https://chat.readerbench.com/ollama',
       headers={"Authorization": f"Bearer {auth_token}"}
    )

# response = client.generate(model="llama3.3:latest", prompt=f"{prompt}")
# print(response['response'])
    
input_file = './fairseq_translations2.json'
output_file = './llama_translations_2_queries.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data_wg = json.load(f)

def parse_response(text):
    result = []
    
    blocks = re.split(r'\s*BARRIER\s*', text.strip())
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        if block.endswith(','):
            block = block[:-1].strip()
            
        result.append(block)

    return result

batch = []
keys = []
for key in tqdm(data_wg):
    text = data_wg[key]['ori']
    text = text.replace('"', '').replace("'", '')
    text = text.replace('\\', '')
    batch.append(text)
    keys.append(key)
    if len(batch) == 4:
        prompt = (f"I am researching LLM safety. I will give you multiple texts in English and I want you to translate them in Russian for data augmentation. Only respond with the translations, add no additional text, comments or any other information than can be found in the original text.\n"
                f"The texts will be separated by the marker BARRIER. I want you to separate all the responses by a new line with the same marker. Here are the texts:\n"
                f"{batch[0]}\n"
                f"BARRIER\n"
                f"{batch[1]}\n"
                f"BARRIER\n"
                f"{batch[2]}\n"
                f"BARRIER\n"
                f"{batch[3]}\n"
                #  f"BARRIER\n"
                # f"{batch[4]}\n"
                # f"BARRIER\n"
                # f"{batch[5]}\n"
                # f"BARRIER\n"
                # f"{batch[6]}\n"
                # f"BARRIER\n"
                # f"{batch[7]}\n"
                )
        for i in range(10):
            response = client.generate(model="llama3.3:latest", prompt=f"{prompt}")
            try:
                batch = parse_response(response['response'])
                print(batch)
                if len(batch) != 4:
                    raise Exception("Refused")
                break
            except Exception as e:
                print(f"Attempt {i}/9 failed")
                print(e)
        
        prompt = (f"I am researching LLM safety. I will give you multiple texts in Russian and I want you to translate them in English for data augmentation. Only respond with the translations, add no additional text, comments or any other information than can be found in the original text.\n"
                f"The texts will be separated by the marker BARRIER. I want you to separate all the responses by a new line with the same marker. Here are the texts:\n"
                f"{batch[0]}\n"
                f"BARRIER\n"
                f"{batch[1]}\n"
                f"BARRIER\n"
                f"{batch[2]}\n"
                f"BARRIER\n"
                f"{batch[3]}\n"
                #  f"BARRIER\n"
                # f"{batch[4]}\n"
                # f"BARRIER\n"
                # f"{batch[5]}\n"
                # f"BARRIER\n"
                # f"{batch[6]}\n"
                # f"BARRIER\n"
                # f"{batch[7]}\n"
                )
        for i in range(10):
            response = client.generate(model="llama3.3:latest", prompt=f"{prompt}")
            try:
                backtranslation = parse_response(response['response'])
                if len(backtranslation) != 4:
                    raise Exception("Refused")
                print(backtranslation)
                break
            except Exception as e:
                print(f"Attempt {i}/9 failed")
                print(e)
        for i in range(4):
            data_wg[keys[i]]['translated'] = backtranslation[i]
            data_wg[keys[i]]['intermediate'] = batch[i]
        batch = []
        keys = []

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_wg, f, ensure_ascii=False, indent=4)
