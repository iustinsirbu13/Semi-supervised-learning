import json
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch.nn.functional as F
from tqdm import tqdm
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_similarity(entry):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', truncation_side='left')
    model = AutoModel.from_pretrained('microsoft/deberta-v3-base').to(device)
    model.eval()

    ori_text = entry["ori"]
    translated_text = entry["translated"][-1]

    # print("Original:", ori_text)
    # print("Translated:", translated_text)

    # Tokenize
    # input_ids = self.tokenizer.encode_plus(f['text'][0], f['text'][1], max_length=self.max_length, truncation=True, padding=False)['input_ids']
    encoding_ori = tokenizer.encode_plus(ori_text[0], ori_text[1], return_tensors='pt', max_length=1024, truncation=True, padding=True)
    encoding_translated = tokenizer.encode_plus(translated_text[0], translated_text[1], return_tensors='pt', max_length=1024, truncation=True, padding=True)

    input_ids_ori = encoding_ori['input_ids'].to(device)
    attention_mask_ori = encoding_ori['attention_mask'].to(device)
    input_ids_translated = encoding_translated['input_ids'].to(device)
    attention_mask_translated = encoding_translated['attention_mask'].to(device)

    with torch.no_grad():
        outputs_ori = model(input_ids_ori, attention_mask=attention_mask_ori)
        outputs_trans = model(input_ids_translated, attention_mask=attention_mask_translated)

    ori_emb = (outputs_ori.last_hidden_state * attention_mask_ori.unsqueeze(-1)).sum(1) / attention_mask_ori.sum(1, keepdim=True)
    trans_emb = (outputs_trans.last_hidden_state * attention_mask_translated.unsqueeze(-1)).sum(1) / attention_mask_translated.sum(1, keepdim=True)

    cos_sim = F.cosine_similarity(ori_emb, trans_emb)

    return len(input_ids_ori.squeeze(0)), cos_sim.item()


if __name__ == '__main__':
    with open('../../data/wildguardmix_RH_paraphrase_mistral/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    sampled_keys = random.sample(list(data.keys()), 1000)

    similarities = []

    for key in tqdm(sampled_keys):
        entry = data[key]
        _, sim = get_similarity(entry)
        similarities.append(sim)

    average_similarity = sum(similarities) / len(similarities)
    print("Average cosine similarity:", average_similarity)