import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import umap.umap_ as umap
from tqdm import tqdm

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", truncation_side="left")
model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device)
model.eval()

# Load JSON and extract (prompt, response) from list values
def load_samples(json_path):
    print(f"Loading: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(entry["ori"][0], entry["ori"][1]) for entry in data.values() if "ori" in entry and len(entry["ori"]) >= 2]

# Get [CLS] token embedding using tokenizer.encode_plus
def get_cls_embedding(prompt, response):
    # print(prompt, flush=True)
    # print(response, flush=True)
    encoded = tokenizer.encode_plus(
        prompt,
        response,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="max_length"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    cls_embed = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls_embed

# File paths
wildguard_path = "../../data/wildguardmix_RH_paraphrase_mistral/test.json"
oai_path = "../../data/aegis2.0_RH_eda/test.json"

# Load data
wildguard_samples = load_samples(wildguard_path)
oai_samples = load_samples(oai_path)

# Get embeddings
embeddings = []
labels = []

print("Generating embeddings...")
for prompt, response in tqdm(wildguard_samples):
    emb = get_cls_embedding(prompt, response)
    embeddings.append(emb)
    labels.append("Wildguard")

for prompt, response in tqdm(oai_samples):
    emb = get_cls_embedding(prompt, response)
    embeddings.append(emb)
    labels.append("Aegis 2.0")

# Reduce to 2D using UMAP
print("Reducing dimensions with UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Plot
print("Plotting...")
colors = {"Wildguard": "blue", "Aegis 2.0": "red"}
plt.figure(figsize=(10, 6))
for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    points = embeddings_2d[indices]
    plt.scatter(points[:, 0], points[:, 1], c=colors[label], label=label, alpha=0.6)
plt.title("UMAP Projection of CLS Embeddings (Prompt + Response)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
