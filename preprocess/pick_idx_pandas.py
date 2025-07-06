import pandas as pd

df = pd.read_csv('../../data/wildguardmix_PH_paraphrase/train.csv')
df_filtered = df[df['id'].isin(index_list)]
# write as { id: { … } } JSON
df_filtered.set_index('id')\
           .to_json("index_llama.json", 
                    orient='index', 
                    force_ascii=False, 
                    indent=2)