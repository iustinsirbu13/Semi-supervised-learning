import yaml
from collections import OrderedDict

def represent_ordered_dict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

yaml.add_representer(dict, represent_ordered_dict)

def represent_bool(dumper, data):
    value = "True" if data else "False"
    return dumper.represent_scalar('tag:yaml.org,2002:bool', value)

yaml.add_representer(bool, represent_bool)

def create_yaml_config(save_name, base_path="./saved_models/usb_llm_safety", output_file="config.yaml"):
    config = {
        "algorithm": "fixmatch",
        "save_dir": base_path,
        "save_name": save_name,
        "resume": True,
        "load_path": f"{base_path}/{save_name}/latest_model.pth",
        "overwrite": True,
        "use_tensorboard": True,
        "use_wandb": False,
        "epoch": 1,
        "num_train_iter": 102400,
        "num_warmup_iter": 5120,
        "num_log_iter": 256,
        "num_eval_iter": 2048,
        "num_labels": 2000,
        "batch_size": 16,
        "eval_batch_size": 16,
        "ema_m": 0.0,
        "hard_label": True,
        "T": 0.5,
        "p_cutoff": 0.95,
        "ulb_loss_ratio": 1.0,
        "uratio": 1,
        "use_cat": False,
        "optim": "AdamW",
        "lr": 5e-05,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "layer_decay": 0.65,
        "amp": False,
        "clip": 0.0,
        "net": "deberta_v3_base",
        "net_from_name": False,
        "data_dir": "./data",
        "dataset": "aegis2.0_PH_eda_2",
        "train_sampler": "RandomSampler",
        "num_classes": 2,
        "num_workers": 4,
        "max_length": 256,
        "seed": 1,
        "world_size": 1,
        "rank": 0,
        "multiprocessing_distributed": False,
        "dist_url": "tcp://127.0.0.1:10008",
        "dist_backend": "nccl",
        "gpu": 0,
        "include_lb_to_ulb": False,
        "text_weak_aug": "eda_synonym",
        "text_strong_aug": "eda_full",
        "save_pseudolabels_stats": True
    }

    with open(output_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

models = [
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_0_deberta_paraphrase',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_1_deberta_paraphrase',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_2_deberta_paraphrase',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_0_deberta_paraphrase_mistral',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_1_deberta_paraphrase_mistral',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_2_deberta_paraphrase_mistral',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_0_deberta_paraphrase_mistral_llama',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_1_deberta_paraphrase_mistral_llama',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_2_deberta_paraphrase_mistral_llama',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_0_deberta_translate',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_1_deberta_translate',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_2_deberta_translate',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_0_deberta_translate_paraphrase',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_1_deberta_translate_paraphrase',
    'fixmatch_wildguardmix_PH_200_translate_hate_seed_2_deberta_translate_paraphrase',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_paraphrase',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_1_deberta_paraphrase',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_2_deberta_paraphrase',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_paraphrase_2aug',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_1_deberta_paraphrase_2aug',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_2_deberta_paraphrase_2aug',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_paraphrase_mistral',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_1_deberta_paraphrase_mistral',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_2_deberta_paraphrase_mistral',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_paraphrase_mistral_1_llama_2',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_1_deberta_paraphrase_mistral_1_llama_2',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_2_deberta_paraphrase_mistral_1_llama_2',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_paraphrase_mistral_llama',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_1_deberta_paraphrase_mistral_llama',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_2_deberta_paraphrase_mistral_llama',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_translate',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_1_deberta_translate',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_2_deberta_translate',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_translate_paraphrase',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_1_deberta_translate_paraphrase',
    'fixmatch_wildguardmix_PH_2000_translate_hate_seed_2_deberta_translate_paraphrase'
]

for model in models:
    create_yaml_config(
        save_name=model,
        output_file=f"../generated_configs5/{model}.yaml"
    )
