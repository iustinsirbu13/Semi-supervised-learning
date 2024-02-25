import sys
sys.path.append("..")

import torch
from semilearn import get_config
from disaster_tweet.wrappers.fixmatch_mmbt_bert_wrapper import FixMatchMMBTBertWrapper

def parse_config():
    config = {
        # optimization configs
        'epoch': 1,
        'num_train_iter': 10, 
        'num_eval_iter': 5,   
        'num_log_iter': 5,    
        'optim': 'AdamW',
        'lr': 5e-4,
        'layer_decay': 0.5,
        'batch_size': 16,
        'eval_batch_size': 16,

        # dataset configs
        'task': 'humanitarian',
        'num_labels': 40,
        'crop_ratio': 0.875,
        'train_filename': 'train_eda.jsonl',
        'dev_filename': 'dev.jsonl',
        'test_filename': 'test.jsonl',
        'unlabeled_filename': 'unlabeled_eda.jsonl',
        'data_dir': 'C:\\Users\\Robert_Popovici\\Desktop\\licenta\\datasets\\humanitarian_orig',
        'img_dir': 'C:\\Users\\Robert_Popovici\\Desktop\\licenta\\datasets',
        'weak_text_tag': 'eda_02',
        'strong_text_tag': 'back_translate',

        # algorithm specific configs
        'hard_label': True,
        'uratio': 2,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': -1,
        'world_size': 1,
        "num_workers": 2,
        'distributed': False,
    }
    
    config = get_config(config)
    

    if config.gpu == -1:
        config.device = 'cpu'
    else:
        config.device = torch.device('cuda', config.gpu)
    
    return config


def main():
    config = parse_config()
    fix_match_bert_mmbt_wrapper = FixMatchMMBTBertWrapper(config)
    fix_match_bert_mmbt_wrapper.train_evaluate()

if __name__ == '__main__':
    main()