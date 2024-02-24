import sys
sys.path.append("..")

from semilearn import get_config
from wrappers.fixmatch_img_wrapper import FixMatchImgWrapper

def parse_config():
    config = {
        'net': 'vit_tiny_patch2_32',
        'use_pretrain': True,
        'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth',

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
        'train_filename': 'train.jsonl',
        'dev_filename': 'dev.jsonl',
        'test_filename': 'test.jsonl',
        'unlabeled_filename': 'unlabeled_eda.jsonl',
        'data_dir': 'C:\\Users\\Robert_Popovici\\Desktop\\licenta\\datasets\\humanitarian_orig',
        'img_dir': 'C:\\Users\\Robert_Popovici\\Desktop\\licenta\\datasets',

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
    
    return get_config(config)


def main():
    config = parse_config()
    fix_match_img_wrapper = FixMatchImgWrapper(config)
    fix_match_img_wrapper.train_evaluate()

if __name__ == '__main__':
    main()