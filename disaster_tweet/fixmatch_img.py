import sys
sys.path.append('..')

import numpy as np
from torchvision import transforms

from semilearn import get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from semilearn.datasets.cv_datasets.disasterdataset import DisasterDataset

import os
import random
import jsonlines

from utils.randaugment import RandAugmentMC

def parse_config():
    config = {
        'algorithm': 'fixmatch',
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
        'dataset': 'CrisisMMD',
        'num_labels': 40,
        'num_classes': 5,
        'img_size': 32,
        'crop_ratio': 0.875,
        'train_filename': 'train.jsonl',
        'dev_filename': 'dev.jsonl',
        'test_filename': 'test.jsonl',
        'unlabeled_filename': 'unlabeled_eda.jsonl',
        'labels': ['not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort', 'infrastructure_and_utility_damage', 'affected_individuals'],
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

    config['num_entries_per_class'] = config['num_labels'] // config['num_classes']
    
    return get_config(config)


class FixMatchImgWrapper:
    def __init__(self, config):
        self.config = config

        # read train, dev and test datasets
        self.train_x, self.train_target = self.read_train_dataset(config)
        self.dev_x, self.dev_target = self.read_dev_dataset(config)
        self.test_x, self.test_target = self.read_test_dataset(config)

        # read unlabeled dataset
        self.unlabeled_x = self.read_unlabeled_dataset(config)

        # prepare USB algorithm
        self.algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

        # prepare USB datasets and loaders
        self.prepare_datasets(config)
        self.prepare_dataloaders(config)
        
    
    def get_weak_transform(self, img_size, crop_ratio):
        return transforms.Compose([
                                transforms.Resize([img_size, img_size]),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=img_size, padding=int(img_size * (1.0 - crop_ratio)), padding_mode='reflect'),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
    
    def get_strong_transform(self, img_size, crop_ratio):
        return transforms.Compose([
                                transforms.Resize([img_size, img_size]),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=img_size, padding=int(img_size * (1.0 - crop_ratio)), padding_mode='reflect'),
                                RandAugmentMC(n=2, m=10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])

    def get_eval_transform(self, img_size):
        return transforms.Compose([
                                    transforms.Resize([img_size, img_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

    def read_labeled_dataset(self, filename, config, num_entries_per_class=0):
        filepath = os.path.join(config.data_dir, filename)
        dataset = [row for row in jsonlines.open(filepath)]

        if num_entries_per_class == 0:
            x = dataset
        else:
            x = []
            for label in config.labels:
                dataset_by_label = [row for row in dataset if row['label'] == label]
                dataset_by_label = random.sample(dataset_by_label, num_entries_per_class)
                x += dataset_by_label

        random.shuffle(x)
        target = [config.labels.index(row['label']) for row in x]

        return x, target

    def read_train_dataset(self, config):
        return self.read_labeled_dataset(config.train_filename, config, num_entries_per_class=config.num_entries_per_class)

    def read_dev_dataset(self, config):
        return self.read_labeled_dataset(config.dev_filename, config)

    def read_test_dataset(self, config):
        return self.read_labeled_dataset(config.test_filename, config)

    def read_unlabeled_dataset(self, config):
        filepath = os.path.join(config.data_dir, config.unlabeled_filename)

        dataset = [row for row in jsonlines.open(filepath)]
        x = random.sample(dataset, len(self.train_x) * config.uratio)

        random.shuffle(x)

        return x
        

    def prepare_datasets(self, config):
        # weak and strong image transformation
        transform_weak = self.get_weak_transform(config.img_size, config.crop_ratio)
        transform_strong = self.get_strong_transform(config.img_size, config.crop_ratio)
        
        # transformation used for dev and test datasets
        eval_tranform = self.get_eval_transform(config.img_size)

        self.train_dataset = DisasterDataset(self.train_x,
                                            self.train_target,
                                            config.num_classes,
                                            transform_weak,
                                            is_ulb=False,
                                            img_dir=config.img_dir,
                                            labels=config.labels,
                                            img_size=config.img_size)
        
        self.dev_dataset = DisasterDataset(self.dev_x,
                                            self.dev_target,
                                            config.num_classes,
                                            eval_tranform,
                                            is_ulb=False,
                                            img_dir=config.img_dir,
                                            labels=config.labels,
                                            img_size=config.img_size)


        self.test_dataset = DisasterDataset(self.test_x,
                                            self.test_target,
                                            config.num_classes,
                                            eval_tranform,
                                            is_ulb=False,
                                            img_dir=config.img_dir,
                                            labels=config.labels,
                                            img_size=config.img_size)


        self.unlabeled_dataset = DisasterDataset(self.unlabeled_x,
                                                None,
                                                config.num_classes,
                                                transform_weak,
                                                is_ulb=True,
                                                strong_transform=transform_strong,
                                                img_dir=config.img_dir,
                                                labels=config.labels,
                                                img_size=config.img_size)


    def prepare_dataloaders(self, config):
        self.train_loader = get_data_loader(config, self.train_dataset, config.batch_size)
        self.dev_loader = get_data_loader(config, self.dev_dataset, config.eval_batch_size)
        self.test_loader = get_data_loader(config, self.test_dataset, config.eval_batch_size)
        self.unlabeled_loader = get_data_loader(config, self.unlabeled_dataset, config.batch_size * config.uratio)


    def train(self):
        trainer = Trainer(self.config, self.algorithm)
        trainer.fit(self.train_loader, self.unlabeled_loader, self.test_loader)
        return trainer

    def evaluate(self, trainer):
        trainer.evaluate(self.test_loader)

    def train_evaluate(self):
        trainer = self.train()
        self.evaluate(trainer)


def main():
    config = parse_config()
    fix_match_img_wrapper = FixMatchImgWrapper(config)
    fix_match_img_wrapper.train_evaluate()

if __name__ == '__main__':
    main()