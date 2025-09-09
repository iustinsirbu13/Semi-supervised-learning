# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import numpy as np
import random

from semilearn.datasets.utils import split_ssl_data
from .datasetbase import BasicDataset, LLMSafetyDataset

import logging
logger = logging.getLogger(__name__)

def get_json_dset(args, alg='fixmatch', dataset='acmIb', num_labels=40, num_classes=20, data_dir='./data', index=None, include_lb_to_ulb=False, onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeled data)
        """
        json_dir = os.path.join(data_dir, dataset)
        
        # Supervised top line using all data as labeled data.
        with open(os.path.join(json_dir,'train.json'),'r') as json_data:
            train_data = json.load(json_data)
            train_sen_list = []
            train_label_list = []
            for idx in train_data:
                train_sen_list.append((train_data[idx]['ori'],train_data[idx]['aug_0'],train_data[idx]['aug_1']))
                train_label_list.append(int(train_data[idx]['label']))
        with open(os.path.join(json_dir,'dev.json'),'r') as json_data:
            dev_data = json.load(json_data)
            dev_sen_list = []
            dev_label_list = []
            for idx in dev_data:
                dev_sen_list.append((dev_data[idx]['ori'],'None','None'))
                dev_label_list.append(int(dev_data[idx]['label']))
        with open(os.path.join(json_dir,'test.json'),'r') as json_data:
            test_data = json.load(json_data)
            test_sen_list = []
            test_label_list = []
            for idx in test_data:
                test_sen_list.append((test_data[idx]['ori'],'None','None'))
                test_label_list.append(int(test_data[idx]['label']))
        dev_dset = BasicDataset(alg, dev_sen_list, dev_label_list, num_classes, False, onehot)
        test_dset = BasicDataset(alg, test_sen_list, test_label_list, num_classes, False, onehot)
        if alg == 'fullysupervised':
            lb_dset = BasicDataset(alg, train_sen_list, train_label_list, num_classes, False,onehot)
            return lb_dset, None, dev_dset, test_dset
        include_lb_to_ulb = False

        lb_sen_list, lb_label_list, ulb_sen_list, ulb_label_list = split_ssl_data(args, train_sen_list, train_label_list, num_classes, 
                                                                    lb_num_labels=num_labels,
                                                                    ulb_num_labels=args.ulb_num_labels,
                                                                    lb_imbalance_ratio=args.lb_imb_ratio,
                                                                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                    include_lb_to_ulb=include_lb_to_ulb)
        
        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(num_classes)]
        for c in train_label_list:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(dataset) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
            
        lb_dset = BasicDataset(alg, lb_sen_list, lb_label_list, num_classes, False, onehot)
        ulb_dset = BasicDataset(alg, ulb_sen_list, ulb_label_list, num_classes, True, onehot)
        return lb_dset, ulb_dset, dev_dset, test_dset

def get_json_dset_aug_list(args, alg='fixmatch', dataset='acmIb', num_labels=40, num_classes=20, data_dir='./data', index=None, include_lb_to_ulb=False, onehot=False, text_weak_aug=None, text_strong_aug=None, load_labeled=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeled data)
        """
        json_dir = os.path.join(data_dir, dataset)
        if load_labeled:
            with open(os.path.join(json_dir,'train_lb.json'),'r') as json_data:
                train_lb_data = json.load(json_data)
                lb_sen_list = []
                lb_label_list = []
                for idx in train_lb_data:
                    lb_sen_list.append((train_lb_data[idx]['ori'],train_lb_data[idx][text_weak_aug],train_lb_data[idx][text_strong_aug]))
                    lb_label_list.append(int(train_lb_data[idx]['label']))
            
            with open(os.path.join(json_dir,'train_ulb.json'),'r') as json_data:
                train_ulb_data = json.load(json_data)
                ulb_sen_list = []
                ulb_label_list = []
                for idx in train_ulb_data:
                    ulb_sen_list.append((train_ulb_data[idx]['ori'],train_ulb_data[idx][text_weak_aug],train_ulb_data[idx][text_strong_aug]))
                    ulb_label_list.append(int(train_ulb_data[idx]['label']))

            if num_labels > 0:
                logger.info(f'Original num labeled examples: {len(lb_sen_list)}')
                lb_sen_list, lb_label_list, _, _ = split_ssl_data(
                    args, lb_sen_list, lb_label_list, num_classes, 
                    lb_num_labels=num_labels,
                    ulb_num_labels=args.ulb_num_labels,
                    lb_imbalance_ratio=args.lb_imb_ratio,
                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                    include_lb_to_ulb=include_lb_to_ulb
                )
                lb_label_list = lb_label_list.tolist()
                logger.info(f'Sampled num labeled examples: {len(lb_sen_list)}')

            train_label_list = lb_label_list + ulb_label_list
            with open(os.path.join(json_dir,'dev.json'),'r') as json_data:
                dev_data = json.load(json_data)
                dev_sen_list = []
                dev_label_list = []
                for idx in dev_data:
                    dev_sen_list.append((dev_data[idx]['ori'],'None','None'))
                    dev_label_list.append(int(dev_data[idx]['label']))
            with open(os.path.join(json_dir,'test.json'),'r') as json_data:
                test_data = json.load(json_data)
                test_sen_list = []
                test_label_list = []
                for idx in test_data:
                    test_sen_list.append((test_data[idx]['ori'],'None','None'))
                    test_label_list.append(int(test_data[idx]['label']))
            dev_dset = LLMSafetyDataset(alg, dev_sen_list, dev_label_list, num_classes, False, onehot)
            test_dset = LLMSafetyDataset(alg, test_sen_list, test_label_list, num_classes, False, onehot)
        else:
            # Supervised top line using all data as labeled data.
            with open(os.path.join(json_dir,'train.json'),'r') as json_data:
                train_data = json.load(json_data)
                train_sen_list = []
                train_label_list = []
                for idx in train_data:
                    train_sen_list.append((train_data[idx]['ori'],train_data[idx][text_weak_aug],train_data[idx][text_strong_aug]))
                    train_label_list.append(int(train_data[idx]['label']))
            with open(os.path.join(json_dir,'dev.json'),'r') as json_data:
                dev_data = json.load(json_data)
                dev_sen_list = []
                dev_label_list = []
                for idx in dev_data:
                    dev_sen_list.append((dev_data[idx]['ori'],'None','None'))
                    dev_label_list.append(int(dev_data[idx]['label']))
            with open(os.path.join(json_dir,'test.json'),'r') as json_data:
                test_data = json.load(json_data)
                test_sen_list = []
                test_label_list = []
                for idx in test_data:
                    test_sen_list.append((test_data[idx]['ori'],'None','None'))
                    test_label_list.append(int(test_data[idx]['label']))
            dev_dset = LLMSafetyDataset(alg, dev_sen_list, dev_label_list, num_classes, False, onehot)
            test_dset = LLMSafetyDataset(alg, test_sen_list, test_label_list, num_classes, False, onehot)
            if alg == 'fullysupervised':
                lb_dset = LLMSafetyDataset(alg, train_sen_list, train_label_list, num_classes, False,onehot)
                return lb_dset, None, dev_dset, test_dset
            include_lb_to_ulb = False
            
            lb_sen_list, lb_label_list, ulb_sen_list, ulb_label_list = split_ssl_data(args, train_sen_list, train_label_list, num_classes, 
                                                                    lb_num_labels=num_labels,
                                                                    ulb_num_labels=args.ulb_num_labels,
                                                                    lb_imbalance_ratio=args.lb_imb_ratio,
                                                                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                    include_lb_to_ulb=include_lb_to_ulb)
        # raise Exception(f'''
        #     dev set size: {len(dev_label_list)} | unique values: {sorted(set(dev_label_list))}
        #     test set size: {len(test_label_list)} | unique values: {sorted(set(test_label_list))}
        #     lb set size: {len(lb_label_list)} | unique values: {sorted(set(lb_label_list))}
        #     ulb set size: {len(ulb_label_list)} | unique values: {sorted(set(ulb_label_list))}
        # ''')

        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(num_classes)]
        for c in lb_label_list:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(dataset) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        lb_dset = LLMSafetyDataset(alg, lb_sen_list, lb_label_list, num_classes, False, onehot)
        ulb_dset = LLMSafetyDataset(alg, ulb_sen_list, ulb_label_list, num_classes, True, onehot)
        logger.info(f'Dataset sizes: labeled - {len(lb_dset)}, unlabeled - {len(ulb_dset)}, dev - {len(dev_dset)}, test - {len(test_dset)}')
        return lb_dset, ulb_dset, dev_dset, test_dset