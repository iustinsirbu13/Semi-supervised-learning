# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import os
import numpy as np 
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import torch

from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import get_onehot


class DisasterDataset(Dataset):
    
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 weak_transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 img_dir=None,
                 labels=[],
                 img_size=256,
                 *args, 
                 **kwargs):
        
        super(DisasterDataset, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        self.img_dir = img_dir
        assert self.img_dir is not None

        self.labels = labels
        assert len(self.labels) == self.num_classes

        self.img_size = img_size

        assert self.weak_transform is not None        
        if self.is_ulb:
            assert self.targets is None
        else:
            assert self.strong_transform is None

        
    def __get_random_image__(self):
        return Image.fromarray(128 * np.ones((self.img_size, self.img_size, 3), dtype=np.uint8))


    def __get_image__(self, img_name):
        try:
            if img_name:
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path).convert('RGB')
            else:
                image = self.__get_random_image__()
        except:
            image = self.__get_random_image__()

        return image
    

    def __sample__(self, idx):
        image = self.__get_image__(self.data[idx]['image'])
        
        if self.is_ulb:
            target = None
        else:
            label = self.data[idx]['label']
            target = self.labels.index(label)
            
        return image, target


    def __getitem__(self, idx):
        image, target = self.__sample__(idx)
        weak_aug_image = self.weak_transform(image)

        if not self.is_ulb:
            return {'x_lb': weak_aug_image, 'y_lb': target}
        else:
            strong_aug_image = self.strong_transform(image)
            return {'x_ulb_w': weak_aug_image, 'x_ulb_s': strong_aug_image}


    def __len__(self):
        return len(self.data)