# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset


class DisasterDatasetImage(Dataset):
    
    def __init__(self,
                 data,
                 targets,
                 num_classes,
                 labels,
                 weak_transform,
                 img_dir,
                 img_size,
                 strong_transform=None,
                 *args,
                 **kwargs):
        
        super(DisasterDatasetImage, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = targets is None

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        self.img_dir = img_dir
        assert self.img_dir is not None

        self.labels = labels
        assert len(self.labels) == self.num_classes

        self.img_size = img_size

        assert self.weak_transform is not None        
        if not self.is_ulb:
            assert self.strong_transform is None


    def get_weak_image(self, image):
        return self.weak_transform(image)


    def get_strong_image(self, image):
        if self.strong_transform is None:
            return None
        
        return self.strong_transform(image)

        
    def get_random_image(self):
        return Image.fromarray(128 * np.ones((self.img_size, self.img_size, 3), dtype=np.uint8))


    def get_image(self, img_name):
        try:
            if img_name:
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path).convert('RGB')
            else:
                image = self.get_random_image()
        except:
            print(f'Error processing image {img_name}')
            image = self.get_random_image()

        return image
    

    def sample(self, idx):
        image = self.get_image(self.data[idx]['image'])
        
        if self.is_ulb:
            target = None
        else:
            label = self.data[idx]['label']
            target = self.labels.index(label)
            
        return image, target


    def __getitem__(self, idx):
        image, target = self.sample(idx)
        weak_image = self.get_weak_image(image)

        if not self.is_ulb:
            return {'x_lb': weak_image, 'y_lb': target}
        else:
            strong_image = self.get_strong_image(image)
            return {'x_ulb_w': weak_image, 'x_ulb_s': strong_image, 'idx_ulb': idx}


    def __len__(self):
        return len(self.data)