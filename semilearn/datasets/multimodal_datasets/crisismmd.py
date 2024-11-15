import random
import torch
import os
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset


def get_crisismmd(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    raise NotImplementedError('TODO')


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
                image = self.__get_random_image__()
        except:
            image = self.__get_random_image__()

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
            return {'x_ulb_w': weak_image, 'x_ulb_s': strong_image}


    def __len__(self):
        return len(self.data)


class DisasterDatasetMMBT(DisasterDatasetImage):

    def __init__(self,
                 data,
                 targets,
                 num_classes,
                 labels,
                 weak_image_transform,
                 img_dir,
                 img_size,
                 weak_text_tag,
                 text_dir,
                 max_seq_length,
                 tokenizer,
                 vocab,
                 model,
                 num_image_embeds,
                 strong_image_transform=None,
                 strong_text_tag=None,
                 *args, 
                 **kwargs):
        
        super().__init__(data=data, 
                        targets=targets,
                        num_classes=num_classes,
                        weak_transform=weak_image_transform,
                        strong_transform=strong_image_transform,
                        labels=labels,
                        img_dir=img_dir,
                        img_size=img_size)
        
        self.weak_text_tag = weak_text_tag
        assert self.weak_text_tag is not None

        self.strong_text_tag = strong_text_tag
        if self.is_ulb:
            assert self.strong_text_tag is not None
        
        self.text_dir = text_dir
        assert self.text_dir is not None
        
        self.max_seq_length = max_seq_length
        assert self.max_seq_length is not None

        self.model = model
        assert self.model is not None

        if self.model == 'mmbt_bert':
            assert num_image_embeds is not None
            self.max_seq_length -= num_image_embeds

        self.tokenizer = tokenizer
        assert self.tokenizer is not None

        self.vocab = vocab
        assert self.vocab is not None

        self.text_start_token = ["[CLS]"] if self.model != "mmbt_bert" else ["[SEP]"]

        self.TEXT_SIZE = 192
    

    def get_sentence_segment(self, text):
        tokenized_text = self.tokenizer(text)
        sentence = self.text_start_token + tokenized_text[:(self.max_seq_length - 1)]
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        
        # ToDo: clarify this???
        if self.model == 'mmbt_bert':
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment
        
    def sample(self, idx):
        image, target = super().sample(idx)

        weak_text = self.data[idx][self.weak_text_tag]
        if self.weak_text_tag != 'text':
            weak_text = random.choice(weak_text)

        strong_text = None
        if self.strong_text_tag is not None:
            strong_text = random.choice(self.data[idx][self.strong_text_tag])

        return image, target, weak_text, strong_text

    
    def get_sentence_segment_mask_tensor(self, sentence, segment):
        length = min(self.TEXT_SIZE, len(sentence))

        sentence_tensor = torch.zeros(self.TEXT_SIZE).long()
        segment_tensor = torch.zeros(self.TEXT_SIZE).long()
        mask_tensor = torch.zeros(self.TEXT_SIZE).long()

        sentence_tensor[:length] = sentence[:length]
        segment_tensor[:length] = segment[:length]
        mask_tensor[:length] = 1

        return sentence_tensor, segment_tensor, mask_tensor


    def __getitem__(self, idx):
        image, target, weak_text, strong_text = self.sample(idx)

        weak_image = self.get_weak_image(image)
        strong_image = self.get_strong_image(image)

        weak_sentence, weak_segment = self.get_sentence_segment(weak_text)
        if strong_text is not None:
            strong_sentence, strong_segment = self.get_sentence_segment(strong_text)
        
        items = {}
        if not self.is_ulb:
            items['lb_weak_image'] = weak_image

            weak_text_tensors = self.get_sentence_segment_mask_tensor(weak_sentence, weak_segment)
            items['lb_weak_sentence'] = weak_text_tensors[0]
            items['lb_weak_segment'] = weak_text_tensors[1]
            items['lb_weak_mask'] = weak_text_tensors[2]

            items['lb_target'] = torch.LongTensor([target])
        else:
            items['ulb_weak_image'] = weak_image
            items['ulb_strong_image'] = strong_image

            weak_text_tensors = self.get_sentence_segment_mask_tensor(weak_sentence, weak_segment)
            items['ulb_weak_sentence'] = weak_text_tensors[0]
            items['ulb_weak_segment'] = weak_text_tensors[1]
            items['ulb_weak_mask'] = weak_text_tensors[2]

            strong_text_tensors = self.get_sentence_segment_mask_tensor(strong_sentence, strong_segment)
            items['ulb_strong_sentence'] = strong_text_tensors[0]
            items['ulb_strong_segment'] = strong_text_tensors[1]
            items['ulb_strong_mask'] = strong_text_tensors[2]

        return items
