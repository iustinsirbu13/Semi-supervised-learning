# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers import BertModel
import os

class ClassificationBertMultiheadLight(nn.Module):
    def __init__(self, name, num_classes=2, num_heads=3, adjust_clf_size=False):
        super(ClassificationBertMultiheadLight, self).__init__()

        self.num_heads = num_heads
        
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained(name)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        assert adjust_clf_size == False, 'Cant adjust clf_size for the light variant.'

        self.classifier_part1 = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.GELU(),
        ])
        _classifier_fn = lambda: nn.Linear(self.num_features, num_classes)
        self.classifier = self.multihead_constructor(_classifier_fn)

    def multihead_constructor(self, constructor):
        return nn.ModuleList([constructor() for _ in range(self.num_heads)])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        
        if only_feat:
            return pooled_output
        
        # logits = self.classifier(pooled_output)
        logits_intermmediate = self.classifier_part1(pooled_output)
        logits = [head_classifier(logits_intermmediate) for head_classifier in self.classifier]

        result_dict = {'logits':logits, 'feat':pooled_output}

        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]
            
        return result_dict
        
        
    def extract(self, x):
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}bert.embeddings'.format(prefix), blocks=r'^{}bert.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []



def bert_base_cased_multihead_light(args, **kwargs):
    model = ClassificationBertMultiheadLight('bert-base-cased', args.num_classes, args.num_heads, args.adjust_clf_size, **kwargs)
    return model


def bert_base_uncased_multihead_light(args, **kwargs):
    model = ClassificationBertMultiheadLight('bert-base-uncased', args.num_classes, args.num_heads, args.adjust_clf_size, **kwargs)
    return model
