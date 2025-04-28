# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers import BertModel
import os



class ClassificationBertMultinet(nn.Module):
    def __init__(self, name, num_classes=2, num_nets=2):
        super(ClassificationBertMultinet, self).__init__()

        self.num_nets = num_nets
        assert self.num_nets == 2

        self.bert1 = BertModel.from_pretrained(name)
        self.bert2 = BertModel.from_pretrained(name)
        self.dropout1 = torch.nn.Dropout(p=0.1, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.num_features_h = self.num_features

        
        self.classifier1 = nn.Sequential(*[
            nn.Linear(self.num_features, num_classes)
        ])
        
        self.classifier2 = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features_h),
            nn.GELU(),
            nn.Linear(self.num_features_h, num_classes)
        ])
        

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        if only_fc:
            raise NotImplementedError()
            logits = self.classifier(x)
            return logits
        
        out_dict1 = self.bert1(**x, output_hidden_states=True, return_dict=True)
        last_hidden1 = out_dict1['last_hidden_state']
        drop_hidden1 = self.dropout1(last_hidden1)
        pooled_output1 = torch.mean(drop_hidden1, 1)

        out_dict2 = self.bert2(**x, output_hidden_states=True, return_dict=True)
        last_hidden2 = out_dict2['last_hidden_state']
        drop_hidden2 = self.dropout2(last_hidden2)
        pooled_output2 = torch.mean(drop_hidden2, 1)
        
        if only_feat:
            return [pooled_output1, pooled_output2]
        
        logits1 = self.classifier1(pooled_output1)
        logits2 = self.classifier2(pooled_output2)

        result_dict = {'logits': [logits1, logits2], 'feat': [pooled_output1, pooled_output2]}

        if return_embed:
            result_dict['embed'] = [out_dict1['hidden_states'][0], out_dict2['hidden_states'][0]]
            
        return result_dict
        
        
    def extract(self, x):
        # out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        # last_hidden = out_dict['last_hidden_state']
        # drop_hidden = self.dropout(last_hidden)
        # pooled_output = torch.mean(drop_hidden, 1)
        
        out_dict1 = self.bert1(**x, output_hidden_states=True, return_dict=True)
        last_hidden1 = out_dict1['last_hidden_state']
        drop_hidden1 = self.dropout1(last_hidden1)
        pooled_output1 = torch.mean(drop_hidden1, 1)

        out_dict2 = self.bert2(**x, output_hidden_states=True, return_dict=True)
        last_hidden2 = out_dict2['last_hidden_state']
        drop_hidden2 = self.dropout2(last_hidden2)
        pooled_output2 = torch.mean(drop_hidden2, 1)

        return [pooled_output1, pooled_output2]

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}bert.embeddings'.format(prefix), blocks=r'^{}bert.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []

#############
class TextClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))
        
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        predict = self.linear(pooled_output)
        return predict
###########

def bert_base_cased_multinet(args, **kwargs):
    model = ClassificationBertMultinet('bert-base-cased', args.num_classes)
    return model


def bert_base_uncased_multinet(args, **kwargs):
    model = ClassificationBertMultinet('bert-base-uncased', args.num_classes)
    return model
