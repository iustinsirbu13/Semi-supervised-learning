import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from inspect import signature


@ALGORITHMS.register('fixmatch_mmbt_bert')
class FixMatchMMBTBert(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

    def set_model(self):
        """
        initialize model
        """
        model = self.net_builder(self.args)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(self.args)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model
    
    def train_step(lb_weak_image,
                   lb_weak_sentence, lb_weak_segment, lb_weak_mask,
                   target,
                   ulb_weak_image, ulb_strong_image,
                   ulb_weak_sentence, ulb_weak_segment, ulb_weak_mask,
                   ulb_strong_sentence, ulb_strong_segment, ulb_strong_mask):
        pass
    
    