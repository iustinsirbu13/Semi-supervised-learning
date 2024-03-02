import torch
import torch.nn.functional as F

from semilearn.algorithms.disaster.fixmatch_disaster import FixMatchDisaster
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('fixmatch_multihead')
class FixMatchMultihead(FixMatchDisaster):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

    # @overrides
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
       
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        logits_x_lb = outputs['logits'][:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
        feats_x_lb = outputs['feat'][:num_lb]
        feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
       
        


        return out_dict, log_dict
