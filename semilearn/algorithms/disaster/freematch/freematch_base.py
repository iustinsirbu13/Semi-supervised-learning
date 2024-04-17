import torch
import torch.nn.functional as F
import numpy as np
from semilearn.algorithms.disaster.freematch.freematch_hook import FreeMatchHook
from semilearn.core.algorithmbase import AlgorithmBase

class FreeMatchBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log=tb_log, logger=logger)

    # @overrides
    def set_model(self):
        """
        initialize model
        """
        model = self.net_builder(self.args)
        return model

    # @overrides
    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(self.args)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model        

    # @overrides
    def set_hooks(self):
        super().set_hooks()
        self.register_hook(FreeMatchHook(self.args), 'FreeMatchHook')

    # @overrides
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['global_threshold'] = self.hooks_dict['FreeMatchHook'].global_threshold
        save_dict['local_thresholds'] = self.hooks_dict['FreeMatchHook'].local_thresholds
        return save_dict

    # @overrides
    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['FreeMatchHook'].global_threshold = checkpoint['global_threshold']
        self.hooks_dict['FreeMatchHook'].local_thresholds = checkpoint['local_thresholds']
        self.print_fn("additional parameter loaded")
        return checkpoint

    def get_supervised_loss(self, lb_logits, lb_target):
        return F.cross_entropy(lb_logits, lb_target)

    def get_pseudo_labels(self, ulb_weak_logits):
        # normalized logits (range [0, 1])
        ulb_weak_logits_norm = torch.softmax(ulb_weak_logits.detach(), dim=-1)

        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        max_probs, pseudo_labels = torch.max(ulb_weak_logits_norm, dim=-1)

        threshold_mask = self.call_hook('update', 'FreeMatchHook', ulb_weak_logits=ulb_weak_logits_norm)

        return pseudo_labels, threshold_mask

    def get_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, threshold_mask):
        if 1 not in threshold_mask:
            return torch.tensor(0).to(self.args.device)

        ulb_strong_logits = ulb_strong_logits[threshold_mask == 1]
        pseudo_labels = pseudo_labels[threshold_mask == 1]

        return F.cross_entropy(ulb_strong_logits, pseudo_labels)

    def get_loss(self, lb_loss, ulb_loss):
        return lb_loss + self.lambda_u * ulb_loss
