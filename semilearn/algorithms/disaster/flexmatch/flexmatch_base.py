import torch
import torch.nn.functional as F
import numpy as np
from semilearn.core.algorithmbase import AlgorithmBase

class FlexMatchBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log=tb_log, logger=logger)

        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff)
    
    def init(self, T, p_cutoff):
        self.T = T
        self.p_cutoff = p_cutoff

        self.thresholds = np.ones(self.num_classes)
        self.threshold_count = np.zeros(self.num_classes)

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

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(self.update_thresholds, 'after_train_epoch')

    def update_thresholds(self):
        max_count = np.max(self.threshold_count)
        if max_count == 0:
            return
        
        self.thresholds = self.threshold_count / max_count
        self.print_fn(f'Threshold_count: f{self.threshold_count}')
        self.print_fn(f'Thresholds: f{self.thresholds}')

        self.threshold_count.fill(0)

    def get_supervised_loss(self, lb_logits, lb_target):
        return F.cross_entropy(lb_logits, lb_target)

    def get_pseudo_labels(self, ulb_weak_logits):
        # normalized logits (range [0, 1])
        ulb_weak_logits_norm = torch.softmax(ulb_weak_logits.detach(), dim=-1)

        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        max_probs, pseudo_labels = torch.max(ulb_weak_logits_norm, dim=-1)

        threshold_mask = torch.zeros(pseudo_labels.shape[0], dtype=torch.bool).to(self.args.device)
        for i in range(pseudo_labels.shape[0]):
            if max_probs[i] >= self.thresholds[pseudo_labels[i]]:
                threshold_mask[i] = True
                self.threshold_count[pseudo_labels[i]] += 1

        return pseudo_labels, threshold_mask

    def get_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, threshold_mask):
        if 1 not in threshold_mask:
            return torch.tensor(0).to(self.args.device)

        ulb_strong_logits = ulb_strong_logits[threshold_mask == 1]
        pseudo_labels = pseudo_labels[threshold_mask == 1]

        return F.cross_entropy(ulb_strong_logits, pseudo_labels)

    def get_loss(self, lb_loss, ulb_loss):
        return lb_loss + self.lambda_u * ulb_loss
