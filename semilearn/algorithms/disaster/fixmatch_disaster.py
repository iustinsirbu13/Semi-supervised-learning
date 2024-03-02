import torch
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase

class FixMatchDisaster(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log=tb_log, logger=logger)

        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.hard_label = hard_label

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
    
    def get_supervised_loss(self, lb_logits, lb_target):
        return F.cross_entropy(lb_logits, lb_target)

    def get_pseudo_labels(self, ulb_weak_logits):
        # normalized logits (range [0, 1])
        ulb_weak_logits_norm = torch.softmax(ulb_weak_logits.detach(), dim=-1)

        if self.hard_label:
            # max probability for each logit tensor
            # index with highest probability for each logit tensor
            max_probs, pseudo_labels = torch.max(ulb_weak_logits_norm, dim=-1)
            threshold_mask = max_probs.ge(self.p_cutoff).float()
        else:
            pseudo_labels = ulb_weak_logits_norm
            threshold_mask = torch.ones(pseudo_labels.shape[0]).float()

        return pseudo_labels, threshold_mask

    def get_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, threshold_mask):
        # Should work for both "hard" and "soft" labels (if "soft" labels are used, threshold_mask is ignored)
        entropy = F.cross_entropy(ulb_strong_logits, pseudo_labels, reduction='none')
        if self.hard_label:
            return (threshold_mask * entropy).mean()
        return entropy.mean()

    def get_loss(self, lb_loss, ulb_loss):
        # ToDo: Add support for linear lambda_u
        loss = lb_loss + self.lambda_u * ulb_loss
        return loss