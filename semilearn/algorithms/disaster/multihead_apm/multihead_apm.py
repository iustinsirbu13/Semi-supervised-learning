import torch
import torch.nn.functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('multihead_apm')
class MultiheadAPM(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

        # multihead specific arguments
        self.num_heads = args.num_heads

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

    def get_head_logits(self, head_id, logits, num_lb):
        head_logits = logits[head_id]
        logits_x_lb = head_logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = head_logits[num_lb:].chunk(2)
        return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s

    def get_head_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, threshold_masks, head_id):
        # ToDo:
        pass

    def get_supervised_loss(self, lb_logits, lb_target):
        head_losses = [F.cross_entropy(lb_logits[head_id], lb_target) for head_id in range(self.num_heads)]
        return sum(head_losses)

    def get_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, threshold_masks):
        head_losses = [self.get_head_unsupervised_loss(ulb_strong_logits, pseudo_labels, threshold_masks, head_id) for head_id in range(self.num_heads)]
        return sum(head_losses) / self.num_heads

    # @overrides
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
       
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        inputs = inputs.to(self.args.device)
        logits = self.model(inputs)['logits']
        
        logits_x_lb = torch.zeros(self.num_heads, num_lb, self.num_classes).to(self.args.device)
        logits_x_ulb_w = torch.zeros(self.num_heads, num_ulb, self.num_classes).to(self.args.device)
        logits_x_ulb_s = torch.zeros(self.num_heads, num_ulb, self.num_classes).to(self.args.device)

        # ToDo:
        pass
    
    # @overrides
    def get_logits(self, data, out_key):
        x = data['x_lb']
        if isinstance(x, dict):
            x = {k: v.to(self.args.device) for k, v in x.items()}
        else:
            x = x.to(self.args.device)  
        
        logits = self.model(x)[out_key]

        # Use all heads for prediction
        return sum(logits) / self.num_heads
