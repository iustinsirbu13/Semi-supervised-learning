import torch
import torch.nn.functional as F

from semilearn.algorithms.disaster.fixmatch_disaster import FixMatchDisaster
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('fixmatch_multihead')
class FixMatchMultihead(FixMatchDisaster):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

        # multihead specific arguments
        self.num_heads = args.num_heads        

    def get_head_logits(self, head_id, logits, num_lb):
        head_logits = logits[head_id]
        logits_x_lb = head_logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = head_logits[num_lb:].chunk(2)
        return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s
    

    @staticmethod
    def get_majority_element(elements, ignore):
        count = 0
        for id, element in enumerate(elements):
            if id in ignore:
                continue
            
            candidate = element if count == 0 else candidate
            count += 1 if candidate == element else -1
            

        count = 0
        for id, element in enumerate(elements):
            if id in ignore:
                continue

            count += 1 if candidate == element else -1

        return candidate if count > 0 else -1


    def get_head_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, head_id):
        pseudo_labels_transpose = torch.transpose(pseudo_labels, 0, 1)
        num_ulb = ulb_strong_logits[head_id].shape[0]

        multihead_labels = torch.zeros(num_ulb, dtype=torch.int64).to(self.args.device)
        for i in range(num_ulb):
            multihead_labels[i] = FixMatchMultihead.get_majority_element(pseudo_labels_transpose[i], {head_id})

        return super(FixMatchMultihead, self).get_unsupervised_loss(ulb_strong_logits[head_id], multihead_labels, multihead_labels != -1)

    # @overrides
    def get_supervised_loss(self, lb_logits, lb_target):
        head_losses = [super(FixMatchMultihead, self).get_supervised_loss(lb_logits[head_id], lb_target) for head_id in range(self.num_heads)]
        return sum(head_losses)

    # @overrides
    def get_unsupervised_loss(self, ulb_strong_logits, pseudo_labels):
        head_losses = [self.get_head_unsupervised_loss(ulb_strong_logits, pseudo_labels, head_id) for head_id in range(self.num_heads)]
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

        for head_id in range(self.num_heads):
            logits_x_lb[head_id], logits_x_ulb_w[head_id], logits_x_ulb_s[head_id] = \
                self.get_head_logits(head_id, logits, num_lb)

        # Supervised loss
        lb_loss = self.get_supervised_loss(logits_x_lb, y_lb)

        # Pseudo labels
        pseudo_labels = torch.stack([self.get_pseudo_labels(logits_x_ulb_w[head_id])[0] for head_id in range(self.num_heads)])

        # Unsupervised loss
        ulb_loss = self.get_unsupervised_loss(logits_x_ulb_s, pseudo_labels)

        # Total loss
        loss = self.get_loss(lb_loss, ulb_loss)

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(sup_loss=lb_loss.item(), 
                                         unsup_loss=ulb_loss.item(), 
                                         total_loss=loss.item())
        
        return out_dict, log_dict
    
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
