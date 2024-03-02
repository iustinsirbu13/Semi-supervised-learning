import torch
import torch.nn.functional as F

from semilearn.algorithms.disaster.fixmatch_disaster import FixMatchDisaster
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('fixmatch_multihead')
class FixMatchMultihead(FixMatchDisaster):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

        # multihead specific arguments
        self.init(args.num_heads)

    def init(self, num_heads=3):
        self.num_heads = num_heads

    def get_head_logits(self, head_id, outputs, num_lb):
        output = outputs[head_id]
        logits_x_lb = output['logits'][:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = output['logits'][num_lb:].chunk(2)
        return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s
    
    # @overrides
    def get_supervised_loss(self, lb_logits, lb_target):
        lb_loss = 0
        for head_id in range(self.num_heads):
            head_lb_loss = super().get_supervised_loss(lb_logits[head_id], lb_target)
            lb_loss += head_lb_loss

        return lb_loss / lb_target.shape[0]
    

    @staticmethod
    def get_majority_element(elements, ignore):
        count = 0
        for id, element in enumerate(elements):
            if id in ignore:
                continue
            
            if count == 0:
                candidate = element

            if candidate == element:
                count += 1
            else:
                count -= 1

        if count == 0:
            return -1
        
        return candidate


    def get_head_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, head_id):
        pseudo_labels_transpose = torch.transpose(pseudo_labels)
        num_ulb = ulb_strong_logits[head_id].shape[0]

        multihead_labels = torch.zeros(num_ulb, dtype=torch.int).to(self.args.device)
        for i in range(num_ulb):
            multihead_labels[i] = FixMatchMultihead.get_majority_element(pseudo_labels_transpose[i], {head_id})

        multihead_mask = multihead_labels.ge(0).float()

        return super().get_unsupervised_loss(ulb_strong_logits[head_id], multihead_labels, multihead_mask)
            

    # @overrides
    def get_unsupervised_loss(self, ulb_strong_logits, pseudo_labels):
        ulb_loss  = 0
        for head_id in range(self.num_heads):
            head_ulb_loss = self.get_head_unsupervised_loss(ulb_strong_logits, pseudo_labels, head_id)
            ulb_loss += head_ulb_loss

        return ulb_loss / self.num_heads

    # @overrides
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
       
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        
        logits_x_lb = torch.zeros(self.num_heads, num_lb, self.num_classes, dtype=torch.float32).to(self.args.device)
        logits_x_ulb_w = torch.zeros(self.num_heads, num_ulb, self.num_classes, dtype=torch.float32).to(self.args.device)
        logits_x_ulb_s = torch.zeros(self.num_heads, num_ulb, self.num_classes, dtype=torch.float32).to(self.args.device)

        for head_id in range(self.num_heads):
            logits_x_lb[head_id], logits_x_ulb_w[head_id], logits_x_ulb_s[head_id] = \
                self.get_head_logits(head_id, outputs, num_lb)

        # Supervised loss
        lb_loss = self.get_supervised_loss(logits_x_lb, y_lb)

        # Pseudo labels
        pseudo_labels = [self.get_pseudo_labels(logits_x_ulb_w[head_id])[0] for head_id in range(self.num_heads)]

        # Unsupervised loss
        ulb_loss = self.get_unsupervised_loss(logits_x_ulb_s, pseudo_labels)

        # Total loss
        loss = lb_loss + self.lambda_u * ulb_loss

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(sup_loss=lb_loss.item(), 
                                         unsup_loss=ulb_loss.item(), 
                                         total_loss=loss.item())
        
        return out_dict, log_dict
