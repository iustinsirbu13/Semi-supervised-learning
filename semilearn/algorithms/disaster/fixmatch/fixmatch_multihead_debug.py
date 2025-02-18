import torch
import torch.nn.functional as F

from semilearn.algorithms.disaster.fixmatch.fixmatch_base import FixMatchBase
from semilearn.core.utils import ALGORITHMS


import jsonlines
import os

@ALGORITHMS.register('fixmatch_multihead_debug')
class FixMatchMultiheadDebug(FixMatchBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

        # multihead specific arguments
        self.num_heads = args.num_heads        
        self.use_head_cutoff = args.use_head_cutoff

    def get_head_logits(self, head_id, logits, num_lb):
        head_logits = logits[head_id]
        logits_x_lb = head_logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = head_logits[num_lb:].chunk(2)
        return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s
    

    @staticmethod
    def get_majority_element(elements, active):
        count = 0
        for id, element in enumerate(elements):
            if not active[id]:
                continue
            
            candidate = element if count == 0 else candidate
            count += 1 if candidate == element else -1
            
        count = 0
        for id, element in enumerate(elements):
            if not active[id]:
                continue

            count += 1 if candidate == element else -1

        return candidate if count > 0 else -1


    def get_head_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, threshold_masks, y_ulb, head_id):
        num_ulb = ulb_strong_logits[head_id].shape[0]

        pseudo_labels = torch.transpose(pseudo_labels, 0, 1)
        threshold_masks = torch.transpose(threshold_masks, 0, 1)

        multihead_labels = torch.zeros(num_ulb, dtype=torch.int64).to(self.args.device)
        for i in range(num_ulb):
            if self.use_head_cutoff:
                active_heads = threshold_masks[i].cpu()
            else:
                active_heads = torch.ones(self.num_heads, dtype=torch.bool)

            active_heads[head_id] = False
        
            multihead_labels[i] = FixMatchMultiheadDebug.get_majority_element(pseudo_labels[i], active_heads)

        head_pseudo_labels = pseudo_labels[:, head_id]
        my_stats_dict = {
            'head_id': head_id,
            'mask_rate': (multihead_labels == -1).float().mean().item(),
            'impurity': (head_pseudo_labels[multihead_labels != -1] == y_ulb[multihead_labels != -1]).float().mean().item(),
        }
        self._my_stats_log(my_stats_dict)

        return super(FixMatchMultiheadDebug, self).get_unsupervised_loss(ulb_strong_logits[head_id], multihead_labels, multihead_labels != -1)

    # @overrides
    def get_supervised_loss(self, lb_logits, lb_target):
        head_losses = [super(FixMatchMultiheadDebug, self).get_supervised_loss(lb_logits[head_id], lb_target) for head_id in range(self.num_heads)]
        return sum(head_losses)

    # @overrides
    def get_unsupervised_loss(self, ulb_strong_logits, pseudo_labels, threshold_masks, y_ulb):
        head_losses = [self.get_head_unsupervised_loss(ulb_strong_logits, pseudo_labels, threshold_masks, y_ulb, head_id) for head_id in range(self.num_heads)]
        return sum(head_losses) / self.num_heads

    # @overrides
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]

        if self.use_cat:
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
            feat_dict = None
        else:
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            feats_x_ulb_s = outs_x_ulb_s['feat']
            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

        # Supervised loss
        lb_loss = self.get_supervised_loss(logits_x_lb, y_lb)

        # Pseudo labels   
        pseudo_labels = torch.stack([self.get_pseudo_labels(logits_x_ulb_w[head_id])[0] for head_id in range(self.num_heads)])
        threshold_masks = torch.stack([self.get_pseudo_labels(logits_x_ulb_w[head_id])[1] for head_id in range(self.num_heads)])

        # Unsupervised loss
        ulb_loss = self.get_unsupervised_loss(logits_x_ulb_s, pseudo_labels, threshold_masks, y_ulb)

        # Total loss
        loss = self.get_loss(lb_loss, ulb_loss)

        if feat_dict:
            out_dict = self.process_out_dict(loss=loss, feat=feat_dict)
        else:
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

    
    def _my_stats_log(self, d):
        d['epoch'] = self.epoch
        d['it'] = self.it
        with jsonlines.open(os.path.join(self.args.save_dir, self.args.save_name, 'my_stats.jsonl'), mode='a') as writer:
            writer.write(d)
