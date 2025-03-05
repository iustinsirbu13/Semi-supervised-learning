# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()
    
    def triplet_loss(self, features, labels, mask=None, margin=1.0):
        batch_size = features.size()[0]
        loss = torch.tensor(0.0, device=features.device)
        triplet_cnt = 0

        for i in range(batch_size):
            if mask is not None and mask[i] == 0:
                continue

            anchor = features[i]

            # select all examples with same label as anchor
            similar_idx = ((labels == labels[i]) & (mask != 0)).nonzero(as_tuple=False).squeeze()
            similar_idx = similar_idx[similar_idx != i]

            # make tensor if there is only 1 example
            if similar_idx.dim() == 0:
                similar_idx = similar_idx.unsqueeze(0)
            similar_examples = features[similar_idx]
        
            #select all examples with different label than anchor
            dissimilar_idx = ((labels != labels[i]) & (mask != 0)).nonzero(as_tuple=False).squeeze()

            # make tensor if there is only 1 example
            if dissimilar_idx.dim() == 0:
                dissimilar_idx = dissimilar_idx.unsqueeze(0)
            dissimilar_examples = features[dissimilar_idx]

            for similar in similar_examples:
                for dissimilar in dissimilar_examples:
                    # calculating distance with cosine similarity: 1 if similar, 0 if different
                    # unsqueeze to have size of batch (1) in first dimension for the function to work
                    d_similar = torch.norm(anchor - similar, p=2)
                    d_dissimilar = torch.norm(anchor - dissimilar, p=2)
                    curr_loss = F.relu(d_similar - d_dissimilar + margin).squeeze()
                    loss += curr_loss
                    triplet_cnt += 1

        # divide loss by number of triplets
        if triplet_cnt > 0:
            loss /= triplet_cnt
        return loss

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
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

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)
            
            triplet_loss_lb = self.triplet_loss(feats_x_lb, y_lb)
            triplet_loss_ulb = self.triplet_loss(feats_x_ulb_w, pseudo_label, mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + triplet_loss_lb + triplet_loss_ulb

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         triplet_loss_lb=triplet_loss_lb,
                                         triplet_loss_ulb=triplet_loss_ulb,
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]