# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import jsonlines
import os


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
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
        device = getattr(args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.projection_head = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=128).to(device)
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def _my_stats_log(self, d):
        d['epoch'] = self.epoch
        d['it'] = self.it
        with jsonlines.open(os.path.join(self.args.save_dir, self.args.save_name, 'my_stats.jsonl'), mode='a') as writer:
            writer.write(d)

    def triplet_loss(self, features, labels, mask=None, margin=1.0):
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.uint8)
        batch_size = features.size(0)
        loss = torch.tensor(0.0, device=features.device)
        triplet_cnt = 0

        for i in range(batch_size):
            # If mask is provided, skip elements with mask[i]==0.
            if mask is not None and mask[i] == 0:
                continue

            anchor = features[i]

            # Find all indices of examples with the same label (excluding the anchor)
            similar_idx = ((labels == labels[i]) & (mask != 0)).nonzero(as_tuple=False).squeeze()
            similar_idx = similar_idx[similar_idx != i]
            if similar_idx.dim() == 0:
                similar_idx = similar_idx.unsqueeze(0)
            similar_examples = features[similar_idx]

            # Find all indices of examples with a different label
            dissimilar_idx = ((labels != labels[i]) & (mask != 0)).nonzero(as_tuple=False).squeeze()
            if dissimilar_idx.dim() == 0:
                dissimilar_idx = dissimilar_idx.unsqueeze(0)
            dissimilar_examples = features[dissimilar_idx]

            for similar in similar_examples:
                for dissimilar in dissimilar_examples:
                    # Compute cosine distance (mapped to [0,1])
                    d_similar = (1 - F.cosine_similarity(anchor.unsqueeze(0), similar.unsqueeze(0), dim=1)) / 2
                    d_dissimilar = (1 - F.cosine_similarity(anchor.unsqueeze(0), dissimilar.unsqueeze(0), dim=1)) / 2
                    curr_loss = F.relu(d_similar - d_dissimilar + margin).squeeze()
                    loss += curr_loss
                    triplet_cnt += 1

        if triplet_cnt > 0:
            loss /= triplet_cnt
        return loss
    
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
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

            if self.args.save_pseudolabels_stats:
                my_stats_dict = {
                    'mask_rate': (mask == 0).float().mean().item(),
                    'impurity': (pseudo_label[mask != 0] == y_ulb[mask != 0]).float().mean().item(),
                }
                self._my_stats_log(my_stats_dict)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            proj_lb = self.projection_head(feats_x_lb)
            proj_ulb_w = self.projection_head(feats_x_ulb_w)

            triplet_loss_lb = self.triplet_loss(proj_lb, y_lb)
            triplet_loss_ulb = self.triplet_loss(proj_ulb_w, pseudo_label, mask)

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