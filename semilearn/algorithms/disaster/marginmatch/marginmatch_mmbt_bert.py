
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

from semilearn.algorithms.disaster.marginmatch.marginmatch_hook import MarginMatchHook
from semilearn.algorithms.disaster.marginmatch.marginmatch_log_hook import MarginMatchLogHook
from semilearn.algorithms.disaster.flexmatch.flexmatch_log_hook import FlexMatchLogHook
from semilearn.algorithms.freematch.utils import FreeMatchThresholdingHook
from semilearn.algorithms.disaster.freematch.freematch_log_hook import FreeMatchLogHook
from semilearn.algorithms.flexmatch.utils import FlexMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('marginmatch_mmbt_bert')
class MarginMatchMMBTBert(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh,
                  p_cutoff=args.p_cutoff, thresh_warmup=args.thresh_warmup, threshold_algo=args.threshold_algo)
        
        super().__init__(args, net_builder, tb_log, logger) 
    

    def init(self, T, p_cutoff, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False, thresh_warmup=True, threshold_algo='freematch'):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        
        self.thresh_warmup = thresh_warmup

        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh

        self.threshold_algo = threshold_algo

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
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")

        self.register_hook(MarginMatchHook(self.args), "PartialMarginHook")
        self.register_hook(MarginMatchLogHook(), "MarginMatchLogHook")

        if self.threshold_algo == 'flexmatch':
            self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
            self.register_hook(FlexMatchLogHook(), "FlexMatchLogHook")
        else:
            self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
            self.register_hook(FreeMatchLogHook(), 'FreeMatchLogHook')

        super().set_hooks()


    def train_step(self,
                   lb_weak_image,
                   lb_weak_sentence, lb_weak_segment, lb_weak_mask,
                   lb_target,
                   ulb_weak_image, ulb_strong_image,
                   ulb_weak_sentence, ulb_weak_segment, ulb_weak_mask,
                   ulb_strong_sentence, ulb_strong_segment, ulb_strong_mask,
                   idx_ulb):
        lb_batch_size = lb_weak_image.shape[0]
        idx_ulb = idx_ulb.to(self.args.device)

        images = torch.cat((lb_weak_image, ulb_weak_image, ulb_strong_image))
        sentences = torch.cat((lb_weak_sentence, ulb_weak_sentence, ulb_strong_sentence))
        segments = torch.cat((lb_weak_segment, ulb_weak_segment, ulb_strong_segment))
        masks = torch.cat((lb_weak_mask, ulb_weak_mask, ulb_strong_mask))

        images = images.to(self.args.device)
        sentences = sentences.to(self.args.device)
        segments = segments.to(self.args.device)
        masks = masks.to(self.args.device)
        lb_target = lb_target.to(self.args.device)

        logits = self.model(sentences, masks, segments, images)
        logits_x_lb = logits[:lb_batch_size]
        logits_x_ulb_w, logits_x_ulb_s = logits[lb_batch_size:].chunk(2)

        sup_loss = self.ce_loss(logits_x_lb, lb_target, reduction='mean')

        # calculate mask
        if self.threshold_algo == 'freematch':
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
        else:
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)

        
        # Average Partial Margin mask
        apm_mask = self.call_hook("masking", "PartialMarginHook", logits_x_ulb_w=logits_x_ulb_w, logits_x_ulb_s=logits_x_ulb_s, idx_ulb=idx_ulb)

        # Keep unlabeled data that passed both masking checks
        mask = mask * apm_mask

        # generate unlabeled targets using pseudo label hook
        pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                        logits=logits_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T)
        
        # calculate unlabeled loss
        unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                        pseudo_label,
                                        'ce',
                                        mask=mask)
        
        total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()

        # additional saving arguments
        if self.threshold_algo == 'freematch':
            save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
            save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        else:
            save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
            save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()

        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        
        if self.threshold_algo == 'freematch':
            self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
            self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        else:
            self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
            self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
            
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]
    
    # @overrides
    def get_logits(self, data, *args, **kwargs):
        images = data['lb_weak_image'].to(self.args.device)
        sentences = data['lb_weak_sentence'].to(self.args.device)
        segments = data['lb_weak_segment'].to(self.args.device)
        masks = data['lb_weak_mask'].to(self.args.device)
        return self.model(sentences, masks, segments, images)

    # @overrides
    def get_targets(self, data, *args, **kwargs):
        targets = data['lb_target'].to(self.args.device)
        return torch.squeeze(targets)