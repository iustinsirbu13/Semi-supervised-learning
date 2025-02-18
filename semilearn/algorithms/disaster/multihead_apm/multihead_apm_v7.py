import torch
import torch.nn.functional as F

from semilearn.algorithms.utils import SSL_Argument, str2bool
# from semilearn.algorithms.disaster.multihead_apm.apm_hook import APMHook
# from semilearn.algorithms.disaster.multihead_apm.apm_hook_v3 import APMHook as APMHookV3
from semilearn.algorithms.disaster.multihead_apm.apm_hook_v7 import APMHook as APMHookV7
from semilearn.algorithms.disaster.multihead_apm.apm_log_hook import APMLogHook
from semilearn.algorithms.disaster.multihead_apm.debug_hook import DebugHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('multihead_apm_v7')
class MultiheadAPMv7(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.use_debug = args.use_debug

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

    # @overrides
    def set_hooks(self):
        # if self.args.multihead_apm_variant == 'original':
        #     self.register_hook(APMHook(self.args, APMLogHook()), "APMHook")
        #     raise NotImplementedError(f'The classical APMHook is not adated for the V7 variant yet.')
        # elif self.args.multihead_apm_variant in ['v3', 'v4', 'v5', 'v6']:
        #     self.register_hook(APMHookV7(self.args, APMLogHook()), "APMHook")
        # else:
        #     raise ValueError(f'{self.args.multihead_apm_variant} is not valid.')

        self.register_hook(APMHookV7(self.args, APMLogHook()), "APMHook")
        if self.use_debug:
            self.register_hook(DebugHook(), "DebugHook")

        super().set_hooks()

    def get_head_logits(self, head_id, logits, num_lb):
        head_logits = logits[head_id]
        logits_x_lb = head_logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = head_logits[num_lb:].chunk(2)
        return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s
    
    def get_pseudo_labels(self, ulb_weak_logits):
        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        _, pseudo_labels = torch.max(ulb_weak_logits, dim=-1)
        return pseudo_labels
    
    def get_supervised_loss(self, lb_logits, lb_target):
        head_losses = [F.cross_entropy(lb_logits[head_id], lb_target) for head_id in range(self.num_heads)]
        return sum(head_losses)

    def get_head_unsupervised_loss(self, ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb, head_id):
        '''
        This works only for 3 heads
        '''
        if head_id == 0:
            head_id1, head_id2 = 1, 2
        elif head_id == 1:
            head_id1, head_id2 = 0, 2
        else:
            head_id1, head_id2 = 0, 1

        num_ulb = idx_ulb.shape[0]
        multihead_labels = torch.ones(num_ulb, dtype=torch.int64).to(self.args.device) * -1

        for i in range(num_ulb):
            label1 = pseudo_labels[head_id1][i]
            label2 = pseudo_labels[head_id2][i]
            multihead_labels[i] = self.call_hook("get_apm_label", "APMHook", head_id=head_id, head_id1=head_id1, head_id2=head_id2, idx=idx_ulb[i], label1=label1, label2=label2)
        
        mask = multihead_labels != -1
        if 1 not in mask:
            return torch.tensor(0).to(self.args.device)

        return F.cross_entropy(ulb_strong_logits[head_id][mask == 1], multihead_labels[mask == 1])

    def get_head_unsupervised_loss_v2(self, ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb, head_id):
        '''
        This works only for 3 heads
        '''
        if head_id == 0:
            head_id1, head_id2 = 1, 2
        elif head_id == 1:
            head_id1, head_id2 = 0, 2
        else:
            head_id1, head_id2 = 0, 1

        num_ulb = idx_ulb.shape[0]
        multihead_labels = torch.ones(num_ulb, dtype=torch.int64).to(self.args.device) * -1
        multihead_agreement_types = torch.ones(num_ulb, dtype=torch.int64).to(self.args.device) * -1
        agreement_types_mask = torch.ones(num_ulb, dtype=torch.int64).to(self.args.device) * -1

        for i in range(num_ulb):
            label1 = pseudo_labels[head_id1][i]
            label2 = pseudo_labels[head_id2][i]
            multihead_labels[i], multihead_agreement_types[i], agreement_types_mask[i] = self.call_hook(
                "get_apm_label_v2", "APMHook", head_id=head_id, head_id1=head_id1, head_id2=head_id2, idx=idx_ulb[i], label1=label1, label2=label2)
        
        if self.args.multihead_apm_variant in ['v4', 'v5', 'v6']:
            multihead_labels[multihead_labels == -1] = 0 # can't have labels -1, even though the weight will be 0
            samples_weights = (agreement_types_mask == 0) * self.args.apm_disagreement_weight + (agreement_types_mask == 1) * 1
            return (F.cross_entropy(ulb_strong_logits[head_id], multihead_labels, reduction='none') * samples_weights).mean()

        if self.args.apm_disagreement_weight == -1:
            # legacy code
            mask = multihead_labels != -1
            if 1 not in mask:
                return torch.tensor(0).to(self.args.device)
            return F.cross_entropy(ulb_strong_logits[head_id][mask == 1], multihead_labels[mask == 1])
        else:
            if 0 in agreement_types_mask:
                ce_disagreement = F.cross_entropy(ulb_strong_logits[head_id][agreement_types_mask == 0], multihead_labels[agreement_types_mask == 0]) 
            else:
                ce_disagreement = torch.tensor(0).to(self.args.device)
            if 1 in agreement_types_mask:
                ce_agreement = F.cross_entropy(ulb_strong_logits[head_id][agreement_types_mask == 1], multihead_labels[agreement_types_mask == 1]) 
            else:
                ce_agreement = torch.tensor(0).to(self.args.device)
            return self.args.apm_disagreement_weight * ce_disagreement + (1 - self.args.apm_disagreement_weight) * ce_agreement

    def get_unsupervised_loss(self, ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb):
        for head_id in range(self.num_heads):
            self.call_hook("update", "APMHook", logits_x_ulb_w=ulb_weak_logits[head_id], logits_x_ulb_s=ulb_strong_logits[head_id], idx_ulb=idx_ulb, head_id=head_id)
        
        head_losses = [self.get_head_unsupervised_loss_v2(ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb, head_id) for head_id in range(self.num_heads)]
        return sum(head_losses) / self.num_heads
    
    def get_loss(self, lb_loss, ulb_loss):
        return lb_loss + self.lambda_u * ulb_loss
    
    def _post_process_logits(self, logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, y_lb, idx_ulb, feat_dict=None):
         # Supervised loss
        lb_loss = self.get_supervised_loss(logits_x_lb, y_lb)

        # Pseudo labels   
        pseudo_labels = torch.stack([self.get_pseudo_labels(logits_x_ulb_w[head_id]) for head_id in range(self.num_heads)])

        # Unsupervised loss
        ulb_loss = self.get_unsupervised_loss(logits_x_ulb_w, logits_x_ulb_s, pseudo_labels, idx_ulb)

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

    def train_step_base(self, logits, y_lb, idx_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = idx_ulb.shape[0]

        logits_x_lb = torch.zeros(self.num_heads, num_lb, self.num_classes).to(self.args.device)
        logits_x_ulb_w = torch.zeros(self.num_heads, num_ulb, self.num_classes).to(self.args.device)
        logits_x_ulb_s = torch.zeros(self.num_heads, num_ulb, self.num_classes).to(self.args.device)

        for head_id in range(self.num_heads):
            logits_x_lb[head_id], logits_x_ulb_w[head_id], logits_x_ulb_s[head_id] = \
                self.get_head_logits(head_id, logits, num_lb)

        return self._post_process_logits(logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, y_lb, idx_ulb)


    # @overrides
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_ulb):       
        idx_ulb = idx_ulb.to(self.args.device)

        if self.use_cat:
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            inputs = inputs.to(self.args.device)
            logits = self.model(inputs)['logits']
            return self.train_step_base(logits, y_lb, idx_ulb)
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

            return self._post_process_logits(logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, y_lb, idx_ulb, feat_dict=feat_dict)
    
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
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--use_debug', str2bool, False),
            SSL_Argument('--num_heads', int, 3),
            SSL_Argument('--smoothness', float, 0.997),
            SSL_Argument('--no_low', str2bool, False),
            SSL_Argument('--apm_disagreement_weight', float, -1), # in [0, 1] if set
            SSL_Argument('--adjust_clf_size', str2bool, False),
            SSL_Argument('--multihead_apm_variant', str, "original"),
            SSL_Argument('--num_recalibrate_iter', int, 0), # if 0, it will be done every epoch
        ]
