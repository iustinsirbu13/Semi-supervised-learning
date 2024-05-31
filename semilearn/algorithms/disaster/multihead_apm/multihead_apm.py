import torch
import torch.nn.functional as F

from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.disaster.multihead_apm.apm_hook import APMHook
from semilearn.algorithms.disaster.multihead_apm.apm_log_hook import APMLogHook
from semilearn.algorithms.disaster.multihead_apm.debug_hook import DebugHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('multihead_apm')
class MultiheadAPM(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

        # multihead specific arguments
        self.num_heads = args.num_heads

        self.use_debug = args.use_debug

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
        self.register_hook(APMHook(self.args, APMLogHook()), "APMHook")
        
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

        self.call_hook("update", "APMHook", logits_x_ulb_w=ulb_weak_logits[head_id], idx_ulb=idx_ulb, head_id=head_id)

        num_ulb = idx_ulb.shape[0]
        multihead_labels = torch.ones(num_ulb, dtype=torch.int64).to(self.args.device) * -1

        for i in range(num_ulb):
            label1 = pseudo_labels[head_id1][i]
            label2 = pseudo_labels[head_id2][i]

            # If both labels are equal we use the current sample without checking the APM threshold
            if label1 == label2:
                multihead_labels[i] = label1
                continue
            
            multihead_labels[i] = self.call_hook("get_apm_label", "APMHook", head_id=head_id, idx=idx_ulb[i], label1=label1, label2=label2)
            
        mask = multihead_labels != -1
        if 1 not in mask:
            return torch.tensor(0).to(self.args.device)

        return F.cross_entropy(ulb_strong_logits[head_id][mask == 1], multihead_labels[mask == 1])


    def get_unsupervised_loss(self, ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb):
        head_losses = [self.get_head_unsupervised_loss(ulb_weak_logits, ulb_strong_logits, pseudo_labels, idx_ulb, head_id) for head_id in range(self.num_heads)]
        return sum(head_losses) / self.num_heads
    
    def get_loss(self, lb_loss, ulb_loss):
        return lb_loss + self.lambda_u * ulb_loss
    
    def train_step_base(self, logits, y_lb, idx_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = idx_ulb.shape[0]

        logits_x_lb = torch.zeros(self.num_heads, num_lb, self.num_classes).to(self.args.device)
        logits_x_ulb_w = torch.zeros(self.num_heads, num_ulb, self.num_classes).to(self.args.device)
        logits_x_ulb_s = torch.zeros(self.num_heads, num_ulb, self.num_classes).to(self.args.device)

        for head_id in range(self.num_heads):
            logits_x_lb[head_id], logits_x_ulb_w[head_id], logits_x_ulb_s[head_id] = \
                self.get_head_logits(head_id, logits, num_lb)
            
        # Supervised loss
        lb_loss = self.get_supervised_loss(logits_x_lb, y_lb)

        # Pseudo labels   
        pseudo_labels = torch.stack([self.get_pseudo_labels(logits_x_ulb_w[head_id]) for head_id in range(self.num_heads)])

        # Unsupervised loss
        ulb_loss = self.get_unsupervised_loss(logits_x_ulb_w, logits_x_ulb_s, pseudo_labels, idx_ulb)

        # Total loss
        loss = self.get_loss(lb_loss, ulb_loss)

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(sup_loss=lb_loss.item(), 
                                         unsup_loss=ulb_loss.item(), 
                                         total_loss=loss.item())
        
        return out_dict, log_dict


    # @overrides
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_ulb):       
        idx_ulb = idx_ulb.to(self.args.device)

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        inputs = inputs.to(self.args.device)
        logits = self.model(inputs)['logits']

        return self.train_step_base(logits, y_lb, idx_ulb)
    
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
            SSL_Argument('--use_debug', str2bool, False)
        ]
