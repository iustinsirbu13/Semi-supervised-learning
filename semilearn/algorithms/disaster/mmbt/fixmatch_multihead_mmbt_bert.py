import torch
from semilearn.algorithms.disaster.fixmatch_multihead import FixMatchMultihead
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('fixmatch_multihead_mmbt_bert')
class FixMatchMultiheadMMBTBert(FixMatchMultihead):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    # @overrides
    def train_step(self,
                   lb_weak_image,
                   lb_weak_sentence, lb_weak_segment, lb_weak_mask,
                   lb_target,
                   ulb_weak_image, ulb_strong_image,
                   ulb_weak_sentence, ulb_weak_segment, ulb_weak_mask,
                   ulb_strong_sentence, ulb_strong_segment, ulb_strong_mask):
        
        lb_batch_size = lb_weak_image.shape[0]
        ulb_batch_size = ulb_weak_image.shape[0]
        lb_target = torch.squeeze(lb_target)

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
        
        logits_x_lb = torch.zeros(self.num_heads, lb_batch_size, self.num_classes).to(self.args.device)
        logits_x_ulb_w = torch.zeros(self.num_heads, ulb_batch_size, self.num_classes).to(self.args.device)
        logits_x_ulb_s = torch.zeros(self.num_heads, ulb_batch_size, self.num_classes).to(self.args.device)

        for head_id in range(self.num_heads):
            logits_x_lb[head_id], logits_x_ulb_w[head_id], logits_x_ulb_s[head_id] = \
                self.get_head_logits(head_id, logits, lb_batch_size)

        # Supervised loss
        lb_loss = self.get_supervised_loss(logits_x_lb, lb_target)

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
    def get_logits(self, data, *args, **kwargs):
        images = data['lb_weak_image'].to(self.args.device)
        sentences = data['lb_weak_sentence'].to(self.args.device)
        segments = data['lb_weak_segment'].to(self.args.device)
        masks = data['lb_weak_mask'].to(self.args.device)
        return self.model(sentences, masks, segments, images)[0]

    # @overrides
    def get_targets(self, data, *args, **kwargs):
        targets = data['lb_target'].to(self.args.device)
        return torch.squeeze(targets)