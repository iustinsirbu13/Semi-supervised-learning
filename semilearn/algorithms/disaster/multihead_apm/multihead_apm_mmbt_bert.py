import torch
from semilearn.algorithms.disaster.multihead_apm.multihead_apm import MultiheadAPM
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('multihead_apm_mmbt_bert')
class MultiheadAPMMMBTBert(MultiheadAPM):
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

        # ToDo:
        pass


    # @overrides
    def get_logits(self, data, *args, **kwargs):
        images = data['lb_weak_image'].to(self.args.device)
        sentences = data['lb_weak_sentence'].to(self.args.device)
        segments = data['lb_weak_segment'].to(self.args.device)
        masks = data['lb_weak_mask'].to(self.args.device)

        logits = self.model(sentences, masks, segments, images)

        # Use all heads for prediction
        return sum(logits) / self.num_heads

    # @overrides
    def get_targets(self, data, *args, **kwargs):
        targets = data['lb_target'].to(self.args.device)
        return torch.squeeze(targets)
