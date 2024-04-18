import torch
from semilearn.core.hooks import Hook

class FlexMatchHook(Hook):
    def __init__(self, args):
        self.p_cutoff = args.p_cutoff
        self.num_classes = args.num_classes
        self.ulb_size = len(args.wrapper.unlabeled_dataset)
        self.args = args

        # class counts should be 0 at the beginning of each epoch
        self.threshold_count = torch.zeros(self.num_classes).to(self.args.device)

        # set initial class thresholds equal to the global threshold (should be 0 according to paper, but this looks strange)
        self.thresholds = torch.ones(self.num_classes).to(self.args.device) * self.p_cutoff

    
    def update(self, algorithm, ulb_weak_logits):
        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        max_probs, pseudo_labels = torch.max(ulb_weak_logits, dim=-1)

        # label exceeds global threshold => increase count for the corresponding class
        self.threshold_count = torch.bincount(pseudo_labels[max_probs >= self.p_cutoff])
        self.threshold_count = torch.cat((self.threshold_count, torch.zeros(self.num_classes - self.threshold_count.shape[0]).to(self.args.device)))

        threshold_mask = torch.zeros(pseudo_labels.shape[0], dtype=torch.bool).to(self.args.device)
        for i in range(pseudo_labels.shape[0]):
            label = pseudo_labels[i]

            # label exceeds class threshold => mark it as True inside mask
            if max_probs[i] >= self.thresholds[label]:
                threshold_mask[i] = True

        return threshold_mask


    def after_train_epoch(self, algorithm):
        max_threshold_count = torch.max(self.threshold_count)
        sum_threshold_count = torch.sum(self.threshold_count)

        # use formula from paper (https://arxiv.org/pdf/2110.08263.pdf)
        self.thresholds = self.p_cutoff * (self.threshold_count / torch.max(max_threshold_count, self.ulb_size - sum_threshold_count))

        # class count is 0 => we will use global threshold in this case
        self.thresholds[self.threshold_count == 0] = self.p_cutoff

        algorithm.print_fn(f'Class threshold counts = {self.threshold_count}')
        algorithm.print_fn(f'Class thresholds = {self.thresholds}')

        # prepare for new epoch => reset class counts to 0
        self.threshold_count = torch.zeros(self.num_classes).to(self.args.device)
