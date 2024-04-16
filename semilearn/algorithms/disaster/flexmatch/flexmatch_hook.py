import torch
import numpy as np

class FlexMatchHook:
    def __init__(self, args):
        self.p_cutoff = args.p_cutoff
        self.num_classes = args.num_classes
        self.ulb_dataset_size = len(args.wrapper.unlabeled_dataset)
        self.args = args

        # class counts should be 0 at the beginning of each epoch
        self.threshold_count = np.zeros(self.num_classes)

        # set initial class thresholds equal to the global threshold (should be 0 according to paper, but this looks strange)
        self.thresholds = np.empty(self.num_classes)
        self.thresholds.fill(self.p_cutoff)

    
    def update(self, ulb_weak_logits):
        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        max_probs, pseudo_labels = torch.max(ulb_weak_logits, dim=-1)

        threshold_mask = torch.zeros(pseudo_labels.shape[0], dtype=torch.bool).to(self.args.device)
        for i in range(pseudo_labels.shape[0]):
            label = pseudo_labels[i]

            # label exceeds class threshold => mark it as True inside mask
            if max_probs[i] >= self.thresholds[label]:
                threshold_mask[i] = True

            # label exceeds global threshold => increase count for the corresponding class
            if max_probs[i] >= self.p_cutoff:
                self.threshold_count[label] += 1

        return threshold_mask


    def after_train_epoch(self):
        max_threshold_count = np.max(self.threshold_count)
        sum_threshold_count = np.sum(self.threshold_count)
        
        for label in range(self.num_classes):
            count = self.threshold_count[label]
            if count == 0:
                # class count is 0 => we will use global threshold in this case
                self.thresholds[label] = self.p_cutoff
            else:
                # use formula from paper (https://arxiv.org/pdf/2110.08263.pdf)
                self.thresholds[label] = self.p_cutoff * (count / max(max_threshold_count, self.ulb_size - sum_threshold_count))

        # prepare for new epoch => reset class counts to 0
        self.threshold_count.fill(0)
