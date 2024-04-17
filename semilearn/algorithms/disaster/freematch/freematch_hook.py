import torch
import numpy as np
from semilearn.core.hooks import Hook

class FreeMatchHook(Hook):
    def __init__(self, args):
        self.p_cutoff = args.p_cutoff
        self.num_classes = args.num_classes
        self.ulb_size = len(args.wrapper.unlabeled_dataset)
        self.ema_decay = args.freematch_ema_decay
        self.args = args

        # initial value of global_threshold is 1 / C
        self.global_threshold = 1.0 / self.num_classes

        # initial value of local thresholds is 1 / C
        self.local_thresholds = np.ones(self.num_classes) / self.num_classes

        # combine global and local threshold to compute each class threshold
        self.thresholds = self.get_class_threholds(self.global_threshold, self.local_thresholds)

        # global and local thresholds for next epoch
        self.new_global_threshold = 0
        self.new_local_thresholds = np.zeros(self.num_classes)


    def get_class_threholds(self, global_threshold, local_thresholds):
        '''
        Self-adaptive threshold (https://arxiv.org/pdf/2205.07246.pdf)
        '''
        max_local_threshold = np.max(local_thresholds)
        return global_threshold * (local_thresholds / max_local_threshold)

    
    def update(self, algorithm, ulb_weak_logits):
        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        max_probs, pseudo_labels = torch.max(ulb_weak_logits, dim=-1)

        threshold_mask = torch.zeros(pseudo_labels.shape[0], dtype=torch.bool).to(self.args.device)
        for i in range(pseudo_labels.shape[0]):
            label = pseudo_labels[i]

            # label exceeds class threshold => mark it as True inside mask
            if max_probs[i] >= self.thresholds[label]:
                threshold_mask[i] = True

            # update the global threshold for the next epoch
            self.new_global_threshold += (1.0 / self.ulb_size) * max_probs[i]
            
            # update the local thresholds for the next epoch
            for label in range(self.num_classes):
                self.new_local_thresholds[label] += (1.0 / self.ulb_size) * ulb_weak_logits[i][label]

        return threshold_mask


    def after_train_epoch(self, algorithm):
        # use EMA decay to update global/local thresholds
        self.global_threshold = self.ema_decay * self.global_threshold + (1 - self.ema_decay) * self.new_global_threshold
        self.local_thresholds = self.ema_decay * self.local_thresholds + (1 - self.ema_decay) * self.new_local_thresholds

        # compute the new class thresholds
        self.thresholds = self.get_class_threholds(self.global_threshold, self.local_thresholds)

        # reset global/local thresholds for the next epoch
        self.new_global_threshold = 0
        self.new_local_thresholds = np.zeros(self.num_classes)


