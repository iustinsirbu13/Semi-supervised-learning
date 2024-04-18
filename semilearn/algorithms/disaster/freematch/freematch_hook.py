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
        self.global_threshold = torch.tensor(1.0 / self.num_classes).to(self.args.device)

        # initial value of local thresholds is 1 / C
        self.local_thresholds = torch.ones(self.num_classes).to(self.args.device) / self.num_classes

        # combine global and local threshold to compute each class threshold
        self.thresholds = self.get_class_threholds(self.global_threshold, self.local_thresholds)

        # global and local thresholds for next epoch
        self.new_global_threshold = 0
        self.new_local_thresholds = torch.zeros(self.num_classes).to(self.args.device)

        # initial histogram value is 1 / C (we use probability distribution instead of count)
        self.hist = torch.ones(self.num_classes).to(self.args.device) / self.num_classes

        # histogram for the next epoch
        self.new_hist = torch.zeros(self.num_classes).to(self.args.device)


    def get_class_threholds(self, global_threshold, local_thresholds):
        '''
        Self-adaptive threshold (https://arxiv.org/pdf/2205.07246.pdf)
        '''
        return global_threshold * (local_thresholds / torch.max(local_thresholds))

    
    def update(self, algorithm, ulb_weak_logits):
        # max probability for each logit tensor
        # index with highest probability for each logit tensor
        max_probs, pseudo_labels = torch.max(ulb_weak_logits.detach(), dim=-1)

        # update the global threshold for the next epoch
        self.new_global_threshold += (1.0 / self.ulb_size) * torch.sum(max_probs)

        # update the local thresholds for the next epoch
        self.new_local_thresholds += (1.0 / self.ulb_size) * torch.sum(ulb_weak_logits, dim=0)

        threshold_mask = torch.zeros(pseudo_labels.shape[0], dtype=torch.bool).to(self.args.device)
        for i in range(pseudo_labels.shape[0]):
            label = pseudo_labels[i]

            # label exceeds class threshold
            if max_probs[i] >= self.thresholds[label]:
                # mark it as True inside mask
                threshold_mask[i] = True

                # update histogram for the next epoch
                self.new_hist[label] += 1
        

        return threshold_mask


    def after_train_epoch(self, algorithm):
        # use EMA decay to update global/local thresholds
        self.global_threshold = self.ema_decay * self.global_threshold + (1 - self.ema_decay) * self.new_global_threshold
        self.local_thresholds = self.ema_decay * self.local_thresholds + (1 - self.ema_decay) * self.new_local_thresholds

        # compute the new class thresholds
        self.thresholds = self.get_class_threholds(self.global_threshold, self.local_thresholds)

        # use EMA decay to update histogram
        self.hist = self.ema_decay * self.hist + (1 - self.ema_decay) * (self.new_hist / torch.sum(self.new_hist))

        algorithm.print_fn(f'Global threshold = {self.global_threshold}')
        algorithm.print_fn(f'Local class thresholds = {self.local_thresholds}')
        algorithm.print_fn(f'Self-adaptive class thresholds = {self.thresholds}')
        algorithm.print_fn(f'Histogram = {self.hist}')

        # reset global/local thresholds for the next epoch
        self.new_global_threshold = torch.tensor(0).to(self.args.device)
        self.new_local_thresholds = torch.zeros(self.num_classes).to(self.args.device)

        # reset histogram
        self.new_hist = torch.zeros(self.num_classes).to(self.args.device)


