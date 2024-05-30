import torch
from semilearn.core.hooks import Hook

class APMHook(Hook):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.num_heads = args.num_heads
        self.num_classes = args.num_classes
        self.ulb_dest_len = args.ulb_dest_len

        self.smoothness = args.smoothness
        self.apm = torch.zeros((self.num_heads, self.ulb_dest_len, self.num_classes)).to(args.device)

        # apm_cutoff[head_id][class_id] = APM threshold for class_id when head_id is excluded
        self.apm_cutoff = torch.zeros((self.num_heads, self.num_classes)).to(args.device)

        # predictions for weakly augmented data for each head (head pseudolabels)
        self.weak_labels = torch.ones((self.num_heads, self.ulb_dest_len)).to(args.device) * -1

    def update(self, algorithm, logits_x_ulb_w, idx_ulb, head_id):
        epoch = algorithm.epoch
        
        max_probs, pseudo_labels = torch.topk(logits_x_ulb_w, 2)
        for i, idx in enumerate(idx_ulb.tolist()):
            pm = logits_x_ulb_w[i] - max_probs[i][0]
            pm[pseudo_labels[i][0]] = max_probs[i][0] - max_probs[i][1]

            self.apm[head_id][idx] = pm * self.smoothness / (1 + epoch) + self.apm[head_id][idx] * (1 - self.smoothness / (1 + epoch))

            self.weak_labels[head_id][idx] = pseudo_labels[i][0]
    
    def get_apm_label(self, algorithm, idx, label1, label2, head_id):
        # We don't have default apm_cutoff values for the first epoch
        if algorithm.epoch == 0:
            return -1
        
        apm_pass1 = self.apm[head_id][idx][label1] >= self.apm_cutoff[head_id][label1]
        apm_pass2 = self.apm[head_id][idx][label2] >= self.apm_cutoff[head_id][label2]

        # Both labels exceed the thresholds => We don't know what to choose, so ignore
        if apm_pass1 and apm_pass2:
            return -1
        
        if apm_pass1:
            return label1
        
        if apm_pass2:
            return label2
        
        # Neither of the 2 labels has passed the threshold
        return -1


    # @overrides
    def after_train_epoch(self, algorithm):
        '''
        Update apm_cutoff as 95th percentile for the 2 non-excluded heads
        '''

        for head_id, head_id1, head_id2 in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]:
            for label in range(self.num_classes):
                # Both heads need to choose the selected label
                mask = self.weak_labels[head_id1] == self.weak_labels[head_id2]
                mask = mask & (self.weak_labels[head_id1] == label)

                # Nothing was found => Use 0 as apm_cutoff
                if 1 not in mask:
                    self.apm_cutoff[head_id][label] = 0
                    break
                
                # Merge apm values for head1 and head2 for the current label
                apm_set1 = self.apm[head_id1][mask]
                apm_set2 = self.apm[head_id2][mask]
                apm_set = torch.cat(apm_set1[:, label], apm_set2[:, label])

                # Use 95th percentile as the new threshold
                self.apm_cutoff[head_id][label] = torch.quantile(apm_set.float(), 0.95)




                

