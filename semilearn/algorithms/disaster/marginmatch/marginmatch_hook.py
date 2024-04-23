import torch
from semilearn.core.hooks import Hook

class MarginMatchHook(Hook):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.num_classes = args.num_classes
        self.ulb_dest_len = args.ulb_dest_len

        self.apm_cutoff = args.apm_cutoff
        self.smoothness = args.smoothness
        self.apm = torch.zeros((self.ulb_dest_len, self.num_classes)).to(args.device)

        # predicitions for weakly augmented data (used for monitoring class APM after each epoch)
        self.weak_labels = torch.ones(self.ulb_dest_len).to(args.device) * -1

    def update(self, algorithm, logits_x_ulb_w, logits_x_ulb_s, idx_ulb):
        epoch = algorithm.epoch
        
        max_probs, pseudo_labels = torch.topk(logits_x_ulb_w, 2)
        for i, idx in enumerate(idx_ulb.tolist()):
            pm = logits_x_ulb_w[i] - max_probs[i][0]
            pm[pseudo_labels[i][0]] = max_probs[i][0] - max_probs[i][1]

            self.apm[idx] = pm * self.smoothness / (1 + epoch) + self.apm[idx] * (1 - self.smoothness / (1 + epoch))

            self.weak_labels[idx] = pseudo_labels[i][0]

    def masking(self, algorithm, logits_x_ulb_w, logits_x_ulb_s, idx_ulb):
        self.update(algorithm, logits_x_ulb_w, logits_x_ulb_s, idx_ulb)

        if self.apm_cutoff is not None:
            max_probs, pseudo_labels = torch.max(logits_x_ulb_w, dim=-1)
            
            mask = torch.zeros(idx_ulb.shape[0]).to(self.args.device)

            for i, idx in enumerate(idx_ulb.tolist()):
                if self.apm[idx][pseudo_labels[i]] >= self.apm_cutoff:
                    mask[i] = self.apm[idx][pseudo_labels[i]] = 1

            return mask
        
        raise Exception("APM Threshold estimation is not supported")

