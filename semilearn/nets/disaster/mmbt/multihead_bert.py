import torch.nn as nn
from semilearn.nets.disaster.mmbt.bert import MultimodalBertEncoder

class MultiheadMultimodelBertClf(nn.Module):
    def __init__(self, args):
        super(MultiheadMultimodelBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        self.num_heads = args.num_heads
        self.clf = self.multihead_constructor(lambda: nn.Linear(args.hidden_sz, args.num_classes))

    def forward(self, txt, mask, segment, img):
        x = self.enc(txt, mask, segment, img)
        return [self.clf[head_id](x) for head_id in range(self.num_heads)]
    
    def multihead_constructor(self, constructor):
        return [constructor() for _ in range(self.num_heads)]
    
    # @overrides
    def modules(self):
        from itertools import chain
        return chain(super().modules(), self.clf)

def multihead_mmbt_bert(args):
    model = MultiheadMultimodelBertClf(args)
    return model
