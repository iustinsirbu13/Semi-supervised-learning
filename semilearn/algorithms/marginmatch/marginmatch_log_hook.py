import torch
from semilearn.core.hooks import Hook

class MarginMatchLogHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    # @overrides
    def after_train_epoch(self, algorithm):
        hook = algorithm.hooks_dict['PartialMarginHook']

        for label in range(algorithm.num_classes):
            algorithm.print_fn(f"Average APM class[{label}]: {torch.mean(hook.apm[hook.weak_labels == label].detach(), dim=0)}")
