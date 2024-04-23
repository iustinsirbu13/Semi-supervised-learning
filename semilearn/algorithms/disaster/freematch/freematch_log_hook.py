import torch
from semilearn.core.hooks import Hook

class FreeMatchLogHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    # @overrides
    def after_train_step(self, algorithm):
        if not super().every_n_iters(algorithm, algorithm.num_log_iter):
            return
        
        hook = algorithm.hooks_dict['MaskingHook']
        algorithm.print_fn(f"Global class threshold: {hook.time_p}")
        algorithm.print_fn(f"Local class thresholds: {hook.p_model}")
        algorithm.print_fn(f"Class thresholds: {hook.time_p * hook.p_model / torch.max(hook.p_model, dim=-1)[0]}")
