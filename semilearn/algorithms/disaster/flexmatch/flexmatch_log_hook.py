
from semilearn.core.hooks import Hook

class FlexMatchLogHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    # @overrides
    def after_train_step(self, algorithm):
        if not super().every_n_iters(algorithm, algorithm.num_log_iter):
            return
        
        hook = algorithm.hooks_dict['MaskingHook']
        algorithm.print_fn(f"Pseudo label counter: {hook.pseudo_counter}")
        algorithm.print_fn(f"Classwise acc: f{hook.classwise_acc}")
        algorithm.print_fn(f"Class thresholds: f{algorithm.p_cutoff * (hook.classwise_acc / (2. - hook.classwise_acc))}")