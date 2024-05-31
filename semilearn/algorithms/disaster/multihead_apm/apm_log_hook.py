import torch
from semilearn.core.hooks import Hook

class APMLogHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    # @overrides
    def after_train_epoch(self, algorithm):
        hook = algorithm.hooks_dict['APMHook']

        for head_id, head_id1, head_id2 in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]:
            # Percent of samples where both heads have reached an agreement
            agreement_mask = (hook.weak_labels[head_id1] == hook.weak_labels[head_id2])
            agreement_percent = agreement_mask.float().mean()
            algorithm.print_fn(f"HEAD[{head_id}]: Agreement percent={agreement_percent}")

            # Number of samples that were included per class using APM threshold (heads didn't reach agreement)
            apm_mask = (hook.apm_labels[head_id] != -1)
            apm_percent = apm_mask.float().mean()
            algorithm.print_fn(f"HEAD[{head_id}]: APM included samples per class={torch.bincount(hook.apm_labels[head_id][apm_mask])}")

            # Total unlabeled percent used
            algorithm.print_fn(f"HEAD[{head_id}]: Total unlabeled percent={agreement_percent + apm_percent}")

            # APM class thresholds for current head
            algorithm.print_fn(f"HEAD[{head_id}]: APM class thresholds={hook.apm_cutoff[head_id]}")


