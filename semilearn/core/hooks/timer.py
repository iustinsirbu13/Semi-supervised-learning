# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .hook import Hook


class TimerHook(Hook):
    """
    Timer Hook
    """
    
    def before_run(self, algorithm):
        # Remove CUDA!!!
        # algorithm.start_batch = torch.cuda.Event(enable_timing=True)
        # algorithm.end_batch = torch.cuda.Event(enable_timing=True)

        # algorithm.start_run = torch.cuda.Event(enable_timing=True)
        # algorithm.end_run = torch.cuda.Event(enable_timing=True)
        # algorithm.start_batch.record()
        pass
    
    def before_train_step(self, algorithm):
        # algorithm.end_batch.record()
        pass
    
    def after_train_step(self, algorithm):
        # algorithm.log_dict['lr'] = algorithm.optimizer.param_groups[-1]['lr']
        # algorithm.log_dict['train/prefetch_time'] = algorithm.start_batch.elapsed_time(algorithm.end_batch) / 1000.
        # algorithm.start_batch.record()
        pass