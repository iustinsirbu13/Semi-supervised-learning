import torch
from semilearn.core.hooks import Hook

class DebugHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def debug_network(self, network, epoch):
        if epoch >= 5:
            return
        
        for head_id in range(len(network.clf)):
            with open(f'debug/clf_{epoch}_{head_id}', 'w') as fd:
                fd.write(network.clf[head_id].weights)

    # @overrides
    def before_train_epoch(self, algorithm):
        self.debug_network(algorithm.model, algorithm.epoch)
