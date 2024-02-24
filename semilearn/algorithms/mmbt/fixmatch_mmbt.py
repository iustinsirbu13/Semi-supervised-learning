import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.fixmatch import FixMatch
from inspect import signature


@ALGORITHMS.register('fixmatch_mmbt')
class FixMatchMMBT(AlgorithmBase):
    def train_step():
        pass    
    
    