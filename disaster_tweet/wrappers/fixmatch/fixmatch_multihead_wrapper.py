from disaster_tweet.wrappers.build import ALGO_WRAPPERS
from disaster_tweet.utils.arg_check import has_argument
from disaster_tweet.wrappers.fixmatch.fixmatch_base_wrapper import FixMatchBaseWrapper


class FixMatchMultiheadWrapper(FixMatchBaseWrapper):

    def __init__(self, config, build_algo=True):
        assert config.algorithm == ALGO_WRAPPERS.FIXMATCH_MULTIHEAD
        assert config.net in ['wrn_multihead_28_2', 'wrn_multihead_28_8']
        
        if not has_argument(config, 'num_heads'):
            config.num_heads = 3
        
        if not has_argument(config, 'img_size'):
            config.img_size = 32

        if not has_argument(config, 'use_head_cutoff'):
            config.use_head_cutoff = False

        super().__init__(config, build_algo)

        # Multihead cotraining requires hard pseudo-labels
        assert config.hard_label == True
        
        # wrn requires 32x32 images
        assert config.img_size == 32
