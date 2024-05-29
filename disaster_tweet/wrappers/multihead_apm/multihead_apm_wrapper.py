from disaster_tweet.wrappers.build import ALGO_WRAPPERS
from disaster_tweet.utils.arg_check import has_argument
from disaster_tweet.wrappers.multihead_apm.multihead_apm_base_wrapper import MultiheadAPMBaseWrapper


class MultiheadAPMWrapper(MultiheadAPMBaseWrapper):

    def __init__(self, config, build_algo=True):
        assert config.algorithm == ALGO_WRAPPERS.MULTIHEAD_APM
        assert config.net in ['wrn_multihead_28_2', 'wrn_multihead_28_8']
        
        if not has_argument(config, 'num_heads'):
            config.num_heads = 3
        
        if not has_argument(config, 'img_size'):
            config.img_size = 32

        super().__init__(config, build_algo)

        # Multihead cotraining requires hard pseudo-labels
        assert config.hard_label == True

        # We only support 3 heads
        assert config.num_heads == 3
        
        # wrn requires 32x32 images
        assert config.img_size == 32