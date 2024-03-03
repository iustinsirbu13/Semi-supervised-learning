from disaster_tweet.wrappers.build import ALGO_WRAPPERS
from disaster_tweet.utils.arg_check import has_argument
from disaster_tweet.wrappers.fixmatch_mmbt_bert_wrapper import FixMatchMMBTBertWrapper

class FixMatchMultiheadMMBTBertWrapper(FixMatchMMBTBertWrapper):
    def __init__(self, config, build_algo=True):

        if not has_argument(config, 'num_heads'):
            config.num_heads = 3

        super().__init__(config, build_algo)

        # Multihead cotraining requires hard pseudo-labels
        assert config.hard_label == True

    # @overrides
    def validate_algo(self, config):
        assert config.algorithm == ALGO_WRAPPERS.FIXMATCH_MULTIHEAD_MMBT_BERT
        config.net = 'multihead_mmbt_bert'