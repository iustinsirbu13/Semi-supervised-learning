import sys
sys.path.append('../..')

from disaster_tweet.wrappers.fixmatch_classic_wrapper import FixMatchClassicWrapper


class FixMatchImgWrapper(FixMatchClassicWrapper):

    def __init__(self, config, build_algo=True):
        assert config.algorithm == 'fixmatch'

        if config.net in ['vit_tiny_patch2_32', 'vit_small_patch2_32']:
            config.img_size = 32
        elif config.net in ['vit_small_patch16_224', 'vit_base_patch16_224']:
            config.img_size = 224
        else:
            raise Exception(f'Net {config.net} is not supported')

        super().__init__(config, build_algo)
