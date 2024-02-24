import sys
sys.path.append('../..')

from wrappers.fixmatch_disaster_wrapper import FixMatchDisasterWrapper


class FixMatchImgWrapper(FixMatchDisasterWrapper):

    def __init__(self, config):
        config.algorithm = 'fixmatch'

        assert config.net is not None
        if config.net in ['vit_tiny_patch2_32', 'vit_small_patch2_32']:
            config.img_size = 32
        elif config.net in ['vit_small_patch16_224', 'vit_base_patch16_224']:
            config.img_size = 224

        super().__init__(config)
