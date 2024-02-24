import sys
sys.path.append('../..')

from utils.arg_check import has_argument

from semilearn.datasets.disaster_datasets.disaster_img import DisasterDatasetImage

from wrappers.fixmatch_disaster_wrapper import FixMatchDisasterWrapper


class FixMatchImgWrapper(FixMatchDisasterWrapper):

    def __init__(self, config):
        config.algorithm = 'fixmatch'

        assert has_argument(config, 'net')
        if config.net in ['vit_tiny_patch2_32', 'vit_small_patch2_32']:
            config.img_size = 32
        elif config.net in ['vit_small_patch16_224', 'vit_base_patch16_224']:
            config.img_size = 224
        else:
            raise Exception(f'Net {config.net} is not supported')

        super().__init__(config)

    # @overrides
    def get_dataset(self, x, targets, weak_image_transform, strong_image_transform=None):
        config = self.config
        return super().get_dataset(
                        DisasterDatasetImage, # T
                        data=x,
                        targets=targets,
                        num_classes=config.num_classes,
                        weak_transform=weak_image_transform,
                        img_dir=config.img_dir,
                        labels=config.labels,
                        img_size=config.img_size,
                        strong_transform=strong_image_transform)


    # @overrides
    def prepare_datasets(self, config):
        # weak/strong/eval image transformations
        weak_image_transform = self.get_weak_image_transform(config.img_size, config.crop_ratio)
        strong_image_transform = self.get_strong_image_transform(config.img_size, config.crop_ratio)
        eval_image_tranform = self.get_eval_image_transform(config.img_size)

        self.train_dataset = self.get_dataset(self.train_x, self.train_target, weak_image_transform)
        self.dev_dataset = self.get_dataset(self.dev_x, self.dev_target, eval_image_tranform)
        self.test_dataset = self.get_dataset(self.test_x, self.test_target, eval_image_tranform)
        self.unlabeled_dataset = self.get_dataset(self.unlabeled_x, None, weak_image_transform,
                                                    strong_image_transform=strong_image_transform)
