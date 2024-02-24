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


    def prepare_datasets(self, config):
        # weak and strong image transformation
        transform_weak = self.get_weak_transform(config.img_size, config.crop_ratio)
        transform_strong = self.get_strong_transform(config.img_size, config.crop_ratio)
        
        # transformation used for dev and test datasets
        eval_tranform = self.get_eval_transform(config.img_size)

        self.train_dataset = DisasterDatasetImage(
                                data=self.train_x,
                                targets=self.train_target,
                                num_classes=config.num_classes,
                                weak_transform=transform_weak,
                                img_dir=config.img_dir,
                                labels=config.labels,
                                img_size=config.img_size)
        
        self.dev_dataset = DisasterDatasetImage(
                                data=self.dev_x,
                                targets=self.dev_target,
                                num_classes=config.num_classes,
                                weak_transform=eval_tranform,
                                img_dir=config.img_dir,
                                labels=config.labels,
                                img_size=config.img_size)


        self.test_dataset = DisasterDatasetImage(
                                data=self.test_x,
                                targets=self.test_target,
                                num_classes=config.num_classes,
                                weak_transform=eval_tranform,
                                img_dir=config.img_dir,
                                labels=config.labels,
                                img_size=config.img_size)


        self.unlabeled_dataset = DisasterDatasetImage(
                                    data=self.unlabeled_x,
                                    targets=None,
                                    num_classes=config.num_classes,
                                    weak_transform=transform_weak,
                                    strong_transform=transform_strong,
                                    img_dir=config.img_dir,
                                    labels=config.labels,
                                    img_size=config.img_size)
