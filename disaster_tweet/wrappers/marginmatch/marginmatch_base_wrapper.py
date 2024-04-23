from semilearn.datasets.disaster_datasets.disaster_img import DisasterDatasetImage
from disaster_tweet.wrappers.ssl_disaster_wrapper import SSLDisasterWrapper
from disaster_tweet.utils.arg_check import has_argument

class MarginMatchBaseWrapper(SSLDisasterWrapper):

    def __init__(self, config, build_algo=True):
        # Not sure what value to use (0 is definetely correct, but might be too small)
        if not has_argument(config, 'apm_cutoff'):
            config.apm_cutoff = 0

        if not has_argument(config, 'threshold_algo'):
            config.threshold_algo = 'freematch'
        assert config.threshold_algo in ['freematch', 'flexmatch']

        if config.threshold_algo == 'freematch':
            if not has_argument(config, 'ema_p'):
                config.ema_p = 0.999

        # Value from paper (https://arxiv.org/pdf/2308.09037.pdf)
        if not has_argument(config, 'smoothness'):
            config.smoothness = 0.997

        super().__init__(config, build_algo)

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
