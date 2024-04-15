import torch
from torchvision import transforms
from disaster_tweet.utils.arg_check import has_argument
from disaster_tweet.utils.randaugment import RandAugmentMC
from disaster_tweet.wrappers.ssl_base_wrapper import SSLBaseWrapper

class SSLDisasterWrapper(SSLBaseWrapper):

    PRECOMPUTED_MEANS = {
        'informative': (0.485, 0.456, 0.406),    # ImageNet
        'humanitarian': (0.485, 0.456, 0.406),   # ImageNet
    }

    PRECOMPUTED_STDS = {
        'informative': (0.229, 0.224, 0.225),    # ImageNet
        'humanitarian': (0.229, 0.224, 0.225),   # ImageNet
    }

    def __init__(self, config, build_algo):
        assert has_argument(config, 'data_dir')
        
        assert has_argument(config, 'dataset')
        assert config.dataset == 'disaster'

        assert has_argument(config, 'task')
        if config.task == 'humanitarian':
            config.labels = ['not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort', 'infrastructure_and_utility_damage', 'affected_individuals']
        elif config.task == 'informative':
            config.labels = ['informative', 'uninformative']
        elif config.task == 'damage':
            config.labels = ['damage', 'no damage']
        else:
            raise Exception(f'Task {config.task} is not supported')

        config.num_classes = len(config.labels)

        if has_argument(config, 'num_labels'):
            config.num_entries_per_class = config.num_labels // config.num_classes
        else:
            config.num_entries_per_class = 0

        if not has_argument(config, 'hard_label'):
            config.hard_label = True

        if not has_argument(config, 'p_cutoff'):
            config.p_cutoff = 0.95

        if not has_argument(config, 'uratio'):
            config.uratio = 2

        if not has_argument(config, 'ulb_loss_ratio'):
            config.ulb_loss_ratio = 1

        if not has_argument(config, 'T'):
            config.T = 0.5

        if config.gpu is None or config.gpu == 'None':
            config.gpu = None
            config.device = 'cpu'
        else:
            config.device = torch.device('cuda', config.gpu)

        if not has_argument(config, 'build_algo'):
            config.build_algo = True
        
        super().__init__(config, build_algo)


    def get_mean(self):
        if self.config.task in SSLDisasterWrapper.PRECOMPUTED_MEANS:
            return SSLDisasterWrapper.PRECOMPUTED_MEANS[self.config.task]
        return [0.5, 0.5, 0.5]

    def get_std(self):
        if self.config.task in SSLDisasterWrapper.PRECOMPUTED_STDS:
            return SSLDisasterWrapper.PRECOMPUTED_STDS[self.config.task]
        return [0.5, 0.5, 0.5]
    
        
    def get_weak_image_transform(self, img_size, crop_ratio):
        return transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=img_size, padding=int(img_size * (1.0 - crop_ratio)), padding_mode='reflect'),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.get_mean(), std=self.get_std())
                            ])
    
    def get_strong_image_transform(self, img_size, crop_ratio):
        return transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=img_size, padding=int(img_size * (1.0 - crop_ratio)), padding_mode='reflect'),
                                RandAugmentMC(n=2, m=10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.get_mean(), std=self.get_std())
                            ])

    def get_eval_image_transform(self, img_size):
        return transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.CenterCrop(size=img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.get_mean(), std=self.get_std())
                                ])


    def get_dataset(self, T, **kwargs):
        return T(**kwargs)
