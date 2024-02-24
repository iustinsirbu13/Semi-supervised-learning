from torchvision import transforms

from utils.arg_check import has_argument
from utils.randaugment import RandAugmentMC

from semilearn import get_data_loader

from wrappers.fixmatch_base_wrapper import FixMatchBaseWrapper

class FixMatchDisasterWrapper(FixMatchBaseWrapper):

    def __init__(self, config):
        assert has_argument(config, 'data_dir')
        config.dataset = 'CrisisMMD'

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
        
        super().__init__(config)
    
        
    def get_weak_transform(self, img_size, crop_ratio):
        return transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=img_size, padding=int(img_size * (1.0 - crop_ratio)), padding_mode='reflect'),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
    
    def get_strong_transform(self, img_size, crop_ratio):
        return transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(size=img_size, padding=int(img_size * (1.0 - crop_ratio)), padding_mode='reflect'),
                                RandAugmentMC(n=2, m=10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])

    def get_eval_transform(self, img_size):
        return transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.CenterCrop(size=img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

    # @overrides
    def prepare_dataloaders(self, config):
        self.train_loader = get_data_loader(config, self.train_dataset, config.batch_size)
        self.dev_loader = get_data_loader(config, self.dev_dataset, config.eval_batch_size)
        self.test_loader = get_data_loader(config, self.test_dataset, config.eval_batch_size)
        self.unlabeled_loader = get_data_loader(config, self.unlabeled_dataset, config.batch_size * config.uratio)