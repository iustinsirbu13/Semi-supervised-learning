import sys
sys.path.append('../..')

import os, random, jsonlines

from semilearn import get_data_loader, get_net_builder, get_algorithm, Trainer


class FixMatchBaseWrapper:
    def __init__(self, config):
        self.config = config

        # read train, dev and test datasets
        self.train_x, self.train_target = self.read_train_dataset(config)
        self.dev_x, self.dev_target = self.read_dev_dataset(config)
        self.test_x, self.test_target = self.read_test_dataset(config)

        # read unlabeled dataset
        self.unlabeled_x = self.read_unlabeled_dataset(config)

        # prepare USB algorithm
        self.algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

        # prepare USB datasets and loaders
        self.prepare_datasets(config)
        self.prepare_dataloaders(config)
    
    
    def read_labeled_dataset(self, filename, config, num_entries_per_class=0):
        filepath = os.path.join(config.data_dir, filename)
        dataset = [row for row in jsonlines.open(filepath)]

        if num_entries_per_class == 0:
            x = dataset
        else:
            x = []
            for label in config.labels:
                dataset_by_label = [row for row in dataset if row['label'] == label]
                dataset_by_label = random.sample(dataset_by_label, num_entries_per_class)
                x += dataset_by_label

        random.shuffle(x)
        target = [config.labels.index(row['label']) for row in x]

        return x, target

    def read_train_dataset(self, config):
        return self.read_labeled_dataset(config.train_filename, config, num_entries_per_class=config.num_entries_per_class)

    def read_dev_dataset(self, config):
        return self.read_labeled_dataset(config.dev_filename, config)

    def read_test_dataset(self, config):
        return self.read_labeled_dataset(config.test_filename, config)

    def read_unlabeled_dataset(self, config):
        filepath = os.path.join(config.data_dir, config.unlabeled_filename)

        dataset = [row for row in jsonlines.open(filepath)]
        x = random.sample(dataset, len(self.train_x) * config.uratio)

        random.shuffle(x)

        return x
    
    def prepare_dataloaders(self, config):
        self.train_loader = get_data_loader(config, self.train_dataset, config.batch_size)
        self.dev_loader = get_data_loader(config, self.dev_dataset, config.eval_batch_size)
        self.test_loader = get_data_loader(config, self.test_dataset, config.eval_batch_size)
        self.unlabeled_loader = get_data_loader(config, self.unlabeled_dataset, config.batch_size * config.uratio)


    def train(self, T=Trainer):
        trainer = T(self.config, self.algorithm)
        trainer.fit(self.train_loader, self.unlabeled_loader, self.test_loader)
        return trainer

    def evaluate(self, trainer):
        trainer.evaluate(self.test_loader)

    def train_evaluate(self, T=Trainer):
        trainer = self.train(T)
        self.evaluate(trainer)

    ########################
    ##  Abstract Methods  ##
    ########################
    
    def prepare_datasets(self):
        raise NotImplementedError
