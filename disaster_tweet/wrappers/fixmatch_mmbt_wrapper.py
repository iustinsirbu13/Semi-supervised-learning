from disaster_tweet.utils.arg_check import has_argument
from disaster_tweet.wrappers.fixmatch_disaster_wrapper import FixMatchDisasterWrapper

class FixMatchMMBTWrapper(FixMatchDisasterWrapper):

    def __init__(self, config, build_algo):
        assert has_argument(config, 'weak_text_tag')
        assert has_argument(config, 'strong_text_tag')

        self.tokenizer = self.get_tokenizer(config).tokenize
        self.vocab = self.get_vocab(config)
        
        config.vocab = self.vocab

        super().__init__(config, build_algo)


    # @overrides
    def prepare_datasets(self, config):
        # weak/strong/eval image transformations
        weak_image_transform = self.get_weak_image_transform(config.img_size, config.crop_ratio)
        strong_image_transform = self.get_strong_image_transform(config.img_size, config.crop_ratio)
        eval_image_tranform = self.get_eval_image_transform(config.img_size)

        self.train_dataset = self.get_dataset(self.train_x, self.train_target, weak_image_transform, config.weak_text_tag)
        self.dev_dataset = self.get_dataset(self.dev_x, self.dev_target, eval_image_tranform, 'text')
        self.test_dataset = self.get_dataset(self.test_x, self.test_target, eval_image_tranform, 'text')
        self.unlabeled_dataset = self.get_dataset(self.unlabeled_x, None, weak_image_transform, config.weak_text_tag,
                                        strong_image_transform=strong_image_transform, strong_text_tag=config.strong_text_tag)
        
    # @overrides
    def get_dataset(self, T, **kwargs):
        config = self.config
        return super().get_dataset(
                        T,
                        num_classes=config.num_classes,
                        img_dir=config.img_dir,
                        labels=config.labels,
                        img_size=config.img_size,
                        text_dir=config.data_dir,
                        max_seq_length=config.max_seq_length,
                        tokenizer=self.tokenizer,
                        vocab=self.vocab,
                        model=config.net,
                        num_image_embeds=config.num_image_embeds,
                        **kwargs)

    ########################
    ##  Abstract Methods  ##
    ########################
    
    def get_vocab(self):
        raise NotImplementedError
    
    def get_tokenizer(self):
        raise NotImplementedError
