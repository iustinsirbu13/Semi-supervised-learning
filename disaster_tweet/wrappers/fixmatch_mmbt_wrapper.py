import sys
sys.path.append('../..')

from pytorch_pretrained_bert import BertTokenizer

from utils.arg_check import has_argument
from utils.vocab import Vocab

from semilearn.datasets.disaster_datasets.disaster_mmbt import DisasterDatasetMMBT

from wrappers.fixmatch_disaster_wrapper import FixMatchDisasterWrapper

class FixMatchMMBTWrapper(FixMatchDisasterWrapper):
    
    def __init__(self, config):
        config.algorithm = 'fixmatch_mmbt'
        config.net = 'mmbt_bert'
        config.img_size = 224

        assert has_argument(config, 'weak_text_tag')
        assert has_argument(config, 'strong_text_tag')

        FixMatchMMBTWrapper.fill_optional_config(config)

        self.tokenizer = self.get_tokenizer(config).tokenize
        self.vocab = self.get_vocab(config)

        super().__init__(config)


    def get_tokenizer(self, config):
        return BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    def get_vocab(self, config):
        bert_tokenizer = self.get_tokenizer(config)

        vocab = Vocab()
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

        return vocab

    # @overrides
    def prepare_datasets(self, config):
        # weak and strong image transformation
        weak_image_transform = self.get_weak_transform(config.img_size, config.crop_ratio)
        strong_image_transform = self.get_strong_transform(config.img_size, config.crop_ratio)
        
        # transformation used for dev and test datasets
        eval_image_tranform = self.get_eval_transform(config.img_size)

        self.train_dataset = DisasterDatasetMMBT(
                                data=self.train_x,
                                targets=self.train_target,
                                num_classes=config.num_classes,
                                weak_image_transform=weak_image_transform,
                                img_dir=config.img_dir,
                                labels=config.labels,
                                img_size=config.img_size,
                                weak_text_tag=config.weak_text_tag,
                                text_dir=config.data_dir,
                                max_seq_length=config.max_seq_length,
                                tokenizer=self.tokenizer,
                                vocab=self.vocab)
        
        
        self.dev_dataset = DisasterDatasetMMBT(
                                data=self.dev_x,
                                targets=self.dev_target,
                                num_classes=config.num_classes,
                                weak_image_transform=eval_image_tranform,
                                img_dir=config.img_dir,
                                labels=config.labels,
                                img_size=config.img_size,
                                weak_text_tag='text',
                                text_dir=config.data_dir,
                                max_seq_length=config.max_seq_length,
                                tokenizer=self.tokenizer,
                                vocab=self.vocab)


        self.test_dataset = DisasterDatasetMMBT(
                                data=self.test_x,
                                targets=self.test_target,
                                num_classes=config.num_classes,
                                weak_image_transform=eval_image_tranform,
                                img_dir=config.img_dir,
                                labels=config.labels,
                                img_size=config.img_size,
                                weak_text_tag='text',
                                text_dir=config.data_dir,
                                max_seq_length=config.max_seq_length,
                                tokenizer=self.tokenizer,
                                vocab=self.vocab)


        self.unlabeled_dataset = DisasterDatasetMMBT(
                                    data=self.unlabeled_x,
                                    targets=None,
                                    num_classes=config.num_classes,
                                    weak_image_transform=weak_image_transform,
                                    img_dir=config.img_dir,
                                    labels=config.labels,
                                    img_size=config.img_size,
                                    weak_text_tag=config.weak_text_tag,
                                    text_dir=config.data_dir,
                                    max_seq_length=config.max_seq_length,
                                    strong_image_transform=strong_image_transform,
                                    strong_text_tag=config.strong_text_tag,
                                    tokenizer=self.tokenizer,
                                    vocab=self.vocab)


    @staticmethod
    def fill_optional_config(config):
        if not has_argument(config, 'bert_model'):
            config.bert_model = 'bert-base-uncased'
        else:
            assert config.bert_model in ['bert-base-uncased', 'bert-large-uncased']
        
        if not has_argument(config, 'img_hidden_sz'):
            config.img_hidden_sz = 2048

        if not has_argument(config, 'hidden_sz'):
            config.hidden_sz = 768

        if not has_argument(config, 'drouput'):
            config.dropout = 0.2
        
        if not has_argument(config, 'num_image_embeds'):
            config.num_image_embeds = 3

        if not has_argument(config, 'img_embed_pool_type'):
            config.img_embed_pool_type = 'avg'
        else:
            assert config.img_embed_pool_type in ['avg', 'max']

        if not has_argument(config, 'max_seq_length'):
            config.max_seq_length = 512
