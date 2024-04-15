from pytorch_pretrained_bert import BertTokenizer
from disaster_tweet.wrappers.build import ALGO_WRAPPERS
from disaster_tweet.utils.arg_check import has_argument
from disaster_tweet.utils.vocab import Vocab
from semilearn.datasets.disaster_datasets.disaster_mmbt import DisasterDatasetMMBT
from disaster_tweet.wrappers.fixmatch.fixmatch_mmbt_base_wrapper import FixMatchMMBTBaseWrapper


class FixMatchMMBTBertWrapper(FixMatchMMBTBaseWrapper):
    
    def __init__(self, config, build_algo=True):
        self.validate_algo(config)

        if not has_argument(config, 'img_size'):
            config.img_size = 224
        
        if not has_argument(config, 'bert_model'):
            config.bert_model = 'bert-base-uncased'
        else:
            assert config.bert_model in ['bert-base-uncased', 'bert-large-uncased']
        
        if not has_argument(config, 'img_hidden_sz'):
            config.img_hidden_sz = 2048

        if not has_argument(config, 'hidden_sz'):
            config.hidden_sz = 768

        if not has_argument(config, 'dropout'):
            config.dropout = 0.2
        
        if not has_argument(config, 'num_image_embeds'):
            config.num_image_embeds = 3

        if not has_argument(config, 'img_embed_pool_type'):
            config.img_embed_pool_type = 'avg'
        else:
            assert config.img_embed_pool_type in ['avg', 'max']

        if not has_argument(config, 'max_seq_length'):
            config.max_seq_length = 512

        super().__init__(config, build_algo)


    # @overrides
    def get_tokenizer(self, config):
        return BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # @overrides
    def get_vocab(self, config):
        bert_tokenizer = self.get_tokenizer(config)

        vocab = Vocab()
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

        return vocab
    
    # @overrides
    def get_dataset(self, x, targets, weak_image_transform, weak_text_tag,
                    strong_image_transform=None, strong_text_tag=None):
        return super().get_dataset(
                        DisasterDatasetMMBT,
                        data=x,
                        targets=targets,
                        weak_image_transform=weak_image_transform,
                        weak_text_tag=weak_text_tag,
                        strong_image_transform=strong_image_transform,
                        strong_text_tag=strong_text_tag)


    def validate_algo(self, config):
        assert config.algorithm == ALGO_WRAPPERS.FIXMATCH_MMBT_BERT
        config.net = 'mmbt_bert'
