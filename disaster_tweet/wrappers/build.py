class ALGO_WRAPPERS:
    FIXMATCH = 'fixmatch'
    FIXMATCH_MMBT_BERT = 'fixmatch_mmbt_bert'
    FIXMATCH_MULTIHEAD = 'fixmatch_multihead'


def build_wrapper(dataset, algorithm, config, build_algo=True):
    if dataset not in ['disaster']:
        return None
    
    if algorithm == ALGO_WRAPPERS.FIXMATCH:
        from disaster_tweet.wrappers.fixmatch_img_wrapper import FixMatchImgWrapper
        return FixMatchImgWrapper(config, build_algo)
    
    elif algorithm == ALGO_WRAPPERS.FIXMATCH_MMBT_BERT:
        from disaster_tweet.wrappers.fixmatch_mmbt_bert_wrapper import FixMatchMMBTBertWrapper
        return FixMatchMMBTBertWrapper(config, build_algo)
    
    elif algorithm == ALGO_WRAPPERS.FIXMATCH_MULTIHEAD:
        from disaster_tweet.wrappers.fixmatch_multihead_wrapper import FixMatchMultiheadWrapper
        return FixMatchMultiheadWrapper(config, build_algo)
    
    raise Exception(f'Unknown disaster algorithm: {algorithm}')
    