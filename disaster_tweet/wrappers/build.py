class ALGO_WRAPPERS:
    FIXMATCH = 'fixmatch'
    FIXMATCH_MMBT_BERT = 'fixmatch_mmbt_bert'
    FIXMATCH_MULTIHEAD = 'fixmatch_multihead'
    FIXMATCH_MULTIHEAD_MMBT_BERT = 'fixmatch_multihead_mmbt_bert'

    FLEXMATCH = 'flexmatch'
    FLEXMATCH_MMBT_BERT = 'flexmatch_mmbt_bert'

    FREEMATCH = 'freematch'
    FREEMATCH_MMBT_BERT = 'freematch_mmbt_bert'

    MARGINMATCH = 'marginmatch'
    MARGINMATCH_MMBT_BERT = 'marginmatch_mmbt_bert'


def build_wrapper(dataset, algorithm, config, build_algo=True):
    if dataset not in ['disaster']:
        return None
    
    #####################
    ##  Fixmatch Start ##
    #####################

    if algorithm == ALGO_WRAPPERS.FIXMATCH:
        from disaster_tweet.wrappers.fixmatch.fixmatch_wrapper import FixMatchWrapper
        return FixMatchWrapper(config, build_algo)
    
    elif algorithm == ALGO_WRAPPERS.FIXMATCH_MMBT_BERT:
        from disaster_tweet.wrappers.fixmatch.fixmatch_mmbt_bert_wrapper import FixMatchMMBTBertWrapper
        return FixMatchMMBTBertWrapper(config, build_algo)
    
    elif algorithm == ALGO_WRAPPERS.FIXMATCH_MULTIHEAD:
        from disaster_tweet.wrappers.fixmatch.fixmatch_multihead_wrapper import FixMatchMultiheadWrapper
        return FixMatchMultiheadWrapper(config, build_algo)
    
    elif algorithm == ALGO_WRAPPERS.FIXMATCH_MULTIHEAD_MMBT_BERT:
        from disaster_tweet.wrappers.fixmatch.fixmatch_multihead_mmbt_bert_wrapper import FixMatchMultiheadMMBTBertWrapper
        return FixMatchMultiheadMMBTBertWrapper(config, build_algo)
    

    ######################
    ##  Flexmatch Start ##
    ######################

    elif algorithm == ALGO_WRAPPERS.FLEXMATCH:
        from disaster_tweet.wrappers.flexmatch.flexmatch_wrapper import FlexMatchWrapper
        return FlexMatchWrapper(config, build_algo)
    
    elif algorithm == ALGO_WRAPPERS.FLEXMATCH_MMBT_BERT:
        from disaster_tweet.wrappers.flexmatch.flexmatch_mmbt_bert_wrapper import FlexMatchMMBTBertWrapper
        return FlexMatchMMBTBertWrapper(config, build_algo)

    
    raise Exception(f'Unknown disaster algorithm: {algorithm}')
    
