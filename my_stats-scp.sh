#!/bin/bash

# for file in iustin.sirbu@fep.grid.pub.ro:/export/home/acs/prof/iustin.sirbu/SSL/USB/saved_models/usb_nlp/debug/*
# do
#     scp "${file:99:100}/my_stats.jsonl" "/mnt/d/GitHub/Semi-supervised-learning/preprocess/my_stats_1/${file:99:100}.jsonl"
# done


declare -a arr=(
    # "freematch_aclImdb_100_0"
    # "freematch_aclImdb_imb-100_0"
    # "freematch_aclImdb_imb100_0"
    # "freematch_ag_news_imb-100_0_real"
    # "freematch_ag_news_imb100_0"
    # "freematch_ag_news_imb100_0_real"
    # "marginmatch_aclImdb_imb-100_0"
    # "marginmatch_aclImdb_imb100_0"
    # "marginmatch_fl_aclImdb_100_0"
    # "marginmatch_fl_ag_news_200_0"
    # "marginmatch_fl_ag_news_imb-100_0"
    # "marginmatch_fl_ag_news_imb100_0"
    # "multihead_apm_plusFR_v4nl_dw3_lu3_aclImdb_100_0"
    # "multihead_apm_plusFR_v4nl_dw3_lu3_aclImdb_imb-100_0"
    # "multihead_apm_plusFR_v4nl_dw3_lu3_aclImdb_imb100_0"
    # "multihead_apm_plusFR_v4nl_dw3_lu3_ag_news_200_0"
    # "multihead_apm_plusFR_v4nl_dw3_lu3_ag_news_imb-100_0"
    # "multihead_apm_plusFR_v4nl_dw3_lu3_ag_news_imb100_0"
    # "multihead_cotraining_aclImdb_100_0"
    # "multihead_cotraining_aclImdb_imb-100_0"
    # "multihead_cotraining_aclImdb_imb100_0"
    # "multihead_cotraining_ag_news_200_0"
    # "multihead_cotraining_ag_news_imb-100_0"
    # "multihead_cotraining_ag_news_imb100_0"
    # "fixmatch_ag_news_imb100_0"
    # "fixmatch_ag_news_200_0"
    # "fixmatch_ag_news_200_01"
    "fixmatch_ag_news_imb100_01"
)
## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   # or do whatever with individual element of the array
   scp iustin.sirbu@fep.grid.pub.ro:/export/home/acs/prof/iustin.sirbu/SSL/USB/saved_models/usb_nlp/debug/$i/my_stats.jsonl /mnt/d/GitHub/Semi-supervised-learning/preprocess/my_stats_1/$i.jsonl
done