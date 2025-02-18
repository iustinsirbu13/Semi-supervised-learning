# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Create the .yaml for each experiment
"""
import os

MAX_PRECISION = 6
def format_floats(v):
    s = f'{v:.{MAX_PRECISION}f}'.rstrip('0')
    if s.endswith('.'):
        s = f'{s}0'
    return s

def create_configuration(cfg, cfg_file, imbalance, save_suffix=''):
    cfg["save_name"] = "{alg}_{dataset}_{num_lb}_{seed}".format(
        alg=cfg["algorithm"],
        dataset=cfg["dataset"],
        num_lb=cfg["num_labels"] if not imbalance else f'imb{imbalance}',
        seed=cfg["seed"],
    )

    ###
    # if cfg["algorithm"].startswith('multihead_apm') or cfg["algorithm"].startswith('marginmatch') or cfg['algorithm'].startswith('multihead_cotraining'):
        # cfg["algorithm"] = 'multihead_apm'
    cfg["save_dir"] = f"./saved_models/usb_nlp/{cfg['algorithm']}" if not imbalance else f"./saved_models/usb_nlp/{cfg['algorithm']}_imb"

    # resume
    cfg["resume"] = True
    cfg["load_path"] = "{}/{}/latest_model.pth".format(
        cfg["save_dir"], cfg["save_name"]
    )

    if imbalance:
        alg_file = cfg_file + cfg["algorithm"] + f"_imb{save_suffix}/"
    else:
        alg_file = cfg_file + cfg["algorithm"] + f"{save_suffix}/"
    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    ###
    if cfg['algorithm'].startswith('multihead_apm_plus'):
        cfg["algorithm"] = 'multihead_apm_plus'
    elif cfg["algorithm"].startswith('multihead_apm'):
        cfg["algorithm"] = 'multihead_apm'

    elif cfg["algorithm"].startswith('marginmatch'):
        cfg["algorithm"] = 'marginmatch'
    elif cfg['algorithm'].startswith('multihead_cotraining'):
        cfg['algorithm'] = 'fixmatch_multihead'
    elif cfg['algorithm'] == 'fixmatch_abc':
        cfg['algorithm'] = 'fixmatch'
    elif cfg['algorithm'] == 'freematch_abc':
        cfg['algorithm'] = 'freematch'
    else:
        raise NotImplementedError()

    print(alg_file + cfg["save_name"] + ".yaml")
    with open(alg_file + cfg["save_name"] + ".yaml", "w", encoding="utf-8") as w:
        lines = []
        for k, v in cfg.items():
            if isinstance(v, float):
                line = str(k) + ": " +  format_floats(v)
                # print(line)
            else:
                line = str(k) + ": " + str(v)
            # if k == 'lr':
            #     print(k, v, type(v), line)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write("\n")


def create_usb_nlp_config(
    alg,
    seed,
    dataset,
    net,
    num_classes,
    num_labels,
    port,
    lr,
    weight_decay,
    layer_decay,
    max_length,
    warmup_epoch=5,
    amp=False,
    imbalance=None,
):
    cfg = {}
    cfg["algorithm"] = alg

    # save config
    cfg["save_dir"] = "./saved_models/usb_nlp"

    # if alg.startswith('multihead_apm'):
        # cfg["algorithm"] = 'multihead_apm'
        # cfg["save_dir"] = "./saved_models/usb_nlp/multihead_apm"

    cfg["save_name"] = None
    cfg["resume"] = True
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = True
    cfg["use_wandb"] = False
    # cfg["use_aim"] = False

    # algorithm config
    cfg["epoch"] = 100
    cfg["num_train_iter"] = 1024 * 100
    cfg["num_warmup_iter"] = int(1024 * warmup_epoch)
    cfg["num_log_iter"] = 256
    cfg["num_eval_iter"] = 2048
    if imbalance is None:
        cfg["num_labels"] = num_labels
    else:
        assert imbalance in [-100, 100, 4, 5, '1k02']
        if imbalance == 100:
            cfg["num_labels"] = 1000
            cfg['lb_imb_ratio'] = 100
            cfg['ulb_imb_ratio'] = 100
            cfg['ulb_num_labels'] = 10000
        elif imbalance == -100:
            cfg["num_labels"] = 1000
            cfg['lb_imb_ratio'] = 100
            cfg['ulb_imb_ratio'] = -100
            cfg['ulb_num_labels'] = 10000
        elif imbalance == '1k02':
            cfg["num_labels"] = 1000
            cfg['lb_imb_ratio'] = 100
            cfg['ulb_imb_ratio'] = 100
            cfg['ulb_num_labels'] = 4000
        else:
            cfg["num_labels"] = num_labels // num_classes
            cfg['lb_imb_ratio'] = imbalance
            cfg['ulb_imb_ratio'] = imbalance
            cfg['ulb_num_labels'] = cfg["num_labels"] * 100

    cfg["batch_size"] = 8
    cfg["eval_batch_size"] = 8
    cfg["ema_m"] = 0.0

    if alg.startswith("fixmatch"):
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        if alg == 'fixmatch_abc':
            cfg['imb_algorithm'] = 'abc'
            cfg['abc_p_cutoff'] = 0.95
            cfg['abc_loss_ratio'] = 1.0

    elif alg == "adamatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["ema_p"] = 0.999
    elif alg == "flexmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["thresh_warmup"] = True
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
    elif alg == "uda":
        cfg["tsa_schedule"] = "exp"
        cfg["T"] = 0.4
        cfg["p_cutoff"] = 0.8
        cfg["ulb_loss_ratio"] = 1.0
    elif alg == "pseudolabel":
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["unsup_warm_up"] = 0.4
    elif alg == "mixmatch":
        cfg["mixup_alpha"] = 0.5
        cfg["T"] = 0.5
        cfg["ulb_loss_ratio"] = 100
        cfg["unsup_warm_up"] = 0.4
        cfg["mixup_manifold"] = True
    elif alg == "remixmatch":
        cfg["mixup_alpha"] = 0.75
        cfg["T"] = 0.5
        cfg["kl_loss_ratio"] = 0.5
        cfg["ulb_loss_ratio"] = 1.5
        cfg["rot_loss_ratio"] = 0.0
        cfg["unsup_warm_up"] = 1 / 64
        cfg["mixup_manifold"] = True
    elif alg == "crmatch":
        cfg["hard_label"] = True
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["rot_loss_ratio"] = 0.0
    elif alg == "comatch":
        cfg["hard_label"] = False
        cfg["p_cutoff"] = 0.95
        cfg["contrast_p_cutoff"] = 0.8
        cfg["contrast_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 64
        cfg["queue_batch"] = 128
        cfg["smoothing_alpha"] = 0.9
        cfg["T"] = 0.2
        cfg["da_len"] = 32
    elif alg == "simmatch":
        cfg["p_cutoff"] = 0.95
        cfg["in_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 128
        cfg["K"] = 256
        cfg["da_len"] = 32
        cfg["smoothing_alpha"] = 0.9
        cfg["T"] = 0.1
        cfg["ema_m"] = 0.0
    elif alg == "meanteacher":
        cfg["ulb_loss_ratio"] = 50
        cfg["unsup_warm_up"] = 0.4
    elif alg == "pimodel":
        cfg["ulb_loss_ratio"] = 10
        cfg["unsup_warm_up"] = 0.4
        cfg["ema_m"] = 0.999
    elif alg == "dash":
        cfg["gamma"] = 1.27
        cfg["C"] = 1.0001
        cfg["rho_min"] = 0.05
        cfg["num_wu_iter"] = 2048
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
    elif alg == "mpl":
        cfg["tsa_schedule"] = "exp"
        cfg["T"] = 0.7
        cfg["p_cutoff"] = 0.6
        cfg["ulb_loss_ratio"] = 8.0
        cfg["teacher_lr"] = 0.03
        cfg["label_smoothing"] = 0.1
        cfg["num_uda_warmup_iter"] = 5000
        cfg["num_stu_wait_iter"] = 3000
    elif alg == "vat":
        cfg["vat_embed"] = True
    elif alg.startswith("freematch"):
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["ema_p"] = 0.999
        cfg["ent_loss_ratio"] = 0.001
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 1.0
        if alg == 'freematch_abc':
            cfg['imb_algorithm'] = 'abc'
            cfg['abc_p_cutoff'] = 0.95
            cfg['abc_loss_ratio'] = 1.0

    elif alg == "softmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["dist_align"] = True
        cfg["dist_uniform"] = True
        cfg["per_class"] = False
        cfg["ema_p"] = 0.999
        cfg["ulb_loss_ratio"] = 1.0
        cfg["n_sigma"] = 2
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 1.0
    elif alg == "defixmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 0.5
    elif alg.startswith('marginmatch'):
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg['threshold_algo'] = 'flexmatch'
        cfg['apm_cutoff'] = 0.0
        cfg['smoothness'] = 0.997
        if alg == 'marginmatch_abc':
            cfg['imb_algorithm'] = 'abc'
            cfg['abc_p_cutoff'] = 0.95
            cfg['abc_loss_ratio'] = 1.0
        elif alg == 'marginmatch':
            pass
        else:
            raise NotImplementedError(f'{alg}')
    elif alg.startswith('multihead_cotraining'):
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg['use_head_cutoff'] = False
        cfg['num_heads'] = 3
        cfg['adjust_clf_size'] = False
        if alg == 'multihead_cotraining_abc':
            cfg['imb_algorithm'] = 'abc'
            cfg['abc_p_cutoff'] = 0.95
            cfg['abc_loss_ratio'] = 1.0


    elif alg.startswith('multihead_apm'):
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg['num_heads'] = 3
        cfg['use_agreement_apm'] = True
        cfg['apm_percentile'] = 0.05
        if alg == 'multihead_apm':
            pass
        elif alg == 'multihead_apm_nolow':
            cfg['no_low'] = True
        elif alg == 'multihead_apm_dw9':
            cfg['apm_disagreement_weight'] = 0.9
        elif alg == 'multihead_apm_dw9_lu10':
            cfg['apm_disagreement_weight'] = 0.9
            cfg["ulb_loss_ratio"] = 10.0
        elif alg == 'multihead_apm_p10':
            cfg['apm_percentile'] = 0.1
        elif alg == 'multihead_apm2_nolow_dw9_lu10_p10':
            cfg['no_low'] = True
            cfg['apm_disagreement_weight'] = 0.9
            cfg["ulb_loss_ratio"] = 10.0
            cfg['apm_percentile'] = 0.1
        elif alg == 'multihead_apm3_light_nolow_dw5_lu2_p20':
            cfg['no_low'] = True
            cfg['apm_disagreement_weight'] = 0.5
            cfg["ulb_loss_ratio"] = 2.0
            cfg['apm_percentile'] = 0.2
        elif alg == 'multihead_apm4_light_nolow_dw5_lu2_p5':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_disagreement_weight'] = 5
            cfg["ulb_loss_ratio"] = 2.0
            cfg['apm_percentile'] = 0.05
        elif alg == 'multihead_apm5_light_nolow_dw5_lu1_p5':
            cfg['multihead_apm_variant'] = 'v5'
            cfg['no_low'] = True
            cfg['apm_disagreement_weight'] = 5
            cfg["ulb_loss_ratio"] = 1.0
            cfg['apm_percentile'] = 0.05
        elif alg == 'multihead_apm6_light_dw5_lu1':
            cfg['multihead_apm_variant'] = 'v6'
            cfg['apm_disagreement_weight'] = 5
            cfg["ulb_loss_ratio"] = 1.0
        elif alg == 'multihead_apm_nolow_abc':
            cfg['no_low'] = True
            cfg['imb_algorithm'] = 'abc'
            cfg['abc_p_cutoff'] = 0.95
            cfg['abc_loss_ratio'] = 1.0
        elif alg == 'multihead_apm6_light_dw5_lu1_abc':
            cfg['multihead_apm_variant'] = 'v6'
            cfg['apm_disagreement_weight'] = 5
            cfg["ulb_loss_ratio"] = 1.0
            cfg['imb_algorithm'] = 'abc'
            cfg['abc_p_cutoff'] = 0.95
            cfg['abc_loss_ratio'] = 1.0
        elif alg == 'multihead_apm_plusFR_v8_dw5':
            cfg['multihead_apm_variant'] = 'v8'
            cfg['apm_disagreement_weight'] = 5
            cfg["ulb_loss_ratio"] = 1.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusFR_v8_dw3_lu3':
            cfg['multihead_apm_variant'] = 'v8'
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusFR_v6_dw5':
            cfg['multihead_apm_variant'] = 'v6'
            cfg['apm_disagreement_weight'] = 5
            cfg["ulb_loss_ratio"] = 1.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusFR_v4nl_dw5':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 5
            cfg["ulb_loss_ratio"] = 1.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997

        # main
        elif alg == 'multihead_apm_plusFR_v4nl_dw3_lu3':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusFR_v4nl_dw3_lu3_abc':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
            cfg['imb_algorithm'] = 'abc'
            cfg['abc_p_cutoff'] = 0.95
            cfg['abc_loss_ratio'] = 1.0
        elif alg == 'multihead_apm_plusFR_v4nl_dw1_lu3':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 1
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusFR_v4nl_dw0_lu3':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 0
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusN_v4nl_dw3_lu3':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'none'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusFR_v-apm_dw3_lu3':
            cfg['multihead_apm_variant'] = 'v-apm'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
        elif alg == 'multihead_apm_plusFR_v4-cutoff_dw3_lu3':
            cfg['multihead_apm_variant'] = 'v4-cutoff'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997

        elif alg == 'multihead_apm_plusFR_v4nl_dw3_avg':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 1.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
            cfg['average_losses'] = True
        elif alg == 'multihead_apm_plusFR_v4nl_dw10_avg':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 10
            cfg["ulb_loss_ratio"] = 1.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997
            cfg['average_losses'] = True
        elif alg == 'multihead_apm_plusFR_v4l_dw3_lu3':
            cfg['multihead_apm_variant'] = 'v4'
            cfg['no_low'] = False
            cfg['apm_percentile'] = 0.05
            cfg['apm_disagreement_weight'] = 3
            cfg["ulb_loss_ratio"] = 3.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997

        elif alg == 'multihead_apm_plusFR_v1nolow':
            cfg['multihead_apm_variant'] = 'original'
            cfg['no_low'] = True
            cfg['apm_percentile'] = 0.05
            cfg["ulb_loss_ratio"] = 1.0
            cfg['threshold_algo'] = 'freematch'
            cfg['apm_cutoff'] =  0.0
            cfg['smoothness'] = 0.997

        else:
            raise NotImplementedError(f'{alg}')
    else:
        raise NotImplementedError(f'{alg}')

    cfg["uratio"] = 1
    cfg["use_cat"] = False

    # optim config
    cfg["optim"] = "AdamW"
    cfg["lr"] = lr
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["layer_decay"] = layer_decay
    cfg["amp"] = amp
    cfg["clip"] = 0.0

    # net config
    cfg["net"] = net
    cfg["net_from_name"] = False

    # data config
    cfg["data_dir"] = "./data"
    cfg["dataset"] = dataset
    cfg["train_sampler"] = "RandomSampler"
    cfg["num_classes"] = num_classes
    cfg["num_workers"] = 4
    cfg["max_length"] = max_length

    # basic config
    cfg["seed"] = seed

    # distributed config
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = False
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"
    cfg["gpu"] = 0

    # other config
    cfg["overwrite"] = True

    return cfg


def exp_usb_nlp(label_amount, imbalances=None):
    config_file = r'./disaster_tweet/config/cluster/usb_nlp/'
    save_path = r"./saved_models/usb_nlp"

    if not os.path.exists(config_file):
        os.mkdir(config_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    algs = [
        # "flexmatch",
        # "fixmatch",
        # "uda",
        # "pseudolabel",
        # "fullysupervised",
        # "supervised",
        # "remixmatch",
        # "mixmatch",
        # "meanteacher",
        # "pimodel",
        # "vat",
        # "dash",
        # "comatch",
        # "crmatch",
        # "simmatch",
        # "adamatch",
        # "softmatch",
        # "freematch",
        # "defixmatch",

        # "multihead_apm",
        # "multihead_apm_nolow",
        # "multihead_apm_dw9",
        # "multihead_apm_dw9_lu10",
        # "multihead_apm_p10",
        # "multihead_apm2_nolow_dw9_lu10_p10",
        # "multihead_apm3_light_nolow_dw5_lu2_p20",
        # "marginmatch",
        # "multihead_apm4_light_nolow_dw5_lu2_p5",
        # "multihead_apm5_light_nolow_dw5_lu1_p5",
        # "multihead_apm6_light_dw5_lu1",

        # "marginmatch_abc",
        # "multihead_apm_nolow_abc",
        # "multihead_apm6_light_dw5_lu1_abc",

        # "multihead_apm_plusFR_v8_dw5",
        # "multihead_apm_plusFR_v6_dw5",
        # "multihead_apm_plusFR_v4nl_dw5",
        # "multihead_apm_plusFR_v4nl_dw3_lu3_abc",
        # "multihead_apm_plusFR_v4nl_dw3_lu3",
        # "multihead_apm_plusFR_v4nl_dw1_lu3",
        "multihead_apm_plusFR_v4nl_dw0_lu3",
        # "multihead_apm_plusN_v4nl_dw3_lu3",
        # "multihead_apm_plusFR_v-apm_dw3_lu3",
        # "multihead_apm_plusFR_v4-cutoff_dw3_lu3",

        # "multihead_apm_plusFR_v4nl_dw3_avg",
        # "multihead_apm_plusFR_v4nl_dw10_avg",

        # "multihead_apm_plusFR_v8_dw3_lu3",
        # "multihead_apm_plusFR_v1nolow",
        # "multihead_cotraining",
        # "multihead_cotraining_abc",
        # "fixmatch_abc",
        # "freematch_abc",
        # "multihead_apm_plusFR_v4l_dw3_lu3",
    ]
    datasets = [
        "aclImdb",
        "ag_news",
        "amazon_review",
        # "dbpedia",
        "yahoo_answers",
        "yelp_review",
    ]

    # datasets = [
    #     'wildguardmix_HT_subcat',
    #     'wildguardmix_HT_cat',
    #     'wildguardmix_HT_topcat',
    # ]

    seeds = [0, 1, 2]
    # seeds = [0]

    dist_port = range(10001, 31120, 1)
    count = 0
    net = "bert_base_uncased"
    # net = "bert_base_uncased_multihead"
    # net = "bert_base_uncased_multihead_light"

    # weight_decay = 5e-4
    weight_decay = 0.0005
    max_length = 512

    for alg in algs:
        if alg.startswith('multihead'):
            net = "bert_base_uncased_multihead"
        print(f'Using net {net} for alg {alg} !!!')

        for dataset in datasets:
            for seed in seeds:
                # change the configuration of each dataset
                if dataset == "aclImdb":
                    num_classes = 2
                    num_labels = label_amount[0] * num_classes

                    # lr = 5e-5
                    lr = 0.00005
                    layer_decay = 0.75

                elif dataset == "ag_news":
                    num_classes = 4
                    num_labels = label_amount[1] * num_classes

                    # lr = 5e-5
                    lr = 0.00005
                    layer_decay = 0.65

                elif dataset == "amazon_review":
                    num_classes = 5
                    num_labels = label_amount[2] * num_classes

                    # lr = 1e-5
                    lr = 0.00001
                    layer_decay = 0.75

                elif dataset == "dbpedia":
                    num_classes = 14
                    num_labels = label_amount[3] * num_classes
                elif dataset == "yahoo_answers":
                    num_classes = 10
                    num_labels = label_amount[4] * num_classes

                    # lr = 1e-4
                    lr = 0.0001
                    layer_decay = 0.65

                elif dataset == "yelp_review":
                    num_classes = 5
                    num_labels = label_amount[5] * num_classes

                    # lr = 5e-5
                    lr = 0.00005
                    layer_decay = 0.75

                elif dataset == 'wildguardmix_HT_subcat':
                    num_classes = 15
                    num_labels = 6940
                    lr = 0.00005
                    layer_decay = 0.75
                elif dataset == 'wildguardmix_HT_topcat':
                    num_classes = 2
                    num_labels = 6940
                    lr = 0.00005
                    layer_decay = 0.75
                elif dataset == 'wildguardmix_HT_cat':
                    num_classes = 6
                    num_labels = 6940
                    lr = 0.00005
                    layer_decay = 0.75

                port = dist_port[count]
                # prepare the configuration file
                cfg = create_usb_nlp_config(
                    alg,
                    seed,
                    dataset,
                    net,
                    num_classes,
                    num_labels,
                    port,
                    lr,
                    weight_decay,
                    layer_decay,
                    max_length,
                    imbalance=imbalances[dataset] if imbalances else None
                )
                count += 1
                if dataset.startswith('wildguardmix_HT'):
                    cfg['lb_imb_ratio'] = 0.0
                    cfg['include_lb_to_ulb'] = False
                    create_configuration(cfg, config_file, imbalance=None, save_suffix='_wg')
                else:
                    create_configuration(cfg, config_file, imbalance=imbalances[dataset] if imbalances else None)


if __name__ == "__main__":
    if not os.path.exists("./saved_models/usb_nlp/"):
        os.makedirs("./saved_models/usb_nlp/", exist_ok=True)
    if not os.path.exists('./disaster_tweet/config/cluster/usb_nlp/'):
        os.makedirs('./disaster_tweet/config/cluster/usb_nlp/', exist_ok=True)


    # usb nlp
    label_amount = {
        "s": [10, 10, 50, 5, 50, 50],
        "m": [50, 50, 200, 20, 200, 200],
    }

    for i in label_amount:
        exp_usb_nlp(label_amount=label_amount[i])


    # imbalanced
    # label_amount = {
    #     "s": [10, 10, 50, 5, 50, 50],
    #     "m": [50, 50, 200, 20, 200, 200],
    # }
    # # label_amount_imbalanced = [50, 50, 200, 20, 200, 200]
    # small_imbalances = {
    #     "aclImdb": 5,
    #     "ag_news": 5,
    #     "amazon_review": 4,
    #     "dbpedia": 4,
    #     "yahoo_answers": 4,
    #     "yelp_review": 4,
    # }
    # big_imbalances = {k: 100 for k in small_imbalances}
    # neg_imbalances = {k: -100 for k in small_imbalances}
    # # abc_imbalances = {k: '1k02' for k in small_imbalances}
    # # exp_usb_nlp(label_amount=label_amount['m'], imbalances=small_imbalances)
    # exp_usb_nlp(label_amount=label_amount['m'], imbalances=big_imbalances)
    # exp_usb_nlp(label_amount=label_amount['m'], imbalances=neg_imbalances)
    # # exp_usb_nlp(label_amount=label_amount['m'], imbalances=abc_imbalances)


    # WG
    # exp_usb_nlp(label_amount=None)
