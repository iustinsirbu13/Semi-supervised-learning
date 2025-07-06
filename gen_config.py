#!/usr/bin/env python3
import os
import yaml
import itertools
from collections import OrderedDict

# Custom type for floats that we want to display in plain format.
class PlainFloat(float):
    pass

def represent_plain_float(dumper, data):
    # Format the float with exactly 5 decimal places.
    formatted = format(data, '.5f')
    return dumper.represent_scalar('tag:yaml.org,2002:float', formatted)

yaml.add_representer(PlainFloat, represent_plain_float)

# Register a custom representer for OrderedDict so that it outputs a plain mapping.
def represent_ordereddict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)

# Custom representer for booleans to print them as "True"/"False".
def represent_bool(dumper, data):
    value = "True" if data else "False"
    return dumper.represent_scalar('tag:yaml.org,2002:bool', value)

yaml.add_representer(bool, represent_bool)

def create_yaml_file(file_path, config):
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    yaml_str = yaml_str.rstrip('\n')
    with open(file_path, 'w') as f:
        f.write(yaml_str)
    print(f"Saved configuration to {file_path}")


def generate_configurations(config_template):
    """
    Separates keys with list values from fixed keys,
    then creates all combinations for the keys with multiple values.
    """
    list_keys = {}    # keys that have a list of possible values
    fixed_keys = {}   # keys that have a single value
    for key, value in config_template.items():
        if isinstance(value, list):
            list_keys[key] = value
        else:
            fixed_keys[key] = value

    # Get the keys that have multiple values and compute their cartesian product.
    keys = list(list_keys.keys())
    product_values = list(itertools.product(*(list_keys[key] for key in keys)))

    # Create a configuration for each combination of values.
    configs = []
    for values in product_values:
        new_config = fixed_keys.copy()
        new_config.update(dict(zip(keys, values)))
        configs.append(new_config)
    return configs

def main():
    # Define the configuration template as an OrderedDict with keys in the exact order.
    # For keys that you want to vary, provide a list of possible values.
    config_template = OrderedDict([
        ('algorithm', 'fixmatch'),
        ('save_dir', './saved_models/usb_llm_safety'),
        # These will be updated dynamically based on current config:
        ('save_name', 'fixmatch_wildguardmix_RH_200_translate_hate_seed_0_deberta'),
        ('resume', True),
        # This will be updated dynamically based on current config:
        ('load_path', './saved_models/usb_llm_safety/fixmatch_wildguardmix_RH_200_translate_hate_seed_0_deberta/latest_model.pth'),
        ('overwrite', True),
        ('use_tensorboard', True),
        ('use_wandb', False),
        ('epoch', 100),
        ('num_train_iter', 1300),
        ('num_warmup_iter', 13),
        ('num_log_iter', 5),
        ('num_eval_iter', 13),
        ('num_labels', 200),
        # Keys with multiple values:
        ('batch_size', 16),
        ('eval_batch_size', 16),
        ('ema_m', 0.0),
        ('hard_label', True),
        ('T', 0.5),
        ('p_cutoff', 0.95),
        ('ulb_loss_ratio', 1.0),
        ('uratio', 3),
        ('use_cat', False),
        ('optim', 'AdamW'),
        ('lr', PlainFloat(5e-05)),
        ('momentum', 0.9),
        ('weight_decay', 0.0005),
        ('layer_decay', 0.75),
        ('amp', False),
        ('clip', 0.0),
        ('net', 'deberta_v3_base'),
        ('net_from_name', False),
        ('data_dir', './data'),
        ('dataset', 'wildguardmix_RH_translate_MarianMT'),
        ('train_sampler', 'RandomSampler'),
        ('num_classes', 2),
        ('num_workers', 4),
        ('max_length', 512),
        ('seed', [0, 1, 2]),
        ('world_size', 1),
        ('rank', 0),
        ('multiprocessing_distributed', False),
        ('dist_url', 'tcp://127.0.0.1:10008'),
        ('dist_backend', 'nccl'),
        ('gpu', 0),
        ('include_lb_to_ulb', False),
        ('text_weak_aug', 'orig'),
        ('text_strong_aug', 'translated'),
        ('save_pseudolabels_stats', True)
    ])


    # config_template = OrderedDict([
    #     ('algorithm', 'multimatch'),
    #     ('save_dir', './saved_models/usb_llm_safety/multimatch'),
    #     # These will be updated dynamically based on current config:
    #     ('save_name', 'multimatch_wildguard_PH_200_translate_deberta_seed_0'),
    #     ('resume', True),
    #     # This will be updated dynamically based on current config:
    #     ('load_path', './saved_models/usb_llm_safety/multimatch/multimatch_wildguard_PH_200_translate_deberta_seed_0/latest_model.pth'),
    #     ('overwrite', True),
    #     ('use_tensorboard', True),
    #     ('use_wandb', False),
    #     ('epoch', 100),
    #     ('num_train_iter', 1300),
    #     ('num_warmup_iter', 13),
    #     ('num_log_iter', 5),
    #     ('num_eval_iter', 13),
    #     ('num_labels', 2000),
    #     # Keys with multiple values:
    #     ('batch_size', 16),
    #     ('eval_batch_size', 16),
    #     ('ema_m', 0.0),
    #     ('hard_label', True),
    #     ('T', 0.5),
    #     ('p_cutoff', 0.95),
    #     ('ulb_loss_ratio', [10.0]),
    #     ('num_heads', 3),
    #     ('use_agreement_apm', True),
    #     ('apm_percentile', 0.05),
    #     ('multihead_apm_variant', 'v4'),
    #     ('no_low', False),
    #     ('apm_disagreement_weight', 3),
    #     ('threshold_algo', 'freematch'),
    #     ('apm_cutoff', 0.0),
    #     ('smoothness', 0.997),
    #     ('uratio', 3),
    #     ('use_cat', False),
    #     ('optim', 'AdamW'),
    #     ('lr', PlainFloat(5e-05)),
    #     ('momentum', 0.9),
    #     ('weight_decay', 0.0005),
    #     ('layer_decay', 0.75),
    #     ('amp', False),
    #     ('clip', 0.0),
    #     ('net', 'deberta_v3_base_multihead'),
    #     ('net_from_name', False),
    #     ('data_dir', './data'),
    #     ('dataset', 'wildguardmix_PH_translate_MarianMT_test'),
    #     ('train_sampler', 'RandomSampler'),
    #     ('num_classes', 2),
    #     ('num_workers', 4),
    #     ('max_length', 512),
    #     ('seed', [0, 1, 2]),
    #     ('world_size', 1),
    #     ('rank', 0),
    #     ('multiprocessing_distributed', False),
    #     ('dist_url', 'tcp://127.0.0.1:10001'),
    #     ('dist_backend', 'nccl'),
    #     ('gpu', 0),
    #     ('text_weak_aug', 'orig'),
    #     ('text_strong_aug', 'translated'),
    #     ('save_pseudolabels_stats', True)
    # ])
    # Generate all configuration combinations.
    configs = generate_configurations(config_template)

    # Create an output directory for generated YAML files.
    output_dir = "generated_configs"
    os.makedirs(output_dir, exist_ok=True)

    # Process each configuration: update dynamic fields and write YAML files.
    for conf in configs:
        # Update 'save_name' to include batch_size, uratio, and seed.
        conf['save_name'] = (
            f"multimatch_wildguard_PH_2000_translate_deberta_ulbloss_{conf['ulb_loss_ratio']}_"
            f"seed_{conf['seed']}.yaml"
        )
        # Update 'load_path' accordingly.
        conf['load_path'] = os.path.join(conf['save_dir'], conf['save_name'], 'latest_model.pth')

        conf['eval_batch_size'] = conf['batch_size']
        iter_one_pass = conf['num_labels'] // conf['batch_size']
        conf['num_train_iter'] = conf['epoch'] * iter_one_pass
        conf['num_warmup_iter'] = iter_one_pass
        conf['num_log_iter'] = int(iter_one_pass // 4)
        conf['num_eval_iter'] = iter_one_pass

        # Build an OrderedDict for the configuration in the same order as the template.
        ordered_conf = OrderedDict()
        for key in config_template.keys():
            ordered_conf[key] = conf[key]

        # Create a file name that includes batch_size, uratio, and seed.
        file_name = (
            f"fixmatch_wildguardmix_PH_2000_translate_hate_seed_0_deberta_ulbloss_{conf['ulb_loss_ratio']}_"
            f"seed_{conf['seed']}.yaml"
        )
        file_path = os.path.join(output_dir, file_name)
        create_yaml_file(file_path, ordered_conf)

if __name__ == '__main__':
    main()
