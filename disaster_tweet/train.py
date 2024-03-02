import sys
sys.path.append("..")

from semilearn import get_config
from wrappers.build import build_wrapper

def parse_config():
    config = get_config({})

    from semilearn.core.utils.misc import over_write_args_from_file
    over_write_args_from_file(config, config.c)

    print('Config:')
    for k, v in vars(config).items():
        print(f'\t{k}: {v}')

    return config

def main():
    config = parse_config()
    wrapper = build_wrapper(config.dataset, config.algorithm, config)
    wrapper.train_evaluate()

if __name__ == '__main__':
    main()