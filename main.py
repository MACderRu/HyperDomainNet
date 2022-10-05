from omegaconf import OmegaConf
from trainers import trainer_registry

from core.utils.common import setup_seed
from core.utils.arguments import load_config
from pprint import pprint


def run_experiment(exp_config):
    pprint(OmegaConf.to_container(exp_config))
    setup_seed(exp_config.exp.seed)
    trainer = trainer_registry[exp_config.exp.trainer](exp_config)
    trainer.setup()
    trainer.train_loop()


def run_experiment_from_ckpt():
    ...


if __name__ == '__main__':
    base_config = load_config()

    if base_config.get('checkpoint'):
        ...

    run_experiment(base_config)
