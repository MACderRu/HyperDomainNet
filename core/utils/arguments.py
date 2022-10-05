import os

from omegaconf import OmegaConf
from core.uda_models import uda_models
from core.utils.class_registry import ClassRegistry


args = ClassRegistry()
generator_args = uda_models.make_dataclass_from_args()
args.add_to_registry("generator_args")(generator_args)
additional_arguments = args.make_dataclass_from_classes()

DEFAULT_CONFIG_DIR = 'configs'


def get_generator_args(generator_name, base_args, conf_args):
    return OmegaConf.create(
        {generator_name: OmegaConf.merge(base_args, conf_args)}
    )

    
def load_config():
    base_gen_args_config = OmegaConf.structured(additional_arguments)

    # config.exp.config = conf_cli.exp.config
    # config.exp.config_dir = conf_cli.exp.config_dir

    # config_path = os.path.join(config.exp.config_dir, config.exp.config)
    # conf_file = OmegaConf.load(config_path)
    # config = OmegaConf.merge(config, conf_file)
    # config = OmegaConf.merge(config, conf_cli)
    
    
    conf_cli = OmegaConf.from_cli()
    conf_cli.exp.config_dir = DEFAULT_CONFIG_DIR
    if not conf_cli.get('exp', False):
        raise ValueError("No config")

    config_path = os.path.join(conf_cli.exp.config_dir, conf_cli.exp.config)
    conf_file = OmegaConf.load(config_path)
    
    conf_generator_args = conf_file.generator_args
    
    generator_args = get_generator_args(
        conf_file.training.generator, 
        base_gen_args_config.generator_args[conf_file.training.generator],
        conf_generator_args
    )
    
    gen_args = OmegaConf.create({
        'generator_args': generator_args
    })
        
    config = OmegaConf.merge(conf_file, conf_cli)
    config = OmegaConf.merge(config, gen_args)
    return config
