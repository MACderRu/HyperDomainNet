import abc
import wandb
import collections
import logging
import os

from glob import glob
from pathlib import Path
from core.utils.common import get_valid_exp_dir_name
from omegaconf import OmegaConf


class LoggingManager:
    def __init__(self, trainer_config):
        self.config = trainer_config
        config_for_logger = OmegaConf.to_container(self.config)
        config_for_logger["PID"] = os.getpid()

        self.cached_latents_local_path = None  # processed in _self._init_local_dir
        self._init_local_dir()
        config_for_logger['local_dir'] = self.local_dir

        self.exp_logger = WandbLogger(
            project=trainer_config.exp.project,
            name=trainer_config.exp.name,
            dir=trainer_config.exp.root,
            tags=tuple(trainer_config.exp.tags) if trainer_config.exp.tags else None,
            notes=trainer_config.exp.notes,
            config=config_for_logger,
        )
        self.run_dir = self.exp_logger.run_dir
        self.console_logger = ConsoleLogger(trainer_config.exp.name)

    def log_values(self, iter_num, num_iters, iter_info, **kwargs):
        self.console_logger.log_iter(
            iter_num, num_iters, iter_info, **kwargs
        )
        self.exp_logger.log(dict(itern=iter_num, **iter_info.to_dict()))

    def log_images(self, iter_num, images):
        self.exp_logger.log_images(iter_num, images)
    
    def log_info(self, output_info):
        self.console_logger.logger.info(output_info)

    def _init_local_dir(self):
        cached_latents_dir = Path('image_domains/cached_latents')
        cached_latents_dir.mkdir(exist_ok=True)
        self.cached_latents_local_path = cached_latents_dir

        project_root = Path(__file__).resolve().parent.parent.parent
        exp_path = get_valid_exp_dir_name(project_root, self.config.exp.name)
        print("Experiment dir: ", exp_path)
        self.local_dir = str(exp_path)
        os.makedirs(self.local_dir)

        with open(os.path.join(self.local_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(config=self.config, f=f.name)

        self.checkpoint_dir = os.path.join(self.local_dir, "checkpoints")
        os.mkdir(self.checkpoint_dir)
        self.models_dir = os.path.join(self.local_dir, "models")
        os.mkdir(self.models_dir)


class WandbLogger:
    def __init__(self, **kwargs):
        wandb.init(**kwargs)
        self.run_dir = wandb.run.dir
        code = wandb.Artifact("project-source", type="code")
        dirs = [
            'core',
            'utils',
            'gan_models',
        ]

        pathes = []

        for dir_p in dirs:
            pathes.extend(glob(f"{dir_p}/*.py"))

        for path in pathes + ['trainers.py', 'main.py', 'main_multi.py']:
            if Path(path).exists():
                code.add_file(path, name=path)
        wandb.run.log_artifact(code)
    
    def finish(self):
        wandb.finish()

    def log(self, data):
        wandb.log(data)
    
    def log_images(self, iter_num: int, images: dict):
        data = {k: wandb.Image(v, caption=f"iter = {iter_num}") for k, v in images.items()}
        wandb.log(data)
    

class ConsoleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        self.logger.propagate = False

    @staticmethod
    def format_info(info):
        if not info:
            return str(info)
        log_groups = collections.defaultdict(dict)
        for k, v in info.to_dict().items():
            prefix, suffix = k.split("/", 1)
            log_groups[prefix][suffix] = (
                f"{v:.3f}" if isinstance(v, float) else str(v)
            )
        formatted_info = ""
        max_group_size = len(max(log_groups, key=len)) + 2
        max_k_size = (
            max([len(max(g, key=len)) for g in log_groups.values()]) + 1
        )
        max_v_size = (
            max([len(max(g.values(), key=len)) for g in log_groups.values()])
            + 1
        )
        for group, group_info in log_groups.items():
            group_str = [
                f"{k:<{max_k_size}}={v:>{max_v_size}}"
                for k, v in group_info.items()
            ]
            max_g_size = len(max(group_str, key=len)) + 2
            group_str = "".join([f"{g:>{max_g_size}}" for g in group_str])
            formatted_info += f"\n{group + ':':<{max_group_size}}{group_str}"
        return formatted_info

    def log_iter(
        self, iter_num, num_iters, iter_info, event="epoch"
    ):
        output_info = (
            f"{event.upper()} ITER {iter_num}/{num_iters}:"
        )

        output_info += self.format_info(iter_info)
        self.logger.info(output_info)
