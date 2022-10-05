import numpy as np

from omegaconf import OmegaConf
from trainers import trainer_registry

from core.utils.common import setup_seed
from core.utils.arguments import load_config
from pprint import pprint
from pathlib import Path


def run_experiment(exp_config):
    print("Exp launched")
    pprint(OmegaConf.to_container(exp_config))
    setup_seed(exp_config.exp.seed)
    trainer = trainer_registry[exp_config.exp.trainer](exp_config)
    trainer.setup()
    trainer.train_loop()


def modificate_config(base_config, key_fn, name_fn, values):
    resulted_configs = []
    for value in values:
        modification = OmegaConf.create(key_fn(value))
        exp_config = OmegaConf.merge(base_config, modification)
        exp_config.exp.name = f'{base_config.exp.name}_{name_fn(value)}'
        resulted_configs.append(exp_config)
    return resulted_configs


class MultiConfig:
    def __init__(self, exp_config, exp_prefix):
        self.base_config = exp_config
        self.exp_prefix = exp_prefix
        self.base_config.exp.name = ''
        self.configs = [self.base_config]
        
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.configs):
            run_config = self.configs[self.i]
            self.i += 1
            run_config.exp.name = f'{self.exp_prefix}_{run_config.exp.name}'
            return run_config
        else:
            raise StopIteration
    
    def expand_config(self, key_fn, name_fn, values):
        new_configs = []
        for c in self.configs:
            new_configs.extend(modificate_config(c, key_fn, name_fn, values))
        self.configs = new_configs
        return self


domain_to_name = {
    'Anime Painting': 'anime',
    '3D Render in the Style of Pixar': 'pixar',
    'Werewolf': 'werewolf',
    'The Joker': 'joker',
    'Dog': 'dog',
    'Car': 'car',
    'Church': 'church',
    'Mona Lisa Painting': 'monalisa',
    'Sketch': "sketch",
    'A painting in Ukiyo-e style': 'ukiyo-e',
    'Fernando Botero Painting': 'botero',
    'Werewolf': 'werewolf',
    'Zombie': 'zombie',
    'The Joker': 'joker',
    'Neanderthal': 'neanderthal',
}


def run_several_seeds(base_config, prefix='seeds', values=(0, 15, 128, 322, 988)):
    mult_config = MultiConfig(base_config, prefix)
    
    key_fn = lambda a: {'exp': {'seed': a}}
    name_fn = lambda a: f'sd_{a}'
    
    mult_config.expand_config(
        key_fn, name_fn, values
    )

    key_fn = lambda pair: {'training': {'target_class': pair[0], 'source_class': pair[1]}}
    name_fn = lambda pair: domain_to_name[pair[0]]
    
    values = [
        ('Anime Painting', 'Photo'),
        ('3D Render in the Style of Pixar', 'Photo')
    ]

    mult_config.expand_config(
        key_fn, name_fn, values
    )
    
    for c in mult_config:
        run_experiment(c)
        

def multi_lr(base_config, prefix, values):
    mult_config = MultiConfig(base_config, prefix)
    
    key_fn = lambda a: {
        'optimisation_setup': {
            'optimizer': {
                'lr': a
            } 
        }
    }
    name_fn = lambda a: f'lr_{a}'
    mult_config.expand_config(
        key_fn, name_fn, values
    )

#     key_fn = lambda pair: {'training': {'target_class': pair[0], 'source_class': pair[1]}}
#     name_fn = lambda pair: domain_to_name[pair[0]]
    
#     values = [
#         ('Anime Painting', 'Photo'),
#         ('3D Render in the Style of Pixar', 'Photo'),
#     ]


#     mult_config.expand_config(
#         key_fn, name_fn, values
#     )
    
    for c in mult_config:
        run_experiment(c)
        
    
def run_td_sereval_domains(base_config):
    config_modifications_style_domains = [
        {"training": {"target_class": "Anime Painting", "source_class": "Photo"}},
        {"training": {"target_class": "Mona Lisa Painting", "source_class": "Photo"}},
        {"training": {"target_class": "3D render in the style of Pixar", "source_class": "Photo"}},
        {"training": {"target_class": "Sketch", "source_class": "Photo"}},
        {"training": {"target_class": "A painting in Ukiyo-e style", "source_class": "Photo"}},
        {"training": {"target_class": "Fernando Botero Painting", "source_class": "Photo"}},
        {"training": {"target_class": "Werewolf", "source_class": "Human"}},
        {"training": {"target_class": "Zombie", "source_class": "Human"}},
        {"training": {"target_class": "The Joker", "source_class": "Human"}},
        {"training": {"target_class": "Neanderthal", "source_class": "Human"}},
    ]
    
    # config_modifications_style_domains = [
    #     {"training": {"target_class": "Disney Princess", "source_class": "Human"}},
    #     {"training": {"target_class": "Cubism Painting", "source_class": "Photo"}},
    #     {"training": {"target_class": "Tolkien Elf", "source_class": "Human"}},
    #     {"training": {"target_class": "Impressionism Painting", "source_class": "Photo"}},
    #     {"training": {"target_class": "Pop Art", "source_class": "Photo"}},
    #     {"training": {"target_class": "Modigliani Painting", "source_class": "Photo"}},
    #     {"training": {"target_class": "Surreal Painting with colored Face", "source_class": "Photo of person"}},
    #     {"training": {"target_class": "The Thanos", "source_class": "Human"}},
    #     {"training": {"target_class": "Edvard Munch Painting", "source_class": "Photo"}},
    #     {"training": {"target_class": "Dali Painting", "source_class": "Photo"}},
    # ]
    
    # config_modifications_style_domains = [
    #     {"training": {"target_class": "1920", "source_class": "2015"}},
    #     {"training": {"target_class": "Dali Painting", "source_class": "Photo"}},
    #     {"training": {"target_class": "Ghost Car", "source_class": "Car"}},
    #     {"training": {"target_class": "Gold Car", "source_class": "Car"}},
    #     {"training": {"target_class": "TRON wheels", "source_class": "Chrome Wheels"}},
    #     {"training": {"target_class": "Pop Art", "source_class": "Photo"}},
    #     {"training": {"target_class": "Tesla", "source_class": "Car"}},
    #     {"training": {"target_class": "Lightning McQueen", "source_class": "Car"}},
    #     {"training": {"target_class": "Future Concept Car", "source_class": "Car"}},
    # ]
    
    # config_modifications_style_domains = [
    #     {"training": {"target_class": "Hut", "source_class": "Church"}},
    #     {"training": {"target_class": "Snowy Mountain", "source_class": "Church"}},
    #     {"training": {"target_class": "Ancient Underwater Ruin", "source_class": "Church"}},
    #     {"training": {"target_class": "The Shire", "source_class": "Church"}},
    #     {"training": {"target_class": "Cryengine render of Shibuya at night", "source_class": "Photo of a Church"}},
    #     {"training": {"target_class": "Cryengine render of New York", "source_class": "Photo of a Church"}},
    #     {"training": {"target_class": "Modern residential building", "source_class": "Church"}},
    #     {"training": {"target_class": "Egyptian pyramids", "source_class": "Church"}},
    #     {"training": {"target_class": "Church at night", "source_class": "Church at noon"}},
    # ]
    
    # config_modifications_style_domains = [
    #     {"training": {"target_class": "Capybara", "source_class": "Cat"}},
    #     {"training": {"target_class": "Lion", "source_class": "Cat"}},
    #     {"training": {"target_class": "Panda", "source_class": "Cat"}},
    #     {"training": {"target_class": "Meerkat", "source_class": "Cat"}},
    #     {"training": {"target_class": "Raccoon", "source_class": "Cat"}},
    #     {"training": {"target_class": "Koala", "source_class": "Cat"}},
    #     {"training": {"target_class": "Fox", "source_class": "Cat"}},
    #     {"training": {"target_class": "Hedgehog", "source_class": "Cat"}},
    #     {"training": {"target_class": "Lemur", "source_class": "Cat"}},
    # ]
    
    
def multi_im2im(base_config, prefix):
    mult_config = MultiConfig(base_config, prefix)
    
    
    key_fn = lambda value: {'training': {'target_class': value}}
    name_fn = lambda value: Path(value).stem
    values = [
        "./image_domains/sketch.png",
        "./image_domains/anastasia.png",
        "./image_domains/digital_painting_jing.png",
        "./image_domains/mermaid.png",
        "./image_domains/speed_paint.png",
        "./image_domains/titan_armin.png",
        "./image_domains/titan_erwin.png"
    ]


    mult_config.expand_config(
        key_fn, name_fn, values
    )
    
    for c in mult_config:
        run_experiment(c)
    
    
def run_several_loss_setups(base_config):
    config_loss_modifications = [
        {'optimization_setup': {
            'loss_coefs': [1.0, 1.5]
        }},
        {'optimization_setup': {
            'loss_coefs': [1.0, 0.]
        }},
        {'optimization_setup': {
            'loss_coefs': [1.0, 1.0, 0.1]
        }},
        {'optimization_setup': {
            'loss_coefs': [1.0, 1.0, 0.5]
        }},
    ]


def multi_model_grid(base_config, prefix='multi_model_grid'):
    mult_config = MultiConfig(base_config, prefix)
    
    key_fn = lambda pair: {'training': {'target_class': pair[0], 'source_class': pair[1]}}
    name_fn = lambda pair: domain_to_name[pair[0]]
    
    values = [
        ('3D Render in the Style of Pixar', 'Photo'),
        ('Anime Painting', 'Photo'),
        ('Werewolf', 'Human'),
        ('The Joker', 'Human'),
        ('Dog', 'Human'),
        ('Car', 'Human'),
        ('Church', 'Human'),
    ]

    mult_config.expand_config(
        key_fn, name_fn, values
    )
    
    def value_to_model(value):
        phase_to_lr = {
            'mapping': 0.2,
            'affine': 0.01,
            'conv_kernel': 0.002
        }
        if value[0] == 'original':
            return {
                'training': {'patch_key': 'original', 'phase': value[1], 'iter_num': 600},
                'optimization_setup': {
                    'optimizer': {
                        'lr': phase_to_lr[value[1]]
                    }
                }
            }
        
        else:
            return {
                'training': {'patch_key': value[0], 'iter_num': 120},
                'optimization_setup': {
                    'optimizer': {
                        'lr': 0.1
                    }
                }
            }
    
    name_fn = lambda pair: '_'.join(pair)
    
    values = [
        ('original', 'mapping'),
        ('original', 'affine'),
        ('original', 'conv_kernel'),
        ('s_delta', 'st'),
    ]

    mult_config.expand_config(
        value_to_model, name_fn, values
    )
    
    for c in mult_config:
        run_experiment(c)


def multi_domains_td(base_config, prefix='global'):
    mult_config = MultiConfig(base_config, prefix)

    key_fn = lambda pair: {'training': {'target_class': pair[0], 'source_class': pair[1]}}
    name_fn = lambda pair: domain_to_name[pair[0]]
    values = [
        ("Anime Painting", "Photo"),
        ("3D Render in the Style of Pixar", "Photo"),
        ("Sketch", "Photo"),
        ("A painting in Ukiyo-e style", "Photo"),
        ("Fernando Botero Painting", "Photo"),
        ("Werewolf", "Human"),
        ("Zombie", "Human"),
        ("Neanderthal", "Human"),
    ]

    mult_config.expand_config(
        key_fn, name_fn, values
    )

    #     key_fn = lambda value: {'optimization_setup': {'optimizer': {'lr': value}}}
    #     name_fn = lambda value: f'{value}'

    #     values = [
    #         0.05, 0.1, 0.2, 0.25
    #     ]

    #     mult_config.expand_config(
    #         key_fn, name_fn, values
    #     )

    for c in mult_config:
        run_experiment(c)

        
value_to_abbrev = {
    'original': 'orig',
    'channelwise_sep_mult': 'ch',
    'direction_original': 'dir_orig',
    'indomain_angle': 'in',
    'direction_mean': 'dir_mean',
    'angle': 'an',
    'regularization': 'reg',
    'global': 'glob'
}
    

if __name__ == '__main__':
    base_config = load_config()
    # run_several_seeds(base_config, prefix='seeds')
    # multi_model_grid(base_config, prefix='mult_models')
    # multi_lr(base_config, prefix='lr_mapping', values=(0.2, 0.1))
    # multi_im2im(base_config, prefix='im2im_s_delta')
    # multi_domains_td(base_config, prefix='td_')

