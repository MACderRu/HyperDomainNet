from argparse import Namespace

################# II2S options for style image embedding  Note: p_norm_lambda = 1e-2 not 1e-3
opts = Namespace()

# StyleGAN2 setting
opts.size = 1024
opts.ckpt = "pretrained/StyleGAN2/stylegan2-ffhq-config-f.pt"
opts.channel_multiplier = 2
opts.latent = 512
opts.n_mlp = 8

# loss options
opts.percept_lambda = 1.0
opts.l2_lambda = 1.0
opts.p_norm_lambda = 1e-3

# arguments
opts.device = 'cuda'
opts.seed = 2
opts.tile_latent = False
opts.opt_name = 'adam'
opts.learning_rate = 0.01
opts.lr_schedule = 'fixed'
opts.steps = 1000
opts.save_intermediate = False
opts.save_interval = 300
opts.verbose = False

II2S_s_opts = opts