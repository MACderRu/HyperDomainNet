# HyperDomainNet: Universal Domain Adaptation for Generative Adversarial Networks (NeurIPS 2022)

Editing Playground: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QMylWjzPxvHtxm74U4lWRQXwquw5AaFL#scrollTo=si2tLKYLT-kV)

> **HyperDomainNet: Universal Domain Adaptation for Generative Adversarial Networks**<br>
> Aibek Alanov*, Vadim Titov*, Dmitry Vetrov <br>
> *Equal contribution <br>
> https://arxiv.org/abs/ <br>
>
>**Abstract:** Domain adaptation framework of GANs has achieved great progress in recent years as a main successful approach of training contemporary GANs in the case of very limited training data. In this work, we significantly improve this framework by proposing an extremely compact parameter space for fine-tuning the generator. We introduce a novel domain-modulation technique that allows to optimize only 6 thousand-dimensional vector instead of 30 million weights of StyleGAN2 to adapt to a target domain. We apply this parameterization to the state-of-art domain adaptation methods and show that it has almost the same expressiveness as the full parameter space. Additionally, we propose a new regularization loss that considerably enhances the diversity of the fine-tuned generator. Inspired by the reduction in the size of the optimizing parameter space we consider the problem of multi-domain adaptation of GANs, i.e. setting when the same model can adapt to several domains depending on the input query. We propose the HyperDomainNet that is a hypernetwork that predicts our parameterization given the target domain. We empirically confirm that it can successfully learn a number of domains at once and may even generalize to unseen domains.

## Description
Official Implementation of HyperDomainNet, a method of domain adaptation technique utilizes both text-driven and one-shot setups.

Our method consist of several types of adaptation setups: 
- text-driven single domain adaptation
- image2image domain adaptation
- HyperDomainNet for any textual description
- HyperDomainNet for any given image (would be improved in future research).

There are three type of models: 
- Fine-tuned aligned child generator.
- Specific target domain modulation operator.
- Universal HyperDomainNet.

## Updates

**12/10/2022** Initial version

## Setup 
For all the methods described in the paper, is it required to have:
- Anaconda
- [CLIP](https://github.com/openai/CLIP)
- required packages from requirements.txt

Specific requirements for each method are described in its section. 
To install CLIP please run the following commands:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```

## Overall  

### Setup

Here, the code relies on the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2.
Some parts of the StyleGAN implementation were modified, so that the whole implementation is native pytorch. 

In addition to the requirements mentioned before, a pretrained StyleGAN2 generator will attempt to be downloaded with script *download.py*. All base requirements could be installed via 

```shell
./setup_environment.sh
```

## Model training

Here, we provide the code for the training. Each model is trained according its config setup. Config explanation could be found in `examples/train_example.ipynb`.

### Usage

When training ends model checkpoints could be found in `local_logged_exps/`. Each `ckpt_name.pt` could be inferenced using a helper classes `InferenceWrapper` in `core/utils/example_utils`.

## Generating data from target domain  

Here, we provide the code for using pretrained checkpoints for inference.

### Setup

Pretrained models for various stylisation are provided. Please refer to `download.py` with flags `--td_single` or `--im2im_single`. First option downloads text-driven single domain adaptation pretrained checkpoints, second is for one-shot domain adaptation models.

### Usage

Given a pretrained checkpoint for certain target domain, one can edit a given image
This operation can be done through the `examples/inference_playground.ipynb` notebook

### Code details

Core functions are 
* mixing_noise (latent code generation)
* InferenceWrapper (checkpoint processer)


## Model evaluation   

Here, we provide the code for evaluation based on clip metrics.  

### Setup

Before evaluation trained models needed to be got with one of two mentioned ways (`trained/pretrained`).

### Usage

Given a pretrained checkpoint for certain target domain, one can be evaluated through the `examples/evaluation.ipynb` notebook

### Code details

/# TODO:

## Related Works

Main idea is based on one-shot (text-drive, image2image) methods [StyleGAN-NADA](https://arxiv.org/abs/2108.00946) and [MindTheGap](https://arxiv.org/abs/2110.08398).

To edit real images, we inverted them to the StyleGAN's latent space using [pSp](https://github.com/eladrich/pixel2style2pixel).

## Citation

If you use this code for your research, please cite our paper:

```
@InProceedings{Alanov_2022_NeurIPS,
    author    = {Alanov, Aibek and Titov, Vadim and Vetrov, Dmitry},
    title     = {HyperDomainNet: Universal Domain Adaptation for Generative Adversarial Networks},
    month     = {October},
    year      = {2022},
}
```