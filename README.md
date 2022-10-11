# HyperDomainNet: Universal Domain Adaptation for Generative Adversarial Networks (NeurIPS 2022)


> **HyperDomainNet: Universal Domain Adaptation for Generative Adversarial Networks**<br>
> Aibek Alanov*, Vadim Titov*, Dmitry Vetrov <br>
> *Equal contribution <br>
> https://arxiv.org/abs/ <br>
>
>**Abstract:** Domain adaptation framework of GANs has achieved great progress in recent years as a main successful approach of training contemporary GANs in the case of very limited training data. In this work, we significantly improve this framework by proposing an extremely compact parameter space for fine-tuning the generator. We introduce a novel domain-modulation technique that allows to optimize only 6 thousand-dimensional vector instead of 30 million weights of StyleGAN2 to adapt to a target domain. We apply this parameterization to the state-of-art domain adaptation methods and show that it has almost the same expressiveness as the full parameter space. Additionally, we propose a new regularization loss that considerably enhances the diversity of the fine-tuned generator. Inspired by the reduction in the size of the optimizing parameter space we consider the problem of multi-domain adaptation of GANs, i.e. setting when the same model can adapt to several domains depending on the input query. We propose the HyperDomainNet that is a hypernetwork that predicts our parameterization given the target domain. We empirically confirm that it can successfully learn a number of domains at once and may even generalize to unseen domains.
