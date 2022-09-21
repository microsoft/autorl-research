# Towards Applicable Reinforcement Learning: Improving the Generalization and Sample Efficiency with Policy Ensemble
This is the experiment code for our IJCAI 2022 paper "[Towards Applicable Reinforcement Learning: Improving the Generalization and Sample Efficiency with Policy Ensemble](https://seqml.github.io/eppo/)".

## Abstract
> It is challenging for reinforcement learning (RL) algorithms to succeed in real-world applications like financial trading and logistic system due to the noisy observation and environment shifting between training and evaluation. Thus, it requires both high sample efficiency and generalization for resolving real-world tasks. However, directly applying typical RL algorithms can lead to poor performance in such scenarios. Considering the great performance of ensemble methods on both accuracy and generalization in supervised learning (SL), we design a robust and applicable method named Ensemble Proximal Policy Optimization (EPPO), which learns ensemble policies in an end-to-end manner. Notably, EPPO combines each policy and the policy ensemble organically and optimizes both simultaneously. In addition, EPPO adopts a diversity enhancement regularization over the policy space which helps to generalize to unseen states and promotes exploration. We theoretically prove EPPO increases exploration efficacy, and through comprehensive experimental evaluations on various tasks, we demonstrate that EPPO achieves higher efficiency and is robust for real-world applications compared with vanilla policy optimization algorithms and other ensemble methods. Code and supplemental materials are available at https://seqml.github.io/eppo.

## Environment Dependencies
### Dependencies
```
pip install -r requirements.txt
```

### Running
Take `Pong` environment in Atari benchmarks as an example, to run EPPO, you can do the following.
```
python code/tools/train_on_atari.py exp/atari_local.yml
```

To run EPPO-Ens, please set the `center_policy_coef` in `exp/atari_local.yml` to 0.

To run EPPO-Div, please set the `diverse_coef` in `exp/atari_local.yml` to 0.

## Reference
You are more than welcome to cite our paper:
```
@article{yang2022towards,
  title={Towards Applicable Reinforcement Learning: Improving the Generalization and Sample Efficiency with Policy Ensemble},
  author={Yang, Zhengyu and Ren, Kan and Luo, Xufang and Liu, Minghuan and Liu, Weiqing and Bian, Jiang and Zhang, Weinan and Li, Dongsheng},
  journal={arXiv preprint arXiv:2205.09284},
  year={2022}
}
```
