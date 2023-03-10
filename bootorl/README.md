# Bootstrapped Transformer
Source code for NeurIPS 2022 paper *[Bootstrapped Transformer for Offline Reinforcement Learning](https://seqml.github.io/bootorl/)*.

## Abstract
> Offline reinforcement learning (RL) aims at learning policies from previously collected static trajectory data without interacting with the real environment. Recent works provide a novel perspective by viewing offline RL as a generic sequence generation problem, adopting sequence models such as Transformer architecture to model distributions over trajectories, and repurposing beam search as a planning algorithm. However, the training datasets utilized in general offline RL tasks are quite limited and often suffer from insufficient distribution coverage, which could be harmful to training sequence generation. In this paper, we propose a novel algorithm named Bootstrapped Transformer, which incorporates the idea of bootstrapping and leverages the learned model to self-generate more offline data to further boost the sequence model training. We conduct extensive experiments on two offline RL benchmarks and demonstrate that our model can largely remedy the existing offline RL training limitations and beat other strong baseline methods. We also analyze the generated pseudo data and the revealed characteristics may shed some light on offline RL training. The codes and supplementary materials are available at https://seqml.github.io/bootorl.

## Dependencies

Python dependencies are listed in [`./environment.yml`](./environment.yml).

We also provides an extra dockerfile as [`./Dockerfile`](./Dockerfile) for reproducibility. 

## Usage

To train the model, run with 
```
python main/train.py --dataset hopper-medium-replay-v2 \
                     --bootstrap True \
                     --bootstrap_type once \
                     --generation_type autoregressive
```
or
```
python main/train.py --dataset hopper-medium-replay-v2 \
                     --bootstrap True \
                     --bootstrap_type repeat \
                     --generation_type teacherforcing
```
depending on your choice of hyperparameters and bootstrap schemes. All default hyperparameters used in our experiments are placed at [`./utils/argparser.py`](`./utils/argparser.py`). You can find it in `DEFAULT_ARGS` at the beginning of this file. By default, training logs and saved models are output to `./logs/<environment>-<dataset_level>/` directory.

To evaluate the performance of trained model, run with
```
python main/plan.py --dataset hopper-medium-replay-v2 \
                    --checkpoint <checkpoint_directory> \
                    --suffix <output_directory_suffix>
```
where `checkpoint_directory` should be the directory containing your model `state_*.pt`. By default, evaluation results are output to `./logs/<environment>-<dataset_level>/<suffix>` directory.


## Acknowledgements
Some source codes of this work have been implemented on top of *Trajectory Transformer* (https://arxiv.org/abs/2106.02039).
*Trajectory Transformer* uses GPT implementation from Andrej Karpathy's *minGPT* repo.

## Citation
You are more than welcome to cite our paper:
```
@article{wang2022bootstrapped,
  title={Bootstrapped Transformer for Offline Reinforcement Learning},
  author={Wang, Kerong and Zhao, Hanye and Luo, Xufang and Ren, Kan and Zhang, Weinan and Li, Dongsheng},
  journal={arXiv preprint arXiv:2206.08569},
  year={2022}
}
```