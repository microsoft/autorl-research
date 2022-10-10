# A2LS: Reinforcement Learning with Automated Auxiliary Loss Search

Code for NeurIPS 2022 paper [Reinforcement Learning with Automated Auxiliary Loss Search](https://seqml.github.io/A2LS/).

This repository is the implementation of A2LS based on the official implementation of [CURL](https://mishalaskin.github.io/curl/) for the DeepMind control experiments.

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
We present some running examples of training RL with auxiliary losses with our code base.
### `Forward dynamics prediction`
$\mathcal{L}_{\text{forward \ dynamics}} = \| h(g_\theta(s_t, a_t)) - g_{\hat{\theta}}(s_{t+1}) \|_2$

To train a agent on the with `Forward dynamics prediction` on image-based Cheetah-Run with default hyper-parameters (please refer to appendix for detailed hyper-parameters for each experiment setting): 
```
python train.py \
    --domain_name cheetah_run \
    --encoder_type pixel \
    --agent auxi_sac \ 
    --auxi_pred_horizon 1 \
    --auxi_pred_input_s 1 --auxi_pred_input_a 1 --auxi_pred_input_r 0 --auxi_pred_input_s_ 0\
    --auxi_pred_output_s 0 --auxi_pred_output_a 0 --auxi_pred_output_r 0 --auxi_pred_output_s_ 1\
    --similarity_metric mse
```
To train a agent on the with `Forward dynamics prediction` on vector-based Cheetah-Run with default hyper-parameters (please refer to appendix for detailed hyper-parameters for each experiment setting): 
```
python train.py \
    --domain_name cheetah_run \
    --encoder_type ofe --encoder_hidden_size 40 --num_layers 1  \
    --agent auxi_sac \ 
    --auxi_pred_horizon 1 \
    --auxi_pred_input_s 1 --auxi_pred_input_a 1 --auxi_pred_input_r 0 --auxi_pred_input_s_ 0\
    --auxi_pred_output_s 0 --auxi_pred_output_a 0 --auxi_pred_output_r 0 --auxi_pred_output_s_ 1\
    --similarity_metric mse
```

### `A2-winner`
$\mathcal{L}_{\text{A2-winner}} = \| h(g_\theta(s_{t+1}, a_{t+1}, a_{t+2}, a_{t+3})) - g_{\hat{\theta}}(r_t, r_{t+1}, s_{t+2}, s_{t+3}) \|_2$

To train a agent on the with `A2-winner` on image-based Cheetah-Run with default hyper-parameters (please refer to appendix for detailed hyper-parameters for each experiment setting): 
```
python train.py \
    --domain_name cheetah_run \
    --encoder_type pixel \
    --agent auxi_sac \ 
    --auxi_pred_horizon 4 \
    --auxi_pred_input_s 1000 --auxi_pred_input_a 1111 --auxi_pred_input_r 1101 --auxi_pred_input_s_ 0\
    --auxi_pred_output_s 0111 --auxi_pred_output_a 0000 --auxi_pred_output_r 0000 --auxi_pred_output_s_ 1\
    --similarity_metric mse
```

To train a agent on the with `A2-winner` on vector-based Cheetah-Run with default hyper-parameters (please refer to appendix for detailed hyper-parameters for each experiment setting): 
```
python train.py \
    --domain_name cheetah_run \
    --encoder_type ofe --encoder_hidden_size 40 --num_layers 1  \
    --agent auxi_sac \ 
    --auxi_pred_horizon 4 \
    --auxi_pred_input_s 1000 --auxi_pred_input_a 1111 --auxi_pred_input_r 1101 --auxi_pred_input_s_ 0\
    --auxi_pred_output_s 0111 --auxi_pred_output_a 0000 --auxi_pred_output_r 0000 --auxi_pred_output_s_ 1\
    --similarity_metric mse
```

### `A2-winner-v`

$\mathcal{L}_{\text{A2-winner-v}} = \| h(g_\theta(s_{t}, a_{t}, a_{t+1}, s_{t+2} a_{t+2}, a_{t+3}, r_{t+3}, a_{t+4}, r_{t+4}, a_{t+5}, a_{t+7}, s_{t+8}, a_{t+8}, r_{t+8})) - g_{\hat{\theta}}(s_{t+1}, s_{t+3}, a_{t+4}, s_{t+6}, s_{t+9}) \|_2$

To train a agent on the with `A2-winner` on image-based Cheetah-Run with default hyper-parameters (please refer to appendix for detailed hyper-parameters for each experiment setting): 
```
python train.py \
    --domain_name cheetah_run \
    --encoder_type pixel \
    --agent auxi_sac \ 
    --auxi_pred_horizon 9 \
    --auxi_pred_input_s 101000001 --auxi_pred_input_a 111111011 --auxi_pred_input_r 000110001 --auxi_pred_input_s_ 0\
    --auxi_pred_output_s 010100100 --auxi_pred_output_a 000010000 --auxi_pred_output_r 000000000 --auxi_pred_output_s_ 1\
    --similarity_metric mse
```

To train a agent on the with `A2-winner` on vector-based Cheetah-Run with default hyper-parameters (please refer to appendix for detailed hyper-parameters for each experiment setting): 
```
python train.py \
    --domain_name cheetah_run \
    --encoder_type ofe --encoder_hidden_size 40 --num_layers 1  \
    --agent auxi_sac \ 
    --auxi_pred_horizon 9 \
    --auxi_pred_input_s 101000001 --auxi_pred_input_a 111111011 --auxi_pred_input_r 000110001 --auxi_pred_input_s_ 0\
    --auxi_pred_output_s 010100100 --auxi_pred_output_a 000010000 --auxi_pred_output_r 000000000 --auxi_pred_output_s_ 1\
    --similarity_metric mse
```

## Issues

For GPU accelerated rendering, make sure EGL is installed on your machine and set `export MUJOCO_GL=egl`. 

For environment troubleshooting issues, see the DeepMind control documentation.