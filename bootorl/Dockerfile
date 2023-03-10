# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

WORKDIR /workspace

# Install new cuda-keyring package
# Noted at https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y zlib1g zlib1g-dev libosmesa6-dev libgl1-mesa-glx libglfw3 libglew2.0 cmake git \
    && ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

# Install MuJoCo 2.1.0.
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin
RUN mkdir -p /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz \
    && tar -xvzf mujoco210.tar.gz -C /root/.mujoco \
    && rm mujoco210.tar.gz

# Install packages, mainly d4rl, which will also install corresponding dependencies automatically.
RUN pip install -U scikit-learn pandas \
    && pip install git+https://github.com/rail-berkeley/d4rl.git@d842aa194b416e564e54b0730d9f934e3e32f854 \
    && pip install git+https://github.com/openai/gym.git@66c431d4b3072a1db44d564dab812b9d23c06e14

# Pre-download dataset if necessary
# RUN python -c "import gym; import d4rl; [gym.make(f'{game}-{level}-v2').unwrapped.get_dataset() for level in \
#     ['medium', 'medium-replay', 'medium-expert', 'expert'] for game in ['halfcheetah', 'hopper', 'walker2d']];"
