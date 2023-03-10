# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import gym
import os
import contextlib
import mujoco_py as mjc
from PIL import Image

@contextlib.contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull) as err, contextlib.redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    import d4rl

ANTMAZE_BOUNDS = {
    'antmaze-umaze-v0': (-3, 11),
    'antmaze-medium-play-v0': (-3, 23),
    'antmaze-medium-diverse-v0': (-3, 23),
    'antmaze-large-play-v0': (-3, 39),
    'antmaze-large-diverse-v0': (-3, 39),
}


def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def save_gif(image_list, images_savepath):
    w, h = image_list[0].shape[0], image_list[0].shape[1]
    images = []
    for i, img in enumerate(image_list):
        img_ = Image.fromarray(img)
        w_ = int(w * i / len(image_list))
        pbar = Image.new('RGBA', (w_, h // 50), (0, 0, 255, 0))  # blue progress bar
        img_.paste(pbar, (0, h - h // 50))
        images.append(img_)
    images[0].save(images_savepath, format="GIF", append_images=images[1:], save_all=True, duration=len(images)/24, loop=0)


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    qstate_dim = qpos_dim + qvel_dim

    if 'ant-' in env.name:
        ypos = np.zeros(1)
        state = np.concatenate([ypos, state])

    if state.size == qpos_dim - 1 or state.size == qstate_dim - 1:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])

    if state.size == qpos_dim:
        qvel = np.zeros(qvel_dim)
        state = np.concatenate([state, qvel])

    if 'ant-' in env.name:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])[:qstate_dim]
    
    if state.size > qpos_dim + qvel_dim:
        state = state[:qstate_dim]

    assert state.size == qpos_dim + qvel_dim

    env.set_state(state[:qpos_dim], state[qpos_dim:])


class Renderer:
    def __init__(self, env, observation_dim=None, action_dim=None):
        self.env = load_environment(env) if type(env) is str else env
        self.observation_dim = observation_dim or np.prod(self.env.observation_space.shape)
        self.action_dim = action_dim or np.prod(self.env.action_space.shape)
        self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        self.set_viewer()

    def set_viewer(self, render_kwargs=None):
        if render_kwargs is None:
            if self.env.name.startswith("antmaze"):
                pos = sum(ANTMAZE_BOUNDS.get(self.env.name)) / 2
                render_kwargs = {
                    'trackbodyid': 2,
                    'distance': 3,
                    'lookat': [pos, pos, pos * 4],
                    'elevation': -90
                }
            else:
                render_kwargs = {
                    'trackbodyid': 2,
                    'distance': 2,
                    'lookat': [0, -0.5, 1],
                    'elevation': -20
                }
        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

    def render(self, observation, dim=512):
        observation = to_np(observation)
        set_state(self.env, observation)
        dim = (dim, dim) if type(dim) == int else dim
        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def render_observations(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return images
    
    def save_gif(self, image_list, images_savepath):
        save_gif(image_list, images_savepath)


if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from dataset import DiscretizedDataset
    dataset = DiscretizedDataset(
        logger=None,
        env="antmaze-large-play-v0",
        n_bins=100,
        sequence_length=10,
        penalty=0,
        discount=0.99,
    )
    traj_len = dataset.path_lengths[0]
    traj = dataset.joined_segmented[0, :traj_len]
    print(traj.shape)

    env = load_environment("antmaze-large-play-v0")
    renderer = Renderer(env, dataset.observation_dim, dataset.action_dim)
    imgs = renderer.render_observations(traj)

    images_savepath = "test.gif"
    save_gif(imgs, images_savepath)
