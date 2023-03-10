# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import numpy as np
from tqdm import tqdm

from .sample import sample_n
from .timer import Timer

VALUE_PLACEHOLDER = 0


def make_prefix(discretizer, context, obs, device, prefix_context=True):
    obs_discrete = discretizer.discretize(obs, subslice=[0, obs.size])
    obs_discrete = torch.tensor(obs_discrete, dtype=torch.long, device=device)
    prefix = torch.cat(context + [obs_discrete], dim=-1) if prefix_context else obs_discrete
    return prefix


def extract_actions(x, observation_dim, action_dim, t=None):
    assert x.shape[1] == observation_dim + action_dim + 2
    actions = x[:, observation_dim:observation_dim+action_dim]
    return actions[t] if t is not None else actions


def update_context(context, discretizer, obs, act, rew, device, max_context_transitions):
    # use a placeholder for value because input values are masked out by model
    rew_val = np.array([rew, VALUE_PLACEHOLDER])
    transition = np.concatenate([obs, act, rew_val])
    # discretize transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition)
    transition_discrete = torch.tensor(transition_discrete, dtype=torch.long, device=device)
    # add new transition to context
    context.append(transition_discrete)
    # crop context if necessary
    context = context[-max_context_transitions:]
    return context


@torch.no_grad()
def beam_plan(
    model, value_fn, x, rollout_steps, beam_width, n_expand, obs_dim, act_dim,
    discount=0.99, max_trans=None, k_obs=None, k_act=None, k_rew=1, cdf_obs=None, cdf_act=None, cdf_rew=None,
    with_tqdm=True, return_values=False
):
    # convert max number of transitions to max number of tokens
    transition_dim = obs_dim + act_dim + 2
    max_block = max_trans * transition_dim - 1 if max_trans else None

    # pass in max numer of tokens to sample function
    sample_kwargs = {'max_block': max_block, 'crop_increment': transition_dim}

    # repeat input for search
    x = x.repeat(beam_width, 1)

    # construct reward and discount tensors for estimating values
    rewards = torch.zeros(beam_width, rollout_steps + 1, device=x.device)
    discounts = discount ** torch.arange(rollout_steps + 1, device=x.device)

    pbar = tqdm(range(rollout_steps), leave=False) if with_tqdm else range(rollout_steps)
    for t in pbar:
        # repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        # sample actions
        x, _ = sample_n(model, x, act_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        # sample reward and value estimate
        x, r_probs = sample_n(model, x, 2, topk=k_rew, cdf=cdf_rew, **sample_kwargs)
        r_t, V_t = value_fn(r_probs)
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        # estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        # get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        # index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        # sample next observation (unless we have reached the end of the planning horizon)
        if t < rollout_steps - 1:
            x, _ = sample_n(model, x, obs_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)
        
        if with_tqdm:
            pbar.set_description(f"Context shape: {list(x.shape)} | "
                                 f"V_(t) estimate: [{V_t.min():.2f}, {V_t.max():.2f}] | "
                                 f"V_(t+{t}) estimate [{values.min():.2f}, {values.max():.2f}]")

    x = x.view(beam_width, -1, transition_dim)
    x = x[:, -rollout_steps:]

    # return best sequence
    argmax = values.argmax()
    best_sequence = x[argmax]
    best_value = (rewards * discounts)[argmax]

    if not return_values:
        return best_sequence
    else:
        return best_sequence, best_value


def plan(args, env, dataset, model, logger, device='cuda:0'):
    timer = Timer(total_num=env.max_episode_steps)

    discretizer = dataset.discretizer
    discount = dataset.discount
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    value_fn = lambda x: discretizer.value_fn(x, args.percentile)

    # main loop
    observation = env.reset()
    total_reward = 0

    ## observations for rendering
    rollout_states = [observation.copy().tolist()]
    predict_states = [observation.copy().tolist()]
    rollout_values = []
    real_rewards = []

    ## previous (tokenized) transitions for conditioning transformer
    context = []

    T = env.max_episode_steps
    for t in range(T):
        prefix = make_prefix(discretizer, context, observation, device, args.prefix_context)
        sequence, best_value = beam_plan(
            model, value_fn, prefix, args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
            discount, args.max_context_transitions, with_tqdm=args.with_tqdm, return_values=True,
            k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
        )

        sequence_recon = discretizer.reconstruct(sequence)
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)
        pred_observation = sequence_recon[0, :observation_dim]
        next_observation, reward, terminal, _ = env.step(action)

        rollout_states.append(next_observation.copy().tolist())
        predict_states.append(pred_observation.copy().tolist())
        rollout_values.append(best_value.cpu().numpy().tolist())
        real_rewards.append(reward)

        total_reward += reward
        score = env.get_normalized_score(total_reward)
        context = update_context(context, discretizer, observation, action, reward, device, args.max_context_transitions)

        diff_time, total_time, eta = timer()
        logger.info(f"Step: {t:>4} / {T:>4} | r': {best_value[0]:.2f} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f}")
        logger.debug(f"Previous time: {diff_time} | Total time: {total_time} | ETA: {eta}")

        if terminal:
            break
        observation = next_observation

    info = {
        "rollout_states": rollout_states[1:],
        "predict_states": predict_states[1:],
        "rollout_values": rollout_values,
        "real_rewards": real_rewards,
    }

    return info
