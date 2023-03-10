# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import time


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('inf')
    return out


def filter_cdf(logits, threshold):
    batch_inds = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    bins_inds = torch.arange(logits.shape[-1], device=logits.device)
    probs = logits.softmax(dim=-1)
    probs_sorted, _ = torch.sort(probs, dim=-1)
    probs_cum = torch.cumsum(probs_sorted, dim=-1)
    ## get minimum probability p such that the cdf up to p is at least `threshold`
    mask = probs_cum < threshold
    masked_inds = torch.argmax(mask * bins_inds, dim=-1)
    probs_threshold = probs_sorted[batch_inds, masked_inds]
    ## filter
    out = logits.clone()
    logits_mask = probs <= probs_threshold.unsqueeze(dim=-1)
    # Not -inf to prevent error: Assertion `cumdist[size - 1] > static_cast<scalar_t>(0)` failed
    out[logits_mask] = -1000
    return out


def round_to_multiple(x, N):
    pad = (N - x % N) % N
    return x + pad


def forward(model, x, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    model.train(False)
    bs = model.module.get_block_size() if hasattr(model, "module") else model.get_block_size()
    block_size = min(bs, max_block or np.inf)
    if x.shape[1] > block_size:
        assert allow_crop
        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]
    logits, _, _ = model(x, return_info=False, **kwargs)
    return logits


def sample(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    logits = forward(model, x, **forward_kwargs)
    logits = logits[:, -1] / temperature
    raw_probs = logits.softmax(dim=-1)
    if cdf is not None:
        logits = filter_cdf(logits, cdf)
    if topk is not None:
        logits = top_k_logits(logits, topk)
    probs = logits.softmax(dim=-1)
    indices = torch.multinomial(probs, num_samples=1)
    return indices, raw_probs


@torch.no_grad()
def sample_n(model, x, N, **sample_kwargs):
    batch_size = len(x)
    vs = model.module.vocab_size if hasattr(model, "module") else model.vocab_size
    probs = torch.zeros(batch_size, N, vs + 1, device=x.device)
    for n in range(N):
        indices, p = sample(model, x, **sample_kwargs)
        x = torch.cat((x, indices), dim=1)
        probs[:, n] = p
    return x, probs

@torch.no_grad()
def sample_rollout(model, x, N, temperature=1.0, cdf=None, topk=None):
    bs = len(x)
    vs = model.module.vocab_size if hasattr(model, "module") else model.vocab_size
    probs = torch.zeros(bs, N, vs + 1, device=x.device)
    for n in range(N):
        logits, _, _ = model(x, return_info=False)
        logits = logits[:, -1] / temperature
        raw_probs = logits.softmax(dim=-1)
        if cdf is not None:
            logits = filter_cdf(logits, cdf)
        if topk is not None:
            logits = top_k_logits(logits, topk)
        real_probs = logits.softmax(dim=-1)
        indices = torch.multinomial(real_probs, num_samples=1)
        x = torch.cat((x, indices), dim=1)
        probs[:, n] = raw_probs
    return x, probs
