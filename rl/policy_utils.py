"""Prompt encoding and batched completion log-probabilities for the RL stage."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from generator.sft import IMAGE_TOKEN_INDEX

def encode_prompt(tok, user_text: str):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "<image>\n" + user_text}],
        add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    post_ids = tok(post, add_special_tokens=False).input_ids
    return pre_ids + [IMAGE_TOKEN_INDEX] + post_ids

def comp_logprobs(model, prompt_ids: list[int], comps: list[list[int]],
                  px: torch.Tensor, requires_grad: bool):

    dev = model.device
    K = len(comps)
    max_T = max(len(c) for c in comps)
    L = len(prompt_ids) + max_T
    pad = model.config.eos_token_id or 0
    ids = torch.full((K, L), pad, dtype=torch.long)
    attn = torch.zeros((K, L), dtype=torch.long)
    for i, c in enumerate(comps):
        n = len(prompt_ids) + len(c)
        ids[i, :n] = torch.tensor(prompt_ids + c)
        attn[i, :n] = 1
    ids, attn = ids.to(dev), attn.to(dev)
    px = px.to(dev, dtype=next(model.parameters()).dtype)
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=ids, attention_mask=attn,
                       images=px.expand(K, *px.shape[1:])).logits
    n_img = logits.shape[1] - L + 1
    lp = torch.full((K, max_T), 0.0, device=dev)
    mask = torch.zeros((K, max_T), dtype=torch.bool, device=dev)
    for i, c in enumerate(comps):
        T = len(c)
        Li = len(prompt_ids) - 1 + n_img + T
        cl = logits[i, Li - T - 1: Li - 1].float()
        lsm = F.log_softmax(cl, dim=-1)
        lp[i, :T] = lsm.gather(-1, torch.tensor(c, device=dev)[:, None])[:, 0]
        mask[i, :T] = True
    return lp, mask
