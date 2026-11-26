# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

import torch

LOOP_WINDOW_SIZE = 128


class LoopCache(Protocol):
    layer_name: str
    kv_cache: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class LoopPass:
    local_cache: LoopCache
    topk_indices: torch.Tensor
    swa_indices: torch.Tensor


_loop_pass: ContextVar[LoopPass | None] = ContextVar("glm53_loop_pass", default=None)


def get_loop_pass() -> LoopPass | None:
    return _loop_pass.get()


@contextmanager
def loop_attention_pass(
    local_cache: LoopCache, topk_indices: torch.Tensor, swa_indices: torch.Tensor
) -> Iterator[None]:
    token = _loop_pass.set(LoopPass(local_cache, topk_indices, swa_indices))
    try:
        yield
    finally:
        _loop_pass.reset(token)


def register_loop_cache(attn_wrapper, vllm_config) -> LoopCache:
    # Keep the tensor helpers importable without vLLM/NPU.
    from vllm_ascend.attention.sfa_v1 import Glm53LoopCache

    return Glm53LoopCache(attn_wrapper.mla_attn, vllm_config)


def build_loop_swa_indices(positions: torch.Tensor) -> torch.Tensor:
    """Build TND sparse indices with all valid slots before the -1 padding."""
    positions = positions.to(torch.int32).unsqueeze(-1)
    offsets = torch.arange(LOOP_WINDOW_SIZE, dtype=torch.int32, device=positions.device)
    indices = (positions - (LOOP_WINDOW_SIZE - 1)).clamp_min(0) + offsets
    return indices.masked_fill_(indices > positions, -1).unsqueeze(1)


def pack_loop_attention_partial(
    output: torch.Tensor, softmax_max: torch.Tensor, softmax_sum: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack SFA's [KV heads, tokens, groups] statistics in output row order."""
    row_max = softmax_max.permute(1, 0, 2).reshape(-1).float()
    row_sum = softmax_sum.permute(1, 0, 2).reshape(-1).float()
    empty = row_sum == 0
    lse = (row_max + row_sum.log()).masked_fill_(empty, -torch.inf)
    rows = output.flatten(0, 1).float().masked_fill(empty[:, None], 0)
    return rows, lse
