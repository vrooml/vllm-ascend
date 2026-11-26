# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

import torch

LOOP_WINDOW_SIZE = 128
# Bound the gathered pages and fp32 attention workspace independently of
# the number of tokens scheduled for a prefill step.
_QUERY_TILE_SIZE = 16


class LoopCache(Protocol):
    layer_name: str
    kv_cache: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class LoopPass:
    local_cache: LoopCache
    topk_indices: torch.Tensor


_loop_pass: ContextVar[LoopPass | None] = ContextVar("glm53_loop_pass", default=None)


def get_loop_pass() -> LoopPass | None:
    return _loop_pass.get()


@contextmanager
def loop_attention_pass(local_cache: LoopCache, topk_indices: torch.Tensor) -> Iterator[None]:
    token = _loop_pass.set(LoopPass(local_cache, topk_indices))
    try:
        yield
    finally:
        _loop_pass.reset(token)


def register_loop_cache(attn_wrapper, vllm_config) -> LoopCache:
    # Keep the tensor reference implementation importable without vLLM/NPU.
    from vllm_ascend.attention.sfa_v1 import Glm53LoopCache

    return Glm53LoopCache(attn_wrapper.mla_attn, vllm_config)


def _gather_bank(
    cache: tuple[torch.Tensor, torch.Tensor],
    block_table: torch.Tensor,
    requests: torch.Tensor,
    key_positions: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent, rope = cache
    block_size = latent.shape[1]
    valid = (key_positions >= 0) & (key_positions <= query_positions[:, None])
    logical_pages = key_positions.clamp_min(0) // block_size
    valid &= logical_pages < block_table.shape[1]

    pages = block_table[requests[:, None], logical_pages.clamp_max(block_table.shape[1] - 1)].long()
    valid &= (pages >= 0) & (pages < latent.shape[0])
    pages = torch.where(valid, pages, 0)
    offsets = key_positions.remainder(block_size)
    latent = latent[pages, offsets, 0].float().masked_fill(~valid[..., None], 0)
    rope = rope[pages, offsets, 0].float().masked_fill(~valid[..., None], 0)
    return latent, rope, valid


def joint_attention(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    global_cache: tuple[torch.Tensor, torch.Tensor],
    local_cache: tuple[torch.Tensor, torch.Tensor],
    global_table: torch.Tensor,
    local_table: torch.Tensor,
    token_requests: torch.Tensor,
    positions: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    window_size: int = LOOP_WINDOW_SIZE,
) -> torch.Tensor:
    """Attend to the sparse first pass and the sliding window of the second pass."""
    if query_latent.shape[0] == 0:
        return torch.empty_like(query_latent)

    window_offsets = torch.arange(1 - window_size, 1, device=positions.device)
    output = torch.empty_like(query_latent, memory_format=torch.contiguous_format)
    for start in range(0, query_latent.shape[0], _QUERY_TILE_SIZE):
        rows = slice(start, start + _QUERY_TILE_SIZE)
        requests = token_requests[rows]
        query_positions = positions[rows]
        global_latent, global_rope, global_valid = _gather_bank(
            global_cache, global_table, requests, topk_indices[rows].long(), query_positions
        )
        local_latent, local_rope, local_valid = _gather_bank(
            local_cache, local_table, requests, query_positions[:, None] + window_offsets, query_positions
        )

        # Equal token positions in the two banks represent different hidden
        # states and both participate in the same normalization.
        latent = torch.cat((global_latent, local_latent), dim=1)
        rope = torch.cat((global_rope, local_rope), dim=1)
        valid = torch.cat((global_valid, local_valid), dim=1).unsqueeze(1)
        scores = torch.bmm(query_latent[rows].float(), latent.transpose(1, 2))
        scores += torch.bmm(query_rope[rows].float(), rope.transpose(1, 2))
        scores *= scale
        scores.masked_fill_(~valid, torch.finfo(scores.dtype).min)
        probabilities = scores.softmax(dim=-1).masked_fill_(~valid, 0)
        output[rows] = torch.bmm(probabilities, latent)
    return output
