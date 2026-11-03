# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Graph agreement for an Embedding group when MC2 permits independent DP."""

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp as _dispatch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_embed_tp_group


def enable_target_embedding_inputs(runner):
    """Use MRV2's existing precomputed-embedding path for text targets."""
    config = runner.ascend_config
    if (
        getattr(config, "embedding_tp_dynamic_padding", False) is not True
        or config.finegrained_tp_config.embedding_tensor_parallel_size <= 1
        or runner.uses_inputs_embeds
        or not runner.is_first_pp_rank
    ):
        return
    state = runner.model_state
    state.inputs_embeds = torch.zeros(
        (state.max_num_tokens, state.inputs_embeds_size), dtype=state.dtype, device=state.device
    )
    state.dynamic_embedding_inputs = True
    runner.uses_inputs_embeds = True


def dispatch_cg_and_sync_dp(
    cudagraph_manager,
    num_reqs,
    num_tokens,
    uniform_token_count,
    dp_size,
    dp_rank,
    max_query_len=None,
    need_eager=False,
    num_active_loras=0,
    parallel_config=None,
    allow_ubatching=False,
    uniform_decode=False,
    dp_sync_state=None,
):
    config = get_ascend_config()
    kwargs = dict(
        max_query_len=max_query_len,
        need_eager=need_eager,
        num_active_loras=num_active_loras,
        parallel_config=parallel_config,
        allow_ubatching=allow_ubatching,
        uniform_decode=uniform_decode,
        dp_sync_state=dp_sync_state,
    )
    enabled = (
        config.embedding_tp_dynamic_padding is True
        and config.finegrained_tp_config.embedding_tensor_parallel_size > 1
        and should_skip_dp_coordination()
        and dp_size > 1
    )
    if not enabled or dp_sync_state is not None:
        # Non-MC2 paths still need the real full-DP token vector for MoE.
        # Reusing target agreement for draft prefill needs no new collective.
        return _dispatch(cudagraph_manager, num_reqs, num_tokens, uniform_token_count, dp_size, dp_rank, **kwargs)
    if allow_ubatching:
        raise ValueError("embedding_tp_dynamic_padding does not support microbatching.")

    local_desc, _ = _dispatch(cudagraph_manager, num_reqs, num_tokens, uniform_token_count, 1, 0, **kwargs)
    group = get_embed_tp_group()
    metadata = torch.zeros((5, group.world_size), dtype=torch.int64, device="cpu")
    metadata[:, group.rank_in_group] = torch.tensor(
        [num_tokens, local_desc.cg_mode.value, uniform_token_count or 0, max_query_len or -1, num_reqs]
    )
    dist.all_reduce(metadata, group=group.cpu_group)
    counts, modes, uniform_counts, query_bounds, request_counts = metadata
    agreed_uniform = int(uniform_counts[0])
    if agreed_uniform == 0 or not bool(torch.all(uniform_counts == agreed_uniform)):
        agreed_uniform = None
    agreed_query_bound = int(query_bounds.max()) if bool(torch.all(query_bounds > 0)) else None
    eager = int(modes.min()) == CUDAGraphMode.NONE.value
    agreed_num_reqs = int(request_counts.max())

    kwargs.update(max_query_len=agreed_query_bound, need_eager=eager)
    desc, _ = _dispatch(
        cudagraph_manager,
        num_reqs if eager else agreed_num_reqs,
        num_tokens if eager else int(counts.max()),
        agreed_uniform,
        1,
        0,
        **kwargs,
    )
    eager = desc.cg_mode == CUDAGraphMode.NONE
    if eager:
        # A combined bound may miss all graphs even if local candidates fit.
        kwargs["need_eager"] = True
        desc, _ = _dispatch(cudagraph_manager, num_reqs, num_tokens, agreed_uniform, 1, 0, **kwargs)
        # MC2 already permits a synthetic DP vector when coordination is
        # skipped. Repeat this group's counts so the local DP entry stays
        # accurate, without waiting for unrelated Embedding groups.
        across_dp = counts.to(torch.int32).repeat(dp_size // group.world_size)
    else:
        across_dp = torch.full((dp_size,), desc.num_tokens, dtype=torch.int32)
    state = DPSyncState(
        num_tokens_across_dp=across_dp,
        uniform_token_count=agreed_uniform,
        eager=eager,
        num_reqs=desc.num_reqs if not eager and desc.num_reqs is not None else agreed_num_reqs,
    )
    return desc, state
