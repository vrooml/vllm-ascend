# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.worker.v2 import embedding_dp_utils as module
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.mark.parametrize(
    "enabled,prepared,first", [(True, False, True), (False, False, True), (True, True, True), (True, False, False)]
)
def test_text_target_embedding_buffer(enabled, prepared, first):
    state = SimpleNamespace(max_num_tokens=8, inputs_embeds_size=4, dtype=torch.float32, device="cpu")
    sentinel = object()
    state.inputs_embeds = sentinel
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(
            embedding_tp_dynamic_padding=enabled,
            finegrained_tp_config=SimpleNamespace(embedding_tensor_parallel_size=8),
        ),
        uses_inputs_embeds=prepared,
        is_first_pp_rank=first,
        model_state=state,
    )
    module.enable_target_embedding_inputs(runner)
    if enabled and not prepared and first:
        assert runner.uses_inputs_embeds
        assert state.dynamic_embedding_inputs
        assert state.inputs_embeds.shape == (8, 4)
        assert torch.count_nonzero(state.inputs_embeds) == 0
    else:
        assert state.inputs_embeds is sentinel
        assert runner.uses_inputs_embeds is prepared


@pytest.mark.parametrize("num_tokens", [0, 3])
def test_text_target_capture_passes_prepared_embeddings(num_tokens):
    state = object.__new__(AscendModelState)
    state.supports_mm_inputs = False
    state.prompt_embeds_state = None
    state.rope_state = None
    state.dynamic_embedding_inputs = True
    state.inputs_embeds = torch.zeros(8, 4)
    state.prepare_engram_dummy_inputs = lambda reqs, tokens: {}
    inputs = state.prepare_dummy_inputs(1, num_tokens)
    assert inputs["inputs_embeds"].shape == (num_tokens, 4)
    assert inputs["inputs_embeds"].data_ptr() == state.inputs_embeds[:num_tokens].data_ptr()


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("eager_peer", [False, True])
def test_embedding_group_dispatch(monkeypatch, rank, eager_peer):
    """Groups agree independently; the local full-DP vector entry stays valid."""
    group = SimpleNamespace(world_size=4, rank_in_group=rank % 4, cpu_group=object())
    counts = [6, 12, 18, 24] if rank < 4 else [30, 36, 42, 48]
    requests = [n // 6 for n in counts]
    modes = [CUDAGraphMode.FULL.value] * 4
    if eager_peer:
        modes[0] = CUDAGraphMode.NONE.value
    config = SimpleNamespace(
        embedding_tp_dynamic_padding=True,
        finegrained_tp_config=SimpleNamespace(embedding_tensor_parallel_size=4),
    )
    monkeypatch.setattr(module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(module, "get_embed_tp_group", lambda: group)
    monkeypatch.setattr(module, "should_skip_dp_coordination", lambda: True)

    def reduce(metadata, group):
        assert group is group_cpu
        assert metadata[:, rank % 4].tolist() == [counts[rank % 4], CUDAGraphMode.FULL.value, 6, 6, requests[rank % 4]]
        metadata[:] = torch.tensor([counts, modes, [6] * 4, [6] * 4, requests])

    def dispatch(manager, reqs, tokens, uniform, dp_size, dp_rank, **kwargs):
        assert dp_size == 1 and dp_rank == 0
        return SimpleNamespace(
            cg_mode=CUDAGraphMode.NONE if kwargs.get("need_eager") else CUDAGraphMode.FULL,
            num_tokens=tokens,
            num_reqs=reqs,
        ), None

    group_cpu = group.cpu_group
    reduce_mock = Mock(side_effect=reduce)
    monkeypatch.setattr(module.dist, "all_reduce", reduce_mock)
    monkeypatch.setattr(module, "_dispatch", dispatch)
    desc, sync = module.dispatch_cg_and_sync_dp(
        object(), requests[rank % 4], counts[rank % 4], 6, 8, rank, max_query_len=6
    )
    reduce_mock.assert_called_once()
    assert sync.num_tokens_across_dp[rank] == (counts[rank % 4] if eager_peer else max(counts))
    assert desc.num_tokens == (counts[rank % 4] if eager_peer else max(counts))
    assert sync.eager is eager_peer
    assert sync.num_tokens_across_dp.dtype == torch.int32
    assert sync.uniform_token_count == 6


@pytest.mark.parametrize("enabled,skip,reuse", [(False, True, None), (True, False, None), (True, True, object())])
def test_embedding_dispatch_preserves_other_paths(monkeypatch, enabled, skip, reuse):
    config = SimpleNamespace(
        embedding_tp_dynamic_padding=enabled,
        finegrained_tp_config=SimpleNamespace(embedding_tensor_parallel_size=4),
    )
    monkeypatch.setattr(module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(module, "should_skip_dp_coordination", lambda: skip)
    original = Mock(return_value=(object(), object()))
    monkeypatch.setattr(module, "_dispatch", original)
    monkeypatch.setattr(module, "get_embed_tp_group", Mock(side_effect=AssertionError("No subgroup work expected")))
    assert module.dispatch_cg_and_sync_dp(None, 2, 12, 6, 8, 1, dp_sync_state=reuse) == original.return_value
    original.assert_called_once()
