# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import torch

from vllm_ascend.attention.glm53_loop import get_loop_pass
from vllm_ascend.models.glm54_loop import Glm53LoopModel


def test_forward_reuses_layers_and_keeps_each_first_pass_topk_group():
    scratch = torch.empty((2, 3), dtype=torch.int32)
    loop_caches = tuple(SimpleNamespace(layer_name=f"local.{index}") for index in range(26, 54))
    calls = []
    segment_inputs = []

    class Layer:
        def __init__(self, index):
            self.index = index
            self.self_attn = SimpleNamespace(mla_attn=SimpleNamespace(topk_indices_buffer=scratch))

        def __call__(self, positions, hidden_states, residual, scaling):
            active_pass = get_loop_pass()
            calls.append((self.index, active_pass is not None))
            if active_pass is not None:
                local_index = self.index - 26
                assert active_pass.local_cache is loop_caches[local_index]
                expected_topk = torch.full_like(scratch, 26 + 4 * (local_index // 4))
                torch.testing.assert_close(active_pass.topk_indices, expected_topk)
            elif 26 <= self.index < 54 and (self.index - 26) % 4 == 0:
                scratch.fill_(self.index)
            if residual is None:
                residual = hidden_states
                segment_inputs.append((self.index, hidden_states[0].item()))
            else:
                residual = hidden_states + residual
            return torch.full_like(hidden_states, 2), residual

    model = SimpleNamespace(
        config=SimpleNamespace(),
        layers=[Layer(index) for index in range(78)],
        loop_local_caches=loop_caches,
        loop_gates=(0.25, 0.5),
        norm=lambda hidden_states, residual: (hidden_states + residual, residual),
    )
    output = Glm53LoopModel.forward(model, None, torch.arange(2), inputs_embeds=torch.full((2,), 3.0))

    assert calls == [
        *((index, False) for index in range(54)),
        *((index, True) for index in range(26, 54)),
        *((index, False) for index in range(54, 78)),
    ]
    assert segment_inputs == [(0, 3), (26, 55), (26, 69), (54, 97)]
    assert get_loop_pass() is None
    torch.testing.assert_close(output, torch.full((2,), 145.0))
