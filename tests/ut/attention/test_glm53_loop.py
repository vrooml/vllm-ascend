# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.glm53_loop import joint_attention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_attention_uses_both_banks_and_request_page_tables(dtype):
    generator = torch.Generator().manual_seed(7)
    block_size = 4
    num_positions = 140
    num_blocks = num_positions // block_size
    num_pages = 2 * num_blocks
    global_table = torch.randperm(num_pages, generator=generator).view(2, num_blocks)
    local_table = torch.randperm(num_pages, generator=generator).view(2, num_blocks)
    global_cache = (
        torch.randn(num_pages, block_size, 1, 3, generator=generator).to(dtype),
        torch.randn(num_pages, block_size, 1, 2, generator=generator).to(dtype),
    )
    local_cache = (
        torch.randn(num_pages, block_size, 1, 3, generator=generator).to(dtype),
        torch.randn(num_pages, block_size, 1, 2, generator=generator).to(dtype),
    )
    token_requests = torch.tensor([0, 0, 1, 1])
    positions = torch.tensor([2, 3, 129, 130])
    topk = torch.tensor([[2, 0, -1, 3], [-1, -1, -1, -1], [0, 1, 2, 129], [2, 3, 130, 131]])
    query_latent = torch.randn(4, 2, 3, generator=generator).to(dtype)
    query_rope = torch.randn(4, 2, 2, generator=generator).to(dtype)
    scale = 0.625

    output = joint_attention(
        query_latent,
        query_rope,
        global_cache,
        local_cache,
        global_table,
        local_table,
        token_requests,
        positions,
        topk,
        scale,
    )

    expected = []
    for token, (request, position) in enumerate(zip(token_requests.tolist(), positions.tolist())):
        global_positions = [key for key in topk[token].tolist() if 0 <= key <= position]
        local_positions = range(max(0, position - 127), position + 1)
        latents, rotary = [], []
        for table, bank, keys in (
            (global_table, global_cache, global_positions),
            (local_table, local_cache, local_positions),
        ):
            for key in keys:
                page = table[request, key // block_size]
                latents.append(bank[0][page, key % block_size, 0])
                rotary.append(bank[1][page, key % block_size, 0])
        latents = torch.stack(latents).double()
        rotary = torch.stack(rotary).double()
        logits = query_latent[token].double() @ latents.T + query_rope[token].double() @ rotary.T
        expected.append((logits * scale).softmax(dim=-1) @ latents)

    torch.testing.assert_close(output, torch.stack(expected).to(output.dtype))


def test_joint_attention_keeps_overlapping_positions_as_two_entries():
    query = torch.zeros(1, 1, 1)
    global_cache = (torch.full((1, 1, 1, 1), 2.0), query.reshape(1, 1, 1, 1))
    local_cache = (torch.full((1, 1, 1, 1), 10.0), query.reshape(1, 1, 1, 1))
    page_table = torch.zeros(1, 1, dtype=torch.int32)
    position = torch.zeros(1, dtype=torch.int64)

    output = joint_attention(
        query,
        query,
        global_cache,
        local_cache,
        page_table,
        page_table,
        position,
        position,
        page_table,
        1.0,
    )

    torch.testing.assert_close(output, torch.full_like(query, 6.0))
