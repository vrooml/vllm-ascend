# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.glm53_loop import build_loop_swa_indices, pack_loop_attention_partial


def test_loop_swa_indices_keep_valid_keys_left_aligned():
    positions = torch.tensor([0, 127, 128, 130])
    indices = build_loop_swa_indices(positions)
    assert indices.shape == (4, 1, 128)
    assert indices.dtype == torch.int32
    for row, position in zip(indices[:, 0], positions.tolist(), strict=True):
        keys = torch.arange(max(position - 127, 0), position + 1, dtype=torch.int32)
        torch.testing.assert_close(row[: len(keys)], keys)
        torch.testing.assert_close(row[len(keys) :], torch.full_like(row[len(keys) :], -1))


def _pack_cpu_partial(scores, values):
    maxima = scores.amax(dim=-1)
    maxima = maxima.masked_fill(torch.isneginf(maxima), 0)
    sums = (scores - maxima.unsqueeze(-1)).exp().sum(dim=-1)
    # SFA uses max = sum = 0 for empty rows. Also check that a NaN partial
    # output from an empty softmax cannot propagate through the merge.
    output = (scores.softmax(dim=-1) @ values.float()).to(values.dtype)
    return pack_loop_attention_partial(output, maxima.unsqueeze(0), sums.unsqueeze(0))


def _update_lse_cpu(outputs, lses):
    normalized_weights = lses.softmax(dim=0)
    return (outputs * normalized_weights.unsqueeze(-1)).sum(dim=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_lse_partials_match_one_softmax(dtype):
    generator = torch.Generator().manual_seed(9)
    scores = torch.randn(2, 3, 4, 5, generator=generator) * 4
    values = torch.randn(2, 3, 5, 6, generator=generator).to(dtype)
    # A query with no global selection still has causal local keys. Also
    # exercise unequal bank lengths and nontrivial [token, head] row order.
    scores[0, 0] = -torch.inf
    scores[1, 1, :, 2:] = -torch.inf

    global_partial = _pack_cpu_partial(scores[0], values[0])
    local_partial = _pack_cpu_partial(scores[1], values[1])
    packed_outputs = torch.stack((global_partial[0], local_partial[0]))
    packed_lses = torch.stack((global_partial[1], local_partial[1]))
    assert packed_outputs.shape == (2, 12, 6)
    assert packed_lses.shape == (2, 12)
    assert packed_outputs.dtype == packed_lses.dtype == torch.float32
    assert torch.isneginf(packed_lses[0, :4]).all()
    torch.testing.assert_close(packed_outputs[0, :4], torch.zeros((4, 6)))

    result = _update_lse_cpu(packed_outputs, packed_lses).view(3, 4, 6)
    joint_scores = torch.cat(scores.unbind(), dim=-1).double()
    joint_values = torch.cat(values.unbind(), dim=-2).double()
    expected = (joint_scores.softmax(dim=-1) @ joint_values).float()
    tolerance = 0.01 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)


def _loop_attention_cpu(
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
):
    partial_outputs = []
    partial_lses = []
    local_indices = build_loop_swa_indices(positions).squeeze(1)
    for (latent, rope), table, keys in (
        (global_cache, global_table, topk),
        (local_cache, local_table, local_indices),
    ):
        block_size = latent.shape[1]
        valid = (keys >= 0) & (keys <= positions[:, None])
        keys = torch.where(valid, keys, 0).long()
        physical_page = table[token_requests[:, None], keys // block_size]
        latent_keys = latent[physical_page, keys % block_size, 0]
        rotary_keys = rope[physical_page, keys % block_size, 0]
        scores = query_latent.float() @ latent_keys.float().transpose(1, 2)
        scores += query_rope.float() @ rotary_keys.float().transpose(1, 2)
        scores = (scores * scale).masked_fill(~valid.unsqueeze(1), -torch.inf)
        output, lse = _pack_cpu_partial(scores, latent_keys)
        partial_outputs.append(output)
        partial_lses.append(lse)
    return _update_lse_cpu(torch.stack(partial_outputs), torch.stack(partial_lses)).view_as(query_latent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_lse_uses_both_banks_and_request_page_tables(dtype):
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

    output = _loop_attention_cpu(
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

    tolerance = 0.01 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(output, torch.stack(expected).float(), rtol=tolerance, atol=tolerance)


def test_update_lse_keeps_overlapping_positions_as_two_entries():
    query = torch.zeros(1, 1, 1)
    global_cache = (torch.full((1, 1, 1, 1), 2.0), query.reshape(1, 1, 1, 1))
    local_cache = (torch.full((1, 1, 1, 1), 10.0), query.reshape(1, 1, 1, 1))
    page_table = torch.zeros(1, 1, dtype=torch.int32)
    position = torch.zeros(1, dtype=torch.int64)

    output = _loop_attention_cpu(
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
