# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.attention.glm53_loop import build_loop_swa_indices, pack_loop_attention_partial
from vllm_ascend.attention.utils import merge_prefill_chunks
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Exercises the A3 paged SFA LSE ABI")
def test_glm53_loop_paged_sfa_attention_update():
    enable_custom_op()
    generator = torch.Generator().manual_seed(928)
    page_size, num_pages, heads = 128, 6, 8
    scale = 1 / 24
    positions = torch.tensor([3, 4, 126, 127, 128, 129, 259], dtype=torch.int32)
    request_ids = (0, 0, 1, 1, 1, 1, 2)
    num_tokens = positions.numel()
    # Two partial prefills and one decode, in TND row order.
    q_lens = torch.tensor([2, 6, 7], dtype=torch.int32).npu()
    kv_lens = torch.tensor([5, 130, 260], dtype=torch.int32).npu()
    query = torch.randn(num_tokens, heads, 512, dtype=torch.bfloat16, generator=generator)
    query_rope = torch.randn(num_tokens, heads, 64, dtype=torch.bfloat16, generator=generator)
    query[3:5, :, 0] = 24
    query_npu, query_rope_npu = query.npu(), query_rope.npu()

    global_table = torch.tensor([[3, 0, 0], [1, 4, 0], [2, 5, 0]], dtype=torch.int32)
    local_table = torch.tensor([[2, 0, 0], [5, 1, 0], [0, 4, 3]], dtype=torch.int32)
    global_keys = ([], [0, 4], [1, 126], [5, 127], [0, 128], [9, 127, 129], list(range(260)))
    global_indices = torch.full((num_tokens, 1, 2048), -1, dtype=torch.int32)
    for row, keys in enumerate(global_keys):
        global_indices[row, 0, : len(keys)] = torch.tensor(keys, dtype=torch.int32)
    local_indices = build_loop_swa_indices(positions.npu())

    def make_bank():
        return tuple(
            torch.randn(num_pages, page_size, 1, dim, dtype=torch.bfloat16, generator=generator) for dim in (512, 64)
        )

    global_bank, local_bank = make_bank(), make_bank()
    # Token zero has a high score for positions 127 and 128, so losing it one
    # position early or retaining it one position late changes the result.
    local_bank[0][local_table[1, 0], 0, 0, 0] = 12

    def partial(bank, table, indices):
        latent, rope = (component.npu() for component in bank)
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            query=query_npu,
            key=latent,
            value=latent,
            sparse_indices=indices,
            scale_value=scale,
            sparse_block_size=1,
            block_table=table.npu(),
            actual_seq_lengths_query=q_lens,
            actual_seq_lengths_kv=kv_lens,
            query_rope=query_rope_npu,
            key_rope=rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )

    local_rows, local_lse = pack_loop_attention_partial(*partial(local_bank, local_table, local_indices))
    global_output, global_max, global_sum = partial(global_bank, global_table, global_indices.npu())
    assert torch.count_nonzero(global_sum[:, 0].cpu()) == 0
    global_rows, global_lse = pack_loop_attention_partial(global_output, global_max, global_sum)
    merged, _, _ = merge_prefill_chunks([local_rows, global_rows], [local_lse, global_lse], None, return_lse=False)

    expected = []
    for row, (request, position) in enumerate(zip(request_ids, positions.tolist(), strict=True)):
        local_keys = range(max(0, position - 127), position + 1)
        latent_parts, rope_parts = [], []
        for bank, table, keys in ((global_bank, global_table, global_keys[row]), (local_bank, local_table, local_keys)):
            keys = torch.tensor(list(keys), dtype=torch.long)
            pages = table[request, keys // page_size].long()
            offsets = keys % page_size
            latent_parts.append(bank[0][pages, offsets, 0])
            rope_parts.append(bank[1][pages, offsets, 0])
        latent = torch.cat(latent_parts).double()
        rope = torch.cat(rope_parts).double()
        logits = (query[row].double() @ latent.T + query_rope[row].double() @ rope.T) * scale
        expected.append(logits.softmax(-1) @ latent)

    expected = torch.stack(expected).float()
    torch.testing.assert_close(merged.cpu().reshape_as(expected), expected, atol=0.03, rtol=0.01)
