# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated A3 W8A8 MegaMoE shape probe, without a model or config bypass.

Run one shape per fresh torchrun process. This calls the installed CANN API
directly to distinguish its shape contract from the vLLM configuration gate.
"""

import argparse
import datetime
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe


def emit(stage, **values):
    print(json.dumps({"rank": int(os.environ.get("RANK", -1)), "stage": stage, **values}), flush=True)


def weight_for(expert, hidden, intermediate):
    generator = torch.Generator(device="cpu").manual_seed(20261015 + expert)
    w13 = torch.randint(-16, 17, (2 * intermediate, hidden), dtype=torch.int8, generator=generator)
    w2 = torch.randint(-16, 17, (hidden, intermediate), dtype=torch.int8, generator=generator)
    s13 = torch.full((2 * intermediate,), 0.002, dtype=torch.float32)
    s2 = torch.full((hidden,), 0.002, dtype=torch.float32)
    return w13, w2, s13, s2


def packed_scale(scale):
    return torch.from_numpy(scale.numpy().view(np.int32).astype(np.int64)).npu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--compare-padding", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    # Match the production platform setup. Otherwise format_cast silently
    # returns ND and the test never reaches MegaMoE's dimension validation.
    torch.npu.config.allow_internal_format = True
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True, parents=True)
    record = {"args": vars(args), "rank": rank, "world": world, "status": "started"}
    context = None
    try:
        emit("initializing", args=vars(args))
        dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=120))
        group = dist.new_group(backend="hccl", timeout=datetime.timedelta(seconds=120))
        group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        emit("group_initialized")
        local_experts = args.experts // world
        w13, w2, s13, s2 = [], [], [], []
        for expert in range(rank * local_experts, (rank + 1) * local_experts):
            first, second, first_scale, second_scale = weight_for(expert, args.hidden, args.intermediate)
            w13.append(torch_npu.npu_format_cast(first.T.contiguous().npu(), 29))
            w2.append(torch_npu.npu_format_cast(second.T.contiguous().npu(), 29))
            s13.append(packed_scale(first_scale))
            s2.append(packed_scale(second_scale))
        formats = [int(torch_npu.get_npu_format(weight)) for weight in (*w13, *w2)]
        if any(fmt != 29 for fmt in formats):
            raise RuntimeError(f"expected FRACTAL_NZ weights, got {formats}")
        emit("weights_prepared", formats=formats)
        generator = torch.Generator(device="cpu").manual_seed(23 + rank)
        x = torch.randn((args.tokens, args.hidden), generator=generator).to(torch.bfloat16).npu()
        # Cover the entire EP domain, including experts owned by the last rank.
        # A low consecutive-ID pattern would leave most EP16 ranks without FFN work.
        ids_cpu = torch.rand((args.tokens, args.experts), generator=generator).topk(args.topk, dim=-1).indices
        expected_counts = torch.bincount(ids_cpu.flatten(), minlength=args.experts).to(torch.int32).npu()
        dist.all_reduce(expected_counts)
        expected_local_counts = expected_counts[rank * local_experts : (rank + 1) * local_experts].cpu()
        ids = ids_cpu.to(torch.int32).npu()
        probabilities = torch.full((args.tokens, args.topk), 1.0 / args.topk, dtype=torch.float32).npu()
        context = get_symm_buffer_for_mega_moe(
            group,
            args.experts,
            args.tokens,
            args.topk,
            args.hidden,
            2 * args.intermediate,
            max_recv_token_num=args.tokens * world * min(args.topk, local_experts),
            dispatch_quant_mode=2,
            dispatch_quant_out_dtype=258,
        )
        emit("buffer_allocated", ccl_buffer_size=int(context.ccl_buffer_size))
        torch.npu.synchronize()
        start = time.monotonic()

        def fused(first, second, first_scale, second_scale):
            return mega_moe(
                x,
                ids,
                probabilities,
                first,
                second,
                context,
                l1_weights_sf=first_scale,
                l2_weights_sf=second_scale,
                weight1_type=258,
                weight2_type=258,
                activation="swiglu",
                activation_clamp=10.0,
            )

        y, counts = fused(w13, w2, s13, s2)
        torch.npu.synchronize()
        elapsed = time.monotonic() - start
        actual = y.cpu().float()
        torch.testing.assert_close(counts.cpu().to(torch.int32), expected_local_counts, rtol=0, atol=0)
        if not bool(torch.isfinite(actual).all()):
            raise RuntimeError("nonfinite native output")
        if not bool(actual.abs().max() > 0):
            raise RuntimeError("unexpected all-zero native output")
        record.update(
            status="native_shape_executed",
            elapsed_seconds=elapsed,
            output_shape=list(y.shape),
            output_abs_max=float(actual.abs().max()),
            expert_counts=counts.cpu().tolist(),
            expected_local_tokens=int(expected_local_counts.sum()),
            numerical_equivalence_verified=False,
            cann_build=getattr(torch.version, "cann", None),
        )
        emit("native_shape_executed", **{k: v for k, v in record.items() if k not in {"rank", "args"}})
        if args.compare_padding:
            padded = (args.intermediate + 511) // 512 * 512
            pw13, pw2, ps13, ps2 = [], [], [], []
            for expert in range(rank * local_experts, (rank + 1) * local_experts):
                first, second, first_scale, second_scale = weight_for(expert, args.hidden, args.intermediate)
                first_padded = torch.zeros((2 * padded, args.hidden), dtype=torch.int8)
                first_padded[: args.intermediate] = first[: args.intermediate]
                first_padded[padded : padded + args.intermediate] = first[args.intermediate :]
                second_padded = torch.zeros((args.hidden, padded), dtype=torch.int8)
                second_padded[:, : args.intermediate] = second
                scale_padded = torch.ones((2 * padded,), dtype=torch.float32)
                scale_padded[: args.intermediate] = first_scale[: args.intermediate]
                scale_padded[padded : padded + args.intermediate] = first_scale[args.intermediate :]
                pw13.append(torch_npu.npu_format_cast(first_padded.T.contiguous().npu(), 29))
                pw2.append(torch_npu.npu_format_cast(second_padded.T.contiguous().npu(), 29))
                ps13.append(packed_scale(scale_padded))
                ps2.append(packed_scale(second_scale))
            reference, reference_counts = fused(pw13, pw2, ps13, ps2)
            torch.npu.synchronize()
            expected = reference.cpu().float()
            difference = actual - expected
            rmse = float((difference.square().mean() / expected.square().mean()).sqrt())
            cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
            counts_equal = bool(torch.equal(counts.cpu(), reference_counts.cpu()))
            record["padding_comparison"] = dict(
                padded_intermediate=padded,
                relative_rmse=rmse,
                cosine=cosine,
                max_abs_error=float(difference.abs().max()),
                counts_equal=counts_equal,
            )
            emit("padding_comparison", **record["padding_comparison"])
            if not (rmse < 0.01 and cosine > 0.9999 and counts_equal):
                raise AssertionError(f"original/padded output mismatch: {record['padding_comparison']}")
            record["numerical_equivalence_verified"] = True
        if args.graph:
            for _ in range(3):
                fused(w13, w2, s13, s2)
            torch.npu.synchronize()
            dist.barrier()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured, captured_counts = fused(w13, w2, s13, s2)
            for _ in range(3):
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(captured.cpu().float(), actual, rtol=0, atol=0)
            record["graph_replays_equal"] = 3
            emit("graph_replays_verified", replays=3)
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
        emit("failed", error=str(error))
        raise
    finally:
        (output / f"rank{rank}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        # Let the isolated process own communicator teardown, including failed
        # kernel/tiling paths; no unrelated process or device reset is used.


if __name__ == "__main__":
    main()
