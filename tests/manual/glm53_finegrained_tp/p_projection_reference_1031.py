"""Target-only local reference; no additional collective and no output edits."""

import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open


def attach(worker):
    runner = worker.model_runner
    path = Path(runner.vllm_config.model_config.model)
    index = json.loads(next(path.glob("*.safetensors.index.json")).read_text())["weight_map"]
    result = []
    for name, layer in runner.model.named_modules():
        match = re.search(r"layers\.(0|3|44)\.self_attn\.o_proj$", name)
        if match is None:
            continue
        number = int(match.group(1))
        op = getattr(layer, "custom_op", None)
        if op is None or type(op).__name__ != "OProjRowParallelOp":
            continue
        keys = [key for key in index if re.search(rf"layers\.{number}\.self_attn\.o_proj\.weight$", key)]
        if len(keys) != 1:
            raise RuntimeError(keys)
        key = keys[0]
        with safe_open(str(path / index[key]), framework="pt", device="cpu") as source:
            weight = source.get_tensor(key)
        if weight.dtype != torch.bfloat16:
            raise RuntimeError(f"Expected BF16 OProj, got {weight.dtype}")
        width = weight.shape[1] // op.tp_size
        expected = weight[:, op.tp_rank * width : (op.tp_rank + 1) * width].contiguous()
        exact = torch.equal(expected, layer.weight.detach().cpu().contiguous())
        if not exact:
            raise RuntimeError(f"OProj weight partition mismatch at {key}")
        op._p_reference_weight = weight.to(layer.weight.device)
        op._p_reference_checks = []
        result.append(dict(prefix=op.prefix, checkpoint=key, weight_exact=exact, shape=list(weight.shape)))
    if len(result) != 3:
        raise RuntimeError(f"Expected three reference projections: {result}")
    return result


def check(op, x, y, n):
    if not hasattr(op, "_p_reference_weight") or len(op._p_reference_checks) >= 4 or n <= 0:
        return
    rows = sorted(set([0, min(1, n - 1), n - 1]))
    inp = x[rows]
    weight = op._p_reference_weight
    width = weight.shape[1] // 4
    with torch.no_grad():
        parts = [
            torch.nn.functional.linear(inp[:, i * width : (i + 1) * width], weight[:, i * width : (i + 1) * width])
            for i in range(4)
        ]
        chunk = (parts[0] + parts[2]) + (parts[1] + parts[3])
        full = torch.nn.functional.linear(inp, weight)
        actual = y[rows]

        def metrics(a, b):
            delta = a.float() - b.float()
            return dict(
                max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / b.float().norm().clamp_min(1e-20)),
                finite=bool(torch.isfinite(a).all()),
            )

        event = dict(
            prefix=op.prefix,
            pid=os.getpid(),
            rows=rows,
            input_shape=list(x.shape),
            output_shape=list(y.shape),
            actual_vs_chunks=metrics(actual, chunk),
            actual_vs_full=metrics(actual, full),
        )
    op._p_reference_checks.append(event)
    print("P_PROJECTION_REFERENCE " + json.dumps(event), flush=True)


def evidence(worker):
    return [
        event
        for module in worker.model_runner.model.modules()
        for event in getattr(getattr(module, "custom_op", None), "_p_reference_checks", [])
    ]
