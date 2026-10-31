"""Experimental P-only eager row alignment, using the existing DP agreement.

Draft execution disables the fine-grained knobs in main_only_scope, so this
adapter leaves draft dispatch and inputs untouched. No extra collectives.
"""

import functools
import json
import os
from dataclasses import replace

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as gpu_runner

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_last_aligned = None
_last_local = None
_last_eager = False
_step_dummy = False
_in_execute = False
_target_config = None
_minimum_alignment = 1
_mlp_input_rows = {}
_events = []
_finite_events = []


def enabled():
    if os.environ.get("FG_ROLE") != "p":
        return False
    ft = get_ascend_config().finegrained_tp_config
    return (
        ft.oproj_tensor_parallel_size > 1
        or ft.mlp_tensor_parallel_size > 1
        or (os.environ.get("P_UNIFORM_TARGET") == "1" and ft is _target_config)
    )


def eager_projection_step():
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return (
        enabled()
        and _in_execute
        and _last_eager
        and is_forward_context_available()
        and not getattr(get_forward_context(), "capturing", False)
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
    )


def install():
    # Import the compatibility patch before wrapping its final dispatch.
    from vllm_ascend.patch.worker.patch_v2 import patch_model_runner  # noqa: F401

    original_dispatch = gpu_runner.dispatch_cg_and_sync_dp

    @functools.wraps(original_dispatch)
    def dispatch(*args, **kwargs):
        global _last_aligned, _last_local, _last_eager
        desc, sync = original_dispatch(*args, **kwargs)
        if enabled():
            _last_local = kwargs.get("num_tokens", args[2] if len(args) > 2 else 0)
            _last_eager = sync is not None and sync.eager
        if enabled() and sync is not None and sync.eager:
            rows = sync.num_tokens_across_dp.tolist()
            count = max(max(rows), _minimum_alignment)
            if os.environ.get("P_LOCAL_PROJECTION_PADDING") != "1":
                desc = replace(desc, num_tokens=count, num_reqs=max(desc.num_reqs or 0, 1))
                sync = replace(sync, num_tokens_across_dp=torch.full_like(sync.num_tokens_across_dp, count))
            _last_aligned = count
            if len(_events) < 12:
                event = dict(
                    rows=rows,
                    padded=count,
                    local_descriptor=desc.num_tokens,
                    local_projection_padding=os.environ.get("P_LOCAL_PROJECTION_PADDING") == "1",
                    pid=os.getpid(),
                )
                _events.append(event)
                print("P_TARGET_EAGER_ALIGNMENT " + json.dumps(event), flush=True)
        return desc, sync

    gpu_runner.dispatch_cg_and_sync_dp = dispatch

    original_execute = NPUModelRunner.execute_model

    @functools.wraps(original_execute)
    def execute(self, *args, **kwargs):
        global _step_dummy, _in_execute, _target_config, _minimum_alignment
        speculative = self.vllm_config.speculative_config
        _minimum_alignment = speculative.num_speculative_tokens + 1 if speculative else 1
        if _target_config is None:
            _target_config = get_ascend_config().finegrained_tp_config
        previous = _step_dummy
        previous_execute = _in_execute
        _in_execute = True
        _step_dummy = kwargs.get("dummy_run", args[2] if len(args) > 2 else False) or kwargs.get("is_profile", False)
        try:
            return original_execute(self, *args, **kwargs)
        finally:
            _step_dummy = previous
            _in_execute = previous_execute

    NPUModelRunner.execute_model = execute

    original_check = NPUModelRunner._check_finegrained_tp_graph_step

    @functools.wraps(original_check)
    def check(self, mode):
        if enabled() and mode == CUDAGraphMode.NONE:
            if _last_aligned is None:
                raise RuntimeError("P target eager step has no DP row agreement")
            return
        return original_check(self, mode)

    NPUModelRunner._check_finegrained_tp_graph_step = check

    original_prepare = NPUModelRunner._prepare_inputs_impl

    @functools.wraps(original_prepare)
    def prepare(self, scheduler_output, req_state, desc):
        if os.environ.get("P_LOCAL_PROJECTION_PADDING") == "1":
            return original_prepare(self, scheduler_output, req_state, desc)
        if enabled() and desc.cg_mode == CUDAGraphMode.NONE:
            if desc.num_tokens != _last_aligned:
                raise RuntimeError("P target descriptor lost its uniform row count")
            n = req_state.num_tokens
            end = desc.num_tokens
            if end > self.max_num_tokens:
                raise RuntimeError("P target aligned rows exceed input capacity")
            self.input_buffers.is_padding[:n].fill_(False)
            self.input_buffers.is_padding[n:end].fill_(True)
            self.input_buffers.input_ids[n:end].zero_()
            self.input_buffers.positions[n:end].zero_()
        return original_prepare(self, scheduler_output, req_state, desc)

    NPUModelRunner._prepare_inputs_impl = prepare

    if any(os.environ.get(key) == "1" for key in ("P_FINITE_PROBE", "P_ZERO_FG_PADDING", "P_REFERENCE_PROBE")):
        from vllm_ascend.ops.linear_op import MLPColumnParallelOp, MLPRowParallelOp, OProjRowParallelOp

        def wrap_apply(original_apply):
            def apply(self, tensor):
                eager = eager_projection_step()
                original_rows = tensor.shape[0]
                if eager and isinstance(self, MLPColumnParallelOp):
                    _mlp_input_rows[self.prefix.rsplit(".", 1)[0]] = original_rows
                if eager and os.environ.get("P_FORCE_FG_ROWS") == "1":
                    if original_rows > _last_aligned:
                        raise RuntimeError("Projection rows exceed DP agreement")
                    if original_rows < _last_aligned:
                        print(
                            "P_PROJECTION_ROW_PADDING "
                            + json.dumps(
                                dict(
                                    prefix=self.prefix,
                                    pid=os.getpid(),
                                    original_rows=original_rows,
                                    padded_rows=_last_aligned,
                                )
                            ),
                            flush=True,
                        )
                        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, _last_aligned - original_rows))
                watch = eager and _last_local is not None and 0 < _last_local < 1024
                n = 0 if _step_dummy else min(_last_local or 0, tensor.shape[0])
                event = None
                if watch and os.environ.get("P_FINITE_PROBE") == "1":
                    valid_finite = bool(torch.isfinite(tensor[:n]).all().item())
                    pad_finite = bool(torch.isfinite(tensor[n:]).all().item())
                    event = dict(
                        prefix=self.prefix,
                        valid_rows=n,
                        total_rows=tensor.shape[0],
                        valid_input_finite=valid_finite,
                        padding_input_finite=pad_finite,
                        pid=os.getpid(),
                    )
                if eager and os.environ.get("P_ZERO_FG_PADDING") == "1" and n < tensor.shape[0]:
                    tensor = tensor.clone()
                    tensor[n:].zero_()
                result = original_apply(self, tensor)
                if eager and os.environ.get("P_REFERENCE_PROBE") == "1":
                    from p_projection_reference_1031 import check as reference_check

                    reference_check(self, tensor, result[0], n)
                if event is not None:
                    event["valid_output_finite"] = bool(torch.isfinite(result[0][:n]).all().item())
                    if len(_finite_events) < 8 or not event["valid_input_finite"] or not event["valid_output_finite"]:
                        if len(_finite_events) < 256:
                            _finite_events.append(event)
                            print("P_TARGET_FINITE_PROBE " + json.dumps(event), flush=True)
                if eager and os.environ.get("P_FORCE_FG_ROWS") == "1" and isinstance(self, OProjRowParallelOp):
                    result = (result[0][:original_rows], result[1])
                return result

            return apply

        for cls in (OProjRowParallelOp, MLPColumnParallelOp):
            cls.apply_impl = wrap_apply(cls.apply_impl)
        original_mlp_row = MLPRowParallelOp.apply_impl

        @functools.wraps(original_mlp_row)
        def mlp_row(self, tensor):
            result = original_mlp_row(self, tensor)
            if eager_projection_step() and os.environ.get("P_FORCE_FG_ROWS") == "1":
                rows = _mlp_input_rows[self.prefix.rsplit(".", 1)[0]]
                result = (result[0][:rows], result[1])
            return result

        MLPRowParallelOp.apply_impl = mlp_row


def evidence():
    return dict(
        events=_events,
        last_aligned_rows=_last_aligned,
        local_projection_padding=os.environ.get("P_LOCAL_PROJECTION_PADDING") == "1",
        finite_events=_finite_events,
        draft_adapter="main_only_scope",
        additional_collectives=0,
    )
