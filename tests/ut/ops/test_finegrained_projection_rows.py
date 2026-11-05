# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Execute projection methods on CPU with ordered, emulated collectives."""

import ast
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from pathlib import Path
from threading import Condition
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
LINEAR = "vllm_ascend/ops/linear_op.py"
CONTEXT = "vllm_ascend/ascend_forward_context.py"


def load(path, names, scope, owner=None):
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    if owner is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), scope)


class World:
    def __init__(self, size):
        self.size = size
        self.condition = Condition()
        self.steps = [0] * size
        self.calls = {}

    def exchange(self, kind, rank, tensor):
        with self.condition:
            step = self.steps[rank]
            self.steps[rank] += 1
            call = self.calls.setdefault(step, {"kind": kind, "inputs": {}})
            assert call["kind"] == kind
            call["inputs"][rank] = tensor.clone()
            if len(call["inputs"]) == self.size:
                values = [call["inputs"][r] for r in range(self.size)]
                if kind == "all_gather":
                    call["outputs"] = [torch.cat(values)] * self.size
                elif kind == "reduce_scatter":
                    call["outputs"] = torch.stack(values).sum(0).chunk(self.size)
                else:
                    chunks = [value.chunk(self.size) for value in values]
                    call["outputs"] = [torch.cat([parts[r] for parts in chunks]) for r in range(self.size)]
                self.condition.notify_all()
            assert self.condition.wait_for(lambda: "outputs" in call, timeout=10), "Collective desynchronized"
            return call["outputs"][rank].clone()


class Group:
    def __init__(self, world, rank):
        self.world = world
        self.world_size = world.size
        self.rank_in_group = rank
        self.device_group = self

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        return self.world.exchange("all_gather", self.rank_in_group, tensor)

    def reduce_scatter(self, tensor, dim=0):
        assert dim == 0
        return self.world.exchange("reduce_scatter", self.rank_in_group, tensor)


def scope_for_projection():
    context = ContextVar("test_forward_context")
    group = ContextVar("test_group")

    def all_to_all(recv, send, group):
        recv.copy_(group.world.exchange("all_to_all", group.rank_in_group, send))

    scope = {
        "torch": torch,
        "dist": SimpleNamespace(all_to_all_single=all_to_all),
        "get_forward_context": context.get,
        "is_forward_context_available": lambda: context.get(None) is not None,
        "get_otp_group": group.get,
        "get_mlp_tp_group": group.get,
    }
    load(
        LINEAR,
        [
            "CustomLinearOp",
            "CustomColumnParallelOp",
            "CustomRowParallelOp",
            "MLPColumnParallelOp",
            "MLPRowParallelOp",
            "OProjRowParallelOp",
            "_pad_target_projection_input",
        ],
        scope,
    )
    return scope, context, group


def make_op(cls, weight, prefix, partition):
    layer = SimpleNamespace(
        weight=weight,
        bias=None,
        prefix=prefix,
        output_size=weight.shape[0],
        input_is_parallel=True,
        input_size_per_partition=partition,
        skip_bias_add=False,
        return_bias=False,
        reduce_results=True,
        gather_output=False,
        quant_method=SimpleNamespace(apply=lambda layer, x, bias=None: x @ layer.weight.T),
    )
    op = cls(layer)
    op.update_attrs()
    return op


@pytest.mark.parametrize("kind", ["oproj", "mlp"])
@pytest.mark.parametrize("counts", [(1, 5), (0, 3), (2, 1, 5, 0)])
def test_uneven_and_idle_peers_match_unsharded_projection(kind, counts):
    torch.manual_seed(42)
    scope, context, group = scope_for_projection()
    size = len(counts)
    world = World(size)
    hidden = 8
    intermediate = 12
    weights = [
        torch.randn(hidden, hidden),
        torch.randn(intermediate, hidden),
        torch.randn(intermediate, hidden),
        torch.randn(hidden, intermediate),
    ]
    inputs = [torch.randn(count, hidden) for count in counts]
    aligned = max(6, max(counts))

    def run(rank):
        real = counts[rank]
        local = inputs[rank] if real else torch.full((6, hidden), float("nan"))
        saved = local.clone()
        ctx = SimpleNamespace(
            additional_kwargs={
                "finegrained_tp_rows": (real, aligned),
                "finegrained_tp_mlp_rows": {},
                "capturing": False,
            }
        )
        context.set(ctx)
        group.set(Group(world, rank))
        if kind == "oproj":
            weight = weights[0].chunk(size, dim=1)[rank]
            op = make_op(scope["OProjRowParallelOp"], weight, "model.layers.0.self_attn.o_proj", hidden // size)
            result, _ = op.apply_impl(local)
            expected = inputs[rank] @ weights[0].T
        else:
            gate, up = [weight.chunk(size)[rank] for weight in weights[1:3]]
            column = make_op(scope["MLPColumnParallelOp"], torch.cat([gate, up]), "model.layers.0.mlp.gate_up_proj", 0)
            down = make_op(
                scope["MLPRowParallelOp"],
                weights[3].chunk(size, dim=1)[rank],
                "model.layers.0.mlp.down_proj",
                intermediate // size,
            )
            packed, _ = column.apply_impl(local)
            gate_values, up_values = packed.chunk(2, dim=-1)
            result, _ = down.apply_impl(torch.nn.functional.silu(gate_values) * up_values)
            expected = (
                torch.nn.functional.silu(inputs[rank] @ weights[1].T) * (inputs[rank] @ weights[2].T)
            ) @ weights[3].T
            assert ctx.additional_kwargs["finegrained_tp_mlp_rows"] == {}
        assert result.shape == local.shape
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result[:real], expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(local, saved, equal_nan=True)

    with ThreadPoolExecutor(max_workers=size) as pool:
        list(pool.map(run, range(size)))
    assert len(world.calls) == 2


@pytest.mark.parametrize("rows,capturing", [(None, False), ((1, 4), True)])
def test_padding_bypasses_draft_and_capture(rows, capturing):
    scope, context, _ = scope_for_projection()
    tensor = torch.randn(1, 8)
    context.set(SimpleNamespace(additional_kwargs={"finegrained_tp_rows": rows, "capturing": capturing}))
    assert scope["_pad_target_projection_input"](tensor) is tensor


def test_padding_zeros_stale_rows_without_mutating_input():
    scope, context, _ = scope_for_projection()
    context.set(SimpleNamespace(additional_kwargs={"finegrained_tp_rows": (1, 4), "capturing": False}))
    tensor = torch.cat([torch.ones(1, 8), torch.full((3, 8), float("nan"))])
    padded = scope["_pad_target_projection_input"](tensor)
    assert torch.equal(padded, torch.cat([torch.ones(1, 8), torch.zeros(3, 8)]))
    assert torch.isnan(tensor[1:]).all()
    context.set(SimpleNamespace(additional_kwargs={"finegrained_tp_rows": (1, 3), "capturing": False}))
    with pytest.raises(RuntimeError, match="exceed"):
        scope["_pad_target_projection_input"](tensor)


class GraphMode(Enum):
    NONE = 0
    FULL = 1


@pytest.mark.parametrize("feature", ["oproj_tensor_parallel_size", "mlp_tensor_parallel_size"])
def test_prefill_configuration_keeps_tp_and_pcp_guards(monkeypatch, feature):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", SimpleNamespace(CUDAGraphMode=GraphMode))
    scope = {"logger": Mock()}
    load("vllm_ascend/ascend_config.py", ["_validate_preconditions"], scope, "FinegrainedTPConfig")
    finegrained = SimpleNamespace(
        oproj_tensor_parallel_size=0,
        mlp_tensor_parallel_size=0,
        embedding_tensor_parallel_size=0,
        lmhead_tensor_parallel_size=0,
    )
    setattr(finegrained, feature, 4)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, prefill_context_parallel_size=1, data_parallel_size=16),
        compilation_config=SimpleNamespace(
            cudagraph_mode=GraphMode.NONE, max_cudagraph_capture_size=48, cudagraph_capture_sizes=[48]
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384, max_num_seqs=128),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        model_config=SimpleNamespace(is_moe=True),
    )
    validate = scope["_validate_preconditions"]
    validate(finegrained, config)
    assert getattr(finegrained, feature) == 4
    config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(AssertionError, match="tensor_parallel_size == 1"):
        validate(finegrained, config)
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(AssertionError, match="prefill_context_parallel_size"):
        validate(finegrained, config)
    config.parallel_config.prefill_context_parallel_size = 1
    config.kv_transfer_config = SimpleNamespace(is_kv_producer=False, is_kv_consumer=True)
    with pytest.raises(AssertionError, match="graph mode"):
        validate(finegrained, config)
    config.compilation_config.cudagraph_mode = GraphMode.FULL
    validate(finegrained, config)
    assert getattr(finegrained, feature) == 1  # Current main disables an uncaptured split with size=1.


def test_runner_accepts_prefill_eager_and_preserves_decode_graph_guard():
    scope = {"CUDAGraphMode": GraphMode}
    load("vllm_ascend/worker/v2/model_runner.py", ["_check_finegrained_tp_graph_step"], scope, "NPUModelRunner")
    kv_config = SimpleNamespace(is_kv_producer=True, is_kv_consumer=False)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=kv_config), _finegrained_tp_requires_graph=True
    )
    check = scope["_check_finegrained_tp_graph_step"]
    check(runner, GraphMode.NONE)
    kv_config.is_kv_producer = False
    kv_config.is_kv_consumer = True
    with pytest.raises(RuntimeError, match="captured graph"):
        check(runner, GraphMode.NONE)
    check(runner, GraphMode.FULL)


def test_context_has_fresh_dp_rows_and_excludes_draft_profile_and_warmup():
    scope = {
        "ContextVar": ContextVar,
        "contextmanager": contextmanager,
        "CUDAGraphMode": GraphMode,
        "_MRV2_TARGET_ROWS": ContextVar("target_rows", default=None),
        "_MRV2_DRAFT_ROWS": ContextVar("draft_rows", default=None),
        "get_ascend_config": lambda: SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(oproj_tensor_parallel_size=4, mlp_tensor_parallel_size=0)
        ),
    }
    load(CONTEXT, ["override_mrv2_target_rows", "get_target_projection_rows"], scope)
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False))
    get_rows = scope["get_target_projection_rows"]
    assert get_rows(config, object(), 13, GraphMode.NONE) is None
    with scope["override_mrv2_target_rows"](2, 6):
        assert get_rows(config, object(), 13, GraphMode.NONE) == (2, 13)
        assert get_rows(config, object(), 13, GraphMode.NONE, is_draft_model=True) is None
        with scope["override_mrv2_target_rows"](None):
            assert get_rows(config, object(), 13, GraphMode.NONE) is None
        assert get_rows(config, object(), 1, GraphMode.NONE) == (2, 6)
        assert get_rows(config, object(), 13, GraphMode.FULL) is None
        with pytest.raises(RuntimeError, match="coordination"):
            get_rows(config, None, 13, GraphMode.NONE)
    assert get_rows(config, object(), 13, GraphMode.NONE) is None


def test_platform_passes_agreed_rows_to_runtime_projection(monkeypatch):
    context_scope = {
        "ContextVar": ContextVar,
        "contextmanager": contextmanager,
        "CUDAGraphMode": GraphMode,
        "_MRV2_TARGET_ROWS": ContextVar("target_rows", default=None),
        "_MRV2_DRAFT_ROWS": ContextVar("draft_rows", default=None),
        "get_ascend_config": lambda: SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(oproj_tensor_parallel_size=4, mlp_tensor_parallel_size=0)
        ),
    }
    load(CONTEXT, ["override_mrv2_target_rows", "get_target_projection_rows"], context_scope)
    modules = {
        "vllm": SimpleNamespace(),
        "vllm.config": SimpleNamespace(CUDAGraphMode=GraphMode),
        "vllm.distributed": SimpleNamespace(
            get_dp_group=lambda: SimpleNamespace(world_size=4), get_tensor_model_parallel_world_size=lambda: 1
        ),
        "vllm_ascend": SimpleNamespace(),
        "vllm_ascend.ascend_forward_context": SimpleNamespace(
            _is_decode_only_node=lambda config: False,
            get_mc2_mask=lambda: None,
            get_mrv2_in_profile_run=lambda: False,
            sync_v2_extra_kwargs=lambda config: None,
            select_moe_comm_method=lambda *args: None,
            use_cann_megamoe=lambda config: False,
            get_target_projection_rows=context_scope["get_target_projection_rows"],
        ),
        "vllm_ascend.ops.fused_moe.moe_comm_method": SimpleNamespace(get_moe_comm_method=lambda kind: None),
        "vllm_ascend.quantization.utils": SimpleNamespace(get_dynamic_mx_quant_scale_alg=lambda config: 0),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    scope = {"math": math, "is_moe_model": lambda config: False}
    load("vllm_ascend/platform.py", ["set_additional_forward_context"], scope, "NPUPlatform")
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
    )
    with context_scope["override_mrv2_target_rows"](2, 6):
        extras = scope["set_additional_forward_context"].__func__(
            object,
            None,
            config,
            SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([2, 13, 0, 4])),
            num_tokens=2,
            cudagraph_runtime_mode=GraphMode.NONE,
        )
    assert extras["num_tokens"] == 2
    assert extras["finegrained_tp_rows"] == (2, 13)
    assert extras["finegrained_tp_mlp_rows"] == {}
    operators, context, _ = scope_for_projection()
    context.set(SimpleNamespace(additional_kwargs=extras))
    padded = operators["_pad_target_projection_input"](torch.ones(2, 8))
    assert padded.shape == (13, 8)
    assert padded[:2].eq(1).all() and padded[2:].eq(0).all()


@pytest.mark.parametrize("instance_method", [False, True])
def test_idle_embedding_joins_before_dummy_and_restores_method(instance_method):
    scope = {"contextmanager": contextmanager}
    load("vllm_ascend/worker/v2/model_runner.py", ["_target_embedding_dummy_collective"], scope)
    events = []

    class State:
        def dummy_inputs_embeds(self, num_tokens):
            events.append("dummy")
            return num_tokens

    state = State()
    if instance_method:
        state.dummy_inputs_embeds = lambda n: events.append("instance") or n
    original = state.dummy_inputs_embeds
    finegrained = SimpleNamespace(embedding_tensor_parallel_size=4)
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(finegrained_tp_config=finegrained),
        uses_inputs_embeds=True,
        is_first_pp_rank=True,
        model_state=state,
        input_buffers=SimpleNamespace(input_ids=torch.arange(6)),
        model=SimpleNamespace(embed_input_ids=lambda x: events.append("embedding")),
    )
    with pytest.raises(ValueError, match="restore"), scope["_target_embedding_dummy_collective"](runner):
        assert state.dummy_inputs_embeds(3) == 3
        finegrained.embedding_tensor_parallel_size = 0
        assert state.dummy_inputs_embeds(2) == 2
        raise ValueError("restore")
    assert state.dummy_inputs_embeds == original
    assert ("dummy_inputs_embeds" in state.__dict__) == instance_method
    assert events == [
        "embedding",
        "instance" if instance_method else "dummy",
        "instance" if instance_method else "dummy",
    ]


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("feature", ["oproj_tensor_parallel_size", "mlp_tensor_parallel_size"])
def test_native_target_and_draft_projection_shards_keep_dp_coordination(draft, feature):
    finegrained = SimpleNamespace(oproj_tensor_parallel_size=1, mlp_tensor_parallel_size=1)
    setattr(finegrained, feature, 4)
    scope = {"get_ascend_config": lambda: SimpleNamespace(finegrained_tp_config=finegrained)}
    load("vllm_ascend/utils.py", ["should_skip_allreduce_across_dp_group"], scope)
    assert scope["should_skip_allreduce_across_dp_group"](SimpleNamespace(), is_draft_model=draft) is False


def draft_rows_scope(profile=False, enabled=True):
    finegrained = SimpleNamespace(oproj_tensor_parallel_size=4 if enabled else 1, mlp_tensor_parallel_size=1)
    scope = {
        "contextmanager": contextmanager,
        "CUDAGraphMode": GraphMode,
        "_MRV2_TARGET_ROWS": ContextVar("target_rows", default=None),
        "_MRV2_DRAFT_ROWS": ContextVar("draft_rows", default=None),
        "get_ascend_config": lambda: SimpleNamespace(finegrained_tp_config=finegrained),
        "get_mrv2_in_profile_run": lambda: profile,
    }
    load(CONTEXT, ["override_mrv2_target_rows", "override_mrv2_draft_rows", "get_target_projection_rows"], scope)
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
                             speculative_config=SimpleNamespace(num_speculative_tokens=3))
    return scope, config


@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("local", [1, 4, 13])
def test_eager_draft_uses_its_own_rows_and_restores_target(local, dummy):
    scope, config = draft_rows_scope()
    get_rows = scope["get_target_projection_rows"]
    agreed = torch.tensor([1, 4, 13, 2])
    with scope["override_mrv2_target_rows"](77, 80):
        with pytest.raises(ValueError, match="restore"), scope["override_mrv2_draft_rows"](
            config, local, agreed, GraphMode.NONE, is_dummy=dummy
        ):
            assert get_rows(config, object(), 13, GraphMode.NONE, is_draft_model=True) == (0 if dummy else local, 13)
            raise ValueError("restore")
        assert get_rows(config, object(), 80, GraphMode.NONE) == (77, 80)
    assert get_rows(config, None, None, GraphMode.NONE) is None


@pytest.mark.parametrize("case", ["decoder", "both", "standalone", "graph", "profile", "disabled", "warmup"])
def test_draft_scope_preserves_non_prefill_and_profile_paths(case):
    scope, config = draft_rows_scope(profile=case == "profile", enabled=case != "disabled")
    mode = GraphMode.FULL if case == "graph" else GraphMode.NONE
    if case in {"decoder", "both"}:
        config.kv_transfer_config = SimpleNamespace(is_kv_producer=case == "both", is_kv_consumer=True)
    elif case == "standalone":
        config.kv_transfer_config = None
    with scope["override_mrv2_target_rows"](77, 80):
        agreed = None if case == "warmup" else torch.tensor([4, 13])
        with scope["override_mrv2_draft_rows"](config, 4, agreed, mode, runtime_batch=case != "warmup"):
            assert scope["get_target_projection_rows"](config, None, 13, mode) is None
        assert scope["_MRV2_TARGET_ROWS"].get() == (77, 80)


def test_runtime_draft_rejects_missing_or_inconsistent_dp_agreement():
    scope, config = draft_rows_scope()
    with pytest.raises(RuntimeError, match="requires DP"):
        with scope["override_mrv2_draft_rows"](config, 4, None, GraphMode.NONE):
            pass
    with pytest.raises(RuntimeError, match="exceed"):
        with scope["override_mrv2_draft_rows"](config, 5, torch.tensor([1, 4]), GraphMode.NONE):
            pass


@pytest.mark.parametrize("local,dummy", [(1, False), (4, True)])
def test_draft_minimum_rows_preserves_idle_mtp_buffer(local, dummy):
    scope, config = draft_rows_scope()
    with scope["override_mrv2_draft_rows"](config, local, torch.tensor([1, 0]), GraphMode.NONE, is_dummy=dummy):
        assert scope["get_target_projection_rows"](config, object(), 1, GraphMode.NONE) == (0 if dummy else 1, 4)


def test_eager_native_draft_ragged_and_idle_oproj_matches_dense():
    torch.manual_seed(42)
    scope, config = draft_rows_scope()
    operators, context, group = scope_for_projection()
    counts = (2, 0, 5, 1)
    world = World(len(counts))
    weight = torch.randn(8, 8)
    inputs = [torch.randn(n, 8) for n in counts]

    def run(rank):
        real = counts[rank]
        local = inputs[rank] if real else torch.full((4, 8), float("nan"))
        with scope["override_mrv2_draft_rows"](config, len(local), torch.tensor([2, 4, 5, 1]), GraphMode.NONE,
                                              is_dummy=not real):
            rows = scope["get_target_projection_rows"](config, object(), 5, GraphMode.NONE, is_draft_model=True)
            context.set(SimpleNamespace(additional_kwargs={"finegrained_tp_rows": rows, "capturing": False}))
            group.set(Group(world, rank))
            op = make_op(operators["OProjRowParallelOp"], weight.chunk(4, dim=1)[rank], "mtp.o_proj", 2)
            result, _ = op.apply_impl(local)
            assert result.shape == local.shape
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result[:real], inputs[rank] @ weight.T)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))
    assert len(world.calls) == 2


@pytest.mark.parametrize("dummy", [False, True])
def test_native_speculator_wrapper_applies_draft_scope_on_real_and_idle(dummy):
    scope, config = draft_rows_scope()

    class Parent:
        def _run_model(self, *args):
            rows = scope["get_target_projection_rows"](config, object(), 13, GraphMode.NONE, is_draft_model=True)
            return rows, rows

    path = ROOT / "vllm_ascend/worker/v2/spec_decode/target_dependent_ar/speculator.py"
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    owner = next(n for n in nodes if isinstance(n, ast.ClassDef) and
                 any(isinstance(f, ast.FunctionDef) and f.name == "_run_model" for f in n.body))
    method = next(f for f in owner.body if isinstance(f, ast.FunctionDef) and f.name == "_run_model")
    cls = ast.ClassDef(name="Probe", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
                        type_ignores=[])
    scope.update(torch=torch, Parent=Parent, AscendPCPManager=SimpleNamespace(
        broadcast_replicated_hidden_states=lambda last, hidden, *args, **kwargs: (last, hidden)))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    speculator = scope["Probe"]()
    speculator.vllm_config = config
    speculator._eplb_dummy_run = dummy
    speculator.replicated_pcp = False
    # Both real and native idle propose receive execute_model_state.input_batch.
    speculator.input_batch = object()
    expected = (0 if dummy else 4, 13)
    assert speculator._run_model(4, None, None, torch.tensor([4, 13]), GraphMode.NONE) == (expected, expected)
    assert scope["_MRV2_DRAFT_ROWS"].get() is None


def test_kv_control_context_does_not_require_model_row_agreement():
    scope, config = draft_rows_scope()
    get_rows = scope["get_target_projection_rows"]
    with scope["override_mrv2_target_rows"](0, 4):
        assert get_rows(config, None, None, GraphMode.NONE) is None
        with pytest.raises(RuntimeError, match="requires DP"):
            get_rows(config, None, 4, GraphMode.NONE)
    with scope["override_mrv2_draft_rows"](config, 4, torch.tensor([4, 13]), GraphMode.NONE):
        assert get_rows(config, None, None, GraphMode.NONE, is_draft_model=True) is None
