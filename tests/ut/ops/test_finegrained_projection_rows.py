# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Regression coverage for native fine-grained TP coordination."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load(path, names, scope, owner=None):
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    if owner is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), scope)


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
