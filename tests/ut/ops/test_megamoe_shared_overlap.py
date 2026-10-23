# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for whole-MLP submission around routed MegaMoE.

Device kernels are mocked; these tests verify backend gating, host submission
order, stream dependencies and output collectives, not NPU numerical accuracy.
"""

import itertools
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend import ascend_config as config_module
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as runner_module
from vllm_ascend.ops.fused_moe import shared_experts as shared_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, SharedExpertParallelMode


class TestMegaMoESharedOverlap(unittest.TestCase):
    def test_config_normalization_and_fallback_matrix(self):
        for requested, installed, supported, overlap, deepseek in itertools.product(
            (0, 1, 2), (False, True), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                mode=requested,
                installed=installed,
                supported=supported,
                overlap=overlap,
                deepseek=deepseek,
            ):
                config = SimpleNamespace(
                    weight_nz_mode=1,
                    enable_mc2_hierarchy_comm=False,
                    enable_fused_mc2=requested,
                    multistream_overlap_shared_expert=overlap,
                    _is_megamoe_supported_by_config=MagicMock(return_value=supported),
                )
                vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(
                        architectures=["DeepseekV4ForCausalLM" if deepseek else "Qwen3MoeForCausalLM"]
                    )
                )
                with (
                    patch.object(config_module, "_MEGA_MOE_SUPPORTED", None),
                    patch.object(
                        config_module.importlib.util, "find_spec", return_value=object() if installed else None
                    ),
                ):
                    AscendConfig._validate_user_input_ranges(config)
                    AscendConfig._validate_fused_mc2_shared_expert_overlap(config, vllm_config)
                    mega_moe = requested == 2 and installed
                    expected_mode = 0 if requested == 0 or (mega_moe and not supported) else 1
                    self.assertEqual(config.enable_fused_mc2, expected_mode)
                    self.assertEqual(config_module.is_mega_moe_supported(), mega_moe)
                    fused_disallows_overlap = expected_mode == 1 and (not mega_moe or not deepseek)
                    self.assertEqual(
                        config.multistream_overlap_shared_expert,
                        overlap and not fused_disallows_overlap,
                    )
                    self.assertEqual(config._is_megamoe_supported_by_config.call_count, int(mega_moe))

    def run_forward(
        self,
        mode,
        *,
        eligible=True,
        comm=MoECommType.FUSED_MC2,
        use_mega_moe=True,
        defer_wait=False,
        decode_only=False,
    ):
        order = []
        hidden, logits, routed_out, shared_out = object(), object(), object(), object()
        shared_ready, router_ready, routed_done = object(), object(), object()
        stream_state = {}

        def wait_event(event):
            self.assertIs(stream_state["current"], auxiliary)
            self.assertIs(event, routed_done)
            order.append("wait_routed_finalize")

        def wait_stream(stream):
            self.assertIs(stream_state["current"], main)
            self.assertIs(stream, auxiliary)
            order.append("join_shared")

        def record_event():
            self.assertIs(stream_state["current"], main)
            order.append("record_routed_finalize")
            return routed_done

        main = SimpleNamespace(wait_stream=wait_stream, record_event=record_event)
        auxiliary = SimpleNamespace(wait_event=wait_event)
        stream_state["current"] = main

        @contextmanager
        def switch_stream(stream, *, enabled=True):
            previous = stream_state["current"]
            if enabled:
                stream_state["current"] = stream
            try:
                yield
            finally:
                stream_state["current"] = previous

        def mlp(gate_up, milestones):
            self.assertIs(stream_state["current"], auxiliary)
            self.assertIs(gate_up, submitted_gate_up)
            self.assertFalse(milestones.has_fine_grained_stage_events)
            order.append("shared_mlp")
            return shared_out

        def reduce_scatter(output):
            self.assertIs(stream_state["current"], auxiliary)
            self.assertIs(output, shared_out)
            order.append("shared_reduce_scatter")
            return shared_out

        def gather(output, metadata):
            self.assertIs(stream_state["current"], main)
            self.assertIs(output, shared_out)
            self.assertEqual(metadata, (3, 1))
            order.append("shared_all_gather")
            return shared_out

        shared = AscendSharedExperts.__new__(AscendSharedExperts)
        shared.multistream_overlap = True
        shared.parallel_mode = lambda: mode
        shared._finish_shared_mlp = MagicMock(side_effect=mlp)
        shared._pad_and_reduce_scatter = MagicMock(side_effect=reduce_scatter)
        shared._finalize_local_dp_output = MagicMock(side_effect=gather)
        shared.prepare_input_before_routed = lambda states: PreparedSharedExpertInput(
            states, is_gathered=mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        )
        submitted_gate_up = SimpleNamespace(
            local_dp_metadata=(3, 1) if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY else None
        )

        def prepare_router(*unused):
            order.append("shared_gate_up")
            return logits, shared_ready, router_ready, submitted_gate_up

        def routed(**kwargs):
            self.assertIs(stream_state["current"], main)
            self.assertIs(kwargs["hidden_states"], hidden)
            self.assertIs(context.moe_comm_type, comm)
            self.assertEqual(context.use_mega_moe, use_mega_moe)
            order.append("routed")
            return routed_out, RoutedMoEMilestones()

        runner = SimpleNamespace(
            ascend_shared_experts=shared,
            is_internal_router=False,
            _overlap_mega_moe=eligible,
            routed_input_transform=None,
            routed_output_transform=object() if defer_wait else None,
            _sequence_parallel_context=nullcontext,
            _can_overlap_sp_shared_with=lambda transform: defer_wait and transform is not None,
            _prepare_router_and_milestones=prepare_router,
            routed_experts=SimpleNamespace(forward_impl=routed),
        )
        context = SimpleNamespace(
            moe_comm_type=comm,
            use_mega_moe=use_mega_moe,
            is_decode_only_node=decode_only,
        )
        with (
            patch.object(runner_module, "_EXTRA_CTX", context),
            patch.object(shared_module, "npu_stream_switch", switch_stream),
            patch.object(shared_module, "shared_experts_calculation_stream", lambda: auxiliary),
            patch.object(shared_module.torch.npu, "current_stream", lambda: stream_state["current"]),
        ):
            result = AscendMoERunner._forward_impl(runner, hidden, logits, shared_experts_input=None)
            self.assertIs(result[0], shared_out)
            self.assertIs(result[1], routed_out)
            shared._finish_shared_mlp.assert_called_once()
            if defer_wait:
                self.assertNotIn("join_shared", order)
                shared.wait_for_output()
        self.assertEqual(order.count("join_shared"), 1)
        if mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
            self.assertLess(order.index("routed"), order.index("record_routed_finalize"))
            self.assertLess(order.index("record_routed_finalize"), order.index("wait_routed_finalize"))
            self.assertLess(order.index("wait_routed_finalize"), order.index("shared_reduce_scatter"))
            self.assertLess(order.index("shared_reduce_scatter"), order.index("join_shared"))
        else:
            shared._pad_and_reduce_scatter.assert_not_called()
        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            self.assertLess(order.index("routed"), order.index("join_shared"))
            self.assertLess(order.index("join_shared"), order.index("shared_all_gather"))
        else:
            shared._finalize_local_dp_output.assert_not_called()
        return order

    def test_decode_only_keeps_selected_megamoe_backend(self):
        for eligible, mode in itertools.product((False, True), SharedExpertParallelMode):
            with self.subTest(eligible=eligible, layout=mode):
                self.run_forward(mode, eligible=eligible, decode_only=True)

    def test_mega_moe_submits_complete_shared_mlp_before_routed(self):
        for mode in SharedExpertParallelMode:
            with self.subTest(layout=mode):
                order = self.run_forward(mode)
                self.assertLess(order.index("shared_gate_up"), order.index("shared_mlp"))
                self.assertLess(order.index("shared_mlp"), order.index("routed"))
                self.assertLess(order.index("routed"), order.index("join_shared"))

    def test_other_backends_keep_staged_shared_submission(self):
        for eligible, comm, use_mega_moe in (
            (False, MoECommType.FUSED_MC2, True),
            (True, MoECommType.FUSED_MC2, False),
            (True, MoECommType.MC2, True),
            (True, MoECommType.ALLTOALL, True),
        ):
            with self.subTest(eligible=eligible, backend=comm, mega_moe=use_mega_moe):
                order = self.run_forward(
                    SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP,
                    eligible=eligible,
                    comm=comm,
                    use_mega_moe=use_mega_moe,
                )
                self.assertLess(order.index("routed"), order.index("shared_mlp"))

    def test_sp_output_join_can_remain_deferred_until_routed_transform(self):
        order = self.run_forward(SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, defer_wait=True)
        self.assertLess(order.index("shared_mlp"), order.index("routed"))

    def test_early_submission_requires_overlap(self):
        shared = AscendSharedExperts.__new__(AscendSharedExperts)
        shared.multistream_overlap = False
        with self.assertRaisesRegex(RuntimeError, "requires multistream overlap"):
            shared.enqueue_mlp_for_fused_routed(object())

    def test_reused_output_requires_its_submission_and_overlap(self):
        for overlap, gate_up in ((False, object()), (True, None)):
            with self.subTest(overlap=overlap, gate_up=gate_up):
                shared = AscendSharedExperts.__new__(AscendSharedExperts)
                shared.multistream_overlap = overlap
                with self.assertRaisesRegex(RuntimeError, "requires its Gate-Up submission"):
                    shared.forward(
                        PreparedSharedExpertInput(object()),
                        RoutedMoEMilestones(),
                        submitted_gate_up=gate_up,
                        submitted_mlp_output=object(),
                    )


if __name__ == "__main__":
    unittest.main()
