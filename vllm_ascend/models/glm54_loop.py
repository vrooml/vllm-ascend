# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model, _get_llama_4_scaling
from vllm.sequence import IntermediateTensors

from vllm_ascend.attention.glm53_loop import loop_attention_pass, register_loop_cache
from vllm_ascend.models.deepseek_mtp import AscendGlmMoeDsaForCausalLM
from vllm_ascend.utils import should_reuse_topk

NUM_LAYERS = 78
LOOP_START = 26
LOOP_END = 54
INDEX_GROUP_SIZE = 4
MAX_REFERENCE_CONTEXT = 4096
INDEX_TOPK_PATTERN = ("F", "F") + ("F", "S", "S", "S") * ((NUM_LAYERS - 2) // INDEX_GROUP_SIZE)


def _prepare_loop_config(vllm_config: VllmConfig) -> tuple[float, float]:
    model_config = vllm_config.model_config
    config = model_config.hf_config
    if (
        config.model_type != "glm_moe_dsa"
        or config.num_hidden_layers != NUM_LAYERS
        or not getattr(config, "index_topk", None)
    ):
        raise ValueError("Glm53LoopForCausalLM requires a full 78-layer GLM-5.3 DSA checkpoint.")

    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.data_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1, 1):
        raise ValueError("Glm53LoopForCausalLM requires PP=DP=PCP=DCP=1.")
    if not model_config.enforce_eager:
        raise ValueError("Glm53LoopForCausalLM requires --enforce-eager.")
    if model_config.max_model_len > MAX_REFERENCE_CONTEXT:
        raise ValueError(f"Glm53LoopForCausalLM requires --max-model-len <= {MAX_REFERENCE_CONTEXT}.")
    if vllm_config.speculative_config is not None or vllm_config.kv_transfer_config is not None:
        raise ValueError("Glm53LoopForCausalLM supports target-only generation in a colocated engine.")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("Glm53LoopForCausalLM requires --no-enable-prefix-caching.")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16", "float16"):
        raise ValueError("Glm53LoopForCausalLM requires floating-point MLA KV cache (--kv-cache-dtype auto).")
    if getattr(vllm_config.attention_config, "hisparse_config", None) is not None:
        raise ValueError("Glm53LoopForCausalLM requires device-resident MLA KV cache.")

    gates = getattr(config, "glm53_loop_gates", (1.0, 1.0))
    if (
        not isinstance(gates, (tuple, list))
        or len(gates) != 2
        or any(type(gate) not in (int, float) or not 0.0 <= gate <= 1.0 for gate in gates)
    ):
        raise ValueError("glm53_loop_gates must contain two numbers in [0, 1].")

    indexer_types = getattr(config, "indexer_types", None)
    if (
        not isinstance(indexer_types, (list, tuple))
        or len(indexer_types) != NUM_LAYERS
        or any(
            not isinstance(indexer_type, str) or indexer_type.lower() != ("full" if pattern == "F" else "shared")
            for indexer_type, pattern in zip(indexer_types, INDEX_TOPK_PATTERN, strict=True)
        )
    ):
        raise ValueError("GLM-5.3 checkpoint indexer_types must match the GLM-5.4 full/shared index groups.")

    if any(
        should_reuse_topk(config, layer_idx) != (pattern == "S") for layer_idx, pattern in enumerate(INDEX_TOPK_PATTERN)
    ):
        config.index_topk_pattern = list(INDEX_TOPK_PATTERN)
    config.use_index_cache = True
    return float(gates[0]), float(gates[1])


def _apply_pass_gate(
    pass_input: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    gate: float,
) -> torch.Tensor:
    if gate == 0.0:
        return pass_input
    candidate = hidden_states + residual
    if gate == 1.0:
        return candidate
    return torch.lerp(pass_input, candidate, gate)


class Glm53LoopModel(DeepseekV2Model):
    """Execute GLM-5.4's text loop schedule with the physical GLM-5.3 layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        loop_gates = _prepare_loop_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.loop_gates = loop_gates
        self.loop_local_caches = tuple(
            register_loop_cache(self.layers[layer_idx].self_attn.mla_attn, vllm_config)
            for layer_idx in range(LOOP_START, LOOP_END)
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.embed_input_ids(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds is required.")

        llama_4_scaling = None
        if scaling_config := getattr(self.config, "llama_4_scaling", None):
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=scaling_config["original_max_position_embeddings"],
                scaling_beta=scaling_config["beta"],
                positions=positions,
            )

        residual = None
        for layer_idx in range(LOOP_START):
            hidden_states, residual = self.layers[layer_idx](positions, hidden_states, residual, llama_4_scaling)

        # DeepSeek layers return the MLP output and residual separately. At each
        # loop boundary combine them before applying the synthetic write gate.
        loop_input = hidden_states + residual
        hidden_states, residual = loop_input, None
        group_topk: list[torch.Tensor] = []
        for layer_idx in range(LOOP_START, LOOP_END):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)
            if (layer_idx - LOOP_START) % INDEX_GROUP_SIZE == 0:
                indices = layer.self_attn.mla_attn.topk_indices_buffer
                # Each indexer writes the same scratch buffer in the normal
                # model. The second loop pass needs all seven groups.
                group_topk.append(indices[: positions.shape[0]].clone())

        loop_input = _apply_pass_gate(loop_input, hidden_states, residual, self.loop_gates[0])
        hidden_states, residual = loop_input, None
        for local_idx, local_cache in enumerate(self.loop_local_caches):
            layer = self.layers[LOOP_START + local_idx]
            with loop_attention_pass(local_cache, group_topk[local_idx // INDEX_GROUP_SIZE]):
                hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

        hidden_states = _apply_pass_gate(loop_input, hidden_states, residual, self.loop_gates[1])
        residual = None
        for layer_idx in range(LOOP_END, NUM_LAYERS):
            hidden_states, residual = self.layers[layer_idx](positions, hidden_states, residual, llama_4_scaling)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Glm53LoopForCausalLM(AscendGlmMoeDsaForCausalLM):
    """Opt-in 5.4-shaped forward pass for the existing 5.3 checkpoint."""

    model_cls = Glm53LoopModel
