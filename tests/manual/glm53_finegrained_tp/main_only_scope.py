"""Test-only GLM5Next scope for target-only fine-grained TP validation.

The GLM-5.3-Flash validation changes only the target model. The draft/MTP
constructor and capture/propose calls therefore run with all fine-grained TP
knobs disabled. Target configuration is restored in ``finally`` blocks.

Embedding and LM Head normally alias the target modules. When either target
module is sharded differently, this adapter keeps an independent TP2 draft
module and reloads the exact baseline target checkpoint slice used by MTP.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import sys
from pathlib import Path


MTP_MODEL_MODULE = "vllm_ascend.models.glm5next.mtp"
MTP_SPECULATOR_MODULE = "vllm_ascend.worker.v2.spec_decode.mtp.speculator"
KNOBS = (
    "oproj_tensor_parallel_size",
    "mlp_tensor_parallel_size",
    "embedding_tensor_parallel_size",
    "lmhead_tensor_parallel_size",
)


def _disabled_config(config):
    current = config.finegrained_tp_config
    return type(current)(**{name: 0 for name in KNOBS})


def scoped_init(original, config_getter):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        config = config_getter()
        saved = config.finegrained_tp_config
        target_knobs = {name: getattr(saved, name) for name in KNOBS}
        config.finegrained_tp_config = _disabled_config(config)
        try:
            result = original(self, *args, **kwargs)
        finally:
            config.finegrained_tp_config = saved
        self._main_only_scope_evidence = {
            "target_knobs": target_knobs,
            "draft_constructor_knobs": {name: 0 for name in KNOBS},
            "target_config_restored": config.finegrained_tp_config is saved,
        }
        self._main_only_model_path = kwargs["vllm_config"].model_config.model
        return result

    return wrapped


def scoped_execution(original, config_getter):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        config = config_getter()
        saved = config.finegrained_tp_config
        disabled = getattr(self, "_main_only_disabled_config", None)
        if disabled is None:
            disabled = _disabled_config(config)
            self._main_only_disabled_config = disabled
        config.finegrained_tp_config = disabled
        try:
            return original(self, *args, **kwargs)
        finally:
            config.finegrained_tp_config = saved

    return wrapped


def _tensor_digest(tensor):
    import torch

    raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _indexed_weight(model_path: Path, key: str):
    matches = []
    for index in model_path.glob("*.index.json"):
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        if key in weight_map:
            matches.append((index, weight_map[key]))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one indexed checkpoint entry for {key}")
    return matches[0]


def _load_checkpoint_slice(model_path, key, module):
    import torch
    from safetensors import safe_open

    index, shard = _indexed_weight(model_path, key)
    if module.tp_size <= 0 or module.num_embeddings % module.tp_size:
        raise RuntimeError(f"Unsupported TP layout for {key}")
    if module.num_embeddings_padded != module.num_embeddings:
        raise RuntimeError(f"Padded vocabulary is unsupported for {key}")
    width = module.num_embeddings // module.tp_size
    start = module.tp_rank * width
    end = start + width
    with safe_open(str(model_path / shard), framework="pt", device="cpu") as checkpoint:
        source = checkpoint.get_slice(key)
        expected_shape = [module.num_embeddings, module.embedding_dim]
        if source.get_shape() != expected_shape:
            raise RuntimeError(f"Checkpoint shape mismatch for {key}")
        part = source[start:end, :].contiguous()
    if part.shape != module.weight.shape or part.dtype != module.weight.dtype:
        raise RuntimeError(f"Loaded TP shard mismatch for {key}")
    with torch.no_grad():
        module.weight.copy_(part)
    source_hash = _tensor_digest(part)
    loaded_hash = _tensor_digest(module.weight.detach().cpu())
    if source_hash != loaded_hash:
        raise RuntimeError(f"Bytewise checkpoint verification failed for {key}")
    return {
        "checkpoint_index": str(index),
        "checkpoint_shard": shard,
        "checkpoint_key": key,
        "vocab_slice": [start, end],
        "tp_size": module.tp_size,
        "tp_rank": module.tp_rank,
        "shape": list(module.weight.shape),
        "source_sha256": source_hash,
        "loaded_sha256": loaded_hash,
    }


def isolated_weight_load(original):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        knobs = self._main_only_scope_evidence["target_knobs"]
        model_path = Path(self._main_only_model_path)
        if knobs["lmhead_tensor_parallel_size"]:
            proofs = [
                _load_checkpoint_slice(model_path, "lm_head.weight", layer.shared_head.head)
                for layer in self.model.layers.values()
            ]
            self._main_only_baseline_head_evidence = {
                "exact_checkpoint_slice_match": True,
                "layers": proofs,
                "sharing_prevented": 0,
            }
        if knobs["embedding_tensor_parallel_size"]:
            proof = _load_checkpoint_slice(
                model_path,
                "model.language_model.embed_tokens.weight",
                self.model.embed_tokens,
            )
            self._main_only_baseline_embedding_evidence = {
                "exact_checkpoint_slice_match": True,
                **proof,
                "sharing_prevented": 0,
            }
        return result

    return wrapped


def isolated_module_sharing(original):
    @functools.wraps(original)
    def wrapped(eagle, flag, draft, target):
        knobs = getattr(eagle, "_main_only_scope_evidence", {}).get("target_knobs", {})
        evidence_name = None
        if flag == "has_own_lm_head" and knobs.get("lmhead_tensor_parallel_size", 0):
            evidence_name = "_main_only_baseline_head_evidence"
        if flag == "has_own_embed_tokens" and knobs.get("embedding_tensor_parallel_size", 0):
            evidence_name = "_main_only_baseline_embedding_evidence"
        if evidence_name is not None:
            proof = getattr(eagle, evidence_name, None)
            if not proof or not proof["exact_checkpoint_slice_match"]:
                raise RuntimeError("Refuse draft isolation without exact checkpoint proof")
            proof["sharing_prevented"] += 1
            return False
        return original(eagle, flag, draft, target)

    return wrapped


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        from vllm_ascend.ascend_config import get_ascend_config

        if module.__name__ == MTP_MODEL_MODULE:
            cls = module.Glm5NextMTP
            cls.__init__ = scoped_init(cls.__init__, get_ascend_config)
            cls.load_weights = isolated_weight_load(cls.load_weights)
            from vllm.v1.worker.gpu.spec_decode.eagle import utils as sharing

            if not getattr(sharing._should_share, "_main_only_guard", False):
                sharing._should_share = isolated_module_sharing(sharing._should_share)
                sharing._should_share._main_only_guard = True
        else:
            cls = module.AscendMTPSpeculator
            for name in ("capture", "propose"):
                setattr(cls, name, scoped_execution(getattr(cls, name), get_ascend_config))


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (MTP_MODEL_MODULE, MTP_SPECULATOR_MODULE):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = Loader(spec.loader)
        return spec


def install():
    if any(name in sys.modules for name in (MTP_MODEL_MODULE, MTP_SPECULATOR_MODULE)):
        raise RuntimeError("Install target-only scope before importing GLM5Next MTP")
    sys.meta_path.insert(0, Finder())
