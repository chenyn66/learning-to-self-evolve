from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
)

try:
    # Linux-only; used for a simple cross-process file lock.
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from torch.distributed._tensor import Placement

try:
    # torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    from torch.distributed._tensor import DTensor  # type: ignore


def _default_cache_root() -> Path:
    """Where converted HF models are stored.

    Priority:
    - $LSE_MODEL_CACHE_DIR (if set)
    - /data-fast/models (if /data-fast exists; we'll create models/ if needed)
    - ~/models
    """
    env = os.environ.get("LSE_MODEL_CACHE_DIR")
    if env:
        root = Path(env).expanduser()
    else:
        root = Path.home() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_fsdp_checkpoint_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "fsdp_config.json").is_file():
        return False
    # At least one rank shard should exist.
    return any(path.glob("model_world_size_*_rank_0.pt"))


def detect_fsdp_checkpoint_root(name_or_path: str) -> Optional[Path]:
    """Return the FSDP checkpoint *root* dir if detected, else None.

    We consider a directory an FSDP checkpoint if it contains:
    - fsdp_config.json
    - model_world_size_*_rank_0.pt (and other rank shards)
    """
    p = Path(name_or_path).expanduser()
    if not p.exists():
        return None

    if p.is_dir() and _is_fsdp_checkpoint_root(p):
        return p

    # Common user mistake: point to the embedded config dir.
    if p.is_dir() and p.name == "huggingface" and _is_fsdp_checkpoint_root(p.parent):
        return p.parent

    # Another common structure: a "global_step_x/" dir containing "actor/" (and sometimes other roles).
    # If exactly one direct child dir is an FSDP checkpoint root, use it.
    # If there are multiple, prefer "actor" if present, otherwise treat as ambiguous.
    if p.is_dir():
        candidates = [d for d in p.iterdir() if d.is_dir() and _is_fsdp_checkpoint_root(d)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            actor = next((d for d in candidates if d.name == "actor"), None)
            if actor is not None:
                return actor

    return None


def _cache_target_dir(fsdp_root: Path, cache_root: Path) -> Path:
    # Human-readable tail (directory name).
    # If you need the legacy collision-avoidance behavior, set:
    #   LSE_FSDP_MERGE_USE_HASH=1
    parts = fsdp_root.resolve().parts[-4:]
    tail = "-".join(parts)
    tail = re.sub(r"[^A-Za-z0-9_.-]+", "-", tail).strip("-")
    if not tail:
        tail = "fsdp_ckpt"
    if len(tail) > 80:
        tail = tail[-80:]
    if os.environ.get("LSE_FSDP_MERGE_USE_HASH", "0") == "1":
        h = hashlib.sha1(str(fsdp_root.resolve()).encode("utf-8")).hexdigest()[:12]
        tail = f"{tail}--{h}"
    return cache_root / "lse_fsdp_merged" / tail


def _looks_like_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    # A few common weight filename patterns.
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return True
    if any(path.glob("model-*.safetensors")):
        return True
    if any(path.glob("pytorch_model-*.bin")):
        return True
    return False


def resolve_model_path(name_or_path: str | None, trust_remote_code: bool = True) -> str | None:
    """If `name_or_path` points to an FSDP checkpoint, convert+cache and return HF dir.

    Otherwise returns `name_or_path` unchanged.
    """
    if name_or_path is None:
        return None

    if os.environ.get("LSE_DISABLE_FSDP_AUTO_MERGE", "0") == "1":
        return name_or_path

    fsdp_root = detect_fsdp_checkpoint_root(name_or_path)
    if fsdp_root is None:
        return name_or_path

    print(f"Detected FSDP checkpoint at {fsdp_root}")
    cache_root = _default_cache_root()
    target_dir = _cache_target_dir(fsdp_root, cache_root)
    print(f"Converting to HuggingFace model at {target_dir}")
    merged_dir = merge_fsdp_checkpoint_to_hf(
        local_dir=fsdp_root,
        target_dir=target_dir,
        trust_remote_code=trust_remote_code,
    )
    return str(merged_dir)


def merge_fsdp_checkpoint_to_hf(
    *,
    local_dir: Path,
    target_dir: Path,
    trust_remote_code: bool = True,
) -> Path:
    """Merge an FSDP checkpoint directory into a HuggingFace `save_pretrained` directory."""
    local_dir = Path(local_dir).expanduser()
    target_dir = Path(target_dir).expanduser()
    hf_config_dir = local_dir / "huggingface"

    if not _is_fsdp_checkpoint_root(local_dir):
        raise ValueError(f"Not an FSDP checkpoint directory: {local_dir}")
    if not hf_config_dir.is_dir():
        raise FileNotFoundError(f"Expected HuggingFace config dir at {hf_config_dir}")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir.with_suffix(target_dir.suffix + ".lock")

    # A cheap cross-process lock; safe enough for shared node usage.
    lock_fh = None
    if fcntl is not None:
        lock_fh = open(lock_path, "a+", encoding="utf-8")
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

    try:
        if _looks_like_hf_model_dir(target_dir):
            return target_dir

        # Clean stale temp dirs from prior interrupted runs (best-effort).
        for p in target_dir.parent.glob(target_dir.name + ".tmp-*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

        tmp_dir = target_dir.with_name(target_dir.name + f".tmp-{os.getpid()}-{int(time.time())}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        print(f"[lse] Detected FSDP checkpoint at {local_dir}")
        print(f"[lse] Converting to HuggingFace model at {target_dir}")

        try:
            _merge_fsdp_checkpoint_impl(
                local_dir=local_dir,
                target_dir=tmp_dir,
                publish_dir=target_dir,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            # Avoid leaving behind confusing ".tmp-*" dirs if a merge fails.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        # Best-effort atomic publish.
        if target_dir.exists():
            shutil.rmtree(target_dir)
        tmp_dir.rename(target_dir)
        return target_dir
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fh.close()


# ----------------------------
# Minimal HF tokenizer helpers
# ----------------------------


def _set_pad_token_id(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(
            f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=2
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=2)


def hf_tokenizer(name_or_path: str, correct_pad_token: bool = True, correct_gemma2: bool = True, **kwargs):
    """Small subset of `verl.utils.hf_tokenizer`."""
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.",
            stacklevel=2,
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        _set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path: str, **kwargs):
    """Small subset of `verl.utils.hf_processor`."""
    from transformers import AutoConfig, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        processor.config = config
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                processor.get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, processor)
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                processor.get_rope_index = types.MethodType(Qwen2_5_VLModel.get_rope_index, processor)
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                processor.get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, processor)
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                processor.get_rope_index = types.MethodType(Glm4vModel.get_rope_index, processor)
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")
    except Exception as e:
        processor = None
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=2)

    # Avoid loading tokenizer indirectly via ProcessorAuto.
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


# ----------------------------
# FSDP checkpoint merge internals
# ----------------------------


def _get_world_size(local_dir: Path) -> int:
    config_path = local_dir / "fsdp_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    world_size = config.get("world_size")
    if world_size is None:
        # Fallback to parsing filenames if needed.
        for p in local_dir.glob("model_world_size_*_rank_0.pt"):
            m = re.search(r"model_world_size_(\d+)_rank_0\.pt$", p.name)
            if m:
                return int(m.group(1))
        raise ValueError(f"world_size not found in {config_path}")
    return int(world_size)


def _extract_device_mesh_info(state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        mesh = np.array([world_size], dtype=np.int64)
        mesh_dim_names = ("fsdp",)
    return mesh, mesh_dim_names


def _calculate_shard_configuration(mesh: np.ndarray, mesh_dim_names: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        # TODO: not supported yet
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    return total_shards, mesh_shape


def _merge_by_placement(tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise NotImplementedError(f"Unsupported placement: {placement}")


def _load_and_merge_state_dicts(
    local_dir: Path,
    *,
    world_size: int,
    total_shards: int,
    mesh_shape: tuple[int, ...],
    mesh_dim_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    model_state_dict_lst: list[Optional[dict]] = [None] * total_shards

    def process_one_shard(rank: int) -> None:
        model_path = local_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 8)) as executor:
        futures = [executor.submit(process_one_shard, rank) for rank in range(total_shards)]
        for f in futures:
            f.result()

    assert model_state_dict_lst[0] is not None
    # Merge state dicts from all shards
    merged: dict[str, list[torch.Tensor] | torch.Tensor] = {}
    param_placements: dict[str, tuple] = {}

    for key in set(model_state_dict_lst[0].keys()):
        merged[key] = []
        for shard_dict in model_state_dict_lst:
            assert shard_dict is not None
            tensor = shard_dict.pop(key)
            if isinstance(tensor, DTensor):
                merged[key].append(tensor._local_tensor.bfloat16())  # type: ignore[attr-defined]
                placements = tuple(tensor.placements)
                # Replicated placement at dp dimension can be discarded.
                if mesh_dim_names[0] in ("dp", "ddp"):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                merged[key].append(tensor.bfloat16())

    # Allow shard dicts to be freed ASAP.
    del model_state_dict_lst
    gc.collect()

    # Merge tensors
    state_dict: dict[str, torch.Tensor] = {}
    for key in sorted(merged):
        shards_or_tensor = merged[key]
        if not isinstance(shards_or_tensor, list):
            state_dict[key] = shards_or_tensor
            continue

        if key in param_placements:
            placements = param_placements[key]
            if len(mesh_shape) == 1:
                assert len(placements) == 1
                state_dict[key] = _merge_by_placement(shards_or_tensor, placements[0])  # type: ignore[arg-type]
            else:
                raise NotImplementedError("FSDP + TP is not supported yet")
        else:
            state_dict[key] = torch.cat(shards_or_tensor, dim=0)

    return state_dict


def _get_transformers_auto_model_class(model_config) -> type:
    has_remote_code = hasattr(model_config, "auto_map") and any(
        model_config.architectures[0] in val for val in model_config.auto_map.values()
    )
    if has_remote_code:
        auto_class = next(k for k, v in model_config.auto_map.items() if model_config.architectures[0] in v)
        match auto_class:
            case "AutoModelForCausalLM":
                return AutoModelForCausalLM
            case "AutoModelForTokenClassification":
                return AutoModelForTokenClassification
            case "AutoModelForVision2Seq":
                return AutoModelForVision2Seq
            case _:
                raise NotImplementedError(f"Unknown auto class {auto_class}")

    arch = model_config.architectures[0]
    if "ForTokenClassification" in arch:
        return AutoModelForTokenClassification
    if "ForCausalLM" in arch:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in arch:
        return AutoModelForVision2Seq
    raise NotImplementedError(f"Unknown architecture {model_config.architectures}")


def _patch_model_generation_config(model, hf_model_config_path: str, trust_remote_code: bool):
    if model.can_generate():
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                hf_model_config_path, trust_remote_code=trust_remote_code
            )
        except OSError:
            print(
                f"[lse] Warning: generation config not found in {hf_model_config_path}; using config-derived defaults."
            )
    return model


def _save_lora_adapter(state_dict: dict[str, torch.Tensor], target_dir: Path) -> Optional[Path]:
    lora_params_names = [name for name in state_dict.keys() if "lora_" in name]
    if not lora_params_names:
        return None

    from collections import OrderedDict

    import peft
    from safetensors.torch import save_file

    lora_params: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    target_modules: set[str] = set()
    lora_key: Optional[str] = None

    for name in lora_params_names:
        lora_key = name.replace(".default.weight", ".weight")
        target_modules.add(lora_key.split(".")[-3])
        lora_params[lora_key] = state_dict.pop(name)

    assert lora_key is not None
    lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
    peft_dict = {
        "r": lora_rank,
        "lora_alpha": 0,  # must be filled by user for actual use
        "target_modules": list(target_modules),
    }
    peft_config = peft.LoraConfig(**peft_dict).to_dict()
    peft_config["task_type"] = peft_config["task_type"].value if peft_config["task_type"] else None
    peft_config["peft_type"] = peft_config["peft_type"].value if peft_config["peft_type"] else None
    peft_config["target_modules"] = list(peft_config["target_modules"])

    lora_path = target_dir / "lora_adapter"
    lora_path.mkdir(parents=True, exist_ok=True)
    with open(lora_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(peft_config, f, ensure_ascii=False, indent=4)
    save_file(lora_params, str(lora_path / "adapter_model.safetensors"))

    # Normalize some common key prefixes.
    for name in list(state_dict.keys()):
        key = (
            name.replace("base_model.model.", "")
            .replace(".base_layer.weight", ".weight")
            .replace(".base_layer.bias", ".bias")
        )
        state_dict[key] = state_dict.pop(name)

    return lora_path


def _merge_fsdp_checkpoint_impl(
    *, local_dir: Path, target_dir: Path, trust_remote_code: bool, publish_dir: Optional[Path] = None
) -> None:
    world_size = _get_world_size(local_dir)
    rank0 = torch.load(
        local_dir / f"model_world_size_{world_size}_rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )

    mesh, mesh_dim_names = _extract_device_mesh_info(rank0, world_size)
    total_shards, mesh_shape = _calculate_shard_configuration(mesh, mesh_dim_names)

    merged_state_dict = _load_and_merge_state_dicts(
        local_dir,
        world_size=world_size,
        total_shards=total_shards,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )

    hf_model_config_path = str((local_dir / "huggingface").resolve())
    model_config = AutoConfig.from_pretrained(hf_model_config_path, trust_remote_code=trust_remote_code)
    auto_model_class = _get_transformers_auto_model_class(model_config)

    with init_empty_weights():
        model = auto_model_class.from_config(
            model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
        )
    # Keep parity with the upstream merger: allocate empty weights on CPU.
    model.to_empty(device="cpu")
    model = _patch_model_generation_config(model, hf_model_config_path, trust_remote_code=trust_remote_code)

    lora_path = _save_lora_adapter(merged_state_dict, target_dir)
    if lora_path is not None:
        print(f"[lse] Saved LoRA adapter to {lora_path}")

    print(f"[lse] Saving merged model to {publish_dir or target_dir}")
    model.save_pretrained(str(target_dir), state_dict=merged_state_dict)
    del merged_state_dict
    del model
    gc.collect()

    # Tokenizer / processor
    processor = hf_processor(hf_model_config_path, trust_remote_code=trust_remote_code)
    tokenizer = hf_tokenizer(hf_model_config_path, trust_remote_code=trust_remote_code)
    if processor is not None:
        processor.save_pretrained(str(target_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(target_dir))

    # Copy any extra metadata files from the checkpoint's huggingface/ dir.
    src_hf_dir = local_dir / "huggingface"
    for p in src_hf_dir.iterdir():
        if not p.is_file():
            continue
        dst = target_dir / p.name
        if dst.exists():
            continue
        shutil.copy2(p, dst)

    # Minimal provenance marker (helps debugging cache issues).
    meta = {
        "source_fsdp_dir": str(local_dir.resolve()),
        "created_at_unix": int(time.time()),
        "world_size": world_size,
    }
    with open(target_dir / "lse_fsdp_merge_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

