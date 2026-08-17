from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .core import AminoAcidTokenizer, CognateCVAE
from .core.lora import LoRALinear
from .paths import default_checkpoint_path, default_config_path


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    config: dict
    device: str
    config_path: Path
    checkpoint_path: Path
    offline: bool = True
    _esm_tokenizer: Any = None

    def get_esm_tokenizer(self):
        esm_cfg = self.config.get("model", {}).get("esm", {})
        if not esm_cfg.get("enabled", False):
            return None
        if self._esm_tokenizer is None:
            from transformers import EsmTokenizer

            self._esm_tokenizer = EsmTokenizer.from_pretrained(
                esm_cfg["model_name"], local_files_only=self.offline
            )
        return self._esm_tokenizer


def _select_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint_obj.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint_obj


def _remap_lora_checkpoint_keys(model, checkpoint_state, lora_types):
    model_state = model.state_dict()
    lora_module_names = {
        name for name, module in model.named_modules() if isinstance(module, lora_types)
    }

    remapped = {}
    for key, value in checkpoint_state.items():
        target_key = key
        if key not in model_state:
            for module_name in lora_module_names:
                prefix = f"{module_name}."
                if key.startswith(prefix):
                    suffix = key[len(prefix) :]
                    candidate = f"{module_name}.base_layer.{suffix}"
                    if candidate in model_state:
                        target_key = candidate
                    break
        remapped[target_key] = value
    return remapped


def load_cognate(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    offline: bool = True,
) -> ModelBundle:
    if offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    config_file = Path(config_path or default_config_path()).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path or default_checkpoint_path()).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    config = load_config(config_file)

    target_device = _select_device(device)
    tokenizer = AminoAcidTokenizer()
    model = CognateCVAE(tokenizer.vocab_size, config)

    checkpoint_obj = torch.load(
        checkpoint_file, map_location="cpu", weights_only=True, mmap=True
    )
    checkpoint_state = _extract_state_dict(checkpoint_obj)
    checkpoint_state.pop("target_encoder.segment_embedding.weight", None)
    checkpoint_state = _remap_lora_checkpoint_keys(model, checkpoint_state, (LoRALinear,))
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    missing = len(getattr(incompatible, "missing_keys", []))
    unexpected = len(getattr(incompatible, "unexpected_keys", []))
    if (missing or unexpected) and os.environ.get("COGNATE_VERBOSE") == "1":
        print(f"[COGNATE] checkpoint load: missing={missing}, unexpected={unexpected}")

    model.to(target_device)
    if target_device.startswith("mps"):
        # MPS lacks the nested-tensor encoder path; use the regular path.
        for module in model.modules():
            if isinstance(module, torch.nn.TransformerEncoder):
                module.enable_nested_tensor = False
                module.use_nested_tensor = False
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=target_device,
        config_path=config_file,
        checkpoint_path=checkpoint_file,
        offline=offline,
    )
