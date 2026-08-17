from __future__ import annotations

import math
import re

import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.scaling = float(alpha) / int(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            base_layer.weight.new_zeros((int(rank), self.in_features))
        )
        self.lora_B = nn.Parameter(
            base_layer.weight.new_zeros((self.out_features, int(rank)))
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.base_layer.requires_grad_(False)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, inputs):
        update = F.linear(self.lora_dropout(inputs), self.lora_A)
        update = F.linear(update, self.lora_B) * self.scaling
        return self.base_layer(inputs) + update


def _layer_index(name: str) -> int | None:
    match = re.search(r"encoder\.layer\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def apply_lora_to_model(backbone: nn.Module, config: dict) -> list[str]:
    targets = tuple(config.get("target_modules", ["query", "value"]))
    last_n_layers = int(config.get("last_n_layers", 4))
    num_layers = int(backbone.config.num_hidden_layers)
    first_layer = max(0, num_layers - last_n_layers)
    replacements = []

    for name, module in backbone.named_modules():
        layer = _layer_index(name)
        if (
            isinstance(module, nn.Linear)
            and layer is not None
            and layer >= first_layer
            and any(name.endswith(target) for target in targets)
        ):
            replacements.append((name, module))

    for name, module in replacements:
        parent = backbone
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            parts[-1],
            LoRALinear(
                module,
                rank=int(config.get("r", 4)),
                alpha=float(config.get("alpha", 8)),
                dropout=float(config.get("dropout", 0.05)),
            ),
        )

    if not replacements:
        raise ValueError(
            f"LoRA matched no ESM2 layers; check target_modules={list(targets)} "
            f"and last_n_layers={last_n_layers}"
        )
    return [name for name, _ in replacements]
