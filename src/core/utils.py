from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_mask(size: int, device: torch.device | str) -> torch.Tensor:
    return torch.triu(
        torch.ones((size, size), device=device, dtype=torch.bool), diagonal=1
    )
