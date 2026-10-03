"""Small helpers for inspecting and reproducing WADOE training runs."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

__all__ = ["count_parameters", "expert_usage", "set_seed"]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters of a module.

    Args:
        model: Any :class:`torch.nn.Module`.

    Returns:
        A tuple ``(total, trainable)`` of parameter counts. During the usual
        WADOE finetuning schedule the backbone is frozen, so ``trainable`` is
        much smaller than ``total``; the gap is a useful sanity check that only
        the experts and router are being updated.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def expert_usage(weights: torch.Tensor) -> torch.Tensor:
    """Compute per-expert usage frequency from routing weights.

    The result is the mean routing weight per expert over all batch and token
    positions, and therefore sums to 1. A healthy run keeps every entry away
    from both 0 and 1; a single entry near 1 signals router collapse.

    Args:
        weights: Routing weights of shape ``[B, T, n_experts]`` (or any shape
            with experts last).

    Returns:
        A 1-D tensor of length ``n_experts`` summing to 1.

    Raises:
        ValueError: If ``weights`` is not at least 2-D.
    """
    if weights.dim() < 2:
        raise ValueError(
            f"expected weights of shape [B, T, E], got {tuple(weights.shape)}"
        )
    return weights.mean(dim=tuple(range(weights.dim() - 1)))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs.

    Router collapse is sensitive to initialisation, so a fixed seed makes
    routing behaviour comparable across ablation runs.

    Args:
        seed: Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)
