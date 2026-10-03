"""Soft routing for WADOE Mixture-of-Experts layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WADOERouter"]


class WADOERouter(nn.Module):
    """Produce per-token mixture weights over experts.

    The router projects each token representation to ``n_experts`` logits,
    divides by ``temperature`` and applies a softmax to obtain soft weights::

        logits = Linear(x)                       # [B, T, E]
        w      = softmax(logits / temperature)   # [B, T, E]

    If ``top_k`` is set and smaller than ``n_experts``, only the ``top_k``
    largest entries are retained, the remainder set to zero, and the surviving
    weights renormalised to sum to one.

    Router weights are initialised from ``N(0, 0.01)`` so that routing starts
    close to uniform, which is important because a badly initialised router can
    lock onto a single expert before any useful signal arrives.

    ``temperature`` and ``top_k`` are plain mutable attributes, so sparsity can
    be annealed during training without rebuilding the module::

        router.top_k = None   # dense soft routing
        router.top_k = 2      # sparse top-2 routing

    Args:
        d_model: Model (hidden) dimension.
        n_experts: Number of experts to route to. Must be positive.
        temperature: Softmax temperature. Must be positive; lower values give
            sharper (more peaked) routing.
        top_k: If not ``None``, keep only the ``top_k`` experts per token.

    Raises:
        ValueError: If ``n_experts`` or ``temperature`` is invalid.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive when set, got {top_k}")

        self.d_model = int(d_model)
        self.n_experts = int(n_experts)
        self.temperature = float(temperature)
        self.top_k = top_k

        self.linear = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def _validate_top_k(self) -> None:
        """Check the mutable ``top_k`` attribute at call time.

        ``top_k`` may be reassigned between forward passes, so validation is
        performed here rather than only in ``__init__``.
        """
        if self.top_k is not None and self.top_k > self.n_experts:
            raise ValueError(
                f"top_k={self.top_k} exceeds n_experts={self.n_experts}; "
                "omit top_k to use dense routing."
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mixture weights for every token.

        Args:
            x: Input tensor of shape ``[B, T, d_model]``.

        Returns:
            A tuple ``(weights, logits)`` where ``weights`` has shape
            ``[B, T, n_experts]`` and sums to 1 over the last dimension, and
            ``logits`` has shape ``[B, T, n_experts]`` and is the *raw*
            (pre-temperature, pre-top-k) router projection. Auxiliary losses
            such as :func:`wadoe.losses.load_balance_loss` expect these raw
            logits.

        Raises:
            ValueError: If ``x`` is not 3-D or the final dimension does not
                match ``d_model``, or if ``temperature``/``top_k`` are invalid.
        """
        if x.dim() != 3:
            raise ValueError(f"expected input of shape [B, T, d_model], got shape {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, got {x.shape[-1]}"
            )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        self._validate_top_k()

        logits = self.linear(x)  # [B, T, E]
        weights = F.softmax(logits / self.temperature, dim=-1)

        if self.top_k is not None and self.top_k < self.n_experts:
            top_indices = torch.topk(weights, k=self.top_k, dim=-1).indices
            mask = torch.zeros_like(weights).scatter(-1, top_indices, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return weights, logits

    def extra_repr(self) -> str:
        """Return a compact representation for ``print(model)``."""
        return (
            f"d_model={self.d_model}, n_experts={self.n_experts}, "
            f"temperature={self.temperature}, top_k={self.top_k}"
        )
