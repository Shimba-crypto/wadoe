"""The WADOE layer: a stack of experts plus a soft router."""

from __future__ import annotations

import torch
import torch.nn as nn

from .experts import WADOEExpert
from .losses import routing_entropy
from .router import WADOERouter

__all__ = ["WADOELayer"]


class WADOELayer(nn.Module):
    """A soft Mixture-of-Experts layer that aggregates weighted expert outputs.

    Implements the mixture

    .. math::

        y = \\sum_{i=1}^{E} w_i(x)\\, E_i(x)

    where :math:`E_i` are :class:`wadoe.WADOEExpert` bottlenecks and
    :math:`w_i(x)` are per-token mixing weights produced by
    :class:`wadoe.WADOERouter`. Unlike hard top-1 routing, every expert with a
    non-zero weight contributes a differentiable term, so all experts receive
    gradient on any given token (subject to ``top_k`` sparsity).

    The layer is shape-preserving: ``[B, T, d_model]`` in, ``[B, T, d_model]``
    out. Drop it into a transformer block in place of the FFN sub-layer.

    Args:
        d_model: Model (hidden) dimension.
        d_ff: Bottleneck intermediate dimension of each expert.
        n_experts: Number of experts. Must be positive.
        rank: Rank of each expert's low-rank adaptation.
        temperature: Router softmax temperature.
        top_k: Keep only the ``top_k`` experts per token. ``None`` means dense
            soft routing over all experts. If ``top_k >= n_experts`` the setting
            is a no-op and dense routing is used instead, so ``top_k=2`` works
            even for one- or two-expert layers.
        dropout: Dropout applied inside each expert after GELU.

    Raises:
        ValueError: If ``n_experts`` is not positive.

    Example:
        >>> layer = WADOELayer(d_model=32, d_ff=64, n_experts=4, rank=8, top_k=2)
        >>> out, aux = layer(torch.randn(2, 8, 32))
        >>> out.shape
        torch.Size([2, 8, 32])
        >>> aux["weights"].shape
        torch.Size([2, 8, 4])
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 6,
        rank: int = 16,
        temperature: float = 1.0,
        top_k: int | None = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.n_experts = int(n_experts)

        # Keeping all experts is equivalent to dense routing, so top_k >= n_experts
        # is treated as "no sparsity" rather than an error.
        effective_top_k = top_k if (top_k is None or top_k < n_experts) else None

        self.experts = nn.ModuleList(
            [WADOEExpert(d_model, d_ff, rank=rank, dropout=dropout) for _ in range(n_experts)]
        )
        self.router = WADOERouter(
            d_model=d_model,
            n_experts=n_experts,
            temperature=temperature,
            top_k=effective_top_k,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_expert_outputs: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Route and mix the experts.

        Args:
            x: Input tensor of shape ``[B, T, d_model]``.
            return_expert_outputs: If ``True``, also stack and return every
                expert output. This costs an extra ``[B, T, n_experts, d_model]``
                tensor, so leave it ``False`` during normal training and enable
                it only for analysis.

        Returns:
            A tuple ``(out, aux)`` where ``out`` has shape
            ``[B, T, d_model]``, and ``aux`` is a dict containing:

            * ``"logits"`` -- ``[B, T, n_experts]`` raw router logits.
            * ``"weights"`` -- ``[B, T, n_experts]`` routing weights, summing to
              1 over the last dimension.
            * ``"entropy"`` -- scalar mean routing entropy.
            * ``"expert_outputs"`` -- ``[B, T, n_experts, d_model]``, present
              only when ``return_expert_outputs=True``.
        """
        weights, logits = self.router(x)

        if return_expert_outputs:
            # [B, T, E, D]
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
            out = (weights.unsqueeze(-1) * expert_outputs).sum(dim=2)
        else:
            out = torch.zeros_like(x)
            for index, expert in enumerate(self.experts):
                out = out + expert(x) * weights[..., index].unsqueeze(-1)

        aux: dict[str, torch.Tensor] = {
            "logits": logits,
            "weights": weights,
            "entropy": routing_entropy(logits),
        }
        if return_expert_outputs:
            aux["expert_outputs"] = expert_outputs

        return out, aux

    def extra_repr(self) -> str:
        """Return a compact representation for ``print(model)``."""
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, n_experts={self.n_experts}, "
            f"temperature={self.router.temperature}, top_k={self.router.top_k}"
        )
