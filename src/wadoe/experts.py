"""Expert modules for WADOE soft Mixture-of-Experts layers.

Each :class:`WADOEExpert` is a bottleneck feed-forward network (the standard
transformer FFN) plus a rank-constrained additive correction. The low-rank term
is applied at the **output** dimension (``d_model``), which keeps its parameter
count independent of the hidden width growth and lets it act as a cheap
specialisation on top of the shared FFN path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WADOEExpert"]


class WADOEExpert(nn.Module):
    """A single bottleneck expert with a low-rank additive adaptation.

    Architecture::

        x -> LayerNorm -> Linear(d_model, d_ff) -> GELU -> dropout
          -> Linear(d_ff, d_model) + low_rank(h)
          -> [B, T, d_model]

    where the low-rank branch is ``(h @ a) @ b * scale`` with ``a`` of shape
    ``[d_ff, rank]`` and ``b`` of shape ``[rank, d_model]``.

    ``a`` is initialised from ``N(0, 0.02)`` and ``b`` from zeros, so at
    initialisation the expert is numerically identical to a plain FFN and the
    low-rank branch only becomes active once ``b`` receives gradient.

    Args:
        d_model: Model (hidden) dimension.
        d_ff: Bottleneck intermediate dimension.
        rank: Rank of the low-rank adaptation. Must be positive.
        dropout: Dropout probability applied after GELU.

    Raises:
        ValueError: If ``rank`` or ``dropout`` is invalid, or ``d_ff < d_model``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rank: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError(f"d_model and d_ff must be positive, got {d_model}, {d_ff}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.rank = int(rank)
        self.scale = float(d_ff) ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Low-rank adaptation at the output dimension.
        self.a = nn.Parameter(torch.zeros(d_ff, rank))
        self.b = nn.Parameter(torch.zeros(rank, d_model))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise weights so the expert starts as a plain FFN.

        The low-rank branch is a no-op at initialisation because ``b`` is zero.
        """
        nn.init.normal_(self.a, mean=0.0, std=0.02)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the expert on a sequence of hidden states.

        Args:
            x: Input tensor of shape ``[B, T, d_model]``.

        Returns:
            Tensor of shape ``[B, T, d_model]``.
        """
        h = self.down(self.norm(x))  # [B, T, d_ff]
        h = F.gelu(h)
        h = self.dropout(h)

        out = self.up(h)  # [B, T, d_model]
        out = out + (h @ self.a) @ self.b * self.scale
        return out

    def extra_repr(self) -> str:
        """Return a compact representation for ``print(model)``."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}, rank={self.rank}"
