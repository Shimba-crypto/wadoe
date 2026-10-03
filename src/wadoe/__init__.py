"""WADOE: a generic soft Mixture-of-Experts research library.

WADOE stands for *Weighted Attention-Deficit/Overactivity Experts*, the acronym
used in our research. The acronym is historical: the implementation shipped here
is a **domain-agnostic soft Mixture-of-Experts toolkit** and makes no
assumptions about, and makes no claims about, any medical or clinical
application. See :mod:`wadoe.losses` and the project README for details.

The central abstraction is :class:`WADOELayer`, which replaces a transformer
feed-forward sub-layer with ``n_experts`` bottleneck experts mixed by a learned
soft router::

    y = sum_i w_i(x) * E_i(x)

Quick start::

    import torch
    from wadoe import WADOELayer, wadoe_loss

    layer = WADOELayer(d_model=32, d_ff=64, n_experts=4, rank=8)
    out, aux = layer(torch.randn(2, 8, 32))

    task_loss = out.mean()
    total, parts = wadoe_loss(
        task_loss,
        logits=aux["logits"],
        weights=aux["weights"],
        experts=layer.experts,
    )
    total.backward()

Not a medical device. This library is not intended for clinical diagnosis,
treatment, or any other clinical use.
"""

from __future__ import annotations

from .experts import WADOEExpert
from .losses import (
    diversity_loss,
    load_balance_loss,
    router_aux_loss,
    routing_entropy,
    wadoe_loss,
)
from .model import WADOELayer
from .router import WADOERouter
from .utils import count_parameters, expert_usage, set_seed

__version__ = "0.1.0"

__all__ = [
    "WADOEExpert",
    "WADOERouter",
    "WADOELayer",
    "load_balance_loss",
    "diversity_loss",
    "routing_entropy",
    "router_aux_loss",
    "wadoe_loss",
    "count_parameters",
    "expert_usage",
    "set_seed",
    "__version__",
]
