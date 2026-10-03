"""Auxiliary objectives for training WADOE soft Mixture-of-Experts layers.

Routing a mixture of experts is unstable without supervision. Two failure modes
dominate, and each has a dedicated penalty here:

* **Router collapse** -- the router sends (almost) every token to a single
  expert, so the other experts receive no gradient and never learn. Penalised by
  :func:`load_balance_loss`.
* **Expert homogenisation** -- every expert converges to the same function.
  Penalised by :func:`diversity_loss`.

:func:`routing_entropy` is a diagnostic rather than a penalty, and
:func:`router_aux_loss` is an optional supervised signal that teaches the router
which expert *should* handle a given sequence.

All functions assume the layout ``[batch, tokens, experts]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "load_balance_loss",
    "diversity_loss",
    "routing_entropy",
    "router_aux_loss",
    "wadoe_loss",
]


def load_balance_loss(logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Encourage uniform expert usage across the batch.

    Implements the Switch-Transformer style importance/coefficient-of-variation
    proxy. Let ``f`` be the mean routing weight per expert (how much traffic each
    expert actually receives) and ``P`` the mean router probability per expert
    (how much traffic it was nominally offered)::

        L = n_experts * (f * P).sum()

    Properties:

    * ``L == 1.0`` at perfectly uniform routing (every token equally likely to
      reach every expert).
    * ``L == n_experts`` at total collapse (one expert receives everything).

    So the loss is minimised at uniform and maximised at collapse, and adding it
    to the task loss actively fights collapse.

    Args:
        logits: Raw router logits of shape ``[B, T, n_experts]``, i.e. the
            pre-temperature output of :meth:`wadoe.WADOERouter.forward`.
        weights: Corresponding routing weights of shape ``[B, T, n_experts]``.
            With ``top_k`` routing these are the renormalised sparse weights,
            which still sum to 1 across the last dimension.

    Returns:
        Scalar tensor in ``[1.0, n_experts]``.

    Raises:
        ValueError: If shapes are inconsistent or ``logits`` is not 3-D.
    """
    if logits.dim() != 3:
        raise ValueError(f"expected logits of shape [B, T, E], got {tuple(logits.shape)}")
    if logits.shape != weights.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != weights shape {tuple(weights.shape)}"
        )

    n_experts = weights.shape[-1]
    # Empirical traffic per expert.
    importance = weights.mean(dim=(0, 1))
    # Average router probability per expert.
    probability = F.softmax(logits, dim=-1).mean(dim=(0, 1))
    return n_experts * (importance * probability).sum()


def diversity_loss(experts: Sequence[nn.Module]) -> torch.Tensor:
    """Penalise experts whose low-rank factors point in the same direction.

    Builds the Gram matrix of the L2-normalised, flattened ``a`` matrices of the
    given experts and penalises deviation from the identity::

        G = normalize([a_1, ..., a_E]) @ normalize([a_1, ..., a_E]).T
        L = (G - I).pow(2).mean()

    Reading the value:

    * ``L == 0`` when the ``a`` matrices are mutually **orthogonal** -- the
      desired state, since each expert occupies its own subspace.
    * ``L == 1`` when they are mutually **identical** (or all equal to a single
      shared direction) -- the maximum penalty, which is what expert
      homogenisation looks like.

    Note that identical experts do *not* score zero; the penalty is maximal
    there by design. Minimising this term pushes experts apart rather than
    pushing them together.

    Args:
        experts: Sequence or ``ModuleList`` of :class:`wadoe.WADOEExpert`
            modules. Must be non-empty.

    Returns:
        Scalar tensor in ``[0, 1]``.

    Raises:
        ValueError: If ``experts`` is empty or any expert lacks a ``.a``
            parameter or has a mismatched ``a`` shape.
    """
    matrices: list[torch.Tensor] = []
    for index, expert in enumerate(experts):
        a = getattr(expert, "a", None)
        if a is None:
            raise ValueError(
                f"experts[{index}] of type {type(expert).__name__} has no 'a' parameter; "
                "diversity_loss requires WADOEExpert modules."
            )
        matrices.append(a.flatten())

    reference = matrices[0]
    for index, matrix in enumerate(matrices):
        if matrix.shape != reference.shape:
            raise ValueError(
                f"experts[{index}] has 'a' with {matrix.numel()} elements, "
                f"expected {reference.numel()}"
            )

    stacked = F.normalize(torch.stack(matrices), dim=-1)
    gram = stacked @ stacked.t()
    n_experts = gram.shape[0]
    identity = torch.eye(n_experts, device=gram.device, dtype=gram.dtype)
    return (gram - identity).pow(2).mean()


def routing_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean per-token entropy of the router distribution.

    Use this as a diagnostic. Entropy near ``log(n_experts)`` means the router is
    spreading mass evenly (every expert gets used, none dominates). Entropy near
    ``0`` means collapse onto a single expert. An ideal soft-MoE sits in the
    middle: clearly differentiated, but not one-hot.

    Note that this expects *raw* logits. If you want the entropy of the
    temperature-scaled distribution, pass ``logits / temperature``.

    Args:
        logits: Router logits of shape ``[B, T, n_experts]`` or ``[*, n_experts]``.

    Returns:
        Scalar tensor with the mean token entropy in nats.
    """
    log_prob = F.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    return -(prob * log_prob).sum(dim=-1).mean()


def router_aux_loss(pooled_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    """Supervise the router with known per-sequence expert/domain labels.

    .. warning::
       ``pooled_logits`` **must already be pooled over the token dimension**,
       i.e. shape ``[B, n_experts]``. Passing per-token logits of shape
       ``[B, T, n_experts]`` alongside labels of shape ``[B]`` is a shape error
       at best.

    .. warning::
       Do **not** pass ``logits.argmax(-1)`` as ``pooled_logits``. ``argmax``
       produces integer class indices of shape ``[B]``, which
       :func:`torch.nn.functional.cross_entropy` will reject (it expects
       floating-point logits, not already-selected indices), and hard argmax also
       removes the gradient signal the router needs to learn.

    Pool per-token logits first, then take the cross-entropy::

        pooled_logits = logits.mean(dim=1)          # [B, E]  <- pool FIRST
        loss = F.cross_entropy(pooled_logits, labels)  # [B] int64

    Args:
        pooled_logits: Token-pooled router logits of shape ``[B, n_experts]``,
            typically ``logits.mean(dim=1)``.
        domain_labels: Integer targets of shape ``[B]`` with values in
            ``[0, n_experts)`` and dtype ``torch.long``.

    Returns:
        Scalar cross-entropy tensor.

    Raises:
        ValueError: If ``pooled_logits`` is not 2-D or not floating-point, shapes
            disagree, or labels are out of range.
    """
    if pooled_logits.dim() != 2:
        raise ValueError(
            "pooled_logits must be pooled over the token dimension and have shape "
            f"[B, n_experts], got {tuple(pooled_logits.shape)}. "
            "Use logits.mean(dim=1) rather than passing per-token logits."
        )
    if not pooled_logits.is_floating_point():
        raise ValueError(
            f"pooled_logits must be floating-point scores, got dtype {pooled_logits.dtype}. "
            "An integer tensor such as logits.argmax(-1) is a hard class index, not a "
            "score; pass the pooled logits themselves (e.g. logits.mean(dim=1))."
        )
    if domain_labels.dim() != 1:
        raise ValueError(f"domain_labels must have shape [B], got {tuple(domain_labels.shape)}")
    if pooled_logits.shape[0] != domain_labels.shape[0]:
        raise ValueError(
            f"batch mismatch: pooled_logits has {pooled_logits.shape[0]} rows but "
            f"domain_labels has {domain_labels.shape[0]}"
        )
    n_experts = pooled_logits.shape[-1]
    if domain_labels.numel() and (
        int(domain_labels.min()) < 0 or int(domain_labels.max()) >= n_experts
    ):
        raise ValueError(f"domain_labels must lie in [0, {n_experts})")

    return F.cross_entropy(pooled_logits, domain_labels.long())


def wadoe_loss(
    task_loss: torch.Tensor,
    logits: torch.Tensor,
    weights: torch.Tensor,
    experts: Iterable[nn.Module],
    domain_labels: torch.Tensor | None = None,
    pooled_logits: torch.Tensor | None = None,
    lambda_balance: float = 0.01,
    lambda_div: float = 0.05,
    lambda_route: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine the task loss with the WADOE auxiliary objectives.

    Computes::

        L = L_task
          + lambda_balance * L_balance
          + lambda_div     * L_diversity
          + lambda_route   * L_route

    The routing auxiliary term is included only when ``domain_labels`` is given,
    or when ``lambda_route > 0``.

    A sensible schedule is to raise ``lambda_balance`` substantially (e.g. 0.1)
    for the first few hundred steps while the router is still near-uniform, then
    relax it to 0.01 for the remainder of training.

    Args:
        task_loss: Scalar primary objective, e.g. cross-entropy from the task
            head.
        logits: Raw router logits ``[B, T, n_experts]`` from
            :class:`wadoe.WADOELayer` aux.
        weights: Routing weights ``[B, T, n_experts]`` from the same aux.
        experts: The expert modules, used by the diversity penalty.
        domain_labels: Optional integer expert/domain labels ``[B]``. Enables
            the routing auxiliary term.
        pooled_logits: Optional pre-pooled router logits ``[B, n_experts]``. If
            omitted while ``domain_labels`` is given, computed as
            ``logits.mean(dim=1)``.
        lambda_balance: Weight on the load-balancing penalty.
        lambda_div: Weight on the expert diversity penalty.
        lambda_route: Weight on the supervised routing auxiliary loss.

    Returns:
        A tuple ``(total_loss, components)``. ``total_loss`` is a scalar tensor
        ready for ``backward()``. ``components`` maps the name of each term to
        its *unweighted* value, for logging; it also contains the weighted
        ``"balance"``, ``"diversity"`` and ``"route"`` contributions as
        ``"balance_weighted"``, ``"diversity_weighted"`` and ``"route_weighted"``.

    Raises:
        ValueError: If ``task_loss`` is not a scalar, or if router tensor shapes
            are inconsistent.
    """
    if task_loss.dim() != 0:
        raise ValueError(f"task_loss must be a scalar, got shape {tuple(task_loss.shape)}")

    balance = load_balance_loss(logits, weights)
    diversity = diversity_loss(list(experts))

    total = task_loss + lambda_balance * balance + lambda_div * diversity
    components: dict[str, torch.Tensor] = {
        "task": task_loss.detach(),
        "balance": balance.detach(),
        "diversity": diversity.detach(),
        "balance_weighted": (lambda_balance * balance).detach(),
        "diversity_weighted": (lambda_div * diversity).detach(),
        "route": torch.zeros((), device=task_loss.device),
    }

    if domain_labels is not None:
        if pooled_logits is None:
            # Pool over the token dimension before cross-entropy.
            pooled_logits = logits.mean(dim=1)
        route = router_aux_loss(pooled_logits, domain_labels)
        if lambda_route != 0.0:
            total = total + lambda_route * route
        components["route"] = route.detach()
        components["route_weighted"] = (lambda_route * route).detach()

    return total, components
