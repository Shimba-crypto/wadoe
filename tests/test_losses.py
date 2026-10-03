"""Behaviour of the auxiliary objectives in :mod:`wadoe.losses`."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from wadoe import (
    WADOEExpert,
    WADOELayer,
    diversity_loss,
    load_balance_loss,
    router_aux_loss,
    routing_entropy,
    set_seed,
    wadoe_loss,
)

N_EXPERTS = 4


# --------------------------------------------------------------------------- #
# load_balance_loss
# --------------------------------------------------------------------------- #
def test_load_balance_is_one_at_uniform() -> None:
    """All-zero logits are uniform, which is the minimum of the penalty."""
    logits = torch.zeros(2, 3, N_EXPERTS)
    weights = torch.softmax(logits, dim=-1)

    assert load_balance_loss(logits, weights).item() == pytest.approx(1.0, abs=1e-6)


def test_load_balance_approaches_n_experts_at_collapse() -> None:
    """Routing everything to one expert is the maximum of the penalty."""
    logits = torch.zeros(2, 3, N_EXPERTS)
    logits[..., 0] = 100.0
    weights = torch.softmax(logits, dim=-1)

    value = load_balance_loss(logits, weights).item()

    assert value == pytest.approx(float(N_EXPERTS), abs=1e-4)
    assert value > 1.0


def test_load_balance_interpolates_monotonically() -> None:
    """The penalty grows smoothly with how skewed routing becomes."""
    values = []
    for scale in [0.0, 0.5, 1.0, 2.0, 4.0]:
        logits = torch.zeros(8, 16, N_EXPERTS)
        logits[..., 0] = scale
        weights = torch.softmax(logits, dim=-1)
        values.append(load_balance_loss(logits, weights).item())

    for low, high in zip(values, values[1:]):
        assert high >= low - 1e-6
    assert values[0] == pytest.approx(1.0, abs=1e-6)


def test_load_balance_rejects_bad_shapes() -> None:
    """Inconsistent or non-3-D router tensors are rejected."""
    logits = torch.zeros(2, 3, N_EXPERTS)

    with pytest.raises(ValueError, match=r"\[B, T, E\]"):
        load_balance_loss(torch.zeros(N_EXPERTS), torch.zeros(N_EXPERTS))
    with pytest.raises(ValueError, match="!="):
        load_balance_loss(logits, torch.zeros(2, 3, N_EXPERTS + 1))


# --------------------------------------------------------------------------- #
# diversity_loss
# --------------------------------------------------------------------------- #
def _orthogonal_experts(n: int, size: int = 8) -> list[WADOEExpert]:
    """Build ``n`` experts whose `a` matrices are mutually orthogonal.

    Expert ``i`` is given the flattened standard basis vector ``e_i``, so the
    Gram matrix of the L2-normalised factors is exactly the identity.
    """
    experts = [WADOEExpert(d_model=size, d_ff=n, rank=n) for _ in range(n)]
    for i, expert in enumerate(experts):
        with torch.no_grad():
            expert.a.zero_()
            expert.a[i, 0] = 1.0
    return experts


def test_diversity_loss_is_zero_for_orthogonal_experts() -> None:
    """Orthogonal low-rank factors are the desired state: zero penalty."""
    experts = _orthogonal_experts(N_EXPERTS)

    assert diversity_loss(experts).item() == pytest.approx(0.0, abs=1e-6)


def test_diversity_loss_penalises_identical_experts() -> None:
    """Identical experts are maximally penalised, not zero.

    ``(G - I).mean(square)`` measures deviation from orthogonality, so aligned
    experts sit at the *top* of the penalty range rather than the bottom. For
    ``n`` identical experts the value is exactly ``1 - 1/n``.
    """
    experts = [WADOEExpert(d_model=8, d_ff=N_EXPERTS, rank=N_EXPERTS) for _ in range(N_EXPERTS)]
    shared = torch.randn(N_EXPERTS, N_EXPERTS)
    for expert in experts:
        with torch.no_grad():
            expert.a.copy_(shared)

    value = diversity_loss(experts).item()

    assert value == pytest.approx(1.0 - 1.0 / N_EXPERTS, abs=1e-6)
    assert value > 0.0


def test_diversity_loss_orders_orthogonal_below_identical() -> None:
    """Aligned experts always score worse than orthogonal ones."""
    aligned = [WADOEExpert(d_model=8, d_ff=N_EXPERTS, rank=N_EXPERTS) for _ in range(N_EXPERTS)]
    shared = torch.randn(N_EXPERTS, N_EXPERTS)
    for expert in aligned:
        with torch.no_grad():
            expert.a.copy_(shared)

    orthogonal = diversity_loss(_orthogonal_experts(N_EXPERTS)).item()
    identical = diversity_loss(aligned).item()

    assert 0.0 <= orthogonal < identical <= 1.0


def test_diversity_loss_detects_partial_alignment() -> None:
    """Making two experts identical raises the penalty above the orthogonal case."""
    experts = _orthogonal_experts(N_EXPERTS)
    baseline = diversity_loss(experts).item()

    with torch.no_grad():
        experts[1].a.copy_(experts[0].a)

    assert diversity_loss(experts).item() > baseline


def test_diversity_loss_rejects_non_expert_modules() -> None:
    """Modules without a low-rank `a` parameter produce a clear error."""
    with pytest.raises(ValueError, match="has no 'a' parameter"):
        diversity_loss([nn.Linear(4, 4), nn.Linear(4, 4)])


# --------------------------------------------------------------------------- #
# routing_entropy
# --------------------------------------------------------------------------- #
def test_routing_entropy_bounds() -> None:
    """Entropy lives between 0 (collapsed) and log(E) (uniform)."""
    uniform = torch.zeros(4, 8, N_EXPERTS)
    assert routing_entropy(uniform).item() == pytest.approx(math.log(N_EXPERTS), abs=1e-6)

    collapsed = torch.zeros(4, 8, N_EXPERTS)
    collapsed[..., 0] = 100.0
    assert routing_entropy(collapsed).item() == pytest.approx(0.0, abs=1e-6)


def test_routing_entropy_increases_with_temperature() -> None:
    """Flattening the distribution with temperature raises entropy."""
    set_seed(0)
    logits = torch.randn(4, 8, N_EXPERTS) * 2.0

    sharp = routing_entropy(logits / 0.2).item()
    soft = routing_entropy(logits / 4.0).item()

    assert sharp < soft
    assert soft <= math.log(N_EXPERTS) + 1e-6


# --------------------------------------------------------------------------- #
# router_aux_loss
# --------------------------------------------------------------------------- #
def test_router_aux_loss_runs_on_pooled_logits() -> None:
    """Pooled [B, E] logits with integer [B] labels are accepted."""
    set_seed(0)
    pooled_logits = torch.randn(6, N_EXPERTS, requires_grad=True)
    labels = torch.randint(0, N_EXPERTS, (6,))

    loss = router_aux_loss(pooled_logits, labels)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pooled_logits.grad is not None
    assert torch.all(torch.isfinite(pooled_logits.grad))


def test_router_aux_loss_pooling_from_token_logits() -> None:
    """Pooling per-token logits by the mean produces a valid input."""
    set_seed(0)
    logits = torch.randn(6, 10, N_EXPERTS)
    labels = torch.randint(0, N_EXPERTS, (6,))

    loss = router_aux_loss(logits.mean(dim=1), labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_router_aux_loss_rejects_unpooled_logits() -> None:
    """Passing raw [B, T, E] logits is an error, per the documented warning."""
    logits = torch.randn(6, 10, N_EXPERTS)
    labels = torch.randint(0, N_EXPERTS, (6,))

    with pytest.raises(ValueError, match="pooled over the token dimension"):
        router_aux_loss(logits, labels)


def test_router_aux_loss_rejects_argmax() -> None:
    """Hard argmax indices must not be passed as logits; they lose gradient.

    ``logits.argmax(-1)`` on ``[B, T, E]`` yields ``[B, T]`` of integers: 2-D, so
    it survives a naive shape check, but it is a hard class index rather than a
    score. The dtype guard is what catches it.
    """
    logits = torch.randn(6, 10, N_EXPERTS)
    labels = torch.randint(0, N_EXPERTS, (6,))

    with pytest.raises(ValueError, match="floating-point scores"):
        router_aux_loss(logits.argmax(dim=-1), labels)


def test_router_aux_loss_rejects_bad_labels() -> None:
    """Out-of-range labels and shape mismatches are rejected."""
    pooled_logits = torch.randn(6, N_EXPERTS)

    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        router_aux_loss(pooled_logits, torch.full((6,), N_EXPERTS))
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        router_aux_loss(pooled_logits, torch.full((6,), -1))
    with pytest.raises(ValueError, match="batch mismatch"):
        router_aux_loss(pooled_logits, torch.zeros(5, dtype=torch.long))


# --------------------------------------------------------------------------- #
# wadoe_loss
# --------------------------------------------------------------------------- #
def test_wadoe_loss_returns_scalar_and_components() -> None:
    """The helper returns a differentiable scalar plus unweighted diagnostics."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=N_EXPERTS, rank=8)
    x = torch.randn(4, 8, 32)
    out, aux = layer(x)

    task_loss = out.pow(2).mean()
    total, components = wadoe_loss(
        task_loss,
        logits=aux["logits"],
        weights=aux["weights"],
        experts=layer.experts,
    )

    assert total.dim() == 0
    assert torch.isfinite(total)
    for key in ["task", "balance", "diversity", "balance_weighted", "diversity_weighted"]:
        assert key in components

    expected = (
        task_loss
        + 0.01 * load_balance_loss(aux["logits"], aux["weights"])
        + 0.05 * diversity_loss(layer.experts)
    )
    assert total.item() == pytest.approx(expected.item(), abs=1e-6)


def test_wadoe_loss_includes_route_term_when_labels_given() -> None:
    """Domain labels enable the supervised routing auxiliary loss."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=N_EXPERTS, rank=8)
    x = torch.randn(4, 8, 32)
    out, aux = layer(x)

    labels = torch.randint(0, N_EXPERTS, (4,))
    total, components = wadoe_loss(
        out.pow(2).mean(),
        logits=aux["logits"],
        weights=aux["weights"],
        experts=layer.experts,
        domain_labels=labels,
        lambda_route=0.1,
    )

    assert "route" in components
    assert components["route"].item() > 0.0

    pooled = aux["logits"].mean(dim=1)
    expected = (
        out.pow(2).mean()
        + 0.01 * load_balance_loss(aux["logits"], aux["weights"])
        + 0.05 * diversity_loss(layer.experts)
        + 0.1 * router_aux_loss(pooled, labels)
    )
    assert total.item() == pytest.approx(expected.item(), abs=1e-6)


def test_wadoe_loss_omits_route_term_without_labels() -> None:
    """Without domain labels the routing term contributes nothing."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=N_EXPERTS, rank=8)
    _, aux = layer(torch.randn(4, 8, 32))

    total, components = wadoe_loss(
        torch.tensor(1.0),
        logits=aux["logits"],
        weights=aux["weights"],
        experts=layer.experts,
        lambda_route=1.0,
    )

    assert components["route"].item() == 0.0
    assert total.item() == pytest.approx(1.0 + 0.01 * components["balance"].item() + 0.05 * components["diversity"].item(), abs=1e-6)


def test_wadoe_loss_rejects_non_scalar_task_loss() -> None:
    """A non-scalar task loss is a caller bug."""
    set_seed(0)
    layer = WADOELayer(d_model=8, d_ff=16, n_experts=2, rank=4)
    _, aux = layer(torch.randn(2, 4, 8))

    with pytest.raises(ValueError, match="scalar"):
        wadoe_loss(
            torch.zeros(2, 4),
            logits=aux["logits"],
            weights=aux["weights"],
            experts=layer.experts,
        )
