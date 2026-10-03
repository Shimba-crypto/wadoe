"""Sparsity and temperature behaviour of the router."""

from __future__ import annotations

import math

import pytest
import torch

from wadoe import WADOELayer, WADOERouter, routing_entropy, set_seed

N_EXPERTS = 6


def test_top_k_keeps_at_most_k_nonzero() -> None:
    """With top_k=2 each token activates at most two experts."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=N_EXPERTS, rank=8, top_k=2)
    x = torch.randn(3, 16, 32)

    _, aux = layer(x)
    weights = aux["weights"]

    nonzero = (weights > 0).sum(dim=-1)
    assert torch.all(nonzero <= 2)
    # top_k=2 out of 6 experts: exactly two slots are used per token.
    assert torch.all(nonzero == 2)

    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_top_k_one_is_hard_selection() -> None:
    """top_k=1 reduces to hard argmax routing."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=N_EXPERTS, rank=8, top_k=1)
    _, aux = layer(torch.randn(3, 16, 32))
    weights = aux["weights"]

    assert torch.all((weights > 0).sum(dim=-1) == 1)
    assert torch.allclose(weights.max(dim=-1).values, torch.ones(3, 16), atol=1e-5)


def test_dense_routing_keeps_all_experts_active() -> None:
    """With top_k=None every expert receives strictly positive weight."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=N_EXPERTS, rank=8, top_k=None)
    _, aux = layer(torch.randn(3, 16, 32))
    weights = aux["weights"]

    assert torch.all(weights > 0)
    assert torch.all((weights > 0).sum(dim=-1) == N_EXPERTS)


def test_top_k_is_mutable() -> None:
    """top_k can be switched between sparse and dense without rebuilding."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=N_EXPERTS, rank=8, top_k=2)
    x = torch.randn(3, 16, 32)

    _, sparse = layer(x)
    assert torch.all((sparse["weights"] > 0).sum(dim=-1) == 2)

    layer.router.top_k = None
    _, dense = layer(x)
    assert torch.all((dense["weights"] > 0).sum(dim=-1) == N_EXPERTS)

    layer.router.top_k = 3
    _, sparse_again = layer(x)
    assert torch.all((sparse_again["weights"] > 0).sum(dim=-1) == 3)


def test_invalid_top_k_raises() -> None:
    """A top_k larger than n_experts is rejected at call time."""
    set_seed(0)
    router = WADOERouter(d_model=32, n_experts=4, top_k=2)
    x = torch.randn(2, 8, 32)

    router.top_k = 9
    with pytest.raises(ValueError, match="exceeds n_experts"):
        router(x)


def test_router_starts_near_uniform() -> None:
    """The small init std keeps initial routing close to uniform."""
    set_seed(0)
    router = WADOERouter(d_model=32, n_experts=N_EXPERTS)
    _, logits = router(torch.randn(64, 8, 32))

    # A uniform router has entropy log(E); we should be close to it.
    assert routing_entropy(logits).item() == pytest.approx(
        math.log(N_EXPERTS), abs=0.05
    )


def test_uniform_logits_have_max_entropy() -> None:
    """Zero logits give a uniform distribution of maximum entropy."""
    logits = torch.zeros(4, 5, N_EXPERTS)
    assert routing_entropy(logits).item() == pytest.approx(math.log(N_EXPERTS), abs=1e-6)


def test_temperature_changes_entropy_monotonically() -> None:
    """Lower temperature sharpens routing, so entropy decreases monotonically."""
    set_seed(0)
    base = torch.randn(4, 6, N_EXPERTS) * 3.0

    temperatures = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    entropies = [routing_entropy(base / t).item() for t in temperatures]

    # entropies must be non-decreasing as temperature grows.
    for low, high in zip(entropies, entropies[1:]):
        assert high >= low - 1e-6

    # Sharpening collapses toward one-hot (entropy 0), flattening approaches the
    # uniform maximum log(E). The gap to log(E) at large T decays like 1/T^2.
    assert entropies[0] < 0.1
    assert entropies[-1] < math.log(N_EXPERTS) + 1e-6
    assert entropies[-1] > math.log(N_EXPERTS) - 0.1


def test_temperature_attribute_changes_routing() -> None:
    """Setting router.temperature reshapes the returned weights."""
    set_seed(0)
    router = WADOERouter(d_model=32, n_experts=N_EXPERTS, top_k=None)
    x = torch.randn(4, 8, 32)

    router.temperature = 0.1
    sharp, _ = router(x)
    sharp_entropy = -(
        sharp * sharp.clamp_min(1e-12).log()
    ).sum(-1).mean()

    router.temperature = 8.0
    smooth, _ = router(x)
    smooth_entropy = -(
        smooth * smooth.clamp_min(1e-12).log()
    ).sum(-1).mean()

    assert sharp_entropy.item() < smooth_entropy.item()

    router.temperature = 0.0
    with pytest.raises(ValueError, match="temperature"):
        router(x)
