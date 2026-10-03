"""Shape and structural guarantees for :class:`wadoe.WADOELayer`."""

from __future__ import annotations

import pytest
import torch

from wadoe import WADOEExpert, WADOELayer, WADOERouter, count_parameters, expert_usage, set_seed


def test_layer_preserves_shape() -> None:
    """The layer is shape-preserving: [B, T, D] in, [B, T, D] out."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=4, rank=8)
    x = torch.randn(2, 8, 32)

    out, aux = layer(x, return_expert_outputs=True)

    assert out.shape == (2, 8, 32)
    assert aux["weights"].shape == (2, 8, 4)
    assert aux["logits"].shape == (2, 8, 4)
    assert aux["expert_outputs"].shape == (2, 8, 4, 32)
    assert aux["entropy"].dim() == 0


@pytest.mark.parametrize("top_k", [1, 2, 3, None])
def test_weights_sum_to_one(top_k: int | None) -> None:
    """Routing weights are a convex combination over the expert axis."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=4, rank=8, top_k=top_k)
    x = torch.randn(2, 8, 32)

    _, aux = layer(x)

    weights = aux["weights"]
    assert weights.shape == (2, 8, 4)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_expert_outputs_omitted_by_default() -> None:
    """`return_expert_outputs=False` avoids the [B, T, E, D] memory cost."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=4, rank=8)
    x = torch.randn(2, 8, 32)

    out_sparse, aux_sparse = layer(x, return_expert_outputs=False)
    out_dense, aux_dense = layer(x, return_expert_outputs=True)

    assert "expert_outputs" not in aux_sparse
    assert "expert_outputs" in aux_dense
    # Both code paths must produce the same result.
    assert torch.allclose(out_sparse, out_dense, atol=1e-6)


def test_expert_shapes() -> None:
    """An expert maps [B, T, d_model] to [B, T, d_model]."""
    set_seed(0)
    expert = WADOEExpert(d_model=32, d_ff=64, rank=8)
    x = torch.randn(2, 8, 32)

    out = expert(x)

    assert out.shape == (2, 8, 32)
    assert expert.a.shape == (64, 8)
    assert expert.b.shape == (8, 32)


def test_expert_starts_as_plain_ffn() -> None:
    """`b` is zero-initialised, so the low-rank branch is inert at step 0."""
    set_seed(0)
    expert = WADOEExpert(d_model=32, d_ff=64, rank=8)

    assert torch.allclose(expert.b, torch.zeros_like(expert.b))

    # With b == 0 the low-rank term contributes exactly nothing.
    x = torch.randn(2, 8, 32)
    h = torch.nn.functional.gelu(expert.down(expert.norm(x)))
    expected = expert.up(h)
    assert torch.allclose(expert(x), expected, atol=1e-6)


def test_router_shapes() -> None:
    """The router returns weights and logits of shape [B, T, E]."""
    set_seed(0)
    router = WADOERouter(d_model=32, n_experts=5)
    x = torch.randn(2, 8, 32)

    weights, logits = router(x)

    assert weights.shape == (2, 8, 5)
    assert logits.shape == (2, 8, 5)


def test_top_k_at_or_above_n_experts_is_dense() -> None:
    """top_k >= n_experts is a no-op and falls back to dense routing."""
    set_seed(0)
    layer = WADOELayer(d_model=16, d_ff=32, n_experts=2, rank=4, top_k=2)
    assert layer.router.top_k is None

    _, aux = layer(torch.randn(2, 4, 16))
    assert torch.all(aux["weights"] > 0)


def test_count_parameters() -> None:
    """Parameter counting reports totals and honours requires_grad."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=4, rank=8)

    total, trainable = count_parameters(layer)
    expected = sum(p.numel() for p in layer.parameters())
    assert total == expected
    assert trainable == expected

    for param in layer.router.parameters():
        param.requires_grad_(False)
    _, frozen_trainable = count_parameters(layer)
    assert frozen_trainable < total


def test_expert_usage() -> None:
    """expert_usage returns per-expert frequencies summing to 1."""
    set_seed(0)
    layer = WADOELayer(d_model=32, d_ff=64, n_experts=4, rank=8, top_k=None)
    _, aux = layer(torch.randn(2, 8, 32))

    usage = expert_usage(aux["weights"])

    assert usage.shape == (4,)
    assert pytest.approx(1.0, abs=1e-5) == usage.sum().item()


def test_invalid_arguments_raise() -> None:
    """Invalid hyperparameters fail loudly rather than silently."""
    with pytest.raises(ValueError):
        WADOELayer(d_model=8, d_ff=16, n_experts=0)
    with pytest.raises(ValueError):
        WADOEExpert(d_model=8, d_ff=16, rank=0)
    with pytest.raises(ValueError):
        WADOEExpert(d_model=8, d_ff=16, dropout=1.5)
    with pytest.raises(ValueError):
        WADOERouter(d_model=8, n_experts=2, temperature=0.0)
    with pytest.raises(ValueError):
        WADOELayer(d_model=8, d_ff=16, n_experts=2)(torch.randn(1, 1, 7))
