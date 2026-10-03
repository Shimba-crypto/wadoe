"""Gradient flow through the router and every expert."""

from __future__ import annotations

import torch

from wadoe import WADOELayer, load_balance_loss, set_seed

B, T, D, E, RANK = 2, 8, 16, 4, 4


def _layer(top_k: int | None) -> WADOELayer:
    set_seed(0)
    return WADOELayer(d_model=D, d_ff=32, n_experts=E, rank=RANK, top_k=top_k)


def _forward_backward(layer: WADOELayer, seed: int = 0):
    set_seed(seed)
    x = torch.randn(B, T, D, requires_grad=True)
    out, aux = layer(x)
    loss = out.pow(2).mean()
    loss.backward()
    return x, out, aux, loss


def test_every_router_parameter_receives_gradient() -> None:
    """The router must not be silently detached from the graph."""
    layer = _layer(top_k=None)
    _forward_backward(layer)

    params = list(layer.router.parameters())
    assert params, "router has no parameters"
    for name, param in layer.router.named_parameters():
        assert param.grad is not None, f"router.{name} has no gradient"
        assert torch.all(torch.isfinite(param.grad)), f"router.{name} has non-finite grad"
        assert param.grad.abs().sum() > 0, f"router.{name} has an all-zero gradient"


def test_every_expert_parameter_receives_gradient() -> None:
    """With dense routing all experts are active, so all must get gradients."""
    layer = _layer(top_k=None)
    _forward_backward(layer)

    for index, expert in enumerate(layer.experts):
        for name, param in expert.named_parameters():
            assert param.grad is not None, f"experts[{index}].{name} has no gradient"
            assert torch.all(torch.isfinite(param.grad)), (
                f"experts[{index}].{name} has non-finite grad"
            )


def test_top_k_routing_produces_finite_gradients() -> None:
    """Sparse top-k routing must not introduce NaN or Inf gradients."""
    layer = _layer(top_k=2)
    _forward_backward(layer)

    for index, expert in enumerate(layer.experts):
        for name, param in expert.named_parameters():
            if param.grad is None:
                # An expert unused by top-k in this batch legitimately has no grad.
                continue
            assert torch.all(torch.isfinite(param.grad)), (
                f"experts[{index}].{name} has non-finite grad under top_k routing"
            )

    for name, param in layer.router.named_parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad)), f"router.{name} has non-finite grad"


def test_gradients_finite_across_many_batches_with_top_k() -> None:
    """Repeated batches under top-k routing stay finite and non-zero."""
    layer = _layer(top_k=1)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)

    for step in range(5):
        set_seed(100 + step)
        out, _ = layer(torch.randn(B, T, D))
        loss = out.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()

        for name, param in layer.named_parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), f"step {step}: {name}"
        optimizer.step()

    assert torch.all(torch.isfinite(loss))


def test_low_rank_branch_activates_after_first_step() -> None:
    """`a` has zero gradient at init (because `b` starts at zero) then becomes active."""
    layer = _layer(top_k=None)
    optimizer = torch.optim.Adam(layer.experts.parameters(), lr=1e-1)

    set_seed(0)
    out, _ = layer(torch.randn(B, T, D))
    out.pow(2).mean().backward()

    # At initialisation the low-rank term contributes nothing, so d L / d a == 0.
    for expert in layer.experts:
        assert expert.a.grad is not None
        assert torch.allclose(expert.a.grad, torch.zeros_like(expert.a))
        assert expert.b.grad.abs().sum() > 0

    optimizer.step()

    # After `b` becomes non-zero the `a` matrices start receiving gradient too.
    set_seed(1)
    out, _ = layer(torch.randn(B, T, D))
    out.pow(2).mean().backward()

    for expert in layer.experts:
        assert expert.b.abs().sum() > 0
        assert expert.a.grad is not None
        assert expert.a.grad.abs().sum() > 0


def test_gradients_flow_to_input() -> None:
    """The layer must not detach its input."""
    layer = _layer(top_k=2)
    x, _, _, _ = _forward_backward(layer)

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert x.grad.abs().sum() > 0


def test_training_step_reduces_loss() -> None:
    """A short optimisation run should actually decrease the objective."""
    set_seed(0)
    layer = WADOELayer(d_model=D, d_ff=32, n_experts=E, rank=RANK, top_k=2)
    head = torch.nn.Linear(D, 1)
    params = list(layer.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-3)

    set_seed(7)
    x = torch.randn(16, T, D)
    target = (x.mean(dim=(1, 2)) > 0).float().unsqueeze(-1)

    first = last = None
    for step in range(60):
        out, aux = layer(x)
        logits = head(out.mean(dim=1))
        task = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

        total = task + 0.01 * load_balance_loss(aux["logits"], aux["weights"])
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if step == 0:
            first = total.item()
        last = total.item()

    assert last < first
