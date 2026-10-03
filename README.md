# WADOE

**W**eighted **A**ttention-**D**eficit/**O**veractivity **E**xperts

A small, dependency-light research library for **soft Mixture-of-Experts** layers. `WADOE` is the acronym used in our research; the implementation itself is **generic and domain-agnostic** — it is a soft-MoE toolkit with no assumptions about, and no claims about, any particular application. Each expert is a bottleneck FFN with a low-rank additive adaptation, and a learned router mixes them with per-token softmax weights.

> **Not a medical device.** This library is not intended for clinical diagnosis, treatment, or any other clinical use. See [Not a medical device](#not-a-medical-device).

---

## Why soft routing?

Hard top-1 Mixture-of-Experts routes every token to a single expert. That is efficient, but it makes training brittle: an unlucky router initialisation locks one expert in early, the rest never receive gradient, and the mixture silently degenerates into a single model.

Soft routing keeps every expert in the graph:

```
y = Σ_i w_i(x) · E_i(x)
```

where `E_i(x)` is expert `i` and `w_i(x)` is its learned weight for token `x`. Because the weights are a softmax, all experts with non-zero weight contribute a differentiable term, so gradients reach all of them from the first step. `wadoe` also ships the auxiliary losses that keep this from degenerating in the other direction — into experts that all learn the same function.

---

## Install

```bash
pip install wadoe
```

For a local checkout with the development extras (`pytest`, `ruff`, `build`, `twine`):

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. NumPy is the only other runtime dependency.

---

## Quickstart

```python
import torch
from wadoe import WADOELayer, wadoe_loss

layer = WADOELayer(
    d_model=512,
    d_ff=2048,
    n_experts=6,
    rank=16,
    temperature=1.0,
    top_k=2,          # None for dense soft routing
    dropout=0.0,
)

x = torch.randn(8, 128, 512)                 # [batch, tokens, d_model]
out, aux = layer(x)                          # out is [8, 128, 512]

task_loss = torch.nn.functional.mse_loss(out.mean(1), target)
total, parts = wadoe_loss(
    task_loss,
    logits=aux["logits"],
    weights=aux["weights"],
    experts=layer.experts,
    lambda_balance=0.01,
    lambda_div=0.05,
)

total.backward()

print(parts)                # unweighted diagnostics for logging
print(aux["weights"].shape) # [8, 128, 6]
```

`WADOELayer` is **shape-preserving**: `[B, T, d_model]` in, `[B, T, d_model]` out. Drop it into a transformer block in place of the FFN sub-layer and keep attention dense — that is the cheap 80% of the benefit.

### Public API

| Symbol | Purpose |
| --- | --- |
| `WADOELayer` | Expert stack + router; the main drop-in layer |
| `WADOEExpert` | A single bottleneck expert with low-rank adaptation |
| `WADOERouter` | Soft (optionally top-k) token-level router |
| `load_balance_loss` | Penalises router collapse |
| `diversity_loss` | Penalises expert homogenisation |
| `routing_entropy` | Diagnostic: how peaked is routing? |
| `router_aux_loss` | Optional supervised routing signal |
| `wadoe_loss` | Combines all of the above with the task loss |
| `count_parameters` | Total / trainable parameter counts |
| `expert_usage` | Per-expert routing frequency |
| `set_seed` | Reproducible seeding across Python, NumPy, PyTorch |

---

## Losses

`wadoe_loss` combines a task loss with three auxiliary terms:

```
L = L_task + λ_balance · L_balance + λ_div · L_diversity + λ_route · L_route
```

| Loss | Range | Role |
| --- | --- | --- |
| `load_balance_loss` | `1.0` → `E` | **Router collapse.** `1.0` at uniform routing, `E` when one expert receives everything. Minimising it keeps traffic spread across experts. |
| `diversity_loss` | `0.0` → `1 − 1/E` | **Expert homogenisation.** `0.0` when the experts' low-rank factors are mutually orthogonal (the goal); maximal when every expert has aligned onto the same direction. |
| `router_aux_loss` | `≥ 0` | **Supervision.** Cross-entropy against known per-sequence expert/domain labels, teaching the router which expert *should* fire. Only active when you pass `domain_labels`. |
| `routing_entropy` | `0` → `log E` | *Diagnostic, not a penalty.* `0` means hard collapse onto one expert; `log E` means the router is averaging uniformly and wasting the mixture. You generally want the middle. |

A working schedule: hold `lambda_balance` high (≈ `0.1`) for the first few hundred steps while the router is still near-uniform, then relax it to `0.01`.

### A note on `diversity_loss`

The penalty is `(G − I)²` on the Gram matrix `G` of the L2-normalised low-rank `a` factors, so it measures **deviation from orthogonality**. Consequently identical experts score near the *maximum* (`1 − 1/E`), not zero — minimising the term pushes experts apart rather than together. This is deliberate; the tests assert both endpoints.

### A note on `router_aux_loss`

Pool per-token logits over the token axis *before* the cross-entropy:

```python
pooled = aux["logits"].mean(dim=1)     # [B, E] — pool FIRST
loss = router_aux_loss(pooled, domain_labels)
```

Do **not** pass `aux["logits"]` (shape `[B, T, E]`) directly, and do **not** pass `aux["logits"].argmax(-1)`: the first is a shape error, and the second throws away the gradient the router needs to learn. `wadoe` raises a descriptive `ValueError` for both.

---

## Training recipe

1. **Warm up with the mixture only.** Freeze the backbone entirely; train router + experts. The router is tiny and starts near-uniform, so give it a higher LR than the experts (`1e-3` vs `5e-4` is a reasonable start).
2. **Unfreeze the backbone LoRA.** Typical LR split: router `1e-3`, experts `1e-4`…`5e-4`, backbone LoRA `3e-5`…`1e-4`.
3. **Full unfreeze (optional).** Very small LR (`5e-6`) for a final epoch if the loss plateaus.
4. **Watch two numbers every few hundred steps** — expert usage and routing entropy. They catch both collapse modes immediately:

```python
usage = expert_usage(aux["weights"])      # 1-D, sums to 1
entropy = routing_entropy(aux["logits"])  # scalar
```

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| One expert > 80% of usage | Router collapse | Raise `lambda_balance`, raise router LR, reset router weights, raise temperature |
| Entropy pinned at 0 | Hard collapse | Same as above |
| Entropy pinned at `log E` | Router ignoring the mixture | Lower temperature, sharpen the task signal |
| Usage even but accuracy poor | Experts not specialising | Fewer experts, lower rank, or labels that carry no exploitable structure |
| `diversity_loss` near `1 − 1/E` | Expert homogenisation | Raise `lambda_div`, raise `rank` |

**Placement tip:** putting WADOE in the top half of the layers only (`layers[L//2:]`) is usually better value than everywhere — early-layer experts tend to collapse.

---

## Example

`examples/adhd_classification.py` is a self-contained, runnable script that generates synthetic data, trains a `WADOELayer` behind a classifier head, and prints expert usage and routing entropy each epoch.

```bash
python examples/adhd_classification.py
```

It uses **synthetic random features with an arbitrary label rule and no real patient data of any kind**. Its purpose is to show how the routing diagnostics behave, nothing else. Read the disclaimer at the top of the file before running it.

---

## Not a medical device

This library is provided for software research and experimentation. It is:

- **not a medical device**;
- **not approved, cleared, or registered** for any medical purpose;
- **not intended** for clinical diagnosis, treatment, screening, triage, monitoring, or decision support, of ADHD or of any other condition;
- **not validated** on any clinical dataset.

The acronym references our research history and nothing more. The core package makes no domain assumptions whatsoever and is intended for general sequence-modelling work.

Any clinical application would require, at minimum: ethics approval, privacy and security review, external validation on held-out cohorts, subgroup fairness analysis, calibrated uncertainty reporting, human oversight, and appropriate regulatory review. None of that is in scope here, and nothing in this repository should be read as evidence for it.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
python -m build
```

## Citation

> **To be added.** If you use WADOE in published work, please cite it. A BibTeX entry and the canonical reference will be included here when the paper is released.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
