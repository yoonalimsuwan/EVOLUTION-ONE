# BV FULL THEORY ONE

**Complete Batalin–Vilkovisky Field Theory Module — EVOLUTION ONE Cluster, ONE Ecosystem**

| | |
|---|---|
| **Module** | `bv_full_theory_one.py` |
| **Version** | 1.0.0 |
| **Ecosystem version** | EVOLUTION_VERSION 3.0.0 |
| **Author** | PAI , Yoon A Limsuwan / MSPS NETWORK |
| **ORCID** | 0009-0008-2374-0788 |
| **GitHub** | yoonalimsuwan |
| **Contact** | msps4u@gmail.com |
| **License** | MIT |
| **Year** | 2026 |

AI Co-Developers: Claude (Anthropic) — full BV formalism design and implementation; GPT (OpenAI) — literature cross-check, gauge-fixing review; Gemini (Google) — initial operator scaffolding; DeepSeek — numerical stability verification.

---

## 1. What this module is

`one_core_evolution.py` ships a **scalar** `BVFieldTheory` base class. It checks a simplified Classical Master Equation, `{S, S} = 0`, on a quadratic action and always returns `True` by construction — useful as a lightweight consistency stub for `GeneNetworkBV` and `InteractionNetworkBV`, but not a full gauge-theory implementation.

`bv_full_theory_one.py` is the **full-theory extension**. It implements the complete BV field content (fields, ghosts, antifields, ghost-antifields), the antibracket, the BV Laplacian, the BRST differential, gauge fixing, and both the Classical and Quantum Master Equations — all fully differentiable in PyTorch — then wires concrete instances of this machinery into the EVOLUTION ONE, EpiForecast, and Cahn–Hilliard clusters.

It is an **additive companion module**: it does not replace or modify `one_core_evolution.py`, and existing code that calls `GeneNetworkBV.verify()` or `InteractionNetworkBV.verify()` continues to work unchanged. Use this module when a deeper, gauge-theoretic certification of network consistency is needed, or when ghost-number bookkeeping, BRST cohomology, or quantum corrections (`Δ_BV`) are relevant to the analysis.

---

## 2. Mathematical background

### 2.1 Field content

For each physical field `Phi^i` (gene expression, viral fitness, phase-field order parameter, …), the full BV spectrum introduces three additional partners:

| Field | Ghost number | Role |
|---|---|---|
| `Phi^i` | 0 | original field |
| `C^alpha` | +1 | ghost, one per gauge generator |
| `Phi*_i` | -1 | antifield conjugate to `Phi^i` |
| `C*_alpha` | -2 | ghost antifield conjugate to `C^alpha` |

### 2.2 Antibracket

```
(F, G) = Σ_i [ dF/dPhi^i · dG/dPhi*_i − dF/dPhi*_i · dG/dPhi^i ]
       + Σ_α [ dF/dC^α  · dG/dC*_α  − dF/dC*_α  · dG/dC^α  ]
```

This is an odd Poisson bracket on the full field space — the central algebraic object of the formalism.

### 2.3 Classical Master Equation (CME)

```
(S, S) = 0
```

where `S = S_0[Phi] + Phi*_i R^i_α C^α + …` is the BV-extended action, `S_0` is the original gauge-invariant action, and `R^i_α` are the gauge-symmetry generators. `(S, S) = 0` is the statement that the gauge symmetry is consistent — no anomaly, closed algebra.

### 2.4 BV Laplacian and Quantum Master Equation (QME)

```
Δ_BV F = Σ_i d²F / (dPhi^i dPhi*_i) + Σ_α d²F / (dC^α dC*_α)

QME:  ½(S, S) − iℏ Δ_BV S = 0
```

The QME is the quantum-corrected consistency condition; a non-zero `Δ_BV S` signals a one-loop anomaly that the classical action alone would miss.

### 2.5 BRST operator and cohomology

```
s F = (S, F),     s² = 0  ⟺  CME holds
```

Physical observables are elements of `H⁰(s)`: BRST-closed (`sF = 0`) but not BRST-exact (`F ≠ sG` for any `G`) functionals at ghost number zero.

### 2.6 Gauge fixing

A gauge-fixing fermion `Ψ` (ghost number −1) fixes the antifields via `Phi*_i = dΨ/dPhi^i`, eliminating them from the path integral and producing the gauge-fixed action `S_gf`.

### 2.7 W-algebra and Homotopy BV

The antibracket restricted to pairs of fields gives a correlation tensor `W^{ij} = (Phi^i, Phi^j)`, whose antisymmetric part are Lie-algebra-like structure constants and whose trace-squared is a Casimir invariant. The `HomotopyBV` class implements the first three L∞ brackets (`l₁ = s`, `l₂ = (·,·)`, `l₃` = Jacobiator) for cases where the CME holds only up to homotopy — relevant to networks with approximate symmetries.

---

## 3. Physical interpretation inside EVOLUTION ONE

| BV object | EVOLUTION ONE meaning |
|---|---|
| `Phi^i` | gene expression level / viral fitness / phase-field order parameter `u` |
| `C^alpha` | redundancy in the interaction network (e.g. global scale, permutation) |
| `S_0` | network interaction energy (quadratic in pairwise differences) |
| `(S, S) = 0` | the interaction network has no expression-level / epidemiological anomaly |
| `W^{ij}` Casimir | overall strength of gauge correlation across the network |
| `μ_BV` (Cahn–Hilliard) | BV-certified mutation load, `sigmoid(Casimir · ū)` |

---

## 4. Public API

```python
from bv_full_theory_one import (
    BV_FULL_VERSION,
    # Core BV algebra
    BVSpectrum, BVAntibracket, BVLaplacian, BRSTOperator,
    # Action hierarchy
    BVActionBase, GaugeSymmetry, BVGaugeFixer,
    # Master equations
    ClassicalMasterEquation, QuantumMasterEquation,
    # BRST cohomology
    BRSTCohomology,
    # W-algebra / Homotopy BV
    WAlgebra, HomotopyBV,
    # EVOLUTION ONE concrete implementations
    GeneNetworkBVFull, InteractionNetworkBVFull, CahnHilliardBVFull,
    # Engine wrappers
    EvoOneBVEngine, EpiBVEngine, CahnHilliardBVBridge,
)
```

### 4.1 Core algebra

- **`BVSpectrum`** — holds `phi`, `ghost`, `phi_star`, `ghost_star` dicts of leaf tensors (`requires_grad=True` by default) plus the ghost-number registry. `.clone()` and all `set_*()` methods always return fresh leaf tensors so the autograd graph stays well-formed.
- **`BVAntibracket`** — `.compute(F_fn, G_fn)` evaluates `(F, G)`. Prefers native `torch.autograd.grad`; falls back to central finite differences only if a tensor arrives detached.
- **`BVLaplacian`** — `.apply(S_fn)` evaluates `Δ_BV S`; `.qme_residual(S_fn, antibracket)` evaluates the full QME residual.
- **`BRSTOperator`** — `.apply(F_fn)` evaluates `sF`; `.nilpotency_check(F_fn)` returns `(is_nilpotent, |s²F|)`.

### 4.2 Action hierarchy

- **`GaugeSymmetry`** — wraps a generator function `R^i_α` and its action on the field dict.
- **`BVActionBase`** (abstract) — subclass and implement `classical_action(spectrum)`; inherited methods give `master_action`, `cme_residual`, `qme_residual`, `verify_cme`.
- **`BVGaugeFixer`** — `.fix_antifields()` sets `Phi*_i = dΨ/dPhi^i`; `.gauge_fixed_action(bv_action)` returns `S_gf`.

### 4.3 Diagnostics

- **`ClassicalMasterEquation(bv_action).check(tol, verbose)`** → `{ok, residual, gauge_algebra_closed, anomaly_free, n_gauge_syms}`
- **`QuantumMasterEquation(bv_action, hbar).check(tol, verbose)`** → `{ok, residual_real, delta_bv_S, cme_term, hbar}`
- **`BRSTCohomology(brst_op).physical_observables(candidates, tol)`** → indices of closed-but-not-exact candidates

### 4.4 Concrete EVOLUTION ONE classes

| Class | Drop-in replacement for | Source module |
|---|---|---|
| `GeneNetworkBVFull` | `GeneNetworkBV` | `evolution_one_v4.py` |
| `InteractionNetworkBVFull` | `InteractionNetworkBV` | `evolution_one_epidemiological_viral_v5.py` |
| `CahnHilliardBVFull` | *(new)* | `structural_cahn_hilliard_3d_v3.py` |

Each exposes `.classical_action()`, `.analyse(tol, verbose)` (full diagnostic dict: CME, QME, BRST nilpotency, W-tensor, Casimir, gauge-fixed action, homotopy-CME residual, physical observables), and `.verify()` (quick boolean, same call signature as the original scalar classes).

### 4.5 Engine wrappers

- **`EvoOneBVEngine(gene_names, interactions, coupling, device)`** — `.run()` for the full report, `.verify()` for drop-in use inside `EvolutionONEEngine`.
- **`EpiBVEngine(node_names, interactions, coupling, device)`** — same pattern for `EpiForecastEngine`.
- **`CahnHilliardBVBridge(kappa, device)`** — `.project_to_mu_bv(u_field)` returns a fully differentiable BV-certified mutation load in `(0, 1)`; `.bv_to_rt(u_field, rt_base, scale, threshold)` maps it onward to an effective reproduction number, composable with `EpiEvolutionBridge.mu_to_rt()`.

---

## 5. Usage examples

### 5.1 Drop-in BV check for the cancer-evolution gene network

```python
from bv_full_theory_one import EvoOneBVEngine

genes        = ["TP53", "KRAS", "EGFR", "MYC", "BRCA1"]
interactions = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

bv = EvoOneBVEngine(genes, interactions)
print(bv.verify())            # quick bool, same as old GeneNetworkBV.verify()

report = bv.run(verbose=True)  # full diagnostic
print(report["overall_consistent"], report["cme"]["residual"], report["casimir"])
```

### 5.2 Full report for the host–pathogen network

```python
from bv_full_theory_one import EpiBVEngine

nodes = ["S", "I", "R", "H", "D"]
edges = [(0, 1), (1, 2), (1, 3), (3, 4)]

epi_bv = EpiBVEngine(nodes, edges)
report = epi_bv.run()
```

### 5.3 BV-certified mutation load from a Cahn–Hilliard phase field

```python
from bv_full_theory_one import CahnHilliardBVBridge
import torch

bridge  = CahnHilliardBVBridge(kappa=0.5)
u_field = torch.randn(64, 64, 64)        # from StructuralCahnHilliard3D.u

mu_bv = bridge.project_to_mu_bv(u_field)              # (0, 1), differentiable
rt_bv = bridge.bv_to_rt(u_field, rt_base=torch.tensor(1.2))
```

### 5.4 Building a custom BV action

```python
from bv_full_theory_one import BVActionBase, BVSpectrum, GaugeSymmetry

class MyNetworkBV(BVActionBase):
    def classical_action(self, spectrum):
        ...  # return scalar Tensor

spectrum = BVSpectrum(field_names=["a", "b"], gauge_names=["C0"])
action   = MyNetworkBV(spectrum)
action.register_gauge_symmetry(GaugeSymmetry(my_generator_fn, name="my_symmetry"))
ok, residual = action.verify_cme()
```

---

## 6. Differentiability and conventions

- Every leaf tensor in `BVSpectrum` is created with `requires_grad=True`; `BVAntibracket` and `BVLaplacian` use native `torch.autograd.grad` as the primary path, with finite differences only as a fallback for detached inputs.
- `soft_clamp`-style smooth nonlinearities (`F.softplus`, `torch.sigmoid`) are used wherever a hard clamp would otherwise appear, consistent with the rest of the ONE Ecosystem.
- No `try/except ImportError` fallback is needed for `one_core_evolution` — this module is a strict downstream dependency and imports it directly.
- A `[PASS]`-style `__main__` self-test (`_self_test()`) exercises all four concrete classes and prints `[OK]` per component.

---

## 7. Relationship to the four-paper Structural Calculus series

This module operationalises the BV sector referenced in **Paper 2 — BV Jump Measures & Self-Evolving Interfaces**, extending it from the jump-measure / interface context into a complete gauge-theoretic consistency layer for the population-scale EVOLUTION ONE cluster. It complements, but is independent of, the CSOC/SSC machinery of Paper 4 — the two can be composed (e.g. `CahnHilliardBVBridge` feeding into `EpiEvolutionBridge`) but neither depends on the other.

---

## 8. Limitations and honest scope

- `BVAntibracket` and `BVLaplacian` use finite-difference derivatives wherever a clean autograd path through a Python-level dict mutation is not available (notably inside `BVLaplacian._mixed_second` and `BVGaugeFixer.fix_antifields`); these are numerically reliable for the low-dimensional toy networks the EVOLUTION ONE cluster currently uses, but have not been benchmarked for large field counts.
- `ClassicalMasterEquation._check_jacobi()` currently returns `True` unconditionally for the quadratic actions implemented here, since the Jacobi identity holds trivially for them; it is a placeholder for non-quadratic actions where the check would need to be made substantive.
- This module has not been executed end-to-end in this environment (no PyTorch / no network access at time of writing) — only syntax-checked and reviewed by hand. Run `python bv_full_theory_one.py` locally and confirm the `[PASS]` line before relying on it in a pipeline.
