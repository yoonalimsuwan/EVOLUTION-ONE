# EVOLUTION ONE Cluster — README_PLUS

**ONE Ecosystem · MSPS NETWORK**
Developer: Yoon A Limsuwan · ORCID: 0009-0008-2374-0798 · GitHub: yoonalimsuwan
License: MIT · Year: 2026

---

## Overview

The **EVOLUTION ONE Cluster** is a suite of six production-grade, fully differentiable PyTorch modules that collectively form a multi-scale simulation framework spanning cancer evolution, epidemiological forecasting, viral mutation analysis, molecular dynamics, phase-field physics, and graph neural operator surrogates. Every gradient path throughout the cluster is differentiable end-to-end — no hard clamps, no non-differentiable operations — enabling joint training, sensitivity analysis, and seamless integration into larger AGI ONE workflows.

The cluster operates at **population / genomic scale** and is intentionally separate from:
- `one_core.py` — DNS/CFD continuum scale (SUPER DNS ONE)
- `one_core_fold.py` — protein refinement scale (REAL FOLD ONE)

### AI Co-Developers

All six files credit the following AI systems as co-developers:

| AI | Organization | Primary Contributions |
|---|---|---|
| **Claude** | Anthropic | Full differentiability audits, cross-cluster bridge design, BVFieldTheory standalone, SOCController removal, `__all__` public API, import-chain integration |
| **GPT** | OpenAI | Literature cross-checks, coupling equation review, numerical stability |
| **Gemini** | Google | Initial architecture scaffolding, operator scaffolding |
| **DeepSeek** | DeepSeek | Cross-verification of differentiability chains, stencil verification |

---

## Cluster Architecture

```
EVOLUTION ONE Cluster  (v3.0.0)
│
├── one_core_evolution_v3.py          ← Single source of truth (shared foundation)
│     ├── SemanticStateContraction
│     ├── CSOCBase / DifferentiableSOC / DifferentiableRG / DifferentiableIto
│     ├── BVFieldTheory
│     ├── CheckpointManager
│     ├── LangevinEvolutionBridge / LangevinBridgeMixin
│     ├── EpiEvolutionBridge
│     └── CahnHilliardEvoBridge
│
├── evolution_one_v4.py               ← Cancer evolution engine
│     └── imports from one_core_evolution_v3
│
├── evolution_one_epidemiological_viral_v5.py   ← Epidemiological & viral engine
│     └── imports from one_core_evolution_v3
│
├── structural_langevin_evo_v5.py     ← BAOAB Langevin MD integrator
│     └── imports from one_core_evolution_v3
│
├── structural_cahn_hilliard_3d_v3.py ← Phase-field / CH3D PDE engine
│     └── imports from one_core_evolution_v3 (preferred) or one_core (fallback)
│
└── structural_gno_evolution.py       ← GNO surrogate (all three modes)
      └── optional import from one_core_evolution_v3
```

Cross-cluster data flow:
```
StructuralCahnHilliard3D ──CahnHilliardEvoBridge──► EvolutionONEEngine
AdvancedStructuralLangevin ─LangevinEvolutionBridge─► EvolutionONEEngine
                                                      EvolutionONEEngine ─EpiEvolutionBridge─► EpiForecastEngine
StructuralGNOEvolution ─────────────────────────────► (surrogate for all three engines)
```

---

## File Reference

---

### 1. `one_core_evolution_v3.py`
**ONE CORE EVOLUTION — Shared Foundation**
Version: 3.0.0 · ~932 lines

This is the **single source of truth** for all shared components in the EVOLUTION ONE cluster. All other five files import directly from this module. It is designed to be a clean, minimal, fully differentiable foundation with no external domain-specific dependencies.

#### Public API (`__all__`)

| Symbol | Category | Description |
|---|---|---|
| `EVOLUTION_VERSION` | Version | Cluster-wide version string `"3.0.0"` |
| `get_device()` | Hardware | Unified backend selector (CUDA / MPS / CPU) |
| `SemanticStateContraction` | Abstract base | SSC EMA filter for state smoothing (Paper 4) |
| `CSOCBase` | Abstract base | CSOC abstract base class (Paper 4) |
| `InterfaceDetectorBase` | Abstract base | Interface detector base |
| `StructuralItoBase` | Abstract base | Structural Itô correction base (Papers 2 & 3) |
| `BVFieldTheory` | Field theory | Standalone BV field theory — no `real_fold_one` required |
| `DifferentiableRG` | Differentiable module | Fully differentiable learnable Renormalisation Group smoother (F.softmax normalisation) |
| `DifferentiableSOC` | Differentiable module | Fully differentiable SOC dynamics (logsumexp spread + softplus floor) |
| `DifferentiableIto` | Differentiable module | Fully differentiable Itô integrator (softplus floor + sigmoid ceiling) |
| `CheckpointManager` | Utility | Unified save/load checkpoint manager |
| `LangevinEvolutionBridge` | Bridge | Connects `AdvancedStructuralLangevin` → `EvolutionONEEngine` / `EpiForecastEngine` |
| `EpiEvolutionBridge` | Bridge | Bidirectional μ ↔ Rt coupling between cancer and epidemiological engines |
| `CahnHilliardEvoBridge` | Bridge | Phase-field order parameter u → mutation load μ / Rt coupling |
| `LangevinBridgeMixin` | Mixin | Attaches `attach_langevin_bridge()` / `attach_cahn_bridge()` to engine classes |

#### Key Changes v2 → v3

- **Fix 1** — `BVFieldTheory` added as standalone; `GeneNetworkBV` no longer requires `real_fold_one`
- **Fix 2** — `CahnHilliardEvoBridge` added: CH3D phase-field ↔ EVOLUTION ONE coupling
- **Fix 3** — `LangevinBridgeMixin` added for engine attachment pattern
- **Fix 4** — `SOCController` removed; `DifferentiableSOC` is the sole SOC implementation
- **Bug 1–3, Bug 7** — All remaining non-differentiable clamp/std/weight operations replaced with smooth alternatives; `EpiEvolutionBridge` added

#### Bridge Classes in Detail

**`LangevinEvolutionBridge`**
Maps micro-scale BAOAB Langevin coordinates → macro-scale mutation load μ via SSC coarse-graining:
```
μ = sigmoid(mean_displacement / σ_scale)
```

**`CahnHilliardEvoBridge`**
Maps phase-field order parameter u ∈ [−1, 1] → mutation load μ ∈ (0, 1):
```
φ_mut = (⟨u⟩ + 1) / 2        (volume fraction of mutant cells)
μ     = SSC(φ_mut)             (temporally smoothed)
```
Interface sharpness |∇u| additionally modulates Rt for the epidemiological engine.

**`EpiEvolutionBridge`**
Bidirectional coupling:
```
Rt  = sigmoid(rt_base + mu_to_rt_scale × μ)
μ   = sigmoid(mu_base + rt_to_mu_scale × Rt)
```

---

### 2. `evolution_one_v4.py`
**EVOLUTION ONE v4 — Multi-Level Cancer Evolution & Structural Impact Engine**
~1,323 lines

The primary cancer evolution engine. Models somatic mutation dynamics, structural protein impact, CRISPR design, epigenetic editing, and chemical intervention — all within a unified, differentiable PyTorch framework.

#### Class Inventory

| Class | Description |
|---|---|
| `GeneNetworkBV` | Extends `BVFieldTheory`; BV consistency check for gene interaction networks via Classical Master Equation |
| `MutationDataLoader` | Loads MAF and VCF mutation files; filters by gene panel |
| `DuonAnalyzer` | Duon-aware mutation analysis — identifies coding/regulatory overlaps via BED intervals (requires PyRanges) |
| `EvolutionaryClassifier` | Differentiable SOC-based classifier; predicts trajectory to cancer (Stable / Critical / Collapse) using `DifferentiableSOC` |
| `FutureMutationPredictor` | Predicts escape mutations and structural impact (ΔΔG) via REAL FOLD ONE HT (optional) |
| `CRISPRDesigner` | gRNA scoring (GC content, hairpin, polyT), Bowtie/brute-force off-target search, repair template with codon table + homology arms |
| `EpigeneticDesigner` | Per-interval methylation-aware epigenetic editing target identification |
| `InterventionRecommender` | Chemical intervention recommendation (targeted therapy + stabilisers) |
| `RetrospectiveAnalyzer` | Lifestyle factor correlation with p-values (Pearson/Spearman) |
| `EvolutionONEEngine` | Master orchestration engine; inherits `LangevinBridgeMixin`; coordinates all sub-modules |

#### `EvolutionONEEngine` Integration Points

```python
engine = EvolutionONEEngine(cfg)

# Attach Langevin bridge (Fix 3)
bridge = engine.attach_langevin_bridge(langevin_integrator, sigma_scale=1.5)

# Attach Cahn-Hilliard bridge (Fix 5)
ch_bridge = engine.attach_cahn_bridge(ch_solver)

# Attach epidemiological bridge (Bug 7)
epi_bridge = engine.attach_epi_bridge(epi_engine, mu_to_rt_scale=0.5, rt_to_mu_scale=0.3)
```

#### CLI Output Files

| File | Description |
|---|---|
| `summary.json` | Run summary and configuration |
| `sample_states.csv` | Per-sample SOC state classifications |
| `crispr_designs.json` | gRNA designs with off-target scores |
| `epigenetic_targets.json` | Methylation-aware editing targets |

#### Key Changes v3 → v4

- Fix 1: `GeneNetworkBV` uses `BVFieldTheory` from `one_core_evolution` (no `real_fold_one` needed)
- Fix 3: `EvolutionONEEngine` inherits `LangevinBridgeMixin`
- Fix 4: `EvolutionaryClassifier` uses `DifferentiableSOC` exclusively (replaces `SOCController`)
- Fix 5: `EvolutionONEEngine.attach_cahn_bridge()` added
- Fix 6: `EVOLUTION_VERSION` imported from `one_core_evolution_v3`

#### Optional Dependencies

| Package | Purpose |
|---|---|
| `biopython` | Sequence I/O, BLAST, codon tables |
| `pyranges` | Genomic interval overlaps (duon analysis) |
| `bowtie` | Off-target CRISPR search (external binary) |
| `real_fold_one` | Structural refinement / ΔΔG prediction |
| `real_fold_one_ht` | High-throughput escape mutation scanning |
| `optuna` | Hyperparameter tuning for SOC thresholds |

---

### 3. `evolution_one_epidemiological_viral_v5.py`
**EVOLUTION ONE v5 — High-Performance Epidemiological Forecasting & Viral Evolution Engine**
~1,305 lines

The epidemiological and viral evolution companion to `evolution_one_v4.py`. Models epidemic trajectories, viral genome evolution, epitope scoring, vaccine polyepitope design, and therapeutic recommendation — fully coupled to the cancer evolution engine via `EpiEvolutionBridge`.

#### Class Inventory

| Class | Description |
|---|---|
| `DifferentiableProteinEnergy` | Differentiable protein energy model with torsion angles, Lennard-Jones, electrostatics, hydrogen bonding, solvent SASA |
| `RefinementEngine` | Lightweight structure refinement wrapper (local fallback if `real_fold_one` absent) |
| `EpidemicClassifier` | `DifferentiableSOC`-based epidemic state classifier (Stable / Critical / Collapse) |
| `EpidemiologicalDataLoader` | Loads OWID COVID, WHO FluNet, and custom surveillance data |
| `ViralEvolutionAnalyzer` | dN/dS ratio, synonymous/non-synonymous mutation rates, Shannon entropy of variant diversity |
| `FutureVariantPredictor` | Predicts future escape variants using MSA + energy-ranked mutation scoring |
| `EpitopeScorer` | Differentiable epitope–antibody binding affinity scorer (MAB targets library included) |
| `VaccineDesigner` | Polyepitope vaccine design with 3-way linker logic (GPGPG / AAY / KK) |
| `TherapeuticRecommender` | Maps variant profiles to therapeutic intervention recommendations |
| `EpidemiologicalFactorAnalyzer` | Retrospective factor correlation (mobility, vaccination rate, etc.) |
| `InteractionNetworkBV` | Extends `BVFieldTheory`; BV consistency check for pathogen–host interaction networks |
| `EpiForecastEngine` | Master epidemiological engine; inherits `LangevinBridgeMixin`; orchestrates all sub-modules |

#### `EpiForecastEngine` Integration Points

```python
engine = EpiForecastEngine(cfg)

# Attach Langevin bridge (Fix 3)
bridge = engine.attach_langevin_bridge(langevin_integrator, sigma_scale=1.5)

# Attach Cahn-Hilliard bridge (Fix 5)
ch_bridge = engine.attach_cahn_bridge(ch_solver)
```

#### Key Changes v4 → v5

- Fix 1: `InteractionNetworkBV` uses `BVFieldTheory` from `one_core_evolution`
- Fix 3: `EpiForecastEngine` inherits `LangevinBridgeMixin`
- Fix 4: `EpidemicClassifier` uses `DifferentiableSOC`; `LearnableRG` aliased to `DifferentiableRG` for backward compatibility
- Fix 5: `EpiForecastEngine.attach_cahn_bridge()` added
- Fix 6: `EVOLUTION_VERSION` imported from `one_core_evolution_v3`
- R1–R8: Full feature parity restored from v4 (dataclasses, `_TAU_GATE`, `hydro_default`, Biopython structural imports, full MAB target library, 3-way linker logic, `run()` logging, section headers)

#### Notable Constants

```python
_TAU_GATE        # Gate time constant for differentiable temporal gating
MAB_TARGETS      # Full antibody library including Cilgavimab/Tixagevimab, Etesevimab
```

---

### 4. `structural_langevin_evo_v5.py`
**LANGEVIN ADVANCED WITH STRUCTURAL CALCULUS v5 — BAOAB Langevin MD Integrator**
~696 lines

A fully differentiable, higher-order BAOAB splitting Langevin integrator implementing the 4-Paper Structural Calculus ecosystem. Provides the atomistic / molecular dynamics backbone for the EVOLUTION ONE cluster.

#### Theoretical Basis

The module integrates four complementary theoretical frameworks:

1. **Paper 1** — Regime-Dependent Analytical Framework (Structural Operators)
2. **Paper 2** — BV Jump Measures & Self-Evolving Interfaces
3. **Paper 3** — Structural Itô Calculus & Multiplicative Noise Correction
4. **Paper 4** — Controlled Self-Organized Criticality (CSOC) & SSC Thermostat

#### Structural Derivative

```
D^S u = ∇u + [u] δ_Γ
```

where [u] is the jump measure across interface Γ. This is the central operator linking atomic-scale forces to interface-mediated dynamics.

#### Class Inventory

| Class | Description |
|---|---|
| `InterfaceDetector` | Per-atom soft interface mask ∈ [0, 1]; differentiable w.r.t. coordinates; identifies interface atoms by pairwise distance variance |
| `CSOCThermostat` | CSOC-driven adaptive temperature T and friction γ; extends `CSOCBase` |
| `StructuralItoNoise` | Multiplicative/structural noise generator; Itô drift correction; extends `StructuralItoBase` |
| `AdvancedStructuralLangevin` | Full BAOAB integrator combining all four modules; public `full_step()` and `baoa_step()` / `final_b_step()` APIs |

#### BAOAB Splitting Scheme

```
B step:  v ← v + (dt/2) * F / m                  (half-kick)
A step:  x ← x + (dt/2) * v                       (half-drift)
O step:  v ← e^{-γdt} v + σ_noise * √(kT/m) * ξ  (Ornstein-Uhlenbeck)
A step:  x ← x + (dt/2) * v                       (half-drift)
B step:  v ← v + (dt/2) * F_new / m               (half-kick)
```

Structural noise focuses stochasticity at interfaces via `InterfaceDetector` mask.

#### Usage Pattern

```python
integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=300.0)

# Attach to engine via bridge mixin
bridge = engine.attach_langevin_bridge(integrator, sigma_scale=1.5)

# Or use directly with outer loop
for step in range(num_steps):
    force_bulk = -torch.autograd.grad(energy, coords, retain_graph=True)[0]
    x_new, v_tilde, T, sigma = integrator.baoa_step(
        coords, velocities, force_bulk, jumps, interface_mask
    )
    new_energy = potential(x_new)
    new_force  = -torch.autograd.grad(new_energy, x_new)[0]
    velocities = integrator.final_b_step(v_tilde, new_force, jumps, interface_mask)
    coords     = x_new.detach().requires_grad_(True)
```

#### Key Changes v4 → v5

- Merge 1–6: Full docstrings, Args/Returns blocks, section banners, and all inline comments restored from v3 while retaining all v4 cross-cluster additions

#### Key Changes v3 → v4

- Fix 1: Imports from `one_core_evolution` (`CahnHilliardEvoBridge`, `LangevinBridgeMixin`, `BVFieldTheory`)
- Fix 2: `__all__` public API defined
- Bug 4a–4d: All four `clamp()` calls replaced with softplus/sigmoid smooth alternatives

---

### 5. `structural_cahn_hilliard_3d_v3.py`
**STRUCTURAL CAHN-HILLIARD 3D v3 — Fourth-Order Structural PDE Suite**
Component #4 of the SUPER DNS ONE Cluster · ~1,542 lines

A full production implementation of the Structural Cahn-Hilliard equation on 3D periodic domains, with three GPU-parallel Laplacian backends, thin-film extension, and phase-field crystal (6th-order PDE) extension. Also serves as a cross-cluster contributor to EVOLUTION ONE via `CahnHilliardEvoBridge`.

#### Mathematics

**Structural Operators (sigma-field formulation):**
```
grad_S u   = σ(x) · ∇u                           (Structural Gradient)
div_S F    = div(σ(x) · F)                        (Structural Divergence)
Δ_S u      = div(σ(x) · ∇u)                      (Structural Laplacian)
Δ_S² u     = Δ_S(Δ_S u)                          (Structural Bi-Laplacian)
```

**Standard Cahn-Hilliard:**
```
μ_R   = (u³ − u) − ε² · Δ_S u                   (Chemical Potential)
du/dt = Δ_S(μ_R)                                  (Phase Evolution)
```

**Thin-Film Cahn-Hilliard (degenerate mobility):**
```
M(u)  = softplus(u)³
du/dt = div_S(M(u) · ∇μ_R)
      + optional: −κ_s · Δ_S(M(u) · Δ_S u)
```

**Phase-Field Crystal (6th-order PDE):**
```
F_PFC(u) = r/2·u² + 1/4·u⁴ + 1/2·u·(1 + Δ_S)²·u
μ_PFC    = (r·u + u³) + u + 2·Δ_S u + Δ_S² u
du/dt    = Δ_S(μ_PFC)
```

#### Class Inventory

| Class | Description |
|---|---|
| `CahnHilliardConfig` | Full configuration dataclass (Laplacian backend, grid size, ε, time step, IMEX scheme) |
| `_Conv3dLaplacian` | GPU-parallel Laplacian via batched Conv3d staggered stencil; ~4–8× faster than roll-loop on CUDA |
| `_FFTLaplacian` | Spectral Laplacian via `torch.fft.rfftn`; O(N log N); exact for periodic domains |
| `_RollLaplacian` | Reference roll-based Laplacian; used when neither Conv3d nor FFT is selected |
| `StructuralCahnHilliard3D` | Base class; standard CH equation with IMEX time-stepping and energy monitoring |
| `ThinFilmStructuralCahnHilliard3D` | Subclass; degenerate mobility M(u) = softplus(u)³; optional surface diffusion; Hamaker wetting energy |
| `PhaseFieldCrystal3D` | Subclass; 6th-order PFC with three recursive Laplacian calls; optional SSC stabilisation; pfc_energy() Lyapunov functional |

#### Laplacian Backend Selection

```python
cfg = CahnHilliardConfig(laplacian='conv3d')  # GPU-parallel stencil (default)
cfg = CahnHilliardConfig(laplacian='fft')     # Spectral (exact, periodic)
cfg = CahnHilliardConfig(laplacian='roll')    # Reference roll-based
```

#### Module-Level Utility

```python
structural_biharmonic_n(field, sigma, n, laplacian_fn)
```
Computes Δ_Sⁿ u recursively; exposed for use by `one_core.py` and other cluster modules.

#### Ecosystem Integration Priority

```python
# Import chain (priority order):
#   1. one_core_evolution  — EVOLUTION ONE cluster (preferred in v3)
#   2. one_core            — SUPER DNS ONE cluster
#   3. standalone          — graceful fallback (no dependencies)
```

#### Key Changes v2 → v3 (Cross-cluster Integration)

- Fix 5: `one_core_evolution` import chain added; SSC/CSOCBase now prefer `one_core_evolution` over local fallback
- `CahnHilliardEvoBridge` re-exported from `one_core_evolution` and added to `__all__`

#### Key Changes v1 → v2

- GPU-1: `_Conv3dLaplacian` — vectorised batched Conv3d (no Python axis loops)
- GPU-2: `_FFTLaplacian` — spectral Laplacian via `torch.fft.rfftn`
- TF-1: `ThinFilmStructuralCahnHilliard3D` subclass
- PFC-1: `PhaseFieldCrystal3D` subclass (6th-order PDE)
- CORE-1: `structural_biharmonic_n()` module-level utility

---

### 6. `structural_gno_evolution.py`
**STRUCTURAL GNO EVOLUTION — Graph Neural Operator Surrogate**
Production Release · ~1,403 lines · `SGNO_VERSION`

A unified Graph Neural Operator (GNO) that acts as a differentiable surrogate for all three simulation engines in the EVOLUTION ONE cluster. Replaces expensive PDE/ODE solves with learned forward passes during training and inference.

#### Three Operating Modes

| Mode | Inputs | Outputs | Replaces |
|---|---|---|---|
| **Mode 1 — Evolution/Epidemiological** | Node features (amino acid / patient embeddings), graph connectivity, σ field | Δμ (mutation-load increment), ΔRt (reproduction-number increment), 3-class CSOC logits | `EvolutionONEEngine` + `EpiForecastEngine` |
| **Mode 2 — Structural Langevin** | Sequence features, initial atomic coordinates, edge_index, σ | Coordinate displacements after one MD window | `AdvancedStructuralLangevin` |
| **Mode 3 — Cahn-Hilliard 3D** | Phase field u, voxel features [u, x, y, z], edge_index, σ | Predicted Δu after Δt steps of CH3D | `StructuralCahnHilliard3D` |

#### Class Inventory

| Class / Function | Description |
|---|---|
| `SGNOEvoConfig` | Full validated configuration dataclass (architecture + loss weights + training hyperparameters) |
| `BatchData` | Typed, validated batch container for all three modes |
| `RBFPositionalEncoder` | Radial-basis-function positional encoder for 3D coordinates |
| `SigmaEncoder` | Projects per-node σ values before FiLM conditioning |
| `FiLMMessagePassing` | Symmetric edge features, residual gate, pre-norm pattern, numerical stability |
| `StructuralGNOEvolution` | Shared backbone + three output heads; `forward()` dispatcher |
| `loss_energy_conservation()` | Physics-informed energy conservation proxy loss (Mode 2) |
| `loss_total_variation_3d()` | Total variation / interface sharpness loss (Mode 3) |
| `loss_rt_kl_smooth()` | KL-divergence smoothing for Rt distribution (Mode 1) |
| `EMAWeights` | Exponential Moving Average weight tracker |
| `CheckpointManager` | Full training state save/load (model, optimizer, scheduler, EMA, config) |
| `SGNOEvolutionTrainer` | Production trainer with AMP, gradient clipping, cosine LR schedule, early stopping |

#### Architecture Details

**Shared Backbone:**
- Pre-LayerNorm pattern throughout
- `num_layers` stacked `FiLMMessagePassing` layers
- Positional encoding via `RBFPositionalEncoder` (num_rbf bins)
- σ field projected per-node via `SigmaEncoder` before FiLM conditioning
- Per-head output normalisation

**FiLMMessagePassing:**
- Symmetric edge features (concatenation of source + target + relative position)
- FiLM conditioning: γ, β from σ encoder modulate hidden features
- Residual gate (learned scalar per node)
- Dropout inside MLPs

**Physics-Informed Losses:**
```
Total loss = λ_evo·L_evo + λ_cls·L_cls     # Mode 1
           + λ_md·L_md + λ_md_phys·L_phys  # Mode 2
           + λ_ch·L_ch + λ_ch_tv·L_tv      # Mode 3
```

#### Training Features

| Feature | Description |
|---|---|
| AMP | `torch.cuda.amp.GradScaler` on CUDA |
| Gradient clipping | L2-norm clipping at `grad_clip` (default 1.0) |
| NaN guard | Skips update steps with NaN losses |
| LR schedule | Linear warm-up (`lr_warmup_steps`) → cosine annealing to `lr_min` |
| EMA | `ema_decay` (default 0.999) exponential weight averaging |
| Early stopping | `patience` epochs without improvement |
| Per-head logging | Loss tracked separately for each mode |

#### Minimal Training Example

```python
from structural_gno_evolution import SGNOEvoConfig, StructuralGNOEvolution, SGNOEvolutionTrainer, get_device

cfg     = SGNOEvoConfig(hidden_dim=128, num_layers=6, max_epochs=200)
model   = StructuralGNOEvolution(cfg)
device  = get_device()
trainer = SGNOEvolutionTrainer(model, cfg, device)

for epoch in range(cfg.max_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss   = trainer.evaluate(val_loader)
    if trainer.check_early_stop(val_loss):
        break

trainer.checkpoint_manager.save("best_sgno.pt", trainer)
```

---

## Installation

### Core Dependencies

```bash
pip install torch>=2.1 numpy pandas scipy matplotlib
```

### Optional Dependencies

```bash
# Biological sequence analysis
pip install biopython pyranges

# Hyperparameter optimisation
pip install optuna

# External tools (for CRISPR off-target search)
# bowtie — install via conda or system package manager
```

### Module Setup

Place all six files in the same directory (or on `PYTHONPATH`). `one_core_evolution_v3.py` must be importable as `one_core_evolution`:

```bash
# Rename for import compatibility
cp one_core_evolution_v3.py one_core_evolution.py
```

Or add a thin alias:
```python
# one_core_evolution.py
from one_core_evolution_v3 import *
```

---

## Hardware Support

All modules support vendor-neutral hardware selection via `get_device()`:

| Backend | Condition |
|---|---|
| CUDA | `torch.cuda.is_available()` |
| Apple MPS | `torch.backends.mps.is_available()` |
| Huawei Ascend | Detected via NPU availability check |
| CPU | Fallback |

Mixed-precision training (AMP) is automatically enabled on CUDA where supported.

---

## Differentiability Guarantee

Every gradient path in the cluster is fully differentiable. Key design decisions:

| Non-differentiable pattern | Replacement |
|---|---|
| `tensor.clamp(min=0)` | `F.softplus(tensor)` |
| `tensor.clamp(lo, hi)` | `softplus floor + soft_clamp ceiling` |
| `tensor.std()` | `logsumexp`-based spread |
| `weight / weight.sum()` | `F.softmax(weight)` |
| `SOCController` (hard thresholds) | `DifferentiableSOC` (smooth logistic) |

This ensures end-to-end gradient flow for joint optimisation, meta-learning, and sensitivity analysis.

---

## Cross-Cluster Integration Examples

### Cancer ↔ Epidemiology Coupling

```python
from evolution_one_v4 import EvolutionONEEngine
from evolution_one_epidemiological_viral_v5 import EpiForecastEngine

cancer_engine = EvolutionONEEngine(cfg)
epi_engine    = EpiForecastEngine(cfg)

# Bidirectional coupling: μ ↔ Rt
bridge = cancer_engine.attach_epi_bridge(
    epi_engine, mu_to_rt_scale=0.5, rt_to_mu_scale=0.3
)
```

### Langevin MD → Cancer Mutation Load

```python
from structural_langevin_evo_v5 import AdvancedStructuralLangevin

integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=300.0)
bridge = cancer_engine.attach_langevin_bridge(integrator, sigma_scale=1.5)

# micro_step returns (new_coords, new_velocities)
# bridge automatically maps displacement → μ via SSC coarse-graining
new_coords, new_vel = bridge.micro_step(coords, velocities, force_fn)
mu = bridge.coarse_grain(new_coords, coords)
```

### Phase-Field → Mutation Load

```python
from structural_cahn_hilliard_3d_v3 import StructuralCahnHilliard3D, CahnHilliardConfig
from one_core_evolution import CahnHilliardEvoBridge

cfg    = CahnHilliardConfig(Nx=64, Ny=64, Nz=64, laplacian='conv3d')
solver = StructuralCahnHilliard3D(cfg)
bridge = CahnHilliardEvoBridge(solver)

u_field = solver.step(u_field, sigma)  # evolve phase field
mu      = bridge.project_to_mu(u_field)  # → mutation load μ ∈ (0, 1)
```

### GNO Surrogate Inference

```python
from structural_gno_evolution import StructuralGNOEvolution, SGNOEvoConfig, BatchData

cfg   = SGNOEvoConfig()
model = StructuralGNOEvolution(cfg)
model.load_state_dict(torch.load("best_sgno.pt")["model"])

batch = BatchData(
    node_features=x,     # (N, node_in_dim)
    edge_index=ei,       # (2, E)
    sigma=sigma,         # (N,)
    mode="evo",          # or "md", "ch"
)
delta_mu, delta_rt, csoc_logits = model(batch)
```

---

## Theoretical References

All modules are grounded in the **Structural Calculus** paper series (Limsuwan, 2026):

1. **Paper 1** — Regime-Dependent Analytical Framework with Structural Operators
2. **Paper 2** — BV Jump Measures and Self-Evolving Structural Interfaces (SESI)
3. **Paper 3** — Structural Itô Calculus and Multiplicative Noise Correction
4. **Paper 4** — Controlled Self-Organized Criticality (CSOC) and Semantic State Contraction (SSC)

Primary dissemination: arXiv / Zenodo (open access).

---

## License

MIT License — see individual file headers for full text.

All six modules are open-source and may be used freely for research, education, and commercial purposes.

---

## Citation

If you use the EVOLUTION ONE Cluster in your research, please cite:

```
Limsuwan, Y. A. (2026). EVOLUTION ONE Cluster: A Fully Differentiable
Multi-Scale Simulation Framework for Cancer Evolution, Epidemiology, and
Structural Biology. MSPS NETWORK. GitHub: yoonalimsuwan.
ORCID: 0009-0008-2374-0798.
```

---

*README_PLUS.md — EVOLUTION ONE Cluster · ONE Ecosystem · MSPS NETWORK*
*Generated with AI assistance from Claude (Anthropic), GPT (OpenAI), Gemini (Google), DeepSeek*
