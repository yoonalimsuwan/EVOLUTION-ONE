# EVOLUTION ONE Cluster — README PLUS

**Developer:** Yoon A Limsuwan / MSPS NETWORK
**ORCID:** 0009-0008-2374-0788
**GitHub:** yoonalimsuwan
**License:** MIT
**Version:** 3.0 / Core v2.0.0
**Year:** 2026

---

## Overview

The **EVOLUTION ONE Cluster** is a suite of four fully differentiable PyTorch-based simulation frameworks that model biological evolution at two complementary population scales: **somatic (cancer genomics)** and **epidemiological (viral spread)**. Both scales share a common physical substrate — Structural Itô Calculus, Self-Organised Criticality (SOC), Semantic State Contraction (SSC), and Renormalisation Group (RG) filtering — implemented once in a shared foundation module and reused throughout the cluster.

The cluster is intentionally **vendor-neutral**: it runs on CPU (≥ 3 GB RAM), Google Colab T4, Apple MPS, Huawei Ascend (NPU), and multi-GPU nodes without code changes.

---

## File Map

```
one_core_evolution_v2.py                      ← Shared foundation (single source of truth)
evolution_one_v3.py                           ← Cancer / somatic evolution engine
evolution_one_epidemiological_viral_v4.py     ← Epidemiological / viral evolution engine
structural_langevin_evo_v3.py                 ← BAOAB Langevin MD integrator
```

All four files share a strict dependency order:

```
one_core_evolution_v2
        ↑
        ├── evolution_one_v3
        ├── evolution_one_epidemiological_viral_v4
        └── structural_langevin_evo_v3
```

---

## File 1 — `one_core_evolution_v2.py`
### ONE CORE EVOLUTION — Shared Foundation

This module is the **single source of truth** for all components that are conceptually identical across the cancer engine, the epidemiological engine, and the Langevin integrator. It eliminates code duplication and ensures that every bug fix propagates automatically to all consumers.

### Exported Classes & Functions

| Symbol | Type | Role |
|---|---|---|
| `get_device()` | function | Hardware-backend selector: CUDA → MPS → NPU → CPU |
| `SemanticStateContraction` | `nn.Module` | SSC EMA low-pass filter for structural stress σ (Paper 4) |
| `CSOCBase` | abstract `nn.Module` | Base class for all CSOC adaptive-parameter modules |
| `InterfaceDetectorBase` | abstract `nn.Module` | Base class for differentiable interface detectors |
| `StructuralItoBase` | abstract `nn.Module` | Base class for Structural Itô drift-correction modules |
| `DifferentiableRG` | `nn.Module` | Fully differentiable learnable 1-D RG smoothing kernel |
| `DifferentiableSOC` | `nn.Module` | Fully differentiable SOC temperature modulation |
| `DifferentiableIto` | `nn.Module` | Euler-Maruyama Itô SDE integrator with double-well energy |
| `CheckpointManager` | class | Unified pickle-based checkpoint save / load |
| `LangevinEvolutionBridge` | class | Bridges BAOAB Langevin MD ↔ population-level μ / Rt |
| `EpiEvolutionBridge` | class | Bidirectional coupling μ (cancer) ↔ Rt (epidemic) |
| `EVOLUTION_VERSION` | `str` | Ecosystem-wide version string (`"2.0.0"`) |

### Key Design Details

**SemanticStateContraction (SSC)**
Implements an Exponential Moving Average (EMA) low-pass filter over structural stress σ. The canonical fix over prototype versions is the use of a boolean `_initialized` buffer rather than the brittle `prev == 0.0` check, which failed silently when the true first stress value was exactly zero. The `reset()` method clears both the EMA buffer and the flag, enabling safe reuse across independent patient cohorts or viral lineages. The module auto-migrates to the device of the incoming tensor.

**DifferentiableRG**
A learnable 1-D convolution kernel with `nn.Parameter` weights. The critical fix (Bug 2) replaces `weight / (sum + 1e-8)` normalisation with `F.softmax`, which is numerically stable and has well-defined gradients everywhere. The kernel is end-to-end trainable together with downstream classifiers.

**DifferentiableSOC**
Evolves a mutation-load or Rt tensor through SOC relaxation dynamics. Bug 1 fixes replace `torch.std()` (which has zero gradient at constant tensors) with a `logsumexp`-based soft spread estimator, and replace `clamp(0, 10)` with a `softplus` floor and a mirrored `softplus` soft ceiling, preserving gradients across the entire range.

**DifferentiableIto**
Euler-Maruyama integration of a double-well SDE: `dx = −∇E dt + √(2 k_B T dt) dW`, where `E(x) = ½(x−0.5)² + 0.1 sin(2πx)`. Bug 3 replaces `clamp(0, 1)` with a `softplus` floor and a `sigmoid` S-curve ceiling so that gradients flow through the temperature parameter `T_param` for end-to-end training.

**LangevinEvolutionBridge**
Provides the coarse-graining step from atomic coordinates (Å) to a scalar mutation-load µ ∈ [0, 1]:
```
μ = sigmoid( mean_displacement / sigma_scale − 1.0 )
```
The displacement is SSC-filtered before projection to suppress high-frequency numerical noise. The bridge exposes `micro_step()` (one BAOAB sub-step) and `run()` (full trajectory with µ logging).

**EpiEvolutionBridge (Bug 7 fix)**
Couples the cancer engine (mutation load μ) to the epidemiological engine (effective reproduction number Rt) via two differentiable sigmoid transfer functions:
```
Rt_coupled = Rt_base + scale · σ( (μ_ssc − μ_threshold) · 10 )
μ_coupled  = μ_base  + scale · σ( (Rt_ssc − Rt_threshold) · 5 )
```
Both directions are individually SSC-filtered and fully differentiable, enabling joint training across scales.

---

## File 2 — `evolution_one_v3.py`
### EVOLUTION ONE — Multi-Level Cancer Evolution & Structural Impact Engine

This engine models cancer evolution from raw somatic mutation data to patient-level risk classification, structural impact prediction, intervention recommendation, CRISPR guide design, and epigenetic target identification.

### Class Reference

| Class | Role |
|---|---|
| `GeneNetworkBV` | Batalin–Vilkovisky consistency check on a gene interaction network |
| `MutationDataLoader` | Loads MAF and VCF mutation files; builds per-sample mutation matrices |
| `DuonAnalyzer` | Detects mutations landing in duon regions (coding + regulatory overlap via BED intervals and PyRanges) |
| `EvolutionaryClassifier` | Classifies patients into stable / critical / collapse states from mutation load µ |
| `FutureMutationPredictor` | Predicts escape mutations via ΔΔG scanning using REAL FOLD ONE HT or the embedded relaxation engine |
| `CRISPRDesigner` | Designs gRNA sequences (Doench 2016 scoring) and HDR repair templates; off-target search via Bowtie or brute-force |
| `EpigeneticDesigner` | Identifies duon-guided methylation editing targets |
| `InterventionRecommender` | Maps mutation profiles to targeted therapies and structural stabilisers |
| `RetrospectiveAnalyzer` | Correlates lifestyle / exposure factors with somatic mutation burden |
| `EvolutionONEEngine` | Top-level orchestrator: runs the full pipeline end-to-end |

### EvolutionaryClassifier — Technical Details

The classifier represents the core SOC model for cancer evolution. Mutation load µ is derived from the per-sample mutation matrix, then passed through the following differentiable pipeline:

1. **RG smoothing** via `DifferentiableRG` (kernel size 5) — temporal denoising.
2. **SOC evolution** via `DifferentiableSOC` — relaxation toward criticality.
3. **Itô evolution** via `DifferentiableIto` — stochastic trajectory exploration in double-well potential.
4. **Soft classification** via log-softmax over three logits (stable / critical / collapse).

Thresholds `threshold_stable` and `threshold_collapse` can be tuned against clinical labels using either **SciPy differential evolution** or **Optuna Bayesian optimisation**.

### Full Pipeline (EvolutionONEEngine.run)

```
Input MAF/VCF
    │
    ▼
MutationDataLoader  →  mutation matrix (samples × genes)
    │
    ▼
μ computation  →  DifferentiableRG smoothing
    │
    ▼
DuonAnalyzer  →  duon mutation rate overlay
    │
    ▼
EvolutionaryClassifier  →  per-sample state {stable, critical, collapse}
    │
    ▼
FutureMutationPredictor  →  predicted escape mutations + ΔΔG
    │
    ▼
CRISPRDesigner  →  gRNA + repair templates
EpigeneticDesigner  →  methylation targets
GeneNetworkBV  →  network consistency check
    │
    ▼
InterventionRecommender  →  therapy suggestions
RetrospectiveAnalyzer  →  lifestyle factor correlations
    │
    ▼
Results dict  +  optional checkpoint
```

### External Dependencies

| Package | Use | Fallback |
|---|---|---|
| Biopython | Sequence I/O, CRISPR reverse complement | — (CRISPR disabled) |
| PyRanges | Genomic interval overlaps for duon detection | Dictionary-based fallback |
| REAL FOLD ONE | High-throughput ΔΔG structural scanning | Embedded PyTorch energy engine |
| Bowtie | CRISPR off-target search | Brute-force Hamming-distance scan |
| Optuna | Bayesian threshold tuning | SciPy differential evolution |

---

## File 3 — `evolution_one_epidemiological_viral_v4.py`
### EVOLUTION ONE — Epidemiological Forecasting & Viral Evolution Engine

This engine mirrors the cancer engine at the population-epidemic scale. It ingests case-count time series and viral sequence data, estimates the effective reproduction number Rt, predicts future escape variants, scores epitopes, and designs poly-epitope mRNA vaccine constructs.

### Class Reference

| Class | Role |
|---|---|
| `RefinementConfig` | Dataclass for the embedded protein relaxation engine |
| `DifferentiableProteinEnergy` | Differentiable Lennard-Jones + electrostatic energy model (DIFF-FIX 4: soft neighbour count via sigmoid) |
| `RefinementEngine` | Local protein structure relaxation using differentiable energy minimisation (DIFF-FIX 5: scatter-based assembly) |
| `EpidemicClassifier` | Classifies epidemic states (controlled / outbreak / pandemic) from Rt; thresholds enforced via cumulative softplus (DIFF-FIX 8) |
| `EpidemiologicalDataLoader` | Loads OWID-style case CSV; memory-efficient FASTA sequence streaming |
| `ViralEvolutionAnalyzer` | Mutation frequency analysis; dN/dS ratio via Nei–Gojobori method |
| `FutureVariantPredictor` | ΔΔG-based escape variant prediction (REAL FOLD ONE HT or embedded engine) |
| `EpitopeScorer` | Differentiable neural epitope scoring (DIFF-FIX 6: no detach on sigmoid outputs) |
| `VaccineDesigner` | Ranks epitopes and assembles poly-epitope mRNA constructs |
| `TherapeuticRecommender` | Maps protein mutation profiles to antiviral recommendations |
| `EpidemiologicalFactorAnalyzer` | Correlates external factors (mobility, temperature, humidity) with Rt |
| `InteractionNetworkBV` | BV consistency check for host–pathogen interaction networks |
| `EpiForecastEngine` | Top-level orchestrator; includes AMP GradScaler training (DIFF-FIX 7) |

### EpidemicClassifier — Technical Details

The classifier takes a time series of Rt values and returns log-probabilities over three states: controlled (Rt < 1), outbreak (1 ≤ Rt < pandemic threshold), pandemic (Rt ≥ pandemic threshold).

The threshold parameters are stored as raw scalars and converted via a cumulative softplus chain:
```
threshold_outbreak = softplus(raw_outbreak)
threshold_pandemic = threshold_outbreak + softplus(raw_gap)
```
This guarantees `threshold_outbreak < threshold_pandemic` at all times without a hard constraint, preserving gradient flow (DIFF-FIX 8). The output uses `F.log_softmax` over three logits rather than sigmoid subtraction, which produced unnormalised probabilities in v1 (DIFF-FIX 3).

### Full Pipeline (EpiForecastEngine.run)

```
Input case CSV  +  FASTA sequences
    │
    ▼
EpidemiologicalDataLoader  →  prevalence matrix (time × variant)
    │
    ▼
Rt estimation  →  DifferentiableRG smoothing
    │
    ▼
ViralEvolutionAnalyzer  →  mutation hotspots + dN/dS
    │
    ▼
EpidemicClassifier  →  per-timepoint state {controlled, outbreak, pandemic}
    │
    ▼
FutureVariantPredictor  →  predicted escape variants + ΔΔG
    │
    ▼
EpitopeScorer  →  ranked epitope list
VaccineDesigner  →  poly-epitope mRNA construct
InteractionNetworkBV  →  host-pathogen network consistency
    │
    ▼
TherapeuticRecommender  →  antiviral / mAb recommendations
EpidemiologicalFactorAnalyzer  →  factor correlations
    │
    ▼
Results dict  +  optional checkpoint
```

### v3 → v4 Changes (Integration with ONE Core Evolution)

| Bug | Fix |
|---|---|
| Bug 6 | Removed local duplicate definitions of `DifferentiableSOC`, `LearnableRG`, `CheckpointManager`, `LangevinEvolutionBridge`; all imported from `one_core_evolution` |
| Bug 7 | `EpiEvolutionBridge` imported from `one_core_evolution`; `attach_evo_bridge()` added to `EpiForecastEngine` for bidirectional µ ↔ Rt coupling |

### v1/v2 → v3 Differentiability Fixes

| DIFF-FIX | Location | Fix |
|---|---|---|
| 1 | `LearnableRG` | Hard clamp on kernel weights → `softplus`; conv weight normalised via `softmax` |
| 2 | `DifferentiableSOC` | `std()` + hard scale → `logsumexp` soft spread |
| 3 | `EpidemicClassifier` | `p_outbreak` subtraction → `log_softmax` over logits |
| 4 | `DifferentiableProteinEnergy` | Hard neighbour count → differentiable soft-count via `sigmoid` |
| 5 | `RefinementEngine.relax_local` | In-place index writes → `scatter`-based differentiable assembly |
| 6 | `EpitopeScorer` | `sigmoid` outputs kept in computation graph (removed `detach`) |
| 7 | `EpiForecastEngine.train_classifier` | AMP `GradScaler` added for mixed-precision CUDA training |
| 8 | `EpidemicClassifier` thresholds | Double `softplus` ordering guaranteed via cumulative sum |

---

## File 4 — `structural_langevin_evo_v3.py`
### ADVANCED STRUCTURAL LANGEVIN — BAOAB Integrator with Structural Calculus

This module implements the **microscale** layer of the EVOLUTION ONE cluster. It integrates atomic or coarse-grained coordinates under a physically rigorous BAOAB Langevin scheme extended by the four-paper Structural Calculus framework:

| Paper | Contribution |
|---|---|
| Paper 1 | Regime-Dependent Analytical Framework (Structural Operators: D^S u = ∇u + [u] δ_Γ) |
| Paper 2 | BV Jump Measures & Self-Evolving Interfaces (multiplicative structural noise at Γ) |
| Paper 3 | Structural Itô Calculus & Multiplicative Noise Correction (½ G(x) ∇_x G(x) drift) |
| Paper 4 | Controlled Self-Organised Criticality (CSOC) & SSC Adaptive Thermostat |

### Class Reference

| Class | Inherits from | Role |
|---|---|---|
| `InterfaceDetector` | `InterfaceDetectorBase` | Differentiable per-atom soft interface mask ∈ [0, 1] |
| `CSOCThermostat` | `CSOCBase` | Adaptive temperature T and friction γ from real-time structural stress |
| `StructuralItoNoise` | `StructuralItoBase` | Multiplicative noise + Itô drift correction via autograd |
| `AdvancedStructuralLangevin` | `nn.Module` | Full BAOAB integrator combining all four papers |

### InterfaceDetector

Computes a **soft interface score** per atom from the variance of pairwise distances within a cutoff radius `r_cut`:

```
w_ij = sigmoid( sharpness · (r_cut − dist_ij) )   [soft neighbourhood weight]
var_d = weighted variance of distances              [heterogeneity measure]
mask_i = sigmoid( sharpness · (std_d − mean_d · 0.3) )
```

All operations are differentiable with respect to atomic coordinates. Bug 4a replaces `w_sum.clamp(min=1e-8)` with a `softplus` floor; Bug 4b replaces `var_d.clamp(min=0.0)` with `F.softplus`.

### CSOCThermostat

Inherits `CSOCBase` and `SemanticStateContraction` from `one_core_evolution`. On each step it:

1. Computes raw structural stress σ = mean per-atom displacement.
2. Applies SSC low-pass filtering.
3. Converts normalised deviation `dev = (σ − σ_target) / σ_target` into adaptive temperature and friction via sigmoid interpolation.

Bug 4c replaces `adaptive_T.clamp(T_lo, T_hi)` with a double-sided `softplus` bound:
```
adaptive_T = T_lo + softplus(adaptive_T − T_lo)    # soft floor
adaptive_T = T_hi − softplus(T_hi − adaptive_T)    # soft ceiling
```

### StructuralItoNoise

Computes the Structural Itô drift correction (Paper 3):
```
G(x) = 1 + amp · mask(x)
drift = ½ G(x) ∇_x G(x)
```
The gradient `∇_x G` is computed via `torch.autograd.grad` with `interface_detector` called inside an `enable_grad()` context so that `mask` is a differentiable function of coordinates. Bug 4d replaces `(1 − c1²).clamp(min=0)` in the BAOAB O-step with `F.softplus`.

### AdvancedStructuralLangevin — BAOAB Splitting

The integrator separates one timestep into five sub-operations:

```
B  ─  half-step velocity update with structural force F^S = ∇U + [u] δ_Γ
A  ─  half-step position update
O  ─  Ornstein–Uhlenbeck friction + noise (thermostat)
A  ─  second half-step position update
B  ─  second half-step velocity update with new force
```

Structural noise during the O-step uses per-atom amplitude `G(x)` from `StructuralItoNoise`, focusing stochasticity near self-evolving interfaces. The Itô drift correction is added as an additive bias to the velocity at each B-step.

**Convenience wrapper:** `full_step(coords, velocities, force_fn, jumps)` executes the complete BAOAB cycle in a single call, returning `(new_coords, new_velocities, T_scalar, sigma_scalar)`.

**State management:** `_prev_coords` and `_state_ready` are `register_buffer` tensors so they are saved/loaded with `torch.save` / `torch.load` automatically. `reset()` clears all buffers and the thermostat SSC filter between independent trajectories.

---

## Cross-Engine Integration

### Langevin MD → Population Scale

```python
from structural_langevin_evo_v3 import AdvancedStructuralLangevin
from one_core_evolution_v2 import LangevinEvolutionBridge

integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=310.0)
bridge = LangevinEvolutionBridge(integrator, sigma_scale=1.0)

# Run MD trajectory; obtain µ history for EvolutionaryClassifier
final_coords, mu_history = bridge.run(
    coords, coords_ref, force_fn, n_steps=1000)

# Feed into EvolutionaryClassifier
state = classifier.mu_to_state(mu_history[-1])
```

### Cancer Engine ↔ Epidemiological Engine

```python
from evolution_one_v3 import EvolutionONEEngine
from evolution_one_epidemiological_viral_v4 import EpiForecastEngine

evo = EvolutionONEEngine(cfg)
epi = EpiForecastEngine(cfg)

# Attach bidirectional bridge (Bug 7 fix)
bridge = epi.attach_evo_bridge(evo, mu_to_rt_scale=0.5)

# Couple mutation load into Rt
mu_tensor = torch.tensor(evo.results['mu_smooth'])
rt_base   = torch.tensor(epi.results['rt_smooth'])
rt_coupled = bridge.mu_to_rt(mu_tensor, rt_base)

# Couple Rt back into µ (immune pressure on tumour)
mu_coupled = bridge.rt_to_mu(rt_base, mu_tensor)
```

---

## Shared Theoretical Framework

All four files are grounded in the same four-paper Structural Calculus ecosystem:

| Concept | Symbol | Role in cluster |
|---|---|---|
| Semantic State Contraction | SSC / σ_ε | EMA low-pass filter; shared across all scales |
| Controlled SOC | CSOC | Adaptive temperature & friction in Langevin; threshold evolution in classifiers |
| Renormalisation Group | RG | Smoothing of µ and Rt time series before classification |
| Structural Itô Calculus | ½ G ∇G | Drift correction for multiplicative noise at interfaces |
| BV Jump Measures | [u] δ_Γ | Structural force enhancement at evolving interfaces |
| EpiEvolutionBridge | µ ↔ Rt | Cross-scale coupling: viral mutation pressure ↔ epidemic transmission |

---

## Installation

```bash
# Core requirements
pip install torch numpy pandas scipy matplotlib

# Recommended (full feature set)
pip install biopython pyranges optuna

# Optional (external binaries)
# Bowtie 1  →  for CRISPR off-target search
# REAL FOLD ONE  →  for high-throughput ΔΔG structural scanning
```

The cluster auto-detects available packages at import time and activates fallback code paths gracefully when optional dependencies are absent.

---

## Hardware Selection

```python
from one_core_evolution_v2 import get_device

device = get_device("cuda")   # CUDA → MPS → NPU → CPU (auto-fallback)
device = get_device("mps")    # Apple Silicon preferred
device = get_device("cpu")    # forced CPU
```

---

## Checkpointing

```python
from one_core_evolution_v2 import CheckpointManager

# Save
CheckpointManager.save("run_001.ckpt", {
    "threshold_stable":   classifier.threshold_stable,
    "threshold_collapse": classifier.threshold_collapse,
    "results":            engine.results,
})

# Resume
ckpt = CheckpointManager.load("run_001.ckpt")
if ckpt:
    classifier.threshold_stable   = ckpt["threshold_stable"]
    classifier.threshold_collapse = ckpt["threshold_collapse"]
```

---

## Version History

### `one_core_evolution` v1 → v2
- Bug 1: `DifferentiableSOC` — `std()` + `clamp(0,10)` → `logsumexp` spread + `softplus` floor/ceiling
- Bug 2: `DifferentiableRG` — `weight / sum` → `F.softmax` normalisation
- Bug 3: `DifferentiableIto` — `clamp(0,1)` → `softplus` floor + `sigmoid` ceiling
- Bug 7: `EpiEvolutionBridge` added; `EpiForecastEngine` ↔ `EvolutionONEEngine` coupling

### `structural_langevin_evo` v2 → v3
- Bug 4a: `w_sum.clamp(min=1e-8)` → `softplus` floor in `InterfaceDetector`
- Bug 4b: `var_d.clamp(min=0.0)` → `F.softplus` in `InterfaceDetector`
- Bug 4c: `adaptive_T.clamp(lo, hi)` → double-sided `softplus` in `CSOCThermostat`
- Bug 4d: `(1−c1²).clamp(min=0)` → `F.softplus` in BAOAB O-step

### `evolution_one_epidemiological_viral` v3 → v4
- Bug 6: Removed local duplicate shared classes; canonical imports from `one_core_evolution`
- Bug 7: `EpiEvolutionBridge` integration and `attach_evo_bridge()` method

---

## Citation

If you use the EVOLUTION ONE cluster in research, please cite:

> Yoon A Limsuwan (2026). *EVOLUTION ONE: A Fully Differentiable PyTorch Framework for Multi-Scale Biological Evolution Simulation*. MSPS NETWORK. MIT License. ORCID: 0009-0008-2374-0788. GitHub: yoonalimsuwan.

---

*This README was generated for EVOLUTION ONE Cluster v3 / ONE Core Evolution v2.0.0.*
