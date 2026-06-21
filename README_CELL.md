# CELL POPULATION ONE — Agent-Based Cell Population Layer

**Module covered:** `cell_population_one.py`
**Cluster:** EVOLUTION ONE — sits alongside `evolution_one_v4.py`, `evolution_one_epidemiological_viral_v5.py`, `structural_cahn_hilliard_3d_v3.py`, `one_core_evolution_v3.py`
**Version:** see `CELL_POPULATION_VERSION` in the file

A batched, fully differentiable **agent-based cell population model**. Each agent ("cell") carries a genotype, a dynamic multi-channel phenotype, and a spatial position, and undergoes stochastic division/death every step. This is the layer that was missing from the ecosystem before this file: every other module operates at atomic, continuum-field, population-summary, or genomic-cohort scale — none of them resolve an actual population of individual, trackable agents.

---

## 1. Genotype vs. phenotype — both exist now, and they're different things

This distinction is worth stating plainly, since the two terms are easy to conflate:

| | Means | What this module stores |
|---|---|---|
| **Genotype** | The genetic identity a cell carries | `CellPopulationState.genotype` — a fixed, heritable integer index into a clone/mutation-matrix row (e.g. "this cell's lineage descends from Tumor_Sample_Barcode #3"). Never changes once a cell is born. |
| **Phenotype** | The *expressed*, time-varying trait that genotype + environment jointly produce | `PhenotypeLayer.state.expression` — a per-cell, per-channel (proliferation / stress-response / differentiation, by default) value that evolves every step via its own gene-regulatory ODE, driven by gene-network input and the local CH3D σ field. |

The key structural difference: **genotype is a label, phenotype is a state**. A clone's genotype never changes, so without a phenotype layer, every cell sharing a genotype was assumed to behave identically forever (a flat `genotype_fitness` lookup table — see §3). The phenotype layer breaks that assumption: two cells with the *same* genotype can now have *different*, *diverging* expression — because they sat in different local environments, or because of stochastic drift at division (`inherit_noise`) — and that divergence feeds back into different division/death rates for those two cells specifically, not just for "their clone" as a whole.

Concretely, the causal chain is now:

```
genotype (fixed label)
   │
   ├──► gene_drive (via GeneNetworkBV.phi → gene_drive_from_bv_network())  ──┐
   │                                                                          ├──► phenotype ODE ──► expression(t)
   └──► genotype_fitness (separate, simpler per-clone lookup)                │         ▲
                                                                              │         │
                                                          local CH3D σ field ─┘    fitness_contribution()
                                                                                         │
                                                          division_logit / death_logit ◄─┴── (+ genotype_fitness term, additive)
```

Both pathways — the direct `genotype_fitness` term and the phenotype-mediated term — are additive in the division/death logit (see §3). This is a strict superset of the genotype-only model: with no `PhenotypeLayer` attached, behaviour is identical to the version of this file that had no phenotype concept at all.

---

## 2. What problem this solves

Before this file, the ecosystem could compute:

- A single protein's atomic structure (REAL FOLD ONE)
- A continuum phase field over a 3-D voxel grid (CH3D / `structural_cahn_hilliard_3d_v3.py`)
- A scalar population-level mutation load μ or reproduction number Rt (EVOLUTION ONE / epidemiological)
- Per-sample mutation calls from real cancer genomics cohorts (MAF rows, `Tumor_Sample_Barcode`)
- A gene-interaction network's static BV consistency check (`GeneNetworkBV`, no dynamics of its own)

None of these track *individual cells*, and none of them give a gene network anywhere to actually *act through* — `GeneNetworkBV.phi` existed before this file only as values to be checked for consistency, never as a signal driving anything downstream. `CellPopulation` + `PhenotypeLayer` is the layer that both resolves individual agents *and* gives the gene network a place to matter dynamically.

```
EvolutionONEEngine                 GeneNetworkBV              StructuralCahnHilliard3D
  mutation matrix                    .phi (per-gene)             u(x,y,z), sigma(x,y,z)
  (samples × genes)                       │                             │
        │                                 │ gene_drive_from_              │ CellPopulationCahnHilliardBridge
        │ fitness_from_                   │ bv_network()                 │
        │ mutation_matrix()               ▼                              ▼
        ▼                          PhenotypeLayer.set_gene_drive()
   genotype_fitness ────┐                 │
                        │                 ▼
                        └────────►  CellPopulation.step()
                                    (motility → phenotype ODE update →
                                     division/death rates → division/death)
                                               │
                                               ▼
                     clone_frequencies(), population_mutation_load()
                                               │
                                               ▼
                      EpiEvolutionBridge.mu_to_rt() (existing bridge, unmodified)
```

---

## 3. Core pieces

### `CellPopulationConfig`
Dataclass following the same convention as the rest of the ecosystem (`Seq2CoarseConfig`, `CahnHilliardConfig`, etc.) — every field validated in `__post_init__`, sensible defaults, no hidden global state.

Key groups:
- **Population sizing**: `n_max` (fixed tensor capacity), `n_init`, `n_genotypes`.
- **Spatial substrate**: `grid_shape` (must match the CH3D field shape this population is coupled to), `box_size`, `motility`.
- **Division/death dynamics**: `base_division_rate`, `base_death_rate`, `sigma_division_gain`, `fitness_division_gain`, `fitness_sign`, plus `death_rate_floor` / `division_rate_ceiling` bounds.
- **Validation note**: `base_division_rate` and `base_death_rate` must lie *strictly between* `death_rate_floor` and `division_rate_ceiling` — not just in `[0, 1]`. This was tightened after an earlier version allowed `base_division_rate=1.0`, which silently broke the "rate(logit=0) == base_rate exactly" guarantee described below (a real bug, caught and fixed during verification — see §7).

### `CellPopulationState`
A "structure of arrays", not "array of structs": `position` (n_max, 3), `genotype` (n_max,), `age` (n_max,), `alive` (n_max,) bool mask — all flat tensors. Dead slots keep stale data rather than being zeroed (irrelevant once masked out, and zeroing every step would cost a write for no benefit).

**Why batched tensors instead of a list of Cell objects:** a Python for-loop over thousands–millions of agents would be both slow and effectively non-differentiable. Keeping everything as flat tensors means the whole population update is one vectorised, GPU-resident, backprop-through-able operation.

**Why fixed capacity (`n_max`) instead of a growing list:** resizing a tensor every step breaks the fixed-shape assumption that autograd graphs rely on in a training loop. Growth/shrinkage is instead handled by an alive-mask: new cells from division are written into currently-dead slots, up to capacity (a standard logistic carrying-capacity assumption, not a crash).

### `PhenotypeConfig` / `PhenotypeState` / `PhenotypeLayer` — the dynamic phenotype layer

**`PhenotypeConfig`**: `channel_names` (default `("proliferation", "stress_response", "differentiation")` — extend or rename freely, every other field's length must match), per-channel `decay_rate`, `sigma_gain`, `fitness_weights`, plus scalar `gene_gain`, `dt`, `inherit_noise`.

**`PhenotypeState`**: a single `(n_max, n_channels)` tensor — same structure-of-arrays convention as `CellPopulationState`.

**`PhenotypeLayer`** — the engine. Every `CellPopulation.step()` call, *if a `PhenotypeLayer` is attached*, runs one explicit-Euler step of:

```
dE_c/dt = -decay_rate_c · E_c + tanh(gene_gain · gene_drive_c + sigma_gain_c · (σ_local - 1))
```

per channel `c`, for every live cell. This is a deliberately *minimal* gene-regulatory model — a saturating-drive, linear-decay ODE per channel, not a mechanistic reaction network — chosen because it's the smallest dynamic state that actually earns the name "phenotype" (time-varying, environment-responsive, distinct from the static genotype that partially drives it) without requiring detailed kinetic data most callers won't have.

Key methods:
- **`set_gene_drive(gene_drive)`** — feeds in a `(n_max, n_channels)` drive tensor, typically from `gene_drive_from_bv_network()` below.
- **`update(local_sigma, alive)`** — called automatically by `CellPopulation.step()`; advances the ODE one step.
- **`fitness_contribution()`** — `(n_max,)` weighted sum of expression across channels (using `fitness_weights`), added into the division logit (and, sign-flipped, the death logit) on top of the existing `genotype_fitness` term.
- **`inherit_slots(parent_idx, child_slots)`** / **`reset_slots(dead_slots)`** — called automatically by `CellPopulation.step()` on division/death respectively, so daughter cells inherit (a noisy copy of) their parent's expression state, and a slot's expression resets to baseline the moment its occupant dies (before that slot can be reused by a different lineage).

### `gene_drive_from_bv_network(...)`
Builds a `(n_max, n_channels)` drive tensor directly from a `GeneNetworkBV` instance's `.phi` values (`evolution_one_v4.py`) — no reformatting step needed to wire EVOLUTION ONE's existing gene-interaction network into a `PhenotypeLayer`. You supply a `channel_gene_map` (e.g. `{"proliferation": ["KRAS", "BRAF"], "stress_response": ["TP53"]}`); each channel's drive is the mean φ-value of its mapped genes. Currently broadcasts the same population-level drive to every cell regardless of genotype (see the function's docstring for how to extend this to genotype-specific drive, which `GeneNetworkBV` itself has no native axis for).

### `CellPopulation`
The engine. `step(sigma_field=None, hard=None)` does, every call:

1. **Motility** — small Gaussian random-walk position update for live cells.
2. **Phenotype update** *(if attached)* — samples local CH3D σ, advances the phenotype ODE one step, computes `fitness_contribution()`.
3. **Rate computation** — combines local σ, `genotype_fitness`, and (if attached) the phenotype's `fitness_contribution()` into a division logit and a death logit — all three terms additive.
4. **Death** — sampled from the death rate; on death, any attached phenotype's expression for that slot resets to baseline.
5. **Division** — live cells past their division-probability draw spawn a same-genotype daughter into a free (dead) slot, capped by available capacity; on division, any attached phenotype's expression is copied (with optional noise) from parent to daughter.

`hard` controls the sampling mode:
- `hard=True` (default at eval time): `torch.bernoulli` — genuinely discrete alive/dead outcomes, no gradient through the sampling step itself.
- `hard=False` (default in training mode): a Gumbel-sigmoid relaxation, so gradients flow through *which* cells divided/died, not just through the rates feeding them.

`attach_phenotype_layer(cfg)` constructs and attaches a `PhenotypeLayer`, force-overriding `cfg.n_max` to match the population's own `n_max` regardless of what was passed in (the two must be shape-compatible — this is enforced automatically rather than left as a footgun).

#### The rate formula, and why it's written the way it is

```
rate(logit) = floor + (ceiling - floor) · sigmoid(logit + shift)
```

`shift` is a constant solved once at construction time so that `rate(0) == base_rate` **exactly** (not approximately) — i.e., under a neutral CH3D σ (σ=1), zero genotype-fitness bias, and zero phenotype contribution, a cell's division/death probability is precisely `cfg.base_division_rate` / `cfg.base_death_rate`. The function is a plain sigmoid of a shifted argument, so it's smooth everywhere (no piecewise branches, no kinks in the gradient) and asymptotes cleanly to `floor`/`ceiling` at extreme logits.

### `fitness_from_mutation_matrix(...)`
Builds a per-clone fitness vector directly from `MutationDataLoader.build_mutation_matrix`'s output in `evolution_one_v4.py` — no reformatting needed to go from a real cancer-genomics cohort to `CellPopulation.set_genotype_fitness(...)`. Without a ΔΔG dict, fitness is just each clone's total mutation count (normalised). With a `{gene: ΔΔG}` dict (e.g. from REAL FOLD ONE HT's per-gene scan), each mutated gene contributes its ΔΔG instead of a flat 1.

### `CellPopulationCahnHilliardBridge`
Completes the loop with CH3D two-way:

- **CH3D → cells**: σ field sampled at each cell's position feeds `CellPopulation.step` (and, transitively, the phenotype ODE if attached).
- **Cells → CH3D**: live "mutant" cells' positions (genotype index ≥ `mutant_genotype_floor`, default 1) are projected onto the grid as a small Gaussian-kernel density source term, added to `u` before the next CH3D step.

This is a sibling to `CahnHilliardEvoBridge` (`one_core_evolution_v3.py`), not a replacement — that bridge remains the cheap, agent-free mean-field path for when no agent population is in play.

### `CellPopulationMixin`
Adds `attach_cell_population(cfg)` to any engine class, mirroring `LangevinBridgeMixin.attach_langevin_bridge()` exactly.

---

## 4. Quick start

### Standalone population, genotype-only (no phenotype layer)

```python
from cell_population_one import CellPopulationConfig, CellPopulation

cfg = CellPopulationConfig(n_max=4096, n_init=256, n_genotypes=8, grid_shape=(32, 32, 32))
pop = CellPopulation(cfg)

for t in range(100):
    out = pop.step()
    print(t, pop.state.n_alive(), out["n_divided"].item(), out["n_died"].item())

print(pop.clone_frequencies())
```

### Adding a dynamic phenotype layer

```python
from cell_population_one import CellPopulationConfig, CellPopulation, PhenotypeConfig

pop = CellPopulation(CellPopulationConfig(n_max=4096, n_init=256, n_genotypes=8, grid_shape=(32, 32, 32)))
phenotype = pop.attach_phenotype_layer(PhenotypeConfig(
    channel_names=("proliferation", "stress_response", "differentiation"),
    fitness_weights=(0.6, -0.5, 0.0),   # proliferation helps division, stress hurts it
))

for t in range(200):
    out = pop.step()
    if t % 50 == 0:
        print(t, "mean expression:", phenotype.state.expression.mean(dim=0))
```

### Driving the phenotype layer from a real gene-interaction network

```python
from evolution_one_v4 import GeneNetworkBV
from cell_population_one import gene_drive_from_bv_network

bv = GeneNetworkBV(gene_names=["TP53", "KRAS", "BRAF"], interactions=[(0, 1), (1, 2)])
channel_map = {"proliferation": ["KRAS", "BRAF"], "stress_response": ["TP53"]}
channel_idx = {name: i for i, name in enumerate(phenotype.cfg.channel_names)}

drive = gene_drive_from_bv_network(bv, pop.state.genotype, channel_map, n_channels=3, channel_index=channel_idx)
phenotype.set_gene_drive(drive)
pop.step()  # this step's phenotype ODE update now uses the gene network's φ values
```

### Driven by a real mutation cohort (genotype side)

```python
from cell_population_one import CellPopulationConfig, CellPopulation, fitness_from_mutation_matrix
from evolution_one_v4 import MutationDataLoader

loader = MutationDataLoader()
mut_df = loader.load_maf("cohort.maf")
genes = ["TP53", "KRAS", "BRAF", "EGFR"]
M, samples = loader.build_mutation_matrix(mut_df, genes)

cfg = CellPopulationConfig(n_max=8192, n_init=512, n_genotypes=len(samples), grid_shape=(32, 32, 32))
pop = CellPopulation(cfg)
pop.set_genotype_fitness(fitness_from_mutation_matrix(M)[: cfg.n_genotypes])

for t in range(200):
    pop.step()
```

### Two-way coupled with CH3D

```python
import torch
from cell_population_one import CellPopulationConfig, CellPopulation, CellPopulationCahnHilliardBridge
from structural_cahn_hilliard_3d_v3 import StructuralCahnHilliard3D, CahnHilliardConfig

grid = (32, 32, 32)
ch = StructuralCahnHilliard3D(CahnHilliardConfig(dx=1.0))
pop = CellPopulation(CellPopulationConfig(grid_shape=grid, box_size=32.0, n_genotypes=2))
bridge = CellPopulationCahnHilliardBridge(ch, pop, mutant_genotype_floor=1)

u = torch.randn(*grid) * 0.01
for t in range(500):
    u, sigma_used = bridge.coupled_step(u)
```

### Via the engine mixin

```python
from evolution_one_v4 import EvolutionONEEngine
from cell_population_one import CellPopulationConfig

engine = EvolutionONEEngine(cfg={...})
population = engine.attach_cell_population(CellPopulationConfig(n_genotypes=10))
```

---

## 5. Differentiability — what gradients flow through what

| Quantity | Differentiable w.r.t. | Not differentiable w.r.t. |
|---|---|---|
| `division_rate`, `death_rate` (returned every `step()` call) | `genotype_fitness`, the sampled `sigma_field` values, and (if attached) `PhenotypeLayer`'s `fitness_weights` / `gene_drive` | — |
| `population_mutation_load()` | `genotype_fitness` (through whichever clones are currently alive) | the discrete `alive` mask / `genotype` assignment itself |
| `PhenotypeLayer.fitness_contribution()` | `fitness_weights`, and (through `update()`) `gene_drive`, `decay_rate`, `sigma_gain` | — |
| Division/death **outcomes** (which cells actually divide/die) | only with `hard=False` (Gumbel-sigmoid relaxation) | with `hard=True` (hard `torch.bernoulli`) |
| `clone_frequencies()` | nothing — pure counting/diagnostic readout | everything (by design; use `population_mutation_load()` for a gradient pathway) |

Practical implication: if you want to train something (e.g. fit `fitness_weights` or `sigma_gain` to match an observed clonal-sweep outcome), use `hard=False` during the training loop and read gradients through `step()`'s returned `division_rate`/`death_rate`/`phenotype_term`.

---

## 6. Scope — what this deliberately does not do

- No tissue/organ mechanics (no extracellular matrix, no explicit cell-cell adhesion forces).
- No immune-cell interactions.
- No spatial neighbor-to-neighbor signalling between cells (the only spatial coupling is *through* the shared CH3D field, not direct cell-cell messages).
- The phenotype layer is a minimal per-channel ODE, not a mechanistic gene-regulatory network simulator — it has no notion of transcription/translation delay, no stochastic gene-expression bursting, and `gene_drive_from_bv_network`'s default broadcast is population-wide, not genotype-specific (see that function's docstring for how to extend it).

This is the minimum agent-based layer needed to ask population-of-individuals, genotype-vs-phenotype-divergence questions — clonal competition, spatial structure effects on a sweep, environment-driven phenotypic plasticity within a single clone — that no other module in the ecosystem currently answers. Each gap above is a legitimate next layer if/when needed, not an oversight in this one.

---

## 7. Verification

```bash
python cell_population_one.py
```

20 `[PASS]`/`[FAIL]` checks. Genotype-only checks (1–13): initial-state shapes; population staying within `n_max` capacity; division preserving genotype identity; high-division/zero-death growing the population and the reverse shrinking it; `fitness_sign=±1` behaviour; CH3D σ field measurably changing division rate; gradient flow through `population_mutation_load()` and through `step()`'s rates in relaxed-sampling mode; `fitness_from_mutation_matrix`'s plain and ΔΔG-weighted outputs; the CH3D bridge's source field and coupled step; checkpoint round-tripping; mixin wiring.

Phenotype-layer checks (14–20): `attach_phenotype_layer()` force-matching `n_max`; the ODE rising toward its expected fixed point under sustained drive and decaying to baseline under none; `fitness_contribution()`'s channel weighting against hand-computed values; an attached `PhenotypeLayer` measurably biasing `CellPopulation.step()`'s division rate end-to-end; `inherit_slots()`/`reset_slots()` correctness (exact copy at `inherit_noise=0`, measurable perturbation at `inherit_noise>0`, baseline reset on death) plus a live 10-step smoke test with real stochastic division/death; `gene_drive_from_bv_network()`'s per-channel averaging against hand-computed values; and a regression guard confirming that with no `PhenotypeLayer` attached, `step()`'s output is bit-for-bit what it was before this layer existed.

Two real bugs were caught and fixed during this verification pass, both via independent pure-Python re-derivation before either was adopted:
1. An earlier draft of the rate formula (§3) did not reproduce `base_rate` at neutral conditions (~0.475 instead of 0.05 in one check).
2. `CellPopulationConfig` originally allowed `base_division_rate`/`base_death_rate` anywhere in `[0, 1]`, but values outside `(death_rate_floor, division_rate_ceiling)` silently broke the same "exact at neutral conditions" guarantee for a different reason (the shift-derivation's target probability fell outside `(0, 1)`); validation was tightened to catch this at config-construction time instead of producing a quietly-wrong rate at runtime.

If you modify the rate formula, the ODE, or any of the bridge math, re-run the suite — this module's guarantees are testable, falsifiable claims, not assumptions.

---

## 8. Dependencies

| Dependency | Required for | Notes |
|---|---|---|
| `torch` | Everything. | `nn.functional.grid_sample` is used for trilinear CH3D field sampling — no special build requirements beyond a normal PyTorch install. |
| `one_core_evolution` | `SemanticStateContraction`, `CheckpointManager`, `get_device`, `EVOLUTION_VERSION` | Optional — the file ships standalone fallbacks for all four and logs a warning if `one_core_evolution` isn't importable. |
| `evolution_one_v4.MutationDataLoader` / `GeneNetworkBV` | `fitness_from_mutation_matrix`'s real-cohort path / `gene_drive_from_bv_network`'s gene-network path | Optional — both functions only need plain array-like / duck-typed `.phi`/`.gene_names` inputs; neither imports `evolution_one_v4` directly. |
| `structural_cahn_hilliard_3d_v3.StructuralCahnHilliard3D` | `CellPopulationCahnHilliardBridge` | Only needed if you use the bridge — `CellPopulation` on its own has no CH3D dependency. |
