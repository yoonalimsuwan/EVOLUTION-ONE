# Organelle Layer — Sub-Cellular Modelling for the ONE Ecosystem

**Module:** `cell_population_one.py` (v1.1.0+)
**Companion module:** `structural_gno_evolution_bv_standalone.py` (v1.2.0+)
**Cluster:** EVOLUTION ONE
**Status:** Production / actively developed

This document describes the `OrganelleLayer` — a per-cell sub-cellular
model that sits beneath `PhenotypeLayer` in `cell_population_one.py` — and
how it is wired into the `structural_gno_evolution_bv_standalone.py`
graph-neural-operator (GNO) training pipeline as the **POP-2** feature.

---

## 1. What problem this solves

`CellPopulation` already modelled cells as agents with a position, a
genotype, and (via `PhenotypeLayer`) a gene-expression state. What it
*didn't* model was **why** a cell's machinery might be failing at a level
finer than "expression went up or down." `OrganelleLayer` adds that next
level down: four coupled sub-cellular subsystems — mitochondria, nucleus,
lysosome, and endoplasmic reticulum — each with its own small ODE state,
each coupled to the others the way real organelle crosstalk works.

This is not cosmetic detail. It changes *why* a cell divides or dies:

- A cell can now die from **DNA-damage checkpoint failure** (a p53-style
  mechanism), not only from an unfavourable local environment.
- A cell can now **fail to divide because it is starved of ATP**, even if
  every other signal favours division.
- Damage in one organelle now **propagates** to the others (ROS → DNA
  damage; ATP scarcity → unfolded protein → UPR stress → membrane
  potential drag), instead of every fitness signal being independent.

---

## 2. The four organelles

| Organelle | State tracked | Core mechanism |
|---|---|---|
| **Mitochondria** | `mito_atp`, `mito_psi`, `mito_ros` | ATP is produced from the local CH3D sigma field (a nutrient-availability proxy), gated by membrane potential (`mito_psi`). Production leaks ROS, more so as `mito_psi` falls (electron-transport-chain leak). |
| **Nucleus** | `nuc_damage`, `nuc_repair_capacity` | DNA damage accumulates from ROS and from structural/replication stress (`\|sigma − 1\|`), repaired at a rate set by `nuc_repair_capacity` — which itself degrades under ATP scarcity (repair is energetically expensive). Above a threshold, damage triggers a smooth, p53-like checkpoint penalty. |
| **Lysosome** | `lyso_capacity`, `lyso_autophagy_flux` | Capacity degrades slowly with cell age (age-related lysosomal exhaustion) and is partially restored on division. Flux tracks current clearance *demand* (ROS + unfolded protein), not just raw capacity — an undamaged cell shows near-zero flux. |
| **Endoplasmic reticulum** | `er_unfolded`, `er_upr_stress` | Unfolded protein accumulates under ATP scarcity (folding is ATP-dependent) and is cleared by lysosomal flux. Past a threshold, UPR activation engages smoothly and drags mitochondrial membrane potential down — the real ER–mitochondria Ca²⁺ stress-crosstalk pathway.

Every threshold-based effect (the nuclear checkpoint, UPR activation, the
ATP division gate) is implemented with `softplus`/`sigmoid`, never a hard
cutoff — the whole layer stays fully differentiable end to end.

---

## 3. Two pathways into the rest of the ecosystem

Organelle state affects a cell's fate through two distinct, additive
pathways:

1. **Slow pathway — into `PhenotypeLayer`.**
   `organelle_drive_to_phenotype()` broadcasts a single scalar
   `organelle_health()` summary into every phenotype channel, added on top
   of whatever gene-regulatory drive a caller already set via
   `PhenotypeLayer.set_gene_drive()`. This is injected **per step**
   through `PhenotypeLayer.update()`'s `extra_drive` argument — it does
   **not** mutate the persistent `_gene_drive` buffer, so it can never
   silently accumulate step over step.

2. **Fast pathway — directly into `CellPopulation`'s division/death logit.**
   `OrganelleLayer.fitness_contribution()` is added straight into the
   division/death logits computed in `CellPopulation.step()`, bypassing
   `PhenotypeLayer`'s ODE lag entirely. This matters biologically: a
   nucleus with damage past the checkpoint threshold should be able to
   trigger apoptosis *this step*, not several steps later once a slow
   expression ODE catches up.

On top of both pathways, `OrganelleLayer.atp_division_gate()` applies a
**multiplicative** (never additive) gate on `division_rate` — a starved
cell's division probability is suppressed regardless of how favourable
every other signal is, which an additive logit term cannot guarantee.

```
                 ┌─────────────────────┐
   local sigma → │   OrganelleLayer    │
   (CH3D field)  │  Mito / Nuc / Lyso  │
                 │        / ER         │
                 └──────────┬──────────┘
                  fast │            │ slow
        fitness_contribution()   organelle_drive_to_phenotype()
                  │                  │
                  ▼                  ▼
       division/death logit   PhenotypeLayer.update()
       (CellPopulation.step)   (extra_drive, per-step only)
```

---

## 4. Attaching the layer

```python
from cell_population_one import CellPopulation, CellPopulationConfig, OrganelleConfig

pop = CellPopulation(CellPopulationConfig(n_max=4096, n_genotypes=8))
organelle = pop.attach_organelle_layer(OrganelleConfig(n_max=4096))

# Optional: layer PhenotypeLayer on top, so organelle health also
# shapes gene expression (not just division/death directly).
phenotype = pop.attach_phenotype_layer()

for _ in range(100):
    out = pop.step(sigma_field=my_ch3d_field)
    # out["organelle_term"]     — this step's fast-pathway fitness contribution
    # out["organelle_state"]    — the full OrganelleState (9 tensors)
    # out["atp_division_gate"]  — this step's multiplicative ATP gate
```

With no `OrganelleLayer` attached, `CellPopulation.step()` behaves exactly
as it did before this feature existed — every organelle-derived term
defaults to a neutral no-op. This is a strict superset of the
pre-organelle model; nothing breaks for existing callers.

`attach_organelle_layer()` always force-matches `OrganelleConfig.n_max` to
the population's own `n_max`, the same convention `attach_phenotype_layer()`
already uses — you cannot accidentally attach a shape-incompatible layer.

### Checkpointing

`CellPopulation.save_checkpoint()` / `load_checkpoint()` persist organelle
state automatically when a layer is attached, and remain fully backward
compatible with checkpoints saved before v1.1.0 (the `organelle_state` key
is simply absent from old files, and skipped on load if no
`OrganelleLayer` is currently attached).

---

## 5. Integration with the GNO bridge (`structural_gno_evolution_bv_standalone.py`)

The training pipeline's `CellPopulationTrainingBridge` (Mode 4 / POP-1)
attaches both `OrganelleLayer` and `PhenotypeLayer` automatically when the
right config flags are set and a compatible `cell_population_one` (v1.1.0+)
is importable:

```python
from structural_gno_evolution_bv_standalone import SGNOEvoBVConfig, StructuralGNOEvolutionBV

cfg = SGNOEvoBVConfig(
    enable_cell_population=True,
    enable_cellpop_organelle=True,   # default: True
    enable_cellpop_phenotype=True,   # default: True
    lambda_organelle_distill=0.15,   # default
    organelle_health_rt_scale=0.3,   # default
)
model = StructuralGNOEvolutionBV(cfg)

model.cellpop_bridge.organelle  # the attached OrganelleLayer, or None
model.cellpop_bridge.phenotype  # the attached PhenotypeLayer, or None
```

### POP-2: organelle ↔ GNO cross-modal distillation

This is the same composition pattern as **BV-1** (the existing
`bv_full_theory_one` distillation term), one rung further down the
cross-cluster stack: instead of distilling an exact analytic gauge theory,
**POP-2** distills `OrganelleLayer`'s own exact, parameter-free
sub-cellular health signal into the *same* Mode-1 ΔRt channel BV-1 already
targets.

```
CellPopulationTrainingBridge.rollout(pred_u)
        │
        ▼
CellPopulationRollout.organelle_health_trace   (differentiable, per-step)
        │
        ▼
organelle_health_and_boost(rollout)             ← detached, tanh-bounded
        │
        ▼
loss_organelle_distillation(mu_rt_pred, boost)  ← trains ONLY the Mode-1 head
```

Like BV-1's target, `organelle_boost` is **detached** before use — this is
a one-directional self-distillation signal. Gradients never flow from the
target back through the rollout into the GNO's Mode-3/Mode-4 graph a
second time; `loss_organelle_distillation` only ever updates the Mode-1
evolution head. It introduces **zero new trainable parameters**.

If both BV-1 and POP-2 are active in the same training run, their two
targets are simply summed onto the same `[Δμ, ΔRt]` channel before the
gradient step — they are complementary distillation signals, not
competing ones.

### What's logged

| Log key | Meaning |
|---|---|
| `loss_organelle_distill` | POP-2's weighted loss contribution this step |
| `organelle_health_final` | Mean `organelle_health()` at rollout end |
| `organelle_atp_final` | Mean `mito_atp` at rollout end (diagnostic) |

These appear in `train_step()`'s returned log dict whenever a Mode-1 batch
is present and the bridge's `OrganelleLayer` is available, and are
included in the periodic `pop_cert_every` summary line alongside the
existing Mode-4 (POP-1) diagnostics.

### Graceful degradation

| Condition | Result |
|---|---|
| `cell_population_one` not importable at all | Mode 4 (POP-1) and POP-2 both disabled; Modes 1–3 and BV-1/2/3 unaffected. |
| `cell_population_one` is v1.0.0 (pre-organelle) | POP-1 still works; POP-2 disables itself with a single `RuntimeWarning`. |
| `enable_cellpop_organelle=False` | POP-2 disabled; POP-1 unaffected. |
| `enable_cellpop_phenotype=False` | `PhenotypeLayer` not attached; organelle→phenotype slow pathway inactive, but POP-2 (which only needs the organelle layer) is unaffected. |

At every one of these boundaries, training and inference degrade rather
than hard-fail — exactly the same convention `bv_full_theory_one`'s
optional dependency chain already follows.

---

## 6. Key configuration reference

### `OrganelleConfig` (in `cell_population_one.py`)

The most commonly tuned fields:

| Field | Default | Effect |
|---|---|---|
| `atp_division_floor` | `0.25` | Below this `mito_atp`, division probability is smoothly suppressed. |
| `nuc_checkpoint_threshold` | `0.6` | DNA damage level above which the apoptosis-style checkpoint penalty engages. |
| `nuc_checkpoint_gain` | `3.0` | How sharply the checkpoint penalty ramps up past threshold. |
| `lyso_capacity_age_decay` | `0.0015` | Rate of age-related lysosomal exhaustion. |
| `er_upr_threshold` | `0.5` | Unfolded-protein level above which UPR activation engages. |
| `phenotype_drive_gain` | `1.0` | Strength of the organelle→phenotype slow pathway. |

See the `OrganelleConfig` docstring in `cell_population_one.py` for the
full field list (30+ parameters across all four organelles).

### `SGNOEvoBVConfig` additions (in `structural_gno_evolution_bv_standalone.py`)

| Field | Default | Effect |
|---|---|---|
| `enable_cellpop_organelle` | `True` | Master switch for attaching `OrganelleLayer`. |
| `enable_cellpop_phenotype` | `True` | Master switch for attaching `PhenotypeLayer` on top of the organelle layer. |
| `cellpop_phenotype_channels` | `("proliferation", "stress_response", "differentiation")` | Channel names for the attached `PhenotypeLayer`. |
| `lambda_organelle_distill` | `0.15` | Loss weight for the POP-2 distillation term. |
| `organelle_health_rt_scale` | `0.3` | Max magnitude of the organelle-derived Rt boost target. |

---

## 7. Testing

`cell_population_one.py`'s built-in smoke test (`python cell_population_one.py`)
includes dedicated organelle coverage:

- ATP response to favourable vs. starvation conditions.
- ROS-driven membrane potential decay.
- Smooth checkpoint penalty engagement above the DNA-damage threshold.
- ATP division gate suppression under starvation.
- A full live-population rollout with finite state across all 8 tracked channels.
- A regression guard confirming the organelle→phenotype pathway never
  mutates `PhenotypeLayer._gene_drive` (the accumulation bug this design
  specifically avoids).

`structural_gno_evolution_bv_standalone.py`'s smoke test
(`python structural_gno_evolution_bv_standalone.py`) additionally verifies:

- Gradient flow from `organelle_health_trace` back into the GNO's
  `ch3d_head` parameters.
- `organelle_health_and_boost()` produces a bounded, detached target.
- `loss_organelle_distillation` trains the Mode-1 head correctly.
- `phenotype_expr_final` is finite when a `PhenotypeLayer` is attached.

All organelle-specific checks `[SKIP]` cleanly (rather than failing) when
run against a `cell_population_one` older than v1.1.0.

---

## 8. Design notes worth remembering

- **Nothing here is parameter-free vs. learned is mixed up.** Every
  organelle ODE in `cell_population_one.py` is exact and parameter-free —
  there is nothing for gradient descent to update inside `OrganelleLayer`
  itself. "Training" in this context always means using organelle state as
  a *target* for the GNO's own (genuinely trainable) Mode-1/Mode-3 heads,
  never the reverse.
- **The slow pathway is non-accumulating by construction.** Early in
  development, the natural-looking implementation mutated
  `PhenotypeLayer._gene_drive` directly each step, which silently
  accumulated the organelle contribution across the rollout. The shipped
  design instead threads the organelle signal through `update()`'s
  `extra_drive` parameter, which is consumed once per call and never
  persisted. Test 27 in `cell_population_one.py`'s smoke test exists
  specifically to guard against this regression resurfacing.
- **Version compatibility is checked, not assumed.** Any code attaching
  `OrganelleLayer` should check `cell_population_one`'s `CELL_POPULATION_VERSION`
  (≥ `"1.1.0"`) before relying on it being present — see
  `_CELLPOP_HAS_ORGANELLE` in `structural_gno_evolution_bv_standalone.py`
  for the reference pattern.

---

## Credits

OrganelleLayer (v1.1.0) and the POP-2 GNO integration (v1.2.0) were
developed by Claude (Anthropic) as part of the ONE Ecosystem, under
MSPS NETWORK / Yoon A. Limsuwan (ORCID: 0009-0008-2374-0788).
