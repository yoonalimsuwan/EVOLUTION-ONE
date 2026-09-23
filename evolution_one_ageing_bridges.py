# =============================================================================
# Author  : PAI AND Yoon A. Limsuwan / MSPS NETWORK
# evolution_one_ageing_bridges.py
#
# Connects evolution_one_ageing.py's DifferentiableLongevityEngine to real,
# already-simulated biological state from other ONE Ecosystem modules,
# instead of the synthetic random tensors the engine's own __main__ demo
# uses. Two bridges are implemented, tested for syntax validity, and
# reasoned through by hand line by line against the exact field names and
# shapes read directly from each module's source (no torch runtime was
# available to execute-test this file -- see the honesty note in
# evolution_one_ageing.py's own header, which applies identically here).
# Three further modules are NOT bridged, with the specific reason for each
# stated in Section 3 rather than left unexplained.
#
# Developed with Claude as AI co-developer.
# =============================================================================

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import torch

from evolution_one_ageing import DifferentiableLongevityEngine, LongevityEngineConfig

Tensor = torch.Tensor


# =============================================================================
# Section 1 -- CellPopulation bridge (the well-grounded one).
#
# cell_population_one.py's CellPopulationState carries real 3-D cell
# positions (`.position`, (n_max,3)) and real per-cell chronological age
# (`.age`, (n_max,)); its optional OrganelleLayer tracks nine literal
# hallmarks-of-ageing biomarkers (mito_atp, mito_psi, mito_ros, nuc_damage,
# nuc_repair_capacity, lyso_capacity, lyso_autophagy_flux, er_unfolded,
# er_upr_stress), and its optional PhenotypeLayer tracks further per-cell
# expression channels. This is a real, non-fabricated source of exactly the
# kind of data DifferentiableLongevityEngine's omic_markers/spatial_integrity
# inputs are for.
#
# Design choice, stated rather than hidden: each LIVE CELL is treated as one
# "patient" (batch_size = n_alive, varying call to call as cells divide and
# die), giving a genuine per-cell biological-age prediction directly
# supervisable against real chronological age (`.age`) -- the standard
# "epigenetic clock" framing. The alternative design -- the whole population
# as ONE spatial tissue graph, with cells as graph nodes -- would give a
# genuinely spatially-coupled bio-age signal, but cell_population_one.py's
# OrganelleState tracks only whole-cell scalar aggregates, not sub-cellular
# spatial data, so there is no real geometry to build a meaningful spatial
# Laplacian from at that level; that richer design is not attempted here,
# and is left as a documented, not implemented, extension.
# =============================================================================


class CellPopulationAgeingBridge:
    """One CellPopulation <-> one DifferentiableLongevityEngine, rebuilt
    only when the live-cell count (or the biomarker feature width, e.g. if
    a PhenotypeLayer is attached/detached) changes by more than
    `rebuild_tolerance`, not on every call -- PolyharmonicTissueDecayCore's
    tissue graph is fixed-batch-size by construction (see its own
    docstring), so rebuilding it every step, when the population is
    barely changing, would dominate cost for no benefit.
    """

    def __init__(
        self,
        population,
        cfg: Optional[LongevityEngineConfig] = None,
        spatial_dim: int = 32,
        rebuild_tolerance: float = 0.1,
    ) -> None:
        self.population = population
        self.base_cfg = cfg or LongevityEngineConfig(spatial_dim=spatial_dim, graph_source="chain")
        self.rebuild_tolerance = rebuild_tolerance
        self.engine: Optional[DifferentiableLongevityEngine] = None
        self._last_n_alive: Optional[int] = None
        self._last_in_features: Optional[int] = None

    def _needs_rebuild(self, n_alive: int, in_features: int) -> bool:
        if self.engine is None or self._last_n_alive is None:
            return True
        if in_features != self._last_in_features:
            return True
        tol = max(1, int(self.rebuild_tolerance * self._last_n_alive))
        return abs(n_alive - self._last_n_alive) > tol

    def _rebuild(self, n_alive: int, in_features: int) -> DifferentiableLongevityEngine:
        cfg = replace(self.base_cfg, batch_size=max(n_alive, 1), in_features=in_features)
        self.engine = DifferentiableLongevityEngine(cfg)
        self._last_n_alive = n_alive
        self._last_in_features = in_features
        return self.engine

    def population_to_ageing_inputs(self) -> Optional[Dict[str, Tensor]]:
        """Pulls live-cell organelle (+ phenotype, if attached) state into
        omic_markers, and a generic internal-structural-integrity proxy
        into spatial_integrity on the engine's toy internal chain graph.

        Honesty note, stated once here rather than re-litigated per call:
        OrganelleState's fields are whole-cell scalar aggregates, not
        sub-cellular spatial data, so there is no real per-cell geometry
        to build a genuine spatial tissue graph from at this level (see
        the module-level docstring's design-choice note). The chain-graph
        tissue-decay pathway is used here as a generic internal-integrity
        proxy, seeded from `mito_atp` (a real, single, defensible
        aggregate proxy for a cell's overall energetic/structural state)
        broadcast across the toy chain's nodes -- not a model of real
        sub-cellular geometry, and not presented as one.
        """
        if getattr(self.population, "organelle", None) is None:
            raise RuntimeError(
                "CellPopulationAgeingBridge needs an OrganelleLayer attached "
                "(population.attach_organelle_layer(...)) -- its state is "
                "this bridge's real biomarker source."
            )
        alive = self.population.state.alive
        n_alive = int(alive.sum().item())
        if n_alive == 0:
            return None

        o = self.population.organelle.state
        organelle_fields = [
            o.mito_atp, o.mito_psi, o.mito_ros, o.nuc_damage, o.nuc_repair_capacity,
            o.lyso_capacity, o.lyso_autophagy_flux, o.er_unfolded, o.er_upr_stress,
        ]
        organelle_block = torch.stack([f[alive] for f in organelle_fields], dim=-1)  # (n_alive, 9)

        if getattr(self.population, "phenotype", None) is not None:
            pheno_block = self.population.phenotype.state.expression[alive]  # (n_alive, n_pheno)
            omic_markers = torch.cat([organelle_block, pheno_block], dim=-1)
        else:
            omic_markers = organelle_block

        in_features = omic_markers.shape[-1]
        if self._needs_rebuild(n_alive, in_features):
            engine = self._rebuild(n_alive, in_features)
        else:
            engine = self.engine

        atp = o.mito_atp[alive]  # (n_alive,)
        spatial_integrity = atp.unsqueeze(-1).expand(-1, engine.cfg.spatial_dim).clone()

        chronological_age = self.population.state.age[alive].float()

        return {
            "engine": engine,
            "omic_markers": omic_markers,
            "spatial_integrity": spatial_integrity,
            "chronological_age": chronological_age,
            "alive_mask": alive,
        }

    def compute_bio_age(self, time_steps: int = 5, dt: float = 0.01) -> Optional[Dict[str, Tensor]]:
        """Runs the ageing engine on the population's current live-cell
        state. `bio_age_projected` in the returned dict is the per-cell
        biological-age prediction; `chronological_age` is real ground
        truth from the same call, ready for direct supervision
        (e.g. MSE(bio_age_projected, chronological_age) as a training
        loss -- the standard epigenetic-clock framing: fit bio-age to
        predict chronological age from biomarkers, then examine systematic
        deviations as the signal of actual interest).
        """
        inputs = self.population_to_ageing_inputs()
        if inputs is None:
            return None
        engine = inputs["engine"]
        out = engine(inputs["omic_markers"], inputs["spatial_integrity"], time_steps=time_steps, dt=dt)
        out["chronological_age"] = inputs["chronological_age"]
        out["alive_mask"] = inputs["alive_mask"]
        return out

    def ageing_to_population_feedback(self, bio_age_fitness_weight: float = 0.01) -> Tensor:
        """Accelerated biological age (bio_age_projected exceeding real
        chronological age) becomes an additive fitness penalty -- mirrors
        this codebase's own established CellPopulationCahnHilliardBridge
        pattern (one sub-system's computed signal modulating another's
        dynamics). Returns a (n_max,) penalty tensor for the caller to
        fold into CellPopulation's own fitness computation; this bridge
        does not reach into CellPopulation's internals to apply it
        directly, since exactly where per-step fitness terms combine is
        CellPopulation.step's own concern, not this bridge's.
        """
        out = self.compute_bio_age()
        n_max = self.population.state.age.shape[0]
        if out is None:
            return torch.zeros(n_max, device=self.population.state.age.device)
        alive = out["alive_mask"]
        age_gap = (out["bio_age_projected"].squeeze(-1) - out["chronological_age"]).clamp_min(0.0)
        penalty = torch.zeros(n_max, device=age_gap.device, dtype=age_gap.dtype)
        penalty[alive] = -bio_age_fitness_weight * age_gap
        return penalty


# =============================================================================
# Section 2 -- Cahn-Hilliard phase field bridge (lighter, single-channel).
#
# structural_cahn_hilliard_3d_v3.py solves a genuine 3-D phase-field PDE;
# cell_population_one.py's own CellPopulationCahnHilliardBridge already
# establishes the pattern for sampling that field AT cell positions. Senescent
# cell clustering is a real, published phenomenon sometimes modeled via
# phase-separation dynamics, making "local phase value" a defensible extra
# biomarker channel -- not a claim that this specific CH3D instance models
# senescence, only that plugging its field value in as one more channel is a
# sensible, well-motivated thing to do with it.
# =============================================================================


def append_ch3d_channel(omic_markers: Tensor, ch3d_field_at_cells: Tensor) -> Tensor:
    """`ch3d_field_at_cells`: (n_alive,) or (n_alive,1), the CH3D phase
    field already sampled at each live cell's position (via
    CellPopulationCahnHilliardBridge's own established field-sampling
    method, not reimplemented here -- this function only concatenates the
    result on as one more biomarker channel).
    """
    if ch3d_field_at_cells.dim() == 1:
        ch3d_field_at_cells = ch3d_field_at_cells.unsqueeze(-1)
    return torch.cat([omic_markers, ch3d_field_at_cells], dim=-1)


# =============================================================================
# Section 3 -- What is NOT bridged here, and specifically why.
#
# structural_langevin_evo_v5.py: a real, legitimate, SMALL possible
# connection exists -- its thermostat's own noise scale could calibrate
# RejuvenationResetDynamics's noise_bound_G0 / drift_bound_B0 (currently
# free model parameters with no external calibration source at all) -- but
# this needs CSOCThermostat's exact noise-scale attribute name verified
# against its source before writing code that reads it, which was not done
# in this pass; noted as a real, small, well-scoped next step, not
# implemented speculatively.
#
# bv_full_theory_one.py: its BV/BRST machinery checks GAUGE-SYMMETRY
# consistency of a postulated gene-network master equation -- a materially
# different kind of question from ageing/longevity prediction, with no
# natural "ageing BV action" this codebase defines. Forcing a connection
# without one would repeat exactly the kind of unmotivated vocabulary-bridge
# this project has declined to fabricate elsewhere (the CSL and non-p.c.f.
# fractal reviews earlier in this project).
#
# structural_gno_evolution_bv_standalone.py: a large (3255-line), complex
# training-and-rollout infrastructure module. A real connection may well
# exist (its CellPopulationRollout/CellPopulationTrainingBridge classes
# suggest one), but verifying it correctly, at the level of rigor the rest
# of this project's software has been held to, would require reading and
# hand-tracing a comparable amount of new source as this entire bridges
# file already required for cell_population_one.py alone -- not attempted
# in this pass rather than rushed.
# =============================================================================
