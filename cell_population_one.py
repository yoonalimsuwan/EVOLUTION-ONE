# =============================================================================
# CELL POPULATION ONE — Agent-Based Cell Population Model
# Component #N of the EVOLUTION ONE Cluster — ONE Ecosystem
# =============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026
#
# AI Co-Developers (architecture, differentiable population dynamics,
# cross-cluster bridge design):
#   - Claude   (Anthropic)  — CellPopulationState batched-tensor design,
#                             differentiable soft birth/death kernel,
#                             CellPopulationCahnHilliardBridge (two-way
#                             coupling with structural_cahn_hilliard_3d),
#                             genotype sampling from EvolutionONEEngine's
#                             mutation matrix, CellPopulationMixin
#                             (attach_cell_population, matching the
#                             LangevinBridgeMixin convention), full
#                             [PASS]/[FAIL] verification suite,
#                             OrganelleLayer v1.1.0 (mitochondria / nucleus
#                             / lysosome / endoplasmic reticulum
#                             sub-cellular ODE layer, two-pathway coupling
#                             into PhenotypeLayer's gene_drive and
#                             CellPopulation's division/death logit,
#                             ATP-gated division, checkpointing support)
#
# =============================================================================
# CHANGELOG
# =============================================================================
# v1.2.0 (2026) — Effective population size + capacity-selection fairness
#   step() now returns "n_effective": an exact, per-step Wright variance-
#   effective population size (Ne = N / (1 + CV_k^2)) computed from
#   realized per-cell death/division outcomes over the cells alive at the
#   start of the step. This is a diagnostic only — the model's demographic
#   stochasticity was already genuine (independent per-cell Bernoulli
#   sampling), this just exposes how drift-dominated a given step/rollout
#   was. Also fixed a hidden selection-order bias in capacity-limited
#   division: when more cells win their division draw than there are free
#   slots, which winners actually divide is now chosen by an unbiased
#   random permutation instead of ascending slot-index order (the old
#   behaviour systematically favoured low-slot-index cells whenever
#   capacity bound, regardless of fitness).
#
# v1.1.0 (2026) — Organelle Layer
#   Added OrganelleConfig / OrganelleState / OrganelleLayer: a per-cell
#   sub-cellular layer beneath PhenotypeLayer, modelling mitochondria
#   (ATP / membrane potential / ROS), nucleus (DNA damage / repair
#   capacity, with a p53-like checkpoint), lysosome (autophagy capacity
#   and flux), and endoplasmic reticulum (unfolded-protein load / UPR
#   stress), each coupled to the others via the crosstalk pathways
#   documented in Section 3.5 below. Wired into CellPopulation.step() via
#   attach_organelle_layer(), contributing both a fast direct fitness
#   pathway (bypassing PhenotypeLayer's ODE lag) and a slow pathway into
#   PhenotypeLayer's gene-regulatory drive (via
#   organelle_drive_to_phenotype(), injected per-step without mutating
#   PhenotypeLayer's persistent _gene_drive buffer). With no
#   OrganelleLayer attached, behaviour is unchanged from v1.0.0.
#
# =============================================================================
# WHAT THIS FILE ADDS TO THE ECOSYSTEM
# =============================================================================
#
# Every existing EVOLUTION ONE / REAL FOLD ONE module operates at one of:
#   • atomic scale      (coords of one protein's atoms — REAL FOLD ONE)
#   • continuum scale    (phase field u(x,y,z) on a voxel grid — CH3D)
#   • population scale   (scalar μ / Rt summarising an entire cohort/tumor)
#   • genomic scale       (per-sample mutation calls, MAF/VCF rows)
#
# None of those resolve an actual population of individual cells, each with
# its own identity. This module adds exactly that missing layer: a batched,
# fully differentiable agent population where each agent ("cell") carries
#
#   • a genotype  (a clone index into the same mutation matrix that
#     EvolutionONEEngine.loader.build_mutation_matrix already produces —
#     i.e. "which tumor-sample genotype this cell's lineage descends from"),
#   • a continuous spatial position on the same (Nx, Ny, Nz) voxel grid
#     used by StructuralCahnHilliard3D,
#   • alive/dead status and age,
#
# and undergoes stochastic division and death every step, with rates
# modulated by (a) the local phase-field/sigma value at the cell's position
# (CH3D feedback) and (b) a per-genotype fitness score (driven by mutation
# load μ and, optionally, REAL FOLD ONE HT's ΔΔG per gene).
#
# Design choice: batched tensors, not per-cell Python objects
# --------------------------------------------------------------
# A population of N cells is stored as flat (N_max,)-shaped tensors
# (genotype ids, positions, alive mask, age) rather than a list of N
# Python "Cell" objects. This is deliberate:
#   • It keeps the whole population update vectorised and GPU-resident —
#     a Python for-loop over cells would be both un-differentiable in any
#     useful sense and orders of magnitude slower at population sizes
#     in the thousands-to-millions range this is meant to scale to.
#   • Division/death become differentiable *soft* operations on
#     probabilities (a cell's survival/division probability is a smooth
#     function of its local environment and genotype fitness), so
#     gradients can flow from population-level outcomes (e.g. final clone
#     frequencies) back into the fitness model and into the CH3D fields
#     driving the local environment — matching the "fully differentiable
#     end-to-end" convention used everywhere else in this ecosystem.
#   • The *number of cells alive* is allowed to fluctuate (this is the
#     entire point of a division/death model), which a fixed-size tensor
#     handles via a capacity (N_max) + alive-mask pattern: new cells are
#     written into currently-dead slots rather than the tensor being
#     resized every step (resizing every step would break the
#     fixed-shape assumption autograd graphs in a training loop rely on).
#
# Two-way coupling with the rest of the ecosystem
# ------------------------------------------------
#   CH3D u/sigma  →  per-cell local environment  →  division/death rates
#   cell positions (alive only)  →  local density source term  →  CH3D u
#   EvolutionONEEngine mutation matrix  →  per-clone genotype vectors
#   per-clone mean mutation count  →  per-clone fitness (more mutations in
#       driver-like genes ⇒ different division/death bias, configurable
#       sign so this can model either a fitness *advantage* (default,
#       cancer-clonal-growth framing) or *disadvantage* (e.g. modelling a
#       deleterious mutation load) via cfg.fitness_sign)
#   REAL FOLD ONE HT ΔΔG per gene (optional)  →  refines per-clone fitness
#       beyond the binary mutation-matrix signal alone
#
# This module does NOT attempt to be a full tissue/organ simulator (no
# extracellular matrix mechanics, no explicit cell-cell adhesion forces,
# no immune-cell interactions). It is the minimum agent-based layer needed
# to ask genuinely cell-population-level questions — e.g. "does this clone
# outcompete that one under this phase-field environment", "how does
# spatial structure affect clonal sweep dynamics" — that no other module
# in this ecosystem currently answers.
# =============================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _soft_clamp(x: torch.Tensor, lo: float, hi: float,
                 beta: float = 20.0) -> torch.Tensor:
    """
    Differentiable two-sided clamp into [lo, hi]: near-identity in the
    interior, smooth saturation only near the lo/hi boundaries, nonzero
    gradient everywhere (unlike torch.Tensor.clamp, which has exactly zero
    gradient at every point where the input is outside [lo, hi]).

    Implemented as a softplus floor followed by a softplus ceiling:
        y = lo + softplus(x - lo, beta)   # soft floor at lo
        y = hi - softplus(hi - y, beta)   # soft ceiling at hi

    This is the same "soft-clamped, not hard .clamp()" convention already
    used for death_rate_floor / division_rate_ceiling in
    CellPopulationConfig (see _rate_from_logit), generalised to a plain
    two-sided clamp for use on field tensors (e.g. the CH3D phase field u)
    rather than a (floor, ceiling)-bounded *rate*.
    """
    y = lo + F.softplus(x - lo, beta=beta)
    y = hi - F.softplus(hi - y, beta=beta)
    return y

# =============================================================================
# Ecosystem imports — single source of truth, graceful fallback for
# standalone use (mirrors the pattern used throughout EVOLUTION ONE).
# =============================================================================
try:
    from one_core_evolution import (
        SemanticStateContraction,
        CSOCBase,
        CheckpointManager,
        get_device,
        EVOLUTION_VERSION,
    )
    _HAS_CORE_EVOLUTION = True
except ImportError:
    _HAS_CORE_EVOLUTION = False
    EVOLUTION_VERSION = "unknown"

    def get_device(preferred: str = "cuda") -> torch.device:  # type: ignore[misc]
        """Standalone fallback mirroring one_core_evolution.get_device."""
        p = preferred.lower()
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    class SemanticStateContraction(nn.Module):  # type: ignore[no-redef]
        """Standalone fallback SSC EMA filter — see one_core_evolution for the canonical version."""

        def __init__(self, epsilon_fp: float = 0.0028, sigma_target: float = 1.0) -> None:
            super().__init__()
            self.eps = epsilon_fp
            self.target = sigma_target
            self.register_buffer("prev_sigma", torch.tensor(0.0))
            self.register_buffer("_initialized", torch.tensor(False))

        def reset(self) -> None:
            self.prev_sigma.zero_()
            self._initialized.fill_(False)

        def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
            if self.prev_sigma.device != raw_sigma.device:
                self.prev_sigma = self.prev_sigma.to(raw_sigma.device)
                self._initialized = self._initialized.to(raw_sigma.device)
            if not self._initialized.item():
                self.prev_sigma.data = raw_sigma.detach()
                self._initialized.fill_(True)
                return raw_sigma
            new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
            self.prev_sigma.data = new_sigma.detach()
            return new_sigma

    class CheckpointManager:  # type: ignore[no-redef]
        """Standalone fallback checkpoint manager — see one_core_evolution for the canonical version."""

        @staticmethod
        def save(filepath: str, data: Dict[str, Any]) -> None:
            import os
            import pickle
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        @staticmethod
        def load(filepath: str) -> Optional[Dict[str, Any]]:
            import os
            import pickle
            if not os.path.exists(filepath):
                return None
            with open(filepath, "rb") as f:
                return pickle.load(f)

    import warnings
    warnings.warn(
        "one_core_evolution not found — cell_population_one.py running in "
        "standalone mode with local fallbacks."
    )

CELL_POPULATION_VERSION: str = "1.2.0"

__all__ = [
    "CELL_POPULATION_VERSION",
    "CellPopulationConfig",
    "CellPopulationState",
    "CellPopulation",
    "PhenotypeConfig",
    "PhenotypeState",
    "PhenotypeLayer",
    "gene_drive_from_bv_network",
    "fitness_from_mutation_matrix",
    "CellPopulationCahnHilliardBridge",
    "CellPopulationMixin",
    # Organelle layer (v1.1.0)
    "OrganelleConfig",
    "OrganelleState",
    "OrganelleLayer",
]


# =============================================================================
# 1.  Configuration
# =============================================================================

@dataclass
class CellPopulationConfig:
    """
    Configuration for a batched, differentiable cell population.

    Population sizing
    ------------------
    n_max          : fixed tensor capacity — the maximum number of cells
                     that can ever be alive simultaneously. Division
                     beyond this capacity is silently capped (a population
                     at carrying capacity), matching the standard
                     logistic-growth assumption rather than crashing.
    n_init         : number of cells alive at t=0.
    n_genotypes    : number of distinct clone genotypes available to draw
                     from (e.g. number of rows in EvolutionONEEngine's
                     mutation matrix / number of Tumor_Sample_Barcode
                     values used as founder clones).

    Spatial substrate (shared with StructuralCahnHilliard3D)
    ----------------------------------------------------------
    grid_shape     : (Nx, Ny, Nz) — MUST match the CH3D solver's u/sigma
                     field shape this population is coupled to.
    box_size       : physical box side length (same units as CH3D's dx
                     grid; isotropic cube assumed unless a 3-tuple is given).
    motility       : per-step random-walk standard deviation for cell
                     position updates (same length units as box_size).
                     Differentiable Gaussian diffusion, not an explicit
                     force model.

    Division / death dynamics
    --------------------------
    base_division_rate : division probability per step at sigma == 1.0
                          (CSOC-neutral) and zero genotype fitness bias —
                          the population's intrinsic baseline.
    base_death_rate     : death probability per step under the same
                          neutral conditions.
    sigma_division_gain : how strongly local CH3D sigma modulates division
                          probability. sigma > 1 (above target) increases
                          division probability by up to this much;
                          sigma < 1 decreases it symmetrically. Implemented
                          as a smooth sigmoid gate, never a hard cutoff.
    fitness_division_gain : how strongly per-clone fitness (see
                          CellPopulation.set_genotype_fitness) modulates
                          division probability, on top of the sigma term.
    fitness_sign        : +1.0 (default) treats higher fitness score as a
                          division *advantage* (cancer-clonal-growth
                          framing: more driver-like mutation burden ⇒
                          faster-growing clone). Set to -1.0 to instead
                          treat higher fitness score as a *disadvantage*
                          (e.g. modelling a deleterious mutation load that
                          slows or kills the carrying cell).
    death_rate_floor, division_rate_ceiling : numerical floors/ceilings
                          (soft-clamped, not hard .clamp()) keeping
                          per-step probabilities inside (0, 1) regardless
                          of how extreme the gain terms push them.

    SSC smoothing
    -------------
    epsilon_fp     : SSC EMA blending factor for the population-level
                     mutation-load signal exposed to EvolutionONEEngine /
                     EpiForecastEngine via existing bridges (see
                     CellPopulation.population_mutation_load).

    Device
    ------
    device, dtype  : compute backend.
    """

    n_max:       int = 4096
    n_init:      int = 256
    n_genotypes: int = 8

    grid_shape: Tuple[int, int, int] = (32, 32, 32)
    box_size:   float = 50.0
    motility:   float = 0.5

    base_division_rate: float = 0.05
    base_death_rate:    float = 0.04
    sigma_division_gain:   float = 0.15
    fitness_division_gain: float = 0.10
    fitness_sign: float = 1.0
    death_rate_floor:      float = 1e-4
    division_rate_ceiling: float = 0.95

    epsilon_fp: float = 0.0028

    device: str = "cpu"
    dtype:  torch.dtype = torch.float32

    def __post_init__(self) -> None:
        assert self.n_max >= 1
        assert 1 <= self.n_init <= self.n_max, (
            f"n_init ({self.n_init}) must be in [1, n_max={self.n_max}]."
        )
        assert self.n_genotypes >= 1
        assert all(d >= 1 for d in self.grid_shape), \
            f"grid_shape must have all dims >= 1; got {self.grid_shape!r}."
        assert self.box_size > 0.0
        assert self.motility >= 0.0
        assert 0.0 <= self.base_division_rate <= 1.0
        assert 0.0 <= self.base_death_rate <= 1.0
        assert self.sigma_division_gain >= 0.0
        assert self.fitness_division_gain >= 0.0
        assert self.fitness_sign in (1.0, -1.0), \
            f"fitness_sign must be +1.0 or -1.0; got {self.fitness_sign!r}."
        assert 0.0 < self.death_rate_floor < 1.0
        assert 0.0 < self.division_rate_ceiling < 1.0
        assert self.death_rate_floor < self.division_rate_ceiling, (
            f"death_rate_floor ({self.death_rate_floor}) must be < "
            f"division_rate_ceiling ({self.division_rate_ceiling})."
        )
        # base_division_rate / base_death_rate must lie strictly inside
        # (death_rate_floor, division_rate_ceiling) — not just [0, 1] — 
        # because _rate_from_logit's logit_shift derivation
        # (logit((base_rate - floor) / (ceiling - floor))) requires that
        # ratio to be a valid probability in (0, 1). A base_rate outside
        # this range would silently saturate at the wrong asymptote
        # instead of reproducing base_rate at logit=0 as documented.
        assert self.death_rate_floor < self.base_division_rate < self.division_rate_ceiling, (
            f"base_division_rate ({self.base_division_rate}) must lie strictly "
            f"between death_rate_floor ({self.death_rate_floor}) and "
            f"division_rate_ceiling ({self.division_rate_ceiling})."
        )
        assert self.death_rate_floor < self.base_death_rate < self.division_rate_ceiling, (
            f"base_death_rate ({self.base_death_rate}) must lie strictly "
            f"between death_rate_floor ({self.death_rate_floor}) and "
            f"division_rate_ceiling ({self.division_rate_ceiling})."
        )
        assert 0.0 < self.epsilon_fp < 1.0


# =============================================================================
# 2.  Cell Population State — batched tensors, not per-cell objects
# =============================================================================

@dataclass
class CellPopulationState:
    """
    Flat, fixed-capacity batched state for a cell population.

    All tensors are shape ``(n_max,)`` (or ``(n_max, 3)`` for position) —
    a "structure of arrays" layout, not "array of structs". Dead slots
    keep stale data in ``position``/``genotype``/``age`` (irrelevant once
    masked out by ``alive``) rather than being zeroed every step, which
    would cost an extra full-tensor write for no behavioural benefit.

    Args:
        position : (n_max, 3) continuous spatial position, same units as
                   ``CellPopulationConfig.box_size``.
        genotype : (n_max,) long tensor — clone/genotype index into
                   [0, n_genotypes).
        age       : (n_max,) float tensor — steps survived since birth.
        alive     : (n_max,) bool tensor — True for currently-live slots.
    """

    position: torch.Tensor
    genotype: torch.Tensor
    age:      torch.Tensor
    alive:    torch.Tensor

    def n_alive(self) -> int:
        """Current live population size (a Python int, for logging/control flow)."""
        return int(self.alive.sum().item())

    def to(self, device: torch.device) -> "CellPopulationState":
        return CellPopulationState(
            position=self.position.to(device),
            genotype=self.genotype.to(device),
            age=self.age.to(device),
            alive=self.alive.to(device),
        )


# =============================================================================
# 3.  Phenotype Layer — dynamic, multi-channel expression state
# =============================================================================
#
# Genotype (CellPopulationState.genotype) is a fixed, heritable label — it
# never changes once a cell is born, and every cell sharing a genotype was
# (before this layer existed) assumed to have identical fitness. That's a
# reasonable population-genetics-style simplification, but it collapses
# exactly the layer biology calls "phenotype": the actually-expressed,
# time-varying, environment-responsive state that genotype only partially
# determines.
#
# This section adds that layer back in, as multiple independent expression
# channels per cell (proliferation / stress-response / differentiation by
# default, but configurable), each following a simple gene-regulatory ODE:
#
#     dE_c/dt = -decay_c * E_c + tanh(gene_drive_c + sigma_drive_c)
#
# driven by two upstream signals:
#   • gene_drive  : derived from GeneNetworkBV's phi (gene-expression-like
#                   scalar field) via a configurable channel-mixing matrix
#                   — see gene_drive_from_bv_network() below.
#   • sigma_drive : the same local CH3D sigma field CellPopulation.step()
#                   already samples per cell, so phenotype expression
#                   responds to the same structural-regime environment the
#                   rest of the population dynamics does.
#
# The resulting expression state then feeds BACK into CellPopulation's
# division/death logits via fitness_contribution() — i.e. phenotype, not
# genotype directly, is what proximately drives a cell's fate; genotype
# only acts *through* the phenotype it tends to produce (via gene_drive)
# plus its separate, simpler genotype_fitness term from Section 4. Both
# pathways are additive in the logit (see CellPopulation.step()), so this
# is a strict superset of the genotype-only model from v1.0.0 — with no
# PhenotypeLayer attached, behaviour is identical to before this section
# existed.
# =============================================================================

@dataclass
class PhenotypeConfig:
    """
    Configuration for a per-cell, multi-channel dynamic phenotype layer.

    Channels
    --------
    channel_names : names of the independent expression channels tracked
                    per cell. Order matters — it fixes the column order of
                    every (n_max, n_channels) tensor this layer produces,
                    and of ``fitness_weights`` / ``decay_rate`` /
                    ``sigma_gain`` below. Default: proliferation,
                    stress-response, differentiation — three biologically
                    distinct axes that commonly drive opposite or
                    orthogonal effects on division/death (e.g. high
                    stress-response should plausibly *suppress* division
                    even when proliferation signal is high).
    n_max         : must match the attached CellPopulationConfig.n_max —
                    enforced automatically by
                    CellPopulation.attach_phenotype_layer().

    ODE dynamics
    ------------
    decay_rate    : (n_channels,) per-channel decay rate λ_c in
                    dE/dt = -λ_c·E + tanh(drive_c). Higher = faster return
                    to baseline (0) absent any drive. Must be > 0 for a
                    well-posed (bounded) ODE.
    dt            : integration time step per CellPopulation.step() call —
                    one explicit-Euler update per population step, so
                    phenotype dynamics are synchronised 1:1 with
                    division/death/motility rather than sub-stepped.
    sigma_gain    : (n_channels,) how strongly local CH3D sigma deviation
                    (sigma - 1.0) drives each channel. Sign matters: a
                    positive gain means above-target sigma pushes that
                    channel's expression up; negative means it pushes
                    expression down.
    gene_gain     : scalar multiplier applied to the gene-regulatory drive
                    signal (see gene_drive_from_bv_network) before it
                    enters the same tanh as sigma_drive.

    Fitness coupling
    -----------------
    fitness_weights : (n_channels,) how strongly each channel's expression
                    contributes to the division/death logit fed back into
                    CellPopulation.step() (see PhenotypeLayer.
                    fitness_contribution). Positive weight on
                    "proliferation" and negative on "stress_response" is
                    the natural default sign convention, but this is left
                    fully configurable rather than hard-coded, since which
                    channel promotes vs. suppresses division is a
                    modelling choice, not a universal constant.

    Initial / inherited state
    --------------------------
    init_expression : scalar baseline expression every live cell starts
                    at (default 0.0 — "off" until driven).
    inherit_noise   : standard deviation of Gaussian noise added to a
                    daughter cell's inherited expression state on
                    division (0.0 = exact inheritance, matching how
                    genotype is inherited; > 0 models stochastic
                    epigenetic drift at division).

    Device
    ------
    device, dtype.
    """

    channel_names: Tuple[str, ...] = ("proliferation", "stress_response", "differentiation")
    n_max: int = 4096

    decay_rate: Tuple[float, ...] = (0.1, 0.2, 0.05)
    dt: float = 1.0
    sigma_gain: Tuple[float, ...] = (0.5, -0.3, 0.1)
    gene_gain: float = 1.0

    fitness_weights: Tuple[float, ...] = (0.6, -0.5, 0.0)

    init_expression: float = 0.0
    inherit_noise: float = 0.02

    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        n_ch = len(self.channel_names)
        assert n_ch >= 1, "channel_names must be non-empty."
        for name, vec in (
            ("decay_rate", self.decay_rate),
            ("sigma_gain", self.sigma_gain),
            ("fitness_weights", self.fitness_weights),
        ):
            assert len(vec) == n_ch, (
                f"{name} must have one entry per channel "
                f"(len={len(vec)} != n_channels={n_ch}; channel_names={self.channel_names})."
            )
        assert all(d > 0.0 for d in self.decay_rate), "All decay_rate entries must be > 0."
        assert self.dt > 0.0
        assert self.n_max >= 1
        assert self.inherit_noise >= 0.0

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


@dataclass
class PhenotypeState:
    """
    Flat (n_max, n_channels) expression state — same "structure of
    arrays" convention as CellPopulationState, for the same reasons
    (vectorised, GPU-resident, no per-cell Python objects).

    Args:
        expression : (n_max, n_channels) current expression level per
                     cell per channel. Unconstrained sign (a tanh-driven
                     ODE naturally stays within a bounded range around
                     zero, but is not hard-clamped).
    """

    expression: torch.Tensor

    def to(self, device: torch.device) -> "PhenotypeState":
        return PhenotypeState(expression=self.expression.to(device))


class PhenotypeLayer(nn.Module):
    """
    Per-cell, multi-channel dynamic phenotype state, evolved by a simple
    gene-regulatory ODE and coupled bidirectionally with the rest of the
    ecosystem:

        GeneNetworkBV.phi  →  gene_drive_from_bv_network()  ──┐
                                                                ├─► tanh ─► dE/dt
        local CH3D sigma   →  sigma_gain * (sigma - 1)     ────┘

        PhenotypeLayer.expression  →  fitness_contribution()  →  CellPopulation
                                                                    division/death logit

    This is intentionally a *minimal* gene-regulatory model — a per-channel
    linear-decay-plus-saturating-drive ODE, not a full reaction-network
    simulator. It's the smallest dynamic state that actually earns the
    name "phenotype" (responds to environment over time, distinct from
    the static genotype that partially drives it) without requiring a
    detailed mechanistic gene network most callers won't have data for.

    Args:
        cfg    : PhenotypeConfig instance.
        device : compute device (normally inherited from the attaching
                 CellPopulation, not set independently).
    """

    def __init__(self, cfg: PhenotypeConfig, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._device = device or get_device(cfg.device)

        self.register_buffer("decay_rate", torch.tensor(cfg.decay_rate, dtype=cfg.dtype))
        self.register_buffer("sigma_gain", torch.tensor(cfg.sigma_gain, dtype=cfg.dtype))
        self.register_buffer("fitness_weights", torch.tensor(cfg.fitness_weights, dtype=cfg.dtype))

        self.state = PhenotypeState(
            expression=torch.full(
                (cfg.n_max, cfg.n_channels), cfg.init_expression,
                device=self._device, dtype=cfg.dtype,
            )
        )

        # External gene-regulatory drive, settable per step via
        # set_gene_drive() (e.g. from gene_drive_from_bv_network()).
        # Shape (n_max, n_channels); defaults to zero (no gene-network
        # input — expression then responds to sigma_drive alone).
        self.register_buffer(
            "_gene_drive",
            torch.zeros(cfg.n_max, cfg.n_channels, device=self._device, dtype=cfg.dtype),
        )

    # ------------------------------------------------------------------
    # Gene-regulatory drive input
    # ------------------------------------------------------------------

    def set_gene_drive(self, gene_drive: torch.Tensor) -> None:
        """
        Set the per-cell, per-channel gene-regulatory drive signal used by
        the next ``update()`` call.

        Args:
            gene_drive : (n_max, n_channels) tensor — typically the output
                         of ``gene_drive_from_bv_network()`` broadcast
                         across cells of a given genotype, but can be any
                         tensor of the right shape (e.g. zero to disable
                         gene-network influence entirely and drive
                         expression from sigma alone).
        """
        if gene_drive.shape != (self.cfg.n_max, self.cfg.n_channels):
            raise ValueError(
                f"gene_drive must be shape ({self.cfg.n_max}, {self.cfg.n_channels}); "
                f"got {tuple(gene_drive.shape)}."
            )
        self._gene_drive = gene_drive.to(device=self._device, dtype=self.cfg.dtype)

    # ------------------------------------------------------------------
    # ODE step
    # ------------------------------------------------------------------

    def update(
        self,
        local_sigma: torch.Tensor,
        alive: torch.Tensor,
        extra_drive: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Advance the expression ODE by one explicit-Euler step:

            E_{t+1} = E_t + dt * ( -decay_rate * E_t
                                    + tanh(gene_gain * gene_drive
                                           + extra_drive
                                           + sigma_gain * (sigma - 1)) )

        Only applied to currently-alive cells — dead slots' expression is
        left untouched here (it gets explicitly reset on death via
        ``reset_slots``, not silently frozen mid-decay).

        Args:
            local_sigma : (n_max,) local CH3D sigma value sampled at each
                          cell's position this step (same tensor
                          CellPopulation.step() already computes).
            alive       : (n_max,) bool mask of currently-alive cells.
            extra_drive : optional (n_max, n_channels) additional drive
                          term, added in BEFORE the shared tanh() but
                          OUTSIDE ``cfg.gene_gain`` and not stored into
                          ``self._gene_drive`` — i.e. a one-step,
                          non-persistent addition. This is the hook
                          ``CellPopulation.step()`` uses to fold
                          ``organelle_drive_to_phenotype()`` in every
                          step without mutating the caller-owned
                          ``_gene_drive`` buffer (which would otherwise
                          accumulate the organelle term step after step,
                          since ``_gene_drive`` is a persistent buffer
                          set rarely, not necessarily every step, via
                          ``set_gene_drive()``). Defaults to zero (no-op)
                          when omitted, so this is a strict superset of
                          the pre-organelle-layer ODE.
        Returns:
            (n_max, n_channels) updated expression state (also stored as
            ``self.state.expression``).
        """
        cfg = self.cfg
        sigma_dev = (local_sigma - 1.0).unsqueeze(-1)               # (n_max, 1)
        drive = cfg.gene_gain * self._gene_drive + self.sigma_gain.unsqueeze(0) * sigma_dev
        if extra_drive is not None:
            drive = drive + extra_drive
        dE = -self.decay_rate.unsqueeze(0) * self.state.expression + torch.tanh(drive)

        new_expression = self.state.expression + cfg.dt * dE
        alive_mask = alive.unsqueeze(-1)
        self.state.expression = torch.where(alive_mask, new_expression, self.state.expression)
        return self.state.expression

    # ------------------------------------------------------------------
    # Fitness feedback (phenotype → CellPopulation division/death logit)
    # ------------------------------------------------------------------

    def fitness_contribution(self) -> torch.Tensor:
        """
        (n_max,) scalar contribution to CellPopulation's division logit
        (and, with a sign flip, the death logit) — a weighted sum of this
        cell's per-channel expression, using ``cfg.fitness_weights``.

        This is the proximate fitness signal in this model: a cell's
        division/death fate depends on what its phenotype is currently
        expressing, not directly on its genotype label (genotype shapes
        phenotype via gene_drive, and separately contributes its own
        ``genotype_fitness`` term — see CellPopulation.step() — but the
        two pathways are distinct and additive).
        """
        return (self.state.expression * self.fitness_weights.unsqueeze(0)).sum(dim=-1)

    # ------------------------------------------------------------------
    # Division / death bookkeeping (called by CellPopulation.step())
    # ------------------------------------------------------------------

    def inherit_slots(self, parent_idx: torch.Tensor, child_slots: torch.Tensor) -> None:
        """
        Copy (parent → daughter) expression state on division, with
        optional Gaussian noise (``cfg.inherit_noise``) modelling
        stochastic epigenetic drift at division. Called by
        CellPopulation.step() immediately after it assigns
        ``genotype``/``position`` for the same ``child_slots``.
        """
        inherited = self.state.expression[parent_idx]
        if self.cfg.inherit_noise > 0.0:
            inherited = inherited + torch.randn_like(inherited) * self.cfg.inherit_noise
        self.state.expression[child_slots] = inherited

    def reset_slots(self, dead_slots: torch.Tensor) -> None:
        """
        Reset expression to baseline for slots that just died — called by
        CellPopulation.step() immediately after it clears ``alive`` for
        the same ``dead_slots``, so a later division reusing that slot
        starts the daughter from a clean baseline rather than carrying
        over a stale, irrelevant expression history from whichever cell
        previously occupied it.
        """
        self.state.expression[dead_slots] = self.cfg.init_expression

    def reset(self) -> None:
        """Reset all expression state to baseline (e.g. between independent simulation runs)."""
        self.state.expression.fill_(self.cfg.init_expression)
        self._gene_drive.zero_()


# =============================================================================
# 3.5  Organelle Layer — sub-cellular state (mitochondria / nucleus /
#      lysosome / endoplasmic reticulum)
# =============================================================================
#
# PhenotypeLayer (Section 3) tracks *expression* — the gene-regulatory-
# network-level output of a cell. It does not model anything about *why*
# a cell's machinery might be failing at a level finer than "expression
# went up or down". This section adds that finer level: four coupled
# organelle subsystems, each with its own small ODE state, each
# bidirectionally coupled to the others the way real organelle crosstalk
# works, and each contributing to the cell's fate through two distinct
# pathways:
#
#   1. A slow pathway: organelle_health feeds into PhenotypeLayer's
#      gene_drive (via organelle_drive_to_phenotype below) — i.e.
#      organelle state shapes phenotype expression, exactly the same way
#      GeneNetworkBV's phi does, and the two drives are simply additive
#      before the shared tanh() in PhenotypeLayer.update().
#   2. A fast pathway: OrganelleLayer.fitness_contribution() — a direct,
#      undamped contribution to CellPopulation's division/death logit,
#      bypassing the phenotype ODE's low-pass decay entirely. This matters
#      biologically: a nucleus with DNA damage past a checkpoint threshold
#      should be able to trigger apoptosis *this step*, not several steps
#      later once a slow expression ODE catches up. Both pathways are
#      additive in CellPopulation.step(), mirroring how genotype_fitness
#      and phenotype_term already coexist there.
#
# The four organelles and what each one tracks:
#
#   Mitochondria (mito_atp, mito_psi, mito_ros)
#   --------------------------------------------
#   mito_atp : ATP budget ∈ roughly [0, 1]. Produced from the local CH3D
#              sigma signal (a stand-in for nutrient/oxygen availability —
#              sigma > 1 reads as a metabolically favourable
#              microenvironment) via mito_psi (membrane potential)-gated
#              production, consumed by a fixed per-step
#              maintenance/division cost (motility + a flat per-step
#              upkeep). Division is additionally gated by an ATP
#              sufficiency check inside CellPopulation.step (a cell with
#              mito_atp at the floor cannot pay the energetic cost of
#              dividing — see OrganelleLayer.atp_division_gate).
#   mito_psi : membrane potential ∈ [0, 1], 1 = healthy/polarised. Decays
#              under sustained ROS exposure (the textbook ROS → membrane
#              permeabilisation pathway) and is partially restored by
#              lysosomal mitophagy clearance (lyso_autophagy_flux below).
#   mito_ros : reactive oxygen species level ≥ 0. Produced as a
#              by-product of ATP production (more production at low
#              membrane potential = "leakier", more ROS per unit ATP —
#              the standard electron-transport-chain leak picture), fed
#              forward into nuclear DNA damage and ER stress, and cleared
#              by lysosomal autophagy.
#
#   Nucleus (nuc_damage, nuc_repair_capacity)
#   -------------------------------------------
#   nuc_damage : DNA damage load ∈ [0, 1]. Driven by mito_ros and by raw
#              sigma deviation (structural/replication stress), repaired
#              at a rate set by nuc_repair_capacity. Past
#              cfg.nuc_checkpoint_threshold, contributes a strongly
#              negative term to fitness_contribution() — the p53-like
#              "too much damage ⇒ apoptosis" checkpoint — implemented as
#              a smooth softplus-above-threshold penalty, never a hard
#              cutoff, so it remains differentiable.
#   nuc_repair_capacity : ∈ [0, 1], itself slowly depleted by ATP scarcity
#              (DNA repair is energetically expensive) — i.e. a starved
#              cell's repair machinery degrades over time, compounding
#              damage accumulation rather than the two being independent.
#
#   Lysosome (lyso_capacity, lyso_autophagy_flux)
#   -------------------------------------------------
#   lyso_capacity : ∈ [0, 1], degrades slowly with cell age (the
#              well-documented age-related decline in lysosomal/autophagic
#              function — "lysosomal exhaustion"), restored slightly on
#              division (a fresh daughter cell inherits a partially
#              reset lysosomal pool, not a fully reset one — see
#              cfg.lyso_inherit_fraction).
#   lyso_autophagy_flux : ∈ [0, 1], how much clearance capacity is
#              actually being exerted this step — a function of
#              lyso_capacity gated by how much there currently is to
#              clear (mito_ros + ER unfolded-protein load), so an
#              undamaged cell with full lysosomal capacity still shows
#              near-zero flux (nothing to clear), not constant high flux.
#              This flux is what reduces mito_ros and er_unfolded each
#              step (autophagy/mitophagy and ER-phagy respectively).
#
#   Endoplasmic reticulum (er_unfolded, er_upr_stress)
#   -----------------------------------------------------
#   er_unfolded : unfolded/misfolded protein load ∈ [0, 1]. Driven by
#              ATP scarcity (protein folding is ATP-dependent — a starved
#              cell accumulates misfolded protein faster) and cleared by
#              lyso_autophagy_flux.
#   er_upr_stress : unfolded protein response activation ∈ [0, 1] — a
#              smooth (softplus) function of er_unfolded crossing
#              cfg.er_upr_threshold, feeding back as an *extra* drag on
#              mito_psi (the well-established ER-mitochondria stress
#              crosstalk via Ca2+ signalling at MAM contact sites) rather
#              than being purely a downstream readout.
#
# Design choices mirror the rest of this module: batched (n_max, ) flat
# tensors, explicit-Euler ODE integration synchronised 1:1 with
# CellPopulation.step(), soft/differentiable gating throughout (no hard
# thresholds), and inherit_slots()/reset_slots() bookkeeping called by
# CellPopulation.step() at exactly the same point PhenotypeLayer's
# analogous methods already are.
# =============================================================================

@dataclass
class OrganelleConfig:
    """
    Configuration for the per-cell organelle layer (mitochondria, nucleus,
    lysosome, endoplasmic reticulum).

    n_max : must match the attached CellPopulationConfig.n_max — enforced
            automatically by CellPopulation.attach_organelle_layer(), the
            same way PhenotypeConfig.n_max is.
    dt    : integration time step for all four organelle ODEs, one
            explicit-Euler update per CellPopulation.step() call (same
            convention as PhenotypeConfig.dt).

    Mitochondria
    ------------
    mito_atp_production_gain : how strongly local CH3D sigma (clamped to
            be non-negative as a "nutrient availability" proxy) drives
            ATP production, gated by current mito_psi.
    mito_atp_upkeep          : flat per-step ATP consumption from basal
            maintenance (independent of division — division's *additional*
            ATP cost is mito_division_cost, charged only on steps where
            the cell actually divides).
    mito_division_cost       : extra one-off ATP cost charged to a
            dividing cell's *parent* slot at the moment of division
            (mirrors the real energetic cost of replicating organelles
            and biomass before cytokinesis).
    mito_psi_ros_decay       : how strongly sustained ROS degrades
            membrane potential per step.
    mito_psi_repair_gain     : how strongly lysosomal autophagy flux
            restores membrane potential per step (mitophagy clearing
            damaged mitochondrial components allows repolarisation).
    mito_ros_production_gain : ROS produced per unit of ATP production,
            scaled up at low membrane potential (electron-transport-chain
            leak increases as the chain becomes less efficient/coupled).
    mito_ros_clearance_gain  : how strongly lysosomal autophagy flux
            clears existing ROS per step.
    atp_division_floor       : minimum mito_atp required before a cell's
            division probability is not actively suppressed by the ATP
            gate (see OrganelleLayer.atp_division_gate) — below this, the
            gate smoothly pushes the effective division logit down,
            modelling "a starved cell cannot afford to divide" without a
            hard cutoff.

    Nucleus
    -------
    nuc_damage_ros_gain     : how strongly mito_ros drives DNA damage
            accumulation per step.
    nuc_damage_sigma_gain   : how strongly raw |sigma - 1| (structural/
            replication stress, independent of its sign) drives DNA
            damage accumulation per step.
    nuc_repair_gain         : how strongly nuc_repair_capacity actually
            repairs (reduces) existing nuc_damage per step.
    nuc_repair_atp_decay    : how strongly ATP scarcity (1 - mito_atp)
            depletes nuc_repair_capacity per step (DNA repair is
            energetically expensive machinery to maintain).
    nuc_checkpoint_threshold : nuc_damage level above which the p53-like
            checkpoint penalty in fitness_contribution() begins to engage
            (smoothly, via softplus — never a hard cutoff).
    nuc_checkpoint_gain      : strength of the checkpoint penalty once
            nuc_damage exceeds nuc_checkpoint_threshold.

    Lysosome
    --------
    lyso_capacity_age_decay  : how strongly cell age depletes
            lyso_capacity per step (age-related lysosomal exhaustion).
    lyso_inherit_fraction    : fraction of lyso_capacity restored toward
            1.0 (full) on division for *both* parent and daughter slots —
            0.0 = no restoration (daughter inherits parent's exact, possibly
            degraded, capacity), 1.0 = fully restored to a fresh-cell
            baseline on every division.
    lyso_clearance_demand_gain : how strongly current clearance demand
            (mito_ros + er_unfolded) gates how much of lyso_capacity is
            actually exerted as autophagy_flux this step — i.e. flux
            tracks demand, not just raw capacity (see module docstring).

    Endoplasmic reticulum
    -----------------------
    er_unfolded_atp_gain     : how strongly ATP scarcity (1 - mito_atp)
            drives unfolded-protein accumulation per step.
    er_unfolded_clearance_gain : how strongly lysosomal autophagy flux
            clears existing unfolded protein load per step (ER-phagy).
    er_upr_threshold         : er_unfolded level above which UPR
            activation begins to engage (smooth softplus, as with the
            nuclear checkpoint).
    er_upr_mito_drag         : how strongly active UPR stress additionally
            drags down mito_psi per step — the ER-mitochondria stress
            crosstalk term (Ca2+ signalling at MAM contact sites).

    Fitness coupling
    -----------------
    fitness_weight_atp     : weight on (mito_atp - 0.5) in
            fitness_contribution() — healthy ATP level is a direct
            division/death signal on top of the slower phenotype pathway.
    fitness_weight_psi     : weight on (mito_psi - 0.5).
    fitness_weight_checkpoint : weight (should be negative, or left to
            nuc_checkpoint_gain to supply the sign — see
            fitness_contribution()) on the nuclear checkpoint penalty term.
    fitness_weight_upr     : weight on -er_upr_stress (chronic UPR
            activation is a known pro-apoptotic signal once sustained).

    Phenotype coupling
    --------------------
    phenotype_drive_gain : scalar multiplier applied to
            organelle_drive_to_phenotype()'s output before it is added
            into PhenotypeLayer.set_gene_drive() — lets a caller dial the
            strength of the organelle→phenotype pathway independently of
            organelle→fitness (see CellPopulation.step()'s wiring).

    Initial state / inheritance
    ------------------------------
    init_atp, init_psi, init_lyso_capacity : starting values for the three
            "capacity-like" states (all default to a healthy 1.0 / full).
            nuc_damage, nuc_repair_capacity (default full), mito_ros,
            er_unfolded, er_upr_stress all implicitly start at the
            opposite convention (0.0 = none) and are not separately
            configurable as starting values, since "freshly born = no
            accumulated damage yet" is not a modelling choice this layer
            needs to vary.
    organelle_inherit_noise : standard deviation of Gaussian noise added
            to a daughter's inherited organelle state on division
            (mirrors PhenotypeConfig.inherit_noise) — applied to every
            state channel except lyso_capacity, which uses
            lyso_inherit_fraction instead (a deterministic partial reset,
            not a noisy copy, since lysosomal reset on division is a
            biologically distinct mechanism from stochastic epigenetic
            drift).

    Device
    ------
    device, dtype.
    """

    n_max: int = 4096
    dt: float = 1.0

    # Mitochondria
    mito_atp_production_gain: float = 0.08
    mito_atp_upkeep: float = 0.02
    mito_division_cost: float = 0.15
    mito_psi_ros_decay: float = 0.10
    mito_psi_repair_gain: float = 0.08
    mito_ros_production_gain: float = 0.05
    mito_ros_clearance_gain: float = 0.30
    atp_division_floor: float = 0.25

    # Nucleus
    nuc_damage_ros_gain: float = 0.20
    nuc_damage_sigma_gain: float = 0.05
    nuc_repair_gain: float = 0.15
    nuc_repair_atp_decay: float = 0.02
    nuc_checkpoint_threshold: float = 0.6
    nuc_checkpoint_gain: float = 3.0

    # Lysosome
    lyso_capacity_age_decay: float = 0.0015
    lyso_inherit_fraction: float = 0.4
    lyso_clearance_demand_gain: float = 1.0

    # Endoplasmic reticulum
    er_unfolded_atp_gain: float = 0.10
    er_unfolded_clearance_gain: float = 0.30
    er_upr_threshold: float = 0.5
    er_upr_mito_drag: float = 0.05

    # Fitness coupling
    fitness_weight_atp: float = 0.4
    fitness_weight_psi: float = 0.3
    fitness_weight_checkpoint: float = -1.0
    fitness_weight_upr: float = -0.3

    # Phenotype coupling
    phenotype_drive_gain: float = 1.0

    # Initial state
    init_atp: float = 1.0
    init_psi: float = 1.0
    init_lyso_capacity: float = 1.0
    organelle_inherit_noise: float = 0.01

    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        assert self.n_max >= 1
        assert self.dt > 0.0
        for name in (
            "mito_atp_production_gain", "mito_atp_upkeep", "mito_division_cost",
            "mito_psi_ros_decay", "mito_psi_repair_gain", "mito_ros_production_gain",
            "mito_ros_clearance_gain", "nuc_damage_ros_gain", "nuc_damage_sigma_gain",
            "nuc_repair_gain", "nuc_repair_atp_decay", "nuc_checkpoint_gain",
            "lyso_capacity_age_decay", "lyso_clearance_demand_gain",
            "er_unfolded_atp_gain", "er_unfolded_clearance_gain", "er_upr_mito_drag",
            "organelle_inherit_noise",
        ):
            v = getattr(self, name)
            assert v >= 0.0, f"{name} must be >= 0; got {v}."
        assert 0.0 <= self.atp_division_floor <= 1.0
        assert 0.0 <= self.nuc_checkpoint_threshold <= 1.0
        assert 0.0 <= self.er_upr_threshold <= 1.0
        assert 0.0 <= self.lyso_inherit_fraction <= 1.0
        assert 0.0 <= self.init_atp <= 1.0
        assert 0.0 <= self.init_psi <= 1.0
        assert 0.0 <= self.init_lyso_capacity <= 1.0


@dataclass
class OrganelleState:
    """
    Flat (n_max,) batched organelle state — one tensor per tracked
    quantity, same "structure of arrays" convention as CellPopulationState
    / PhenotypeState.

    Args:
        mito_atp             : ATP budget, nominally in [0, 1] (soft-bounded
                                by the ODE's own production/consumption
                                balance, not hard-clamped every step).
        mito_psi             : mitochondrial membrane potential, [0, 1].
        mito_ros             : reactive oxygen species level, >= 0.
        nuc_damage           : accumulated DNA damage, [0, 1].
        nuc_repair_capacity  : DNA repair machinery integrity, [0, 1].
        lyso_capacity        : lysosomal/autophagic capacity, [0, 1].
        lyso_autophagy_flux  : this-step realised autophagy flux, [0, 1]
                                (recomputed fresh every step — not
                                integrated, so it is exposed for
                                diagnostics but does not need its own
                                inherit/reset handling beyond the states
                                that drive it).
        er_unfolded          : unfolded/misfolded ER protein load, [0, 1].
        er_upr_stress        : unfolded protein response activation, [0, 1].
    """

    mito_atp: torch.Tensor
    mito_psi: torch.Tensor
    mito_ros: torch.Tensor
    nuc_damage: torch.Tensor
    nuc_repair_capacity: torch.Tensor
    lyso_capacity: torch.Tensor
    lyso_autophagy_flux: torch.Tensor
    er_unfolded: torch.Tensor
    er_upr_stress: torch.Tensor

    def to(self, device: torch.device) -> "OrganelleState":
        return OrganelleState(
            mito_atp=self.mito_atp.to(device),
            mito_psi=self.mito_psi.to(device),
            mito_ros=self.mito_ros.to(device),
            nuc_damage=self.nuc_damage.to(device),
            nuc_repair_capacity=self.nuc_repair_capacity.to(device),
            lyso_capacity=self.lyso_capacity.to(device),
            lyso_autophagy_flux=self.lyso_autophagy_flux.to(device),
            er_unfolded=self.er_unfolded.to(device),
            er_upr_stress=self.er_upr_stress.to(device),
        )


class OrganelleLayer(nn.Module):
    """
    Per-cell sub-cellular organelle state: mitochondria, nucleus,
    lysosome, and endoplasmic reticulum, coupled to each other and to the
    rest of the ecosystem as described in the Section 3.5 module
    docstring above.

    Args:
        cfg    : OrganelleConfig instance.
        device : compute device (normally inherited from the attaching
                  CellPopulation, not set independently).
    """

    def __init__(self, cfg: OrganelleConfig, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._device = device or get_device(cfg.device)
        n, dt_ = cfg.n_max, cfg.dtype

        self.state = OrganelleState(
            mito_atp=torch.full((n,), cfg.init_atp, device=self._device, dtype=dt_),
            mito_psi=torch.full((n,), cfg.init_psi, device=self._device, dtype=dt_),
            mito_ros=torch.zeros(n, device=self._device, dtype=dt_),
            nuc_damage=torch.zeros(n, device=self._device, dtype=dt_),
            nuc_repair_capacity=torch.ones(n, device=self._device, dtype=dt_),
            lyso_capacity=torch.full((n,), cfg.init_lyso_capacity, device=self._device, dtype=dt_),
            lyso_autophagy_flux=torch.zeros(n, device=self._device, dtype=dt_),
            er_unfolded=torch.zeros(n, device=self._device, dtype=dt_),
            er_upr_stress=torch.zeros(n, device=self._device, dtype=dt_),
        )

    # ------------------------------------------------------------------
    # ODE step — all four organelles, one explicit-Euler update
    # ------------------------------------------------------------------

    def update(self, local_sigma: torch.Tensor, alive: torch.Tensor) -> OrganelleState:
        """
        Advance all four organelle ODEs by one explicit-Euler step, in a
        fixed dependency order chosen to match real organelle crosstalk
        timing within a single step:

            1. Lysosome flux is computed FIRST, from the *previous* step's
               mito_ros / er_unfolded (i.e. "how much clearance work is
               there to do, based on what's true right now") — this flux
               is then used by mito/ER's updates within this same step, so
               clearance and accumulation are resolved simultaneously
               rather than always lagging by one full step.
            2. Mitochondria update (production gated by current psi and
               local_sigma; ROS production/clearance).
            3. Nucleus update (damage driven by the *new* mito_ros and by
               sigma deviation; repair capacity depleted by ATP scarcity).
            4. ER update (unfolded protein driven by ATP scarcity, cleared
               by lysosomal flux; UPR stress as a smooth threshold
               function of unfolded load).
            5. UPR's mito_psi drag is applied last, after mito_psi's own
               ROS-driven decay/repair this step — so a step in which UPR
               stress newly crosses threshold begins dragging mito_psi
               down starting next step, not retroactively within the same
               step it first appeared (avoids a same-step feedback loop
               that would otherwise need an implicit/iterative solve).

        Only applied to currently-alive cells — dead slots are left
        untouched here (reset explicitly via ``reset_slots`` on death,
        matching PhenotypeLayer's convention).

        Args:
            local_sigma : (n_max,) local CH3D sigma value sampled at each
                          cell's position this step (the same tensor
                          CellPopulation.step() already computes and
                          passes to PhenotypeLayer.update()).
            alive       : (n_max,) bool mask of currently-alive cells.
        Returns:
            The updated OrganelleState (also stored as ``self.state``).
        """
        cfg = self.cfg
        st = self.state
        dt = cfg.dt

        # --- 1. Lysosome: flux tracks current clearance demand ----------
        clearance_demand = st.mito_ros + st.er_unfolded
        demand_gate = torch.tanh(cfg.lyso_clearance_demand_gain * clearance_demand)
        autophagy_flux = st.lyso_capacity * demand_gate
        st.lyso_autophagy_flux = torch.where(alive, autophagy_flux, st.lyso_autophagy_flux)

        # Lysosomal capacity itself decays slowly with age — driven by
        # CellPopulation via age, not directly here (age isn't part of
        # OrganelleState); CellPopulation.step() calls
        # decay_lyso_capacity_with_age() separately each step, after this
        # update(), so the ordering note above still holds for this step's
        # autophagy_flux (computed from *last* step's capacity).

        # --- 2. Mitochondria --------------------------------------------
        nutrient = F.softplus(local_sigma, beta=4.0)  # sigma>=0 "nutrient" proxy, smooth
        atp_production = cfg.mito_atp_production_gain * st.mito_psi * nutrient
        d_atp = atp_production - cfg.mito_atp_upkeep
        new_atp = _soft_clamp(st.mito_atp + dt * d_atp, 0.0, 1.0)

        # ROS scales with production but is amplified at low psi (leaky,
        # poorly-coupled electron transport chain at low membrane potential).
        leak_factor = 1.0 + (1.0 - st.mito_psi)
        d_ros = (
            cfg.mito_ros_production_gain * atp_production * leak_factor
            - cfg.mito_ros_clearance_gain * st.lyso_autophagy_flux * st.mito_ros
        )
        new_ros = F.softplus(st.mito_ros + dt * d_ros, beta=10.0)  # soft floor at 0

        d_psi = (
            -cfg.mito_psi_ros_decay * st.mito_ros
            + cfg.mito_psi_repair_gain * st.lyso_autophagy_flux * (1.0 - st.mito_psi)
            - cfg.er_upr_mito_drag * st.er_upr_stress
        )
        new_psi = _soft_clamp(st.mito_psi + dt * d_psi, 0.0, 1.0)

        st.mito_atp = torch.where(alive, new_atp, st.mito_atp)
        st.mito_ros = torch.where(alive, new_ros, st.mito_ros)
        st.mito_psi = torch.where(alive, new_psi, st.mito_psi)

        # --- 3. Nucleus ----------------------------------------------------
        sigma_stress = (local_sigma - 1.0).abs()
        d_damage = (
            cfg.nuc_damage_ros_gain * st.mito_ros
            + cfg.nuc_damage_sigma_gain * sigma_stress
            - cfg.nuc_repair_gain * st.nuc_repair_capacity * st.nuc_damage
        )
        new_damage = _soft_clamp(st.nuc_damage + dt * d_damage, 0.0, 1.0)

        # atp_scarcity intentionally uses the NEW (just-updated) mito_atp —
        # nucleus repair-capacity depletion and ER unfolded-protein
        # accumulation (below) respond to this step's actual energy state,
        # consistent with mito_ros above also being the new, post-update value.
        atp_scarcity = 1.0 - st.mito_atp
        d_repair_capacity = -cfg.nuc_repair_atp_decay * atp_scarcity
        new_repair_capacity = _soft_clamp(
            st.nuc_repair_capacity + dt * d_repair_capacity, 0.0, 1.0
        )

        st.nuc_damage = torch.where(alive, new_damage, st.nuc_damage)
        st.nuc_repair_capacity = torch.where(alive, new_repair_capacity, st.nuc_repair_capacity)

        # --- 4. Endoplasmic reticulum --------------------------------------
        d_unfolded = (
            cfg.er_unfolded_atp_gain * atp_scarcity
            - cfg.er_unfolded_clearance_gain * st.lyso_autophagy_flux * st.er_unfolded
        )
        new_unfolded = _soft_clamp(st.er_unfolded + dt * d_unfolded, 0.0, 1.0)

        # UPR stress: smooth softplus engagement above er_upr_threshold,
        # itself soft-clamped into [0, 1] (a saturating stress response,
        # not an unbounded one).
        upr_drive = F.softplus(
            (new_unfolded - cfg.er_upr_threshold) * 10.0, beta=1.0
        ) / 10.0
        new_upr = _soft_clamp(upr_drive, 0.0, 1.0)

        st.er_unfolded = torch.where(alive, new_unfolded, st.er_unfolded)
        st.er_upr_stress = torch.where(alive, new_upr, st.er_upr_stress)

        return st

    def decay_lyso_capacity_with_age(self, age: torch.Tensor, alive: torch.Tensor) -> None:
        """
        Apply age-related lysosomal capacity decay. Called by
        CellPopulation.step() once per step, separately from ``update()``,
        since lysosomal capacity depends on ``CellPopulationState.age``
        rather than any quantity tracked inside OrganelleState itself.

        Args:
            age   : (n_max,) current age (steps survived) per cell.
            alive : (n_max,) bool mask of currently-alive cells.
        """
        cfg = self.cfg
        st = self.state
        new_capacity = _soft_clamp(
            st.lyso_capacity - cfg.lyso_capacity_age_decay * age * cfg.dt, 0.0, 1.0
        )
        st.lyso_capacity = torch.where(alive, new_capacity, st.lyso_capacity)

    # ------------------------------------------------------------------
    # Fitness feedback (organelle → CellPopulation division/death logit,
    # bypassing the phenotype ODE — see module docstring's "fast pathway")
    # ------------------------------------------------------------------

    def fitness_contribution(self) -> torch.Tensor:
        """
        (n_max,) scalar contribution to CellPopulation's division logit
        (and, with a sign flip, the death logit) — additive alongside
        genotype_fitness and PhenotypeLayer.fitness_contribution().

        Four terms, all smooth:
          + fitness_weight_atp * (mito_atp - 0.5)     — energy sufficiency
          + fitness_weight_psi * (mito_psi - 0.5)     — membrane health
          + fitness_weight_checkpoint * checkpoint_penalty  — p53-like
                DNA-damage checkpoint (checkpoint_penalty itself is always
                >= 0 via softplus, so the configured *sign* of
                fitness_weight_checkpoint — negative by default — is what
                makes this an actual penalty rather than a bonus)
          + fitness_weight_upr * er_upr_stress         — chronic UPR drag
        """
        cfg = self.cfg
        st = self.state
        checkpoint_penalty = F.softplus(
            (st.nuc_damage - cfg.nuc_checkpoint_threshold) * cfg.nuc_checkpoint_gain
        )
        return (
            cfg.fitness_weight_atp * (st.mito_atp - 0.5)
            + cfg.fitness_weight_psi * (st.mito_psi - 0.5)
            + cfg.fitness_weight_checkpoint * checkpoint_penalty
            + cfg.fitness_weight_upr * st.er_upr_stress
        )

    def atp_division_gate(self) -> torch.Tensor:
        """
        (n_max,) smooth multiplicative gate ∈ (0, 1] on division
        probability based on ATP sufficiency: ≈1.0 when mito_atp is
        comfortably above ``cfg.atp_division_floor``, smoothly falling
        toward 0 as mito_atp drops toward and below the floor — modelling
        "a starved cell cannot afford the energetic cost of division"
        without a hard cutoff. Applied multiplicatively to division_rate
        in CellPopulation.step(), never as an additive logit term (an
        energy gate is naturally multiplicative: zero energy ⇒ zero
        division probability regardless of how favourable every other
        signal is, which an additive logit term cannot guarantee).
        """
        cfg = self.cfg
        return torch.sigmoid((self.state.mito_atp - cfg.atp_division_floor) * 10.0)

    def charge_division_cost(self, parent_idx: torch.Tensor) -> None:
        """
        Deduct the one-off ATP cost of division from the parent slots
        that actually divided this step. Called by CellPopulation.step()
        immediately after it determines ``chosen_parents``.
        """
        if parent_idx.numel() == 0:
            return
        self.state.mito_atp[parent_idx] = _soft_clamp(
            self.state.mito_atp[parent_idx] - self.cfg.mito_division_cost, 0.0, 1.0
        )

    # ------------------------------------------------------------------
    # Phenotype coupling (organelle → PhenotypeLayer gene_drive — see
    # module docstring's "slow pathway", and organelle_drive_to_phenotype
    # below for the actual channel-mapping logic)
    # ------------------------------------------------------------------

    def organelle_health(self) -> torch.Tensor:
        """
        (n_max,) single scalar summary ∈ roughly [-1, 1] of overall
        organelle health this step: positive when mitochondria are
        well-fuelled and polarised with low damage/stress elsewhere,
        negative when ATP/psi are low or damage/stress are high. This is
        the signal organelle_drive_to_phenotype() broadcasts into
        PhenotypeLayer's gene_drive channels — a single aggregate rather
        than exposing all nine raw state tensors to the phenotype layer,
        since "how healthy is this cell's machinery overall" is the
        biologically meaningful quantity gene expression should respond
        to, not each organelle's raw internal state individually.
        """
        st = self.state
        return (
            0.3 * (st.mito_atp - 0.5) * 2.0
            + 0.3 * (st.mito_psi - 0.5) * 2.0
            - 0.2 * st.nuc_damage
            - 0.1 * st.er_upr_stress
            - 0.1 * st.mito_ros.clamp(max=1.0)
        )

    # ------------------------------------------------------------------
    # Division / death bookkeeping (called by CellPopulation.step(),
    # mirroring PhenotypeLayer.inherit_slots / reset_slots exactly)
    # ------------------------------------------------------------------

    def inherit_slots(self, parent_idx: torch.Tensor, child_slots: torch.Tensor) -> None:
        """
        Copy (parent → daughter) organelle state on division.

        Every channel except ``lyso_capacity`` is copied with optional
        Gaussian noise (``cfg.organelle_inherit_noise``), the same
        stochastic-epigenetic-drift convention PhenotypeLayer uses.
        ``lyso_capacity`` instead receives a deterministic partial reset
        toward 1.0 by ``cfg.lyso_inherit_fraction`` — applied to BOTH the
        parent's post-division capacity and the daughter's inherited
        capacity, modelling division as an opportunity for (partial)
        lysosomal pool renewal for both resulting cells, not a
        biologically distinct "epigenetic noise" event.
        """
        st = self.state
        cfg = self.cfg
        noise_std = cfg.organelle_inherit_noise

        def _copy(tensor: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
            inherited = tensor[parent_idx]
            if noise_std > 0.0:
                inherited = inherited + torch.randn_like(inherited) * noise_std
            return _soft_clamp(inherited, lo, hi)

        def _copy_ros(tensor: torch.Tensor) -> torch.Tensor:
            # mito_ros has no natural upper bound (unlike the other
            # [0, 1]-bounded channels) — only a soft floor at 0 is needed,
            # via the same softplus floor _soft_clamp itself uses internally.
            inherited = tensor[parent_idx]
            if noise_std > 0.0:
                inherited = inherited + torch.randn_like(inherited) * noise_std
            return F.softplus(inherited, beta=10.0)

        st.mito_atp[child_slots] = _copy(st.mito_atp)
        st.mito_psi[child_slots] = _copy(st.mito_psi)
        st.mito_ros[child_slots] = _copy_ros(st.mito_ros)
        st.nuc_damage[child_slots] = _copy(st.nuc_damage)
        st.nuc_repair_capacity[child_slots] = _copy(st.nuc_repair_capacity)
        st.er_unfolded[child_slots] = _copy(st.er_unfolded)
        st.er_upr_stress[child_slots] = _copy(st.er_upr_stress)

        # Lysosome: deterministic partial reset toward 1.0 for both
        # parent and daughter (see docstring above) — no noise applied.
        def _renew_lyso(capacity_slots: torch.Tensor) -> torch.Tensor:
            current = st.lyso_capacity[capacity_slots]
            return _soft_clamp(
                current + cfg.lyso_inherit_fraction * (1.0 - current), 0.0, 1.0,
            )

        st.lyso_capacity[child_slots] = _renew_lyso(parent_idx)
        st.lyso_capacity[parent_idx] = _renew_lyso(parent_idx)

    def reset_slots(self, dead_slots: torch.Tensor) -> None:
        """
        Reset organelle state to baseline for slots that just died —
        called by CellPopulation.step() immediately after it clears
        ``alive`` for the same ``dead_slots``, mirroring
        PhenotypeLayer.reset_slots() exactly.
        """
        cfg = self.cfg
        st = self.state
        st.mito_atp[dead_slots] = cfg.init_atp
        st.mito_psi[dead_slots] = cfg.init_psi
        st.mito_ros[dead_slots] = 0.0
        st.nuc_damage[dead_slots] = 0.0
        st.nuc_repair_capacity[dead_slots] = 1.0
        st.lyso_capacity[dead_slots] = cfg.init_lyso_capacity
        st.lyso_autophagy_flux[dead_slots] = 0.0
        st.er_unfolded[dead_slots] = 0.0
        st.er_upr_stress[dead_slots] = 0.0

    def reset(self) -> None:
        """Reset all organelle state to baseline (e.g. between independent simulation runs)."""
        cfg = self.cfg
        st = self.state
        st.mito_atp.fill_(cfg.init_atp)
        st.mito_psi.fill_(cfg.init_psi)
        st.mito_ros.zero_()
        st.nuc_damage.zero_()
        st.nuc_repair_capacity.fill_(1.0)
        st.lyso_capacity.fill_(cfg.init_lyso_capacity)
        st.lyso_autophagy_flux.zero_()
        st.er_unfolded.zero_()
        st.er_upr_stress.zero_()


def organelle_drive_to_phenotype(
    organelle: OrganelleLayer,
    n_channels: int,
    gain: Optional[float] = None,
) -> torch.Tensor:
    """
    Broadcast ``OrganelleLayer.organelle_health()`` into a
    (n_max, n_channels) drive tensor suitable for ADDING into whatever
    gene-regulatory drive a caller is already building (typically the
    output of ``gene_drive_from_bv_network``) before passing the sum to
    ``PhenotypeLayer.set_gene_drive()``.

    Broadcasting the same scalar health signal identically across every
    phenotype channel is a deliberate simplification — organelle health
    affecting every expression channel equally (rather than, say, only
    "stress_response") is the conservative default; a caller wanting
    channel-specific organelle effects can instead read
    ``organelle.state`` directly and build a custom (n_max, n_channels)
    tensor of their own, the same opt-out path
    ``gene_drive_from_bv_network``'s docstring already documents for
    genotype-specific gene drive.

    Args:
        organelle  : an OrganelleLayer instance.
        n_channels : target phenotype channel count (must match
                     PhenotypeConfig.n_channels).
        gain       : multiplier applied to organelle_health() before
                     broadcasting; defaults to
                     organelle.cfg.phenotype_drive_gain if not given.
    Returns:
        (n_max, n_channels) tensor.
    """
    if gain is None:
        gain = organelle.cfg.phenotype_drive_gain
    health = gain * organelle.organelle_health()  # (n_max,)
    return health.unsqueeze(-1).expand(-1, n_channels)


def gene_drive_from_bv_network(
    bv_network: "Any",
    genotype: torch.Tensor,
    channel_gene_map: Dict[str, List[str]],
    n_channels: int,
    channel_index: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    Build a (n_max, n_channels) gene-regulatory drive tensor from a
    ``GeneNetworkBV`` instance (``evolution_one_v4.py``) — i.e. wires
    EVOLUTION ONE's existing BV gene-interaction network directly into
    ``PhenotypeLayer.set_gene_drive()``, with no reformatting step.

    ``GeneNetworkBV.phi`` holds one scalar tensor per gene
    (``phi["phi_{i}"]`` for ``gene_names[i]``). This function averages the
    φ-values of whichever genes are mapped to each phenotype channel
    (via ``channel_gene_map``), giving every *cell* the gene-network-wide
    drive value broadcast across all cells regardless of genotype — i.e.
    this treats φ as a population-level gene-network state shared by the
    whole tumor/cohort, not a per-genotype quantity. If your use case
    needs genotype-specific gene drive (e.g. each clone perturbs the
    network differently), construct ``channel_gene_map``/the underlying
    ``bv_network.phi`` per-genotype yourself and call this function once
    per genotype, scattering the results into the relevant rows by
    ``genotype`` afterward — this function does not attempt that
    decomposition automatically, since ``GeneNetworkBV`` itself has no
    native per-genotype axis to draw it from.

    Args:
        bv_network       : a GeneNetworkBV instance — only ``.phi`` and
                           ``.gene_names`` are read.
        genotype         : (n_max,) genotype index per cell — used only
                           to determine output shape/device, not to vary
                           the drive by genotype (see note above).
        channel_gene_map : {channel_name: [gene_name, ...]} — which genes'
                           φ-values feed into which phenotype channel.
                           Genes not in ``bv_network.gene_names`` are
                           skipped with a warning (not a hard error), so a
                           partially-specified map still produces a usable
                           (if more neutral) drive signal.
        n_channels       : total phenotype channel count (must match
                           ``PhenotypeConfig.n_channels``).
        channel_index    : {channel_name: column_index} — defaults to
                           enumerating ``channel_gene_map``'s keys in
                           insertion order if not given; pass explicitly
                           to guarantee alignment with
                           ``PhenotypeConfig.channel_names``'s order.
    Returns:
        (n_max, n_channels) tensor, broadcastable directly into
        ``PhenotypeLayer.set_gene_drive()``.
    """
    if channel_index is None:
        channel_index = {name: i for i, name in enumerate(channel_gene_map.keys())}

    gene_to_phi_idx = {g: i for i, g in enumerate(bv_network.gene_names)}
    n_max = genotype.shape[0]
    device = genotype.device

    drive = torch.zeros(n_max, n_channels, device=device)
    for channel_name, gene_list in channel_gene_map.items():
        if channel_name not in channel_index:
            logger.warning("Channel %r not found in channel_index; skipping.", channel_name)
            continue
        col = channel_index[channel_name]
        phi_values = []
        for g in gene_list:
            if g not in gene_to_phi_idx:
                logger.warning(
                    "Gene %r not found in bv_network.gene_names; skipping for channel %r.",
                    g, channel_name,
                )
                continue
            phi_values.append(bv_network.phi[f"phi_{gene_to_phi_idx[g]}"].reshape(()))
        if not phi_values:
            continue
        mean_phi = torch.stack(phi_values).mean()
        drive[:, col] = mean_phi.to(device)

    return drive


# =============================================================================
# 4.  CellPopulation — differentiable division / death dynamics
# =============================================================================

class CellPopulation(nn.Module):
    """

    Batched, fully differentiable agent-based cell population.

    Each step:
      1. Cells take a small random-walk position step (motility).
      2. Local CH3D sigma (if attached, else 1.0) and per-clone fitness
         (if set, else 0.0) are combined into a per-cell division
         probability and a per-cell death probability — both smooth
         sigmoid-gated functions, never hard thresholds.
      3. Death is applied: cells "die" (their ``alive`` flag turns off)
         with that probability — sampled via a differentiable
         Gumbel-sigmoid relaxation in training mode (gradients flow into
         the rates), or a hard Bernoulli draw in eval mode (so inference
         runs produce genuinely discrete alive/dead populations).
      4. Division is applied: live cells spawn a daughter (same genotype,
         a small positional jitter away) into a currently-dead slot, up
         to the available capacity (``cfg.n_max``).

    This is a population-dynamics module, not a parametric neural network
    — it has no trainable weights of its own by default. Its
    differentiability matters because the *rates themselves* are smooth
    functions of upstream learnable quantities (CH3D's u/sigma fields,
    EvolutionONEEngine's mutation-derived fitness scores), so a training
    loop can ask "how should the phase field / fitness model change to
    produce this observed clonal outcome" via ordinary backprop through
    this module.

    Args:
        cfg : CellPopulationConfig instance.
    """

    def __init__(self, cfg: Optional[CellPopulationConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CellPopulationConfig()
        self._device = get_device(self.cfg.device)

        self.ssc = SemanticStateContraction(epsilon_fp=self.cfg.epsilon_fp)

        # Per-genotype fitness score, settable via set_genotype_fitness().
        # Shape (n_genotypes,); defaults to all-zero (no fitness bias —
        # every clone behaves identically until a real signal is wired in).
        self.register_buffer(
            "genotype_fitness",
            torch.zeros(self.cfg.n_genotypes, dtype=self.cfg.dtype),
        )

        self.state = self._spawn_initial_state()

        # Precompute the logit shifts that make _rate_from_logit reproduce
        # exactly base_division_rate / base_death_rate at logit=0 (see
        # _rate_from_logit's docstring for the derivation). These are pure
        # functions of cfg's floor/ceiling/base values, so computing them
        # once here (rather than every step()) is exact, not an approximation.
        self._division_logit_shift = self._inverse_sigmoid(
            (self.cfg.base_division_rate - self.cfg.death_rate_floor)
            / (self.cfg.division_rate_ceiling - self.cfg.death_rate_floor)
        )
        self._death_logit_shift = self._inverse_sigmoid(
            (self.cfg.base_death_rate - self.cfg.death_rate_floor)
            / (self.cfg.division_rate_ceiling - self.cfg.death_rate_floor)
        )

        # Optional phenotype layer — attached via attach_phenotype_layer().
        # When None (default), step() behaves exactly as before this
        # feature existed: division/death rates depend only on local CH3D
        # sigma and genotype_fitness. When attached, a PhenotypeLayer's
        # current expression state additionally contributes to the
        # division/death logits every step (see step()'s "phenotype_term").
        self.phenotype: Optional["PhenotypeLayer"] = None

        # Optional organelle layer — attached via attach_organelle_layer().
        # When None (default), step() behaves exactly as before this
        # feature existed. When attached, mitochondria/nucleus/lysosome/ER
        # state additionally (a) contributes its own fast-pathway
        # fitness_contribution() to the division/death logits, (b) gates
        # division_rate multiplicatively via atp_division_gate(), and (c)
        # — if a PhenotypeLayer is ALSO attached — feeds
        # organelle_drive_to_phenotype() into that layer's gene_drive
        # every step (additively on top of whatever gene_drive a caller
        # already set via set_gene_drive(), e.g. from
        # gene_drive_from_bv_network) — see step()'s "organelle_term".
        self.organelle: Optional["OrganelleLayer"] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _spawn_initial_state(self) -> CellPopulationState:
        cfg = self.cfg
        device, dtype = self._device, cfg.dtype

        position = torch.rand(cfg.n_max, 3, device=device, dtype=dtype) * cfg.box_size
        genotype = torch.randint(0, cfg.n_genotypes, (cfg.n_max,), device=device)
        age = torch.zeros(cfg.n_max, device=device, dtype=dtype)
        alive = torch.zeros(cfg.n_max, device=device, dtype=torch.bool)
        alive[: cfg.n_init] = True

        return CellPopulationState(position=position, genotype=genotype, age=age, alive=alive)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-spawn the initial population (and reset the SSC filter)."""
        if seed is not None:
            gen_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            self.state = self._spawn_initial_state()
            torch.random.set_rng_state(gen_state)
        else:
            self.state = self._spawn_initial_state()
        self.ssc.reset()

    # ------------------------------------------------------------------
    # Genotype fitness wiring
    # ------------------------------------------------------------------

    def set_genotype_fitness(self, fitness: torch.Tensor) -> None:
        """
        Set per-genotype fitness scores driving division/death bias.

        Args:
            fitness : (n_genotypes,) tensor. Typically derived from
                      EvolutionONEEngine's mutation matrix (e.g. per-clone
                      mutation count in driver genes, normalised) and/or
                      REAL FOLD ONE HT's per-gene ΔΔG — see
                      ``fitness_from_mutation_matrix`` below for a ready-made
                      construction from the same mutation matrix
                      EvolutionONEEngine.loader.build_mutation_matrix
                      already produces.
        """
        if fitness.shape != (self.cfg.n_genotypes,):
            raise ValueError(
                f"fitness must be shape ({self.cfg.n_genotypes},); got {tuple(fitness.shape)}."
            )
        self.genotype_fitness = fitness.to(device=self._device, dtype=self.cfg.dtype)

    # ------------------------------------------------------------------
    # Local environment sampling (CH3D coupling point)
    # ------------------------------------------------------------------

    def _sample_local_sigma(self, sigma_field: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Differentiable trilinear sampling of a (Nx, Ny, Nz) sigma field at
        every live cell's current position, via grid_sample — the exact
        interpolation strategy used by FoldCahnHilliardBridge
        (one_core_fold.py) for the analogous atomic-coordinate case, so the
        two cross-cluster bridges behave consistently.

        Returns 1.0 everywhere if ``sigma_field`` is None (neutral,
        CSOC-target environment — division/death fall back to their base
        rates with no spatial modulation).
        """
        n_max = self.cfg.n_max
        if sigma_field is None:
            return torch.ones(n_max, device=self._device, dtype=self.cfg.dtype)

        Nx, Ny, Nz = sigma_field.shape
        box = self.cfg.box_size
        pos = self.state.position  # (n_max, 3), Å/units, in [0, box]

        # Normalise to [-1, 1] for grid_sample, matching
        # FoldCahnHilliardBridge.sigma_to_atom_scale's convention exactly.
        cx = pos[:, 0] / box * 2.0 - 1.0
        cy = pos[:, 1] / box * 2.0 - 1.0
        cz = pos[:, 2] / box * 2.0 - 1.0
        grid = torch.stack([cz, cy, cx], dim=-1).view(1, n_max, 1, 1, 3)

        sigma_5d = sigma_field.to(dtype=self.cfg.dtype).unsqueeze(0).unsqueeze(0)
        interp = F.grid_sample(
            sigma_5d, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return interp.view(n_max)

    # ------------------------------------------------------------------
    # One population step
    # ------------------------------------------------------------------

    def step(
        self,
        sigma_field: Optional[torch.Tensor] = None,
        hard: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance the population by one division/death/motility step.

        Args:
            sigma_field : optional (Nx, Ny, Nz) CH3D structural sigma
                          field — sampled at each live cell's position to
                          modulate its division/death probability. Pass
                          the same field StructuralCahnHilliard3D.step
                          consumes/produces for a genuinely two-way
                          coupled simulation (see
                          CellPopulationCahnHilliardBridge below for the
                          reverse direction: cells → CH3D source term).
            hard        : if True, samples discrete Bernoulli division/
                          death events (no gradient through the sampling
                          step itself, though the *rates* feeding it
                          remain differentiable upstream). If False, uses
                          a Gumbel-sigmoid relaxation so gradients flow
                          through the sampled outcome too — useful when
                          training through an outcome that depends on
                          *which* cells divided/died, not just the rates.
                          Defaults to ``not self.training`` (hard sampling
                          at eval time, relaxed sampling in training mode)
                          — matching the predict()/forward() convention
                          used elsewhere in this ecosystem.
        Returns:
            Dict with:
                "n_alive_before" : (scalar) population size before this step.
                "n_alive_after"  : (scalar) population size after this step.
                "n_divided"      : (scalar) number of successful divisions
                                    (capped by available capacity).
                "n_died"         : (scalar) number of deaths.
                "division_rate"  : (n_max,) per-slot division probability
                                    used this step (alive slots only are
                                    meaningful).
                "death_rate"     : (n_max,) per-slot death probability.
                "phenotype_term" : (n_max,) PhenotypeLayer's fitness
                                    contribution this step (all-zero if no
                                    PhenotypeLayer attached).
                "organelle_term" : (n_max,) OrganelleLayer's fast-pathway
                                    fitness contribution this step
                                    (all-zero if no OrganelleLayer attached).
                "phenotype_state" : (n_max, n_channels) current expression
                                    state — only present if a PhenotypeLayer
                                    is attached.
                "organelle_state" : the current OrganelleState — only
                                    present if an OrganelleLayer is attached.
                "atp_division_gate" : (n_max,) the multiplicative ATP
                                    sufficiency gate applied to
                                    division_rate this step — only present
                                    if an OrganelleLayer is attached.
                "n_effective"    : (scalar) Wright variance-effective
                                    population size for THIS step, computed
                                    exactly from realized per-cell death/
                                    division outcomes (see the "Effective
                                    population size" comment in this
                                    method's body). Diagnostic only —
                                    reports how drift-dominated this step
                                    was; does not feed back into the
                                    dynamics.

        Note on capacity-limited division: when more cells win their
        division Bernoulli draw than there are free (dead) slots to place
        offspring into, which winners actually get a slot is chosen by an
        unbiased random permutation (not slot-index order) — see the
        "Randomise which of the wants_to_divide cells..." comment in step
        6 below for why an index-ordered choice would silently favour
        low-slot-index cells regardless of fitness.
        """
        if hard is None:
            hard = not self.training

        cfg = self.cfg
        st = self.state
        device = self._device

        n_before = st.n_alive()

        # --- 1. Motility: small random-walk position update ------------
        if cfg.motility > 0.0:
            noise = torch.randn_like(st.position) * cfg.motility
            st.position = torch.where(
                st.alive.unsqueeze(-1), st.position + noise, st.position
            ).clamp(min=0.0, max=cfg.box_size)
        st.age = torch.where(st.alive, st.age + 1.0, st.age)

        # --- 2. Organelle layer update (optional) ------------------------
        # If an OrganelleLayer is attached, advance its four ODEs FIRST
        # (mitochondria/nucleus/lysosome/ER), using this step's local
        # sigma — so its lyso-capacity-vs-age decay and resulting
        # organelle_health/fitness_contribution feed into the phenotype
        # update (next) and the rate computation (below) using "current",
        # not one-step-stale, organelle state.
        local_sigma = self._sample_local_sigma(sigma_field)        # (n_max,)
        organelle_term = torch.zeros(cfg.n_max, device=device, dtype=cfg.dtype)
        atp_gate = torch.ones(cfg.n_max, device=device, dtype=cfg.dtype)
        if self.organelle is not None:
            self.organelle.decay_lyso_capacity_with_age(age=st.age, alive=st.alive)
            self.organelle.update(local_sigma=local_sigma, alive=st.alive)
            organelle_term = self.organelle.fitness_contribution()
            atp_gate = self.organelle.atp_division_gate()

        # --- 3. Phenotype layer update (optional) -----------------------
        # If a PhenotypeLayer is attached, advance its expression state
        # one ODE step next, using this step's local sigma as one of its
        # drive signals — so the phenotype state used in the rate
        # computation below is always "current", not one step stale. If
        # an OrganelleLayer is ALSO attached, its organelle_health is
        # passed in as update()'s one-step, non-persistent extra_drive
        # argument (see PhenotypeLayer.update's docstring) — added
        # alongside whatever gene_drive a caller already set via
        # set_gene_drive() (e.g. from gene_drive_from_bv_network), without
        # mutating that persistent buffer, so the organelle contribution
        # cannot silently accumulate step over step.
        phenotype_term = torch.zeros(cfg.n_max, device=device, dtype=cfg.dtype)
        if self.phenotype is not None:
            extra_drive = None
            if self.organelle is not None:
                extra_drive = organelle_drive_to_phenotype(
                    self.organelle, self.phenotype.cfg.n_channels,
                )
            self.phenotype.update(local_sigma=local_sigma, alive=st.alive, extra_drive=extra_drive)
            phenotype_term = self.phenotype.fitness_contribution()

        # --- 4. Local environment + fitness → per-cell rates ------------
        clone_fit = self.genotype_fitness[st.genotype]              # (n_max,)
        fit_term = cfg.fitness_sign * clone_fit

        sigma_dev = local_sigma - 1.0  # deviation from CSOC-neutral target
        division_logit = (
            cfg.sigma_division_gain * sigma_dev
            + cfg.fitness_division_gain * fit_term
            + phenotype_term
            + organelle_term
        )
        death_logit = -(
            cfg.sigma_division_gain * sigma_dev
            + cfg.fitness_division_gain * fit_term
        ) - phenotype_term - organelle_term

        division_rate = self._rate_from_logit(
            division_logit, self._division_logit_shift,
            cfg.death_rate_floor, cfg.division_rate_ceiling,
        )
        death_rate = self._rate_from_logit(
            death_logit, self._death_logit_shift,
            cfg.death_rate_floor, cfg.division_rate_ceiling,
        )
        # ATP sufficiency gate: multiplicative, never additive — see
        # OrganelleLayer.atp_division_gate's docstring for why an energy
        # gate must be multiplicative rather than folded into the logit
        # above. No-op (gate == 1.0 everywhere) when no OrganelleLayer is
        # attached, so this is a strict superset of the pre-organelle
        # behaviour.
        division_rate = division_rate * atp_gate

        # Snapshot of who was alive BEFORE death/division mutate st.alive,
        # needed below for the exact per-step effective-population-size
        # diagnostic (n_effective) — every alive-before cell either dies
        # (k=0), survives without dividing (k=1), or survives and divides
        # (k=2), and that per-cell k is what n_effective is computed from.
        alive_before_mask = st.alive.clone()

        # --- 5. Death --------------------------------------------------
        death_event = self._sample_bernoulli(death_rate, hard) * st.alive.to(self.cfg.dtype)
        newly_dead = (death_event > 0.5) & st.alive
        st.alive = st.alive & (~newly_dead)
        n_died = int(newly_dead.sum().item())
        if self.phenotype is not None and n_died > 0:
            self.phenotype.reset_slots(torch.nonzero(newly_dead, as_tuple=False).flatten())
        if self.organelle is not None and n_died > 0:
            self.organelle.reset_slots(torch.nonzero(newly_dead, as_tuple=False).flatten())

        # --- 6. Division -------------------------------------------------
        division_event = self._sample_bernoulli(division_rate, hard) * st.alive.to(self.cfg.dtype)
        wants_to_divide = (division_event > 0.5) & st.alive
        parent_idx_all = torch.nonzero(wants_to_divide, as_tuple=False).flatten()

        dead_slots = torch.nonzero(~st.alive, as_tuple=False).flatten()
        n_divisions = min(parent_idx_all.numel(), dead_slots.numel())
        n_divided = 0
        # Which alive-before cells actually got to divide, once capacity
        # limiting (below) is resolved — used only for the n_effective
        # diagnostic, has no effect on population dynamics itself.
        division_realized = torch.zeros(cfg.n_max, device=device, dtype=torch.bool)
        if n_divisions > 0:
            # Randomise which of the wants_to_divide cells actually claim a
            # free slot when division is capacity-limited (n_divisions <
            # parent_idx_all.numel()). Previously this took an unshuffled
            # prefix of parent_idx_all, i.e. nonzero()'s ascending slot-
            # index order — meaning whenever capacity bound, low-slot-index
            # cells were systematically favoured to reproduce every single
            # time, regardless of fitness/sigma/genotype. That is a hidden
            # selection pressure with no biological meaning, purely a
            # tensor-layout artefact, and it silently distorts clonal
            # competition outcomes (a clone that happens to occupy
            # low-index slots — e.g. simply by having been seeded first —
            # would win capacity-limited competitions more often than its
            # actual fitness warrants). Shuffling before slicing removes
            # this bias: which cells win a capacity-limited slot is now
            # unbiased across slot index, exactly as it should be for
            # among-equal-rate-winners competition.
            perm = torch.randperm(parent_idx_all.numel(), device=device)
            chosen_parents = parent_idx_all[perm][:n_divisions]
            chosen_slots = dead_slots[:n_divisions]

            jitter = torch.randn(n_divisions, 3, device=device, dtype=cfg.dtype) * cfg.motility
            st.position[chosen_slots] = (
                st.position[chosen_parents] + jitter
            ).clamp(min=0.0, max=cfg.box_size)
            st.genotype[chosen_slots] = st.genotype[chosen_parents]
            st.age[chosen_slots] = 0.0
            st.alive[chosen_slots] = True
            n_divided = n_divisions
            division_realized[chosen_parents] = True
            if self.phenotype is not None:
                self.phenotype.inherit_slots(chosen_parents, chosen_slots)
            if self.organelle is not None:
                self.organelle.inherit_slots(chosen_parents, chosen_slots)
                self.organelle.charge_division_cost(chosen_parents)

        n_after = st.n_alive()

        # --- 7. Effective population size (demographic-stochasticity
        # diagnostic) -----------------------------------------------------
        # NOTE: this does NOT "add" drift to the model — the independent
        # per-cell Bernoulli sampling in steps 5-6 above already IS genuine
        # demographic stochasticity; drift emerges from it exactly the way
        # it does in any individual-based birth-death model. What this
        # computes is Wright's variance-effective population size,
        # Ne = N / (1 + CV_k^2), from the *realized* per-cell outcome
        # k_i in {0, 1, 2} this step (0 = died, 1 = survived without
        # dividing, 2 = survived and divided) over the cells alive at the
        # start of this step — an exact per-step value (not the aggregate
        # Poisson-rate approximation used as a fallback in
        # structural_gno_evolution_bv_standalone.py's
        # estimate_effective_population_size, for callers that only have
        # step-level totals rather than per-cell outcomes). Reported
        # alongside n_alive_after purely as a diagnostic; nothing above
        # depends on it.
        k = alive_before_mask.to(cfg.dtype) * (1.0 - newly_dead.to(cfg.dtype))
        k = k + division_realized.to(cfg.dtype)  # +1 more if it also divided
        n_ref = alive_before_mask.sum().clamp_min(1).to(cfg.dtype)
        mean_k = (k * alive_before_mask.to(cfg.dtype)).sum() / n_ref
        var_k = ((k - mean_k) ** 2 * alive_before_mask.to(cfg.dtype)).sum() / n_ref
        cv_k_sq = var_k / mean_k.clamp_min(1e-6) ** 2
        n_effective = n_ref / (1.0 + cv_k_sq)

        result = {
            "n_alive_before": torch.tensor(n_before, device=device),
            "n_alive_after":  torch.tensor(n_after, device=device),
            "n_divided":      torch.tensor(n_divided, device=device),
            "n_died":         torch.tensor(n_died, device=device),
            "division_rate":  division_rate,
            "death_rate":     death_rate,
            "phenotype_term": phenotype_term,
            "organelle_term": organelle_term,
            "n_effective":    n_effective.detach(),
        }
        if self.phenotype is not None:
            result["phenotype_state"] = self.phenotype.state.expression
        if self.organelle is not None:
            result["organelle_state"] = self.organelle.state
            result["atp_division_gate"] = atp_gate
        return result

    def attach_phenotype_layer(
        self, cfg: Optional["PhenotypeConfig"] = None,
    ) -> "PhenotypeLayer":
        """
        Construct and attach a PhenotypeLayer to this population.

        Once attached, every subsequent ``step()`` call advances the
        phenotype layer's expression ODE first (using that step's local
        CH3D sigma as one drive signal) and folds its
        ``fitness_contribution()`` into the division/death logits — on
        top of, not instead of, the existing genotype_fitness term.
        Daughter cells inherit (a copy of) their parent's expression
        state on division; dead slots have their expression reset to
        baseline.

        Args:
            cfg : PhenotypeConfig instance (default-constructed if None).
                  ``cfg.n_max`` is forced to match this population's
                  ``self.cfg.n_max`` regardless of what's passed in, so
                  the two stay shape-compatible automatically.
        Returns:
            The attached PhenotypeLayer instance (also stored as
            ``self.phenotype``).
        """
        cfg = cfg or PhenotypeConfig()
        if cfg.n_max != self.cfg.n_max:
            logger.warning(
                "PhenotypeConfig.n_max=%d does not match CellPopulationConfig.n_max=%d; "
                "overriding to match (phenotype state must be shape-compatible "
                "with the cell population it's attached to).",
                cfg.n_max, self.cfg.n_max,
            )
            import dataclasses
            cfg = dataclasses.replace(cfg, n_max=self.cfg.n_max)
        self.phenotype = PhenotypeLayer(cfg, device=self._device)
        logger.info(
            "%s: PhenotypeLayer attached (channels=%s, n_max=%d).",
            self.__class__.__name__, cfg.channel_names, cfg.n_max,
        )
        return self.phenotype

    def attach_organelle_layer(
        self, cfg: Optional["OrganelleConfig"] = None,
    ) -> "OrganelleLayer":
        """
        Construct and attach an OrganelleLayer (mitochondria / nucleus /
        lysosome / endoplasmic reticulum) to this population.

        Once attached, every subsequent ``step()`` call:
          1. Advances all four organelle ODEs first (using that step's
             local CH3D sigma and the cell's current age).
          2. Folds ``OrganelleLayer.fitness_contribution()`` into the
             division/death logits — additive alongside genotype_fitness
             and (if attached) PhenotypeLayer's own fitness_contribution.
          3. Applies ``OrganelleLayer.atp_division_gate()`` as a
             multiplicative gate on division_rate (a starved cell's
             division probability is suppressed regardless of how
             favourable every other signal is).
          4. If a PhenotypeLayer is ALSO attached, additively folds
             ``organelle_drive_to_phenotype()`` into that layer's
             expression ODE for this step only (via
             ``PhenotypeLayer.update``'s ``extra_drive`` argument, which
             does NOT mutate the persistent ``_gene_drive`` buffer) — so
             organelle health shapes phenotype expression every step
             without accumulating on top of itself over time.
          5. Charges ``mito_division_cost`` in ATP from a cell's organelle
             state at the moment it actually divides, and renews
             ``lyso_capacity`` by ``cfg.lyso_inherit_fraction`` for both
             resulting cells.
        Dead slots have their organelle state reset to baseline; daughter
        cells inherit (a noisy copy of) their parent's organelle state.

        With no OrganelleLayer attached (the default), ``step()`` behaves
        exactly as before this feature existed — every organelle-derived
        term defaults to a neutral no-op (zero fitness contribution, a
        gate of 1.0), so this is a strict superset of the
        pre-OrganelleLayer model.

        Args:
            cfg : OrganelleConfig instance (default-constructed if None).
                  ``cfg.n_max`` is forced to match this population's
                  ``self.cfg.n_max`` regardless of what's passed in, so
                  the two stay shape-compatible automatically (mirrors
                  ``attach_phenotype_layer``'s convention exactly).
        Returns:
            The attached OrganelleLayer instance (also stored as
            ``self.organelle``).
        """
        cfg = cfg or OrganelleConfig()
        if cfg.n_max != self.cfg.n_max:
            logger.warning(
                "OrganelleConfig.n_max=%d does not match CellPopulationConfig.n_max=%d; "
                "overriding to match (organelle state must be shape-compatible "
                "with the cell population it's attached to).",
                cfg.n_max, self.cfg.n_max,
            )
            import dataclasses
            cfg = dataclasses.replace(cfg, n_max=self.cfg.n_max)
        self.organelle = OrganelleLayer(cfg, device=self._device)
        logger.info(
            "%s: OrganelleLayer attached (n_max=%d).",
            self.__class__.__name__, cfg.n_max,
        )
        return self.organelle

    @staticmethod
    def _inverse_sigmoid(p: float) -> float:
        """
        logit(p) = log(p / (1-p)), the inverse of torch.sigmoid, evaluated
        on a plain Python float (used only once at __init__ time to derive
        a constant shift — never inside the per-step tensor math, so a
        non-differentiable plain-float implementation is fine here).
        """
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _rate_from_logit(
        logit: torch.Tensor, logit_shift: float, floor: float, ceiling: float,
    ) -> torch.Tensor:
        """
        rate(logit) = floor + (ceiling - floor) * sigmoid(logit + logit_shift)

        Smooth everywhere (a plain sigmoid of a shifted argument — no
        piecewise branches, so it has a well-defined gradient at every
        point, including logit=0), asymptotes to ``floor`` as
        logit → -∞ and to ``ceiling`` as logit → +∞, and — by
        construction of ``logit_shift`` (see ``__init__``, where
        ``logit_shift = logit((base_rate - floor) / (ceiling - floor))``)
        — returns exactly ``base_rate`` when ``logit == 0`` (i.e. under
        neutral sigma and zero genotype-fitness bias, reproducing
        ``cfg.base_division_rate`` / ``cfg.base_death_rate`` precisely,
        not approximately).
        """
        return floor + (ceiling - floor) * torch.sigmoid(logit + logit_shift)

    @staticmethod
    def _sample_bernoulli(prob: torch.Tensor, hard: bool) -> torch.Tensor:
        """
        Sample a {0, 1} event tensor from per-element probabilities.

        hard=True  : torch.bernoulli — discrete, no gradient through the
                     sampling step itself (matches eval-time determinism
                     conventions used elsewhere, e.g. SeqToCoarseStructure's
                     predict()).
        hard=False : Gumbel-sigmoid relaxation — differentiable w.r.t. prob,
                     converges to the same {0,1} draw as temperature → 0,
                     for training-time use where gradients need to flow
                     through *which* cells divided/died.
        """
        if hard:
            return torch.bernoulli(prob)
        eps = 1e-8
        u1 = torch.rand_like(prob).clamp(eps, 1.0 - eps)
        u2 = torch.rand_like(prob).clamp(eps, 1.0 - eps)
        logit_p = torch.log(prob.clamp_min(eps)) - torch.log((1.0 - prob).clamp_min(eps))
        gumbel_noise = torch.log(torch.log(u1) / torch.log(u2)).neg()  # logistic-ish relaxation noise
        temperature = 0.5
        soft = torch.sigmoid((logit_p + gumbel_noise) / temperature)
        return soft

    # ------------------------------------------------------------------
    # Population-level summaries (feed back into existing bridges)
    # ------------------------------------------------------------------

    def clone_frequencies(self) -> torch.Tensor:
        """
        (n_genotypes,) fraction of the *current live population* belonging
        to each genotype/clone. Differentiable w.r.t. nothing upstream by
        itself (it's a counting operation on the discrete ``alive``/
        ``genotype`` state) — used as a downstream readout/diagnostic, not
        as a gradient pathway. For a differentiable population-level
        signal, use ``population_mutation_load`` instead.
        """
        st = self.state
        alive_genotypes = st.genotype[st.alive]
        if alive_genotypes.numel() == 0:
            return torch.zeros(self.cfg.n_genotypes, device=self._device, dtype=self.cfg.dtype)
        counts = torch.bincount(alive_genotypes, minlength=self.cfg.n_genotypes).to(self.cfg.dtype)
        return counts / counts.sum().clamp_min(1.0)

    def population_mutation_load(self) -> torch.Tensor:
        """
        Scalar population-level mutation load μ ∈ (0, 1), suitable as a
        drop-in for the μ produced by CahnHilliardEvoBridge.project_to_mu
        or LangevinEvolutionBridge.project_to_mu — i.e. this is the
        agent-based-population analogue of those bridges' coarse-grained μ,
        letting EpiEvolutionBridge.mu_to_rt consume it identically.

        Computed as the population-mean (fitness_sign-weighted, SSC-
        smoothed) genotype fitness of the *currently alive* cells,
        squeezed into (0, 1) via a sigmoid — fully differentiable w.r.t.
        ``genotype_fitness`` (gradients do not flow through the discrete
        ``alive``/``genotype`` assignment itself, only through the fitness
        values attached to whichever clones happen to be alive).

        Returns:
            Scalar tensor ∈ (0, 1).
        """
        st = self.state
        if st.alive.sum() == 0:
            raw_mu = torch.zeros((), device=self._device, dtype=self.cfg.dtype)
        else:
            alive_fitness = self.cfg.fitness_sign * self.genotype_fitness[st.genotype]
            raw_mu = (alive_fitness * st.alive.to(self.cfg.dtype)).sum() / st.alive.sum().to(self.cfg.dtype)
        return torch.sigmoid(self.ssc(raw_mu))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath: str) -> None:
        payload: Dict[str, Any] = {
            "version": CELL_POPULATION_VERSION,
            "cfg": self.cfg,
            "state": {
                "position": self.state.position.detach().cpu(),
                "genotype": self.state.genotype.detach().cpu(),
                "age": self.state.age.detach().cpu(),
                "alive": self.state.alive.detach().cpu(),
            },
            "genotype_fitness": self.genotype_fitness.detach().cpu(),
        }
        # Organelle state is checkpointed alongside the core population
        # state when present (v1.1.0+) — omitted entirely for a population
        # with no OrganelleLayer attached, so old (pre-v1.1.0) checkpoints
        # and new organelle-free checkpoints remain byte-for-byte
        # equivalent in shape.
        if self.organelle is not None:
            ost = self.organelle.state
            payload["organelle_cfg"] = self.organelle.cfg
            payload["organelle_state"] = {
                "mito_atp": ost.mito_atp.detach().cpu(),
                "mito_psi": ost.mito_psi.detach().cpu(),
                "mito_ros": ost.mito_ros.detach().cpu(),
                "nuc_damage": ost.nuc_damage.detach().cpu(),
                "nuc_repair_capacity": ost.nuc_repair_capacity.detach().cpu(),
                "lyso_capacity": ost.lyso_capacity.detach().cpu(),
                "lyso_autophagy_flux": ost.lyso_autophagy_flux.detach().cpu(),
                "er_unfolded": ost.er_unfolded.detach().cpu(),
                "er_upr_stress": ost.er_upr_stress.detach().cpu(),
            }
        CheckpointManager.save(filepath, payload)

    def load_checkpoint(self, filepath: str) -> None:
        data = CheckpointManager.load(filepath)
        if data is None:
            return
        s = data["state"]
        self.state = CellPopulationState(
            position=s["position"].to(self._device),
            genotype=s["genotype"].to(self._device),
            age=s["age"].to(self._device),
            alive=s["alive"].to(self._device),
        )
        self.genotype_fitness = data["genotype_fitness"].to(self._device)

        # Organelle state: only restored if BOTH the checkpoint contains it
        # AND an OrganelleLayer has already been attached to this instance
        # via attach_organelle_layer() (mirrors how PhenotypeLayer is
        # never auto-attached by load_checkpoint either — restoring a
        # *layer's existence* from a checkpoint is the caller's
        # responsibility; this method only ever restores layer *state*).
        if "organelle_state" in data and self.organelle is not None:
            os_ = data["organelle_state"]
            dev = self._device
            self.organelle.state = OrganelleState(
                mito_atp=os_["mito_atp"].to(dev),
                mito_psi=os_["mito_psi"].to(dev),
                mito_ros=os_["mito_ros"].to(dev),
                nuc_damage=os_["nuc_damage"].to(dev),
                nuc_repair_capacity=os_["nuc_repair_capacity"].to(dev),
                lyso_capacity=os_["lyso_capacity"].to(dev),
                lyso_autophagy_flux=os_["lyso_autophagy_flux"].to(dev),
                er_unfolded=os_["er_unfolded"].to(dev),
                er_upr_stress=os_["er_upr_stress"].to(dev),
            )
        elif "organelle_state" in data and self.organelle is None:
            logger.warning(
                "Checkpoint contains organelle_state but no OrganelleLayer is "
                "attached to this CellPopulation; call attach_organelle_layer() "
                "before load_checkpoint() to restore it. Organelle state was "
                "NOT restored."
            )


# =============================================================================
# 5.  Genotype fitness construction from EvolutionONEEngine's mutation matrix
# =============================================================================

def fitness_from_mutation_matrix(
    mutation_matrix: "Any",
    gene_ddg: Optional[Dict[str, float]] = None,
    genes: Optional[List[str]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build a per-clone fitness vector from the exact (n_samples, n_genes)
    binary mutation matrix produced by
    ``MutationDataLoader.build_mutation_matrix`` in evolution_one_v4.py —
    so a real cancer-genomics cohort can drive ``CellPopulation``'s
    division/death bias directly, with no reformatting step.

    Without ``gene_ddg``, fitness for clone i is simply its total mutation
    count (row sum), normalised to roughly unit scale across clones — a
    "more mutations ⇒ higher fitness score" baseline (sign/interpretation
    controlled by ``CellPopulationConfig.fitness_sign``, not by this
    function).

    With ``gene_ddg`` (a {gene_name: ΔΔG} dict, e.g. from REAL FOLD ONE
    HT's per-gene mutation scan), each mutated gene contributes its ΔΔG
    instead of a flat 1, so genes with a larger predicted structural/
    stability impact weigh more heavily into that clone's fitness score.

    Args:
        mutation_matrix : (n_samples, n_genes) array-like (numpy array or
                          tensor), as returned by
                          MutationDataLoader.build_mutation_matrix.
        gene_ddg        : optional {gene_name: ddg_value} dict.
        genes           : gene-name list matching the matrix's column
                          order — required if ``gene_ddg`` is given.
        normalize       : if True, min-max normalise the resulting vector
                          into [0, 1] (recommended — keeps fitness on a
                          comparable scale to the sigmoid-gated rates in
                          ``CellPopulation.step`` regardless of cohort size).
    Returns:
        (n_samples,) float tensor — one fitness score per source sample/
        clone, ready to feed (after any resampling/truncation to
        ``cfg.n_genotypes``) into ``CellPopulation.set_genotype_fitness``.
    """
    M = torch.as_tensor(mutation_matrix, dtype=torch.float32)
    if M.dim() != 2:
        raise ValueError(f"mutation_matrix must be 2-D (n_samples, n_genes); got shape {tuple(M.shape)}.")

    if gene_ddg is not None:
        if genes is None:
            raise ValueError("genes must be provided alongside gene_ddg, matching the matrix's column order.")
        if len(genes) != M.shape[1]:
            raise ValueError(
                f"len(genes) ({len(genes)}) must match mutation_matrix's column count ({M.shape[1]})."
            )
        weights = torch.tensor(
            [float(gene_ddg.get(g, 0.0)) for g in genes], dtype=torch.float32
        )
        raw = M @ weights
    else:
        raw = M.sum(dim=1)

    if normalize:
        lo, hi = raw.min(), raw.max()
        span = (hi - lo).clamp_min(1e-8)
        raw = (raw - lo) / span

    return raw


# =============================================================================
# 6.  CellPopulationCahnHilliardBridge — two-way coupling with CH3D
# =============================================================================

class CellPopulationCahnHilliardBridge:
    """
    Bridge connecting ``CellPopulation`` to ``StructuralCahnHilliard3D``
    (structural_cahn_hilliard_3d.py), completing the loop:

        CH3D u, sigma  →  per-cell local environment  →  division/death
                                                              (CellPopulation.step)
        live-cell positions  →  density source term  →  CH3D u
                                                              (this bridge)

    Physical rationale
    -------------------
    ``CahnHilliardEvoBridge`` (one_core_evolution.py) already interprets
    the CH3D phase field u as "the spatial distribution of a somatic
    mutation (u = +1: mutant dominant, u = -1: WT)" at the level of a
    scalar mean-field average (``φ_mut = (mean(u)+1)/2``). This bridge
    makes that interpretation literal and two-way at the agent level: each
    live cell's genotype (mutant clone vs. not, by convention genotype
    index 0 reserved for "wild-type / founder") contributes a small
    Gaussian density bump to the *local* u field at its position — the
    same differentiable-projection strategy ``FoldCahnHilliardBridge``
    (one_core_fold.py) uses for atomic coordinates — so clonal expansion
    or extinction at the agent level now visibly reshapes the continuum
    phase field CahnHilliardEvoBridge.project_to_mu summarises, rather
    than the two staying numerically disconnected.

    This does not replace the existing scalar bridge — it is a sibling:
    ``CahnHilliardEvoBridge`` remains the cheap mean-field path for when
    no agent-based population is in play; this bridge is for when one is.

    Usage::

        ch_solver  = StructuralCahnHilliard3D(ch_cfg)
        population = CellPopulation(pop_cfg)
        bridge     = CellPopulationCahnHilliardBridge(ch_solver, population)

        u = make_sigma_field(...)  # or any (Nx,Ny,Nz) initial field
        for t in range(n_steps):
            u, sigma = bridge.coupled_step(u)

    Args:
        ch_solver  : a StructuralCahnHilliard3D (or subclass) instance.
        population : a CellPopulation instance, sharing ``grid_shape`` /
                     ``box_size`` with ``ch_solver``'s configuration.
        mutant_genotype_floor : genotype indices >= this value are treated
                     as "mutant" for the purposes of the density source
                     term (default 1, i.e. genotype 0 is wild-type/founder
                     and everything else counts as mutant). Set to 0 to
                     treat every live cell as contributing regardless of
                     genotype.
        source_strength : amplitude of each live cell's Gaussian
                     contribution to the u source term.
        source_sigma_vox : Gaussian kernel width (grid units) for the
                     cell→field density projection.
    """

    def __init__(
        self,
        ch_solver: "nn.Module",
        population: CellPopulation,
        mutant_genotype_floor: int = 1,
        source_strength: float = 0.02,
        source_sigma_vox: float = 1.5,
    ) -> None:
        if population.cfg.grid_shape != getattr(ch_solver, "_grid_shape_hint", population.cfg.grid_shape):
            # Best-effort consistency note: StructuralCahnHilliard3D infers
            # its grid shape from whatever u tensor it's called with rather
            # than storing one explicitly, so this is a soft reminder, not
            # an enforced assertion — get the grid shapes wired correctly
            # at the call site (same (Nx, Ny, Nz) tensor shape on both sides).
            pass
        self.ch = ch_solver
        self.population = population
        self.mutant_genotype_floor = mutant_genotype_floor
        self.source_strength = source_strength
        self.source_sigma_vox = source_sigma_vox

    # ------------------------------------------------------------------
    # Cells → CH3D (density source term)
    # ------------------------------------------------------------------

    def cells_to_source_field(self) -> torch.Tensor:
        """
        Project live "mutant" cells' positions onto the CH3D grid as a
        small additive source field, via the same differentiable Gaussian
        kernel strategy as FoldCahnHilliardBridge.coords_to_field.

        Returns:
            (Nx, Ny, Nz) source field, same units/scale as u, intended to
            be *added* to u (not to replace it) before the next CH3D step
            — see ``coupled_step`` below.
        """
        cfg = self.population.cfg
        st = self.population.state
        Nx, Ny, Nz = cfg.grid_shape
        device, dtype = st.position.device, st.position.dtype

        is_mutant = st.genotype >= self.mutant_genotype_floor
        mask = st.alive & is_mutant
        if mask.sum() == 0:
            return torch.zeros(Nx, Ny, Nz, device=device, dtype=dtype)

        coords = st.position[mask]  # (n_mutant, 3), in [0, box_size]

        xs = torch.linspace(0.0, cfg.box_size, Nx, device=device, dtype=dtype)
        ys = torch.linspace(0.0, cfg.box_size, Ny, device=device, dtype=dtype)
        zs = torch.linspace(0.0, cfg.box_size, Nz, device=device, dtype=dtype)
        GX, GY, GZ = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = torch.stack([GX.reshape(-1), GY.reshape(-1), GZ.reshape(-1)], dim=-1)  # (Ng, 3)

        diff = coords.unsqueeze(1) - grid_pts.unsqueeze(0)         # (n_mutant, Ng, 3)
        dist2 = (diff ** 2).sum(dim=-1)                            # (n_mutant, Ng)
        vox_size = cfg.box_size / max(Nx, Ny, Nz)
        kernel = torch.exp(-dist2 / (2.0 * (self.source_sigma_vox * vox_size) ** 2))
        rho = kernel.sum(dim=0).reshape(Nx, Ny, Nz)                 # (Nx, Ny, Nz)

        rho_max = rho.max().clamp_min(1e-12)
        return self.source_strength * (rho / rho_max)

    # ------------------------------------------------------------------
    # Full coupled step
    # ------------------------------------------------------------------

    def coupled_step(
        self,
        u: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One full two-way coupled step:
          1. Sample current u/sigma at each live cell's position; advance
             the population (division/death/motility) using ``sigma``.
          2. Add the (newly updated) population's mutant-density source
             term into u.
          3. Advance the CH3D solver by one dt using the source-augmented u.

        Args:
            u     : (Nx, Ny, Nz) current phase field.
            sigma : optional (Nx, Ny, Nz) structural sigma field; if None,
                    the population step uses the neutral sigma=1 fallback
                    and the CH3D step resolves its own default internally.
        Returns:
            (u_new, sigma_used) — sigma_used is the field actually sampled
            for the population step this call (useful for logging /
            CahnHilliardEvoBridge.interface_sharpness on the same field).
        """
        self.population.step(sigma_field=sigma)
        source = self.cells_to_source_field()
        u_raw = u + source
        # Bug fix (June 2026): hard .clamp(-1, 1) zeroed the gradient
        # wherever (u + source) saturated outside [-1, 1] — exactly the
        # regime where mutant-cell density pushes u hardest, i.e. precisely
        # where the population->field coupling gradient matters most for
        # training. Replaced with the same softplus-pair soft-clamp already
        # used elsewhere in this module for death_rate_floor /
        # division_rate_ceiling (see CellPopulationConfig docstring:
        # "soft-clamped, not hard .clamp()") — near-identity in the
        # interior, smooth saturation only near the +-1 boundary, nonzero
        # gradient everywhere.
        u_sourced = _soft_clamp(u_raw, -1.0, 1.0)
        u_new = self.ch.step(u_sourced, sigma)
        return u_new, (sigma if sigma is not None else torch.ones_like(u))


# =============================================================================
# 7.  CellPopulationMixin — attach_cell_population(), matching the
#     LangevinBridgeMixin convention used by EvolutionONEEngine /
#     EpiForecastEngine
# =============================================================================

class CellPopulationMixin:
    """
    Mixin adding ``attach_cell_population()`` to any EVOLUTION ONE engine
    class, mirroring ``LangevinBridgeMixin.attach_langevin_bridge()``.

    Usage::

        class EvolutionONEEngine(LangevinBridgeMixin, CellPopulationMixin):
            ...

        engine = EvolutionONEEngine(cfg)
        pop_cfg = CellPopulationConfig(n_genotypes=len(samples))
        population = engine.attach_cell_population(pop_cfg)

        # Wire real cohort genotypes in directly:
        M, samples = engine.loader.build_mutation_matrix(mut_df, genes)
        fitness = fitness_from_mutation_matrix(M)
        population.set_genotype_fitness(fitness[: pop_cfg.n_genotypes])

    The population is stored as ``self.cell_population`` and returned.
    """

    def attach_cell_population(
        self,
        cfg: Optional[CellPopulationConfig] = None,
    ) -> CellPopulation:
        """
        Construct and attach a CellPopulation.

        Args:
            cfg : CellPopulationConfig instance (default-constructed if None).
        Returns:
            The attached CellPopulation instance (also stored as
            ``self.cell_population``).
        """
        population = CellPopulation(cfg)
        self.cell_population = population
        logger.info(
            "%s: CellPopulation attached (n_init=%d, n_max=%d, n_genotypes=%d).",
            self.__class__.__name__, population.cfg.n_init,
            population.cfg.n_max, population.cfg.n_genotypes,
        )
        return population


# =============================================================================
# 8.  Verification Suite
# =============================================================================

if __name__ == "__main__":
    import dataclasses

    print("=" * 70)
    print("  Cell Population One — Verification Suite")
    print(f"  CELL_POPULATION_VERSION = {CELL_POPULATION_VERSION}")
    print(f"  one_core_evolution available: {_HAS_CORE_EVOLUTION}")
    print("=" * 70)

    torch.manual_seed(0)
    _device = get_device("cpu")

    _cfg = CellPopulationConfig(
        n_max=512, n_init=64, n_genotypes=4,
        grid_shape=(8, 8, 8), box_size=16.0, motility=0.3,
        device="cpu",
    )
    pop = CellPopulation(_cfg)

    # ── Test 1: initial state shapes and counts ─────────────────────────
    assert pop.state.position.shape == (512, 3)
    assert pop.state.genotype.shape == (512,)
    assert pop.state.alive.shape == (512,)
    assert pop.state.n_alive() == 64
    print(f"[PASS] Initial state shapes correct; n_alive={pop.state.n_alive()}")

    # ── Test 2: step() runs, population stays within capacity ──────────
    for _ in range(20):
        out = pop.step(hard=True)
        assert 0 <= pop.state.n_alive() <= _cfg.n_max
    print(f"[PASS] 20 steps (hard sampling) ran; final n_alive={pop.state.n_alive()}, "
          f"never exceeded n_max={_cfg.n_max}")

    # ── Test 3: division produces same-genotype daughters ──────────────
    pop2 = CellPopulation(CellPopulationConfig(
        n_max=256, n_init=32, n_genotypes=3, grid_shape=(8, 8, 8),
        box_size=16.0, base_division_rate=0.9, base_death_rate=0.0,
        device="cpu",
    ))
    pre_genotypes = set(pop2.state.genotype[pop2.state.alive].tolist())
    for _ in range(3):
        pop2.step(hard=True)
    post_genotypes = set(pop2.state.genotype[pop2.state.alive].tolist())
    assert post_genotypes.issubset(pre_genotypes.union(pre_genotypes)), (
        "Division introduced a genotype that didn't exist in the parent population."
    )
    assert pop2.state.n_alive() > 32, (
        f"High division rate / zero death rate should have grown the population; "
        f"got n_alive={pop2.state.n_alive()} from n_init=32."
    )
    print(f"[PASS] Division preserves genotype identity; population grew "
          f"32 → {pop2.state.n_alive()} under high-division/zero-death config")

    # ── Test 4: high death rate shrinks the population ──────────────────
    pop3 = CellPopulation(CellPopulationConfig(
        n_max=256, n_init=128, n_genotypes=2, grid_shape=(8, 8, 8),
        box_size=16.0, base_division_rate=0.0, base_death_rate=0.9,
        device="cpu",
    ))
    for _ in range(5):
        pop3.step(hard=True)
    assert pop3.state.n_alive() < 128, (
        f"High death rate / zero division rate should have shrunk the population; "
        f"got n_alive={pop3.state.n_alive()} from n_init=128."
    )
    print(f"[PASS] High death rate shrinks population: 128 → {pop3.state.n_alive()}")

    # ── Test 5: genotype fitness biases division (fitness_sign=+1) ─────
    pop4 = CellPopulation(CellPopulationConfig(
        n_max=2000, n_init=200, n_genotypes=2, grid_shape=(8, 8, 8),
        box_size=16.0, base_division_rate=0.1, base_death_rate=0.1,
        fitness_division_gain=0.5, fitness_sign=1.0, device="cpu",
    ))
    # Force exactly half the initial population into genotype 0 (neutral)
    # and half into genotype 1 (high fitness) for a clean comparison.
    half = 100
    pop4.state.genotype[:half] = 0
    pop4.state.genotype[half:200] = 1
    pop4.set_genotype_fitness(torch.tensor([0.0, 1.0]))
    for _ in range(15):
        pop4.step(hard=True)
    freqs = pop4.clone_frequencies()
    assert freqs[1] > freqs[0], (
        f"Higher-fitness genotype (1) should have a higher final frequency than "
        f"the neutral genotype (0) under fitness_sign=+1.0; got freqs={freqs.tolist()}."
    )
    print(f"[PASS] fitness_sign=+1.0: high-fitness clone outcompetes neutral clone "
          f"(final frequencies: genotype0={freqs[0]:.3f}, genotype1={freqs[1]:.3f})")

    # ── Test 6: fitness_sign=-1.0 inverts the bias ──────────────────────
    pop5 = CellPopulation(CellPopulationConfig(
        n_max=2000, n_init=200, n_genotypes=2, grid_shape=(8, 8, 8),
        box_size=16.0, base_division_rate=0.1, base_death_rate=0.1,
        fitness_division_gain=0.5, fitness_sign=-1.0, device="cpu",
    ))
    pop5.state.genotype[:half] = 0
    pop5.state.genotype[half:200] = 1
    pop5.set_genotype_fitness(torch.tensor([0.0, 1.0]))
    for _ in range(15):
        pop5.step(hard=True)
    freqs5 = pop5.clone_frequencies()
    assert freqs5[0] > freqs5[1], (
        f"With fitness_sign=-1.0, the 'fit' genotype (1) should be at a "
        f"disadvantage instead; got freqs={freqs5.tolist()}."
    )
    print(f"[PASS] fitness_sign=-1.0 inverts the bias "
          f"(final frequencies: genotype0={freqs5[0]:.3f}, genotype1={freqs5[1]:.3f})")

    # ── Test 7: sigma field modulates division/death (CH3D coupling point) ──
    pop6 = CellPopulation(CellPopulationConfig(
        n_max=200, n_init=100, n_genotypes=1, grid_shape=(8, 8, 8),
        box_size=16.0, base_division_rate=0.1, base_death_rate=0.1,
        sigma_division_gain=0.8, device="cpu",
    ))
    high_sigma = torch.full((8, 8, 8), 3.0)
    low_sigma = torch.full((8, 8, 8), 0.1)
    out_high = pop6.step(sigma_field=high_sigma, hard=True)
    div_rate_high = out_high["division_rate"][pop6.state.alive].mean().item()
    pop6.reset()
    out_low = pop6.step(sigma_field=low_sigma, hard=True)
    div_rate_low = out_low["division_rate"][pop6.state.alive].mean().item()
    assert div_rate_high > div_rate_low, (
        f"High sigma should increase division rate relative to low sigma; "
        f"got high={div_rate_high:.4f}, low={div_rate_low:.4f}."
    )
    print(f"[PASS] CH3D sigma field modulates division rate "
          f"(high sigma → {div_rate_high:.4f}, low sigma → {div_rate_low:.4f})")

    # ── Test 8: differentiability — gradients flow from population_mutation_load ──
    # ── into genotype_fitness through the (frozen) alive/genotype assignment ──
    pop7 = CellPopulation(CellPopulationConfig(
        n_max=64, n_init=32, n_genotypes=3, grid_shape=(4, 4, 4),
        box_size=8.0, device="cpu",
    ))
    fitness_param = torch.zeros(3, requires_grad=True)
    pop7.genotype_fitness = fitness_param
    mu = pop7.population_mutation_load()
    mu.sum().backward()
    assert fitness_param.grad is not None and torch.isfinite(fitness_param.grad).all() \
        and fitness_param.grad.abs().sum() > 0, \
        "Gradient did not flow from population_mutation_load() back into genotype_fitness."
    print(f"[PASS] Gradients flow from population_mutation_load() into genotype_fitness "
          f"(grad={fitness_param.grad.tolist()})")

    # ── Test 9: soft (relaxed) sampling is differentiable end-to-end through step() ──
    pop8 = CellPopulation(CellPopulationConfig(
        n_max=64, n_init=32, n_genotypes=2, grid_shape=(4, 4, 4),
        box_size=8.0, device="cpu",
    ))
    pop8.train()
    fitness_param2 = torch.zeros(2, requires_grad=True)
    pop8.genotype_fitness = fitness_param2
    out_soft = pop8.step(hard=False)
    loss = out_soft["division_rate"].sum() + out_soft["death_rate"].sum()
    loss.backward()
    assert fitness_param2.grad is not None and torch.isfinite(fitness_param2.grad).all(), \
        "Gradient did not flow through step()'s division_rate/death_rate outputs."
    print("[PASS] Gradients flow through step()'s division_rate/death_rate "
          "(soft/relaxed sampling mode, hard=False)")

    # ── Test 10: fitness_from_mutation_matrix matches build_mutation_matrix's shape ──
    fake_matrix = torch.tensor([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ], dtype=torch.float32)
    fit_plain = fitness_from_mutation_matrix(fake_matrix)
    assert fit_plain.shape == (4,)
    assert torch.isclose(fit_plain[3], torch.tensor(0.0)), "Zero-mutation clone should get minimum (0.0) normalised fitness."
    assert torch.isclose(fit_plain[2], torch.tensor(1.0)), "Max-mutation clone should get maximum (1.0) normalised fitness."
    print(f"[PASS] fitness_from_mutation_matrix (no ΔΔG) → {fit_plain.tolist()}")

    fit_ddg = fitness_from_mutation_matrix(
        fake_matrix,
        gene_ddg={"TP53": 2.0, "KRAS": 0.5, "BRAF": -1.0, "EGFR": 0.0},
        genes=["TP53", "KRAS", "BRAF", "EGFR"],
        normalize=False,
    )
    expected_row2 = 2.0 + 0.5 - 1.0 + 0.0  # all four genes mutated in clone 2
    assert torch.isclose(fit_ddg[2], torch.tensor(expected_row2)), \
        f"ΔΔG-weighted fitness mismatch: expected {expected_row2}, got {fit_ddg[2].item()}."
    print(f"[PASS] fitness_from_mutation_matrix (ΔΔG-weighted, unnormalised) → {fit_ddg.tolist()}")

    # ── Test 11: CellPopulationCahnHilliardBridge produces a valid source field ──
    class _FakeCHSolver(nn.Module):
        """Minimal stand-in for StructuralCahnHilliard3D.step for this test."""
        def step(self, u, sigma=None):
            return u  # identity — only exercising the bridge's plumbing here

    pop9 = CellPopulation(CellPopulationConfig(
        n_max=128, n_init=64, n_genotypes=2, grid_shape=(8, 8, 8),
        box_size=16.0, device="cpu",
    ))
    pop9.state.genotype[:32] = 0   # wild-type / founder
    pop9.state.genotype[32:64] = 1  # mutant
    bridge = CellPopulationCahnHilliardBridge(_FakeCHSolver(), pop9, mutant_genotype_floor=1)
    source = bridge.cells_to_source_field()
    assert source.shape == (8, 8, 8)
    assert torch.isfinite(source).all()
    assert source.max().item() <= bridge.source_strength + 1e-6
    print(f"[PASS] CellPopulationCahnHilliardBridge.cells_to_source_field "
          f"→ shape {tuple(source.shape)}, max={source.max().item():.4f}")

    u0 = torch.zeros(8, 8, 8)
    u_new, sigma_used = bridge.coupled_step(u0)
    assert u_new.shape == (8, 8, 8)
    assert torch.isfinite(u_new).all()
    assert sigma_used.shape == (8, 8, 8)
    print(f"[PASS] CellPopulationCahnHilliardBridge.coupled_step ran end-to-end "
          f"→ u_new shape {tuple(u_new.shape)}")

    # ── Test 12: checkpoint save/load round-trip ────────────────────────
    import tempfile, os as _os
    pop10 = CellPopulation(CellPopulationConfig(
        n_max=32, n_init=16, n_genotypes=2, grid_shape=(4, 4, 4),
        box_size=8.0, device="cpu",
    ))
    pop10.set_genotype_fitness(torch.tensor([0.3, 0.7]))
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = _os.path.join(tmpdir, "pop_ckpt.pkl")
        pop10.save_checkpoint(ckpt_path)
        pop11 = CellPopulation(CellPopulationConfig(
            n_max=32, n_init=16, n_genotypes=2, grid_shape=(4, 4, 4),
            box_size=8.0, device="cpu",
        ))
        pop11.load_checkpoint(ckpt_path)
        assert torch.equal(pop11.state.genotype, pop10.state.genotype)
        assert torch.equal(pop11.state.alive, pop10.state.alive)
        assert torch.allclose(pop11.genotype_fitness, pop10.genotype_fitness)
    print("[PASS] Checkpoint save/load round-trip preserves state and genotype_fitness")

    # ── Test 13: CellPopulationMixin wires attach_cell_population() ────
    class _FakeEngine(CellPopulationMixin):
        pass

    engine = _FakeEngine()
    attached = engine.attach_cell_population(
        CellPopulationConfig(n_max=64, n_init=8, n_genotypes=2, grid_shape=(4, 4, 4), box_size=8.0)
    )
    assert engine.cell_population is attached
    assert attached.state.n_alive() == 8
    print("[PASS] CellPopulationMixin.attach_cell_population() wires self.cell_population correctly")

    # ── Test 14: attach_phenotype_layer() wires self.phenotype, n_max matched ──
    pop_ph = CellPopulation(CellPopulationConfig(
        n_max=128, n_init=32, n_genotypes=2, grid_shape=(4, 4, 4), box_size=8.0,
    ))
    phenotype = pop_ph.attach_phenotype_layer(PhenotypeConfig(n_max=999))  # deliberately wrong n_max
    assert pop_ph.phenotype is phenotype
    assert phenotype.cfg.n_max == 128, (
        f"attach_phenotype_layer should override n_max to match the population "
        f"(128), got {phenotype.cfg.n_max}."
    )
    assert phenotype.state.expression.shape == (128, 3)  # 3 default channels
    print(f"[PASS] attach_phenotype_layer() wires self.phenotype and force-matches "
          f"n_max to the population's (128), regardless of what PhenotypeConfig specified (999)")

    # ── Test 15: phenotype ODE responds to sigma_drive and decays toward 0 ──
    ph_cfg = PhenotypeConfig(
        n_max=16, channel_names=("prolif",), decay_rate=(0.1,),
        sigma_gain=(1.0,), gene_gain=0.0, fitness_weights=(1.0,), dt=1.0,
    )
    ph = PhenotypeLayer(ph_cfg, device=torch.device("cpu"))
    alive_mask = torch.ones(16, dtype=torch.bool)
    high_sigma = torch.full((16,), 3.0)  # sigma - 1 = 2.0 -> positive drive
    for _ in range(50):
        ph.update(local_sigma=high_sigma, alive=alive_mask)
    expr_high = ph.state.expression[:, 0].mean().item()
    assert expr_high > 0.1, (
        f"Expression should rise toward a positive fixed point under sustained "
        f"high sigma; got mean expression={expr_high:.4f}."
    )
    print(f"[PASS] Phenotype ODE rises toward a positive fixed point under sustained "
          f"high sigma (mean expression={expr_high:.4f} after 50 steps)")

    ph.reset()
    low_sigma = torch.full((16,), 1.0)  # sigma - 1 = 0 -> zero drive -> decays to 0
    ph.state.expression.fill_(0.5)  # start away from baseline
    for _ in range(50):
        ph.update(local_sigma=low_sigma, alive=alive_mask)
    expr_neutral = ph.state.expression[:, 0].mean().item()
    assert abs(expr_neutral) < 0.01, (
        f"Expression should decay back to ~0 under neutral sigma (no drive); "
        f"got mean expression={expr_neutral:.4f}."
    )
    print(f"[PASS] Phenotype ODE decays back to baseline (~0) under neutral sigma "
          f"(mean expression={expr_neutral:.4f} after 50 steps from a 0.5 start)")

    # ── Test 16: fitness_contribution reflects fitness_weights sign/magnitude ──
    ph2_cfg = PhenotypeConfig(
        n_max=4, channel_names=("a", "b"), decay_rate=(0.1, 0.1),
        sigma_gain=(0.0, 0.0), fitness_weights=(2.0, -3.0),
    )
    ph2 = PhenotypeLayer(ph2_cfg, device=torch.device("cpu"))
    ph2.state.expression = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    contrib = ph2.fitness_contribution()
    expected = torch.tensor([2.0 - 3.0, 0.0, 2.0, -3.0])
    assert torch.allclose(contrib, expected), f"Expected {expected.tolist()}, got {contrib.tolist()}."
    print(f"[PASS] fitness_contribution() correctly weights channels by fitness_weights "
          f"(got {contrib.tolist()}, expected {expected.tolist()})")

    # ── Test 17: PhenotypeLayer attached to CellPopulation biases division/death ──
    # (end-to-end: a cell with high "proliferation" phenotype should divide
    # more than an otherwise-identical cell with low proliferation expression)
    pop_ph2 = CellPopulation(CellPopulationConfig(
        n_max=8, n_init=2, n_genotypes=1, grid_shape=(4, 4, 4), box_size=8.0,
        base_division_rate=0.1, base_death_rate=0.1, motility=0.0,
    ))
    ph3 = pop_ph2.attach_phenotype_layer(PhenotypeConfig(
        n_max=8, channel_names=("prolif",), decay_rate=(0.1,),
        sigma_gain=(0.0,), gene_gain=0.0, fitness_weights=(2.0,),
    ))
    ph3.state.expression[0, 0] = 5.0   # cell 0: strongly "proliferative"
    ph3.state.expression[1, 0] = -5.0  # cell 1: strongly "anti-proliferative"
    out17 = pop_ph2.step(hard=True)
    div_rate_0 = out17["division_rate"][0].item()
    div_rate_1 = out17["division_rate"][1].item()
    assert div_rate_0 > div_rate_1, (
        f"Cell with high proliferation phenotype should have a higher division rate "
        f"than one with low/negative proliferation phenotype; got cell0={div_rate_0:.4f}, "
        f"cell1={div_rate_1:.4f}."
    )
    print(f"[PASS] Attached PhenotypeLayer biases CellPopulation.step()'s division_rate "
          f"as expected (high-proliferation cell0={div_rate_0:.4f} > "
          f"low-proliferation cell1={div_rate_1:.4f})")

    # ── Test 18: inherit_slots() / reset_slots() correctness (unit-level,
    # decoupled from step()'s stochastic division/death sampling) ──────
    ph4_cfg = PhenotypeConfig(
        n_max=8, channel_names=("x", "y"), decay_rate=(0.01, 0.01),
        sigma_gain=(0.0, 0.0), gene_gain=0.0, fitness_weights=(0.0, 0.0),
        inherit_noise=0.0,
    )
    ph4 = PhenotypeLayer(ph4_cfg, device=torch.device("cpu"))
    ph4.state.expression[0] = torch.tensor([0.77, -0.33])  # parent's expression
    ph4.state.expression[3] = torch.tensor([9.9, 9.9])      # stale data in a "dead" slot

    # Division: slot 3 (currently dead) becomes a daughter of parent slot 0.
    ph4.inherit_slots(parent_idx=torch.tensor([0]), child_slots=torch.tensor([3]))
    daughter_expr = ph4.state.expression[3].tolist()
    assert daughter_expr == [0.77, -0.33], (
        f"Daughter slot should exactly inherit parent's expression "
        f"(inherit_noise=0.0); expected [0.77, -0.33], got {daughter_expr}."
    )
    print(f"[PASS] inherit_slots() copies parent expression exactly "
          f"(inherit_noise=0.0): daughter slot 3 = {daughter_expr}")

    # Death: slot 0 dies -> its expression should reset to cfg.init_expression (0.0).
    ph4.reset_slots(torch.tensor([0]))
    reset_expr = ph4.state.expression[0].tolist()
    assert reset_expr == [0.0, 0.0], f"Expected reset to [0.0, 0.0], got {reset_expr}."
    print(f"[PASS] reset_slots() resets a dead cell's expression to baseline: {reset_expr}")

    # inherit_noise > 0 should perturb (not exactly preserve) the inherited value.
    ph4_noisy = PhenotypeLayer(
        dataclasses.replace(ph4_cfg, inherit_noise=0.1), device=torch.device("cpu")
    )
    ph4_noisy.state.expression[0] = torch.tensor([0.5, 0.5])
    torch.manual_seed(3)
    ph4_noisy.inherit_slots(parent_idx=torch.tensor([0]), child_slots=torch.tensor([1]))
    noisy_expr = ph4_noisy.state.expression[1]
    assert not torch.allclose(noisy_expr, torch.tensor([0.5, 0.5])), (
        "inherit_noise=0.1 should perturb the daughter's inherited expression away "
        "from an exact copy of the parent's."
    )
    print(f"[PASS] inherit_noise > 0 perturbs daughter expression away from an "
          f"exact parent copy (parent=[0.5, 0.5] -> daughter={noisy_expr.tolist()})")

    # End-to-end smoke test: attach to a CellPopulation and run a few steps
    # to confirm step() actually CALLS inherit_slots/reset_slots without
    # erroring, regardless of which cells happen to divide/die this run.
    pop_ph3 = CellPopulation(CellPopulationConfig(
        n_max=32, n_init=8, n_genotypes=1, grid_shape=(4, 4, 4), box_size=8.0,
        base_division_rate=0.3, base_death_rate=0.05, motility=0.0,
    ))
    ph5 = pop_ph3.attach_phenotype_layer(PhenotypeConfig(
        n_max=32, channel_names=("x",), decay_rate=(0.05,), sigma_gain=(0.0,),
        gene_gain=0.0, fitness_weights=(0.0,),
    ))
    for _ in range(10):
        pop_ph3.step(hard=True)
    assert torch.isfinite(ph5.state.expression).all(), (
        "Phenotype expression became non-finite after running step() with "
        "division/death active for 10 steps."
    )
    print(f"[PASS] PhenotypeLayer attached to a live CellPopulation survives 10 "
          f"steps of real division/death activity with finite expression "
          f"(final n_alive={pop_ph3.state.n_alive()})")

    # ── Test 19: gene_drive_from_bv_network reads GeneNetworkBV.phi correctly ──
    class _FakeBVNetwork:
        """Minimal stand-in for evolution_one_v4.GeneNetworkBV for this test."""
        def __init__(self, gene_names, phi_values):
            self.gene_names = gene_names
            self.phi = {f"phi_{i}": torch.tensor([v]) for i, v in enumerate(phi_values)}

    bv = _FakeBVNetwork(
        gene_names=["TP53", "KRAS", "BRAF"],
        phi_values=[0.5, 0.3, -0.2],
    )
    channel_map = {"prolif": ["KRAS", "BRAF"], "stress": ["TP53"]}
    channel_idx = {"prolif": 0, "stress": 1}
    genotype_dummy = torch.zeros(5, dtype=torch.long)
    drive19 = gene_drive_from_bv_network(bv, genotype_dummy, channel_map, n_channels=2, channel_index=channel_idx)
    assert drive19.shape == (5, 2)
    expected_prolif = (0.3 + (-0.2)) / 2.0
    expected_stress = 0.5
    assert torch.allclose(drive19[:, 0], torch.full((5,), expected_prolif), atol=1e-5), (
        f"Expected prolif column = mean(KRAS,BRAF)={expected_prolif}, got {drive19[0,0].item()}."
    )
    assert torch.allclose(drive19[:, 1], torch.full((5,), expected_stress), atol=1e-5), (
        f"Expected stress column = phi(TP53)={expected_stress}, got {drive19[0,1].item()}."
    )
    print(f"[PASS] gene_drive_from_bv_network correctly averages mapped genes' phi "
          f"per channel (prolif={drive19[0,0].item():.3f}, stress={drive19[0,1].item():.3f}), "
          f"broadcast identically across all {drive19.shape[0]} cells")

    # ── Test 20: with no PhenotypeLayer attached, step() is unchanged (regression guard) ──
    torch.manual_seed(7)
    pop_no_ph = CellPopulation(CellPopulationConfig(
        n_max=64, n_init=16, n_genotypes=2, grid_shape=(4, 4, 4), box_size=8.0,
    ))
    assert pop_no_ph.phenotype is None
    out20 = pop_no_ph.step(hard=True)
    assert "phenotype_term" in out20 and torch.allclose(
        out20["phenotype_term"], torch.zeros(64)
    ), "phenotype_term should be all-zero when no PhenotypeLayer is attached."
    assert "phenotype_state" not in out20, (
        "phenotype_state should not appear in step()'s output when no PhenotypeLayer is attached."
    )
    print("[PASS] Regression guard: with no PhenotypeLayer attached, phenotype_term is "
          "all-zero and division/death logits reduce to the original (pre-phenotype) "
          "genotype+sigma-only formula")

    # =========================================================================
    # Organelle Layer tests (v1.1.0)
    # =========================================================================

    # ── Test 21: attach_organelle_layer() wires self.organelle, n_max matched ──
    pop_org = CellPopulation(CellPopulationConfig(
        n_max=128, n_init=32, n_genotypes=2, grid_shape=(4, 4, 4), box_size=8.0,
    ))
    organelle = pop_org.attach_organelle_layer(OrganelleConfig(n_max=999))  # deliberately wrong n_max
    assert pop_org.organelle is organelle
    assert organelle.cfg.n_max == 128, (
        f"attach_organelle_layer should override n_max to match the population "
        f"(128), got {organelle.cfg.n_max}."
    )
    assert organelle.state.mito_atp.shape == (128,)
    print("[PASS] attach_organelle_layer() wires self.organelle and force-matches "
          "n_max to the population's (128), regardless of what OrganelleConfig specified (999)")

    # ── Test 22: ATP rises under high (nutrient-favourable) sigma, falls under starvation ──
    org_cfg = OrganelleConfig(
        n_max=16, mito_atp_production_gain=0.2, mito_atp_upkeep=0.02,
        mito_ros_production_gain=0.0,  # isolate ATP dynamics from ROS feedback for this test
    )
    org1 = OrganelleLayer(org_cfg, device=torch.device("cpu"))
    alive16 = torch.ones(16, dtype=torch.bool)
    high_sigma16 = torch.full((16,), 5.0)
    for _ in range(30):
        org1.update(local_sigma=high_sigma16, alive=alive16)
    atp_fed = org1.state.mito_atp.mean().item()
    assert atp_fed > 0.5, (
        f"ATP should rise toward a high fixed point under sustained favourable "
        f"sigma; got mean mito_atp={atp_fed:.4f}."
    )
    org2 = OrganelleLayer(org_cfg, device=torch.device("cpu"))
    zero_sigma16 = torch.full((16,), -5.0)  # softplus(-5) ≈ 0 -> no nutrient -> pure upkeep drain
    for _ in range(30):
        org2.update(local_sigma=zero_sigma16, alive=alive16)
    atp_starved = org2.state.mito_atp.mean().item()
    assert atp_starved < atp_fed, (
        f"ATP under starvation ({atp_starved:.4f}) should be lower than ATP under "
        f"a favourable, nutrient-rich environment ({atp_fed:.4f})."
    )
    print(f"[PASS] Mitochondrial ATP responds to local sigma as a nutrient proxy "
          f"(favourable sigma -> mean ATP={atp_fed:.4f}, starvation -> {atp_starved:.4f})")

    # ── Test 23: sustained ROS degrades membrane potential; lysosomal flux restores it ──
    org_cfg2 = OrganelleConfig(n_max=8, mito_psi_ros_decay=0.5, mito_psi_repair_gain=0.0)
    org3 = OrganelleLayer(org_cfg2, device=torch.device("cpu"))
    org3.state.mito_ros.fill_(1.0)  # force high ROS directly, isolating the psi-decay pathway
    alive8 = torch.ones(8, dtype=torch.bool)
    neutral_sigma8 = torch.ones(8)
    psi_before = org3.state.mito_psi.mean().item()
    for _ in range(10):
        # Re-assert high ROS each step so we isolate psi's response to ROS,
        # rather than also exercising ROS's own clearance/production dynamics.
        org3.state.mito_ros.fill_(1.0)
        org3.update(local_sigma=neutral_sigma8, alive=alive8)
    psi_after = org3.state.mito_psi.mean().item()
    assert psi_after < psi_before, (
        f"Sustained high ROS should degrade membrane potential; got "
        f"psi_before={psi_before:.4f}, psi_after={psi_after:.4f}."
    )
    print(f"[PASS] Sustained ROS degrades mitochondrial membrane potential "
          f"(psi: {psi_before:.4f} -> {psi_after:.4f})")

    # ── Test 24: DNA damage checkpoint penalty engages smoothly above threshold ──
    org_cfg3 = OrganelleConfig(
        n_max=4, nuc_checkpoint_threshold=0.5, nuc_checkpoint_gain=5.0,
        fitness_weight_checkpoint=-2.0, fitness_weight_atp=0.0, fitness_weight_psi=0.0,
        fitness_weight_upr=0.0,
    )
    org4 = OrganelleLayer(org_cfg3, device=torch.device("cpu"))
    org4.state.nuc_damage = torch.tensor([0.1, 0.4, 0.6, 0.9])  # below/below/above/above threshold
    fit4 = org4.fitness_contribution()
    assert fit4[0] > fit4[2] and fit4[1] > fit4[3], (
        f"Fitness contribution should decrease as nuc_damage rises past the "
        f"checkpoint threshold; got {fit4.tolist()} for damage levels "
        f"[0.1, 0.4, 0.6, 0.9]."
    )
    assert torch.all(fit4[2:] < 0), (
        f"Cells with nuc_damage above nuc_checkpoint_threshold should incur a "
        f"net-negative fitness contribution from the checkpoint penalty alone; "
        f"got {fit4[2:].tolist()}."
    )
    print(f"[PASS] Nuclear DNA-damage checkpoint smoothly penalises fitness above "
          f"threshold (damage=[0.1,0.4,0.6,0.9] -> fitness={[round(x,3) for x in fit4.tolist()]})")

    # ── Test 25: ATP division gate suppresses division_rate for starved cells ──
    pop_atp = CellPopulation(CellPopulationConfig(
        n_max=8, n_init=8, n_genotypes=1, grid_shape=(4, 4, 4), box_size=8.0,
        base_division_rate=0.5, base_death_rate=0.01, motility=0.0,
    ))
    org5 = pop_atp.attach_organelle_layer(OrganelleConfig(n_max=8, atp_division_floor=0.5))
    org5.state.mito_atp[0] = 0.95   # well-fed
    org5.state.mito_atp[1] = 0.01   # starved, well below the floor
    out25 = pop_atp.step(hard=True)
    gate25 = out25["atp_division_gate"]
    assert gate25[0] > gate25[1], (
        f"A well-fed cell's ATP division gate ({gate25[0].item():.4f}) should exceed "
        f"a starved cell's ({gate25[1].item():.4f})."
    )
    assert gate25[1] < 0.1, (
        f"A severely starved cell (mito_atp=0.01, floor=0.5) should have an ATP "
        f"division gate close to 0; got {gate25[1].item():.4f}."
    )
    print(f"[PASS] ATP division gate suppresses a starved cell's effective division "
          f"rate (well-fed gate={gate25[0].item():.4f}, starved gate={gate25[1].item():.4f})")

    # ── Test 26: OrganelleLayer attached to a live CellPopulation survives steps ──
    # (end-to-end smoke test, mirroring Test 18's PhenotypeLayer equivalent)
    pop_org2 = CellPopulation(CellPopulationConfig(
        n_max=32, n_init=8, n_genotypes=1, grid_shape=(4, 4, 4), box_size=8.0,
        base_division_rate=0.3, base_death_rate=0.05, motility=0.0,
    ))
    org6 = pop_org2.attach_organelle_layer(OrganelleConfig(n_max=32))
    for _ in range(15):
        pop_org2.step(hard=True)
    st6 = org6.state
    for name, tensor in (
        ("mito_atp", st6.mito_atp), ("mito_psi", st6.mito_psi), ("mito_ros", st6.mito_ros),
        ("nuc_damage", st6.nuc_damage), ("nuc_repair_capacity", st6.nuc_repair_capacity),
        ("lyso_capacity", st6.lyso_capacity), ("er_unfolded", st6.er_unfolded),
        ("er_upr_stress", st6.er_upr_stress),
    ):
        assert torch.isfinite(tensor).all(), (
            f"OrganelleState.{name} became non-finite after 15 live steps."
        )
    print(f"[PASS] OrganelleLayer attached to a live CellPopulation survives 15 "
          f"steps of real division/death activity with finite state in all 8 "
          f"tracked channels (final n_alive={pop_org2.state.n_alive()})")

    # ── Test 27: OrganelleLayer + PhenotypeLayer together — organelle drive ──
    # reaches phenotype expression WITHOUT mutating PhenotypeLayer._gene_drive
    # (regression guard for the accumulation bug this design specifically avoids)
    pop_both = CellPopulation(CellPopulationConfig(
        n_max=8, n_init=8, n_genotypes=1, grid_shape=(4, 4, 4), box_size=8.0,
        base_division_rate=0.0, base_death_rate=0.0, motility=0.0,
    ))
    ph7 = pop_both.attach_phenotype_layer(PhenotypeConfig(
        n_max=8, channel_names=("prolif",), decay_rate=(0.1,),
        sigma_gain=(0.0,), gene_gain=0.0, fitness_weights=(0.0,),
    ))
    org7 = pop_both.attach_organelle_layer(OrganelleConfig(n_max=8))
    # Force a strong, persistent organelle_health signal via low mito_atp/psi.
    org7.state.mito_atp.fill_(0.0)
    org7.state.mito_psi.fill_(0.0)
    gene_drive_before = ph7._gene_drive.clone()
    for _ in range(5):
        pop_both.step(hard=True)
        # Re-force organelle state each step so organelle_health stays strongly
        # negative throughout, isolating whether _gene_drive itself accumulates.
        org7.state.mito_atp.fill_(0.0)
        org7.state.mito_psi.fill_(0.0)
    assert torch.equal(ph7._gene_drive, gene_drive_before), (
        "PhenotypeLayer._gene_drive should NEVER be mutated by the organelle->"
        "phenotype coupling pathway (it is injected per-step via update()'s "
        "extra_drive argument instead) — found _gene_drive changed after 5 "
        "steps, indicating the accumulation bug this design avoids has resurfaced."
    )
    assert torch.isfinite(ph7.state.expression).all()
    print("[PASS] Organelle->phenotype coupling drives expression via update()'s "
          "extra_drive argument without ever mutating the persistent _gene_drive "
          "buffer (no step-over-step accumulation)")

    print("=" * 70)
    print("  All tests passed.")
    print("=" * 70)
