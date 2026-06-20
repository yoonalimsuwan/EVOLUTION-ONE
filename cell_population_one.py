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
#                             [PASS]/[FAIL] verification suite
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

CELL_POPULATION_VERSION: str = "1.0.0"

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

    def update(self, local_sigma: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """
        Advance the expression ODE by one explicit-Euler step:

            E_{t+1} = E_t + dt * ( -decay_rate * E_t
                                    + tanh(gene_gain * gene_drive + sigma_gain * (sigma - 1)) )

        Only applied to currently-alive cells — dead slots' expression is
        left untouched here (it gets explicitly reset on death via
        ``reset_slots``, not silently frozen mid-decay).

        Args:
            local_sigma : (n_max,) local CH3D sigma value sampled at each
                          cell's position this step (same tensor
                          CellPopulation.step() already computes).
            alive       : (n_max,) bool mask of currently-alive cells.
        Returns:
            (n_max, n_channels) updated expression state (also stored as
            ``self.state.expression``).
        """
        cfg = self.cfg
        sigma_dev = (local_sigma - 1.0).unsqueeze(-1)               # (n_max, 1)
        drive = cfg.gene_gain * self._gene_drive + self.sigma_gain.unsqueeze(0) * sigma_dev
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

        # --- 2. Phenotype layer update (optional) -----------------------
        # If a PhenotypeLayer is attached, advance its expression state
        # one ODE step first, using this step's local sigma as one of its
        # drive signals — so the phenotype state used in the rate
        # computation below is always "current", not one step stale.
        local_sigma = self._sample_local_sigma(sigma_field)        # (n_max,)
        phenotype_term = torch.zeros(cfg.n_max, device=device, dtype=cfg.dtype)
        if self.phenotype is not None:
            self.phenotype.update(local_sigma=local_sigma, alive=st.alive)
            phenotype_term = self.phenotype.fitness_contribution()

        # --- 3. Local environment + fitness → per-cell rates ------------
        clone_fit = self.genotype_fitness[st.genotype]              # (n_max,)
        fit_term = cfg.fitness_sign * clone_fit

        sigma_dev = local_sigma - 1.0  # deviation from CSOC-neutral target
        division_logit = (
            cfg.sigma_division_gain * sigma_dev
            + cfg.fitness_division_gain * fit_term
            + phenotype_term
        )
        death_logit = -(
            cfg.sigma_division_gain * sigma_dev
            + cfg.fitness_division_gain * fit_term
        ) - phenotype_term

        division_rate = self._rate_from_logit(
            division_logit, self._division_logit_shift,
            cfg.death_rate_floor, cfg.division_rate_ceiling,
        )
        death_rate = self._rate_from_logit(
            death_logit, self._death_logit_shift,
            cfg.death_rate_floor, cfg.division_rate_ceiling,
        )

        # --- 3. Death --------------------------------------------------
        death_event = self._sample_bernoulli(death_rate, hard) * st.alive.to(self.cfg.dtype)
        newly_dead = (death_event > 0.5) & st.alive
        st.alive = st.alive & (~newly_dead)
        n_died = int(newly_dead.sum().item())
        if self.phenotype is not None and n_died > 0:
            self.phenotype.reset_slots(torch.nonzero(newly_dead, as_tuple=False).flatten())

        # --- 4. Division -------------------------------------------------
        division_event = self._sample_bernoulli(division_rate, hard) * st.alive.to(self.cfg.dtype)
        wants_to_divide = (division_event > 0.5) & st.alive
        parent_idx = torch.nonzero(wants_to_divide, as_tuple=False).flatten()

        dead_slots = torch.nonzero(~st.alive, as_tuple=False).flatten()
        n_divisions = min(parent_idx.numel(), dead_slots.numel())
        n_divided = 0
        if n_divisions > 0:
            chosen_parents = parent_idx[:n_divisions]
            chosen_slots = dead_slots[:n_divisions]

            jitter = torch.randn(n_divisions, 3, device=device, dtype=cfg.dtype) * cfg.motility
            st.position[chosen_slots] = (
                st.position[chosen_parents] + jitter
            ).clamp(min=0.0, max=cfg.box_size)
            st.genotype[chosen_slots] = st.genotype[chosen_parents]
            st.age[chosen_slots] = 0.0
            st.alive[chosen_slots] = True
            n_divided = n_divisions
            if self.phenotype is not None:
                self.phenotype.inherit_slots(chosen_parents, chosen_slots)

        n_after = st.n_alive()

        result = {
            "n_alive_before": torch.tensor(n_before, device=device),
            "n_alive_after":  torch.tensor(n_after, device=device),
            "n_divided":      torch.tensor(n_divided, device=device),
            "n_died":         torch.tensor(n_died, device=device),
            "division_rate":  division_rate,
            "death_rate":     death_rate,
            "phenotype_term": phenotype_term,
        }
        if self.phenotype is not None:
            result["phenotype_state"] = self.phenotype.state.expression
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
        CheckpointManager.save(filepath, {
            "version": CELL_POPULATION_VERSION,
            "cfg": self.cfg,
            "state": {
                "position": self.state.position.detach().cpu(),
                "genotype": self.state.genotype.detach().cpu(),
                "age": self.state.age.detach().cpu(),
                "alive": self.state.alive.detach().cpu(),
            },
            "genotype_fitness": self.genotype_fitness.detach().cpu(),
        })

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
        u_sourced = (u + source).clamp(-1.0, 1.0)
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

    print("=" * 70)
    print("  All tests passed.")
    print("=" * 70)
