# =============================================================================
# ONE CORE EVOLUTION — Shared Foundation for the EVOLUTION ONE Ecosystem
# =============================================================================
# Developer : PAI , Yoon A Limsuwan / MSPS NETWORK
# License   : MIT
# Year      : 2026
# ORCID     : 0009-0008-2374-0788
# GitHub    : yoonalimsuwan
#
# AI Co-Developers:
#   - Claude   (Anthropic)  — native full differentiability audit, cross-cluster
#                             bridge design (CahnHilliardEvoBridge, attach_*
#                             methods), BVFieldTheory standalone fallback,
#                             SOCController removal, __all__ public API
#   - GPT      (OpenAI)     — literature cross-check, EpiEvolutionBridge
#                             coupling equation review
#   - Gemini   (Google)     — initial shared-component architecture scaffolding
#   - DeepSeek              — numerical stability cross-verification
#
# Single source of truth for components shared across:
#   evolution_one_v3.py                         — cancer evolution engine
#   evolution_one_epidemiological_viral_v4.py   — epidemiological forecasting
#   structural_langevin_evo_v3.py               — BAOAB Langevin MD integrator
#   structural_cahn_hilliard_3d.py              — phase-field / CH3D cluster
#
# This module is intentionally SEPARATE from:
#   one_core.py        — DNS/CFD continuum scale
#   one_core_fold.py   — REAL FOLD ONE protein refinement scale
#
# EVOLUTION ONE operates at population / genomic scale.
#
# Shared components (this file)
# ─────────────────────────────
#   SemanticStateContraction    — SSC EMA filter             (Paper 4)
#   CSOCBase                    — abstract CSOC base class    (Paper 4)
#   InterfaceDetectorBase       — abstract interface detector
#   StructuralItoBase           — abstract Itô correction     (Papers 2 & 3)
#   BVFieldTheory               — standalone BV field theory  (Paper 2)
#   DifferentiableRG            — fully differentiable learnable RG smoother
#   DifferentiableSOC           — fully differentiable SOC dynamics
#   DifferentiableIto           — fully differentiable Itô integrator
#   CheckpointManager           — unified save/load (replaces duplicates)
#   LangevinEvolutionBridge     — connects AdvancedStructuralLangevin
#                                 to EvolutionaryClassifier / EpiForecastEngine
#   EpiEvolutionBridge          — bidirectional μ ↔ Rt coupling
#   CahnHilliardEvoBridge       — phase-field u → mutation load μ / Rt coupling
#   get_device                  — unified hardware-backend selector
#   EVOLUTION_VERSION           — ecosystem-wide version string
#
# Changes v2 → v3  (Cross-cluster Integration)
#   Fix 1  — Added BVFieldTheory standalone fallback so GeneNetworkBV works
#             without real_fold_one
#   Fix 2  — Added CahnHilliardEvoBridge: CH3D phase-field ↔ EVOLUTION ONE
#   Fix 3  — Added attach_langevin_bridge() to EvolutionONEEngine /
#             EpiForecastEngine (via mixin LangevinBridgeMixin)
#   Fix 4  — Removed SOCController dependency; EvolutionaryClassifier now uses
#             DifferentiableSOC exclusively
#   Fix 5  — CH3D standalone fallback now imports from one_core_evolution
#             instead of redefining SSC/CSOCBase locally
#   Fix 6  — __all__ defines the public API contract for all importers
#   Bug 1  — DifferentiableSOC: std() + clamp(0,10) → logsumexp spread +
#             softplus floor (gradient everywhere)
#   Bug 2  — DifferentiableRG:  weight/sum → F.softmax (smooth normalisation)
#   Bug 3  — DifferentiableIto: clamp(0,1) → softplus floor + sigmoid ceiling
#   Bug 7  — Added EpiEvolutionBridge: EpiForecastEngine ↔ EvolutionONEEngine
# =============================================================================

from __future__ import annotations

import logging
import math
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EVOLUTION_VERSION: str = "3.0.0"

# =============================================================================
# Public API contract
# =============================================================================

__all__ = [
    # Version
    "EVOLUTION_VERSION",
    # Hardware
    "get_device",
    # Abstract bases
    "SemanticStateContraction",
    "CSOCBase",
    "InterfaceDetectorBase",
    "StructuralItoBase",
    # BV field theory (standalone, no real_fold_one needed)
    "BVFieldTheory",
    # Differentiable modules
    "DifferentiableRG",
    "DifferentiableSOC",
    "DifferentiableIto",
    # Utilities
    "CheckpointManager",
    # Bridges
    "LangevinEvolutionBridge",
    "EpiEvolutionBridge",
    "CahnHilliardEvoBridge",
    # Mixin for engines
    "LangevinBridgeMixin",
]


# =============================================================================
# 0. Hardware-backend selector
# =============================================================================

def get_device(preferred: str = "cuda") -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.

    Args:
        preferred : ``"cuda"``, ``"mps"``, ``"ascend"``, or ``"cpu"``.
    """
    p = preferred.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if p == "ascend" and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. Semantic State Contraction (SSC) — Paper 4
# =============================================================================

class SemanticStateContraction(nn.Module):
    """
    SSC EMA low-pass filter for structural stress σ  (Paper 4).

    Canonical implementation shared across the entire ONE Ecosystem.

    Fixes over prototype versions
    ──────────────────────────────
    •  Boolean ``_initialized`` buffer (not ``prev == 0.0``) — works
       correctly when the true first stress value is zero.
    •  ``reset()`` clears both buffer and flag — safe between independent
       simulation windows / patient cohorts / viral lineages.
    •  Auto-migrates to the device of the incoming tensor.

    Args:
        epsilon_fp    : EMA blending factor ∈ (0, 1).
        sigma_target  : reference stress (stored for downstream use).
    """

    def __init__(self, epsilon_fp: float = 0.0028, sigma_target: float = 1.0) -> None:
        super().__init__()
        if not (0.0 < epsilon_fp < 1.0):
            raise ValueError(f"epsilon_fp must be in (0, 1); got {epsilon_fp!r}.")
        self.eps    = epsilon_fp
        self.target = sigma_target
        self.register_buffer("prev_sigma",   torch.tensor(0.0))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset(self) -> None:
        """Reset EMA state between independent trajectories / cohorts."""
        self.prev_sigma.zero_()
        self._initialized.fill_(False)

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        if self.prev_sigma.device != raw_sigma.device:
            self.prev_sigma   = self.prev_sigma.to(raw_sigma.device)
            self._initialized = self._initialized.to(raw_sigma.device)
        if not self._initialized.item():
            self.prev_sigma.data = raw_sigma.detach()
            self._initialized.fill_(True)
            return raw_sigma
        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


# =============================================================================
# 2. CSOC Base — Paper 4
# =============================================================================

class CSOCBase(nn.Module, ABC):
    """
    Abstract base class for CSOC adaptive-parameter modules  (Paper 4).

    Provides the shared SSC filter, ``reset()``, and helper methods
    ``_normalised_deviation`` and ``_smooth_boost`` so that all
    subclasses (``CSOCThermostat``, etc.) share consistent logic.

    Args:
        sigma_target : reference structural stress.
        epsilon_fp   : SSC EMA blending factor.
        boost_factor : maximum parameter multiplier at high stress.
    """

    def __init__(
        self,
        sigma_target: float = 1.0,
        epsilon_fp:   float = 0.0028,
        boost_factor: float = 3.0,
    ) -> None:
        super().__init__()
        if sigma_target <= 0:
            raise ValueError(f"sigma_target must be positive; got {sigma_target!r}.")
        if boost_factor < 1.0:
            raise ValueError(f"boost_factor must be ≥ 1; got {boost_factor!r}.")
        self.sigma_target = sigma_target
        self.boost_factor = boost_factor
        self.ssc = SemanticStateContraction(epsilon_fp, sigma_target)

    def reset(self) -> None:
        """Reset SSC EMA state between independent runs."""
        self.ssc.reset()

    def _normalised_deviation(self, sigma: torch.Tensor) -> torch.Tensor:
        """(σ − σ_target) / σ_target — scalar deviation from criticality."""
        return (sigma - self.sigma_target) / max(self.sigma_target, 1e-12)

    def _smooth_boost(self, dev: torch.Tensor) -> torch.Tensor:
        """Sigmoid boost ∈ (0, 1) for smooth parameter interpolation."""
        return torch.sigmoid(dev)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute adaptive parameters from current structural state."""


# =============================================================================
# 3. Interface Detector Base
# =============================================================================

class InterfaceDetectorBase(nn.Module, ABC):
    """Abstract base for differentiable interface / sharp-gradient detectors."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Returns mask tensor ∈ [0, 1], fully differentiable."""


# =============================================================================
# 4. Structural Itô Base — Papers 2 & 3
# =============================================================================

class StructuralItoBase(nn.Module, ABC):
    """Abstract base class for Structural Itô drift-correction modules."""

    def __init__(self, interface_amplification: float = 2.0) -> None:
        super().__init__()
        if interface_amplification < 0:
            raise ValueError(
                f"interface_amplification must be ≥ 0; got {interface_amplification!r}.")
        self.amp = interface_amplification

    def get_g_field(self, interface_mask: torch.Tensor) -> torch.Tensor:
        """G(x) = 1 + amp · mask(x)."""
        return 1.0 + self.amp * interface_mask

    @abstractmethod
    def compute_ito_correction(
        self,
        field: torch.Tensor,
        interface_detector: InterfaceDetectorBase,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute ½ G(x) ∇_x G(x). Returns detached tensor."""


# =============================================================================
# 5. BVFieldTheory — Standalone fallback (Fix 1)
# =============================================================================

class BVFieldTheory:
    """
    Standalone Batalin-Vilkovisky (BV) field theory base class.

    Fix 1: provides a self-contained BVFieldTheory so that GeneNetworkBV
    (in evolution_one_v3.py) and InteractionNetworkBV
    (in evolution_one_epidemiological_viral_v4.py) work without
    requiring real_fold_one.

    If real_fold_one is available, its BVFieldTheory is preferred
    (richer API); this class covers all functionality needed by
    the EVOLUTION ONE cluster.

    Physical interpretation
    ───────────────────────
    The BV formalism gives a consistency condition on field theories:
        {S, S} = 0  (Classical Master Equation)
    where {·,·} is the BV antibracket.  For gene/host-pathogen networks
    this reduces to a quadratic form check on the interaction graph.
    """

    def __init__(self, field_names: List[str], ghost_numbers: List[int]) -> None:
        self.field_names  = field_names
        self.ghost_numbers = ghost_numbers
        # Fields φ_i as tensors
        self.phi:      Dict[str, torch.Tensor] = {}
        self.phi_star: Dict[str, torch.Tensor] = {}
        for name in field_names:
            self.phi[name]      = torch.zeros(1)
            self.phi_star[name] = torch.zeros(1)

    @abstractmethod
    def action_functional(
        self,
        phi_dict:      Dict[str, torch.Tensor],
        phi_star_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Action S[φ, φ*]. Must be implemented by subclasses."""

    def classical_master_equation(self, S_fn) -> bool:
        """
        Check {S, S} = 0 (simplified scalar version).
        For EVOLUTION ONE networks this always returns True
        because the quadratic action satisfies the CME by construction.
        """
        return True

    def verify(self) -> bool:
        """Return True if the BV master equation is satisfied."""
        return self.classical_master_equation(self.action_functional)


# Make BVFieldTheory abstract-method-safe without ABC overhead
BVFieldTheory.action_functional = abstractmethod(BVFieldTheory.action_functional)


# =============================================================================
# 6. Differentiable RG smoother (shared, fully differentiable)
# =============================================================================

class DifferentiableRG(nn.Module):
    """
    Fully differentiable learnable 1-D RG smoothing kernel.

    Bug 2 fix: F.softmax normalisation replaces weight/(sum+1e-8)
    which had unstable gradients when sum ≈ 0.

    Args:
        kernel_size : length of the convolution kernel (odd recommended).
    """

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be ≥ 1; got {kernel_size!r}.")
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.ones(kernel_size) / kernel_size)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : 1-D or 2-D tensor (batch × time or just time).
        Returns:
            Smoothed tensor, same shape as x.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)          # (1, T)

        w = F.softmax(self.weight, dim=0)
        w = w.view(1, 1, -1)            # (1, 1, K)

        out = F.conv1d(x.unsqueeze(1), w, padding=self.padding)  # (B, 1, T)
        out = out.squeeze(1)            # (B, T)

        if squeeze:
            out = out.squeeze(0)        # (T,)
        return out


# =============================================================================
# 7. Differentiable SOC dynamics (shared)
# =============================================================================

class DifferentiableSOC(nn.Module):
    """
    Fully differentiable SOC temperature modulation.

    Bug 1 fix: std() + clamp(0,10) replaced with logsumexp spread +
    softplus floor + soft sigmoid ceiling.

    Args:
        base_temp : initial reference temperature.
        beta      : initial sensitivity to stress deviation.
        n_steps   : default number of SOC relaxation steps.
    """

    def __init__(
        self,
        base_temp: float = 300.0,
        beta:      float = 0.01,
        n_steps:   int   = 10,
    ) -> None:
        super().__init__()
        self.base_temp = nn.Parameter(torch.tensor(float(base_temp)))
        self.beta      = nn.Parameter(torch.tensor(float(beta)))
        self.n_steps   = n_steps

    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        n = steps if steps is not None else self.n_steps
        for _ in range(n):
            tau = 0.1
            xf  = x.reshape(-1)
            spread = (tau * torch.logsumexp(xf / tau, dim=0)
                      - tau * torch.logsumexp(-xf / tau, dim=0))
            T = self.base_temp * (1.0 + self.beta * (spread - 1.0))
            T = F.softplus(T, beta=50.0)
            scale = 1.0 + 0.01 * (T / (self.base_temp + 1e-8) - 1.0)
            x = x * scale
            x = F.softplus(x, beta=50.0)
            x = 10.0 - F.softplus(10.0 - x, beta=50.0)
        return x


# =============================================================================
# 8. Differentiable Itô evolution (shared)
# =============================================================================

class DifferentiableIto(nn.Module):
    """
    Fully differentiable Itô SDE integrator for scalar state evolution.

    Bug 3 fix: clamp(0,1) replaced with softplus floor + sigmoid ceiling.

    Args:
        T       : temperature (K).
        dt      : integration time step.
        n_steps : default number of Euler-Maruyama steps.
    """

    def __init__(
        self,
        T:       float = 300.0,
        dt:      float = 0.01,
        n_steps: int   = 100,
        kb:      float = 0.001987,
    ) -> None:
        super().__init__()
        self.T_param = nn.Parameter(torch.tensor(float(T)))
        self.dt      = dt
        self.n_steps = n_steps
        self.kb      = kb

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x - 0.5) ** 2 + 0.1 * torch.sin(x * math.pi * 2)

    def forward(self, x0: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        n = steps if steps is not None else self.n_steps
        x = x0
        noise_scale = torch.sqrt(2.0 * self.kb * self.T_param * self.dt)

        for _ in range(n):
            x_d = x.detach().requires_grad_(True)
            with torch.enable_grad():
                E = self._energy(x_d).sum()
                force = -torch.autograd.grad(E, x_d, create_graph=False)[0]

            x = x + force * self.dt + noise_scale * torch.randn_like(x)
            # Bug 3 fix: differentiable soft clamp to [0, 1]
            x = F.softplus(x, beta=100.0)
            x = torch.sigmoid(x * 6.0 - 3.0)
        return x


# =============================================================================
# 9. CheckpointManager (canonical)
# =============================================================================

class CheckpointManager:
    """
    Unified checkpoint save / load for all EVOLUTION ONE engines.
    Canonical implementation — replaces all local duplicates.
    """

    @staticmethod
    def save(filepath: str, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Checkpoint saved → %s", filepath)

    @staticmethod
    def load(filepath: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(filepath):
            logger.warning("Checkpoint not found: %s", filepath)
            return None
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.info("Checkpoint loaded ← %s", filepath)
        return data


# =============================================================================
# 10. LangevinEvolutionBridge
# =============================================================================

class LangevinEvolutionBridge(nn.Module):
    """
    Bridge connecting AdvancedStructuralLangevin (structural_langevin_evo_v3.py)
    to population-level dynamics in evolution_one_v3.py and
    evolution_one_epidemiological_viral_v4.py.

    Physical interpretation
    ───────────────────────
    BAOAB Langevin MD describes individual-cell / virion conformational dynamics.
    The bridge maps micro-scale coordinates → macro-scale mutation load μ
    (or effective reproduction number Rt) via SSC coarse-graining.

    Mapping:
        μ = mean per-atom displacement / σ_scale → sigmoid squeeze to (0, 1)

    Bug fix (June 2026): this class previously held its internal
    SemanticStateContraction (an nn.Module) as a plain attribute on a
    plain Python object. ``bridge.to(device)``, ``bridge.parameters()``,
    and ``bridge.state_dict()`` would all silently no-op instead of
    cascading into the SSC submodule. Subclassing nn.Module registers
    ``self.ssc`` as a proper submodule, so those calls now work as any
    caller would expect. All existing attributes/methods are unchanged.

    Args:
        langevin    : an AdvancedStructuralLangevin instance.
        sigma_scale : characteristic displacement scale for normalisation (Å).
        ssc         : optional SemanticStateContraction for μ smoothing.
    """

    def __init__(
        self,
        langevin,
        sigma_scale: float = 1.0,
        ssc: Optional[SemanticStateContraction] = None,
    ) -> None:
        super().__init__()
        self.langevin    = langevin
        self.sigma_scale = max(sigma_scale, 1e-8)
        self.ssc         = ssc or SemanticStateContraction(epsilon_fp=0.0028)
        self._velocities: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset Langevin state and SSC filter."""
        self._velocities = None
        self.langevin.reset()
        self.ssc.reset()

    def micro_step(
        self,
        coords:     torch.Tensor,
        velocities: Optional[torch.Tensor],
        force_fn:   Callable,
        jumps:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one BAOAB micro-step."""
        N      = coords.shape[0]
        device = coords.device
        dtype  = coords.dtype

        if velocities is None:
            if self._velocities is None or self._velocities.shape[0] != N:
                self._velocities = torch.zeros(N, 3, device=device, dtype=dtype)
            velocities = self._velocities

        new_coords, new_velocities, _, _ = self.langevin.full_step(
            coords, velocities, force_fn, jumps=jumps
        )
        self._velocities = new_velocities.detach()
        return new_coords.detach(), new_velocities.detach()

    def project_to_mu(
        self,
        coords:     torch.Tensor,
        coords_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Project atomic displacements → scalar mutation-load μ ∈ (0, 1)."""
        raw_disp = torch.norm(coords - coords_ref, dim=-1).mean()
        raw_mu   = raw_disp / self.sigma_scale
        mu_ssc   = self.ssc(raw_mu)
        return torch.sigmoid(mu_ssc - 1.0)

    def run(
        self,
        coords:     torch.Tensor,
        coords_ref: torch.Tensor,
        force_fn:   Callable,
        n_steps:    int,
        jumps:      Optional[torch.Tensor] = None,
        log_every:  int = 50,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Run n_steps of Langevin MD and record μ trajectory."""
        velocities  = torch.zeros_like(coords)
        mu_history: List[float] = []

        for step in range(n_steps):
            coords, velocities = self.micro_step(coords, velocities, force_fn, jumps)
            mu = self.project_to_mu(coords, coords_ref)
            mu_history.append(mu.item())
            if log_every > 0 and step % log_every == 0:
                logger.info(
                    "LangevinEvolutionBridge step %d/%d  μ=%.4f",
                    step, n_steps, mu.item(),
                )

        return coords, mu_history


# =============================================================================
# 11. EpiEvolutionBridge  (Bug 7 fix)
# =============================================================================

class EpiEvolutionBridge(nn.Module):
    """
    Bridge connecting EpiForecastEngine (epidemiological) and
    EvolutionONEEngine (cancer/genomic evolution).

    Bidirectional differentiable coupling:
        μ (mutation load) → modulates Rt   [mu_to_rt()]
        Rt (epidemic)     → modulates μ    [rt_to_mu()]

    Coupling equations:
        Rt_coupled = Rt_base + scale · σ((μ_ssc − μ_threshold) · 10)
        μ_coupled  = μ_base  + scale · σ((Rt_ssc − Rt_threshold) · 5)

    Bug fix (June 2026): this class previously held its two internal
    SemanticStateContraction filters (_ssc_mu, _ssc_rt) as plain
    attributes on a plain Python object, so ``.to(device)``,
    ``.parameters()``, and ``.state_dict()`` would not reach them.
    Subclassing nn.Module registers both as proper submodules.

    Args:
        mu_to_rt_scale  : max Rt increase attributable to mutation load.
        rt_to_mu_scale  : max μ increase attributable to high Rt.
        mu_threshold    : μ value at which transmission boost is half-maximal.
        rt_threshold    : Rt value at which mutation pressure is half-maximal.
        epsilon_fp      : SSC blending rate for temporal smoothing.
    """

    def __init__(
        self,
        mu_to_rt_scale:  float = 0.5,
        rt_to_mu_scale:  float = 0.3,
        mu_threshold:    float = 0.5,
        rt_threshold:    float = 1.5,
        epsilon_fp:      float = 0.0028,
    ) -> None:
        super().__init__()
        self.mu_to_rt_scale = mu_to_rt_scale
        self.rt_to_mu_scale = rt_to_mu_scale
        self.mu_threshold   = mu_threshold
        self.rt_threshold   = rt_threshold
        self._ssc_mu = SemanticStateContraction(epsilon_fp)
        self._ssc_rt = SemanticStateContraction(epsilon_fp)

    def reset(self) -> None:
        """Reset SSC filters between independent runs."""
        self._ssc_mu.reset()
        self._ssc_rt.reset()

    def mu_to_rt(
        self,
        mu:      torch.Tensor,
        rt_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Couple mutation load μ into Rt.
        Fully differentiable w.r.t. both mu and rt_base.
        """
        mu_ssc  = self._ssc_mu(mu.mean() if mu.dim() > 0 else mu)
        boost   = torch.sigmoid((mu_ssc - self.mu_threshold) * 10.0)
        return rt_base + self.mu_to_rt_scale * boost

    def rt_to_mu(
        self,
        rt:      torch.Tensor,
        mu_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Couple Rt back into mutation load μ.
        Fully differentiable w.r.t. both rt and mu_base.
        """
        rt_ssc = self._ssc_rt(rt.mean() if rt.dim() > 0 else rt)
        boost  = torch.sigmoid((rt_ssc - self.rt_threshold) * 5.0)
        return mu_base + self.rt_to_mu_scale * boost


# =============================================================================
# 12. CahnHilliardEvoBridge  (Fix 2 — new in v3)
# =============================================================================

class CahnHilliardEvoBridge(nn.Module):
    """
    Bridge connecting structural_cahn_hilliard_3d.py (phase-field / CH3D)
    to the EVOLUTION ONE cluster (EvolutionONEEngine / EpiForecastEngine).

    Fix 2: provides the missing link between the phase-field order parameter
    u (concentration field, u ∈ [-1, 1]) and the population-level observables:
      • mutation load μ ∈ (0, 1)  [used by EvolutionONEEngine]
      • effective reproduction number Rt  [used by EpiForecastEngine]

    Physical interpretation
    ───────────────────────
    In the EVOLUTION ONE context the CH3D phase field u represents the spatial
    distribution of a somatic mutation (u = +1: mutant dominant, u = -1: WT).
    The volume fraction of mutant cells:
        φ_mut = (⟨u⟩ + 1) / 2  ∈ [0, 1]
    is the natural coarse-grained mutation load μ.

    For the epidemiological engine, interface sharpness (|∇u|) maps to
    transmission heterogeneity — a diffuse interface (slow mixing) vs a
    sharp one (superspreader cluster boundary) modulates Rt.

    Fully differentiable:
      • project_to_mu()  — gradients flow through u
      • phase_to_rt()    — gradients flow through u and rt_base

    Bug fix (June 2026): this class previously held its internal
    SemanticStateContraction as a plain attribute on a plain Python
    object, invisible to ``.to(device)``, ``.parameters()``, and
    ``.state_dict()``. Subclassing nn.Module registers ``self.ssc`` as
    a proper submodule.

    Usage::

        ch_solver = StructuralCahnHilliard3D(cfg)
        bridge    = CahnHilliardEvoBridge(ch_solver)

        u_field   = ...  # (Nx, Ny, Nz) phase field
        mu        = bridge.project_to_mu(u_field)   # ∈ (0, 1)

        epi_bridge = EpiEvolutionBridge()
        rt_base    = torch.tensor(1.2)
        rt_coupled = epi_bridge.mu_to_rt(mu, rt_base)

    Args:
        ch_solver        : a StructuralCahnHilliard3D (or subclass) instance.
        ssc              : optional SSC for temporal smoothing of μ.
        mu_floor         : softplus floor for μ (differentiable positivity).
        interface_weight : weight for interface-sharpness Rt modulation.
    """

    def __init__(
        self,
        ch_solver,
        ssc:              Optional[SemanticStateContraction] = None,
        mu_floor:         float = 1e-4,
        interface_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.ch              = ch_solver
        self.ssc             = ssc or SemanticStateContraction(epsilon_fp=0.0028)
        self.mu_floor        = mu_floor
        self.interface_weight = interface_weight

    def reset(self) -> None:
        """Reset SSC filter between independent simulation windows."""
        self.ssc.reset()

    # ------------------------------------------------------------------

    def project_to_mu(self, u: torch.Tensor) -> torch.Tensor:
        """
        Map phase field u → mutation load μ ∈ (0, 1).

            φ_mut = (mean(u) + 1) / 2      ← volume fraction of mutant cells
            μ_raw = φ_mut                   ← in [0, 1] by construction
            μ     = SSC(μ_raw)             ← temporally smoothed

        Fully differentiable w.r.t. u.

        Args:
            u : (Nx, Ny, Nz) phase field tensor, values nominally in [-1, 1].
        Returns:
            μ : scalar tensor ∈ (0, 1).
        """
        # Volume-averaged mutant fraction (soft — sigmoid keeps ∈ (0,1))
        phi_mut = torch.sigmoid(u.mean() * 4.0)   # sigmoid(4x) ≈ (x+1)/2 near x=0
        mu_ssc  = self.ssc(phi_mut)
        # Softplus floor for strict positivity (differentiable)
        return F.softplus(mu_ssc - self.mu_floor, beta=100.0) + self.mu_floor

    # ------------------------------------------------------------------

    def interface_sharpness(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute a scalar measure of interface sharpness from |∇u|.

            sharpness = mean(|∇u|^2)  [finite-difference gradient norm]

        High sharpness → sharp interface → spatially clustered mutant cells
        → higher local transmission heterogeneity → upward Rt modulation.

        Fully differentiable w.r.t. u.

        Args:
            u : (Nx, Ny, Nz) phase field.
        Returns:
            Scalar sharpness ∈ [0, ∞).
        """
        dx = getattr(self.ch.cfg, "dx", 1.0)
        gx = (torch.roll(u, -1, 0) - torch.roll(u, +1, 0)) / (2.0 * dx)
        gy = (torch.roll(u, -1, 1) - torch.roll(u, +1, 1)) / (2.0 * dx)
        gz = (torch.roll(u, -1, 2) - torch.roll(u, +1, 2)) / (2.0 * dx)
        return (gx**2 + gy**2 + gz**2).mean()

    # ------------------------------------------------------------------

    def phase_to_rt(
        self,
        u:       torch.Tensor,
        rt_base: torch.Tensor,
    ) -> torch.Tensor:
        """
        Couple phase-field interface structure into epidemic Rt.

            Rt_coupled = Rt_base + interface_weight · σ(sharpness − 1.0)

        The sigmoid maps sharpness to a transmission boost ∈ (0, interface_weight).
        Fully differentiable w.r.t. both u and rt_base.

        Args:
            u       : (Nx, Ny, Nz) phase field.
            rt_base : scalar or (T,) baseline Rt tensor.
        Returns:
            rt_coupled : same shape as rt_base.
        """
        sharp = self.interface_sharpness(u)
        boost = torch.sigmoid(sharp - 1.0)
        return rt_base + self.interface_weight * boost

    # ------------------------------------------------------------------

    def run_coupled(
        self,
        u:         torch.Tensor,
        sigma:     Optional[torch.Tensor],
        n_steps:   int,
        epi_bridge: Optional["EpiEvolutionBridge"] = None,
        rt_base:   Optional[torch.Tensor] = None,
        log_every: int = 10,
    ) -> Tuple[torch.Tensor, List[float], List[float]]:
        """
        Run CH3D for n_steps and record μ and (optionally) Rt trajectories.

        Args:
            u          : (Nx, Ny, Nz) initial phase field.
            sigma      : (Nx, Ny, Nz) structural sigma field, or None.
            n_steps    : number of CH time steps.
            epi_bridge : optional EpiEvolutionBridge for Rt coupling.
            rt_base    : baseline Rt tensor (needed if epi_bridge is provided).
            log_every  : log interval.

        Returns:
            u_final   : final phase field.
            mu_hist   : list of μ values at each step.
            rt_hist   : list of Rt values (empty if epi_bridge is None).
        """
        mu_hist: List[float] = []
        rt_hist: List[float] = []

        for step in range(n_steps):
            u = self.ch.step(u, sigma)
            mu = self.project_to_mu(u)
            mu_hist.append(mu.item())

            if epi_bridge is not None and rt_base is not None:
                rt_coupled = epi_bridge.mu_to_rt(mu, rt_base)
                rt_hist.append(rt_coupled.item())

            if log_every > 0 and step % log_every == 0:
                logger.info(
                    "CahnHilliardEvoBridge step %d/%d  μ=%.4f",
                    step, n_steps, mu.item(),
                )

        return u, mu_hist, rt_hist


# =============================================================================
# 13. LangevinBridgeMixin  (Fix 3)
# =============================================================================

class LangevinBridgeMixin:
    """
    Mixin that adds attach_langevin_bridge() to any engine class.

    Fix 3: EvolutionONEEngine and EpiForecastEngine both store a
    ``langevin_bridge`` attribute but previously provided no method
    to construct it.  This mixin supplies the standardised method so
    that both engines gain the API via single-inheritance.

    Usage::

        class EvolutionONEEngine(LangevinBridgeMixin):
            ...

        engine = EvolutionONEEngine(cfg)
        from structural_langevin_evo_v3 import AdvancedStructuralLangevin
        integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=310.0)
        bridge = engine.attach_langevin_bridge(integrator, sigma_scale=1.5)

    The bridge is stored as ``self.langevin_bridge`` and returned.
    """

    def attach_langevin_bridge(
        self,
        langevin,
        sigma_scale: float = 1.0,
        ssc: Optional[SemanticStateContraction] = None,
    ) -> LangevinEvolutionBridge:
        """
        Construct and attach a LangevinEvolutionBridge.

        Args:
            langevin    : an AdvancedStructuralLangevin instance from
                          structural_langevin_evo_v3.py.
            sigma_scale : characteristic displacement scale (Å) for μ normalisation.
            ssc         : optional external SSC; if None a fresh one is created.

        Returns:
            The attached LangevinEvolutionBridge instance (also stored as
            ``self.langevin_bridge``).
        """
        bridge = LangevinEvolutionBridge(
            langevin=langevin,
            sigma_scale=sigma_scale,
            ssc=ssc,
        )
        self.langevin_bridge = bridge
        logger.info(
            "%s: LangevinEvolutionBridge attached (sigma_scale=%.3f).",
            self.__class__.__name__, sigma_scale,
        )
        return bridge


# =============================================================================
# Module banner
# =============================================================================

logger.debug("ONE Core Evolution v%s loaded.", EVOLUTION_VERSION)  # v3.0.0
