# =============================================================================
# ONE CORE EVOLUTION — Shared Foundation for the EVOLUTION ONE Ecosystem
# =============================================================================
# Developer : Yoon A Limsuwan / MSPS NETWORK
# License   : MIT
# Year      : 2026
# ORCID     : 0009-0008-2374-0788
# GitHub    : yoonalimsuwan
#
# Single source of truth for components shared across:
#   evolution_one.py                         — cancer evolution engine
#   evolution_one_epidemiological_viral.py   — epidemiological forecasting
#   structural_langevin.py                   — BAOAB Langevin MD integrator
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
#   DifferentiableRG            — fully differentiable learnable RG smoother
#   CheckpointManager           — unified save/load (replaces duplicates)
#   LangevinEvolutionBridge     — connects AdvancedStructuralLangevin
#                                 to EvolutionaryClassifier / EpiForecastEngine
#   get_device                  — unified hardware-backend selector
#   EVOLUTION_VERSION           — ecosystem-wide version string
#
# Changes v1 → v2  (Native Full Differentiability)
#   Bug 1 — DifferentiableSOC: std() + clamp(0,10) → logsumexp spread +
#            softplus floor (gradient everywhere)
#   Bug 2 — DifferentiableRG:  weight/sum → F.softmax (smooth normalisation)
#   Bug 3 — DifferentiableIto: clamp(0,1) → softplus floor + sigmoid ceiling
#   Bug 7 — Added EpiEvolutionBridge: EpiForecastEngine ↔ EvolutionONEEngine
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

EVOLUTION_VERSION: str = "2.0.0"


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

    Canonical implementation shared across the EVOLUTION ONE ecosystem.

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
    subclasses (``CSOCThermostat``, ``SOCController``, etc.) share
    consistent logic.

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
# 5. Differentiable RG smoother (shared, fully differentiable)
# =============================================================================

class DifferentiableRG(nn.Module):
    """
    Fully differentiable learnable 1-D RG smoothing kernel.

    Replaces the non-differentiable ``np.convolve`` / ``DiffRGRefiner``
    used in prototype versions of both ``EvolutionaryClassifier`` and
    ``EpiForecastEngine``.

    The kernel weights are learnable parameters (``nn.Parameter``) so the
    smoother can be end-to-end trained together with classifiers.

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

        # Bug 2 fix: softmax normalisation — smooth, gradient everywhere
        # (replaces weight / (sum + 1e-8) which has unstable gradient at sum≈0)
        w = F.softmax(self.weight, dim=0)
        w = w.view(1, 1, -1)            # (1, 1, K)

        # x shape: (B, T) → add channel dim → (B, 1, T)
        out = F.conv1d(x.unsqueeze(1), w, padding=self.padding)  # (B, 1, T)
        out = out.squeeze(1)            # (B, T)

        if squeeze:
            out = out.squeeze(0)        # (T,)
        return out


# =============================================================================
# 6. Differentiable SOC dynamics (shared)
# =============================================================================

class DifferentiableSOC(nn.Module):
    """
    Fully differentiable SOC temperature modulation.

    Replaces manual ``torch.std`` + ``torch.clamp`` loops in
    ``EvolutionaryClassifier.soc_evolve`` with a proper ``nn.Module``
    whose parameters can be gradient-trained.

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
        """
        Evolve a mutation-load / Rt tensor via SOC dynamics.

        Bug 1 fix: fully differentiable everywhere —
          • std() replaced with logsumexp-based soft spread estimator
          • clamp(0, 10) replaced with softplus floor + soft sigmoid ceiling
            so gradients flow even when x is outside [0, 10].

        Args:
            x     : (...) float tensor of mutation loads or Rt values.
            steps : override ``self.n_steps``.
        Returns:
            Evolved tensor, same shape.
        """
        n = steps if steps is not None else self.n_steps
        for _ in range(n):
            # Bug 1a: differentiable spread via logsumexp (replaces std())
            tau = 0.1
            xf  = x.reshape(-1)
            spread = (tau * torch.logsumexp(xf / tau, dim=0)
                      - tau * torch.logsumexp(-xf / tau, dim=0))
            T = self.base_temp * (1.0 + self.beta * (spread - 1.0))
            # Ensure T > 0 via softplus
            T = F.softplus(T, beta=50.0)
            scale = 1.0 + 0.01 * (T / (self.base_temp + 1e-8) - 1.0)
            x = x * scale
            # Bug 1b: differentiable floor + ceiling (replaces clamp)
            x = F.softplus(x, beta=50.0)                   # floor at 0
            x = 10.0 - F.softplus(10.0 - x, beta=50.0)    # soft ceiling at 10
        return x


# =============================================================================
# 7. Differentiable Itô evolution (shared, replaces ito_evolve)
# =============================================================================

class DifferentiableIto(nn.Module):
    """
    Fully differentiable Itô SDE integrator for scalar state evolution.

    Replaces ``EvolutionaryClassifier.ito_evolve`` which instantiated
    a non-shared ``LangevinDynamics`` object internally and returned
    detached tensors.

    The double-well energy E(x) = ½(x−0.5)² + 0.1 sin(2πx) is
    implemented as a differentiable function, so gradients flow
    through the entire trajectory for end-to-end training.

    For connection to ``AdvancedStructuralLangevin`` (BAOAB), use
    :class:`LangevinEvolutionBridge` instead.

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
        """
        Euler-Maruyama integration of dx = −∇E dt + √(2 k_B T dt) dW.

        Fully differentiable — gradients flow through T_param and
        through the deterministic drift.

        Args:
            x0    : initial state tensor (any shape).
            steps : override ``self.n_steps``.
        Returns:
            Final state tensor, same shape as x0.
        """
        n = steps if steps is not None else self.n_steps
        x = x0
        noise_scale = torch.sqrt(2.0 * self.kb * self.T_param * self.dt)

        for _ in range(n):
            x_d = x.detach().requires_grad_(True)
            with torch.enable_grad():
                E = self._energy(x_d).sum()
                force = -torch.autograd.grad(E, x_d, create_graph=False)[0]

            # Euler-Maruyama step (deterministic part differentiable via T_param)
            x = x + force * self.dt + noise_scale * torch.randn_like(x)
            # Bug 3 fix: differentiable soft clamp to [0, 1]
            # softplus floor at 0, sigmoid ceiling at 1
            x = F.softplus(x, beta=100.0)                  # floor at 0
            x = torch.sigmoid(x * 6.0 - 3.0)              # smooth S-curve ∈ (0,1)
        return x


# =============================================================================
# 8. CheckpointManager (canonical, replaces duplicates in both engine files)
# =============================================================================

class CheckpointManager:
    """
    Unified checkpoint save / load for all EVOLUTION ONE engines.

    Previously defined identically in both ``evolution_one.py`` and
    ``evolution_one_epidemiological_viral.py`` — now lives here.
    """

    @staticmethod
    def save(filepath: str, data: Dict[str, Any]) -> None:
        """
        Serialise *data* to *filepath* using pickle.

        Args:
            filepath : destination path (will create parent dirs).
            data     : any pickle-serialisable dict.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Checkpoint saved → %s", filepath)

    @staticmethod
    def load(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.  Returns ``None`` if the file does not exist.

        Args:
            filepath : path written by :meth:`save`.
        """
        if not os.path.exists(filepath):
            logger.warning("Checkpoint not found: %s", filepath)
            return None
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.info("Checkpoint loaded ← %s", filepath)
        return data


# =============================================================================
# 9. LangevinEvolutionBridge
# =============================================================================

class LangevinEvolutionBridge:
    """
    Bridge connecting :class:`structural_langevin.AdvancedStructuralLangevin`
    to the population-level dynamics in ``evolution_one.py`` and
    ``evolution_one_epidemiological_viral.py``.

    Physical interpretation
    ───────────────────────
    The BAOAB Langevin integrator describes **individual-cell** (or
    individual-virion) conformational dynamics.  The bridge maps these
    micro-scale coordinates to a **macro-scale mutation-load scalar** μ
    (or effective reproduction number Rt) that feeds into the
    population-level classifiers.

    Mapping:
        μ = mean per-atom displacement / σ_scale
            → clipped to [0, 1]

    This is analogous to a coarse-graining step: fast atomic fluctuations
    are contracted to a slow collective variable via the SSC filter.

    Usage::

        from one_core_evolution import LangevinEvolutionBridge
        from structural_langevin import AdvancedStructuralLangevin

        integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=310.0)
        bridge = LangevinEvolutionBridge(integrator, sigma_scale=1.0)

        # Micro-step: evolve atomic coordinates
        coords, velocities = bridge.micro_step(coords, velocities, force_fn)

        # Macro projection: get population-level μ
        mu = bridge.project_to_mu(coords, coords_ref)

        # Feed into EvolutionaryClassifier or EpidemicClassifier
        state = classifier.mu_to_state(mu.item())

    Args:
        langevin    : an ``AdvancedStructuralLangevin`` instance.
        sigma_scale : characteristic displacement scale for normalisation (Å).
        ssc         : optional SemanticStateContraction for μ smoothing.
    """

    def __init__(
        self,
        langevin,
        sigma_scale: float = 1.0,
        ssc: Optional[SemanticStateContraction] = None,
    ) -> None:
        self.langevin    = langevin
        self.sigma_scale = max(sigma_scale, 1e-8)
        self.ssc         = ssc or SemanticStateContraction(epsilon_fp=0.0028)
        self._velocities: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset Langevin state and SSC filter."""
        self._velocities = None
        self.langevin.reset()
        self.ssc.reset()

    # ------------------------------------------------------------------

    def micro_step(
        self,
        coords:     torch.Tensor,
        velocities: Optional[torch.Tensor],
        force_fn:   Callable,
        jumps:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one BAOAB micro-step using ``AdvancedStructuralLangevin``.

        Args:
            coords     : (N, 3) atomic coordinates (Å).
            velocities : (N, 3) or None → zero-initialised.
            force_fn   : callable(coords) → (energy, force).
            jumps      : (N, 3) BV jump vectors or None.

        Returns:
            new_coords, new_velocities (both detached).
        """
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

    # ------------------------------------------------------------------

    def project_to_mu(
        self,
        coords:     torch.Tensor,
        coords_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project atomic displacements to a scalar mutation-load μ ∈ [0, 1].

        Applies SSC low-pass filtering for temporal smoothness.

        Args:
            coords     : (N, 3) current positions.
            coords_ref : (N, 3) reference (e.g. wild-type) positions.

        Returns:
            Scalar μ tensor ∈ [0, 1] (differentiable w.r.t. coords).
        """
        raw_disp = torch.norm(coords - coords_ref, dim=-1).mean()
        raw_mu   = raw_disp / self.sigma_scale
        mu_ssc   = self.ssc(raw_mu)
        return torch.sigmoid(mu_ssc - 1.0)   # soft clamp to (0, 1)

    # ------------------------------------------------------------------

    def run(
        self,
        coords:     torch.Tensor,
        coords_ref: torch.Tensor,
        force_fn:   Callable,
        n_steps:    int,
        jumps:      Optional[torch.Tensor] = None,
        log_every:  int = 50,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Run ``n_steps`` of Langevin MD and record μ trajectory.

        Args:
            coords     : (N, 3) initial atomic positions.
            coords_ref : (N, 3) reference positions for μ projection.
            force_fn   : callable(coords) → (energy, force).
            n_steps    : number of integration steps.
            jumps      : (N, 3) BV jump vectors or None.
            log_every  : log diagnostic every this many steps.

        Returns:
            final_coords : (N, 3) final positions.
            mu_history   : list of float μ values (length n_steps).
        """
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
# 10. EpiEvolutionBridge  (Bug 7 fix)
# =============================================================================

class EpiEvolutionBridge:
    """
    Bridge connecting ``EpiForecastEngine`` (epidemiological)
    and ``EvolutionONEEngine`` (cancer/genomic evolution).

    Motivation  (Bug 7 fix)
    ───────────────────────
    Both engines independently compute a scalar state signal:
      • ``EpiForecastEngine``  → Rt (effective reproduction number)
      • ``EvolutionONEEngine`` → μ  (mutation load / fitness shift)

    These two quantities are physically linked: a high mutation load
    (e.g. immune-escape variant) increases Rt.  This bridge provides a
    differentiable soft coupling via SSC filtering so that μ can modulate
    Rt (and vice-versa) in a jointly trainable pipeline.

    Coupling equation (one direction — μ → Rt):
        Rt_coupled = Rt_base + scale · σ(μ_smoothed − μ_threshold)

    Usage::

        bridge = EpiEvolutionBridge(
            epi_engine=epi, evo_engine=evo, mu_to_rt_scale=0.5)
        rt_coupled = bridge.mu_to_rt(mu_tensor, rt_base_tensor)
        mu_from_rt = bridge.rt_to_mu(rt_tensor, mu_base_tensor)

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

        Rt_coupled = Rt_base + scale · σ((μ_ssc − μ_threshold) · 10)

        Fully differentiable w.r.t. both mu and rt_base.

        Args:
            mu      : scalar or (T,) mutation-load tensor.
            rt_base : scalar or (T,) baseline Rt tensor.
        Returns:
            rt_coupled : same shape as rt_base.
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

        μ_coupled = μ_base + scale · σ((Rt_ssc − Rt_threshold) · 5)

        Args:
            rt      : scalar or (T,) Rt tensor.
            mu_base : scalar or (T,) baseline μ tensor.
        Returns:
            mu_coupled : same shape as mu_base.
        """
        rt_ssc = self._ssc_rt(rt.mean() if rt.dim() > 0 else rt)
        boost  = torch.sigmoid((rt_ssc - self.rt_threshold) * 5.0)
        return mu_base + self.rt_to_mu_scale * boost

# =============================================================================
# Module banner
# =============================================================================

logger.debug("ONE Core Evolution v%s loaded.", EVOLUTION_VERSION)  # v2.0.0
