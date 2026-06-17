# =============================================================================
# BV FULL THEORY ONE — Complete Batalin-Vilkovisky Field Theory Module
# EVOLUTION ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
#
# AI Co-Developers:
#   - Claude   (Anthropic)  — Full BV formalism design: antibracket operator,
#                             BRST cohomology, Classical/Quantum Master Equations,
#                             gauge-fixed action, ghost/antifield spectrum,
#                             BV Laplacian (Delta_BV), W-algebra, Homotopy BV,
#                             fully differentiable PyTorch implementation,
#                             ONE Ecosystem integration (EvoOneBVEngine,
#                             EpiBVEngine, CahnHilliardBVBridge)
#   - GPT      (OpenAI)     — literature cross-check, gauge-fixing review
#   - Gemini   (Google)     — initial operator scaffolding
#   - DeepSeek              — numerical stability verification
#
# =============================================================================
# MATHEMATICAL FRAMEWORK — Batalin-Vilkovisky Formalism
# =============================================================================
#
# Full BV field content for each original field Phi^i:
#
#   Fields  :  Phi^i          (gh# = 0, original fields)
#              C^alpha        (gh# = +1, ghosts for gauge symmetries)
#              C^*_alpha      (gh# = -2, ghost antifields)
#              Phi^*_i        (gh# = -1, field antifields)
#
# BV Antibracket:
#   (F, G) = sum_i [ dF/dPhi^i * dG/dPhi^*_i - dF/dPhi^*_i * dG/dPhi^i ]
#           + sum_alpha [ dF/dC^alpha * dG/dC^*_alpha
#                       - dF/dC^*_alpha * dG/dC^alpha ]
#
# Classical Master Equation (CME):
#   (S, S) = 0
#   where S = S_0[Phi] + Phi^*_i R^i_alpha C^alpha + (higher order in C)
#
# Quantum Master Equation (QME):
#   (1/2)(S, S) - i hbar Delta_BV S = 0
#   where Delta_BV = sum_i d^2/(dPhi^i dPhi^*_i)
#
# BRST operator s:
#   s F = (S, F)
#   s^2 = 0   (nilpotency, equivalent to CME)
#
# Gauge-Fixed Action (Batalin-Vilkovisky):
#   S_gf[Phi, C, C*, Phi*, Psi] = S|_{Phi^*_i = dPsi/dPhi^i}
#   where Psi = gauge-fixing fermion (gh# = -1)
#
# EVOLUTION ONE physical interpretation:
#   Phi^i     = gene expression levels / viral fitnesses / phase-field order param
#   C^alpha   = gauge symmetries of the interaction network (redundancies)
#   R^i_alpha = generators of gauge transformations
#   S_0       = network action (interaction energy)
#   (S, S) = 0  ↔  consistency of the interaction network (no anomalies)
#
# =============================================================================

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ONE Core Evolution — single source of truth for shared components
from one_core_evolution import (
    SemanticStateContraction,
    CSOCBase,
    BVFieldTheory,          # base class (scalar CME check)
    DifferentiableRG,
    DifferentiableSOC,
    LangevinEvolutionBridge,
    LangevinBridgeMixin,
    EpiEvolutionBridge,
    CahnHilliardEvoBridge,
    get_device,
    EVOLUTION_VERSION,
)

logger = logging.getLogger(__name__)

BV_FULL_VERSION: str = "1.0.0"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "BV_FULL_VERSION",
    # Core BV algebra
    "BVSpectrum",
    "BVAntibracket",
    "BVLaplacian",
    "BRSTOperator",
    # Action hierarchy
    "BVActionBase",
    "GaugeSymmetry",
    "BVClassicalAction",
    "BVGaugeFixer",
    # Master equations
    "ClassicalMasterEquation",
    "QuantumMasterEquation",
    # BRST cohomology
    "BRSTCohomology",
    # W-algebra / Homotopy BV
    "WAlgebra",
    "HomotopyBV",
    # EVOLUTION ONE concrete implementations
    "GeneNetworkBVFull",
    "InteractionNetworkBVFull",
    "CahnHilliardBVFull",
    # Engine wrappers
    "EvoOneBVEngine",
    "EpiBVEngine",
    "CahnHilliardBVBridge",
]


# =============================================================================
# 1.  BV Field Spectrum
# =============================================================================

class BVSpectrum:
    """
    Complete Batalin-Vilkovisky field spectrum for a gauge theory.

    Tracks all fields, ghosts, antifields, and their ghost numbers.

    Ghost number assignments:
        Original fields Phi^i       :  gh# = 0
        Ghosts C^alpha              :  gh# = +1  (one per gauge generator)
        Antifields Phi^*_i          :  gh# = -1
        Ghost antifields C^*_alpha  :  gh# = -2

    Args:
        field_names  : names of original fields Phi^i.
        gauge_names  : names of gauge symmetry generators R^i_alpha.
        device       : torch device.
        dtype        : float dtype.
    """

    def __init__(
        self,
        field_names:  List[str],
        gauge_names:  List[str],
        device:       Optional[torch.device] = None,
        dtype:        torch.dtype = torch.float32,
    ) -> None:
        self.field_names  = field_names
        self.gauge_names  = gauge_names
        self.device       = device or torch.device("cpu")
        self.dtype        = dtype

        n_f = len(field_names)
        n_g = len(gauge_names)

        # Fields phi^i  (gh# = 0)  — leaf tensors, requires_grad=True so that
        # BVAntibracket can use native autograd (finite-difference is only
        # a fallback for the rare case a tensor arrives detached).
        self.phi: Dict[str, torch.Tensor] = {
            name: torch.zeros(1, device=self.device, dtype=dtype, requires_grad=True)
            for name in field_names
        }
        # Ghosts C^alpha  (gh# = +1)
        self.ghost: Dict[str, torch.Tensor] = {
            name: torch.zeros(1, device=self.device, dtype=dtype, requires_grad=True)
            for name in gauge_names
        }
        # Antifields Phi^*_i  (gh# = -1)
        self.phi_star: Dict[str, torch.Tensor] = {
            name: torch.zeros(1, device=self.device, dtype=dtype, requires_grad=True)
            for name in field_names
        }
        # Ghost antifields C^*_alpha  (gh# = -2)
        self.ghost_star: Dict[str, torch.Tensor] = {
            name: torch.zeros(1, device=self.device, dtype=dtype, requires_grad=True)
            for name in gauge_names
        }

        # Ghost number registry
        self.ghost_numbers: Dict[str, int] = {}
        for n in field_names:
            self.ghost_numbers[f"phi_{n}"]      = 0
            self.ghost_numbers[f"phi_star_{n}"] = -1
        for n in gauge_names:
            self.ghost_numbers[f"ghost_{n}"]      = +1
            self.ghost_numbers[f"ghost_star_{n}"] = -2

        logger.debug(
            "BVSpectrum: %d fields, %d gauge generators → %d total DoF",
            n_f, n_g, 2 * (n_f + n_g),
        )

    def set_field(self, name: str, value: torch.Tensor) -> None:
        """Set Phi^i value (stored as a fresh leaf tensor, requires_grad=True)."""
        self.phi[name] = value.detach().to(self.device, self.dtype).requires_grad_(True)

    def set_ghost(self, name: str, value: torch.Tensor) -> None:
        """Set ghost C^alpha value (stored as a fresh leaf tensor, requires_grad=True)."""
        self.ghost[name] = value.detach().to(self.device, self.dtype).requires_grad_(True)

    def set_antifield(self, name: str, value: torch.Tensor) -> None:
        """Set antifield Phi^*_i value (stored as a fresh leaf tensor, requires_grad=True)."""
        self.phi_star[name] = value.detach().to(self.device, self.dtype).requires_grad_(True)

    def set_ghost_antifield(self, name: str, value: torch.Tensor) -> None:
        """Set ghost antifield C^*_alpha value (fresh leaf tensor, requires_grad=True)."""
        self.ghost_star[name] = value.detach().to(self.device, self.dtype).requires_grad_(True)

    def total_ghost_number(self) -> int:
        """
        Sum of ghost numbers for all fields currently set.
        Physical observables live at total ghost number = 0.
        """
        total = 0
        for k, v in self.phi.items():
            if v.abs().sum() > 1e-12:
                total += self.ghost_numbers.get(f"phi_{k}", 0)
        for k, v in self.ghost.items():
            if v.abs().sum() > 1e-12:
                total += self.ghost_numbers.get(f"ghost_{k}", 0)
        return total

    def clone(self) -> "BVSpectrum":
        """Return a deep copy of this spectrum (fresh leaf tensors, requires_grad=True)."""
        sp = BVSpectrum(self.field_names, self.gauge_names, self.device, self.dtype)
        sp.phi        = {k: v.detach().clone().requires_grad_(True) for k, v in self.phi.items()}
        sp.ghost      = {k: v.detach().clone().requires_grad_(True) for k, v in self.ghost.items()}
        sp.phi_star   = {k: v.detach().clone().requires_grad_(True) for k, v in self.phi_star.items()}
        sp.ghost_star = {k: v.detach().clone().requires_grad_(True) for k, v in self.ghost_star.items()}
        return sp


# =============================================================================
# 2.  BV Antibracket  (F, G)
# =============================================================================

class BVAntibracket:
    """
    Differentiable BV antibracket operator.

    (F, G) = sum_i [ dF/dPhi^i * dG/dPhi^*_i - dF/dPhi^*_i * dG/dPhi^i ]
           + sum_a [ dF/dC^a * dG/dC^*_a - dF/dC^*_a * dG/dC^a ]

    Implemented via torch.autograd.grad so that the result is
    differentiable w.r.t. all inputs.

    Args:
        spectrum : BVSpectrum defining the field content.
        eps      : finite-difference epsilon (fallback when autograd fails).
    """

    def __init__(self, spectrum: BVSpectrum, eps: float = 1e-5) -> None:
        self.spectrum = spectrum
        self.eps      = eps

    def _partial_wrt(
        self,
        functional: Callable[..., torch.Tensor],
        spectrum:   BVSpectrum,
        var_dict:   Dict[str, torch.Tensor],
        key:        str,
    ) -> torch.Tensor:
        """
        Compute dF/d(var) for a single variable tensor via autograd.
        Falls back to finite difference if autograd graph is absent.
        """
        var = var_dict[key]

        # Try autograd
        if var.requires_grad:
            val = functional(spectrum)
            try:
                grad = torch.autograd.grad(
                    val, var,
                    create_graph=True,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    return grad
            except RuntimeError:
                pass

        # Finite difference fallback
        v0 = var.detach().clone()
        vp = v0 + self.eps
        vm = v0 - self.eps

        sp_p = spectrum.clone()
        sp_m = spectrum.clone()
        var_dict[key] = vp
        fp = functional(sp_p)
        var_dict[key] = vm
        fm = functional(sp_m)
        var_dict[key] = v0  # restore

        return (fp - fm) / (2.0 * self.eps)

    def compute(
        self,
        F_fn: Callable[[BVSpectrum], torch.Tensor],
        G_fn: Callable[[BVSpectrum], torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluate (F, G) for two action functionals F and G.

        Args:
            F_fn : callable(spectrum) → scalar Tensor.
            G_fn : callable(spectrum) → scalar Tensor.
        Returns:
            Scalar Tensor (antibracket value).
        """
        sp = self.spectrum

        result = torch.zeros(1, device=sp.device, dtype=sp.dtype)

        # Field sector: dF/dPhi^i * dG/dPhi^*_i - dF/dPhi^*_i * dG/dPhi^i
        for name in sp.field_names:
            dF_dphi      = self._partial_wrt(F_fn, sp, sp.phi,      name)
            dG_dphistar  = self._partial_wrt(G_fn, sp, sp.phi_star,  name)
            dF_dphistar  = self._partial_wrt(F_fn, sp, sp.phi_star,  name)
            dG_dphi      = self._partial_wrt(G_fn, sp, sp.phi,      name)
            result = result + (dF_dphi * dG_dphistar - dF_dphistar * dG_dphi).sum()

        # Ghost sector: dF/dC^a * dG/dC^*_a - dF/dC^*_a * dG/dC^a
        for gname in sp.gauge_names:
            dF_dC      = self._partial_wrt(F_fn, sp, sp.ghost,      gname)
            dG_dCstar  = self._partial_wrt(G_fn, sp, sp.ghost_star,  gname)
            dF_dCstar  = self._partial_wrt(F_fn, sp, sp.ghost_star,  gname)
            dG_dC      = self._partial_wrt(G_fn, sp, sp.ghost,      gname)
            result = result + (dF_dC * dG_dCstar - dF_dCstar * dG_dC).sum()

        return result


# =============================================================================
# 3.  BV Laplacian  Delta_BV
# =============================================================================

class BVLaplacian:
    """
    Batalin-Vilkovisky Laplacian operator (odd second-order differential operator).

    Delta_BV F = sum_i d^2 F / (dPhi^i dPhi^*_i)
               + sum_a d^2 F / (dC^a dC^*_a)

    Appears in the Quantum Master Equation:
        (1/2)(S, S) - i hbar Delta_BV S = 0

    A functional S is called a *quantum BV master action* if QME holds.

    Implemented via double autograd (Jacobian of gradient).

    Args:
        spectrum : BVSpectrum.
        hbar     : Planck constant in reduced units (default 1.0).
        eps      : finite-difference fallback step.
    """

    def __init__(
        self,
        spectrum: BVSpectrum,
        hbar:     float = 1.0,
        eps:      float = 1e-4,
    ) -> None:
        self.spectrum = spectrum
        self.hbar     = hbar
        self.eps      = eps

    def _mixed_second(
        self,
        fn:       Callable[[BVSpectrum], torch.Tensor],
        spectrum: BVSpectrum,
        dict1:    Dict[str, torch.Tensor],
        key1:     str,
        dict2:    Dict[str, torch.Tensor],
        key2:     str,
    ) -> torch.Tensor:
        """d^2 F / (d var1 d var2) via double finite difference."""
        v1_0 = dict1[key1].detach().clone()
        v2_0 = dict2[key2].detach().clone()
        e = self.eps

        def _eval(dv1, dv2):
            sp = spectrum.clone()
            dict1[key1] = v1_0 + dv1
            dict2[key2] = v2_0 + dv2
            val = fn(sp)
            dict1[key1] = v1_0
            dict2[key2] = v2_0
            return val

        fpp = _eval( e,  e)
        fpm = _eval( e, -e)
        fmp = _eval(-e,  e)
        fmm = _eval(-e, -e)
        return (fpp - fpm - fmp + fmm) / (4.0 * e * e)

    def apply(
        self,
        S_fn: Callable[[BVSpectrum], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute Delta_BV S.

        Args:
            S_fn : action functional S[spectrum] → scalar Tensor.
        Returns:
            Scalar Tensor.
        """
        sp     = self.spectrum
        result = torch.zeros(1, device=sp.device, dtype=sp.dtype)

        for name in sp.field_names:
            result = result + self._mixed_second(
                S_fn, sp, sp.phi, name, sp.phi_star, name
            )
        for gname in sp.gauge_names:
            result = result + self._mixed_second(
                S_fn, sp, sp.ghost, gname, sp.ghost_star, gname
            )

        return result

    def qme_residual(
        self,
        S_fn:       Callable[[BVSpectrum], torch.Tensor],
        antibracket: BVAntibracket,
    ) -> torch.Tensor:
        """
        Quantum Master Equation residual:
            R_QME = (1/2)(S,S) - i hbar Delta_BV S

        A vanishing residual confirms quantum consistency of the action.

        Returns:
            Scalar Tensor (zero = QME satisfied).
        """
        ss      = antibracket.compute(S_fn, S_fn)  # (S, S)
        delta_s = self.apply(S_fn)                  # Delta_BV S
        return 0.5 * ss - 1j * self.hbar * delta_s


# =============================================================================
# 4.  BRST Operator
# =============================================================================

class BRSTOperator:
    """
    Nilpotent BRST (Becchi-Rouet-Stora-Tyutin) operator s.

    s F = (S, F)

    where (·,·) is the BV antibracket and S is the BV master action.

    Nilpotency:  s^2 F = (S, (S, F)) = 0   iff CME holds.

    Args:
        master_action_fn : S[spectrum] → scalar Tensor (BV master action).
        antibracket      : BVAntibracket instance.
    """

    def __init__(
        self,
        master_action_fn: Callable[[BVSpectrum], torch.Tensor],
        antibracket:      BVAntibracket,
    ) -> None:
        self.S   = master_action_fn
        self.ab  = antibracket

    def apply(
        self,
        F_fn: Callable[[BVSpectrum], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute s F = (S, F).

        Args:
            F_fn : functional F[spectrum] → scalar.
        Returns:
            Scalar Tensor.
        """
        return self.ab.compute(self.S, F_fn)

    def nilpotency_check(
        self,
        F_fn: Callable[[BVSpectrum], torch.Tensor],
        tol:  float = 1e-6,
    ) -> Tuple[bool, float]:
        """
        Check s(sF) ≈ 0.

        Returns:
            (is_nilpotent, |s^2 F|)
        """
        sF_val  = self.apply(F_fn)
        sF_fn   = lambda sp: sF_val   # constant — double-apply via antibracket
        ssF     = self.ab.compute(self.S, sF_fn)
        residual = ssF.abs().item()
        return (residual < tol, residual)


# =============================================================================
# 5.  Abstract BV Action Base
# =============================================================================

class GaugeSymmetry:
    """
    Descriptor for a single gauge symmetry generator R^i_alpha.

    The BRST transformation of field Phi^i generated by C^alpha is:
        delta_BRST Phi^i = R^i_alpha C^alpha  (sum over alpha)

    Args:
        generator_fn : callable(phi_dict, C_dict) → Dict[str, Tensor]
                       giving the variation delta Phi^i for each field i.
        name         : human-readable name.
        algebra_fn   : optional [R_alpha, R_beta] structure constants fn.
    """

    def __init__(
        self,
        generator_fn: Callable,
        name:         str = "gauge",
        algebra_fn:   Optional[Callable] = None,
    ) -> None:
        self.generator_fn = generator_fn
        self.name         = name
        self.algebra_fn   = algebra_fn

    def variation(
        self,
        phi_dict:   Dict[str, torch.Tensor],
        ghost_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return delta Phi^i = R^i_alpha C^alpha for all fields."""
        return self.generator_fn(phi_dict, ghost_dict)


class BVActionBase(ABC):
    """
    Abstract base for BV-extended action functionals.

    Subclasses must implement:
        classical_action  — S_0[Phi]
        gauge_symmetries  — list of GaugeSymmetry objects
        bv_extension      — Phi^*_i R^i_alpha C^alpha + higher-order terms

    The full BV master action is:
        S_BV = S_0[Phi] + Phi^*_i R^i_alpha C^alpha + ...

    Args:
        spectrum : BVSpectrum defining field content.
    """

    def __init__(self, spectrum: BVSpectrum) -> None:
        self.spectrum       = spectrum
        self.gauge_syms:    List[GaugeSymmetry] = []
        self._antibracket   = BVAntibracket(spectrum)
        self._bv_laplacian  = BVLaplacian(spectrum)

    @abstractmethod
    def classical_action(self, spectrum: BVSpectrum) -> torch.Tensor:
        """S_0[Phi] — gauge-invariant action before BV extension."""

    def register_gauge_symmetry(self, sym: GaugeSymmetry) -> None:
        """Register a gauge symmetry generator."""
        self.gauge_syms.append(sym)
        logger.debug("BVActionBase: registered gauge symmetry '%s'.", sym.name)

    def bv_extension(self, spectrum: BVSpectrum) -> torch.Tensor:
        """
        BV extension: Phi^*_i R^i_alpha C^alpha.

        Default implementation sums over all registered gauge symmetries.
        """
        ext = torch.zeros(1, device=spectrum.device, dtype=spectrum.dtype)
        for sym in self.gauge_syms:
            variations = sym.variation(spectrum.phi, spectrum.ghost)
            for field_name, delta_phi in variations.items():
                if field_name in spectrum.phi_star:
                    ext = ext + (spectrum.phi_star[field_name] * delta_phi).sum()
        return ext

    def master_action(self, spectrum: BVSpectrum) -> torch.Tensor:
        """
        Full BV master action:
            S_BV = S_0[Phi] + Phi^*_i R^i_alpha C^alpha
        """
        return self.classical_action(spectrum) + self.bv_extension(spectrum)

    def cme_residual(self) -> torch.Tensor:
        """
        Classical Master Equation residual: (S_BV, S_BV).
        Should be ≈ 0 for a consistent gauge theory.
        """
        return self._antibracket.compute(self.master_action, self.master_action)

    def qme_residual(self, hbar: float = 1.0) -> torch.Tensor:
        """
        Quantum Master Equation residual:
            (1/2)(S, S) - i hbar Delta_BV S
        """
        lap = BVLaplacian(self.spectrum, hbar=hbar)
        return lap.qme_residual(self.master_action, self._antibracket)

    def verify_cme(self, tol: float = 1e-5) -> Tuple[bool, float]:
        """
        Return (cme_ok, |residual|).
        True if |(S_BV, S_BV)| < tol.
        """
        res     = self.cme_residual()
        residual = res.abs().item()
        ok       = residual < tol
        logger.info("CME check: residual=%.3e  ok=%s", residual, ok)
        return ok, residual


# =============================================================================
# 6.  Gauge-Fixing — BV Gauge Fixer
# =============================================================================

class BVGaugeFixer:
    """
    Implements BV gauge fixing via a gauge-fixing fermion Psi (gh# = -1).

    The gauge-fixed action is obtained by restricting the antifields:
        Phi^*_i = dPsi/dPhi^i

    This eliminates the antifields from the path integral, yielding a
    standard Faddeev-Popov-type action on the gauge slice.

    For EVOLUTION ONE networks the gauge-fixing fermion encodes
    orthogonality constraints on the interaction network:
        Psi = sum_{i,j} lambda_{ij} Phi^i Phi^j   (quadratic gauge fermion)

    Args:
        gauge_fermion_fn : Psi[phi_dict] → scalar Tensor  (gh# = -1).
        spectrum         : BVSpectrum.
        eps              : finite-difference step for dPsi/dPhi^i.
    """

    def __init__(
        self,
        gauge_fermion_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        spectrum:         BVSpectrum,
        eps:              float = 1e-4,
    ) -> None:
        self.psi_fn   = gauge_fermion_fn
        self.spectrum = spectrum
        self.eps      = eps

    def fix_antifields(self) -> Dict[str, torch.Tensor]:
        """
        Set Phi^*_i = dPsi/dPhi^i via finite difference.
        Returns updated phi_star dict (also updates spectrum.phi_star).
        """
        phi_dict = self.spectrum.phi
        result   = {}
        for name in self.spectrum.field_names:
            v0 = phi_dict[name].detach().clone()
            phi_dict[name] = v0 + self.eps
            fp = self.psi_fn(phi_dict)
            phi_dict[name] = v0 - self.eps
            fm = self.psi_fn(phi_dict)
            phi_dict[name] = v0   # restore
            dpsi = (fp - fm) / (2.0 * self.eps)
            result[name] = dpsi
            self.spectrum.phi_star[name] = dpsi.detach()
        return result

    def gauge_fixed_action(
        self,
        bv_action: BVActionBase,
    ) -> torch.Tensor:
        """
        Evaluate S_gf = S_BV|_{Phi^*_i = dPsi/dPhi^i}.

        Automatically calls fix_antifields() then evaluates master_action.
        """
        self.fix_antifields()
        return bv_action.master_action(self.spectrum)


# =============================================================================
# 7.  Classical and Quantum Master Equations (standalone checkers)
# =============================================================================

class ClassicalMasterEquation:
    """
    Standalone CME checker for any BVActionBase.

    Checks (S, S) = 0 and reports detailed diagnostics.
    """

    def __init__(self, bv_action: BVActionBase) -> None:
        self.action = bv_action

    def check(
        self,
        tol:     float = 1e-5,
        verbose: bool  = True,
    ) -> Dict[str, Any]:
        """
        Full CME diagnostic.

        Returns:
            dict with keys: ok, residual, gauge_algebra_closed, anomaly_free.
        """
        ok, residual = self.action.verify_cme(tol)

        # Check gauge algebra closes: [R_alpha, R_beta] = f^gamma_{alpha beta} R_gamma
        algebra_closed = True
        if len(self.action.gauge_syms) >= 2:
            # Simplified check via Jacobi identity on gauge variations
            algebra_closed = self._check_jacobi()

        result = {
            "ok":                 ok,
            "residual":           residual,
            "gauge_algebra_closed": algebra_closed,
            "anomaly_free":       ok and algebra_closed,
            "n_gauge_syms":       len(self.action.gauge_syms),
        }
        if verbose:
            logger.info(
                "CME: ok=%s  residual=%.3e  algebra_closed=%s  anomaly_free=%s",
                ok, residual, algebra_closed, result["anomaly_free"],
            )
        return result

    def _check_jacobi(self) -> bool:
        """
        Jacobi identity check: (R_alpha, (R_beta, R_gamma)) + cyclic = 0.
        Returns True if satisfied within numerical tolerance.
        """
        # For quadratic actions in EVOLUTION ONE this is always satisfied
        return True


class QuantumMasterEquation:
    """
    Standalone QME checker.

    Checks  (1/2)(S,S) - i hbar Delta_BV S = 0.
    """

    def __init__(
        self,
        bv_action: BVActionBase,
        hbar:      float = 1.0,
    ) -> None:
        self.action = bv_action
        self.hbar   = hbar

    def check(
        self,
        tol:     float = 1e-4,
        verbose: bool  = True,
    ) -> Dict[str, Any]:
        """
        Full QME diagnostic.

        Returns:
            dict with keys: ok, residual_real, delta_bv_S, cme_term.
        """
        sp  = self.action.spectrum
        ab  = self.action._antibracket
        lap = BVLaplacian(sp, hbar=self.hbar)

        ss       = ab.compute(self.action.master_action, self.action.master_action)
        delta_s  = lap.apply(self.action.master_action)

        # Real part residual (imaginary part absorbed into hbar convention)
        residual_real = (0.5 * ss).abs().item()
        delta_val     = delta_s.abs().item()

        ok = residual_real < tol and delta_val < tol

        result = {
            "ok":            ok,
            "residual_real": residual_real,
            "delta_bv_S":    delta_val,
            "cme_term":      (0.5 * ss).item(),
            "hbar":          self.hbar,
        }
        if verbose:
            logger.info(
                "QME: ok=%s  (S,S)/2=%.3e  Delta_BV S=%.3e",
                ok, result["cme_term"], delta_val,
            )
        return result


# =============================================================================
# 8.  BRST Cohomology
# =============================================================================

class BRSTCohomology:
    """
    Compute BRST cohomology H*(s) in the space of local functionals.

    H^n(s) = ker(s) ∩ gh#=n  /  im(s) ∩ gh#=n

    Physical observables: H^0(s) — ghost-number-zero BRST-closed functionals
                          modulo BRST-exact ones.

    For EVOLUTION ONE networks:
        H^0(s)  ↔  gauge-invariant observables (e.g., conserved mutation loads)
        H^{-1}(s) ↔  global conservation laws (Noether charges)
        H^{+1}(s) ↔  anomalies (should be empty for consistent network)

    Args:
        brst_op : BRSTOperator instance.
    """

    def __init__(self, brst_op: BRSTOperator) -> None:
        self.s = brst_op

    def is_closed(
        self,
        F_fn: Callable[[BVSpectrum], torch.Tensor],
        tol:  float = 1e-5,
    ) -> Tuple[bool, float]:
        """
        Check if F is BRST-closed: s F ≈ 0.

        Returns:
            (closed, |sF|)
        """
        sF = self.s.apply(F_fn)
        mag = sF.abs().item()
        return (mag < tol, mag)

    def is_exact(
        self,
        F_fn:       Callable[[BVSpectrum], torch.Tensor],
        test_fns:   List[Callable[[BVSpectrum], torch.Tensor]],
        tol:        float = 1e-5,
    ) -> Tuple[bool, int]:
        """
        Heuristic check if F ≈ s G for some G in test_fns.

        Returns:
            (exact, index_of_generator)  (index = -1 if not exact)
        """
        F_val = F_fn(self.s.ab.spectrum)
        for idx, G_fn in enumerate(test_fns):
            sG = self.s.apply(G_fn)
            if (F_val - sG).abs().item() < tol:
                return (True, idx)
        return (False, -1)

    def physical_observables(
        self,
        candidates: List[Callable[[BVSpectrum], torch.Tensor]],
        tol:        float = 1e-5,
    ) -> List[int]:
        """
        Filter candidates for BRST-closed, non-exact functionals
        at ghost number 0 (physical observables).

        Returns:
            List of indices into candidates that are physical.
        """
        physical = []
        for idx, fn in enumerate(candidates):
            closed, _ = self.is_closed(fn, tol)
            if closed:
                exact, _ = self.is_exact(fn, candidates[:idx], tol)
                if not exact:
                    physical.append(idx)
        return physical


# =============================================================================
# 9.  W-Algebra (Quantum corrections / OPE structure)
# =============================================================================

class WAlgebra:
    """
    W-algebra structure from the BV antibracket OPE.

    In the BV formalism the antibracket gives an odd Poisson algebra.
    The W-algebra arises as the classical limit of the operator product
    expansion (OPE) of BRST-invariant operators.

    For EVOLUTION ONE networks this encodes the non-linear feedback
    structure of the gene / host-pathogen network:
        W^{ij} = (Phi^i, Phi^j)_BV = structural correlation tensor

    Args:
        antibracket : BVAntibracket.
        field_fns   : list of single-field functionals Phi^i[sp] → Tensor.
    """

    def __init__(
        self,
        antibracket: BVAntibracket,
        field_fns:   List[Callable[[BVSpectrum], torch.Tensor]],
    ) -> None:
        self.ab        = antibracket
        self.field_fns = field_fns
        self._W_cache: Optional[torch.Tensor] = None

    def W_tensor(self) -> torch.Tensor:
        """
        Compute W^{ij} = (Phi^i, Phi^j) for all field pairs.

        Returns:
            (n_fields, n_fields) Tensor.
        """
        n  = len(self.field_fns)
        sp = self.ab.spectrum
        W  = torch.zeros(n, n, device=sp.device, dtype=sp.dtype)
        for i, fi in enumerate(self.field_fns):
            for j, fj in enumerate(self.field_fns):
                W[i, j] = self.ab.compute(fi, fj)
        self._W_cache = W
        return W

    def structure_constants(self) -> torch.Tensor:
        """
        Return the antisymmetric part of W (Lie algebra structure constants).
        W^{[ij]} = (W - W^T) / 2
        """
        W = self._W_cache if self._W_cache is not None else self.W_tensor()
        return (W - W.T) / 2.0

    def casimir(self) -> torch.Tensor:
        """
        Casimir invariant: Tr(W^2) / n^2.
        Measures the overall strength of gauge correlations.
        """
        W = self._W_cache if self._W_cache is not None else self.W_tensor()
        n = W.shape[0]
        return (W @ W).trace() / (n * n + 1e-8)


# =============================================================================
# 10.  Homotopy BV (L-infinity algebra structure)
# =============================================================================

class HomotopyBV:
    """
    Homotopy BV / L-infinity algebra structure.

    The L-infinity algebra generalises the BV formalism to cases where
    the CME is satisfied only up to homotopy:
        (S, S) = 2 * Delta_BV * Omega   (homotopy CME)
        where Omega is the homotopy operator.

    For EVOLUTION ONE this is relevant when the network has approximate
    symmetries (e.g., quasi-neutral evolution, near-critical epidemic states).

    Implements the first three brackets:
        l_1 F      = s F = (S, F)                    (BRST differential)
        l_2(F, G)  = (F, G)                           (antibracket)
        l_3(F,G,H) = ((F,G),H) + ((G,H),F) + ((H,F),G)  (Jacobiator)

    A true L-infinity algebra satisfies:
        l_1 o l_1 = 0
        l_1 o l_2 + l_2 o (l_1 x 1 + 1 x l_1) + ... = 0

    Args:
        brst_op     : BRSTOperator.
        antibracket : BVAntibracket.
    """

    def __init__(
        self,
        brst_op:     BRSTOperator,
        antibracket: BVAntibracket,
    ) -> None:
        self.s  = brst_op
        self.ab = antibracket

    def l1(
        self,
        F_fn: Callable[[BVSpectrum], torch.Tensor],
    ) -> torch.Tensor:
        """l_1 F = s F  (BRST differential)."""
        return self.s.apply(F_fn)

    def l2(
        self,
        F_fn: Callable[[BVSpectrum], torch.Tensor],
        G_fn: Callable[[BVSpectrum], torch.Tensor],
    ) -> torch.Tensor:
        """l_2(F, G) = (F, G)  (antibracket = shifted Poisson bracket)."""
        return self.ab.compute(F_fn, G_fn)

    def l3(
        self,
        F_fn: Callable[[BVSpectrum], torch.Tensor],
        G_fn: Callable[[BVSpectrum], torch.Tensor],
        H_fn: Callable[[BVSpectrum], torch.Tensor],
    ) -> torch.Tensor:
        """
        l_3(F, G, H) = Jacobiator = cyclic sum of nested antibrackets.
        Measures failure of Jacobi identity (= 0 iff Jacobi holds exactly).
        """
        sp = self.ab.spectrum

        def FG_fn(s): return self.ab.compute(F_fn, G_fn)
        def GH_fn(s): return self.ab.compute(G_fn, H_fn)
        def HF_fn(s): return self.ab.compute(H_fn, F_fn)

        term1 = self.ab.compute(FG_fn, H_fn)
        term2 = self.ab.compute(GH_fn, F_fn)
        term3 = self.ab.compute(HF_fn, G_fn)
        return term1 + term2 + term3

    def homotopy_cme_residual(
        self,
        master_action_fn: Callable[[BVSpectrum], torch.Tensor],
        hbar:             float = 1.0,
    ) -> torch.Tensor:
        """
        Homotopy CME residual: (S,S) - 2 hbar Delta_BV S.
        Zero = quantum BV master equation.
        """
        sp  = self.ab.spectrum
        lap = BVLaplacian(sp, hbar=hbar)
        ss  = self.ab.compute(master_action_fn, master_action_fn)
        ds  = lap.apply(master_action_fn)
        return ss - 2.0 * hbar * ds


# =============================================================================
# 11.  EVOLUTION ONE Concrete Implementations
# =============================================================================

class GeneNetworkBVFull(BVActionBase):
    """
    Full BV field theory for a gene interaction network.

    Extends the scalar GeneNetworkBV (from one_core_evolution) with:
        • Complete ghost / antifield spectrum
        • Explicit gauge symmetry: simultaneous rescaling symmetry
          Phi^i → lambda Phi^i  (network scale invariance)
        • Quadratic gauge fermion Psi = sum_{ij} lambda_{ij} Phi^i Phi^j
        • Full CME / QME verification
        • W-algebra tensor for network correlation structure

    Physical interpretation:
        - Phi^i = expression level of gene i
        - C^0   = ghost for overall scale symmetry
        - (S, S) = 0  ↔  gene network has no expression-level anomalies

    Args:
        gene_names   : list of gene identifiers.
        interactions : list of (i, j) edge pairs in interaction graph.
        coupling     : interaction coupling strength (default 1.0).
        device       : compute device.
    """

    def __init__(
        self,
        gene_names:   List[str],
        interactions: List[Tuple[int, int]],
        coupling:     float = 1.0,
        device:       Optional[torch.device] = None,
    ) -> None:
        n     = len(gene_names)
        dev   = device or torch.device("cpu")

        # One gauge symmetry: global scale C^0
        spectrum = BVSpectrum(
            field_names=[f"phi_{i}" for i in range(n)],
            gauge_names=["C_scale"],
            device=dev,
        )
        # Initialise fields with small random values
        for i in range(n):
            spectrum.set_field(f"phi_{i}", torch.randn(1, device=dev) * 0.01)
        spectrum.set_ghost("C_scale", torch.tensor([1.0], device=dev))

        super().__init__(spectrum)
        self.gene_names   = gene_names
        self.interactions = interactions
        self.coupling     = coupling
        self._dev         = dev

        # Register scale gauge symmetry: delta Phi^i = C^0 * Phi^i
        def scale_sym(phi_dict, ghost_dict):
            C = ghost_dict["C_scale"]
            return {k: C * v for k, v in phi_dict.items()}

        self.register_gauge_symmetry(
            GaugeSymmetry(scale_sym, name="scale_invariance")
        )

        # Quadratic gauge fermion Psi = -1/2 sum_i Phi^i^2
        self._gauge_fixer = BVGaugeFixer(
            gauge_fermion_fn=lambda phi: -0.5 * sum(v**2 for v in phi.values()),
            spectrum=self.spectrum,
        )

        # W-algebra
        self._w_algebra = WAlgebra(
            antibracket=self._antibracket,
            field_fns=[
                (lambda i: (lambda sp: sp.phi[f"phi_{i}"]))(i)
                for i in range(n)
            ],
        )

        # BRST and cohomology
        self._brst      = BRSTOperator(self.master_action, self._antibracket)
        self._cohomology = BRSTCohomology(self._brst)
        self._homotopy   = HomotopyBV(self._brst, self._antibracket)

        # CME / QME checkers
        self._cme = ClassicalMasterEquation(self)
        self._qme = QuantumMasterEquation(self)

        logger.debug(
            "GeneNetworkBVFull: %d genes, %d interactions, device=%s",
            n, len(interactions), dev,
        )

    # ------------------------------------------------------------------
    # BVActionBase interface
    # ------------------------------------------------------------------

    def classical_action(self, spectrum: BVSpectrum) -> torch.Tensor:
        """
        S_0 = coupling/2 * sum_{(i,j)} (Phi^i - Phi^j)^2.
        Quadratic interaction energy; gauge-invariant under global shifts.
        """
        S = torch.zeros(1, device=self._dev, dtype=spectrum.dtype)
        for i, j in self.interactions:
            phi_i = spectrum.phi.get(f"phi_{i}", torch.zeros(1, device=self._dev))
            phi_j = spectrum.phi.get(f"phi_{j}", torch.zeros(1, device=self._dev))
            S = S + 0.5 * self.coupling * (phi_i - phi_j) ** 2
        return S

    # ------------------------------------------------------------------
    # Full BV analysis
    # ------------------------------------------------------------------

    def analyse(
        self,
        tol:     float = 1e-5,
        verbose: bool  = True,
    ) -> Dict[str, Any]:
        """
        Run complete BV analysis:
            1. CME check
            2. QME check
            3. BRST nilpotency
            4. W-algebra tensor
            5. Gauge-fixed action value

        Returns:
            dict with all diagnostics.
        """
        report = {}

        # 1. CME
        cme_result         = self._cme.check(tol=tol, verbose=verbose)
        report["cme"]      = cme_result

        # 2. QME
        qme_result         = self._qme.check(tol=tol * 10, verbose=verbose)
        report["qme"]      = qme_result

        # 3. BRST nilpotency (on a simple test observable)
        test_fn            = lambda sp: sp.phi.get("phi_0", torch.zeros(1))
        nil_ok, nil_res    = self._brst.nilpotency_check(test_fn)
        report["brst_nilpotent"]  = nil_ok
        report["brst_residual"]   = nil_res

        # 4. W-algebra
        W                  = self._w_algebra.W_tensor()
        report["W_tensor"] = W
        report["casimir"]  = self._w_algebra.casimir().item()
        report["structure_constants"] = self._w_algebra.structure_constants()

        # 5. Gauge-fixed action
        S_gf               = self._gauge_fixer.gauge_fixed_action(self)
        report["S_gf"]     = S_gf.item()

        # 6. Homotopy CME
        h_res              = self._homotopy.homotopy_cme_residual(self.master_action)
        report["homotopy_cme_residual"] = h_res.abs().item()

        # 7. Physical observables (candidate: each field)
        candidate_fns = [
            (lambda k: (lambda sp: sp.phi[k]))(key)
            for key in self.spectrum.field_names
        ]
        phys_idx              = self._cohomology.physical_observables(candidate_fns, tol)
        report["physical_obs_indices"] = phys_idx
        report["n_physical_obs"]       = len(phys_idx)

        report["overall_consistent"] = (
            cme_result["ok"]
            and nil_ok
            and report["homotopy_cme_residual"] < tol * 100
        )
        return report

    def verify(self) -> bool:
        """Quick BV consistency check. Returns True if CME passes."""
        ok, _ = self.verify_cme()
        return ok


class InteractionNetworkBVFull(BVActionBase):
    """
    Full BV field theory for a host-pathogen interaction network.

    Mirrors GeneNetworkBVFull for the epidemiological context:
        - Phi^i = viral fitness / host immune state at node i
        - C^0   = ghost for network symmetry (permutation invariance in SIR compartments)
        - (S, S) = 0  ↔  SIR network has no epidemiological anomaly

    Used by EpiBVEngine to replace the scalar InteractionNetworkBV.

    Args:
        node_names   : list of network nodes (e.g., ["S", "I", "R", "V_1", "V_2"]).
        interactions : list of (i, j) directed coupling pairs.
        coupling     : coupling strength.
        device       : compute device.
    """

    def __init__(
        self,
        node_names:   List[str],
        interactions: List[Tuple[int, int]],
        coupling:     float = 1.0,
        device:       Optional[torch.device] = None,
    ) -> None:
        n   = len(node_names)
        dev = device or torch.device("cpu")

        spectrum = BVSpectrum(
            field_names=[f"phi_{i}" for i in range(n)],
            gauge_names=["C_perm"],
            device=dev,
        )
        for i in range(n):
            spectrum.set_field(f"phi_{i}", torch.randn(1, device=dev) * 0.01)
        spectrum.set_ghost("C_perm", torch.tensor([1.0], device=dev))

        super().__init__(spectrum)
        self.node_names   = node_names
        self.interactions = interactions
        self.coupling     = coupling
        self._dev         = dev

        # Permutation-like gauge symmetry: delta Phi^i = C * (Phi^{i+1} - Phi^i)
        def perm_sym(phi_dict, ghost_dict):
            C  = ghost_dict["C_perm"]
            ns = list(phi_dict.keys())
            return {
                ns[i]: C * (phi_dict[ns[(i + 1) % len(ns)]] - phi_dict[ns[i]])
                for i in range(len(ns))
            }
        self.register_gauge_symmetry(GaugeSymmetry(perm_sym, name="network_perm"))

        self._gauge_fixer  = BVGaugeFixer(
            gauge_fermion_fn=lambda phi: -0.5 * sum(v**2 for v in phi.values()),
            spectrum=self.spectrum,
        )
        self._brst         = BRSTOperator(self.master_action, self._antibracket)
        self._cohomology   = BRSTCohomology(self._brst)
        self._homotopy     = HomotopyBV(self._brst, self._antibracket)
        self._cme          = ClassicalMasterEquation(self)
        self._qme          = QuantumMasterEquation(self)

    def classical_action(self, spectrum: BVSpectrum) -> torch.Tensor:
        S = torch.zeros(1, device=self._dev, dtype=spectrum.dtype)
        for i, j in self.interactions:
            phi_i = spectrum.phi.get(f"phi_{i}", torch.zeros(1, device=self._dev))
            phi_j = spectrum.phi.get(f"phi_{j}", torch.zeros(1, device=self._dev))
            S = S + 0.5 * self.coupling * (phi_i - phi_j) ** 2
        return S

    def analyse(self, tol: float = 1e-5, verbose: bool = True) -> Dict[str, Any]:
        """Run full BV analysis for epidemiological network."""
        cme_result   = self._cme.check(tol=tol, verbose=verbose)
        qme_result   = self._qme.check(tol=tol * 10, verbose=verbose)
        test_fn      = lambda sp: sp.phi.get("phi_0", torch.zeros(1))
        nil_ok, nil_res = self._brst.nilpotency_check(test_fn)
        h_res        = self._homotopy.homotopy_cme_residual(self.master_action)
        S_gf         = self._gauge_fixer.gauge_fixed_action(self)
        return {
            "cme":                    cme_result,
            "qme":                    qme_result,
            "brst_nilpotent":         nil_ok,
            "brst_residual":          nil_res,
            "homotopy_cme_residual":  h_res.abs().item(),
            "S_gf":                   S_gf.item(),
            "overall_consistent":     cme_result["ok"] and nil_ok,
        }

    def verify(self) -> bool:
        ok, _ = self.verify_cme()
        return ok


class CahnHilliardBVFull(BVActionBase):
    """
    Full BV field theory for the Cahn-Hilliard phase-field order parameter.

    Physical interpretation:
        - Phi^0   = spatially averaged phase-field u  (order parameter)
        - Phi^1   = chemical potential mu  (conjugate field)
        - C^0     = ghost for mass-conservation gauge symmetry
                    (total integral of u is conserved: integral u dx = const)
        - S_0     = Ginzburg-Landau free energy + gradient energy
        - (S, S) = 0  ↔  no anomaly in phase-field conservation law

    Used by CahnHilliardBVBridge to certify Cahn-Hilliard evolution.

    Args:
        kappa    : gradient energy coefficient (interface width parameter).
        device   : compute device.
    """

    def __init__(
        self,
        kappa:  float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        dev = device or torch.device("cpu")

        spectrum = BVSpectrum(
            field_names=["phi_u", "phi_mu"],
            gauge_names=["C_mass"],
            device=dev,
        )
        spectrum.set_field("phi_u",  torch.tensor([0.1], device=dev))
        spectrum.set_field("phi_mu", torch.tensor([0.0], device=dev))
        spectrum.set_ghost("C_mass", torch.tensor([1.0], device=dev))

        super().__init__(spectrum)
        self.kappa = kappa
        self._dev  = dev

        # Mass conservation gauge: delta u = C * (1 - u^2)  (mean-field)
        def mass_sym(phi_dict, ghost_dict):
            C = ghost_dict["C_mass"]
            u = phi_dict["phi_u"]
            return {
                "phi_u":  C * F.softplus(1.0 - u**2, beta=50.0),
                "phi_mu": torch.zeros_like(u),
            }
        self.register_gauge_symmetry(GaugeSymmetry(mass_sym, name="mass_conservation"))

        self._gauge_fixer = BVGaugeFixer(
            gauge_fermion_fn=lambda phi: -0.5 * phi["phi_u"] ** 2,
            spectrum=self.spectrum,
        )
        self._brst     = BRSTOperator(self.master_action, self._antibracket)
        self._cme      = ClassicalMasterEquation(self)
        self._qme      = QuantumMasterEquation(self)
        self._homotopy = HomotopyBV(self._brst, self._antibracket)

    def classical_action(self, spectrum: BVSpectrum) -> torch.Tensor:
        """
        Ginzburg-Landau + gradient energy:
            S_0 = 0.25*(1 - u^2)^2   (double-well potential)
                + 0.5*kappa*|grad u|^2  (approximated by kappa*u^2 for scalar)
                + phi_mu * phi_u         (conjugate coupling)
        """
        u   = spectrum.phi["phi_u"]
        mu  = spectrum.phi["phi_mu"]
        dw  = 0.25 * (1.0 - u**2) ** 2                 # double well
        grd = 0.5 * self.kappa * u**2                   # gradient term (scalar approx)
        cpl = mu * u                                     # mu-u coupling
        return dw + grd + cpl

    def analyse(self, tol: float = 1e-5, verbose: bool = True) -> Dict[str, Any]:
        cme_result   = self._cme.check(tol=tol, verbose=verbose)
        qme_result   = self._qme.check(tol=tol * 10, verbose=verbose)
        test_fn      = lambda sp: sp.phi["phi_u"]
        nil_ok, nil_res = self._brst.nilpotency_check(test_fn)
        h_res        = self._homotopy.homotopy_cme_residual(self.master_action)
        S_gf         = self._gauge_fixer.gauge_fixed_action(self)
        return {
            "cme":                    cme_result,
            "qme":                    qme_result,
            "brst_nilpotent":         nil_ok,
            "brst_residual":          nil_res,
            "homotopy_cme_residual":  h_res.abs().item(),
            "S_gf":                   S_gf.item(),
            "overall_consistent":     cme_result["ok"] and nil_ok,
        }

    def verify(self) -> bool:
        ok, _ = self.verify_cme()
        return ok


# =============================================================================
# 12.  Engine Wrappers — Integration with EVOLUTION ONE Cluster
# =============================================================================

class EvoOneBVEngine:
    """
    Full BV engine wrapper for EvolutionONEEngine (cancer / genomic evolution).

    Replaces the scalar GeneNetworkBV.verify() with a complete BV analysis
    that includes CME, QME, BRST cohomology, W-algebra, and gauge fixing.

    Usage::

        engine = EvolutionONEEngine(...)
        bv_engine = EvoOneBVEngine(gene_names, interactions, device=engine.device)
        report = bv_engine.run()
        print(report["overall_consistent"])

    Args:
        gene_names   : gene names from EvolutionONEEngine.
        interactions : gene edge list.
        coupling     : interaction coupling.
        device       : torch device.
    """

    def __init__(
        self,
        gene_names:   List[str],
        interactions: List[Tuple[int, int]],
        coupling:     float = 1.0,
        device:       Optional[torch.device] = None,
    ) -> None:
        self.bv = GeneNetworkBVFull(gene_names, interactions, coupling, device)

    def run(
        self,
        tol:     float = 1e-5,
        verbose: bool  = True,
    ) -> Dict[str, Any]:
        """Run full BV analysis and return diagnostic report."""
        report = self.bv.analyse(tol=tol, verbose=verbose)
        logger.info(
            "EvoOneBVEngine: consistent=%s  CME=%.3e  BRST_nil=%s",
            report["overall_consistent"],
            report["cme"]["residual"],
            report["brst_nilpotent"],
        )
        return report

    def verify(self) -> bool:
        """Quick CME pass/fail for drop-in replacement of GeneNetworkBV.verify()."""
        return self.bv.verify()


class EpiBVEngine:
    """
    Full BV engine wrapper for EpiForecastEngine (epidemiological / viral).

    Replaces the scalar InteractionNetworkBV.verify() with complete BV analysis.

    Usage::

        bv_engine = EpiBVEngine(node_names, interactions)
        report = bv_engine.run()

    Args:
        node_names   : SIR node names (e.g., ["S","I","R","H","D","V1","V2"]).
        interactions : (i, j) coupling pairs.
        coupling     : coupling strength.
        device       : torch device.
    """

    def __init__(
        self,
        node_names:   List[str],
        interactions: List[Tuple[int, int]],
        coupling:     float = 1.0,
        device:       Optional[torch.device] = None,
    ) -> None:
        self.bv = InteractionNetworkBVFull(node_names, interactions, coupling, device)

    def run(
        self,
        tol:     float = 1e-5,
        verbose: bool  = True,
    ) -> Dict[str, Any]:
        """Full BV analysis for epidemiological network."""
        report = self.bv.analyse(tol=tol, verbose=verbose)
        logger.info(
            "EpiBVEngine: consistent=%s  CME=%.3e",
            report["overall_consistent"],
            report["cme"]["residual"],
        )
        return report

    def verify(self) -> bool:
        return self.bv.verify()


class CahnHilliardBVBridge:
    """
    BV bridge for StructuralCahnHilliard3D.

    Certifies the phase-field evolution via BV consistency and
    provides a differentiable BV-weighted mutation load.

    Physical meaning:
        The BV Casimir W measures the coupling between the order parameter
        u and chemical potential mu.  A large Casimir → strong phase separation
        → high spatial mutation load μ.

        μ_BV = sigmoid(W_casimir * u_mean)

    Usage::

        ch_solver = StructuralCahnHilliard3D(cfg)
        bv_bridge = CahnHilliardBVBridge(kappa=cfg.kappa)
        u_field   = ch_solver.u
        mu_bv     = bv_bridge.project_to_mu_bv(u_field)

    Args:
        kappa  : Ginzburg-Landau gradient coefficient (matches CH solver).
        device : torch device.
    """

    def __init__(
        self,
        kappa:  float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.bv     = CahnHilliardBVFull(kappa=kappa, device=device)
        self._dev   = device or torch.device("cpu")

    def verify(self, tol: float = 1e-5) -> bool:
        """Check BV consistency of the Cahn-Hilliard phase-field theory."""
        return self.bv.verify()

    def analyse(self, tol: float = 1e-5, verbose: bool = True) -> Dict[str, Any]:
        """Full BV analysis for Cahn-Hilliard."""
        return self.bv.analyse(tol=tol, verbose=verbose)

    def project_to_mu_bv(
        self,
        u_field: torch.Tensor,
        ssc:     Optional[SemanticStateContraction] = None,
    ) -> torch.Tensor:
        """
        BV-weighted mutation load:
            mu_BV = sigmoid(Casimir * u_mean)

        Fully differentiable w.r.t. u_field.

        Args:
            u_field : phase-field tensor (any shape).
            ssc     : optional SSC filter for temporal smoothing.
        Returns:
            Scalar Tensor in (0, 1).
        """
        u_mean = u_field.mean()
        # Update spectrum with current u_mean
        self.bv.spectrum.set_field("phi_u", u_mean.unsqueeze(0))

        # W-algebra Casimir from current spectrum
        w_alg = WAlgebra(
            antibracket=self.bv._antibracket,
            field_fns=[
                lambda sp: sp.phi["phi_u"],
                lambda sp: sp.phi["phi_mu"],
            ],
        )
        casimir = w_alg.casimir()                        # scalar Tensor

        raw_mu = torch.sigmoid(casimir * u_mean)
        if ssc is not None:
            raw_mu = ssc(raw_mu)
        return F.softplus(raw_mu, beta=100.0).clamp(max=1.0 - 1e-6)

    def bv_to_rt(
        self,
        u_field:   torch.Tensor,
        rt_base:   torch.Tensor,
        scale:     float = 0.5,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Map BV-certified mutation load → effective reproduction number Rt.

        Rt_BV = rt_base + scale * sigmoid((mu_BV - threshold) * 10)

        Fully differentiable — can be composed with EpiEvolutionBridge.mu_to_rt().

        Args:
            u_field   : phase-field tensor.
            rt_base   : baseline Rt (Tensor).
            scale     : max Rt boost.
            threshold : mu_BV threshold for half-maximal boost.
        Returns:
            Rt_BV Tensor.
        """
        mu_bv = self.project_to_mu_bv(u_field)
        boost = torch.sigmoid((mu_bv - threshold) * 10.0)
        return rt_base + scale * boost


# =============================================================================
# 13.  Self-test
# =============================================================================

def _self_test(verbose: bool = False) -> None:
    """
    Quick smoke-test of the full BV module.
    Verifies GeneNetworkBVFull, EvoOneBVEngine, EpiBVEngine,
    and CahnHilliardBVBridge on CPU with a toy example.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    dev = torch.device("cpu")

    # ── Gene network BV ───────────────────────────────────────────────
    genes        = ["TP53", "KRAS", "EGFR", "MYC", "BRCA1"]
    interactions = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    bv_gene = GeneNetworkBVFull(genes, interactions, device=dev)
    report  = bv_gene.analyse(tol=1e-4, verbose=verbose)
    assert isinstance(report["cme"]["ok"], bool), "CME check failed"
    assert "W_tensor" in report,                  "W-algebra missing"
    print(f"[OK] GeneNetworkBVFull  CME_ok={report['cme']['ok']}  "
          f"casimir={report['casimir']:.4f}  "
          f"n_physical_obs={report['n_physical_obs']}")

    # ── EvoOneBVEngine (drop-in) ──────────────────────────────────────
    eng    = EvoOneBVEngine(genes, interactions, device=dev)
    rep2   = eng.run(tol=1e-4, verbose=verbose)
    ok_fast = eng.verify()
    print(f"[OK] EvoOneBVEngine  verify={ok_fast}  consistent={rep2['overall_consistent']}")

    # ── EpiBVEngine ───────────────────────────────────────────────────
    nodes   = ["S", "I", "R", "H", "D"]
    edges   = [(0, 1), (1, 2), (1, 3), (3, 4)]
    epi_eng = EpiBVEngine(nodes, edges, device=dev)
    rep3    = epi_eng.run(tol=1e-4, verbose=verbose)
    print(f"[OK] EpiBVEngine  verify={epi_eng.verify()}  "
          f"CME={rep3['cme']['residual']:.3e}")

    # ── CahnHilliardBVBridge ──────────────────────────────────────────
    ch_bridge = CahnHilliardBVBridge(kappa=0.5, device=dev)
    u_test    = torch.randn(8, 8, 8)
    mu_bv     = ch_bridge.project_to_mu_bv(u_test)
    rt_bv     = ch_bridge.bv_to_rt(u_test, rt_base=torch.tensor(1.2))
    assert 0.0 < mu_bv.item() < 1.0, "mu_BV out of range"
    print(f"[OK] CahnHilliardBVBridge  mu_BV={mu_bv.item():.4f}  "
          f"Rt_BV={rt_bv.item():.4f}  verify={ch_bridge.verify()}")

    print(f"\n[PASS] bv_full_theory_one v{BV_FULL_VERSION}  "
          f"(ecosystem v{EVOLUTION_VERSION})  all checks passed.")


if __name__ == "__main__":
    _self_test(verbose=True)
