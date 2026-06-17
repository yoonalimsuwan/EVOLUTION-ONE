# =============================================================================
# STRUCTURAL GNO EVOLUTION  —  BV EDITION
# =============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
#
# AI Co-Developers:
#   - Claude   (Anthropic)  — BV-edition architecture: BVCertificationBridge,
#                             cross-modal BV distillation loss, mass-conservation
#                             physics loss, periodic analytic certification,
#                             StructuralGNOEvolutionBV, SGNOEvolutionBVTrainer,
#                             graceful two-stage ImportError fallback chain.
#   - GPT      (OpenAI)     — early architecture exploration (base SGNO-Evo).
#   - Gemini   (Google)     — v2 unified discrete/continuous extension (base).
#   - DeepSeek              — numerical stability verification (BV core).
#
# Overview
# --------
# StructuralGNOEvolutionBV is the BV-aware special edition of
# StructuralGNOEvolution (SGNO-Evo).  It keeps the entire production GNO
# surrogate intact (Mode 1 Evolution/Epi, Mode 2 Structural Langevin,
# Mode 3 Cahn-Hilliard 3D) and adds a genuine Batalin-Vilkovisky layer on
# top of it, rather than re-deriving a parallel BV machinery from scratch.
# It does this by *composing* with ``bv_full_theory_one.py`` — the exact,
# non-parametric BV gauge-theory engines already validated for the
# EVOLUTION ONE cluster (GeneNetworkBVFull / InteractionNetworkBVFull /
# CahnHilliardBVFull) — instead of duplicating that formalism inside the
# neural surrogate.
#
# Three concrete BV-edition features
# -----------------------------------
#   [BV-1] Cross-modal BV distillation
#       The exact CahnHilliardBVFull gauge theory (via CahnHilliardBVBridge)
#       maps the Mode-3 phase field u -> a certified mu_BV -> a certified
#       Rt structural-coupling boost.  This certified boost is used as an
#       auxiliary training target for the Mode-1 ΔRt channel, so the fast
#       differentiable surrogate is pulled toward consistency with the
#       analytic BV gauge theory of phase-coupled mutation load.  Because
#       BVSpectrum.set_field() always detaches its input, this pathway is
#       a one-directional, training-stable self-distillation signal — it
#       never back-propagates through the analytic engine.
#
#   [BV-2] Mass-conservation physics loss (Mode 3)
#       CahnHilliardBVFull registers an explicit "mass_conservation" gauge
#       symmetry for the order parameter u.  Cahn-Hilliard dynamics
#       conserve integral(u) exactly, which implies integral(delta_u) = 0.
#       This BV-motivated conservation law is enforced softly on every
#       Mode-3 batch, independent of whether bv_full_theory_one is even
#       importable.
#
#   [BV-3] Periodic analytic BV certification
#       Every ``cfg.bv_cert_every`` training steps, a small subgraph of the
#       current Mode-1 batch (capped at ``cfg.bv_cert_max_nodes`` nodes) is
#       handed to the exact EvoOneBVEngine / EpiBVEngine for a full
#       CME / QME / BRST-nilpotency / W-algebra health-check, independent
#       of the neural surrogate.  Results ([PASS]/[FAIL]) are logged and
#       returned in the training log dict as a free-standing certificate.
#
# Two-stage graceful degradation
# -------------------------------
#   structural_gno_evolution_bv
#     -> bv_full_theory_one            (BV-1, BV-3; optional)
#         -> one_core_evolution        (required by bv_full_theory_one)
#
# If either link of that chain is missing, BV-1 and BV-3 disable
# themselves automatically (with a single warning) and
# StructuralGNOEvolutionBV degrades to the plain production GNO surrogate
# plus BV-2 (which has no external dependency). Training and inference
# never hard-fail because of a missing optional module.
#
# Dependencies
# ------------
#   torch ≥ 2.1
#   structural_gno_evolution   (the base production module — required,
#                                must be importable from the same path)
#   bv_full_theory_one         (optional — enables BV-1 / BV-3)
#   one_core_evolution         (optional — required transitively by the line above)
# =============================================================================

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 0 — Required base module (the production GNO this file extends)
# ---------------------------------------------------------------------------
from structural_gno_evolution import (
    SGNOEvoConfig,
    BatchData,
    StructuralGNOEvolution,
    SGNOEvolutionTrainer,
    EMAWeights,
    CheckpointManager,
    get_device,
    loss_rt_kl_smooth,
    loss_energy_conservation,
    loss_total_variation_3d,
    SGNO_VERSION,
)

# ---------------------------------------------------------------------------
# Stage 1 — Optional BV cross-cluster integration
# ---------------------------------------------------------------------------
try:
    from bv_full_theory_one import (
        CahnHilliardBVBridge,
        EvoOneBVEngine,
        EpiBVEngine,
        BV_FULL_VERSION,
    )
    _HAS_BV_FULL_THEORY = True
    logger.info("structural_gno_evolution_bv: bv_full_theory_one v%s loaded.", BV_FULL_VERSION)
except ImportError as _bv_import_err:
    _HAS_BV_FULL_THEORY = False
    CahnHilliardBVBridge = EvoOneBVEngine = EpiBVEngine = None  # type: ignore
    BV_FULL_VERSION = "unavailable"
    logger.debug(
        "structural_gno_evolution_bv: bv_full_theory_one not found (%s) — "
        "BV-1 distillation and BV-3 certification disabled; BV-2 mass "
        "conservation and the base GNO surrogate remain fully functional.",
        _bv_import_err,
    )

__all__ = [
    "SGNOEvoBVConfig",
    "BVCertificationReport",
    "loss_bv_mass_conservation",
    "loss_bv_distillation",
    "BVCertificationBridge",
    "StructuralGNOEvolutionBV",
    "SGNOEvolutionBVTrainer",
    "SGNO_BV_VERSION",
]

SGNO_BV_VERSION: str = "1.0.0"


# =============================================================================
# 1.  Configuration  — extends SGNOEvoConfig with BV-specific fields
# =============================================================================

@dataclass
class SGNOEvoBVConfig(SGNOEvoConfig):
    """
    Configuration for StructuralGNOEvolutionBV.

    All fields of ``SGNOEvoConfig`` are inherited unchanged. New fields:

    BV bridge (cross-modal distillation, BV-1)
    --------------------------------------------
    enable_bv_bridge   : Master switch. If True but bv_full_theory_one is not
                          importable, BV-1/BV-3 silently disable themselves.
    bv_kappa           : Ginzburg-Landau gradient coefficient passed to
                          CahnHilliardBVFull (should match the CH3D solver's
                          own kappa for a meaningful certification).
    bv_rt_scale        : Max Rt boost magnitude in CahnHilliardBVBridge.bv_to_rt.
    bv_rt_threshold    : mu_BV threshold for half-maximal Rt boost.
    lambda_bv_distill  : Loss weight for the cross-modal BV distillation term
                          (Mode-1 ΔRt <-> Mode-3 phase-coupling boost).

    Mass conservation (BV-2, always available)
    --------------------------------------------
    lambda_bv_mass     : Loss weight for the Cahn-Hilliard mass-conservation
                          penalty integral(delta_u) ≈ 0.

    Periodic certification (BV-3)
    --------------------------------------------
    bv_cert_every      : Run a full analytic BV certification every N
                          optimiser steps (0 disables periodic certification).
    bv_cert_max_nodes  : Cap on the number of graph nodes handed to the exact
                          BV engines per certification call (BVAntibracket is
                          O(n) backward passes per check — keep this small).
    """

    # BV bridge
    enable_bv_bridge:  bool  = True
    bv_kappa:          float = 1.0
    bv_rt_scale:       float = 0.5
    bv_rt_threshold:   float = 0.5
    lambda_bv_distill: float = 0.2

    # Mass conservation
    lambda_bv_mass:    float = 0.05

    # Periodic certification
    bv_cert_every:     int   = 100
    bv_cert_max_nodes: int   = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.bv_kappa          > 0,  "bv_kappa must be positive"
        assert self.lambda_bv_distill >= 0, "lambda_bv_distill must be ≥ 0"
        assert self.lambda_bv_mass    >= 0, "lambda_bv_mass must be ≥ 0"
        assert self.bv_cert_every     >= 0, "bv_cert_every must be ≥ 0"
        assert self.bv_cert_max_nodes >= 3, "bv_cert_max_nodes must be ≥ 3"


# =============================================================================
# 2.  BV Certification Report  — typed container for periodic health-checks
# =============================================================================

@dataclass
class BVCertificationReport:
    """
    Result of one periodic analytic BV certification call.

    Attributes:
        step               : global training step at which this was run.
        kind               : ``"evo"`` (GeneNetworkBVFull-style) or
                              ``"epi"`` (InteractionNetworkBVFull-style).
        n_nodes            : number of nodes actually certified (≤ cap).
        cme_ok             : whether the Classical Master Equation residual
                              was within tolerance.
        cme_residual       : |(S, S)| value.
        brst_nilpotent     : whether s(sF) ≈ 0 held for the BRST test functional.
        casimir            : W-algebra Casimir invariant (correlation strength).
        overall_consistent : combined pass/fail flag from the engine's own
                              ``analyse()`` report.
        passed             : convenience alias of ``overall_consistent``.
    """
    step:               int
    kind:               str
    n_nodes:             int
    cme_ok:              bool
    cme_residual:         float
    brst_nilpotent:       bool
    casimir:              float
    overall_consistent:   bool

    @property
    def passed(self) -> bool:
        return self.overall_consistent

    def summary(self) -> str:
        tag = "[PASS]" if self.passed else "[FAIL]"
        return (
            f"{tag} BV-certify({self.kind}) step={self.step} "
            f"n_nodes={self.n_nodes} CME_ok={self.cme_ok} "
            f"|CME|={self.cme_residual:.3e} BRST_nilpotent={self.brst_nilpotent} "
            f"casimir={self.casimir:.4f}"
        )


# =============================================================================
# 3.  BV-2 — Mass-conservation physics loss (no external dependency)
# =============================================================================

def loss_bv_mass_conservation(delta_u: torch.Tensor) -> torch.Tensor:
    """
    Soft penalty enforcing the Cahn-Hilliard mass-conservation law implied
    by CahnHilliardBVFull's registered "mass_conservation" gauge symmetry.

    Cahn-Hilliard dynamics conserve integral(u) dx exactly, which means the
    spatial mean of any one-step update must vanish:

        Loss = mean(delta_u) ** 2

    This is a genuine BV-motivated conservation law (not a free-floating
    smoothness prior like ``loss_total_variation_3d``), and has zero
    dependency on bv_full_theory_one — it remains active even when the
    exact BV engines are unavailable.

    Args:
        delta_u : (Nx, Ny, Nz) predicted phase-field increment.

    Returns:
        Scalar loss tensor.
    """
    return delta_u.mean() ** 2


# =============================================================================
# 4.  BV-1 — Cross-modal BV distillation loss
# =============================================================================

def loss_bv_distillation(
    mu_rt_pred:       torch.Tensor,
    rt_boost_target:  torch.Tensor,
) -> torch.Tensor:
    """
    Cross-modal BV distillation loss.

    Pulls the mean predicted ΔRt channel (Mode 1) toward the certified
    structural-coupling boost implied by the exact CahnHilliardBVFull gauge
    theory acting on the current Mode-3 phase field (see
    ``BVCertificationBridge.mu_bv_and_boost``).

    ``rt_boost_target`` already carries no gradient w.r.t. model parameters
    (BVSpectrum.set_field detaches internally), so this term only ever
    trains the Mode-1 head — it is a one-directional self-distillation
    signal, never a path for gradients to leak backward into Mode 3.

    Args:
        mu_rt_pred      : (N, 2) predicted [Δμ, ΔRt] from the evolution head.
        rt_boost_target : scalar Tensor, the certified Rt boost.

    Returns:
        Scalar loss tensor.
    """
    rt_pred_mean = mu_rt_pred[:, 1].mean()
    target = rt_boost_target.to(device=rt_pred_mean.device, dtype=rt_pred_mean.dtype)
    return F.mse_loss(rt_pred_mean, target)


# =============================================================================
# 5.  BV Certification Bridge — thin composition wrapper, no re-derivation
# =============================================================================

class BVCertificationBridge:
    """
    Composes the exact BV engines from ``bv_full_theory_one`` with the
    neural surrogate's batch representation.

    This class never re-implements BV algebra; it only adapts data shapes
    (GNO tensors <-> BVSpectrum-friendly lists/scalars) and degrades to a
    safe no-op when ``bv_full_theory_one`` is unavailable.

    Args:
        cfg    : SGNOEvoBVConfig.
        device : torch device for the (CPU-cheap) exact BV engines.
    """

    def __init__(self, cfg: SGNOEvoBVConfig, device: Optional[torch.device] = None) -> None:
        self.cfg     = cfg
        # The exact BV engines operate on tiny scalar leaf tensors and are
        # cheap regardless of where the main GNO runs — pin them to CPU so
        # a CUDA/MPS-resident model never trips a device-mismatch error
        # when its outputs are routed through this bridge.
        self.device  = torch.device("cpu")
        self.available = bool(cfg.enable_bv_bridge and _HAS_BV_FULL_THEORY)

        self._ch_bridge: Optional["CahnHilliardBVBridge"] = None
        if self.available:
            self._ch_bridge = CahnHilliardBVBridge(kappa=cfg.bv_kappa, device=self.device)
        elif cfg.enable_bv_bridge and not _HAS_BV_FULL_THEORY:
            warnings.warn(
                "SGNOEvoBVConfig.enable_bv_bridge=True but bv_full_theory_one "
                "is not importable — BV-1 distillation and BV-3 certification "
                "are disabled for this run; the base GNO surrogate and BV-2 "
                "mass conservation remain fully active.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # BV-1 : phase field -> certified mu_BV / Rt-boost
    # ------------------------------------------------------------------

    def mu_bv_and_boost(self, u_field: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Map a Mode-3 phase field through the exact CahnHilliardBVFull gauge
        theory to a certified ``(mu_BV, rt_boost)`` pair.

        Args:
            u_field : (Nx, Ny, Nz) (or any shape) phase field, e.g. ``pred_u``
                      from ``StructuralGNOEvolution._forward_ch3d``. May live
                      on any device — it is detached and moved to the
                      bridge's CPU device internally.

        Returns:
            (mu_bv, rt_boost) scalar CPU Tensors, or ``None`` if the bridge
            is unavailable. Callers combining these with model tensors on
            another device must move them first (see
            ``StructuralGNOEvolutionBV.forward_certified`` for the pattern).
        """
        if not self.available or self._ch_bridge is None:
            return None
        u_field  = u_field.detach().to(self.device)
        mu_bv    = self._ch_bridge.project_to_mu_bv(u_field)
        rt_boost = self._ch_bridge.bv_to_rt(
            u_field,
            rt_base=torch.zeros((), device=self.device),
            scale=self.cfg.bv_rt_scale,
            threshold=self.cfg.bv_rt_threshold,
        )
        return mu_bv, rt_boost

    # ------------------------------------------------------------------
    # BV-3 : periodic analytic certification
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_index_to_capped_interactions(
        edge_index: torch.Tensor,
        max_nodes:  int,
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Down-sample a (2, E) COO edge list to ≤ max_nodes nodes (relabelled
        0..k-1) so the exact BV engines stay tractable.
        """
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        kept_nodes: List[int] = []
        for n in src + dst:
            if n not in kept_nodes:
                kept_nodes.append(n)
            if len(kept_nodes) >= max_nodes:
                break
        relabel = {n: i for i, n in enumerate(kept_nodes)}
        interactions: List[Tuple[int, int]] = []
        for s, d in zip(src, dst):
            if s in relabel and d in relabel and relabel[s] != relabel[d]:
                pair = (relabel[s], relabel[d])
                if pair not in interactions and (pair[1], pair[0]) not in interactions:
                    interactions.append(pair)
        node_names = [f"node_{i}" for i in range(len(kept_nodes))]
        return node_names, interactions

    def certify(
        self,
        edge_index: torch.Tensor,
        step:       int,
        kind:       str = "evo",
        tol:        float = 1e-4,
    ) -> Optional[BVCertificationReport]:
        """
        Run a full analytic BV health-check on a capped subgraph of the
        current batch.

        Args:
            edge_index : (2, E) COO edge list from the active BatchData.
            step       : current global training step (for the report).
            kind       : ``"evo"`` -> EvoOneBVEngine (GeneNetworkBVFull),
                         ``"epi"`` -> EpiBVEngine (InteractionNetworkBVFull).
            tol        : CME / nilpotency tolerance.

        Returns:
            BVCertificationReport, or ``None`` if the bridge is unavailable.
        """
        if not self.available:
            return None

        node_names, interactions = self._edge_index_to_capped_interactions(
            edge_index, self.cfg.bv_cert_max_nodes
        )
        if len(interactions) == 0:
            logger.debug("BVCertificationBridge.certify: no usable interactions, skipping.")
            return None

        if kind == "epi":
            engine = EpiBVEngine(node_names, interactions, device=self.device)
        else:
            engine = EvoOneBVEngine(node_names, interactions, device=self.device)

        report = engine.run(tol=tol, verbose=False)
        cert = BVCertificationReport(
            step=step,
            kind=kind,
            n_nodes=len(node_names),
            cme_ok=bool(report["cme"]["ok"]),
            cme_residual=float(report["cme"]["residual"]),
            brst_nilpotent=bool(report["brst_nilpotent"]),
            casimir=float(report.get("casimir", float("nan"))),
            overall_consistent=bool(report["overall_consistent"]),
        )
        logger.info(cert.summary())
        return cert


# =============================================================================
# 6.  StructuralGNOEvolutionBV — Main Model
# =============================================================================

class StructuralGNOEvolutionBV(StructuralGNOEvolution):
    """
    BV Edition of StructuralGNOEvolution.

    Identical architecture and forward dispatcher to the base production
    model (Mode 1 / 2 / 3 all behave exactly as in StructuralGNOEvolution).
    Adds one new capability: ``forward_certified``, which runs Mode 1 and
    Mode 3 together and attaches the BV-certified cross-modal Rt boost
    (BV-1) to the result for inference-time inspection or downstream use.

    Args:
        cfg : SGNOEvoBVConfig instance.
    """

    def __init__(self, cfg: SGNOEvoBVConfig) -> None:
        super().__init__(cfg)
        self.cfg: SGNOEvoBVConfig = cfg
        self.bv_bridge = BVCertificationBridge(cfg)

    def forward_certified(
        self,
        batch_evo: Optional[BatchData] = None,
        batch_ch:  Optional[BatchData] = None,
    ) -> Dict[str, Any]:
        """
        Run Mode 1 (evolution) and/or Mode 3 (ch3d) and attach the
        BV-certified cross-modal Rt boost when both batches and the BV
        bridge are available.

        Args:
            batch_evo : Mode-1 batch, or None to skip.
            batch_ch  : Mode-3 batch, or None to skip.

        Returns:
            dict with keys from whichever modes ran, plus (when available):
                ``mu_bv``               — certified phase-coupling mu_BV
                ``rt_bv_boost``         — certified Rt structural boost
                ``rt_certified``        — model ΔRt + rt_bv_boost (N,)
        """
        out: Dict[str, Any] = {}
        if batch_evo is not None:
            out.update(self._forward_evolution(batch_evo))
        if batch_ch is not None:
            out.update(self._forward_ch3d(batch_ch))

        if batch_evo is not None and batch_ch is not None and self.bv_bridge.available:
            mu_boost = self.bv_bridge.mu_bv_and_boost(out["pred_u"])
            if mu_boost is not None:
                mu_bv, rt_boost = mu_boost
                model_device = out["mu_rt"].device
                out["mu_bv"]        = mu_bv
                out["rt_bv_boost"]  = rt_boost
                out["rt_certified"] = out["mu_rt"][:, 1] + rt_boost.to(model_device)
        return out


# =============================================================================
# 7.  SGNOEvolutionBVTrainer — adds BV-1 / BV-2 / BV-3 to the production trainer
# =============================================================================

class SGNOEvolutionBVTrainer(SGNOEvolutionTrainer):
    """
    Trainer for StructuralGNOEvolutionBV.

    Extends ``SGNOEvolutionTrainer`` with three additions, all individually
    weighted / toggle-able via ``SGNOEvoBVConfig``:

      • BV-2 mass conservation on every Mode-3 batch (``lambda_bv_mass``).
      • BV-1 cross-modal distillation when both Mode-1 and Mode-3 batches
        are present and the BV bridge is available (``lambda_bv_distill``).
      • BV-3 periodic analytic certification every ``bv_cert_every`` steps,
        logged into the returned loss dict as ``bv_cme_residual`` /
        ``bv_overall_consistent`` whenever it runs.

    All base-trainer behaviour (AMP, EMA, gradient clipping, NaN guards,
    LR scheduling, checkpointing, early stopping) is unchanged.

    Args:
        model    : Initialised StructuralGNOEvolutionBV.
        cfg      : SGNOEvoBVConfig.
        device   : Compute device; defaults to best available.
        ckpt_dir : Directory for checkpoint files.
    """

    def __init__(
        self,
        model:    StructuralGNOEvolutionBV,
        cfg:      SGNOEvoBVConfig,
        device:   Optional[torch.device] = None,
        ckpt_dir: str = "checkpoints",
    ) -> None:
        super().__init__(model, cfg, device=device, ckpt_dir=ckpt_dir)
        self.cfg:   SGNOEvoBVConfig          = cfg
        self.model: StructuralGNOEvolutionBV = model

    # ------------------------------------------------------------------
    # BV-augmented loss computation
    # ------------------------------------------------------------------

    def _compute_bv_losses(
        self,
        batch_evo: Optional[BatchData],
        batch_md:  Optional[BatchData],
        batch_ch:  Optional[BatchData],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute base multi-objective loss plus the BV-1 / BV-2 additions.

        Re-runs the relevant forward passes (kept separate from the base
        trainer's private ``_compute_losses`` for clarity / composability
        rather than raw speed — batches in this research setting are small
        graphs, so the duplicate forward cost is negligible).
        """
        total_loss, log = self._compute_losses(batch_evo, batch_md, batch_ch)
        cfg = self.cfg

        # ── BV-2 : mass conservation (always available) ────────────────
        if batch_ch is not None:
            out_ch      = self.model(batch_ch, mode="ch3d")
            loss_bv_mass = loss_bv_mass_conservation(out_ch["delta_u"])
            l_bv_mass    = cfg.lambda_bv_mass * loss_bv_mass
            total_loss   = total_loss + l_bv_mass
            log["loss_bv_mass"] = l_bv_mass.item()

            # ── BV-1 : cross-modal distillation (requires bridge + Mode 1) ──
            if batch_evo is not None and self.model.bv_bridge.available:
                out_evo  = self.model(batch_evo, mode="evolution")
                mu_boost = self.model.bv_bridge.mu_bv_and_boost(out_ch["pred_u"].detach())
                if mu_boost is not None:
                    mu_bv, rt_boost = mu_boost
                    loss_distill = loss_bv_distillation(out_evo["mu_rt"], rt_boost)
                    l_bv_distill = cfg.lambda_bv_distill * loss_distill
                    total_loss   = total_loss + l_bv_distill
                    log["loss_bv_distill"] = l_bv_distill.item()
                    log["mu_bv"]           = mu_bv.item()

        log["total_loss"] = total_loss.item()
        return total_loss, log

    # ------------------------------------------------------------------
    # Single training step (BV-augmented)
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch_evo: Optional[BatchData] = None,
        batch_md:  Optional[BatchData] = None,
        batch_ch:  Optional[BatchData] = None,
    ) -> Dict[str, float]:
        """
        Identical contract to ``SGNOEvolutionTrainer.train_step``, but the
        loss includes BV-1 / BV-2 terms, and BV-3 certification runs
        automatically every ``cfg.bv_cert_every`` steps when a Mode-1 batch
        with an edge_index is available.
        """
        if batch_evo is None and batch_md is None and batch_ch is None:
            raise ValueError("At least one batch must be non-None.")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self._use_amp):
            total_loss, log = self._compute_bv_losses(batch_evo, batch_md, batch_ch)

        if not torch.isfinite(total_loss):
            logger.warning("Step %d: non-finite loss (%.4g) — skipping update.",
                           self.global_step, total_loss.item())
            return log

        self._scaler.scale(total_loss).backward()
        self._scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        if not torch.isfinite(grad_norm):
            logger.warning("Step %d: non-finite gradient norm — skipping update.", self.global_step)
            self.optimizer.zero_grad(set_to_none=True)
            return log

        self._scaler.step(self.optimizer)
        self._scaler.update()
        self.scheduler.step()
        self.ema.update(self.model)

        log["grad_norm"] = grad_norm.item()
        log["lr"]        = self.scheduler.get_last_lr()[0]
        self.global_step += 1

        # ── BV-3 : periodic analytic certification ─────────────────────
        if (
            self.cfg.bv_cert_every > 0
            and self.model.bv_bridge.available
            and batch_evo is not None
            and self.global_step % self.cfg.bv_cert_every == 0
        ):
            cert = self.model.bv_bridge.certify(
                batch_evo.edge_index, step=self.global_step, kind="evo"
            )
            if cert is not None:
                log["bv_cme_residual"]       = cert.cme_residual
                log["bv_overall_consistent"] = float(cert.overall_consistent)

        if self.global_step % self.cfg.log_every == 0:
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in log.items())
            logger.info("Step %6d | %s", self.global_step, loss_str)

        return log


# =============================================================================
# 8.  Quick Smoke-Test  (python structural_gno_evolution_bv.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    print(f"StructuralGNOEvolutionBV v{SGNO_BV_VERSION}  (base SGNO v{SGNO_VERSION}, "
          f"bv_full_theory_one={'v' + BV_FULL_VERSION if _HAS_BV_FULL_THEORY else 'unavailable'})")
    print("  smoke test")

    torch.manual_seed(42)
    device = get_device()
    print(f"  device : {device}")

    cfg = SGNOEvoBVConfig(
        hidden_dim=64, num_layers=3, max_epochs=2, log_every=1,
        bv_cert_every=2, bv_cert_max_nodes=8,
    )
    model = StructuralGNOEvolutionBV(cfg).to(device)
    print(f"  params : {model.num_parameters():,}")
    print(f"  bv_bridge.available : {model.bv_bridge.available}")

    # ── Synthetic batch helpers ──────────────────────────────────────
    N, E, Nx = 12, 24, 4

    def rand_edge(n, e):
        src = torch.randint(0, n, (e,))
        dst = torch.randint(0, n, (e,))
        return torch.stack([src, dst], dim=0).to(device)

    batch_evo = BatchData(
        feats       = torch.randn(N, cfg.node_in_dim).to(device),
        edge_index  = rand_edge(N, E),
        sigma       = torch.rand(N, 1).to(device),
        true_mu_rt  = torch.rand(N, 2).to(device),
        labels      = torch.randint(0, 3, (N,)).to(device),
    )
    batch_md = BatchData(
        feats              = torch.randn(N, cfg.node_in_dim).to(device),
        edge_index         = rand_edge(N, E),
        sigma              = torch.rand(N, 1).to(device),
        coords             = torch.randn(N, 3).to(device),
        true_future_coords = torch.randn(N, 3).to(device),
    )
    M = Nx ** 3
    batch_ch = BatchData(
        u_init          = torch.randn(Nx, Nx, Nx).to(device),
        grid_feats      = torch.randn(M, cfg.grid_in_dim).to(device),
        grid_edge_index = rand_edge(M, M * 2),
        sigma_3d        = torch.rand(Nx, Nx, Nx).to(device),
        true_future_u   = torch.randn(Nx, Nx, Nx).to(device),
    )

    trainer = SGNOEvolutionBVTrainer(model, cfg, device=device, ckpt_dir="/tmp/sgno_bv_test_ckpts")

    # ── A few train steps (exercises BV-1 + BV-2, and BV-3 at step 2) ──
    for i in range(3):
        log = trainer.train_step(batch_evo, batch_md, batch_ch)
    print(f"  train_step OK  total_loss={log['total_loss']:.4f}  "
          f"grad_norm={log.get('grad_norm', float('nan')):.4f}  "
          f"loss_bv_mass={log.get('loss_bv_mass', float('nan')):.4f}  "
          f"loss_bv_distill={log.get('loss_bv_distill', float('nan')):.4f}")

    # ── forward_certified ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out = model.forward_certified(batch_evo, batch_ch)
    has_cert = "rt_bv_boost" in out
    print(f"  forward_certified OK  keys={sorted(out.keys())}  has_bv_fields={has_cert}")
    model.train()

    # ── Evaluate with EMA ────────────────────────────────────────────
    val_log = trainer.evaluate(batch_evo, batch_md, batch_ch)
    print(f"  evaluate OK    total_loss={val_log['total_loss']:.4f}")

    # ── Checkpoint round-trip ────────────────────────────────────────
    ckpt_path = trainer.ckpt_mgr.save(
        model, trainer.optimizer, trainer.scheduler, trainer.ema,
        {"epoch": 0, "step": trainer.global_step, "best_loss": val_log["total_loss"]}, tag="smoke"
    )
    CheckpointManager.load(ckpt_path, model, device=device)
    print(f"  checkpoint OK  → {ckpt_path}")

    # ── Mini fit loop (base trainer loop; BV terms ride along inside train_step) ──
    history = trainer.fit(
        train_batches=[(batch_evo, batch_md, batch_ch)] * 3,
        val_batches=  [(batch_evo, batch_md, batch_ch)],
    )
    print(f"  fit OK   final train_loss={history['train_loss'][-1]:.4f}")

    # ── Standalone BV-3 certification call ──────────────────────────
    cert = model.bv_bridge.certify(batch_evo.edge_index, step=trainer.global_step, kind="evo")
    if cert is not None:
        print(f"  {cert.summary()}")
    else:
        print("  [SKIP] BV-3 certification (bv_full_theory_one unavailable in this environment)")

    print(f"\n[PASS] structural_gno_evolution_bv v{SGNO_BV_VERSION} — all checks passed.")
    sys.exit(0)
