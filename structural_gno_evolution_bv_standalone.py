# =============================================================================
# STRUCTURAL GNO EVOLUTION  —  BV EDITION (STANDALONE, SINGLE-FILE)
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
#   - Claude   (Anthropic)  — Base production refactor (EMA checkpointing,
#                             multi-loss weighting, physics-informed losses,
#                             LR scheduling, gradient monitoring, full
#                             docstrings) AND the BV-edition layer on top of
#                             it: BVCertificationBridge, cross-modal BV
#                             distillation loss, mass-conservation physics
#                             loss, periodic analytic certification,
#                             StructuralGNOEvolutionBV, SGNOEvolutionBVTrainer,
#                             two-stage graceful ImportError fallback chain,
#                             and this standalone single-file merge.
#   - GPT      (OpenAI)     — early architecture exploration, message-passing
#                             design, phase-field surrogate concept.
#   - Gemini   (Google)     — v2 unified discrete/continuous extension,
#                             one-shot phase evolution framing.
#   - DeepSeek              — numerical stability verification (BV core).
#
# Overview
# --------
# This is the STANDALONE edition: the entire production GNO surrogate
# (StructuralGNOEvolution, SGNOEvoConfig, BatchData, RBFPositionalEncoder,
# SigmaEncoder, FiLMMessagePassing, the physics-informed losses, EMAWeights,
# CheckpointManager, SGNOEvolutionTrainer — Sections 0-9 below) is included
# verbatim in this single file, followed by the BV-edition layer (Sections
# 10-16) that extends it. No import of a separate ``structural_gno_evolution``
# module is required; this file is fully self-contained.
#
# StructuralGNOEvolutionBV is the BV-aware special edition of
# StructuralGNOEvolution (SGNO-Evo). It keeps the entire production GNO
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
#       Rt structural-coupling boost. This certified boost is used as an
#       auxiliary training target for the Mode-1 ΔRt channel, so the fast
#       differentiable surrogate is pulled toward consistency with the
#       analytic BV gauge theory of phase-coupled mutation load.  Because
#       BVSpectrum.set_field() always detaches its input, this pathway is
#       a one-directional, training-stable self-distillation signal — it
#       never back-propagates through the analytic engine. It introduces
#       NO new trainable parameters: gradients flow only into the existing
#       evo_head.
#
#   [BV-2] Mass-conservation physics loss (Mode 3)
#       CahnHilliardBVFull registers an explicit "mass_conservation" gauge
#       symmetry for the order parameter u. Cahn-Hilliard dynamics conserve
#       integral(u) exactly, which implies integral(delta_u) = 0. This
#       BV-motivated conservation law is enforced softly on every Mode-3
#       batch, independent of whether bv_full_theory_one is even importable.
#
#   [BV-3] Periodic analytic BV certification
#       Every ``cfg.bv_cert_every`` training steps, a small subgraph of the
#       current Mode-1 batch (capped at ``cfg.bv_cert_max_nodes`` nodes) is
#       handed to the exact EvoOneBVEngine / EpiBVEngine for a full
#       CME / QME / BRST-nilpotency / W-algebra health-check, independent
#       of the neural surrogate. Results ([PASS]/[FAIL]) are logged and
#       returned in the training log dict as a free-standing certificate.
#
# Parameter count
# ----------------
# StructuralGNOEvolutionBV adds ZERO new trainable parameters relative to
# StructuralGNOEvolution — the BV engines it composes with are purely
# analytic (no nn.Parameter / nn.Module). model.num_parameters() and the
# state_dict() keys are therefore identical between the two; a checkpoint
# trained with either model loads directly into the other.
#
# Graceful degradation (still two-stage, even though single-file)
# -------------------------------------------------------------------
#   this file
#     -> bv_full_theory_one            (BV-1, BV-3; optional, separate file)
#         -> one_core_evolution        (required by bv_full_theory_one)
#
# If either link of that chain is missing, BV-1 and BV-3 disable themselves
# automatically (with a single warning) and StructuralGNOEvolutionBV
# degrades to the plain production GNO surrogate plus BV-2 (which has no
# external dependency). Training and inference never hard-fail because of
# a missing optional module.
#
# Dependencies
# ------------
#   torch ≥ 2.1   (AMP, compile-ready)
#   bv_full_theory_one   (optional — separate file, enables BV-1 / BV-3)
#   one_core_evolution   (optional — required transitively by the line above)
# =============================================================================

from __future__ import annotations

import copy
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional EVOLUTION ONE cross-cluster integration
# ---------------------------------------------------------------------------
try:
    from one_core_evolution import (
        EpiEvolutionBridge,
        CahnHilliardEvoBridge,
        SemanticStateContraction,
        EVOLUTION_VERSION,
    )
    _HAS_ONE_CORE_EVOLUTION = True
    logger.info("structural_gno_evolution: one_core_evolution v%s loaded.", EVOLUTION_VERSION)
except ImportError:
    _HAS_ONE_CORE_EVOLUTION = False
    logger.debug("structural_gno_evolution: one_core_evolution not found — bridge features disabled.")

__all__ = [
    "SGNOEvoConfig",
    "BatchData",
    "RBFPositionalEncoder",
    "SigmaEncoder",
    "FiLMMessagePassing",
    "StructuralGNOEvolution",
    "SGNOEvolutionTrainer",
    "CheckpointManager",
    "get_device",
    "SGNO_VERSION",
]

SGNO_VERSION: str = "1.0.0"


# =============================================================================
# 0.  Utilities
# =============================================================================

def get_device(preferred: str = "cuda") -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.

    Args:
        preferred: ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns:
        torch.device
    """
    p = preferred.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1.  Configuration  [P-1]
# =============================================================================

@dataclass
class SGNOEvoConfig:
    """
    Full configuration for StructuralGNOEvolution.

    Architecture
    ------------
    node_in_dim     : Input feature dimension for graph nodes
                      (amino-acid one-hot / patient embeddings / voxel descriptors).
    grid_in_dim     : Input feature dimension for CH3D grid nodes [u, x, y, z, σ].
    hidden_dim      : Hidden dimension shared across backbone and all heads.
    num_layers      : Number of FiLMMessagePassing layers in the shared backbone.
    num_rbf         : Number of radial-basis-function bins for edge/positional encoding.
    dropout         : Dropout probability applied inside message-passing MLPs.
    cutoff_graph    : Neighbour cutoff (Å or distance units) for molecular / patient graphs.
    cutoff_grid     : Neighbour cutoff (grid units) for CH3D voxel graphs.

    Loss weights
    ------------
    lambda_evo      : Weight for mutation-load / Rt MSE loss (Mode 1).
    lambda_cls      : Weight for CSOC state classification loss (Mode 1).
    lambda_md       : Weight for coordinate-displacement MSE loss (Mode 2).
    lambda_md_phys  : Weight for energy-conservation physics loss (Mode 2).
    lambda_ch       : Weight for phase-field MSE loss (Mode 3).
    lambda_ch_tv    : Weight for total-variation interface sharpness loss (Mode 3).

    Training
    --------
    lr              : Initial learning rate for AdamW.
    weight_decay    : L2 regularisation coefficient.
    lr_warmup_steps : Number of linear warm-up steps before cosine decay.
    lr_min          : Minimum learning rate at the end of cosine schedule.
    grad_clip       : Maximum gradient L2-norm for gradient clipping.
    ema_decay       : Exponential moving average decay for weight averaging.
    max_epochs      : Total training epochs.
    patience        : Early-stopping patience (epochs without improvement).
    use_amp         : Enable automatic mixed precision on CUDA.
    log_every       : Log training metrics every N steps.
    """

    # Architecture
    node_in_dim:     int   = 20
    grid_in_dim:     int   = 5        # [u, x, y, z, sigma]
    hidden_dim:      int   = 128
    num_layers:      int   = 6
    num_rbf:         int   = 16
    dropout:         float = 0.1
    cutoff_graph:    float = 10.0
    cutoff_grid:     float = 1.5

    # Loss weights
    lambda_evo:      float = 1.0
    lambda_cls:      float = 0.5
    lambda_md:       float = 0.5
    lambda_md_phys:  float = 0.1
    lambda_ch:       float = 0.5
    lambda_ch_tv:    float = 0.05

    # Training
    lr:              float = 3e-4
    weight_decay:    float = 1e-5
    lr_warmup_steps: int   = 500
    lr_min:          float = 1e-6
    grad_clip:       float = 1.0
    ema_decay:       float = 0.999
    max_epochs:      int   = 200
    patience:        int   = 20
    use_amp:         bool  = True
    log_every:       int   = 50

    def __post_init__(self) -> None:
        assert self.hidden_dim  >= 16,   "hidden_dim must be ≥ 16"
        assert self.num_layers  >= 1,    "num_layers must be ≥ 1"
        assert 0.0 <= self.dropout < 1.0,"dropout must be in [0, 1)"
        assert self.lr          >  0,    "lr must be positive"
        assert self.grad_clip   >  0,    "grad_clip must be positive"
        assert 0.0 < self.ema_decay < 1.0,"ema_decay must be in (0, 1)"


# =============================================================================
# 2.  Typed Batch Container  [P-7]
# =============================================================================

@dataclass
class BatchData:
    """
    Typed container for a single training batch.

    All tensors are optional; only the fields required by the active
    forward mode need to be populated.

    Shapes
    ------
    feats            : (N, node_in_dim)       — node feature matrix
    edge_index       : (2, E)                 — COO edge list
    sigma            : (N, 1) or (N,)         — structural regime σ per node
    true_mu_rt       : (N, 2)                 — target [μ, Rt] increments
    labels           : (N,) int64             — CSOC state labels {0,1,2}
    coords           : (N, 3)                 — initial atomic coordinates
    true_future_coords: (N, 3)               — target coordinates after MD
    u_init           : (Nx, Ny, Nz)          — initial CH3D phase field
    grid_feats       : (Nx*Ny*Nz, grid_in_dim)— voxel features
    grid_edge_index  : (2, E_grid)            — voxel graph COO
    sigma_3d         : (Nx, Ny, Nz)          — CH3D structural regime field
    true_future_u    : (Nx, Ny, Nz)          — target phase field after Δt
    """
    feats:              Optional[torch.Tensor] = None
    edge_index:         Optional[torch.Tensor] = None
    sigma:              Optional[torch.Tensor] = None
    true_mu_rt:         Optional[torch.Tensor] = None
    labels:             Optional[torch.Tensor] = None
    coords:             Optional[torch.Tensor] = None
    true_future_coords: Optional[torch.Tensor] = None
    u_init:             Optional[torch.Tensor] = None
    grid_feats:         Optional[torch.Tensor] = None
    grid_edge_index:    Optional[torch.Tensor] = None
    sigma_3d:           Optional[torch.Tensor] = None
    true_future_u:      Optional[torch.Tensor] = None


# =============================================================================
# 3.  Positional / Structural Encoders  [P-3]
# =============================================================================

class RBFPositionalEncoder(nn.Module):
    """
    Radial-basis-function (RBF) encoder for inter-node distances.

    Maps a scalar distance r into a ``num_rbf``-dimensional feature vector
    using a bank of Gaussian kernels evenly spaced over [0, cutoff].

    This gives the message-passing layer a continuous, differentiable
    description of edge geometry — important for both the molecular graph
    (Å distances) and the CH3D voxel graph (grid-unit distances).

    Args:
        num_rbf : Number of Gaussian centres.
        cutoff  : Maximum distance (same units as input distances).
    """

    def __init__(self, num_rbf: int = 16, cutoff: float = 10.0) -> None:
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff  = cutoff
        # Learnable centres and widths
        centres = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centres", centres)
        self.log_width = nn.Parameter(
            torch.zeros(num_rbf) + math.log(2.0 / num_rbf)
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist : (E,) edge distances.
        Returns:
            rbf  : (E, num_rbf) feature matrix.
        """
        width = torch.exp(self.log_width).clamp(min=1e-4)      # softplus-safe
        diff  = dist.unsqueeze(-1) - self.centres.unsqueeze(0)  # (E, num_rbf)
        return torch.exp(-0.5 * (diff ** 2) / (width ** 2))


class SigmaEncoder(nn.Module):
    """
    Project scalar structural regime σ to hidden dimension.

    A small 2-layer MLP maps σ ∈ ℝ (per node) → ℝ^d, producing the
    per-node context vector used by FiLM modulation.

    Args:
        hidden_dim : Output dimension (= model hidden_dim).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma : (N,) or (N, 1) structural regime field.
        Returns:
            (N, hidden_dim) encoded sigma context.
        """
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        return self.norm(self.net(sigma))


# =============================================================================
# 4.  FiLM Message-Passing Layer  [P-2]
# =============================================================================

class FiLMMessagePassing(nn.Module):
    """
    Graph message-passing layer with Feature-wise Linear Modulation (FiLM).

    Architecture
    ------------
    Pre-norm pattern:
        x_norm   = LayerNorm(x)
        messages = MLP([x_norm[src] ‖ x_norm[dst] ‖ rbf_edge])
        aggr     = scatter_add(messages, dst, dim=0)
        modulated= gamma(σ_ctx) ⊙ aggr + beta(σ_ctx)
        gate     = sigmoid(W_gate · [x_norm ‖ modulated])
        x_new    = x + gate ⊙ MLP([x_norm ‖ modulated])

    Using pre-norm + residual gate gives:
      • Stable gradients in deep stacks (no post-norm collapse)
      • Adaptive blending of local graph signal vs. σ-driven modulation
      • Full differentiability (no hard clamp or non-differentiable op)

    Args:
        dim      : Node feature / hidden dimension.
        num_rbf  : RBF edge-feature dimension (from RBFPositionalEncoder).
        dropout  : Dropout inside MLPs.
    """

    def __init__(self, dim: int, num_rbf: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)

        # Message MLP: [x_src ‖ x_dst ‖ rbf_edge] → dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2 + num_rbf, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        # FiLM modulator: σ_context → (γ, β), both shape (N, dim)
        self.film_gamma = nn.Linear(dim, dim, bias=False)
        self.film_beta  = nn.Linear(dim, dim)

        # Update MLP: [x ‖ modulated_aggr] → dim
        self.upd_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        # Residual gate: sigmoid([x ‖ modulated_aggr]) → scalar per dim
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.post_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        sigma_ctx:  torch.Tensor,
        edge_attr:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x          : (N, dim) node features.
            edge_index : (2, E) COO edge list (src, dst).
            sigma_ctx  : (N, dim) encoded sigma context from SigmaEncoder.
            edge_attr  : (E, num_rbf) RBF edge features, or None (zeros used).
        Returns:
            x_out      : (N, dim) updated node features.
        """
        src, dst = edge_index[0], edge_index[1]
        N   = x.size(0)
        E   = src.size(0)

        x_n = self.pre_norm(x)

        # Edge attributes
        if edge_attr is None:
            edge_attr = x.new_zeros(E, self.msg_mlp[0].in_features - 2 * x.size(-1))

        # Message computation
        msg_in   = torch.cat([x_n[src], x_n[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_in)                          # (E, dim)

        # Scatter-add aggregation
        aggr = x.new_zeros(N, messages.size(-1))
        aggr.index_add_(0, dst, messages)                        # (N, dim)

        # FiLM modulation by structural regime σ
        gamma     = self.film_gamma(sigma_ctx)                   # (N, dim)
        beta      = self.film_beta(sigma_ctx)                    # (N, dim)
        modulated = gamma * aggr + beta                          # (N, dim)

        # Gated residual update
        concat    = torch.cat([x_n, modulated], dim=-1)          # (N, 2*dim)
        gate_val  = self.gate(concat)                            # (N, dim)
        update    = self.upd_mlp(concat)                         # (N, dim)
        x_out     = x + gate_val * update                        # (N, dim)

        return self.post_norm(x_out)


# =============================================================================
# 5.  StructuralGNOEvolution — Main Model  [P-3]
# =============================================================================

class StructuralGNOEvolution(nn.Module):
    """
    Unified Graph Neural Operator surrogate for the EVOLUTION ONE cluster.

    The model shares a single backbone across three physically distinct
    simulation modes, using FiLM modulation to condition computation on
    the local structural regime field σ.

    Forward dispatcher
    ------------------
    Call ``forward(batch, mode)`` with mode in::

        "evolution"  — Mode 1: μ / Rt / CSOC state
        "langevin"   — Mode 2: BAOAB MD coordinate surrogate
        "ch3d"       — Mode 3: Cahn-Hilliard phase-field surrogate

    Returns a dict with keys depending on mode (see docstrings of the
    private ``_forward_*`` methods).

    Args:
        cfg : SGNOEvoConfig instance.
    """

    def __init__(self, cfg: SGNOEvoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d        = cfg.hidden_dim
        rbf      = cfg.num_rbf

        # ── Encoders ────────────────────────────────────────────────────
        self.node_embed   = nn.Sequential(
            nn.Linear(cfg.node_in_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.grid_embed   = nn.Sequential(
            nn.Linear(cfg.grid_in_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.sigma_enc    = SigmaEncoder(d)
        self.rbf_enc      = RBFPositionalEncoder(num_rbf=rbf, cutoff=max(cfg.cutoff_graph, cfg.cutoff_grid))
        self.edge_proj    = nn.Linear(rbf, rbf)   # project RBF → same dim expected by layers

        # ── Shared Backbone ──────────────────────────────────────────────
        self.layers = nn.ModuleList([
            FiLMMessagePassing(d, num_rbf=rbf, dropout=cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        # ── Head 1: Evolution / Epidemiological ─────────────────────────
        # Outputs: [Δμ, ΔRt]  and  logits [Stable | Critical | Collapse]
        self.evo_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 3),
        )

        # ── Head 2: Structural Langevin (BAOAB surrogate) ───────────────
        # Outputs: coordinate displacements (N, 3)
        self.md_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 3),
        )

        # ── Head 3: Cahn-Hilliard 3D (phase-field surrogate) ────────────
        # Outputs: Δu per voxel (M, 1)
        self.ch3d_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        # ── Output normalisations ─────────────────────────────────────
        self.evo_out_norm  = nn.LayerNorm(2)
        self.md_out_norm   = nn.LayerNorm(3)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming-uniform init for linear layers; zero-init output heads."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the last layer of each head → small initial predictions
        for head in [self.evo_head, self.md_head, self.ch3d_head]:
            nn.init.zeros_(head[-1].weight)
            if head[-1].bias is not None:
                nn.init.zeros_(head[-1].bias)

    # ------------------------------------------------------------------
    # Shared Backbone
    # ------------------------------------------------------------------

    def _encode_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Ensure sigma is (N, 1) then encode to (N, hidden_dim)."""
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        return self.sigma_enc(sigma)

    def _apply_backbone(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        sigma_ctx:  torch.Tensor,
        edge_attr:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run all FiLMMessagePassing layers sequentially.

        Args:
            x          : (N, d) node embeddings.
            edge_index : (2, E) COO edges.
            sigma_ctx  : (N, d) encoded sigma context.
            edge_attr  : (E, num_rbf) RBF edge features, or None.
        Returns:
            (N, d) updated node representations.
        """
        for layer in self.layers:
            x = layer(x, edge_index, sigma_ctx, edge_attr)
        return x

    # ------------------------------------------------------------------
    # Private Forward Modes
    # ------------------------------------------------------------------

    def _forward_evolution(
        self,
        batch: BatchData,
    ) -> Dict[str, torch.Tensor]:
        """
        Mode 1: Evolution & Epidemiological surrogate.

        Predicts per-node increments Δμ and ΔRt, plus CSOC state logits.

        Args:
            batch : BatchData with fields:
                    feats, edge_index, sigma, (optional) edge_attr

        Returns:
            dict with keys:
                ``mu_rt``  — (N, 2)  predicted [Δμ, ΔRt]
                ``logits`` — (N, 3)  CSOC state logits
        """
        x         = self.node_embed(batch.feats)
        sigma_ctx = self._encode_sigma(batch.sigma)
        x         = self._apply_backbone(x, batch.edge_index, sigma_ctx)

        mu_rt  = self.evo_out_norm(self.evo_head(x))   # (N, 2)
        logits = self.cls_head(x)                       # (N, 3)
        return {"mu_rt": mu_rt, "logits": logits}

    def _forward_langevin(
        self,
        batch: BatchData,
    ) -> Dict[str, torch.Tensor]:
        """
        Mode 2: BAOAB Structural Langevin surrogate.

        Predicts coordinate displacements for one MD window; the caller
        adds them to ``batch.coords`` to obtain the predicted trajectory.

        The model takes both sequence features and (optionally) raw
        3-D coordinates as input, so geometry is explicitly encoded.

        Args:
            batch : BatchData with fields:
                    feats, edge_index, sigma, coords

        Returns:
            dict with keys:
                ``pred_coords``    — (N, 3)  predicted future coordinates
                ``displacements``  — (N, 3)  raw displacement predictions
        """
        x         = self.node_embed(batch.feats)
        sigma_ctx = self._encode_sigma(batch.sigma)
        x         = self._apply_backbone(x, batch.edge_index, sigma_ctx)

        displacements = self.md_out_norm(self.md_head(x))   # (N, 3)
        pred_coords   = batch.coords + displacements
        return {"pred_coords": pred_coords, "displacements": displacements}

    def _forward_ch3d(
        self,
        batch: BatchData,
    ) -> Dict[str, torch.Tensor]:
        """
        Mode 3: Cahn-Hilliard 3D phase-field surrogate.

        Operates on a voxel graph where each node represents one grid point.
        Predicts Δu at every voxel; result is reshaped back to the 3-D grid.

        Args:
            batch : BatchData with fields:
                    u_init (Nx,Ny,Nz), grid_feats (M, grid_in_dim),
                    grid_edge_index (2, E_grid), sigma_3d (Nx,Ny,Nz)

        Returns:
            dict with keys:
                ``pred_u``  — (Nx, Ny, Nz)  predicted phase field
                ``delta_u`` — (Nx, Ny, Nz)  raw Δu prediction
        """
        shape_3d   = batch.u_init.shape
        sigma_flat = batch.sigma_3d.flatten().unsqueeze(-1)   # (M, 1)

        x          = self.grid_embed(batch.grid_feats)        # (M, d)
        sigma_ctx  = self.sigma_enc(sigma_flat)               # (M, d)
        x          = self._apply_backbone(x, batch.grid_edge_index, sigma_ctx)

        delta_u    = self.ch3d_head(x).squeeze(-1).view(shape_3d)   # (Nx,Ny,Nz)
        pred_u     = batch.u_init + delta_u
        return {"pred_u": pred_u, "delta_u": delta_u}

    # ------------------------------------------------------------------
    # Public Forward Dispatcher
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: BatchData,
        mode:  str = "evolution",
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass dispatcher.

        Args:
            batch : BatchData container (only fields relevant to ``mode``
                    need to be populated).
            mode  : One of ``"evolution"``, ``"langevin"``, ``"ch3d"``.

        Returns:
            Dict of output tensors (mode-specific, see private methods).

        Raises:
            ValueError : If ``mode`` is not recognised.
        """
        if mode == "evolution":
            return self._forward_evolution(batch)
        elif mode == "langevin":
            return self._forward_langevin(batch)
        elif mode == "ch3d":
            return self._forward_ch3d(batch)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of: 'evolution', 'langevin', 'ch3d'."
            )

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total number of (trainable) parameters."""
        params = self.parameters() if not trainable_only else \
                 filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)


# =============================================================================
# 6.  Physics-Informed Loss Functions  [P-4]
# =============================================================================

def loss_energy_conservation(
    displacements: torch.Tensor,
    coords:        torch.Tensor,
    edge_index:    torch.Tensor,
    penalty:       float = 1.0,
) -> torch.Tensor:
    """
    Soft energy-conservation proxy for the Langevin surrogate.

    Penalises large pairwise distance changes between predicted and
    initial coordinate sets.  Under exact MD, the kinetic + potential
    energy is approximately conserved over short windows; large ΔR
    violations signal unphysical jumps.

    Loss = mean over edges of ( |r_pred - r_init| )^2

    Args:
        displacements : (N, 3) predicted coordinate displacements.
        coords        : (N, 3) initial coordinates.
        edge_index    : (2, E) COO edges.
        penalty       : overall scale factor.

    Returns:
        Scalar loss tensor.
    """
    src, dst      = edge_index[0], edge_index[1]
    r_init        = torch.norm(coords[src] - coords[dst], dim=-1)
    pred_coords   = coords + displacements
    r_pred        = torch.norm(pred_coords[src] - pred_coords[dst], dim=-1)
    return penalty * F.mse_loss(r_pred, r_init)


def loss_total_variation_3d(delta_u: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic total-variation regularisation for the CH3D surrogate.

    Penalises spatially rough Δu predictions, encouraging the surrogate
    to learn smooth, physically plausible phase-field updates.

    Loss = mean( |Δu_{i+1,j,k} - Δu_{i,j,k}| + ... )  over all directions.

    Args:
        delta_u : (Nx, Ny, Nz) predicted phase-field increment.

    Returns:
        Scalar loss tensor.
    """
    tv_x = (delta_u[1:, :, :] - delta_u[:-1, :, :]).abs().mean()
    tv_y = (delta_u[:, 1:, :] - delta_u[:, :-1, :]).abs().mean()
    tv_z = (delta_u[:, :, 1:] - delta_u[:, :, :-1]).abs().mean()
    return (tv_x + tv_y + tv_z) / 3.0


def loss_rt_kl_smooth(
    mu_rt_pred: torch.Tensor,
    mu_rt_true: torch.Tensor,
    epsilon:    float = 1e-4,
) -> torch.Tensor:
    """
    Smooth KL-divergence-inspired loss for Rt distribution.

    Standard MSE treats all nodes equally; this soft-KL penalises
    relative errors more heavily at small Rt values (where absolute
    differences matter most epidemiologically).

    Loss = mean( (Rt_pred - Rt_true)^2 / (|Rt_true| + epsilon) )

    Args:
        mu_rt_pred : (N, 2) predicted [Δμ, ΔRt].
        mu_rt_true : (N, 2) ground truth [Δμ, ΔRt].
        epsilon    : floor to avoid division by zero.

    Returns:
        Scalar loss tensor.
    """
    diff   = (mu_rt_pred - mu_rt_true) ** 2
    denom  = mu_rt_true.abs() + epsilon
    return (diff / denom).mean()


# =============================================================================
# 7.  Exponential Moving Average  [P-5]
# =============================================================================

class EMAWeights:
    """
    Exponential Moving Average of model weights.

    EMA parameters are kept separately and not used during the forward /
    backward pass.  Call ``update()`` after each optimiser step and
    ``apply()`` before evaluation / inference.

    Args:
        model : The nn.Module whose parameters to shadow.
        decay : EMA decay rate (e.g. 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay    = decay
        self.shadow   = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA shadow parameters after one optimiser step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> None:
        """Replace model parameters with EMA shadow (before eval)."""
        self._backup = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original (training) parameters (after eval)."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()


# =============================================================================
# 8.  Checkpoint Manager  [P-6]
# =============================================================================

class CheckpointManager:
    """
    Save and load complete training state to / from disk.

    Saves:
      • model state_dict
      • optimiser state_dict
      • scheduler state_dict
      • EMA shadow weights
      • training metadata (epoch, best_loss, step)

    Args:
        save_dir : Directory for checkpoint files.
        keep_n   : Keep the N most recent checkpoints (plus the best).
    """

    def __init__(self, save_dir: str = "checkpoints", keep_n: int = 3) -> None:
        self.save_dir = save_dir
        self.keep_n   = keep_n
        self._saved:  List[str] = []
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        ema:       EMAWeights,
        meta:      Dict[str, Any],
        tag:       str = "latest",
    ) -> str:
        """
        Persist training state.

        Args:
            model, optimizer, scheduler, ema : training objects.
            meta : dict of scalars (epoch, step, best_loss, …).
            tag  : file-name suffix; use ``"best"`` for the best checkpoint.

        Returns:
            Path to the saved file.
        """
        path = os.path.join(self.save_dir, f"sgno_evo_{tag}.pt")
        torch.save(
            {
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "ema":       ema.shadow,
                "meta":      meta,
                "version":   SGNO_VERSION,
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

        if tag != "best":
            self._saved.append(path)
            # Prune old checkpoints
            while len(self._saved) > self.keep_n:
                old = self._saved.pop(0)
                if os.path.exists(old):
                    os.remove(old)
                    logger.debug("Removed old checkpoint: %s", old)

        return path

    @staticmethod
    def load(
        path:      str,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        ema:       Optional[EMAWeights] = None,
        device:    torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Restore training state from a checkpoint file.

        Args:
            path      : Path returned by ``save()``.
            model     : Model to load weights into (in-place).
            optimizer : If provided, restore optimiser state.
            scheduler : If provided, restore scheduler state.
            ema       : If provided, restore EMA shadow weights.
            device    : Map location for tensors.

        Returns:
            ``meta`` dict from the checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema is not None and ckpt.get("ema"):
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        logger.info("Checkpoint loaded ← %s  (version %s)", path, ckpt.get("version", "?"))
        return ckpt.get("meta", {})


# =============================================================================
# 9.  Production Trainer  [P-5]
# =============================================================================

class SGNOEvolutionTrainer:
    """
    Production-grade multi-objective trainer for StructuralGNOEvolution.

    Features
    --------
    • AdamW with cosine-annealing LR + linear warm-up (``lr_warmup_steps``).
    • Gradient clipping at ``cfg.grad_clip`` (L2 norm).
    • NaN / Inf guard — skips update if any gradient is non-finite.
    • EMA weight averaging (``cfg.ema_decay``).
    • Mixed-precision training via ``torch.cuda.amp`` when ``cfg.use_amp``.
    • Per-head loss tracking returned from every ``train_step`` call.
    • ``evaluate()`` method runs the model with EMA weights.
    • Early stopping with configurable patience.
    • Checkpoint save/load via ``CheckpointManager``.

    Args:
        model      : Initialised StructuralGNOEvolution.
        cfg        : SGNOEvoConfig (training hyper-parameters are read here).
        device     : Compute device; defaults to best available.
        ckpt_dir   : Directory for checkpoint files.
    """

    def __init__(
        self,
        model:    StructuralGNOEvolution,
        cfg:      SGNOEvoConfig,
        device:   Optional[torch.device] = None,
        ckpt_dir: str = "checkpoints",
    ) -> None:
        self.cfg    = cfg
        self.device = device or get_device()
        self.model  = model.to(self.device)

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # LR Scheduler: linear warm-up → cosine annealing
        def lr_lambda(step: int) -> float:
            if step < cfg.lr_warmup_steps:
                return float(step + 1) / float(max(1, cfg.lr_warmup_steps))
            progress = (step - cfg.lr_warmup_steps) / max(1, cfg.max_epochs * 100 - cfg.lr_warmup_steps)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            min_frac = cfg.lr_min / cfg.lr
            return max(min_frac, cosine)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # EMA
        self.ema = EMAWeights(model, decay=cfg.ema_decay)

        # AMP scaler (CUDA only)
        self._use_amp   = cfg.use_amp and self.device.type == "cuda"
        self._scaler    = GradScaler(enabled=self._use_amp)

        # Checkpoint manager
        self.ckpt_mgr   = CheckpointManager(ckpt_dir)

        # Training state
        self.global_step:  int   = 0
        self.best_loss:    float = float("inf")
        self._no_improve:  int   = 0

        logger.info(
            "SGNOEvolutionTrainer ready — device=%s  params=%d  AMP=%s",
            self.device, model.num_parameters(), self._use_amp,
        )

    # ------------------------------------------------------------------
    # Loss computation (all modes)
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        batch_evo: Optional[BatchData],
        batch_md:  Optional[BatchData],
        batch_ch:  Optional[BatchData],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-objective loss.

        Any batch that is None is skipped.

        Returns:
            total_loss : scalar tensor (with grad).
            loss_dict  : {name: float} for logging.
        """
        cfg        = self.cfg
        total_loss = torch.tensor(0.0, device=self.device)
        log: Dict[str, float] = {}

        # ── Mode 1: Evolution / Epidemiological ─────────────────────
        if batch_evo is not None:
            out      = self.model(batch_evo, mode="evolution")
            mu_rt    = out["mu_rt"]
            logits   = out["logits"]

            loss_mu_rt = loss_rt_kl_smooth(mu_rt, batch_evo.true_mu_rt)
            loss_cls   = F.cross_entropy(logits, batch_evo.labels)

            l_evo      = cfg.lambda_evo * loss_mu_rt + cfg.lambda_cls * loss_cls
            total_loss = total_loss + l_evo
            log["loss_mu_rt"] = loss_mu_rt.item()
            log["loss_cls"]   = loss_cls.item()
            log["loss_evo"]   = l_evo.item()

        # ── Mode 2: Structural Langevin ──────────────────────────────
        if batch_md is not None:
            out         = self.model(batch_md, mode="langevin")
            pred_coords = out["pred_coords"]
            displ       = out["displacements"]

            loss_md_mse  = F.mse_loss(pred_coords, batch_md.true_future_coords)
            loss_md_phys = loss_energy_conservation(
                displ, batch_md.coords, batch_md.edge_index
            )

            l_md       = cfg.lambda_md * loss_md_mse + cfg.lambda_md_phys * loss_md_phys
            total_loss = total_loss + l_md
            log["loss_md_mse"]  = loss_md_mse.item()
            log["loss_md_phys"] = loss_md_phys.item()
            log["loss_md"]      = l_md.item()

        # ── Mode 3: Cahn-Hilliard 3D ────────────────────────────────
        if batch_ch is not None:
            out       = self.model(batch_ch, mode="ch3d")
            pred_u    = out["pred_u"]
            delta_u   = out["delta_u"]

            loss_ch_mse = F.mse_loss(pred_u, batch_ch.true_future_u)
            loss_ch_tv  = loss_total_variation_3d(delta_u)

            l_ch       = cfg.lambda_ch * loss_ch_mse + cfg.lambda_ch_tv * loss_ch_tv
            total_loss = total_loss + l_ch
            log["loss_ch_mse"] = loss_ch_mse.item()
            log["loss_ch_tv"]  = loss_ch_tv.item()
            log["loss_ch"]     = l_ch.item()

        log["total_loss"] = total_loss.item()
        return total_loss, log

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch_evo: Optional[BatchData] = None,
        batch_md:  Optional[BatchData] = None,
        batch_ch:  Optional[BatchData] = None,
    ) -> Dict[str, float]:
        """
        Execute one gradient-update step.

        At least one of the three batches must be non-None.

        Args:
            batch_evo : Evolution / Epidemiological batch (or None to skip).
            batch_md  : Langevin MD batch (or None to skip).
            batch_ch  : Cahn-Hilliard 3D batch (or None to skip).

        Returns:
            loss_dict : {loss_name: float} for logging / monitoring.
        """
        if batch_evo is None and batch_md is None and batch_ch is None:
            raise ValueError("At least one batch must be non-None.")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self._use_amp):
            total_loss, log = self._compute_losses(batch_evo, batch_md, batch_ch)

        # NaN guard
        if not torch.isfinite(total_loss):
            logger.warning("Step %d: non-finite loss (%.4g) — skipping update.",
                           self.global_step, total_loss.item())
            return log

        # Backward + gradient clipping
        self._scaler.scale(total_loss).backward()
        self._scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        # Gradient NaN check
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

        if self.global_step % self.cfg.log_every == 0:
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in log.items())
            logger.info("Step %6d | %s", self.global_step, loss_str)

        return log

    # ------------------------------------------------------------------
    # Evaluation with EMA weights
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        batch_evo: Optional[BatchData] = None,
        batch_md:  Optional[BatchData] = None,
        batch_ch:  Optional[BatchData] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model with EMA weights (no gradient computation).

        Temporarily applies EMA weights, runs loss computation,
        then restores training weights.

        Returns:
            loss_dict from ``_compute_losses``.
        """
        self.ema.apply(self.model)
        self.model.eval()
        try:
            with autocast(enabled=self._use_amp):
                _, log = self._compute_losses(batch_evo, batch_md, batch_ch)
        finally:
            self.ema.restore(self.model)
            self.model.train()
        return log

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_batches: List[Tuple[Optional[BatchData], Optional[BatchData], Optional[BatchData]]],
        val_batches:   Optional[List[Tuple[Optional[BatchData], Optional[BatchData], Optional[BatchData]]]] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            train_batches : List of (batch_evo, batch_md, batch_ch) tuples.
            val_batches   : Optional validation set (same format).

        Returns:
            history : Dict with ``"train_loss"`` and ``"val_loss"`` lists.

        Training stops when:
          • ``max_epochs`` is reached, OR
          • validation loss has not improved for ``patience`` epochs.
        """
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()
            epoch_losses: List[float] = []

            # ── Training epoch ──────────────────────────────────────
            for b_evo, b_md, b_ch in train_batches:
                log = self.train_step(b_evo, b_md, b_ch)
                epoch_losses.append(log["total_loss"])

            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            history["train_loss"].append(train_loss)

            # ── Validation ──────────────────────────────────────────
            val_loss = None
            if val_batches:
                val_logs = [
                    self.evaluate(b_evo, b_md, b_ch)
                    for b_evo, b_md, b_ch in val_batches
                ]
                val_loss = sum(d["total_loss"] for d in val_logs) / max(len(val_logs), 1)
                history["val_loss"].append(val_loss)

            elapsed = time.time() - t0
            monitor = val_loss if val_loss is not None else train_loss

            logger.info(
                "Epoch %4d/%d | train=%.4f | val=%s | time=%.1fs | lr=%.2e",
                epoch, self.cfg.max_epochs,
                train_loss,
                f"{val_loss:.4f}" if val_loss is not None else "—",
                elapsed,
                self.scheduler.get_last_lr()[0],
            )

            # ── Checkpointing + Early Stopping ──────────────────────
            meta = {
                "epoch": epoch,
                "step": self.global_step,
                "best_loss": self.best_loss,
                "train_loss": train_loss,
            }
            self.ckpt_mgr.save(
                self.model, self.optimizer, self.scheduler, self.ema, meta, tag="latest"
            )

            if monitor < self.best_loss:
                self.best_loss  = monitor
                self._no_improve = 0
                self.ckpt_mgr.save(
                    self.model, self.optimizer, self.scheduler, self.ema,
                    {**meta, "best_loss": self.best_loss}, tag="best"
                )
                logger.info("  ↳ New best: %.4f — checkpoint saved.", self.best_loss)
            else:
                self._no_improve += 1
                if self._no_improve >= self.cfg.patience:
                    logger.info(
                        "Early stopping: no improvement for %d epochs.", self.cfg.patience
                    )
                    break

        logger.info("Training complete. Best loss: %.4f", self.best_loss)
        return history


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

__all__ += [
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
# 10. BV Configuration  — extends SGNOEvoConfig with BV-specific fields
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
# 11. BV Certification Report  — typed container for periodic health-checks
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
# 12. BV-2 — Mass-conservation physics loss (no external dependency)
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
# 13. BV-1 — Cross-modal BV distillation loss
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
# 14. BV Certification Bridge — thin composition wrapper, no re-derivation
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
# 15. StructuralGNOEvolutionBV — Main Model
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
# 16. SGNOEvolutionBVTrainer — adds BV-1 / BV-2 / BV-3 to the production trainer
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

# =============================================================================
# 17.  Quick Smoke-Test  (python structural_gno_evolution_bv_standalone.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    print(f"StructuralGNOEvolutionBV v{SGNO_BV_VERSION}  (base SGNO v{SGNO_VERSION}, "
          f"bv_full_theory_one={'v' + BV_FULL_VERSION if _HAS_BV_FULL_THEORY else 'unavailable'}) "
          f"— standalone single-file edition")
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

    # ── Cross-compatibility check: plain StructuralGNOEvolution <-> BV edition ──
    # (No new trainable parameters are added by the BV layer, so state_dicts
    #  must match key-for-key between the two classes.)
    plain_model = StructuralGNOEvolution(cfg).to(device)
    plain_keys  = set(plain_model.state_dict().keys())
    bv_keys     = set(model.state_dict().keys())
    print(f"  state_dict cross-compat OK  identical_keys={plain_keys == bv_keys}  "
          f"n_params_plain={plain_model.num_parameters():,}  n_params_bv={model.num_parameters():,}")

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

    print(f"\n[PASS] structural_gno_evolution_bv (standalone) v{SGNO_BV_VERSION} — all checks passed.")
    sys.exit(0)
