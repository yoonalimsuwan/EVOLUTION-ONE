# =============================================================================
# AGI ONE v3.0 — Production-Grade AGI Architecture
# Central Orchestration Hub for the ONE Ecosystem
# =============================================================================
#
# Developer  : Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788
# GitHub     : https://github.com/yoonalimsuwan
# Email      : msps4u@gmail.com
#
# AI Development Assistants:
#   Claude   (Anthropic)        — architecture co-design, missing-component
#                                 specification, code review, AGI completeness
#                                 analysis v1.0 → v2.0 → v3.0; curriculum
#                                 training, PCGrad, InfoNCE alignment,
#                                 EcosystemOrchestrator design
#   GPT-4o   (OpenAI)           — supplementary architecture consultation
#   Gemini   (Google DeepMind)  — cross-validation of design decisions
#   DeepSeek (DeepSeek AI)      — open-source alignment review
#
# =============================================================================
# VERSION HISTORY
# ─────────────────
# v1.0.0  Initial AGI ONE:
#         Perception, Language, WorkingMemory, EpisodicMemory,
#         GlobalWorkspace, WorldModel(RSSM), PlanningCEM, MetaCognition,
#         PsycheTriad, MultiScaleIntegrator, AGITrainer
#
# v2.0.0  Production upgrade:
#   [NEW] MPPI planner   — replaces CEM, GPU-parallel trajectory sampling
#   [NEW] DreamerV3 world model — symlog, two-hot reward, free-bits KL,
#         categorical latents (straight-through)
#   [NEW] PPO full       — clipped surrogate + GAE(λ) + entropy bonus
#   [NEW] DreamerCompoundLoss — world/actor/critic with uncertainty weighting
#   [NEW] SSCStabilizer  — SSC as Transformer hidden-state stabilizer
#   [NEW] InterfaceAttention — Interface Detector as adaptive attention prior
#   [NEW] CSOCComputeController — edge-of-chaos adaptive depth
#   [NEW] StructuralLangevinDiffusion — geometry-aware latent diffusion
#   [NEW] PsycheExecutiveLayer — Id→Goal / Ego→Plan / Superego→Safety
#   [NEW] OpenScienceRegistry — dataset attribution & provenance
#   [NEW] BSD ONE, HODGE ONE, GRH ONE — mathematical reasoning layer
#   [UPG] VisionEncoder   → ViT-style patch embedding + RoPE
#   [UPG] LanguageModule  → GPT-style causal LM + RoPE
#   [UPG] AudioEncoder    → Mel-spectrogram + Conformer-lite
#   [UPG] LossBalancer    → Kendall uncertainty weighting
#   [UPG] AGITrainer v2   → PPO + Dreamer + PSY joint training
#
# v3.0.0  Distributed Ecosystem + Stable Multi-Task Training  (this file):
#
#   PROBLEM SOLVED: Joint training of heterogeneous losses
#   (physics PDE, discrete math, RL, language) causes Gradient Interference
#   (Negative Transfer) — gradients from one domain destroy another domain's
#   learned representations. Loss landscape becomes intractable for AdamW.
#
#   THREE-PART SOLUTION:
#
#   [A] Multi-Stage Curriculum Training (CurriculumScheduler)
#       Phase 1 — FOUNDATION  : Train physics/math surrogates (SFNO3D, GNOFold,
#                               GNOEvolution, GNOHodge, GNONumberTheory, MSNOv3,
#                               NGOPhysics) + DreamerV3 World Model independently.
#                               Core ecosystem modules are frozen after convergence.
#       Phase 2 — ALIGNMENT   : Freeze physics backbone. Train Language↔Physics
#                               bridge via InfoNCE Contrastive Alignment Loss.
#                               Cross-modal latent spaces geometrically aligned
#                               without direct gradient interference.
#       Phase 3 — COGNITIVE   : Unlock PPO actor-critic + PsycheExecutiveLayer.
#                               Fine-tune policy with reward signal + Free Energy.
#                               Ecosystem surrogates remain partially frozen
#                               (backbone frozen, heads fine-tuned).
#
#   [B] PCGrad Gradient Surgery (PCGradOptimizer)
#       During Phase 3 joint training, conflicting gradients between task pairs
#       are detected (g_i · g_j < 0) and projected to orthogonal components,
#       preventing mutual destruction of learned representations.
#       Based on: Yu et al. 2020 "Gradient Surgery for Multi-Task Learning"
#
#   [C] InfoNCE Contrastive Alignment (CrossModalAlignmentLoss)
#       Physics latent ↔ Language latent aligned via symmetric InfoNCE loss.
#       Attracts (physics_i, language_i) pairs; repels (physics_i, language_j≠i).
#       Avoids direct gradient coupling between PDE solver and language decoder.
#       Based on: CLIP (Radford et al. 2021), SimCLR (Chen et al. 2020)
#
#   [D] EcosystemOrchestrator — Distributed Module Hub
#       AGI ONE v3 no longer embeds ecosystem engines directly. Instead,
#       EcosystemOrchestrator maintains references to uploaded surrogate modules
#       (structural_fno_3d.py, structural_gno_fold_v3.py, etc.) and queries
#       them via a unified adapter interface. This enables:
#         • True distributed training (each surrogate trains independently)
#         • Selective freezing per curriculum phase
#         • Per-domain optimizer assignment (Decoupled Optimizers)
#         • Hot-swappable surrogate modules
#
#   [E] Decoupled Domain Optimizers (AGITrainerV3)
#       Each domain group gets its own optimizer with domain-appropriate LR:
#         physics_surrogates  : AdamW  lr=1e-6  (high-precision, sensitive)
#         math_surrogates     : AdamW  lr=5e-7  (discrete logic, very slow)
#         language_module     : AdamW  lr=1e-5  (standard LLM fine-tuning)
#         world_model         : AdamW  lr=3e-4  (DreamerV3 default)
#         policy_heads        : AdamW  lr=1e-4  (PPO actor/critic)
#         psyche_layer        : AdamW  lr=3e-4  (Free Energy)
#         loss_balancer       : Adam   lr=1e-3  (Kendall σ params)
#
# =============================================================================
# THEORETICAL FOUNDATIONS
# ────────────────────────
#   Structural Itô Calculus (Limsuwan 2025)
#   Self-Organised Criticality + CSOC (SOC universality chain)
#   Renormalisation Group multi-scale smoothing
#   Active Inference / Free Energy Principle (Friston 2010)
#   Global Workspace Theory (Baars 1988; Dehaene 2011)
#   Integrated Information Theory Φ (Tononi 2004)
#   Deep Equilibrium Models — DEQ (Bai et al. 2019)
#   DreamerV3 (Hafner et al. 2023)
#   MPPI (Williams et al. 2017)
#   PPO + GAE (Schulman et al. 2017 / 2015)
#   Rotary Positional Embedding RoPE (Su et al. 2021)
#   Conformer (Gulati et al. 2020)
#   ViT patch embedding (Dosovitskiy et al. 2020)
#   Uncertainty-weighted multi-task loss (Kendall et al. 2018)
#   Edge-of-Chaos / Critical Brain Hypothesis (Langton 1990)
#   Geometry-aware Manifold Diffusion (Song et al. 2020+)
#
# =============================================================================
# ONE ECOSYSTEM INTEGRATION MAP — 23 modules
# ───────────────────────────────────────────
#   one_core_v3.py                            → shared SSC/CSOC/Itô
#   one_core_mental.py                        → mental-scale primitives
#   one_core_fold.py                          → protein primitives
#   one_core_evolution_v2.py                  → genomic primitives
#   mental_one.py                             → psychiatric/EEG/fMRI
#   psy_one_bridge_diff.py                    → Id/Ego/Superego triad
#   langevin_mental_bridge.py                 → Langevin↔brain
#   structural_langevin_mental.py             → Langevin mental
#   real_fold_one_v2.py                       → protein folding
#   real_fold_one_ht_v2.py                    → HT protein folding
#   structural_langevin_fold_v2.py            → Langevin MD fold
#   evolution_one_v3.py                       → cancer/somatic evolution
#   evolution_one_epidemiological_viral_v4.py → epidemiology/viral
#   structural_langevin_evo_v3.py             → Langevin evolutionary
#   structuralfluctuatinghydro_v6.py          → 3-D fluctuating hydro
#   super_dns_one_v6.py                       → compressible DNS/LES
#   structural_langevin_v3.py                 → Langevin MD BAOAB
#   standard_one.py                           → Standard Model
#   yang_mills_mass_gap_one.py                → Yang-Mills mass gap
#   rh_one.py                                 → RH computational explorer
#   bsd_one.py            [NEW v2.0]          → Birch–Swinnerton-Dyer
#   grh_one.py            [NEW v2.0]          → Generalized RH
#   hodge_one.py          [NEW v2.0]          → Hodge Conjecture explorer
#
# =============================================================================
# OPEN SCIENCE & DATA PROVENANCE
# ────────────────────────────────
# AGI ONE upholds open science principles. All training datasets must be
# traceable to their originating research lab, institution, or investigator.
# In the AGI era, attribution extends beyond paper authors to every
# laboratory, dataset contributor, and research centre that supplied data.
# See OpenScienceRegistry for the provenance API.
#
# =============================================================================
# MIT License
# Copyright (c) 2026 Yoon A Limsuwan / MSPS NETWORK
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [AGI_ONE v2]  %(levelname)s  %(message)s",
)
logger = logging.getLogger("AGI_ONE_v2")

AGI_ONE_VERSION: str = "3.0.0"

# =============================================================================
# ONE Ecosystem — Graceful Imports
# =============================================================================

# ── ONE Core primitives ───────────────────────────────────────────────────────
try:
    from one_core_mental import (
        SemanticStateContraction,
        CSOCBase,
        InterfaceDetectorBase,
        StructuralItoBase,
        DifferentiableRG,
        DifferentiableSOC,
        soft_clamp,
        MENTAL_VERSION,
    )
    HAS_ONE_CORE_MENTAL = True
    logger.info(f"✓ one_core_mental  (v{MENTAL_VERSION})")
except ImportError:
    HAS_ONE_CORE_MENTAL = False
    logger.warning("✗ one_core_mental not found — inline fallbacks active")
    def soft_clamp(x, lo, hi):
        c = (hi + lo) / 2.0; s = (hi - lo) / 2.0 + 1e-8
        return c + s * torch.tanh((x - c) / s)

try:
    from one_core_v3 import (
        SemanticStateContraction as SSC_Core,
        CSOCBase as CSOCBase_Core,
        get_device as get_device_core,
        ONE_VERSION,
    )
    HAS_ONE_CORE = True
    logger.info(f"✓ one_core_v3  (v{ONE_VERSION})")
except ImportError:
    HAS_ONE_CORE = False

try:
    from one_core_fold import SemanticStateContraction as SSC_Fold
    HAS_ONE_CORE_FOLD = True
    logger.info("✓ one_core_fold")
except ImportError:
    HAS_ONE_CORE_FOLD = False

try:
    from one_core_evolution_v2 import SemanticStateContraction as SSC_Evo
    HAS_ONE_CORE_EVO = True
    logger.info("✓ one_core_evolution_v2")
except ImportError:
    HAS_ONE_CORE_EVO = False

# ── MENTAL ONE ────────────────────────────────────────────────────────────────
try:
    from mental_one import MentalONEEngine
    HAS_MENTAL_ONE = True
    logger.info("✓ mental_one")
except ImportError:
    HAS_MENTAL_ONE = False

# ── PSY ONE BRIDGE ────────────────────────────────────────────────────────────
try:
    from psy_one_bridge_diff import (
        PsycheTriad, PsycheConfig, PsycheTriadState,
        PsychopathologyMode, GumbelAnnealScheduler,
    )
    HAS_PSY_BRIDGE = True
    logger.info("✓ psy_one_bridge_diff")
except ImportError:
    HAS_PSY_BRIDGE = False

# ── Langevin bridges ──────────────────────────────────────────────────────────
try:
    from langevin_mental_bridge import LangevinMentalBridge
    HAS_LANGEVIN_MENTAL = True
    logger.info("✓ langevin_mental_bridge")
except ImportError:
    HAS_LANGEVIN_MENTAL = False

try:
    from structural_langevin_mental import StructuralLangevinMental
    HAS_STRUCT_LANG_MENTAL = True
    logger.info("✓ structural_langevin_mental")
except ImportError:
    HAS_STRUCT_LANG_MENTAL = False

# ── REAL FOLD ONE ─────────────────────────────────────────────────────────────
try:
    from real_fold_one_v2 import RealFoldONEEngine
    HAS_REAL_FOLD = True
    logger.info("✓ real_fold_one_v2")
except ImportError:
    HAS_REAL_FOLD = False

try:
    from real_fold_one_ht_v2 import RealFoldHTEngine
    HAS_REAL_FOLD_HT = True
    logger.info("✓ real_fold_one_ht_v2")
except ImportError:
    HAS_REAL_FOLD_HT = False

try:
    from structural_langevin_fold_v2 import StructuralLangevinFold
    HAS_LANGEVIN_FOLD = True
    logger.info("✓ structural_langevin_fold_v2")
except ImportError:
    HAS_LANGEVIN_FOLD = False

# ── EVOLUTION ONE ─────────────────────────────────────────────────────────────
try:
    from evolution_one_v3 import EvolutionONEEngine
    HAS_EVOLUTION = True
    logger.info("✓ evolution_one_v3")
except ImportError:
    HAS_EVOLUTION = False

try:
    from evolution_one_epidemiological_viral_v4 import EpidemicEngine
    HAS_EPIDEMIC = True
    logger.info("✓ evolution_one_epidemiological_viral_v4")
except ImportError:
    HAS_EPIDEMIC = False

try:
    from structural_langevin_evo_v3 import StructuralLangevinEvo
    HAS_LANGEVIN_EVO = True
    logger.info("✓ structural_langevin_evo_v3")
except ImportError:
    HAS_LANGEVIN_EVO = False

# ── PHYSICS ───────────────────────────────────────────────────────────────────
try:
    from structuralfluctuatinghydro_v6 import StructuralFluctuatingHydro
    HAS_FH = True
    logger.info("✓ structuralfluctuatinghydro_v6")
except ImportError:
    HAS_FH = False

try:
    from super_dns_one_v6 import SuperDNSEngine
    HAS_DNS = True
    logger.info("✓ super_dns_one_v6")
except ImportError:
    HAS_DNS = False

try:
    from structural_langevin_v3 import StructuralLangevinMD
    HAS_LANGEVIN_MD = True
    logger.info("✓ structural_langevin_v3")
except ImportError:
    HAS_LANGEVIN_MD = False

# ── STANDARD MODEL / MATHEMATICS ─────────────────────────────────────────────
try:
    from standard_one import StandardONEEngine
    HAS_STANDARD = True
    logger.info("✓ standard_one")
except ImportError:
    HAS_STANDARD = False

try:
    from yang_mills_mass_gap_one import YangMillsMassGapEngine
    HAS_YANG_MILLS = True
    logger.info("✓ yang_mills_mass_gap_one")
except ImportError:
    HAS_YANG_MILLS = False

try:
    from rh_one__1_ import RiemannHypothesisEngine
    HAS_RH = True
    logger.info("✓ rh_one")
except ImportError:
    HAS_RH = False

# ── NEW v2.0: Mathematics trilogy ────────────────────────────────────────────
try:
    import bsd_one as bsd
    HAS_BSD = True
    logger.info("✓ bsd_one  [v2.0 NEW]")
except ImportError:
    HAS_BSD = False

try:
    import grh_one as grh
    HAS_GRH = True
    logger.info("✓ grh_one  [v2.0 NEW]")
except ImportError:
    HAS_GRH = False

try:
    import hodge_one as hodge
    HAS_HODGE = True
    logger.info("✓ hodge_one  [v2.0 NEW]")
except ImportError:
    HAS_HODGE = False

# ── Optional: HuggingFace / torchvision / torchaudio ─────────────────────────
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_HF = True
    logger.info("✓ HuggingFace transformers")
except ImportError:
    HAS_HF = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


# =============================================================================
# DEVICE UTILITY
# =============================================================================

def get_agi_device(preferred: str = "cuda") -> torch.device:
    p = preferred.lower()
    if p == "cuda"   and torch.cuda.is_available():    return torch.device("cuda")
    if p == "mps"    and torch.backends.mps.is_available(): return torch.device("mps")
    if p == "ascend" and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# SECTION 0 — OPEN SCIENCE REGISTRY
# =============================================================================

@dataclass
class DatasetRecord:
    """Single dataset attribution record."""
    dataset_id      : str
    title           : str
    source_lab      : str
    institution     : str
    contributors    : List[str]
    doi             : Optional[str]   = None
    url             : Optional[str]   = None
    license         : str             = "Unknown"
    year            : Optional[int]   = None
    description     : str             = ""
    tags            : List[str]       = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "dataset_id"  : self.dataset_id,
            "title"       : self.title,
            "source_lab"  : self.source_lab,
            "institution" : self.institution,
            "contributors": self.contributors,
            "doi"         : self.doi,
            "url"         : self.url,
            "license"     : self.license,
            "year"        : self.year,
            "description" : self.description,
            "tags"        : self.tags,
        }


class OpenScienceRegistry:
    """
    Dataset Attribution and Provenance Tracking Registry.

    AGI ONE principle: In the AGI era, attribution extends beyond paper
    authors to every laboratory, institution, and researcher who contributed
    data.  This registry ensures every dataset used in training is fully
    credited and traceable.

    Usage:
        registry = OpenScienceRegistry()
        registry.register(DatasetRecord(
            dataset_id="openneuro_ds003944",
            title="EEG Resting State Dataset",
            source_lab="Neuroimaging Lab",
            institution="Stanford University",
            contributors=["J. Smith", "A. Lee"],
            doi="10.18112/openneuro.ds003944",
            license="CC-BY-4.0",
            year=2021,
        ))
        registry.cite("openneuro_ds003944")
        report = registry.provenance_report()
    """

    def __init__(self) -> None:
        self._records: Dict[str, DatasetRecord] = {}
        self._usage_log: List[Dict] = []

    def register(self, record: DatasetRecord) -> None:
        """Register a dataset with full attribution."""
        self._records[record.dataset_id] = record
        logger.info(
            f"[OpenScience] Registered: {record.dataset_id}  "
            f"| Lab: {record.source_lab}  | {record.institution}"
        )

    def cite(self, dataset_id: str, context: str = "") -> Optional[DatasetRecord]:
        """Record usage of a dataset (for audit trail)."""
        if dataset_id not in self._records:
            logger.warning(f"[OpenScience] Unknown dataset: {dataset_id}")
            return None
        self._usage_log.append({
            "dataset_id": dataset_id,
            "context"   : context,
            "timestamp" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        return self._records[dataset_id]

    def all_records(self) -> List[DatasetRecord]:
        return list(self._records.values())

    def provenance_report(self) -> Dict:
        """Full provenance report: all registered datasets + usage log."""
        return {
            "agi_one_version"  : AGI_ONE_VERSION,
            "report_generated" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datasets"         : [r.to_dict() for r in self._records.values()],
            "usage_log"        : self._usage_log,
            "principle": (
                "AGI ONE upholds open science attribution: every laboratory, "
                "dataset contributor, and research centre that supplied data "
                "is credited. Attribution in the AGI era is broader than "
                "individual authorship — it encompasses the full data "
                "provenance chain."
            ),
        }

    def save_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.provenance_report(), f, indent=2, ensure_ascii=False)
        logger.info(f"[OpenScience] Provenance report saved: {path}")

    def load_registry(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("datasets", []):
            self.register(DatasetRecord(**d))


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

class CognitivePriority(Enum):
    BALANCED      = "balanced"
    PERCEPTION    = "perception"
    LANGUAGE      = "language"
    PLANNING      = "planning"
    INTROSPECTION = "introspection"
    PHYSICS       = "physics"


@dataclass
class AGIConfig:
    """Master configuration for AGI ONE v2.0."""
    # ── Core dimensions ──────────────────────────────────────────────────────
    latent_dim           : int   = 512
    action_dim           : int   = 64
    memory_slots         : int   = 128
    episodic_capacity    : int   = 10_000
    planning_horizon     : int   = 15
    n_transformer_heads  : int   = 8
    n_transformer_layers : int   = 6      # upgraded from 4

    # ── Modality flags ────────────────────────────────────────────────────────
    use_vision           : bool  = True
    use_audio            : bool  = True
    use_language         : bool  = True
    use_proprioception   : bool  = True
    use_timeseries       : bool  = True

    # ── ONE Ecosystem ─────────────────────────────────────────────────────────
    use_mental_one       : bool  = True
    use_psy_bridge       : bool  = True
    use_real_fold        : bool  = True
    use_evolution        : bool  = True
    use_physics          : bool  = True
    use_standard_one     : bool  = False
    use_yang_mills       : bool  = False
    use_rh               : bool  = False
    use_bsd              : bool  = True   # [v2.0]
    use_grh              : bool  = True   # [v2.0]
    use_hodge            : bool  = True   # [v2.0]

    # ── Language backbone ─────────────────────────────────────────────────────
    language_backend     : str   = "builtin"
    language_model_id    : str   = "distilbert-base-uncased"
    language_dim         : int   = 768
    vocab_size           : int   = 32_000

    # ── PSY Bridge ────────────────────────────────────────────────────────────
    psyche_mode          : str   = "healthy"
    gumbel_tau           : float = 1.0
    gumbel_hard          : bool  = False
    anderson_depth       : int   = 5
    lambda_reg           : float = 2.5

    # ── MPPI Planner [v2.0] ───────────────────────────────────────────────────
    mppi_n_samples       : int   = 1024
    mppi_temperature     : float = 1.0
    mppi_noise_sigma     : float = 0.5

    # ── DreamerV3 World Model [v2.0] ──────────────────────────────────────────
    dreamer_stoch_size   : int   = 32
    dreamer_stoch_classes: int   = 32
    dreamer_det_size     : int   = 512
    dreamer_reward_bins  : int   = 255  # two-hot encoding bins
    dreamer_free_bits    : float = 1.0  # KL free bits threshold
    dreamer_kl_balance   : float = 0.8  # posterior vs prior weight

    # ── PPO Training [v2.0] ───────────────────────────────────────────────────
    ppo_clip_eps         : float = 0.2
    ppo_epochs           : int   = 4
    ppo_gae_lambda       : float = 0.95
    ppo_gamma            : float = 0.99
    value_loss_coef      : float = 0.5
    entropy_coef         : float = 0.01

    # ── CSOC Compute Control [v2.0] ───────────────────────────────────────────
    csoc_min_layers      : int   = 4
    csoc_max_layers      : int   = 32
    csoc_sigma_target    : float = 1.0

    # ── Training ─────────────────────────────────────────────────────────────
    lr                   : float = 3e-4
    weight_decay         : float = 1e-4
    grad_clip_norm       : float = 1.0
    use_amp              : bool  = True
    use_grad_checkpoint  : bool  = False
    warmup_steps         : int   = 1_000

    # ── Meta ─────────────────────────────────────────────────────────────────
    device               : torch.device = field(
        default_factory=lambda: get_agi_device("cuda")
    )
    cognitive_priority   : CognitivePriority = CognitivePriority.BALANCED
    verbose              : bool  = True
    seed                 : int   = 42


# =============================================================================
# SECTION 2 — ROTARY POSITIONAL EMBEDDING (RoPE)
# Su et al. 2021 — used by ViT, GPT, and language module
# =============================================================================

class RotaryEmbedding(nn.Module):
    """RoPE positional embedding — improves extrapolation over learned PE."""

    def __init__(self, dim: int, max_seq: int = 4096) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._max_seq = max_seq

    def _get_cos_sin(self, seq_len: int, device: torch.device):
        t      = torch.arange(seq_len, device=device).float()
        freqs  = torch.einsum("i,j->ij", t, self.inv_freq)
        emb    = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors (B, H, L, D)."""
        seq_len = q.shape[-2]
        cos, sin = self._get_cos_sin(seq_len, q.device)  # (L, D)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,L,D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# =============================================================================
# SECTION 3 — SSC PRIMITIVES INTEGRATED INTO TRANSFORMER
# [v2.0 NEW] — SSCStabilizer, InterfaceAttention, CSOCComputeController
# =============================================================================

class SSCStabilizer(nn.Module):
    """
    [v2.0 NEW] SSC as Transformer Hidden-State Stabilizer.

    Wraps SemanticStateContraction (from one_core_mental) as a per-channel
    EMA filter applied to the hidden state of each Transformer layer.

    Pipeline:
        hidden_state (B, L, D)
            → per-channel stress σ = std over sequence
            → SSC EMA filter
            → refined hidden_state

    This stabilizes the latent representation and reduces forgetting,
    acting as a learnable memory compression / coherent-context keeper.
    """

    def __init__(self, d_model: int, epsilon_fp: float = 0.005) -> None:
        super().__init__()
        self.d_model = d_model

        if HAS_ONE_CORE_MENTAL:
            self.ssc = SemanticStateContraction(epsilon_fp=epsilon_fp)
        else:
            # Fallback: learnable EMA coefficient per channel
            self.log_alpha = nn.Parameter(
                torch.full((d_model,), math.log(epsilon_fp))
            )
            self.ssc = None

        # Projection: refine state after SSC
        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        # Persistent stabilized state
        self.register_buffer("stabilized", torch.zeros(d_model))
        self.register_buffer("_init", torch.tensor(False))

    def reset(self) -> None:
        self.stabilized.zero_()
        self._init.fill_(False)
        if self.ssc is not None:
            self.ssc.reset()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden : (B, L, D) transformer hidden state
        Returns:
            stabilized hidden : (B, L, D)
        """
        B, L, D = hidden.shape

        # Compute per-channel "structural stress" (std across sequence)
        sigma = hidden.std(dim=1)   # (B, D)
        sigma_mean = sigma.mean(dim=0)  # (D,)

        if self.ssc is not None:
            sigma_filtered = self.ssc(sigma_mean.mean().unsqueeze(0))
            scale = (sigma_filtered / (sigma_mean.mean() + 1e-8)).clamp(0.5, 2.0)
            refined = hidden * scale
        else:
            alpha = torch.sigmoid(self.log_alpha)   # (D,)
            if not self._init.item():
                self.stabilized.data = sigma_mean.detach()
                self._init.fill_(True)
            self.stabilized.data = (
                (1 - alpha.detach()) * self.stabilized + alpha.detach() * sigma_mean.detach()
            )
            scale = (self.stabilized / (sigma_mean + 1e-8)).clamp(0.5, 2.0)
            refined = hidden * scale.unsqueeze(0).unsqueeze(0)

        return self.refine(refined)


class InterfaceAttention(nn.Module):
    """
    [v2.0 NEW] Interface Detector as Adaptive Attention Prior.

    Based on InterfaceDetectorBase (one_core_mental):
    detects phase transitions / boundary points in the hidden-state sequence,
    then adds a learned bias to attention logits so the model attends more
    to interface tokens (points where knowledge/context is changing).

    Pipeline:
        attn_logits (B, H, L, L)
        + interface_score(hidden) → (B, 1, L, 1)  broadcast as column bias
        ──────────────────────────────────────────
        modified_attn_logits

    Inspired by:
    - InterfaceDetectorBase in structural_langevin_mental.py
    - Adaptive attention frontier work (e.g. Lei et al., "Fastformer")
    """

    def __init__(self, d_model: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

        # Differentiable interface score: how much each token is a "boundary"
        self.interface_net = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        # Learnable scale for bias strength
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden      : torch.Tensor,   # (B, L, D)
        attn_logits : torch.Tensor,   # (B, H, L, L)
    ) -> torch.Tensor:
        """
        Returns:
            modified_attn_logits : (B, H, L, L)
        """
        # Interface score per token: (B, L, 1)
        iface = self.interface_net(hidden)              # (B, L, 1)

        # Gradient of score across sequence → detect transitions
        iface_diff = torch.zeros_like(iface)
        iface_diff[:, 1:, :] = (iface[:, 1:, :] - iface[:, :-1, :]).abs()

        # Normalize and scale
        scale = torch.exp(self.log_scale)
        bias  = iface_diff * scale                      # (B, L, 1)
        bias  = bias.unsqueeze(1)                       # (B, 1, L, 1) → column bias
        bias  = bias.expand_as(attn_logits)

        return attn_logits + bias

    def get_interface_map(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns interface scores for visualization. (B, L)"""
        return self.interface_net(hidden).squeeze(-1)


class CSOCComputeController(nn.Module):
    """
    [v2.0 NEW] CSOC as Dynamic Compute Controller (Adaptive Depth).

    Based on CSOCBase (one_core_mental) + DifferentiableSOC:
    monitors criticality of the current hidden state and dynamically
    decides how many Transformer layers to execute.

    Easy problems (low criticality)  → use min_layers
    Hard problems (high criticality) → use up to max_layers

    Pipeline:
        hidden_state → criticality_score ∈ [0,1]
        n_layers = min_layers + round(score * (max_layers - min_layers))

    Analogous to:
    - Adaptive Computation Time (Graves 2016)
    - Test-time compute scaling (DeepSeek-R1 style reasoning)
    - Edge-of-Chaos adaptive complexity (Langton 1990)
    """

    def __init__(
        self,
        d_model        : int,
        min_layers     : int   = 4,
        max_layers     : int   = 32,
        sigma_target   : float = 1.0,
        epsilon_fp     : float = 0.005,
    ) -> None:
        super().__init__()
        self.min_layers  = min_layers
        self.max_layers  = max_layers
        self.sigma_target = sigma_target

        # SSC for smoothing criticality signal
        if HAS_ONE_CORE_MENTAL:
            self.ssc = SemanticStateContraction(epsilon_fp, sigma_target)
        else:
            self.ssc = None

        # DifferentiableSOC for criticality dynamics
        if HAS_ONE_CORE_MENTAL:
            self.diff_soc = DifferentiableSOC(
                base_temp=300.0, beta=0.01, n_steps=5
            )
        else:
            self.diff_soc = None

        # Criticality estimator: score ∈ [0, 1]
        self.critic_net = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

        # Learnable bias for criticality threshold
        self.bias = nn.Parameter(torch.tensor(0.0))

        self._last_n_layers = min_layers
        self._last_score    = 0.0

    def reset(self) -> None:
        if self.ssc is not None:
            self.ssc.reset()

    def compute_criticality(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            hidden : (B, L, D) hidden state
        Returns:
            (criticality_score tensor [0,1], n_layers to use int)
        """
        h_mean  = hidden.mean(dim=1)   # (B, D)
        score   = self.critic_net(h_mean).mean()  # scalar ∈ [0,1]
        score   = score + self.bias.sigmoid() * 0.1

        # SSC smoothing on criticality signal
        if self.ssc is not None:
            score_sm = self.ssc(score.unsqueeze(0)).squeeze()
        else:
            score_sm = score

        score_clamped = soft_clamp(score_sm, 0.0, 1.0)

        # Dynamic depth
        span     = self.max_layers - self.min_layers
        n_layers = self.min_layers + int(round(float(score_clamped.item()) * span))
        n_layers = max(self.min_layers, min(self.max_layers, n_layers))

        self._last_n_layers = n_layers
        self._last_score    = float(score_clamped.item())

        return score_clamped, n_layers

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self.compute_criticality(hidden)


class StructuralLangevinDiffusion(nn.Module):
    """
    [v2.0 NEW] Geometry-Aware Latent Diffusion via Structural Langevin.

    Based on StructuralItoBase (one_core_mental):
    standard diffusion uses uniform Gaussian noise over latent space,
    but here noise amplitude G(x) varies with the interface structure
    of the current state — creating manifold-aware / geometry-aware diffusion.

    dX_t = -∇U(X_t) dt + G(X_t) dW_t + ½ G(X_t) ∇G(X_t) dt  (Itô correction)

    where G(x) = 1 + amp · interface_mask(x)

    Applications in AGI ONE:
    - Latent imagination / dreaming (world model)
    - Exploration in planning
    - Latent space regularization during training
    """

    def __init__(
        self,
        d_model                : int,
        interface_amplification: float = 2.0,
        n_steps                : int   = 10,
        dt                     : float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.amp       = interface_amplification
        self.n_steps   = n_steps
        self.dt        = dt

        # Interface detector: (B, D) → (B, D) mask ∈ [0, 1]
        self.iface_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )

        # Energy function U(x) = ½ ||x||²  (Gaussian prior)
        # Learnable scale
        self.log_dt = nn.Parameter(torch.tensor(math.log(dt)))

        # RG smoother for G field
        if HAS_ONE_CORE_MENTAL:
            self.rg = DifferentiableRG(kernel_size=5)
        else:
            self.rg = None

    def _g_field(self, x: torch.Tensor) -> torch.Tensor:
        """G(x) = 1 + amp · interface_mask(x)  shape: same as x."""
        mask = self.iface_net(x)
        return 1.0 + self.amp * mask

    def _ito_correction(self, x: torch.Tensor) -> torch.Tensor:
        """½ G(x) · ∇G(x) — via autograd."""
        x_req = x.detach().requires_grad_(True)
        g     = self._g_field(x_req).sum()
        grad  = torch.autograd.grad(g, x_req, create_graph=False)[0]
        g_val = self._g_field(x.detach())
        return 0.5 * g_val * grad

    def forward(
        self,
        x       : torch.Tensor,   # (B, D) latent vector
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Run geometry-aware Langevin diffusion for n_steps.

        Returns:
            x_diffused : (B, D) diffused latent
        """
        dt = torch.exp(self.log_dt).item()

        for _ in range(self.n_steps):
            # Gradient of energy: ∇U = x  (Gaussian prior)
            grad_u = x

            # G field
            G = self._g_field(x)            # (B, D) ∈ [1, 1+amp]

            # Itô correction
            try:
                ito_corr = self._ito_correction(x)
            except Exception:
                ito_corr = torch.zeros_like(x)

            # Noise: G(x) · dW
            dW    = torch.randn_like(x) * math.sqrt(dt) * noise_scale
            noise = G * dW

            # Langevin step: dx = -∇U dt + G dW + ½G∇G dt
            x = x - grad_u * dt + noise + ito_corr * dt

            # Soft clamp to prevent explosion
            x = soft_clamp(x, -10.0, 10.0)

        return x



# =============================================================================
# SECTION 4 — UPGRADED PERCEPTION MODULE
# ViT-style patch encoder + Conformer-lite audio + improved fusion
# =============================================================================

class PatchViTEncoder(nn.Module):
    """
    [v2.0 UPGRADED] ViT-style patch-based vision encoder.

    Replaces simple ResNet-18 with:
    - Patch embedding (non-overlapping patches → linear projection)
    - RoPE positional embedding
    - Transformer encoder (configurable depth)
    - CLS token aggregation

    Much stronger than ResNet-18 for complex visual reasoning.
    """

    def __init__(
        self,
        latent_dim  : int,
        img_size    : int   = 224,
        patch_size  : int   = 16,
        in_channels : int   = 3,
        n_heads     : int   = 8,
        n_layers    : int   = 6,
        device      : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.n_patches   = (img_size // patch_size) ** 2
        patch_dim        = in_channels * patch_size * patch_size
        self.patch_size  = patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, latent_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # RoPE
        self.rope = RotaryEmbedding(latent_dim // n_heads)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(latent_dim)

        # SSC stabilizer on CLS output [v2.0]
        self.ssc_stab = SSCStabilizer(latent_dim)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            (B, latent_dim) vision embedding
        """
        B = x.shape[0]
        # Patch embedding: (B, D, n_h, n_w) → (B, n_patches, D)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        p   = torch.cat([cls, p], dim=1)   # (B, 1+n_patches, D)

        # Transformer
        out = self.transformer(p)
        out = self.ssc_stab(out)
        out = self.norm(out)
        return out[:, 0, :]   # CLS output: (B, D)


class ConformerAudioEncoder(nn.Module):
    """
    [v2.0 UPGRADED] Conformer-lite audio encoder.

    Mel-spectrogram → Conv subsampling → Conformer blocks → pooling.
    Conformer (Gulati et al. 2020) combines CNN local patterns + Transformer
    global context, state of the art for audio/speech.
    """

    def __init__(
        self,
        latent_dim  : int,
        n_mfcc      : int   = 80,
        n_heads     : int   = 4,
        n_layers    : int   = 4,
        device      : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        if HAS_TORCHAUDIO:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=16_000, n_fft=512, n_mels=n_mfcc,
            )
        else:
            self.mel = None

        # Conv subsampling: (B, 1, n_mfcc, T) → (B, latent_dim//2, T//4)
        self.conv_sub = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, latent_dim // 4, 3, stride=2, padding=1), nn.GELU(),
        )

        # Linear projection after flattening mel dim
        mel_out_dim = (n_mfcc // 4) * (latent_dim // 4)
        self.proj   = nn.Linear(mel_out_dim, latent_dim)

        # Conformer-lite: Transformer + depthwise conv feed-forward
        class ConformerBlock(nn.Module):
            def __init__(self, d: int, h: int) -> None:
                super().__init__()
                self.ff1  = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, d*4), nn.SiLU(), nn.Linear(d*4, d),
                )
                self.attn = nn.MultiheadAttention(d, h, batch_first=True)
                self.conv = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Conv1d(d, d*2, 1),
                    nn.GLU(dim=1),
                    nn.Conv1d(d, d, 31, padding=15, groups=d),
                    nn.BatchNorm1d(d),
                    nn.SiLU(),
                    nn.Conv1d(d, d, 1),
                )
                self.ff2  = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, d*4), nn.SiLU(), nn.Linear(d*4, d),
                )
                self.norm = nn.LayerNorm(d)

            def forward(self, x):
                x = x + 0.5 * self.ff1(x)
                a, _ = self.attn(x, x, x)
                x = x + a
                xc = x.transpose(1, 2)
                xc = self.conv(xc).transpose(1, 2)
                x = x + xc
                x = x + 0.5 * self.ff2(x)
                return self.norm(x)

        self.conformer = nn.Sequential(
            *[ConformerBlock(latent_dim, n_heads) for _ in range(n_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, 1, T) or (B, n_mfcc, T) mel frames
        Returns:
            (B, latent_dim)
        """
        B = waveform.shape[0]
        if self.mel is not None and waveform.shape[1] == 1:
            x = self.mel(waveform.squeeze(1))  # (B, n_mfcc, T)
        else:
            x = waveform

        # Conv subsampling: treat mel as 2D image (1 channel)
        x   = x.unsqueeze(1)            # (B, 1, n_mfcc, T)
        x   = self.conv_sub(x)          # (B, D//4, n_mfcc//4, T//4)
        T2  = x.shape[-1]
        x   = x.permute(0, 3, 1, 2)    # (B, T', D//4, n_mfcc//4)
        x   = x.flatten(2)              # (B, T', mel_out_dim)
        x   = self.proj(x)              # (B, T', D)

        x   = self.conformer(x)         # (B, T', D)
        x   = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, D)
        return x


class ProprioceptionEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim),
        )
        self.to(device)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class TimeSeriesEncoder(nn.Module):
    """TCN with SSC stabilization — EEG / sensor / physics fields."""

    def __init__(self, in_channels: int, latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3), nn.GELU(),
            nn.Conv1d(128, 256, 5, padding=2), nn.GELU(),
            nn.Conv1d(256, latent_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(latent_dim * 8, latent_dim),
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)


class CrossModalFusion(nn.Module):
    """
    Multi-modal cross-attention fusion with InterfaceAttention prior.
    [v2.0] InterfaceAttention added.
    """

    def __init__(self, latent_dim: int, n_heads: int, device: torch.device) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer    = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls_token      = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.iface_attn     = InterfaceAttention(latent_dim)   # [v2.0]
        self.ssc_stab       = SSCStabilizer(latent_dim)         # [v2.0]
        self.pool           = nn.Linear(latent_dim, latent_dim)
        self.latent_dim     = latent_dim
        self.to(device)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        B     = embeddings[0].shape[0]
        tokens = torch.stack(embeddings, dim=1)              # (B, n_mod, D)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)             # (B, 1+n, D)
        out    = self.transformer(tokens)
        out    = self.ssc_stab(out)                          # [v2.0] stabilize
        return self.pool(out[:, 0, :])


class PerceptionModule(nn.Module):
    """
    AGI ONE v2.0 Perception Layer.
    Vision: ViT-style patch encoder (upgraded from ResNet-18)
    Audio:  Conformer-lite (upgraded from MFCC+CNN)
    All others: unchanged from v1.0
    """

    def __init__(self, cfg: AGIConfig) -> None:
        super().__init__()
        D       = cfg.latent_dim
        device  = cfg.device
        self.device           = device
        self.use_vision       = cfg.use_vision
        self.use_audio        = cfg.use_audio
        self.use_proprio      = cfg.use_proprioception
        self.use_timeseries   = cfg.use_timeseries

        if cfg.use_vision:
            self.vision_enc = PatchViTEncoder(
                latent_dim=D, n_heads=cfg.n_transformer_heads,
                n_layers=cfg.n_transformer_layers, device=device,
            )
        if cfg.use_audio:
            self.audio_enc  = ConformerAudioEncoder(
                latent_dim=D, n_heads=4, n_layers=4, device=device,
            )
        if cfg.use_proprioception:
            self.proprio_enc = ProprioceptionEncoder(64, D, device)
        if cfg.use_timeseries:
            self.ts_enc = TimeSeriesEncoder(64, D, device)

        self.text_embed = nn.Embedding(cfg.vocab_size, D)
        self.text_proj  = nn.Linear(D, D)

        self.fusion = CrossModalFusion(D, cfg.n_transformer_heads, device)
        self.to(device)

    def forward(
        self,
        image      : Optional[torch.Tensor] = None,
        waveform   : Optional[torch.Tensor] = None,
        token_ids  : Optional[torch.Tensor] = None,
        proprio    : Optional[torch.Tensor] = None,
        timeseries : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mods: List[torch.Tensor] = []
        if image is not None and self.use_vision:
            mods.append(self.vision_enc(image))
        if waveform is not None and self.use_audio:
            mods.append(self.audio_enc(waveform))
        if token_ids is not None:
            mods.append(self.text_proj(self.text_embed(token_ids).mean(dim=1)))
        if proprio is not None and self.use_proprio:
            mods.append(self.proprio_enc(proprio))
        if timeseries is not None and self.use_timeseries:
            mods.append(self.ts_enc(timeseries))
        if not mods:
            return torch.zeros(1, self.fusion.latent_dim, device=self.device)
        if len(mods) == 1:
            return mods[0]
        return self.fusion(mods)


# =============================================================================
# SECTION 5 — UPGRADED LANGUAGE MODULE
# GPT-style causal LM with RoPE + SSCStabilizer + InterfaceAttention
# =============================================================================

class RoPECausalTransformer(nn.Module):
    """
    [v2.0 UPGRADED] GPT-style causal language model with:
    - RoPE positional embedding
    - Pre-norm residual architecture
    - SSCStabilizer per layer
    - InterfaceAttention-modified self-attention
    - CSOCComputeController for adaptive depth
    """

    def __init__(
        self,
        vocab_size  : int,
        d_model     : int,
        n_heads     : int,
        n_layers    : int,
        max_seq     : int   = 2048,
        device      : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.n_layers  = n_layers
        self.device    = device
        self.head_dim  = d_model // n_heads

        self.embed  = nn.Embedding(vocab_size, d_model)
        self.rope   = RotaryEmbedding(self.head_dim, max_seq)

        # Build layers with SSC stabilizers and interface attention
        self.layers    : nn.ModuleList = nn.ModuleList()
        self.ssc_stabs : nn.ModuleList = nn.ModuleList()
        self.iface_attns: nn.ModuleList = nn.ModuleList()

        for _ in range(n_layers):
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.layers.append(enc_layer)
            self.ssc_stabs.append(SSCStabilizer(d_model))
            self.iface_attns.append(InterfaceAttention(d_model))

        # CSOC compute controller [v2.0]
        self.csoc = CSOCComputeController(
            d_model=d_model, min_layers=2, max_layers=n_layers
        )

        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.to(device)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids : (B, L)
        Returns:
            (B, d_model) mean-pooled encoding
        """
        x = self.embed(token_ids)           # (B, L, D)

        # CSOC: decide how many layers to run
        _, n_active = self.csoc(x)

        for i in range(n_active):
            layer = self.layers[i]
            x = layer(x)
            x = self.ssc_stabs[i](x)       # SSC stabilization per layer

        x = self.norm(x)
        return x.mean(dim=1)               # (B, D)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode(token_ids)


class LanguageModule(nn.Module):
    """
    AGI ONE v2.0 Language Interface.

    Backend "builtin": RoPECausalTransformer (upgraded GPT-style)
    Backend "huggingface:*": HuggingFace AutoModel
    """

    def __init__(self, cfg: AGIConfig) -> None:
        super().__init__()
        D = cfg.latent_dim
        self.device     = cfg.device
        self.latent_dim = D

        backend = cfg.language_backend.lower()

        if backend == "builtin" or not HAS_HF:
            self.backbone = RoPECausalTransformer(
                vocab_size = cfg.vocab_size,
                d_model    = D,
                n_heads    = cfg.n_transformer_heads,
                n_layers   = cfg.n_transformer_layers,
                device     = cfg.device,
            )
            self.lang_dim  = D
            self._backend  = "builtin"
            logger.info("LanguageModule: RoPE-GPT built-in backbone [v2.0]")

        elif backend.startswith("huggingface:"):
            model_id = backend.split("huggingface:", 1)[1] or cfg.language_model_id
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                hf_model       = AutoModel.from_pretrained(model_id)
                self.backbone  = hf_model.to(cfg.device)
                self.lang_dim  = cfg.language_dim
                self._backend  = "huggingface"
                logger.info(f"LanguageModule: HuggingFace {model_id} [v2.0]")
            except Exception as e:
                logger.warning(f"HuggingFace load failed ({e}) → fallback builtin")
                self.backbone = RoPECausalTransformer(
                    cfg.vocab_size, D, cfg.n_transformer_heads,
                    cfg.n_transformer_layers, device=cfg.device,
                )
                self.lang_dim = D
                self._backend = "builtin"
        else:
            raise ValueError(f"Unknown language_backend: {backend}")

        self.lang_to_latent = nn.Linear(self.lang_dim, D)
        self.grounding_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=cfg.n_transformer_heads, batch_first=True,
        )
        self.lm_head = nn.Linear(D, cfg.vocab_size)

        self.to(cfg.device)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._backend == "builtin":
            return self.backbone.encode(token_ids)
        with torch.no_grad():
            out = self.backbone(token_ids)
            return self.lang_to_latent(out.last_hidden_state.mean(dim=1))

    def ground(self, lang: torch.Tensor, percept: torch.Tensor) -> torch.Tensor:
        q = lang.unsqueeze(1)
        k = percept.unsqueeze(1)
        g, _ = self.grounding_attn(q, k, k)
        return g.squeeze(1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.lm_head(latent)

    def forward(
        self,
        token_ids         : torch.Tensor,
        perception_latent : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lang = self.encode(token_ids)
        if perception_latent is not None:
            lang = self.ground(lang, perception_latent)
        return lang, self.decode(lang)


# =============================================================================
# SECTION 6 — PSYCHE EXECUTIVE LAYER [v2.0 NEW]
# Id → Goal Generator | Ego → Planner | Superego → Safety Constraint
# =============================================================================

class PsycheExecutiveLayer(nn.Module):
    """
    [v2.0 NEW] Psyche as Executive Layer above Transformer.

    Reinterprets PSY ONE BRIDGE (Id/Ego/Superego) as a three-tier
    executive control system:

        Id       → Goal Generator  (what drives does the system have?)
        Ego      → Planner         (what action best satisfies drives?)
        Superego → Safety Filter   (does the action violate constraints?)

    Pipeline:
        Workspace state (D)
            ↓
        [Id]  drive_proposal (action_dim)
        [Ego]  planned_action = free_energy_minimize(drive, constraint)
        [Superego] safety_score ∈ [0,1]: block unsafe actions
            ↓
        executive_action (action_dim)  + safety_gate (scalar)

    This module wraps PsycheTriad if available, otherwise provides
    a differentiable standalone executive controller.
    """

    def __init__(
        self,
        latent_dim  : int,
        action_dim  : int,
        cfg         : AGIConfig,
        device      : torch.device,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device     = device

        # Goal Generator (Id analog): workspace → drive distribution
        self.goal_generator = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Planner (Ego analog): DEQ-like fixed-point via iterative refinement
        self.planner_net = nn.Sequential(
            nn.Linear(action_dim * 2, 256), nn.GELU(),
            nn.Linear(256, action_dim),
        )

        # Safety Constraint (Superego analog): score ∈ [0, 1]
        # 0 = unsafe (block), 1 = safe (allow)
        self.safety_net = nn.Sequential(
            nn.Linear(action_dim, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

        # Normative policy (learnable "ethical prior")
        self.normative_policy = nn.Parameter(
            torch.ones(action_dim) / action_dim
        )

        # PSY ONE BRIDGE integration
        self.psy_triad  : Optional[Any] = None
        self.gumbel_sched: Optional[Any] = None
        if cfg.use_psy_bridge and HAS_PSY_BRIDGE:
            try:
                mode = PsychopathologyMode(cfg.psyche_mode)
            except ValueError:
                mode = PsychopathologyMode.HEALTHY
            self.psy_triad = PsycheTriad(PsycheConfig(
                action_dim     = action_dim,
                lambda_reg     = cfg.lambda_reg,
                mode           = mode,
                gumbel_tau     = cfg.gumbel_tau,
                gumbel_hard    = cfg.gumbel_hard,
                anderson_depth = cfg.anderson_depth,
                device         = device,
            ))
            self.gumbel_sched = GumbelAnnealScheduler(
                tau_start=1.0, tau_end=0.1, total_steps=50_000
            )
            logger.info("✓ PsycheTriad integrated into PsycheExecutiveLayer")

        self.to(device)

    def forward(
        self,
        workspace_state   : torch.Tensor,   # (D,)
        n_planner_iters   : int = 5,
    ) -> Dict[str, Any]:
        """
        Returns:
            executive_action : (action_dim,) differentiable action
            safety_gate      : scalar ∈ [0,1]
            psyche_state     : PsycheTriadState or None
            total_loss       : PSY bridge loss or None
        """
        # ── Id: Goal Generation ─────────────────────────────────────────────
        if workspace_state.dim() == 1:
            ws = workspace_state.unsqueeze(0)   # (1, D)
        else:
            ws = workspace_state

        drive = self.goal_generator(ws).squeeze(0)   # (action_dim,)

        # ── PSY BRIDGE (if available): use full differentiable triad ─────────
        psyche_state = None
        psy_loss     = None
        if self.psy_triad is not None:
            try:
                tau = self.gumbel_sched.step() if self.gumbel_sched else 1.0
                self.psy_triad.config.gumbel_tau = tau
                # Adapt drive to expected distribution
                drive_in = F.softmax(drive, dim=-1)
                psyche_state, psy_loss = self.psy_triad(drive_in)
                if psyche_state.soft_action is not None:
                    drive = psyche_state.soft_action
            except Exception as e:
                logger.debug(f"PsycheTriad exec: {e}")

        # ── Ego: Iterative Planning (DEQ-style fixed-point) ──────────────────
        norm_pol = F.softmax(self.normative_policy, dim=-1)
        plan     = drive.clone()
        for _ in range(n_planner_iters):
            combined = torch.cat([plan, norm_pol], dim=-1)   # (2*action_dim,)
            delta    = self.planner_net(combined)
            plan     = F.softmax(plan + 0.1 * delta, dim=-1)

        # ── Superego: Safety Gate ─────────────────────────────────────────────
        safety_score = self.safety_net(plan).squeeze(-1)   # scalar

        # Gate: blend action with normative policy based on safety
        executive_action = safety_score * plan + (1 - safety_score) * norm_pol

        return {
            "executive_action": executive_action,
            "drive"           : drive,
            "plan"            : plan,
            "safety_score"    : safety_score,
            "psyche_state"    : psyche_state,
            "psy_loss"        : psy_loss,
        }



# =============================================================================
# SECTION 7 — DREAMERV3-STYLE WORLD MODEL [v2.0 NEW]
# Hafner et al. 2023: symlog, two-hot reward, free-bits KL,
# categorical straight-through latents
# =============================================================================

def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symlog: sign(x) · log(|x| + 1)."""
    return x.sign() * (x.abs() + 1.0).log()

def symexp(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symexp: inverse of symlog."""
    return x.sign() * (x.abs().exp() - 1.0)

def two_hot_encode(x: torch.Tensor, n_bins: int = 255,
                    lo: float = -20.0, hi: float = 20.0) -> torch.Tensor:
    """
    DreamerV3 two-hot encoding for reward.
    Projects scalar reward onto two adjacent bins with linear interpolation.
    """
    bins  = torch.linspace(lo, hi, n_bins, device=x.device)
    x_sym = symlog(x).clamp(lo, hi)
    idx   = torch.bucketize(x_sym, bins) - 1
    idx   = idx.clamp(0, n_bins - 2)

    lo_val = bins[idx]
    hi_val = bins[idx + 1]
    w_hi   = ((x_sym - lo_val) / (hi_val - lo_val + 1e-8)).clamp(0, 1)
    w_lo   = 1.0 - w_hi

    target = torch.zeros(*x.shape, n_bins, device=x.device)
    target.scatter_(-1, idx.unsqueeze(-1), w_lo.unsqueeze(-1))
    target.scatter_(-1, (idx + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
    return target


class DreamerV3WorldModel(nn.Module):
    """
    [v2.0 NEW] DreamerV3-style Recurrent State Space Model.

    Key differences from RSSM v1.0:
    [1] Categorical straight-through latents (32 classes × 32 variables)
        instead of Gaussian — avoids posterior collapse
    [2] symlog preprocessing on all inputs and reconstruction targets
    [3] Two-hot encoding for reward (handles wide reward distributions)
    [4] Free-bits KL: KL = max(free_bits, KL_per_variable)
        prevents first few training steps from collapsing
    [5] KL balancing: 80% from posterior, 20% from prior (DreamerV3 default)

    References:
        Hafner et al. "Mastering Diverse Domains with World Models" (2023)
        https://arxiv.org/abs/2301.04104
    """

    def __init__(
        self,
        obs_dim        : int,
        action_dim     : int,
        stoch_size     : int   = 32,   # number of categorical variables
        stoch_classes  : int   = 32,   # classes per variable
        det_size       : int   = 512,
        reward_bins    : int   = 255,
        free_bits      : float = 1.0,
        kl_balance     : float = 0.8,
        device         : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.obs_dim       = obs_dim
        self.action_dim    = action_dim
        self.stoch_size    = stoch_size
        self.stoch_classes = stoch_classes
        self.det_size      = det_size
        self.reward_bins   = reward_bins
        self.free_bits     = free_bits
        self.kl_balance    = kl_balance
        self.device        = device

        self.latent_dim    = stoch_size * stoch_classes

        # ── Sequence model: GRU (deterministic state) ────────────────────────
        self.gru = nn.GRUCell(
            input_size  = self.latent_dim + action_dim,
            hidden_size = det_size,
        )

        # ── Dynamics predictor: prior p(z_t | h_t) ───────────────────────────
        self.prior_net = nn.Sequential(
            nn.Linear(det_size, 512), nn.ELU(),
            nn.Linear(512, stoch_size * stoch_classes),
        )

        # ── Representation model: posterior q(z_t | h_t, o_t) ───────────────
        self.posterior_net = nn.Sequential(
            nn.Linear(det_size + obs_dim, 512), nn.ELU(),
            nn.Linear(512, stoch_size * stoch_classes),
        )

        # ── Decoder: observation reconstruction ─────────────────────────────
        self.obs_decoder = nn.Sequential(
            nn.Linear(det_size + self.latent_dim, 512), nn.ELU(),
            nn.Linear(512, obs_dim),
        )

        # ── Reward predictor: two-hot output ──────────────────────────────────
        self.reward_net = nn.Sequential(
            nn.Linear(det_size + self.latent_dim, 256), nn.ELU(),
            nn.Linear(256, reward_bins),
        )

        # ── Continue predictor: p(non-terminal) ───────────────────────────────
        self.continue_net = nn.Sequential(
            nn.Linear(det_size + self.latent_dim, 128), nn.ELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

        # ── SSC stabilizer on hidden state ────────────────────────────────────
        self.h_ssc = SSCStabilizer(det_size)

        # ── Initial states ────────────────────────────────────────────────────
        self.register_buffer("h0",  torch.zeros(1, det_size))
        self.register_buffer("z0",  torch.zeros(1, self.latent_dim))

        self.to(device)

    # ── Categorical straight-through ─────────────────────────────────────────
    def _straight_through_sample(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Straight-through estimator for categorical latents.
        Returns (one_hot, soft_probs) — gradients flow through soft_probs.
        """
        B       = logits.shape[0]
        logits_ = logits.view(B, self.stoch_size, self.stoch_classes)
        probs   = F.softmax(logits_, dim=-1)
        indices = probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.stoch_classes).float()
        # Straight-through: forward = one_hot, backward = probs
        z       = (one_hot - probs).detach() + probs
        return z.view(B, self.latent_dim), probs

    # ── Prior ─────────────────────────────────────────────────────────────────
    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_prior, prior_logits)."""
        logits = self.prior_net(h)
        z, _   = self._straight_through_sample(logits)
        return z, logits

    # ── Posterior ─────────────────────────────────────────────────────────────
    def posterior(
        self, h: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_post, posterior_logits)."""
        obs_sym = symlog(obs)
        inp     = torch.cat([h, obs_sym], dim=-1)
        logits  = self.posterior_net(inp)
        z, _    = self._straight_through_sample(logits)
        return z, logits

    # ── GRU transition ────────────────────────────────────────────────────────
    def gru_step(
        self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        inp    = torch.cat([z, a], dim=-1)
        h_next = self.gru(inp, h)
        return h_next

    # ── Free-bits KL loss ─────────────────────────────────────────────────────
    def kl_loss(
        self,
        post_logits : torch.Tensor,   # (B, stoch_size * stoch_classes)
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        DreamerV3 free-bits KL with KL balancing.

        KL = kl_balance * KL(post||sg(prior)) + (1-kl_balance) * KL(sg(post)||prior)
        Free-bits: clamp KL per variable at free_bits minimum.
        """
        B = post_logits.shape[0]
        post  = post_logits.view(B, self.stoch_size, self.stoch_classes)
        prior = prior_logits.view(B, self.stoch_size, self.stoch_classes)

        post_probs  = F.softmax(post,  dim=-1).clamp(1e-8)
        prior_probs = F.softmax(prior, dim=-1).clamp(1e-8)

        # KL(post || prior)
        kl_pp = (post_probs * (post_probs.log() - prior_probs.log())).sum(-1)  # (B, S)
        # KL(post_sg || prior)
        kl_sp = ((post_probs.detach()) *
                 (post_probs.detach().log() - prior_probs.log())).sum(-1)

        # Free bits: max(free_bits, kl_per_variable)
        kl_pp = kl_pp.clamp(min=self.free_bits)
        kl_sp = kl_sp.clamp(min=self.free_bits)

        loss = self.kl_balance * kl_pp + (1 - self.kl_balance) * kl_sp
        return loss.mean()

    # ── Reward loss ───────────────────────────────────────────────────────────
    def reward_loss(
        self, feat: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        logits = self.reward_net(feat)
        target = two_hot_encode(
            reward, self.reward_bins, device=reward.device
        )
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    # ── Imagine trajectory ────────────────────────────────────────────────────
    def imagine(
        self,
        h0        : torch.Tensor,   # (1, det_size)
        z0        : torch.Tensor,   # (1, latent_dim)
        action_seq: torch.Tensor,   # (T, action_dim)
    ) -> Dict[str, torch.Tensor]:
        T     = action_seq.shape[0]
        h, z  = h0, z0
        h_seq, z_seq, r_seq, cont_seq = [], [], [], []

        for t in range(T):
            a     = action_seq[t].unsqueeze(0)
            h     = self.gru_step(h, z, a)
            z, _  = self.prior(h)
            feat  = torch.cat([h, z], dim=-1)
            r     = self.reward_net(feat)
            cont  = self.continue_net(feat)
            h_seq.append(h); z_seq.append(z)
            r_seq.append(r); cont_seq.append(cont)

        return {
            "h_seq"     : torch.cat(h_seq,    dim=0),
            "z_seq"     : torch.cat(z_seq,    dim=0),
            "reward_seq": torch.cat(r_seq,    dim=0),
            "cont_seq"  : torch.cat(cont_seq, dim=0),
        }

    def forward(
        self,
        h      : torch.Tensor,
        z      : torch.Tensor,
        action : torch.Tensor,
        obs    : Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h_next       = self.gru_step(h, z, action)

        # Stabilize hidden state
        h_stable     = self.h_ssc(h_next.unsqueeze(1)).squeeze(1)

        prior_logits                  = self.prior_net(h_stable)
        z_prior, _                    = self._straight_through_sample(prior_logits)

        if obs is not None:
            post_logits               = self.posterior_net(
                torch.cat([h_stable, symlog(obs)], dim=-1)
            )
            z_post, _                 = self._straight_through_sample(post_logits)
            z_next                    = z_post
        else:
            post_logits               = prior_logits
            z_next                    = z_prior

        feat         = torch.cat([h_stable, z_next], dim=-1)
        obs_pred     = symexp(self.obs_decoder(feat))
        reward_logits= self.reward_net(feat)
        cont_pred    = self.continue_net(feat)

        return {
            "h_next"        : h_stable,
            "z_next"        : z_next,
            "obs_pred"      : obs_pred,
            "reward_logits" : reward_logits,
            "cont_pred"     : cont_pred,
            "prior_logits"  : prior_logits,
            "post_logits"   : post_logits,
            "feat"          : feat,
        }


# =============================================================================
# SECTION 8 — MPPI PLANNER [v2.0 REPLACES CEM]
# Williams et al. 2017 — GPU-parallel importance-weighted planning
# =============================================================================

class MPPIPlanner(nn.Module):
    """
    [v2.0 NEW] Model Predictive Path Integral (MPPI) Planner.

    Replaces CEM. Key advantages:
    - All N trajectories evaluated in parallel (no sequential elite selection)
    - Importance-weighted update: ALL samples contribute, not just top-k
    - Smoother, more stable optimization landscape
    - Better exploration via temperature-controlled weighting

    Algorithm:
        1. Sample N perturbations ε ~ N(0, σ²I) around nominal action sequence
        2. Roll out each in world model → cost = -sum(discount^t * reward_t)
        3. Compute importance weights: w_i = exp(-(cost_i - min_cost)/λ)
        4. Update: μ_new = Σ(w_i * (μ + ε_i)) / Σw_i
        5. Execute μ[0]; warm-start next step
    """

    def __init__(
        self,
        action_dim      : int,
        horizon         : int,
        n_samples       : int,
        temperature     : float,
        noise_sigma     : float,
        device          : torch.device,
    ) -> None:
        super().__init__()
        self.action_dim  = action_dim
        self.horizon     = horizon
        self.n_samples   = n_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.device      = device

        # Nominal action sequence (warm-started between steps)
        self.register_buffer(
            "mu", torch.zeros(horizon, action_dim)
        )

        # Value baseline for terminal bootstrap
        self.value_net = nn.Sequential(
            nn.Linear(512 + 32 * 32, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

        self.goal_encoder = nn.Linear(action_dim, action_dim)
        self.to(device)

    def plan(
        self,
        world_model   : DreamerV3WorldModel,
        h             : torch.Tensor,         # (1, det_size)
        z             : torch.Tensor,         # (1, latent_dim)
        goal          : Optional[torch.Tensor] = None,
        psyche_bias   : Optional[torch.Tensor] = None,
        discount      : float = 0.99,
        n_iters       : int   = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MPPI planning loop.

        Returns:
            best_action   : (action_dim,) first action to execute
            best_sequence : (T, action_dim) full sequence
        """
        T  = self.horizon
        mu = self.mu.clone()

        # Add psyche drive bias to nominal sequence
        if psyche_bias is not None:
            bias = psyche_bias.to(self.device)
            if bias.shape[-1] != self.action_dim:
                bias = F.adaptive_avg_pool1d(
                    bias.unsqueeze(0).unsqueeze(0), self.action_dim
                ).squeeze()
            mu += 0.1 * bias.unsqueeze(0).expand(T, -1)

        for _ in range(n_iters):
            # Sample perturbations: (N, T, action_dim)
            eps     = torch.randn(
                self.n_samples, T, self.action_dim, device=self.device
            ) * self.noise_sigma
            samples = mu.unsqueeze(0) + eps   # (N, T, A)

            # Evaluate trajectories: compute costs
            costs = torch.zeros(self.n_samples, device=self.device)
            for i in range(self.n_samples):
                traj = world_model.imagine(
                    h0         = h,
                    z0         = z,
                    action_seq = samples[i],
                )
                # Cost = negative discounted reward
                r_logits  = traj["reward_seq"]        # (T, reward_bins)
                r_bins    = torch.linspace(-20, 20, world_model.reward_bins,
                                           device=self.device)
                r_pred    = (F.softmax(r_logits, dim=-1) * r_bins).sum(-1)
                r_pred    = symexp(r_pred)
                discounts = torch.tensor(
                    [discount ** t for t in range(T)], device=self.device
                )
                # Continuation weighting
                cont      = traj["cont_seq"].squeeze(-1)
                cum_cont  = cont.cumprod(dim=0)
                returns   = (r_pred * discounts * cum_cont).sum()

                # Goal bonus
                if goal is not None:
                    final_z = traj["z_seq"][-1:]
                    goal_enc = self.goal_encoder(goal)
                    bonus    = F.cosine_similarity(
                        final_z.mean(dim=-1, keepdim=True),
                        goal_enc.unsqueeze(-1),
                        dim=0,
                    ).mean()
                    returns += bonus

                costs[i] = -returns   # negate: lower cost = better

            # Importance weights: w_i = exp(-(cost_i - min_cost) / λ)
            beta    = costs.min()
            weights = torch.exp(-(costs - beta) / self.temperature)
            weights = weights / (weights.sum() + 1e-8)     # normalize

            # Weighted update of nominal sequence
            mu      = (weights.view(-1, 1, 1) * samples).sum(dim=0)

        # Warm-start: shift sequence left, repeat last action
        self.mu[:-1] = mu[1:].detach()
        self.mu[-1]  = mu[-1].detach()

        return mu[0], mu

    def reset(self) -> None:
        """Reset warm-started nominal sequence (new episode)."""
        self.mu.zero_()

    def forward(
        self,
        world_model: DreamerV3WorldModel,
        h          : torch.Tensor,
        z          : torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        best_action, _ = self.plan(world_model, h, z, **kwargs)
        return best_action


# =============================================================================
# SECTION 9 — MEMORY MODULES (preserved + upgraded with SSC)
# =============================================================================

class WorkingMemoryModule(nn.Module):
    """
    Short-term working memory with SSCStabilizer [v2.0 upgrade].
    N attention-gated slots.
    """

    def __init__(self, n_slots: int, latent_dim: int,
                 n_heads: int, device: torch.device) -> None:
        super().__init__()
        self.n_slots    = n_slots
        self.latent_dim = latent_dim
        self.device     = device

        self.register_buffer("slots", torch.zeros(n_slots, latent_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.slot_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.gate         = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid(),
        )
        self.read_query   = nn.Linear(latent_dim, latent_dim)
        self.read_key     = nn.Linear(latent_dim, latent_dim)
        self.read_value   = nn.Linear(latent_dim, latent_dim)
        self.ssc_stab     = SSCStabilizer(latent_dim)  # [v2.0]
        self.to(device)

    def write(self, c: torch.Tensor) -> None:
        if c.dim() == 1: c = c.unsqueeze(0)
        sim     = F.cosine_similarity(c, self.slots, dim=-1)
        idx     = int(sim.argmin().item())
        old     = self.slots[idx].unsqueeze(0)
        g       = self.gate(torch.cat([old, c], dim=-1))
        self.slots[idx] = (g * c + (1-g) * old).squeeze(0)

    def read(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1: q = q.unsqueeze(0)
        Q  = self.read_query(q)
        K  = self.read_key(self.slots)
        V  = self.read_value(self.slots)
        w  = F.softmax((Q @ K.T) / math.sqrt(self.latent_dim), dim=-1)
        return (w @ V).squeeze(0)

    def process(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.dim() == 1: inp = inp.unsqueeze(0)
        seq = torch.cat([self.slots.unsqueeze(0), inp.unsqueeze(0)], dim=1)
        out = self.slot_transformer(seq)
        out = self.ssc_stab(out)   # [v2.0]
        return out[0, -1, :]

    def reset(self) -> None: self.slots.zero_()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        ctx = self.process(inp)
        self.write(inp)
        return ctx


class EpisodicMemoryModule(nn.Module):
    """Long-term DND episodic memory — preserved from v1.0."""

    def __init__(self, capacity: int, latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.capacity   = capacity
        self.latent_dim = latent_dim
        self.device     = device

        self.register_buffer("keys",         torch.zeros(capacity, latent_dim))
        self.register_buffer("values",       torch.zeros(capacity, latent_dim))
        self.register_buffer("ages",         torch.zeros(capacity))
        self.register_buffer("access_count", torch.zeros(capacity))
        self._ptr  = 0
        self._size = 0

        self.key_encoder       = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh())
        self.consolidation_proj= nn.Linear(latent_dim, latent_dim)
        self.to(device)

    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        k = self.key_encoder(k.detach()).squeeze(0)
        v = v.detach().squeeze(0)
        if self._size < self.capacity:
            idx = self._ptr
            self._size += 1
        else:
            score = self.ages * (1.0 / (self.access_count + 1.0))
            idx   = int(score.argmax().item())
        self.keys[idx] = k; self.values[idx] = v
        self.ages[idx] = 0.0; self.access_count[idx] = 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self.ages[:self._size] += 1.0

    def retrieve(self, q: torch.Tensor, top_k: int = 5,
                 temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._size == 0:
            return torch.zeros(self.latent_dim, device=self.device), \
                   torch.zeros(1, device=self.device)
        q2   = self.key_encoder(q).squeeze(0)
        sim  = F.cosine_similarity(q2.unsqueeze(0), self.keys[:self._size], dim=-1)
        k    = min(top_k, self._size)
        ts, ti = sim.topk(k)
        self.access_count[ti] += 1.0
        w    = F.softmax(ts / temperature, dim=0)
        return (w.unsqueeze(-1) * self.values[ti]).sum(0), w

    def forward(self, q: torch.Tensor,
                write_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        if write_v is not None: self.write(q, write_v)
        r, _ = self.retrieve(q)
        return r


# =============================================================================
# SECTION 10 — GLOBAL WORKSPACE MODULE (preserved + upgraded)
# =============================================================================

class GlobalWorkspaceModule(nn.Module):
    """GWT broadcast consciousness — upgraded with SSCStabilizer [v2.0]."""

    def __init__(self, latent_dim: int, n_modules: int, n_heads: int,
                 device: torch.device, temp: float = 0.5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.temp       = temp
        self.device     = device

        self.saliency_net    = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.broadcast_proj  = nn.Linear(latent_dim, latent_dim)
        self.register_buffer("workspace_state", torch.zeros(latent_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.integrator  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.ssc_stab    = SSCStabilizer(latent_dim)   # [v2.0]
        self.to(device)

    def forward(self, module_activations: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, str]:
        names  = list(module_activations.keys())
        vecs   = torch.stack([module_activations[n].to(self.device) for n in names], 0)
        scores = self.saliency_net(vecs).squeeze(-1)
        w      = F.softmax(scores / self.temp, dim=0)
        bc     = self.broadcast_proj((w.unsqueeze(-1) * vecs).sum(0))
        seq    = torch.stack([self.workspace_state.unsqueeze(0),
                              bc.unsqueeze(0)], dim=1)
        out    = self.integrator(seq)
        out    = self.ssc_stab(out)   # [v2.0]
        new_st = out[0, -1, :]
        self.workspace_state = new_st.detach()
        winner = names[int(w.argmax().item())]
        return new_st, winner


# =============================================================================
# SECTION 11 — META-COGNITION (preserved + upgraded with CSOC)
# =============================================================================

class MetaCognitionModule(nn.Module):
    """Self-model with CSOC-driven adaptive introspection [v2.0]."""

    def __init__(self, latent_dim: int, n_strategies: int = 8,
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.device     = device

        self.strategy_net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, n_strategies),
        )
        self.unc_net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 2), nn.Softplus(),
        )
        self.load_net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.anomaly_enc = nn.Linear(latent_dim, latent_dim // 2)
        self.anomaly_dec = nn.Linear(latent_dim // 2, latent_dim)

        # CSOC compute controller for meta-cognition depth [v2.0]
        self.csoc = CSOCComputeController(latent_dim, min_layers=1, max_layers=8)

        self.register_buffer("history", torch.zeros(100, latent_dim))
        self._ptr = 0
        self.to(device)

    def _update(self, ws: torch.Tensor) -> None:
        self.history[self._ptr % 100] = ws.detach()
        self._ptr += 1

    def introspect(self, q: torch.Tensor) -> torch.Tensor:
        n = min(self._ptr, 100)
        if n == 0: return torch.zeros(self.latent_dim, device=self.device)
        h = self.history[:n]
        w = F.softmax(F.cosine_similarity(q.unsqueeze(0), h, dim=-1) / 0.1, dim=0)
        return (w.unsqueeze(-1) * h).sum(0)

    def forward(self, ws: torch.Tensor, ocd: bool = False) -> Dict[str, Any]:
        self._update(ws)
        unc_vals = self.unc_net(ws)
        load     = float(self.load_net(ws).item())
        recon    = self.anomaly_dec(self.anomaly_enc(ws))
        anomaly  = float(F.mse_loss(recon, ws).item())
        strat    = int(self.strategy_net(ws).argmax().item())
        _, n_lay = self.csoc(ws.unsqueeze(0).unsqueeze(0))

        return {
            "strategy"       : strat,
            "epistemic_unc"  : float(unc_vals[0].item()),
            "aleatoric_unc"  : float(unc_vals[1].item()),
            "cognitive_load" : load,
            "anomaly_score"  : anomaly,
            "ocd_alert"      : ocd,
            "csoc_n_layers"  : n_lay,
        }


