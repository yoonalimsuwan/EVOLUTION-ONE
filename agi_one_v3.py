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

