"""
evolution_one_crispr_v7.py  (fixed v7.0.1)
============================================
Differentiable CRISPR-Cas Therapeutic Design Engine — v7
Built on: evolution_one_crispr_v6 (all v6 fixes in v6.0.1 required)

# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026
#
# AI Co-Developers (bug fixes, v7.0.1 audit):
#   - Claude (Anthropic) — DDP-safe guide_optimizers fix,
#                          ClinicalReportGeneratorV7 (11 outcomes),
#                          _extra_optimizers pattern, v7.0.1 audit pass

WHAT'S NEW IN v7  (unchanged from v7 original)
------------------------------------------------
1. Plugin-style disease registry (YAML/JSON)
2. Extended mechanism modes (6 → 13)
3. SAFE FALLBACK — HALT on unknown modes (ModeResolutionError)
4. Outcome vector extended 6 → 11
5. Mechanism-specific heads (Triplet, Structural, Mito, Imprinting, Epigenetic)
6. Honest docstring

CHANGELOG v7.0.1
─────────────────
[FIX-5] DDP-unsafe nn.ModuleDict mutation in _optimize_with_spec
  _optimize_with_spec() previously added newly-created DifferentiableGuideOptimizer
  instances directly into self.guide_optimizers (an nn.ModuleDict) at runtime:
      self.guide_optimizers[spec.key] = optimizer
  nn.ModuleDict.__setitem__ called on a DDP-wrapped module silently bypasses
  gradient hooks, causing optimizers for dynamically-registered diseases to
  receive no gradients in multi-GPU runs.
  Fix: added self._extra_optimizers (plain Python dict, not registered with
  DDP) for guide optimizers that were not present at __init__ time. Lookup
  order: guide_optimizers (registered) → _extra_optimizers (dynamic) → create.

[FIX-6] ClinicalReportGeneratorV7 — 11-outcome support
  v7 imports ClinicalReportGenerator from v6, which only displays 6 outcomes.
  repair["probs"] in v7 has shape (B, 11). The v6 reporter silently truncated
  the 5 new outcomes (Triplet_contraction, Large_deletion, Mitochondrial_replace,
  Epigenetic_silence, Imprint_correct).
  Added ClinicalReportGeneratorV7 that:
  - displays all 11 outcomes from OUTCOME_NAMES
  - applies FIX-2 (correct batch indexing: repair_probs[0][i])
  - used in main() and smoke test (v6 ClinicalReportGenerator still exported
    for backward compatibility)

Requires: evolution_one_crispr_v6_fixed.py (v6.0.1) on PYTHONPATH.
"""

from __future__ import annotations
import math, os, sys, json, argparse, logging, warnings
from pathlib import Path
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse v6 primitives — import from the fixed v6 file.
# In production, rename evolution_one_crispr_v6_fixed.py → evolution_one_crispr_v6.py
# or adjust the import below to match your file layout.
try:
    from evolution_one_crispr_v6_fixed import (
        DifferentiableDNAEncoder,
        DifferentiableProteinEncoder,
        DifferentiableGuideEncoder,
        DifferentiableOffTargetScorer,
        DifferentiableRepairPredictor,
        DifferentiableDeliveryScorer,
        DifferentiableGuideOptimizer,
        DiseaseObjectiveHead,
        ClinicalReportGenerator,      # v6 compat (6 outcomes)
        NUCLEOTIDE_VOCAB,
        AA_VOCAB,
        HUMAN_OPT_CODON,
        CRISPRDesignConfig,
        CRISPRTherapeuticEngine as _V6Engine,
    )
except ImportError:
    # Fallback: try the original filename (pre-fix)
    from evolution_one_crispr_v6 import (
        DifferentiableDNAEncoder,
        DifferentiableProteinEncoder,
        DifferentiableGuideEncoder,
        DifferentiableOffTargetScorer,
        DifferentiableRepairPredictor,
        DifferentiableDeliveryScorer,
        DifferentiableGuideOptimizer,
        DiseaseObjectiveHead,
        ClinicalReportGenerator,
        NUCLEOTIDE_VOCAB,
        AA_VOCAB,
        HUMAN_OPT_CODON,
        CRISPRDesignConfig,
        CRISPRTherapeuticEngine as _V6Engine,
    )

from one_core_evolution import (
    EVOLUTION_VERSION,
    LangevinEvolutionBridge,
    LangevinBridgeMixin,
    CahnHilliardEvoBridge,
    get_device as _core_get_device,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("CRISPRv7")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# Section 1: Mechanism modes (13 total)
# =============================================================================

class MechanismMode(str, Enum):
    """Every therapeutic mechanism the engine understands."""
    # ── Direct DNA repair ────────────────────────────────────────────
    HDR                = "HDR"
    NHEJ_KO            = "NHEJ_KO"
    ALLELE_KO          = "ALLELE_KO"
    EXON_SKIP          = "EXON_SKIP"
    # ── Editor-mediated ─────────────────────────────────────────────
    BASE_EDIT          = "BASE_EDIT"
    PRIME_EDIT         = "PRIME_EDIT"
    EPIGENETIC         = "EPIGENETIC"
    # ── Large-scale / chromosomal ───────────────────────────────────
    XIST_INTEGRATION   = "XIST_INTEGRATION"
    STRUCTURAL_VARIANT = "STRUCTURAL_VARIANT"
    # ── Repeat expansion ────────────────────────────────────────────
    TRIPLET_CONTRACTION = "TRIPLET_CONTRACTION"
    # ── Mitochondrial ───────────────────────────────────────────────
    MITOCHONDRIAL      = "MITOCHONDRIAL"
    # ── Imprinting ──────────────────────────────────────────────────
    IMPRINTING         = "IMPRINTING"
    # ── Refusal / unknown ───────────────────────────────────────────
    NO_RECOMMEND       = "NO_RECOMMEND"


class ModeResolutionError(RuntimeError):
    """Raised when a disease's mode cannot be safely resolved."""


# =============================================================================
# Section 2: DiseaseSpec (extensible schema)
# =============================================================================

@dataclass
class DiseaseSpec:
    """
    Complete specification of a disease target.
    Carries explicit MechanismMode — never inferred silently.
    """
    key: str
    gene: str
    mutation: str
    mode: MechanismMode
    objective: str
    cell_type: int = 0
    tissue: str = "unknown"
    preferred_editor: str = "Cas9"
    delivery: str = "systemic"
    note: str = ""
    ethics_flag: Optional[str] = None

    # ── Mechanism-specific extras (all optional) ────────────────────
    repeat_unit: Optional[str] = None
    repeat_count_pathogenic: Optional[int] = None
    variant_size_kb: Optional[float] = None
    mtdna_heteroplasmy_threshold: Optional[float] = None
    imprinted_allele: Optional[str] = None
    epigenetic_mark: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.mode, str):
            try:
                self.mode = MechanismMode(self.mode)
            except ValueError as e:
                raise ModeResolutionError(
                    f"Unknown mode '{self.mode}' for disease '{self.key}'. "
                    f"Valid modes: {[m.value for m in MechanismMode]}"
                ) from e


# =============================================================================
# Section 3: DiseaseRegistry with YAML/JSON plugin loading
# =============================================================================

class DiseaseRegistry:
    """
    Plugin-style registry. Built-in 15 diseases (v6-compatible) + external YAML/JSON.

    Usage:
        registry = DiseaseRegistry()                    # built-ins
        registry.load_yaml("my_diseases.yaml")          # add more
        registry.add(DiseaseSpec(...))                  # programmatic
        spec = registry.get("huntington")
    """

    def __init__(self):
        self._specs: Dict[str, DiseaseSpec] = {}
        self._load_builtin_v6()

    def _load_builtin_v6(self) -> None:
        B = [
            # ── Loss-of-function recessive ────────────────────────────
            DiseaseSpec("cystic_fibrosis", "CFTR", "F508del",
                        MechanismMode.PRIME_EDIT, "restore_channel_function",
                        cell_type=0, tissue="airway_epithelium",
                        preferred_editor="PE2", delivery="LNP_airway"),
            DiseaseSpec("duchenne_md", "DMD", "exon_51_del",
                        MechanismMode.EXON_SKIP, "restore_reading_frame",
                        cell_type=1, tissue="skeletal_muscle",
                        preferred_editor="Cas9_dual_sgRNA", delivery="AAV9"),
            DiseaseSpec("sma", "SMN1", "exon_7_skip",
                        MechanismMode.BASE_EDIT, "restore_SMN1_exon7",
                        cell_type=2, tissue="motor_neuron",
                        preferred_editor="ABE", delivery="AAV9_intrathecal"),
            DiseaseSpec("beta_thalassemia", "HBB", "IVS2-654",
                        MechanismMode.HDR, "restore_HbF",
                        cell_type=3, tissue="HSPC",
                        preferred_editor="Cas12a",
                        delivery="ex_vivo_electroporation"),
            DiseaseSpec("beta_thalassemia_bcl11a", "BCL11A",
                        "erythroid_enhancer_del",
                        MechanismMode.NHEJ_KO, "reactivate_HbF_via_BCL11A_KO",
                        cell_type=3, tissue="HSPC",
                        preferred_editor="Cas9",
                        delivery="ex_vivo_electroporation"),
            DiseaseSpec("gjb2_deafness", "GJB2", "35delG",
                        MechanismMode.PRIME_EDIT, "restore_connexin26",
                        cell_type=4, tissue="cochlear_supporting_cells",
                        preferred_editor="PE2", delivery="AAV2_round_window"),
            DiseaseSpec("leber_congenital_amaurosis", "CEP290",
                        "IVS26_c.2991+1655A>G",
                        MechanismMode.PRIME_EDIT, "restore_photoreceptor",
                        cell_type=5, tissue="retinal_pigment_epithelium",
                        preferred_editor="PE2", delivery="AAV2_subretinal"),
            DiseaseSpec("pku", "PAH", "R408W",
                        MechanismMode.BASE_EDIT, "restore_PAH_activity",
                        cell_type=6, tissue="hepatocyte",
                        preferred_editor="ABE", delivery="LNP_liver"),
            DiseaseSpec("congenital_diabetes", "KCNJ11", "V59M",
                        MechanismMode.BASE_EDIT, "restore_insulin_secretion",
                        cell_type=7, tissue="pancreatic_beta_cell",
                        preferred_editor="ABE", delivery="LNP_pancreas"),

            # ── Gain-of-function dominant ─────────────────────────────
            DiseaseSpec("neurofibromatosis_1", "NF1", "R1925X",
                        MechanismMode.ALLELE_KO, "reduce_dominant_negative",
                        cell_type=8, tissue="Schwann_cell_precursor",
                        preferred_editor="Cas9_allele_specific",
                        delivery="AAV9",
                        note="Most NF1 mutations are LOF; DN subset only."),
            DiseaseSpec("fibrodysplasia_ossificans", "ACVR1", "R206H",
                        MechanismMode.ALLELE_KO, "block_heterotopic_ossification",
                        cell_type=9, tissue="skeletal_muscle_progenitor",
                        preferred_editor="Cas9_allele_specific",
                        delivery="AAV9"),
            DiseaseSpec("noonan_syndrome", "LZTR1", "deep_intronic",
                        MechanismMode.BASE_EDIT, "restore_LZTR1_splicing",
                        cell_type=10, tissue="cardiac_progenitor",
                        preferred_editor="CBE", delivery="AAV9"),
            DiseaseSpec("cerebral_palsy_gnao1", "GNAO1", "G203R",
                        MechanismMode.ALLELE_KO, "reduce_GNAO1_GoF",
                        cell_type=11, tissue="striatal_neuron",
                        preferred_editor="Cas9_allele_specific",
                        delivery="AAV9_intracranial"),
            DiseaseSpec("congenital_heart_disease", "MYH7", "R403Q",
                        MechanismMode.ALLELE_KO, "reduce_HCM",
                        cell_type=12, tissue="cardiomyocyte",
                        preferred_editor="Cas9_allele_specific",
                        delivery="AAV9_intramyocardial"),

            # ── Chromosomal ───────────────────────────────────────────
            DiseaseSpec("down_syndrome", "chr21_XIST", "trisomy_21",
                        MechanismMode.XIST_INTEGRATION, "silence_extra_chr21",
                        cell_type=13, tissue="neural_progenitor",
                        preferred_editor="Cas9_large_HDR",
                        delivery="AAV6_dual",
                        ethics_flag=("ETHICS NOTE: Trisomy 21 silencing is "
                                     "exploratory. Neurodevelopmental "
                                     "outcomes in humans unknown.")),

            # ── DSD — refusal by design ───────────────────────────────
            DiseaseSpec("disorders_sex_development", "SRY_OR_SOX9",
                        "variable", MechanismMode.NO_RECOMMEND,
                        "HALT_ETHICAL_REVIEW_REQUIRED",
                        cell_type=14, tissue="gonadal",
                        preferred_editor="NONE", delivery="NONE",
                        ethics_flag=("ETHICS REVIEW REQUIRED: DSD "
                                     "interventions must not be undertaken "
                                     "without IRB approval, patient/family "
                                     "consent, and non-pathologizing "
                                     "clinical rationale.")),
        ]
        for spec in B:
            self._specs[spec.key] = spec

    # ── External loading ─────────────────────────────────────────────
    def load_yaml(self, path: str) -> int:
        if not HAS_YAML:
            raise ImportError("pip install pyyaml to load YAML registries")
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._load_dict(data, source=path)

    def load_json(self, path: str) -> int:
        with open(path) as f:
            data = json.load(f)
        return self._load_dict(data, source=path)

    def _load_dict(self, data: Any, source: str) -> int:
        if isinstance(data, dict) and "diseases" in data:
            entries = data["diseases"]
        elif isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = [{"key": k, **v} for k, v in data.items()]
        else:
            raise ValueError(f"Unrecognized registry format in {source}")

        added = 0
        for entry in entries:
            try:
                spec = DiseaseSpec(**entry)
                self._specs[spec.key] = spec
                added += 1
            except Exception as e:
                logger.error("Skipping malformed entry %s in %s: %s",
                             entry.get("key", "?"), source, e)
        logger.info("Loaded %d disease specs from %s", added, source)
        return added

    def add(self, spec: DiseaseSpec) -> None:
        self._specs[spec.key] = spec

    def get(self, key: str) -> Optional[DiseaseSpec]:
        return self._specs.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def __iter__(self):
        return iter(self._specs.values())

    def keys(self) -> List[str]:
        return list(self._specs.keys())


# =============================================================================
# Section 4: Mode → outcome preference matrix (data-driven, 11 outcomes)
# =============================================================================

OUTCOME_NAMES = [
    "HDR", "NHEJ_fs", "NHEJ_if", "Base_edit", "Prime_edit", "No_edit",
    "Triplet_contraction", "Large_deletion", "Mitochondrial_replace",
    "Epigenetic_silence", "Imprint_correct",
]
N_OUTCOMES = len(OUTCOME_NAMES)   # 11

_MODE_OUTCOME_WEIGHTS: Dict[MechanismMode, List[float]] = {
    #                        HDR  fs   if   BE   PE  none tri  del  mito epi  imp
    MechanismMode.HDR:                [2.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    MechanismMode.NHEJ_KO:            [0.0, 2.0, 0.3, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    MechanismMode.ALLELE_KO:          [0.0, 1.8, 0.4, 0.5, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    MechanismMode.EXON_SKIP:          [0.0, 0.5, 2.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    MechanismMode.BASE_EDIT:          [0.0, 0.0, 0.0, 2.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    MechanismMode.PRIME_EDIT:         [0.0, 0.0, 0.0, 0.0, 2.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    MechanismMode.EPIGENETIC:         [0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    MechanismMode.XIST_INTEGRATION:   [2.5, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    MechanismMode.STRUCTURAL_VARIANT: [1.5, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    MechanismMode.TRIPLET_CONTRACTION:[0.3, 0.0, 0.0, 0.0, 0.0,-1.0, 2.0, 0.5, 0.0, 0.0, 0.0],
    MechanismMode.MITOCHONDRIAL:      [0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    MechanismMode.IMPRINTING:         [0.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    MechanismMode.NO_RECOMMEND:       [0.0] * 11,
}


def mode_outcome_weights(mode: MechanismMode) -> List[float]:
    """Return the preference vector; raises ModeResolutionError for unknown modes."""
    if mode not in _MODE_OUTCOME_WEIGHTS:
        raise ModeResolutionError(
            f"No outcome weights registered for mode '{mode}'. "
            "Refusing to default to HDR (unsafe). Halting."
        )
    return list(_MODE_OUTCOME_WEIGHTS[mode])


# =============================================================================
# Section 5: Extended repair predictor (11 outcomes)
# =============================================================================

class ExtendedRepairPredictor(nn.Module):
    """
    v7 upgrade of v6's 6-outcome predictor → 11 outcomes.
    Interface-compatible: precision_score and 6 legacy accessors preserved.
    """
    N_OUTCOMES = N_OUTCOMES  # 11

    def __init__(self, d_guide: int = 96, d_context: int = 64,
                 d_cell: int = 32, d_nuclease: int = 16):
        super().__init__()
        self.ctx_enc = DifferentiableDNAEncoder(d_model=d_context, max_len=256)
        self.cell_embed = nn.Embedding(16, d_cell)
        self.nuc_embed  = nn.Embedding(8, d_nuclease)
        self.mlp = nn.Sequential(
            nn.Linear(d_guide + d_context + d_cell + d_nuclease, 128),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, N_OUTCOMES),
        )

    def forward(self, guide_emb, context_tokens, cell_type, nuclease_type):
        ctx  = self.ctx_enc(context_tokens).mean(dim=1)
        cell = self.cell_embed(cell_type)
        nuc  = self.nuc_embed(nuclease_type)
        x    = torch.cat([guide_emb, ctx, cell, nuc], dim=-1)
        probs = F.softmax(self.mlp(x), dim=-1)
        return {
            "probs": probs,
            # legacy 6-outcome accessors
            "hdr_prob":        probs[..., 0],
            "nhej_fs_prob":    probs[..., 1],
            "base_edit_prob":  probs[..., 3],
            "prime_edit_prob": probs[..., 4],
            "precision_score": probs[..., 0] + probs[..., 3] + probs[..., 4],
            # v7 additions
            "triplet_contraction_prob":  probs[..., 6],
            "large_deletion_prob":       probs[..., 7],
            "mito_replace_prob":         probs[..., 8],
            "epigenetic_silence_prob":   probs[..., 9],
            "imprint_correct_prob":      probs[..., 10],
        }


# =============================================================================
# Section 6: Mechanism-specific heads
# =============================================================================

class TripletRepeatHead(nn.Module):
    """Models contraction efficiency for CAG/CTG/GAA/CGG repeat expansions."""
    N_UNITS = 8

    def __init__(self, d_guide: int = 96, d_model: int = 48):
        super().__init__()
        self.unit_embed = nn.Embedding(self.N_UNITS, d_model)
        self.ctx_enc = DifferentiableDNAEncoder(d_model=d_model, max_len=128)
        self.mlp = nn.Sequential(
            nn.Linear(d_guide + d_model * 2 + 2, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 3),   # [contraction, no-change, expansion]
        )

    def forward(self, guide_emb, unit_id, repeat_count, ctx_tokens):
        u  = self.unit_embed(unit_id)
        c  = self.ctx_enc(ctx_tokens).mean(dim=1)
        r  = repeat_count.unsqueeze(-1).float() / 1000.0
        r2 = (repeat_count.unsqueeze(-1).float() / 100.0) ** 2
        x  = torch.cat([guide_emb, u, c, r, r2], dim=-1)
        probs = F.softmax(self.mlp(x), dim=-1)
        return {
            "contraction_prob":     probs[..., 0],
            "no_change_prob":       probs[..., 1],
            "expansion_prob":       probs[..., 2],
            "net_contraction_score": probs[..., 0] - probs[..., 2],
        }


class StructuralVariantHead(nn.Module):
    """Models repair of >100 bp deletions/duplications/inversions."""
    def __init__(self, d_guide: int = 96, d_model: int = 48):
        super().__init__()
        self.ctx_enc = DifferentiableDNAEncoder(d_model=d_model, max_len=256)
        self.mlp = nn.Sequential(
            nn.Linear(d_guide + d_model + 1, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, guide_emb, ctx_tokens, variant_size_kb):
        c    = self.ctx_enc(ctx_tokens).mean(dim=1)
        size = variant_size_kb.unsqueeze(-1).float()
        x    = torch.cat([guide_emb, c, size], dim=-1)
        probs = F.softmax(self.mlp(x), dim=-1)
        return {
            "precise_repair_prob": probs[..., 0],
            "partial_repair_prob": probs[..., 1],
            "failed_prob":         probs[..., 2],
        }


class MitochondrialHead(nn.Module):
    """Models heteroplasmy shift for mtDNA variants."""
    def __init__(self, d_guide: int = 96, d_model: int = 48):
        super().__init__()
        self.ctx_enc = DifferentiableDNAEncoder(d_model=d_model, max_len=128)
        self.mlp = nn.Sequential(
            nn.Linear(d_guide + d_model + 1, 96), nn.GELU(),
            nn.Linear(96, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, guide_emb, ctx_tokens, current_heteroplasmy):
        c = self.ctx_enc(ctx_tokens).mean(dim=1)
        h = current_heteroplasmy.unsqueeze(-1).float()
        x = torch.cat([guide_emb, c, h], dim=-1)
        return {"delta_heteroplasmy": self.mlp(x).squeeze(-1)}


class ImprintingHead(nn.Module):
    """Models allele-specific methylation correction for imprinting disorders."""
    def __init__(self, d_guide: int = 96, d_model: int = 48):
        super().__init__()
        self.ctx_enc = DifferentiableDNAEncoder(d_model=d_model, max_len=256)
        self.allele_embed = nn.Embedding(3, 16)
        self.mlp = nn.Sequential(
            nn.Linear(d_guide + d_model + 16, 96), nn.GELU(),
            nn.Linear(96, 1), nn.Sigmoid(),
        )

    def forward(self, guide_emb, ctx_tokens, imprinted_allele_id):
        c = self.ctx_enc(ctx_tokens).mean(dim=1)
        a = self.allele_embed(imprinted_allele_id)
        x = torch.cat([guide_emb, c, a], dim=-1)
        return {"imprint_correction_score": self.mlp(x).squeeze(-1)}


class EpigeneticHead(nn.Module):
    """Models CRISPRi/CRISPRa via dCas9-KRAB/dCas9-VP64."""
    N_MODALITIES = 3

    def __init__(self, d_guide: int = 96, d_model: int = 48):
        super().__init__()
        self.ctx_enc = DifferentiableDNAEncoder(d_model=d_model, max_len=256)
        self.mod_embed = nn.Embedding(self.N_MODALITIES, 16)
        self.mlp = nn.Sequential(
            nn.Linear(d_guide + d_model + 16, 96), nn.GELU(),
            nn.Linear(96, 1), nn.Sigmoid(),
        )

    def forward(self, guide_emb, ctx_tokens, modality_id):
        c = self.ctx_enc(ctx_tokens).mean(dim=1)
        m = self.mod_embed(modality_id)
        x = torch.cat([guide_emb, c, m], dim=-1)
        return {"epigenetic_efficiency": self.mlp(x).squeeze(-1)}


MECHANISM_HEAD_CLASSES: Dict[MechanismMode, type] = {
    MechanismMode.TRIPLET_CONTRACTION: TripletRepeatHead,
    MechanismMode.STRUCTURAL_VARIANT:  StructuralVariantHead,
    MechanismMode.MITOCHONDRIAL:       MitochondrialHead,
    MechanismMode.IMPRINTING:          ImprintingHead,
    MechanismMode.EPIGENETIC:          EpigeneticHead,
}


# =============================================================================
# Section 7: v7 disease objective head (mode-driven, safe)
# =============================================================================

class DiseaseObjectiveHeadV7(nn.Module):
    """
    Like v6's DiseaseObjectiveHead but:
      - outcome_weights loaded from 11-outcome matrix (register_buffer, non-learnable)
      - raises ModeResolutionError for unknown modes (SAFE FALLBACK)
      - composes with mechanism-specific head when applicable

    Note: outcome_weights is a buffer (fixed, data-driven) not a Parameter
    (learnable). This is intentional: the preference matrix encodes clinical
    domain knowledge and should not drift during optimization.
    """
    def __init__(self, spec: DiseaseSpec, d_guide: int = 96):
        super().__init__()
        self.spec = spec
        self.mode = spec.mode

        if spec.mode == MechanismMode.NO_RECOMMEND:
            self.outcome_weights = None
            self.head            = None
            self.mech_head       = None
            return

        weights = mode_outcome_weights(spec.mode)   # raises on unknown mode
        self.register_buffer(
            "outcome_weights",
            torch.tensor(weights, dtype=torch.float32),
        )
        self.head = nn.Sequential(
            nn.Linear(d_guide + N_OUTCOMES + 1 + 4, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        mech_cls = MECHANISM_HEAD_CLASSES.get(spec.mode)
        self.mech_head = mech_cls(d_guide=d_guide) if mech_cls else None

    def forward(self, guide_emb, repair_probs, off_target_risk,
                delivery_feasibility, **mech_kwargs) -> Dict[str, torch.Tensor]:
        if self.mode == MechanismMode.NO_RECOMMEND:
            raise ModeResolutionError(
                f"Disease '{self.spec.key}' is marked NO_RECOMMEND. "
                "The engine must halt before calling this head."
            )
        outcome_score = (repair_probs * self.outcome_weights).sum(-1, keepdim=True)
        safety_penalty = -2.0 * off_target_risk.unsqueeze(-1)
        delivery_score = delivery_feasibility.sum(-1, keepdim=True)

        x = torch.cat([guide_emb, repair_probs,
                       off_target_risk.unsqueeze(-1),
                       delivery_feasibility], dim=-1)
        learned = self.head(x)

        therapeutic = (learned * 0.6 +
                       torch.sigmoid(outcome_score + safety_penalty
                                     + 0.3 * delivery_score) * 0.4)
        out = {
            "therapeutic_score": therapeutic,
            "outcome_score":     outcome_score,
            "safety_penalty":    safety_penalty,
        }
        if self.mech_head is not None and mech_kwargs:
            mech_out = self.mech_head(guide_emb, **mech_kwargs)
            out["mech"] = mech_out
            out["therapeutic_score"] = (0.7 * therapeutic
                                        + 0.3 * self._mech_scalar(mech_out))
        return out

    @staticmethod
    def _mech_scalar(mech_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key in ("net_contraction_score", "imprint_correction_score",
                    "epigenetic_efficiency", "precise_repair_prob"):
            if key in mech_out:
                s = mech_out[key]
                return torch.sigmoid(s) if (s.min() < 0 or s.max() > 1) else s
        if "delta_heteroplasmy" in mech_out:
            return torch.sigmoid(mech_out["delta_heteroplasmy"])
        vals = [v for v in mech_out.values() if v.dim() >= 1]
        return torch.stack(vals, dim=0).mean(dim=0)


# =============================================================================
# Section 8: v7 Engine
# =============================================================================

class CRISPRTherapeuticEngineV7(_V6Engine):
    """
    v7 engine:
      - DiseaseRegistry (built-in + external YAML/JSON)
      - ExtendedRepairPredictor (11 outcomes)
      - DiseaseObjectiveHeadV7 (mode-driven, SAFE FALLBACK)
      - HALTS on unknown / NO_RECOMMEND modes
      - FIX-5: DDP-safe guide optimizer management
    """

    def __init__(self, cfg: Optional[CRISPRDesignConfig] = None,
                 registry: Optional[DiseaseRegistry] = None):
        super().__init__(cfg)                        # v6 init
        self.registry = registry or DiseaseRegistry()

        # Replace v6 repair predictor with 11-outcome version
        self.repair_predictor = ExtendedRepairPredictor(
            d_guide=self.cfg.d_guide).to(self.device)

        # Rebuild disease heads using v7 spec-driven head
        self.disease_heads = nn.ModuleDict({
            key: DiseaseObjectiveHeadV7(
                self.registry.get(key), d_guide=self.cfg.d_guide
            ).to(self.device)
            for key in self.cfg.disease_keys
            if key in self.registry
        })

        # FIX-5: _extra_optimizers holds optimizers for dynamically-registered
        # diseases added after __init__ (e.g. from YAML at runtime).
        # Plain dict — not an nn.ModuleDict — so DDP wrapping is safe.
        # Optimizers here are kept on self.device but excluded from DDP sync.
        # For distributed training, call design_all() before DDP wrapping,
        # or pre-populate cfg.disease_keys with all expected keys.
        self._extra_optimizers: Dict[str, DifferentiableGuideOptimizer] = {}

        missing = [k for k in self.cfg.disease_keys if k not in self.registry]
        if missing:
            logger.warning(
                "Disease keys not in registry (will HALT, not default): %s",
                missing)

        logger.info("v7 engine ready. Registry size = %d",
                    len(self.registry.keys()))

    # ── SAFE mode resolution ─────────────────────────────────────────
    def _resolve_spec(self, disease_key: str) -> DiseaseSpec:
        spec = self.registry.get(disease_key)
        if spec is None:
            raise ModeResolutionError(
                f"Disease '{disease_key}' not in registry. "
                "Register it via DiseaseRegistry.load_yaml/.add, or "
                "remove it from cfg.disease_keys. "
                "v7 REFUSES to silently default to HDR.")
        return spec

    # ── Guide optimizer lookup (FIX-5) ───────────────────────────────
    def _get_or_create_optimizer(
            self, key: str) -> DifferentiableGuideOptimizer:
        """
        Lookup order:
          1. self.guide_optimizers  — DDP-registered (keys in cfg.disease_keys)
          2. self._extra_optimizers — dynamic (keys added after __init__)
          3. Create new, store in _extra_optimizers (NOT in nn.ModuleDict)
        """
        if key in self.guide_optimizers:
            return self.guide_optimizers[key]
        if key in self._extra_optimizers:
            return self._extra_optimizers[key]
        opt = DifferentiableGuideOptimizer(
            guide_length=20, temp=self.cfg.temp_init).to(self.device)
        self._extra_optimizers[key] = opt
        logger.debug("Created extra optimizer for '%s' (not DDP-synced)", key)
        return opt

    # ── Optimize one disease ─────────────────────────────────────────
    def optimize_guide_for_disease(self, disease_key: str,
                                   target_context: str,
                                   **kwargs) -> Dict[str, Any]:
        spec = self._resolve_spec(disease_key)

        if spec.mode == MechanismMode.NO_RECOMMEND:
            return {
                "disease":         disease_key,
                "status":          "HALTED",
                "reason":          spec.ethics_flag or spec.note or "Mode NO_RECOMMEND.",
                "optimized_guide": None,
            }

        if disease_key not in self.disease_heads:
            raise ModeResolutionError(
                f"Disease '{disease_key}' resolved to mode "
                f"'{spec.mode.value}' but no head was instantiated. "
                "Add it to cfg.disease_keys.")

        return self._optimize_with_spec(spec, target_context, **kwargs)

    def _optimize_with_spec(self, spec: DiseaseSpec, target_context: str,
                            n_steps: Optional[int] = None,
                            n_off_target_candidates: int = 32,
                            route: str = "systemic") -> Dict[str, Any]:
        """Run gradient optimization using the v7 spec."""
        n_steps = n_steps or self.cfg.n_opt_steps

        ctx = self.tokenize_dna(target_context[:256]).unsqueeze(0).to(self.device)
        if ctx.shape[1] < 256:
            pad = torch.full((1, 256 - ctx.shape[1]), 4,
                             dtype=torch.long, device=self.device)
            ctx = torch.cat([ctx, pad], dim=1)

        off_cands = torch.randint(0, 5, (1, n_off_target_candidates, 20),
                                  device=self.device)
        pam_t    = torch.tensor([self._pam_token("NGG")], device=self.device)
        cell_t   = torch.tensor([spec.cell_type], device=self.device)
        nuc_t    = torch.tensor([0], device=self.device)
        tissue_t = torch.tensor([spec.cell_type], device=self.device)
        size_t   = torch.tensor(
            [self.delivery_scorer.EDITOR_SIZES_KB.get(
                spec.preferred_editor, 4.1)], device=self.device)
        route_map = {"systemic": 0, "local": 1, "ex_vivo": 2, "intrathecal": 3}
        route_oh  = F.one_hot(
            torch.tensor([route_map.get(route, 0)], device=self.device),
            num_classes=4).float()

        # FIX-5: use safe lookup instead of direct nn.ModuleDict mutation
        optimizer = self._get_or_create_optimizer(spec.key)
        opt = torch.optim.Adam([optimizer.logits], lr=self.cfg.opt_lr)
        history = []
        head = self.disease_heads[spec.key]

        # Mechanism kwargs — built once outside the loop
        mech_kwargs: Dict[str, Any] = {}
        if spec.mode == MechanismMode.TRIPLET_CONTRACTION:
            mech_kwargs = {
                "unit_id":      torch.tensor([0], device=self.device),
                "repeat_count": torch.tensor([40], device=self.device),
                "ctx_tokens":   ctx,
            }
        elif spec.mode == MechanismMode.STRUCTURAL_VARIANT:
            mech_kwargs = {
                "ctx_tokens":      ctx,
                "variant_size_kb": torch.tensor(
                    [spec.variant_size_kb or 100.0], device=self.device),
            }
        elif spec.mode == MechanismMode.MITOCHONDRIAL:
            mech_kwargs = {
                "ctx_tokens":            ctx,
                "current_heteroplasmy":  torch.tensor([0.7], device=self.device),
            }
        elif spec.mode == MechanismMode.IMPRINTING:
            mech_kwargs = {
                "ctx_tokens":           ctx,
                "imprinted_allele_id":  torch.tensor([0], device=self.device),
            }
        elif spec.mode == MechanismMode.EPIGENETIC:
            mech_kwargs = {
                "ctx_tokens":  ctx,
                "modality_id": torch.tensor([0], device=self.device),
            }

        for step in range(n_steps):
            t = (self.cfg.temp_init
                 + (self.cfg.temp_final - self.cfg.temp_init) * (step / n_steps))
            optimizer.temp = t
            opt.zero_grad()

            gumbel    = optimizer(hard=False)
            soft      = gumbel["soft_tokens"]
            guide_tok = soft.argmax(dim=-1)

            guide_emb      = self.guide_encoder(guide_tok, pam_t)
            guide_emb_soft = (soft.unsqueeze(-1)
                              * self.guide_encoder.dna_enc.embed.weight.unsqueeze(0)
                              ).sum(-2).mean(dim=1)
            guide_emb = guide_emb + (guide_emb_soft - guide_emb_soft.detach())

            off_target = self.off_target_scorer(guide_emb, off_cands)
            repair     = self.repair_predictor(guide_emb, ctx, cell_t, nuc_t)
            delivery   = self.delivery_scorer(tissue_t, size_t, route_oh)
            objective  = head(guide_emb, repair["probs"],
                              off_target["worst_risk"], delivery, **mech_kwargs)

            loss = (
                -objective["therapeutic_score"].mean()
                + self.cfg.lambda_off_target * off_target["worst_risk"].mean()
                + self.cfg.lambda_precision  * (1.0 - repair["precision_score"]).mean()
                + self.cfg.lambda_delivery   * (1.0 - delivery.sum(-1)).mean()
                + self.cfg.lambda_safety     * off_target["worst_risk"].pow(2).mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([optimizer.logits], 1.0)
            opt.step()
            history.append(loss.item())

        final_guide = optimizer.to_sequence()
        return {
            "disease":                  spec.key,
            "mode":                     spec.mode.value,
            "status":                   "DESIGN_READY",
            "optimized_guide":          final_guide,
            "loss_history":             history,
            "final_therapeutic_score":  float(
                objective["therapeutic_score"].mean().detach()),
            "final_off_target_risk":    float(
                off_target["worst_risk"].mean().detach()),
            "final_precision_score":    float(
                repair["precision_score"].mean().detach()),
            "repair_probs":             repair["probs"].detach().cpu().numpy().tolist(),
            "outcome_names":            OUTCOME_NAMES,
            "preferred_editor":         spec.preferred_editor,
            "delivery_route":           spec.delivery,
        }

    # ── design_all: safe iteration ───────────────────────────────────
    def design_all(self, target_context: str,
                   disease_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        keys   = disease_keys or self.cfg.disease_keys
        report: Dict[str, Any] = {}
        for key in keys:
            spec = self.registry.get(key)
            if spec is None:
                report[key] = {
                    "status": "HALTED",
                    "reason": (f"Disease '{key}' not in registry. "
                               "v7 refuses to default unknown modes to HDR."),
                }
                logger.warning("HALT: unregistered disease %s", key)
                continue

            if spec.mode == MechanismMode.NO_RECOMMEND:
                report[key] = {
                    "status": "HALTED",
                    "reason": spec.ethics_flag or spec.note or "Mode NO_RECOMMEND.",
                }
                logger.warning("HALT (ethics): %s", key)
                continue

            try:
                report[key] = self.optimize_guide_for_disease(key, target_context)
                logger.info("%s OK: guide=%s score=%.3f off=%.4f",
                            key, report[key]["optimized_guide"],
                            report[key]["final_therapeutic_score"],
                            report[key]["final_off_target_risk"])
            except ModeResolutionError as e:
                report[key] = {"status": "HALTED", "reason": str(e)}
                logger.error("HALT %s: %s", key, e)
            except Exception as e:
                report[key] = {"status": "ERROR", "error": str(e)}
                logger.exception("ERROR %s", key)

        self.results = report
        return report


# =============================================================================
# Section 9: ClinicalReportGeneratorV7 (FIX-6: 11-outcome support)
# =============================================================================

class ClinicalReportGeneratorV7:
    """
    v7 clinical report generator with full 11-outcome display.

    FIX-6: v6's ClinicalReportGenerator only displayed 6 outcomes.
    v7's repair_probs tensor has shape (B, 11). This class:
      - displays all 11 outcomes using OUTCOME_NAMES
      - applies FIX-2 batch indexing: repair_probs[0][i]
      - adds mode field and v7-only mechanism output where present
      - backward-compatible with v6 HALTED/ERROR result dicts
    """

    DISEASE_DISPLAY_NAMES = {
        "neurofibromatosis_1":        "Neurofibromatosis Type 1 (NF1)",
        "down_syndrome":              "Down Syndrome (Trisomy 21)",
        "congenital_diabetes":        "Congenital Diabetes Mellitus",
        "congenital_heart_disease":   "Congenital Heart Disease",
        "cerebral_palsy_gnao1":       "Cerebral Palsy (GNAO1)",
        "disorders_sex_development":  "Disorders of Sex Development",
        "cystic_fibrosis":            "Cystic Fibrosis",
        "duchenne_md":                "Duchenne Muscular Dystrophy",
        "sma":                        "Spinal Muscular Atrophy",
        "beta_thalassemia":           "Beta-Thalassemia (HBB HDR)",
        "beta_thalassemia_bcl11a":    "Beta-Thalassemia (BCL11A strategy)",
        "gjb2_deafness":              "GJB2-Related Congenital Deafness",
        "leber_congenital_amaurosis": "Leber Congenital Amaurosis",
        "fibrodysplasia_ossificans":  "Fibrodysplasia Ossificans Progressiva",
        "noonan_syndrome":            "Noonan Syndrome (LZTR1)",
        "pku":                        "Phenylketonuria",
        "huntington":                 "Huntington Disease",
        "prader_willi":               "Prader-Willi Syndrome",
        "mELAS":                      "MELAS (MT-TL1 m.3243A>G)",
    }

    def __init__(self):
        self.reports: List[Dict] = []

    def generate(self, engine_results: Dict[str, Any],
                 patient_id: str = "UNKNOWN") -> Dict[str, Any]:
        report = {
            "patient_id":       patient_id,
            "engine_version":   "v7.0.1",
            "evolution_version": EVOLUTION_VERSION,
            "diseases":         {},
        }
        for key, result in engine_results.items():
            display = self.DISEASE_DISPLAY_NAMES.get(key, key)

            if result.get("status") == "HALTED":
                report["diseases"][display] = {
                    "status":                  "HALTED — ethical / registry check",
                    "reason":                  result.get("reason"),
                    "guide":                   None,
                    "clinical_recommendation": None,
                }
                continue
            if result.get("status") == "ERROR":
                report["diseases"][display] = {
                    "status": "ERROR",
                    "error":  result.get("error"),
                }
                continue

            # FIX-6 + FIX-2: repair_probs is [[v0..v10]] (batch=1 × 11)
            rp   = result.get("repair_probs")
            names = result.get("outcome_names", OUTCOME_NAMES)

            def _rp(i: int) -> Optional[float]:
                if rp and len(rp) > 0 and len(rp[0]) > i:
                    return round(float(rp[0][i]), 6)
                return None

            repair_dist = {name: _rp(i) for i, name in enumerate(names)}

            report["diseases"][display] = {
                "status":                    "DESIGN_READY",
                "mode":                      result.get("mode"),
                "optimized_guide":           result.get("optimized_guide"),
                "therapeutic_score":         result.get("final_therapeutic_score"),
                "off_target_risk":           result.get("final_off_target_risk"),
                "precision_score":           result.get("final_precision_score"),
                "repair_outcome_distribution": repair_dist,
                "editor":                    result.get("preferred_editor"),
                "delivery_route":            result.get("delivery_route"),
                "clinical_recommendation":   self._recommend(result),
                "ethics_flag":               self._ethics_check(key),
            }
        self.reports.append(report)
        return report

    def _recommend(self, result: Dict) -> str:
        score = result.get("final_therapeutic_score", 0)
        off   = result.get("final_off_target_risk", 1)
        if off > 0.1:
            return "REVIEW — off-target risk exceeds 10% threshold"
        if score > 0.7:
            return "PROCEED to preclinical validation"
        if score > 0.4:
            return "CONDITIONAL — further guide optimization recommended"
        return "HOLD — therapeutic score below actionable threshold"

    def _ethics_check(self, disease_key: str) -> Optional[str]:
        if disease_key == "disorders_sex_development":
            return ("ETHICS REVIEW REQUIRED: DSD interventions must not "
                    "be undertaken without IRB approval, patient/family "
                    "consent, and non-pathologizing clinical rationale.")
        if disease_key == "down_syndrome":
            return ("ETHICS NOTE: Trisomy 21 silencing is exploratory. "
                    "Neurodevelopmental outcomes in humans unknown.")
        return None

    def export_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.reports, f, indent=2, default=str)
        logger.info("Clinical report (v7) saved to %s", path)


# =============================================================================
# Section 10: CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EVOLUTION ONE v7.0.1 — Differentiable CRISPR Engine")
    parser.add_argument("--context", "-c", required=True)
    parser.add_argument("--diseases", nargs="+", required=True)
    parser.add_argument("--registry", default=None,
                        help="Optional YAML/JSON disease registry")
    parser.add_argument("--output", "-o", default="./crispr_output")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--patient_id", default="PATIENT_001")
    args = parser.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    registry = DiseaseRegistry()
    if args.registry:
        p = Path(args.registry)
        if p.suffix in (".yaml", ".yml"):
            registry.load_yaml(str(p))
        elif p.suffix == ".json":
            registry.load_json(str(p))
        else:
            raise ValueError("Registry must be .yaml/.yml/.json")

    cfg = CRISPRDesignConfig(
        disease_keys=args.diseases,
        n_opt_steps=args.steps,
        use_amp=not args.no_amp,
        device=args.device,
    )
    engine  = CRISPRTherapeuticEngineV7(cfg, registry=registry)
    results = engine.design_all(args.context, disease_keys=args.diseases)

    # FIX-6: use v7 reporter
    reporter = ClinicalReportGeneratorV7()
    report   = reporter.generate(results, patient_id=args.patient_id)
    reporter.export_json(str(out / "crispr_report_v7.json"))
    with open(out / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nOutputs saved to {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # ── Smoke test ───────────────────────────────────────────────
        print("EVOLUTION ONE v7.0.1 — smoke test")
        registry = DiseaseRegistry()

        # Programmatic registration
        registry.add(DiseaseSpec(
            key="huntington", gene="HTT", mutation="CAG_expansion_42",
            mode=MechanismMode.TRIPLET_CONTRACTION,
            objective="reduce_HTT_aggregation",
            cell_type=11, tissue="striatal_neuron",
            preferred_editor="Cas9_nickase_pair", delivery="AAV9_intracranial",
            repeat_unit="CAG", repeat_count_pathogenic=42,
        ))
        registry.add(DiseaseSpec(
            key="prader_willi", gene="15q11-q13", mutation="pat_del",
            mode=MechanismMode.IMPRINTING,
            objective="restore_paternal_gene_expression",
            cell_type=14, tissue="hypothalamic_neuron",
            preferred_editor="dCas9-TET1", delivery="AAV9",
            imprinted_allele="paternal",
        ))
        registry.add(DiseaseSpec(
            key="mELAS", gene="MT-TL1", mutation="m.3243A>G",
            mode=MechanismMode.MITOCHONDRIAL,
            objective="shift_heteroplasmy",
            cell_type=0, tissue="mitochondria_rich_tissue",
            preferred_editor="mito-TALEN", delivery="AAV9",
            mtdna_heteroplasmy_threshold=0.6,
        ))

        print(f"\nRegistry size: {len(registry.keys())}")
        print("Keys:", registry.keys())

        cfg = CRISPRDesignConfig(
            disease_keys=["cystic_fibrosis", "huntington",
                          "prader_willi", "disorders_sex_development",
                          "totally_unregistered_disease"],
            n_opt_steps=30, use_amp=False, device="cpu",
        )
        engine  = CRISPRTherapeuticEngineV7(cfg, registry=registry)
        context = "ATGGTGGTGGAAGGTGTTTTTCTGGATTTATTTGATTTTTCAGGTCTCT"
        results = engine.design_all(context, disease_keys=cfg.disease_keys)

        print("\n=== SMOKE TEST RESULTS ===")
        for k, r in results.items():
            print(f"\n{k}: {r.get('status')}")
            if r.get("status") == "HALTED":
                print(f"  reason: {r.get('reason')}")
            else:
                print(f"  guide : {r.get('optimized_guide')}")
                print(f"  score : {r.get('final_therapeutic_score'):.4f}")
                print(f"  mode  : {r.get('mode')}")

        # FIX-6: use v7 reporter
        reporter = ClinicalReportGeneratorV7()
        report   = reporter.generate(results, patient_id="SMOKE_TEST_V7")
        print("\n=== CLINICAL REPORT (v7) ===")
        for disease, data in report["diseases"].items():
            print(f"\n{disease}:")
            print(f"  Status : {data['status']}")
            print(f"  Rec.   : {data.get('clinical_recommendation')}")
            dist = data.get("repair_outcome_distribution", {})
            if dist:
                top = sorted(dist.items(), key=lambda x: x[1] or 0, reverse=True)[:3]
                print(f"  Top outcomes: {top}")
            if data.get("ethics_flag"):
                print(f"  Ethics : {data['ethics_flag']}")

        print("\nAll smoke-test sections executed without error.")
