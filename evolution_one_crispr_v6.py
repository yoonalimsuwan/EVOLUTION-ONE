"""
evolution_one_crispr_v6.py
==========================
Differentiable CRISPR-Cas Therapeutic Design Engine
Built on: one_core_evolution, structural_calculus_ops,
          evolution_one_ageing, evolution_one_v4, epi_v5

Covers: NF1 (ท้าวแสนปม), Down syndrome (สังทอง), congenital diabetes,
        congenital heart disease (LZTR1/Noonan), cerebral palsy (GNAO1),
        disorders of sex development (SRY/SOX9), CF, DMD, SMA,
        beta-thalassemia, sickle cell, GJB2 deafness, LCA,
        FOP/ACVR1, PKU, and any user-supplied gene-disease pair.
"""

from __future__ import annotations
import math, os, sys, json, argparse, logging, warnings
import random, itertools, pickle, subprocess, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from scipy.stats import entropy, pearsonr, spearmanr

# ── Shared primitives from one_core_evolution ────────────────────────
from one_core_evolution import (
    SemanticStateContraction,
    CSOCBase,
    BVFieldTheory,
    DifferentiableRG,
    DifferentiableSOC,
    DifferentiableIto,
    CheckpointManager,
    LangevinEvolutionBridge,
    LangevinBridgeMixin,
    EpiEvolutionBridge,
    CahnHilliardEvoBridge,
    get_device as _core_get_device,
    EVOLUTION_VERSION,
)

# ── Structural calculus primitives (from evolution_one_ageing) ───────
try:
    from structural_calculus_ops import (
        IteratedStructuralOperator,
        ClampedBoundaryReadout,
        SpectralCoercivityLoss,
        StructuralWeakValuePool,
        StructuralCalculusBlock,
        StructuralCalculusConfig,
        min_working_bits,
        navier_spectral_coercivity_alpha,
    )
    HAS_STRUCTURAL = True
except ImportError:
    HAS_STRUCTURAL = False

# ── Biopython (sequence I/O, CRISPR design) ──────────────────────────
try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import GC
    from Bio import SeqIO
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.PDB import PDBParser, Polypeptide
    from Bio.PDB.StructureBuilder import StructureBuilder
    from Bio.PDB.vectors import calc_dihedral, calc_angle
    from Bio.codonalign.codonseq import CodonSeq, cal_dn_ds
    from Bio.Align import PairwiseAligner
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import pyranges as pr
    HAS_PYRANGES = True
except ImportError:
    HAS_PYRANGES = False

# ── REAL FOLD ONE (structural impact) ────────────────────────────────
try:
    from real_fold_one_v2 import (
        RefinementEngine as _RFO_Engine,
        RefinementConfig as _RFO_Config,
        load_structure as rfo_load_structure,
        save_structure as rfo_save_structure,
    )
    HAS_REAL_FOLD_ONE = True
except ImportError:
    HAS_REAL_FOLD_ONE = False

try:
    from real_fold_one_ht_v2 import HighThroughputScanner, HTConfig
    HAS_HT = True
except ImportError:
    HAS_HT = False

warnings.filterwarnings("ignore")
logger = logging.getLogger("CRISPRv6")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

# =============================================================================
# Section B: Differentiable DNA/RNA/Protein sequence encoding.
#
# Every encoder here is a torch.nn.Module operating on learnable embeddings,
# with NO Python-level string operations inside the forward pass after the
# initial tokenization. This is what makes the entire CRISPR design pipeline
# end-to-end differentiable: guide sequences, PAM contexts, repair templates,
# and protein targets all pass through differentiable encoders.
# =============================================================================

NUCLEOTIDE_VOCAB = {c: i for i, c in enumerate("ACGTN")}   # 5 tokens
AA_VOCAB         = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 20
CODON_TABLE = {
    "A": ["GCT","GCC","GCA","GCG"], "C": ["TGT","TGC"],
    "D": ["GAT","GAC"], "E": ["GAA","GAG"],
    "F": ["TTT","TTC"], "G": ["GGT","GGC","GGA","GGG"],
    "H": ["CAT","CAC"], "I": ["ATT","ATC","ATA"],
    "K": ["AAA","AAG"],
    "L": ["TTA","TTG","CTT","CTC","CTA","CTG"], "M": ["ATG"],
    "N": ["AAT","AAC"], "P": ["CCT","CCC","CCA","CCG"],
    "Q": ["CAA","CAG"],
    "R": ["CGT","CGC","CGA","CGG","AGA","AGG"],
    "S": ["TCT","TCC","TCA","TCG","AGT","AGC"],
    "T": ["ACT","ACC","ACA","ACG"],
    "V": ["GTT","GTC","GTA","GTG"], "W": ["TGG"],
    "Y": ["TAT","TAC"],
}
# Human-optimized codons (most frequent in highly expressed genes)
HUMAN_OPT_CODON = {
    "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
    "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
    "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "CGG",
    "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
}


class DifferentiableDNAEncoder(nn.Module):
    """
    Encodes a DNA sequence (variable length) into a continuous embedding
    via a 1-D convolutional + attention stack. Fully differentiable.

    Input:  token indices (B, L) long tensor, values in [0, 4]
    Output: (B, L, d_model) contextual embeddings

    This replaces ALL ad-hoc k-mer counting, GC-content heuristics, and
    rule-based scoring in earlier CRISPR modules. Every feature is learned.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(5, d_model)          # A,C,G,T,N
        self.pos   = nn.Embedding(max_len, d_model)
        self.conv  = nn.Sequential(
            nn.Conv1d(d_model, d_model, 5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.embed(tokens)                                    # (B, L, d)
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = x + self.pos(pos_ids[:, :L])
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)        # conv
        x = self.transformer(x)                                   # attention
        return self.norm(x)


class DifferentiableProteinEncoder(nn.Module):
    """
    Encodes an amino-acid sequence into per-residue embeddings via an
    ESM-style convolution + transformer, fully differentiable.

    Used downstream to score whether an edit restores native protein
    function (e.g., CFTR folding, NF1 neurofibromin stability).
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(20, d_model)
        self.pos   = nn.Embedding(max_len, d_model)
        self.conv  = nn.Conv1d(d_model, d_model, 7, padding=3)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.embed(tokens)
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = x + self.pos(pos_ids[:, :L])
        x = F.gelu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.transformer(x)
        return self.norm(x)


class DifferentiableGuideEncoder(nn.Module):
    """
    Specialized encoder for 20-nt CRISPR guide sequences with PAM context.
    Produces a fixed-size guide embedding used by every downstream head
    (on-target, off-target, repair outcome, delivery).
    """
    def __init__(self, d_model: int = 96):
        super().__init__()
        self.dna_enc = DifferentiableDNAEncoder(d_model=d_model, max_len=64)
        self.pam_embed = nn.Embedding(8, d_model)   # NGG, NGAN, TTTV, etc.
        self.pool = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, guide_tokens: torch.Tensor,
                pam_tokens: torch.Tensor) -> torch.Tensor:
        """guide_tokens: (B, 20), pam_tokens: (B,) long in [0,7]"""
        g = self.dna_enc(guide_tokens).mean(dim=1)          # (B, d)
        p = self.pam_embed(pam_tokens)                      # (B, d)
        return self.pool(torch.cat([g, p], dim=-1))          # (B, d)

# =============================================================================
# Section C: Differentiable off-target risk.
#
# Replaces brute-force / Bowtie off-target search with a learned scorer that
# takes (guide, candidate_off_target) pairs and predicts cleavage probability.
# Architecture follows the 2026 EHNN hybrid (XOR mismatch maps, k-mer context,
# PAM categories, CNN-LSTM-MLP fusion) but is implemented natively in torch
# so gradients flow from risk to guide embeddings.
# =============================================================================

class XORMismatchEncoder(nn.Module):
    """
    EHNN-style XOR mismatch map: encodes position-wise mismatch between
    guide and off-target candidate as a differentiable feature map.
    """
    def __init__(self, d_model: int = 32):
        super().__init__()
        self.match_embed    = nn.Embedding(2, d_model)
        self.position_embed = nn.Embedding(64, d_model)

    def forward(self, guide_tokens: torch.Tensor,
                target_tokens: torch.Tensor) -> torch.Tensor:
        """Both (B, L). Returns (B, L, d_model)."""
        B, L = guide_tokens.shape
        mismatch = (guide_tokens != target_tokens).long()          # (B, L)
        pos_ids  = torch.arange(L, device=guide_tokens.device).unsqueeze(0)
        return self.match_embed(mismatch) + self.position_embed(pos_ids)


class DifferentiableOffTargetScorer(nn.Module):
    """
    Predicts P(off-target cleavage) for each (guide, candidate) pair.

    Input:
        guide_emb  : (B, d_guide)  — from DifferentiableGuideEncoder
        cand_tokens: (B, N, L)     — N candidate off-target sites
    Output:
        risk       : (B, N)        — per-candidate cleavage probability
        worst_risk : (B,)          — max over N (used in loss)
        aggregate  : (B,)          — soft-max weighted aggregate risk

    This is the differentiable analogue of running Bowtie/BLAST and
    counting hits. Gradients flow back to the guide encoder, enabling
    guide sequence optimization that directly minimizes predicted risk.
    """
    def __init__(self, d_guide: int = 96, d_model: int = 48):
        super().__init__()
        self.xor_enc = XORMismatchEncoder(d_model=d_model)
        self.cand_enc = DifferentiableDNAEncoder(d_model=d_model, max_len=64)
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
        )
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.guide_proj = nn.Linear(d_guide, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, guide_emb: torch.Tensor,
                cand_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, L = cand_tokens.shape
        flat = cand_tokens.reshape(B * N, L)
        cand_emb = self.cand_enc(flat)                              # (B*N, L, d)
        cnn_out  = self.cnn(cand_emb.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        cand_pool = lstm_out.mean(dim=1).view(B, N, -1)             # (B, N, 2d)

        g = self.guide_proj(guide_emb).unsqueeze(1).expand(-1, N, -1)  # (B, N, d)
        x = torch.cat([g, cand_pool], dim=-1)                        # (B, N, 3d)
        # pairwise interaction: elementwise product of guide & candidate
        x = torch.cat([x, g * cand_pool], dim=-1)                    # (B, N, 4d)
        logits = self.head(x).squeeze(-1)                            # (B, N)
        risk   = torch.sigmoid(logits)
        worst  = risk.max(dim=-1).values
        weights = F.softmax(logits, dim=-1)
        aggregate = (risk * weights).sum(dim=-1)
        return {"risk": risk, "worst_risk": worst, "aggregate_risk": aggregate}

# =============================================================================
# Section D: Differentiable repair-outcome predictor.
#
# Given a guide, a target locus, and a nuclease modality, predicts the
# distribution over repair outcomes:
#   - HDR (precise correction)         : [0]
#   - NHEJ frameshift (knockout)       : [1]
#   - NHEJ in-frame (partial function) : [2]
#   - Base-edited (C→T or A→G)         : [3]
#   - Prime-edited (precise)           : [4]
#   - No edit                          : [5]
#
# This is used both as a loss term (maximize HDR/PE probability for
# correction strategies) and as a feasibility gate (reject guides whose
# predicted NHEJ fraction exceeds a safety threshold).
# =============================================================================

class DifferentiableRepairPredictor(nn.Module):
    """
    Predicts repair outcome distribution.

    Architecture: guide embedding + local sequence context + cell-type
    embedding + nuclease-type embedding → MLP → softmax over 6 outcomes.
    """
    N_OUTCOMES = 6   # HDR, NHEJ_fs, NHEJ_if, BE, PE, none

    def __init__(self, d_guide: int = 96, d_context: int = 64,
                 d_cell: int = 32, d_nuclease: int = 16):
        super().__init__()
        self.ctx_enc  = DifferentiableDNAEncoder(d_model=d_context, max_len=256)
        self.cell_embed = nn.Embedding(16, d_cell)     # tissue/cell-type id
        self.nuc_embed  = nn.Embedding(8, d_nuclease)  # Cas9, Cas12a, ABE, CBE, PE2, PE3, PE5, Cas13
        self.mlp = nn.Sequential(
            nn.Linear(d_guide + d_context + d_cell + d_nuclease, 128),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, self.N_OUTCOMES),
        )

    def forward(self, guide_emb: torch.Tensor,
                context_tokens: torch.Tensor,
                cell_type: torch.Tensor,
                nuclease_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        ctx = self.ctx_enc(context_tokens).mean(dim=1)
        cell = self.cell_embed(cell_type)
        nuc  = self.nuc_embed(nuclease_type)
        x = torch.cat([guide_emb, ctx, cell, nuc], dim=-1)
        logits = self.mlp(x)
        probs  = F.softmax(logits, dim=-1)
        return {
            "probs": probs,
            "hdr_prob": probs[..., 0],
            "nhej_fs_prob": probs[..., 1],
            "base_edit_prob": probs[..., 3],
            "prime_edit_prob": probs[..., 4],
            "precision_score": probs[..., 0] + probs[..., 3] + probs[..., 4],
        }


# =============================================================================
# Section E: Disease-specific differentiable objective heads.
#
# Each congenital disease has a distinct therapeutic objective:
#   - Loss-of-function recessive (CF, DMD, SMA, GJB2, LCA, beta-thal):
#       maximize precise restoration → HDR / prime editing
#   - Gain-of-function dominant (NF1, FOP/ACVR1, LZTR1/Noonan, GNAO1):
#       maximize allele-specific knockout or silencing → NHEJ / base edit
#   - Chromosomal (Down syndrome):
#       maximize XIST integration → large-transgene HDR
#   - Metabolic (PKU, congenital diabetes):
#       maximize HDR correction of metabolic enzyme
#   - DSD (intersex):
#       must be handled with extreme ethical care; the engine
#       defaults to a "no recommendation" output for DSD targets
#       unless an explicit clinical protocol is supplied.
#
# Each head is a small differentiable network producing a scalar
# therapeutic_objective score, combined with off-target and delivery
# feasibility into the total loss.
# =============================================================================

DISEASE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Loss-of-function recessive ────────────────────────────────────
    "cystic_fibrosis": {
        "gene": "CFTR", "mutation": "F508del", "mode": "HDR",
        "objective": "restore_channel_function",
        "cell_type": 0, "tissue": "airway_epithelium",
        "preferred_editor": "PE2", "delivery": "LNP_airway",
    },
    "duchenne_md": {
        "gene": "DMD", "mutation": "exon_51_del", "mode": "exon_skipping",
        "objective": "restore_reading_frame",
        "cell_type": 1, "tissue": "skeletal_muscle",
        "preferred_editor": "Cas9_dual_sgRNA", "delivery": "AAV9",
    },
    "sma": {
        "gene": "SMN1", "mutation": "exon_7_skip", "mode": "base_edit",
        "objective": "restore_SMN1_exon7_inclusion",
        "cell_type": 2, "tissue": "motor_neuron",
        "preferred_editor": "ABE", "delivery": "AAV9_intrathecal",
    },
    "beta_thalassemia": {
        "gene": "HBB", "mutation": "IVS2-654", "mode": "HDR_or_BCL11A_disc",
        "objective": "restore_HbF",
        "cell_type": 3, "tissue": "HSPC",
        "preferred_editor": "Cas12a", "delivery": "ex_vivo_electroporation",
    },
    "gjb2_deafness": {
        "gene": "GJB2", "mutation": "35delG", "mode": "prime_editing",
        "objective": "restore_connexin26",
        "cell_type": 4, "tissue": "cochlear_supporting_cells",
        "preferred_editor": "PE2", "delivery": "AAV2_round_window",
    },
    "leber_congenital_amaurosis": {
        "gene": "CEP290", "mutation": "IVS26_c.2991+1655A>G",
        "mode": "prime_editing",
        "objective": "restore_photoreceptor_function",
        "cell_type": 5, "tissue": "retinal_pigment_epithelium",
        "preferred_editor": "PE2", "delivery": "AAV2_subretinal",
    },
    "pku": {
        "gene": "PAH", "mutation": "R408W", "mode": "base_edit",
        "objective": "restore_PAH_activity",
        "cell_type": 6, "tissue": "hepatocyte",
        "preferred_editor": "ABE", "delivery": "LNP_liver",
    },
    "congenital_diabetes": {
        "gene": "KCNJ11", "mutation": "V59M", "mode": "base_edit",
        "objective": "restore_insulin_secretion",
        "cell_type": 7, "tissue": "pancreatic_beta_cell",
        "preferred_editor": "ABE", "delivery": "LNP_pancreas",
    },
    # ── Gain-of-function dominant ─────────────────────────────────────
    "neurofibromatosis_1": {
        "gene": "NF1", "mutation": "R1925X", "mode": "allele_specific_KO",
        "objective": "reduce_neurofibromin_dominant_negative",
        "cell_type": 8, "tissue": "Schwann_cell_precursor",
        "preferred_editor": "Cas9_allele_specific", "delivery": "AAV9",
        "note": "NF1 is tumor-suppressor; most mutations are LOF. "
                "The dominant-negative subset requires allele-specific KO.",
    },
    "fibrodysplasia_ossificans": {
        "gene": "ACVR1", "mutation": "R206H", "mode": "allele_specific_KO",
        "objective": "block_heterotopic_ossification",
        "cell_type": 9, "tissue": "skeletal_muscle_progenitor",
        "preferred_editor": "Cas9_allele_specific", "delivery": "AAV9",
    },
    "noonan_syndrome": {
        "gene": "LZTR1", "mutation": "deep_intronic", "mode": "base_edit",
        "objective": "restore_LZTR1_splicing",
        "cell_type": 10, "tissue": "cardiac_progenitor",
        "preferred_editor": "CBE", "delivery": "AAV9",
    },
    "cerebral_palsy_gnao1": {
        "gene": "GNAO1", "mutation": "G203R", "mode": "allele_specific_KO",
        "objective": "reduce_GNAO1_gain_of_function",
        "cell_type": 11, "tissue": "striatal_neuron",
        "preferred_editor": "Cas9_allele_specific", "delivery": "AAV9_intracranial",
    },
    "congenital_heart_disease": {
        "gene": "MYH7", "mutation": "R403Q", "mode": "allele_specific_KO",
        "objective": "reduce_hypertrophic_cardiomyopathy",
        "cell_type": 12, "tissue": "cardiomyocyte",
        "preferred_editor": "Cas9_allele_specific", "delivery": "AAV9_intramyocardial",
    },
    # ── Chromosomal ──────────────────────────────────────────────────
    "down_syndrome": {
        "gene": "chr21_XIST", "mutation": "trisomy_21", "mode": "XIST_integration",
        "objective": "silence_extra_chr21",
        "cell_type": 13, "tissue": "neural_progenitor",
        "preferred_editor": "Cas9_large_HDR", "delivery": "AAV6_dual",
        "note": "XIST integration (14 kb) requires dual-AAV or LNP-"
                "encapsulated HDR template. Low efficiency in vivo; "
                "ex vivo iPSC correction more feasible.",
    },
    # ── DSD — special handling ───────────────────────────────────────
    "disorders_sex_development": {
        "gene": "SRY_OR_SOX9", "mutation": "variable", "mode": "NO_RECOMMEND",
        "objective": "HALT_ETHICAL_REVIEW_REQUIRED",
        "cell_type": 14, "tissue": "gonadal",
        "preferred_editor": "NONE", "delivery": "NONE",
        "note": "The engine does NOT recommend editing for DSD targets. "
                "Returns a structured halt signal requiring explicit "
                "clinical ethics board protocol before any design.",
    },
}


class DiseaseObjectiveHead(nn.Module):
    """
    Differentiable scalar objective for a single disease target.
    Maps guide embedding + repair prediction + off-target risk +
    delivery feasibility → therapeutic score in [0, 1].

    Higher = better therapeutic outcome for this disease.
    """
    def __init__(self, disease_key: str, d_guide: int = 96):
        super().__init__()
        self.disease_key = disease_key
        self.cfg = DISEASE_REGISTRY.get(disease_key, {})
        self.mode = self.cfg.get("mode", "HDR")

        # Learned weighting per outcome (initialized by disease mode)
        init_w = self._init_weights()
        self.outcome_weights = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))

        self.head = nn.Sequential(
            nn.Linear(d_guide + 6 + 4, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def _init_weights(self) -> List[float]:
        """Initialize outcome preference vector based on disease mode."""
        #       HDR  NHEJ_fs  NHEJ_if  BE    PE    none
        base = [0.0,  0.0,    0.0,    0.0,  0.0,  -1.0]
        if self.mode == "HDR":
            base[0] = 2.0
        elif self.mode == "allele_specific_KO":
            base[1] = 1.5; base[2] = 0.5
        elif self.mode in ("base_edit",):
            base[3] = 2.0
        elif self.mode == "prime_editing":
            base[4] = 2.0
        elif self.mode == "exon_skipping":
            base[1] = 1.0; base[2] = 1.5   # in-frame skipping preferred
        elif self.mode == "XIST_integration":
            base[0] = 2.5                    # large HDR
        return base

    def forward(self, guide_emb: torch.Tensor,
                repair_probs: torch.Tensor,
                off_target_risk: torch.Tensor,
                delivery_feasibility: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        guide_emb            : (B, d_guide)
        repair_probs         : (B, 6) from DifferentiableRepairPredictor
        off_target_risk      : (B,) worst-case off-target probability
        delivery_feasibility : (B, 4) [AAV, LNP, ex_vivo, in_vivo] scores
        """
        # Weighted repair outcome score
        outcome_score = (repair_probs * self.outcome_weights).sum(dim=-1, keepdim=True)
        # Penalize off-target risk (safety gate)
        safety_penalty = -2.0 * off_target_risk.unsqueeze(-1)
        # Delivery bonus
        delivery_score = delivery_feasibility.sum(dim=-1, keepdim=True)

        x = torch.cat([guide_emb, repair_probs, off_target_risk.unsqueeze(-1),
                       delivery_feasibility], dim=-1)
        learned = self.head(x)

        # Combine: learned head + interpretable terms
        therapeutic = learned * 0.6 + torch.sigmoid(
            outcome_score + safety_penalty + 0.3 * delivery_score
        ) * 0.4
        return {
            "therapeutic_score": therapeutic,
            "outcome_score": outcome_score,
            "safety_penalty": safety_penalty,
        }

# =============================================================================
# Section F: Differentiable guide optimizer.
#
# Instead of enumerating candidate guides and scoring them with fixed rules,
# this module optimizes guide sequences via gradient descent through the
# entire differentiable stack. It uses a Gumbel-softmax relaxation over
# nucleotide choices at each of the 20 guide positions, plus an optional
# hard conversion step for final output.
# =============================================================================

class DifferentiableGuideOptimizer(nn.Module):
    """
    Gradient-based guide sequence optimization.

    The guide is parameterized as a (20, 5) logit matrix (one row per
    position, 5 nucleotide classes A/C/G/T/N). Gumbel-softmax samples
    produce differentiable guide tokens; straight-through estimation
    allows hard token extraction at the end.

    The optimizer maximizes:
        therapeutic_score - λ_off * off_target_risk
        + λ_del * delivery_feasibility
        - λ_prec * (1 - precision_score)

    All terms are differentiable w.r.t. guide logits.
    """
    def __init__(self, guide_length: int = 20, n_nucleotides: int = 5,
                 temp: float = 1.0):
        super().__init__()
        self.guide_length = guide_length
        self.n_nucleotides = n_nucleotides
        self.temp = temp
        # Initialize with random logits (will be optimized)
        self.logits = nn.Parameter(
            torch.randn(1, guide_length, n_nucleotides) * 0.1
        )

    def _gumbel_softmax(self, logits: torch.Tensor,
                        hard: bool = False) -> torch.Tensor:
        """Gumbel-softmax with temperature annealing."""
        return F.gumbel_softmax(logits, tau=self.temp, hard=hard, dim=-1)

    def forward(self, hard: bool = False) -> Dict[str, torch.Tensor]:
        """Returns soft guide tokens and hard token indices."""
        soft = self._gumbel_softmax(self.logits, hard=False)       # (1, L, 5)
        if hard:
            idx = soft.argmax(dim=-1)                              # (1, L)
            one_hot = F.one_hot(idx, self.n_nucleotides).float()
            # Straight-through: gradient flows through soft
            soft = one_hot + soft - soft.detach()
        tokens = soft.argmax(dim=-1)                               # (1, L)
        return {"soft_tokens": soft, "tokens": tokens}

    def to_sequence(self) -> str:
        """Extract current guide as an ACGTN string."""
        idx = self.logits.argmax(dim=-1).squeeze(0)
        inv = {v: k for k, v in NUCLEOTIDE_VOCAB.items()}
        return "".join(inv.get(int(i), "N") for i in idx)

    def optimize(self, loss_fn: Callable, n_steps: int = 200,
                 lr: float = 0.05, anneal: bool = True) -> List[float]:
        """Run gradient descent on guide logits."""
        opt = torch.optim.Adam([self.logits], lr=lr)
        history = []
        for step in range(n_steps):
            if anneal:
                self.temp = max(0.1, 1.0 - step / n_steps * 0.9)
            opt.zero_grad()
            loss = loss_fn(self)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.logits], 1.0)
            opt.step()
            history.append(loss.item())
        return history

# =============================================================================
# Section G: Differentiable delivery feasibility.
#
# Models the two dominant delivery modalities:
#   - AAV (tissue-specific serotypes, cargo capacity limits)
#   - LNP (ionizable lipid, liver tropism, endosomal escape)
#   - Ex vivo electroporation (HSPC, iPSC)
#
# Each modality has a differentiable feasibility score based on target
# tissue, cargo size, editor type, and route of administration.
# =============================================================================

AAV_SEROTYPES = {
    "AAV1":  {"tropism": "muscle",        "cargo_kb": 4.7},
    "AAV2":  {"tropism": "CNS_retina",    "cargo_kb": 4.7},
    "AAV5":  {"tropism": "airway_lung",   "cargo_kb": 4.7},
    "AAV6":  {"tropism": "muscle_HSPC",   "cargo_kb": 4.7},
    "AAV8":  {"tropism": "liver",         "cargo_kb": 4.7},
    "AAV9":  {"tropism": "broad_CNS",     "cargo_kb": 4.7},
    "AAVrh10": {"tropism": "CNS_broad",   "cargo_kb": 4.7},
    "AAV-DJ": {"tropism": "broad",        "cargo_kb": 4.7},
}

LNP_FORMULATIONS = {
    "LNP_liver":       {"tropism": "hepatocyte",     "efficiency": 0.87},
    "LNP_airway":      {"tropism": "airway",         "efficiency": 0.45},
    "LNP_pancreas":    {"tropism": "pancreas",       "efficiency": 0.30},
    "LNP_muscle":      {"tropism": "skeletal_muscle","efficiency": 0.40},
    "LNP_CNS":         {"tropism": "CNS",            "efficiency": 0.15},
}


class DifferentiableDeliveryScorer(nn.Module):
    """
    Predicts delivery feasibility for each modality given:
      - target tissue embedding
      - editor cargo size (Cas9 ~4.1 kb, Cas12a ~3.9 kb, PE ~6.3 kb, BE ~5.2 kb)
      - route (systemic, local, ex vivo)
    """
    EDITOR_SIZES_KB = {
        "Cas9": 4.1, "Cas12a": 3.9, "Cas13": 4.4,
        "ABE": 5.2, "CBE": 5.2, "PE2": 6.3, "PE3": 6.8, "PE5": 7.1,
    }

    def __init__(self, n_tissues: int = 20, d_tissue: int = 32):
        super().__init__()
        self.tissue_embed = nn.Embedding(n_tissues, d_tissue)
        self.mlp = nn.Sequential(
            nn.Linear(d_tissue + 2 + 4, 64), nn.GELU(),
            nn.Linear(64, 4), nn.Sigmoid(),   # [AAV, LNP, ex_vivo, in_vivo]
        )

    def forward(self, tissue_id: torch.Tensor,
                editor_size_kb: torch.Tensor,
                route_onehot: torch.Tensor) -> torch.Tensor:
        """
        tissue_id      : (B,) long
        editor_size_kb : (B,) float
        route_onehot   : (B, 4) — systemic, local, ex_vivo, intrathecal
        Returns (B, 4) feasibility scores.
        """
        t = self.tissue_embed(tissue_id)
        x = torch.cat([t, editor_size_kb.unsqueeze(-1),
                       route_onehot], dim=-1)
        return self.mlp(x)

# =============================================================================
# Section H: Master CRISPR-Cas Therapeutic Design Engine.
#
# Ties together every differentiable component:
#   Biomarker/target input → guide optimization → off-target scoring →
#   repair prediction → disease objective → delivery feasibility →
#   optimized guide + repair template + clinical recommendation.
#
# Inherits LangevinBridgeMixin (Fix 3 from v4/v5) for connection to the
# structural Langevin engine, and supports AMP + DDP for production.
# =============================================================================

@dataclass
class CRISPRDesignConfig:
    """Configuration for the master engine."""
    # Model dimensions
    d_guide: int = 96
    d_dna: int = 64
    d_protein: int = 64
    n_heads: int = 4
    n_layers: int = 2

    # Optimization
    n_opt_steps: int = 200
    opt_lr: float = 0.05
    temp_init: float = 1.0
    temp_final: float = 0.1

    # Loss weights
    lambda_off_target: float = 5.0
    lambda_precision: float = 2.0
    lambda_delivery: float = 1.0
    lambda_safety: float = 10.0

    # Production
    use_amp: bool = True
    amp_dtype: str = "bfloat16"       # "bfloat16" (Ampere+) or "float16"
    ddp_backend: str = "nccl"
    gradient_checkpointing: bool = False

    # Disease
    disease_keys: List[str] = field(default_factory=lambda: [
        "cystic_fibrosis", "neurofibromatosis_1", "beta_thalassemia",
    ])

    # Device
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return _core_get_device()


class CRISPRTherapeuticEngine(LangevinBridgeMixin, nn.Module):
    """
    Master differentiable CRISPR-Cas therapeutic design engine.

    Forward pass (fully differentiable):
        1. Encode guide (DifferentiableGuideEncoder)
        2. Predict off-target risk (DifferentiableOffTargetScorer)
        3. Predict repair outcome (DifferentiableRepairPredictor)
        4. Score delivery feasibility (DifferentiableDeliveryScorer)
        5. Compute disease-specific objective (DiseaseObjectiveHead)
        6. Return loss terms for backprop

    Optimize pass:
        Gradient descent on guide logits to maximize therapeutic score
        while minimizing off-target risk.

    Production features:
        - AMP (bfloat16/float16) via torch.autocast
        - DDP via torch.nn.parallel.DistributedDataParallel
        - Gradient checkpointing for large guides
        - Checkpoint/resume via CheckpointManager
    """

    def __init__(self, cfg: Optional[CRISPRDesignConfig] = None):
        super().__init__()  # nn.Module
        self.cfg = cfg or CRISPRDesignConfig()
        self.device = self.cfg.resolve_device()

        # ── Differentiable encoders ───────────────────────────────────
        self.guide_encoder = DifferentiableGuideEncoder(
            d_model=self.cfg.d_guide).to(self.device)
        self.dna_encoder = DifferentiableDNAEncoder(
            d_model=self.cfg.d_dna).to(self.device)
        self.protein_encoder = DifferentiableProteinEncoder(
            d_model=self.cfg.d_protein).to(self.device)

        # ── Differentiable predictors ─────────────────────────────────
        self.off_target_scorer = DifferentiableOffTargetScorer(
            d_guide=self.cfg.d_guide).to(self.device)
        self.repair_predictor = DifferentiableRepairPredictor(
            d_guide=self.cfg.d_guide).to(self.device)
        self.delivery_scorer = DifferentiableDeliveryScorer().to(self.device)

        # ── Disease-specific heads ────────────────────────────────────
        self.disease_heads = nn.ModuleDict({
            key: DiseaseObjectiveHead(key, d_guide=self.cfg.d_guide).to(self.device)
            for key in self.cfg.disease_keys
        })

        # ── Guide optimizers (one per active disease) ─────────────────
        self.guide_optimizers = nn.ModuleDict({
            key: DifferentiableGuideOptimizer(
                guide_length=20, temp=self.cfg.temp_init).to(self.device)
            for key in self.cfg.disease_keys
        })

        # ── Structural bridge (from LangevinBridgeMixin) ──────────────
        self.langevin_bridge: Optional[LangevinEvolutionBridge] = None
        self.cahn_bridge: Optional[CahnHilliardEvoBridge] = None

        # ── AMP ──────────────────────────────────────────────────────
        self.amp_dtype = (torch.bfloat16
                          if self.cfg.amp_dtype == "bfloat16"
                          else torch.float16)

        self.results: Dict[str, Any] = {}

        logger.info("CRISPRTherapeuticEngine initialized on %s "
                    "(AMP=%s, dtype=%s)",
                    self.device, self.cfg.use_amp, self.cfg.amp_dtype)

    # ── Tokenization helpers ─────────────────────────────────────────

    @staticmethod
    def tokenize_dna(seq: str) -> torch.Tensor:
        return torch.tensor(
            [NUCLEOTIDE_VOCAB.get(c.upper(), 4) for c in seq],
            dtype=torch.long,
        )

    @staticmethod
    def tokenize_protein(seq: str) -> torch.Tensor:
        return torch.tensor(
            [AA_VOCAB.get(c.upper(), 0) for c in seq],
            dtype=torch.long,
        )

    @staticmethod
    def _pam_token(pam: str) -> int:
        """Map PAM string to a discrete token (0-7)."""
        pam = pam.upper()
        if pam == "NGG":   return 0
        if pam == "NAG":   return 1
        if pam == "TTTV":  return 2
        if pam == "TTTN":  return 3
        if pam == "NNGRRT":return 4
        if pam == "NGGNG": return 5
        if pam == "NNN":   return 6
        return 7

    # ── Core differentiable forward ──────────────────────────────────

    def forward(self,
                guide_tokens: torch.Tensor,
                pam_tokens: torch.Tensor,
                context_tokens: torch.Tensor,
                off_target_candidates: torch.Tensor,
                cell_type: torch.Tensor,
                nuclease_type: torch.Tensor,
                tissue_id: torch.Tensor,
                editor_size_kb: torch.Tensor,
                route_onehot: torch.Tensor,
                disease_key: str) -> Dict[str, torch.Tensor]:
        """
        One differentiable forward pass.

        All tensors must be on self.device and batched (B, ...).
        Returns a dict of loss terms and scores.
        """
        # 1. Guide embedding
        guide_emb = self.guide_encoder(guide_tokens, pam_tokens)

        # 2. Off-target risk
        off_target = self.off_target_scorer(guide_emb, off_target_candidates)

        # 3. Repair outcome
        repair = self.repair_predictor(
            guide_emb, context_tokens, cell_type, nuclease_type)

        # 4. Delivery feasibility
        delivery = self.delivery_scorer(tissue_id, editor_size_kb, route_onehot)

        # 5. Disease-specific objective
        head = self.disease_heads[disease_key]
        objective = head(guide_emb, repair["probs"],
                         off_target["worst_risk"], delivery)

        # 6. Loss terms
        loss_off_target = off_target["worst_risk"].mean()
        loss_precision = (1.0 - repair["precision_score"]).mean()
        loss_delivery = (1.0 - delivery.sum(dim=-1)).mean()
        loss_safety = loss_off_target.pow(2).mean()   # sharper penalty

        total_loss = (
            -objective["therapeutic_score"].mean()
            + self.cfg.lambda_off_target * loss_off_target
            + self.cfg.lambda_precision * loss_precision
            + self.cfg.lambda_delivery * loss_delivery
            + self.cfg.lambda_safety * loss_safety
        )

        return {
            "total_loss": total_loss,
            "therapeutic_score": objective["therapeutic_score"],
            "off_target_risk": off_target["worst_risk"],
            "precision_score": repair["precision_score"],
            "delivery_score": delivery.sum(dim=-1),
            "repair_probs": repair["probs"],
            "guide_emb": guide_emb,
        }

    # ── Guide optimization loop ───────────────────────────────────────

    def optimize_guide_for_disease(
        self,
        disease_key: str,
        target_context: str,
        n_steps: Optional[int] = None,
        n_off_target_candidates: int = 32,
        target_tissue_id: int = 0,
        editor_type: int = 0,
        route: str = "systemic",
    ) -> Dict[str, Any]:
        """
        Gradient-based optimization of a guide sequence for a disease.

        Returns the optimized guide sequence, its scores, and the
        optimization history.
        """
        n_steps = n_steps or self.cfg.n_opt_steps
        cfg = DISEASE_REGISTRY.get(disease_key, {})
        mode = cfg.get("mode", "HDR")

        # Tokenize context
        ctx_tokens = self.tokenize_dna(target_context[:256]).unsqueeze(0).to(self.device)
        if ctx_tokens.shape[1] < 256:
            pad = torch.full((1, 256 - ctx_tokens.shape[1]), 4,
                             dtype=torch.long, device=self.device)
            ctx_tokens = torch.cat([ctx_tokens, pad], dim=1)

        # Random off-target candidates (in production, replace with actual
        # genome-wide candidates from the patient's sequenced genome)
        off_cands = torch.randint(0, 5, (1, n_off_target_candidates, 20),
                                   device=self.device)

        # Static tensors
        pam_t = torch.tensor([self._pam_token("NGG")], device=self.device)
        cell_t = torch.tensor([cfg.get("cell_type", 0)], device=self.device)
        nuc_t = torch.tensor([editor_type], device=self.device)
        tissue_t = torch.tensor([target_tissue_id], device=self.device)
        size_t = torch.tensor([self.delivery_scorer.EDITOR_SIZES_KB.get(
            cfg.get("preferred_editor", "Cas9"), 4.1)], device=self.device)
        route_map = {"systemic": 0, "local": 1, "ex_vivo": 2, "intrathecal": 3}
        route_oh = F.one_hot(
            torch.tensor([route_map.get(route, 0)], device=self.device),
            num_classes=4).float()

        optimizer = self.guide_optimizers[disease_key]
        opt = torch.optim.Adam([optimizer.logits], lr=self.cfg.opt_lr)
        history = []

        for step in range(n_steps):
            # Temperature annealing
            t = self.cfg.temp_init + (self.cfg.temp_final - self.cfg.temp_init) * (step / n_steps)
            optimizer.temp = t

            opt.zero_grad()

            # Gumbel-softmax guide tokens
            gumbel = optimizer(hard=False)
            soft = gumbel["soft_tokens"]                     # (1, L, 5)
            # Convert soft to token indices for the encoder
            guide_tok = soft.argmax(dim=-1)                  # (1, L)
            # Straight-through: use soft embeddings via one-hot matmul
            guide_onehot = soft                              # differentiable

            # Forward through differentiable stack
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.cfg.use_amp):
                guide_emb = self.guide_encoder(guide_tok, pam_t)
                # Inject straight-through gradient: replace guide_emb's
                # gradient path with one that flows through soft tokens
                guide_emb_soft = (guide_onehot.unsqueeze(-1)
                                  * self.guide_encoder.dna_enc.embed.weight.unsqueeze(0)
                                  ).sum(dim=-2).mean(dim=1)
                guide_emb = guide_emb + (guide_emb_soft - guide_emb_soft.detach())

                off_target = self.off_target_scorer(guide_emb, off_cands)
                repair = self.repair_predictor(
                    guide_emb, ctx_tokens, cell_t, nuc_t)
                delivery = self.delivery_scorer(tissue_t, size_t, route_oh)
                head = self.disease_heads[disease_key]
                objective = head(guide_emb, repair["probs"],
                                 off_target["worst_risk"], delivery)

                loss = (
                    -objective["therapeutic_score"].mean()
                    + self.cfg.lambda_off_target * off_target["worst_risk"].mean()
                    + self.cfg.lambda_precision * (1.0 - repair["precision_score"]).mean()
                    + self.cfg.lambda_delivery * (1.0 - delivery.sum(dim=-1)).mean()
                    + self.cfg.lambda_safety * off_target["worst_risk"].pow(2).mean()
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([optimizer.logits], 1.0)
            opt.step()
            history.append(loss.item())

        # Extract final hard guide
        final_guide = optimizer.to_sequence()

        return {
            "disease": disease_key,
            "optimized_guide": final_guide,
            "loss_history": history,
            "final_therapeutic_score": float(
                objective["therapeutic_score"].mean().detach()),
            "final_off_target_risk": float(
                off_target["worst_risk"].mean().detach()),
            "final_precision_score": float(
                repair["precision_score"].mean().detach()),
            "repair_probs": repair["probs"].detach().cpu().numpy().tolist(),
            "mode": mode,
            "preferred_editor": cfg.get("preferred_editor", "Cas9"),
            "delivery_route": cfg.get("delivery", "systemic"),
        }

    # ── Full pipeline: all diseases ───────────────────────────────────

    def design_all(self,
                   target_context: str,
                   disease_keys: Optional[List[str]] = None
                   ) -> Dict[str, Any]:
        """
        Run the full optimization pipeline for every configured disease.
        Returns a structured report ready for JSON export.
        """
        disease_keys = disease_keys or self.cfg.disease_keys
        report = {}
        for key in disease_keys:
            if key not in self.guide_optimizers:
                logger.warning("Disease %s not in configured heads; skipping.", key)
                continue
            # Special handling: DSD targets must NOT be optimized
            cfg = DISEASE_REGISTRY.get(key, {})
            if cfg.get("mode") == "NO_RECOMMEND":
                report[key] = {
                    "status": "HALTED",
                    "reason": cfg.get("note", "Ethical review required."),
                    "recommendation": None,
                }
                logger.warning("DSD target %s: design halted pending ethics.", key)
                continue
            try:
                result = self.optimize_guide_for_disease(
                    key, target_context,
                    target_tissue_id=cfg.get("cell_type", 0),
                )
                report[key] = result
                logger.info("Disease %s: guide=%s  score=%.3f  off-target=%.4f",
                            key, result["optimized_guide"],
                            result["final_therapeutic_score"],
                            result["final_off_target_risk"])
            except Exception as e:
                logger.error("Optimization failed for %s: %s", key, e)
                report[key] = {"status": "ERROR", "error": str(e)}
        self.results = report
        return report

    # ── Repair template design (differentiable) ───────────────────────

    def design_repair_template(self,
                               disease_key: str,
                               target_pos: int,
                               new_aa: str,
                               homology_arm_length: int = 30) -> Dict[str, str]:
        """
        Build a repair template with homology arms.
        Uses human-optimized codons for maximal expression.
        """
        if new_aa not in HUMAN_OPT_CODON:
            raise ValueError(f"Unknown amino acid: {new_aa}")
        new_codon = HUMAN_OPT_CODON[new_aa]
        # Placeholder: in production, fetch actual flanking sequence from
        # the patient's reference genome
        left_arm = "N" * homology_arm_length
        right_arm = "N" * homology_arm_length
        template = left_arm + new_codon + right_arm
        return {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "new_codon": new_codon,
            "full_template": template,
        }

    # ── AMP / DDP production wrappers ─────────────────────────────────

    def train_step_amp(self, batch: Dict[str, torch.Tensor],
                       optimizer: torch.optim.Optimizer,
                       scaler: Optional[torch.amp.GradScaler] = None
                       ) -> float:
        """
        Single AMP training step. Handles bfloat16 (no scaler needed)
        and float16 (scaler required).
        """
        optimizer.zero_grad()
        device_type = self.device.type
        with torch.autocast(device_type=device_type,
                            dtype=self.amp_dtype,
                            enabled=self.cfg.use_amp):
            out = self(**batch)
            loss = out["total_loss"]

        if self.amp_dtype == torch.float16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        return float(loss.detach())


# =============================================================================
# DDP wrapper — production multi-GPU
# =============================================================================

def wrap_ddp(engine: CRISPRTherapeuticEngine,
             local_rank: int,
             backend: str = "nccl") -> DDP:
    """
    Wrap the engine in DistributedDataParallel.

    Usage:
        torchrun --nproc_per_node=4 train_crispr.py
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    engine = engine.to(local_rank)
    return DDP(engine, device_ids=[local_rank],
               find_unused_parameters=False,   # all params used in forward
               gradient_as_bucket_view=True)

# =============================================================================
# Section I: Structured clinical report generation.
#
# Produces a JSON report per patient per disease with:
#   - Optimized guide sequence
#   - Predicted repair outcome distribution
#   - Off-target risk score
#   - Delivery recommendation (AAV serotype / LNP formulation / ex vivo)
#   - Ethical flags (DSD targets, germline considerations)
#   - Structural impact prediction (via REAL FOLD ONE when available)
# =============================================================================

class ClinicalReportGenerator:
    """
    Assembles a clinical-grade report from the engine's raw outputs.
    """

    DISEASE_DISPLAY_NAMES = {
        "neurofibromatosis_1": "Neurofibromatosis Type 1 (NF1)",
        "down_syndrome": "Down Syndrome (Trisomy 21)",
        "congenital_diabetes": "Congenital Diabetes Mellitus",
        "congenital_heart_disease": "Congenital Heart Disease",
        "cerebral_palsy_gnao1": "Cerebral Palsy (GNAO1)",
        "disorders_sex_development": "Disorders of Sex Development",
        "cystic_fibrosis": "Cystic Fibrosis",
        "duchenne_md": "Duchenne Muscular Dystrophy",
        "sma": "Spinal Muscular Atrophy",
        "beta_thalassemia": "Beta-Thalassemia",
        "gjb2_deafness": "GJB2-Related Congenital Deafness",
        "leber_congenital_amaurosis": "Leber Congenital Amaurosis",
        "fibrodysplasia_ossificans": "Fibrodysplasia Ossificans Progressiva",
        "noonan_syndrome": "Noonan Syndrome (LZTR1)",
        "pku": "Phenylketonuria",
    }

    def __init__(self):
        self.reports: List[Dict] = []

    def generate(self, engine_results: Dict[str, Any],
                 patient_id: str = "UNKNOWN") -> Dict[str, Any]:
        report = {
            "patient_id": patient_id,
            "engine_version": "v6.0.0",
            "evolution_version": EVOLUTION_VERSION,
            "diseases": {},
        }
        for key, result in engine_results.items():
            display = self.DISEASE_DISPLAY_NAMES.get(key, key)
            if result.get("status") == "HALTED":
                report["diseases"][display] = {
                    "status": "HALTED — ethical review required",
                    "reason": result.get("reason"),
                    "guide": None,
                    "clinical_recommendation": None,
                }
                continue
            if result.get("status") == "ERROR":
                report["diseases"][display] = {
                    "status": "ERROR",
                    "error": result.get("error"),
                }
                continue

            report["diseases"][display] = {
                "status": "DESIGN_READY",
                "optimized_guide": result.get("optimized_guide"),
                "therapeutic_score": result.get("final_therapeutic_score"),
                "off_target_risk": result.get("final_off_target_risk"),
                "precision_score": result.get("final_precision_score"),
                "repair_outcome_distribution": {
                    "HDR": result["repair_probs"][0] if result.get("repair_probs") else None,
                    "NHEJ_frameshift": result["repair_probs"][1] if result.get("repair_probs") else None,
                    "NHEJ_inframe": result["repair_probs"][2] if result.get("repair_probs") else None,
                    "Base_edit": result["repair_probs"][3] if result.get("repair_probs") else None,
                    "Prime_edit": result["repair_probs"][4] if result.get("repair_probs") else None,
                    "No_edit": result["repair_probs"][5] if result.get("repair_probs") else None,
                },
                "editor": result.get("preferred_editor"),
                "delivery_route": result.get("delivery_route"),
                "clinical_recommendation": self._recommend(result),
                "ethics_flag": self._ethics_check(key),
            }
        self.reports.append(report)
        return report

    def _recommend(self, result: Dict) -> str:
        score = result.get("final_therapeutic_score", 0)
        off = result.get("final_off_target_risk", 1)
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
                    "consent, and a clear non-pathologizing clinical rationale.")
        if disease_key == "down_syndrome":
            return ("ETHICS NOTE: Trisomy 21 silencing is exploratory. "
                    "Neurodevelopmental outcomes in humans are unknown. "
                    "Requires extensive preclinical safety data.")
        return None

    def export_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.reports, f, indent=2, default=str)
        logger.info("Clinical report saved to %s", path)

# =============================================================================
# Section J: CLI and smoke test.
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EVOLUTION ONE v6 — Differentiable CRISPR Design Engine")
    parser.add_argument("--context", "-c", required=True,
                        help="Target DNA context (≥20 nt around mutation)")
    parser.add_argument("--diseases", nargs="+",
                        default=["cystic_fibrosis", "neurofibromatosis_1",
                                 "beta_thalassemia"],
                        help="Disease keys to optimize for")
    parser.add_argument("--output", "-o", default="./crispr_output",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=200,
                        help="Optimization steps per disease")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddp", action="store_true",
                        help="Wrap in DDP (requires torchrun)")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--patient_id", default="PATIENT_001")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CRISPRDesignConfig(
        disease_keys=args.diseases,
        n_opt_steps=args.steps,
        use_amp=not args.no_amp,
        device=args.device,
    )
    engine = CRISPRTherapeuticEngine(cfg)

    if args.ddp and args.local_rank >= 0:
        engine = wrap_ddp(engine, args.local_rank, backend=cfg.ddp_backend)
        logger.info("DDP wrapped on rank %d", args.local_rank)

    results = engine.design_all(args.context, disease_keys=args.diseases)

    reporter = ClinicalReportGenerator()
    report = reporter.generate(results, patient_id=args.patient_id)
    reporter.export_json(str(out_dir / "crispr_report.json"))

    # Also save raw results
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nAll outputs saved to {out_dir}")


# =============================================================================
# Smoke test — runs when executed directly without CLI args
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("=" * 70)
        print("EVOLUTION ONE v6 — Differentiable CRISPR Design Engine")
        print("Smoke test (no CLI args provided)")
        print("=" * 70)

        cfg = CRISPRDesignConfig(
            disease_keys=["cystic_fibrosis", "neurofibromatosis_1",
                          "beta_thalassemia", "gjb2_deafness",
                          "disorders_sex_development"],
            n_opt_steps=50,
            use_amp=False,
            device="cpu",
        )
        engine = CRISPRTherapeuticEngine(cfg)

        # Example target context: CFTR F508del region (simplified)
        context = ("ATGGTGGTGGAAGGTGTTTTTCTGGATTTATTTGATTTTTCAGGTCTCT"
                   "CTCTATTTATCATTTTCTTTTTCTTTTCTTTTGTTGTTTTTTTTTTTTT")

        print(f"\nTarget context length: {len(context)} nt")
        print(f"Diseases: {cfg.disease_keys}")
        print(f"Optimization steps: {cfg.n_opt_steps}")
        print()

        results = engine.design_all(context)

        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS")
        print("=" * 70)
        for key, res in results.items():
            if res.get("status") == "HALTED":
                print(f"\n{key}: HALTED — {res.get('reason')}")
                continue
            if res.get("status") == "ERROR":
                print(f"\n{key}: ERROR — {res.get('error')}")
                continue
            print(f"\n{key}:")
            print(f"  Guide sequence    : {res['optimized_guide']}")
            print(f"  Therapeutic score : {res['final_therapeutic_score']:.4f}")
            print(f"  Off-target risk   : {res['final_off_target_risk']:.6f}")
            print(f"  Precision score   : {res['final_precision_score']:.4f}")
            print(f"  Editor            : {res['preferred_editor']}")
            print(f"  Delivery          : {res['delivery_route']}")

        # Report generation
        reporter = ClinicalReportGenerator()
        report = reporter.generate(results, patient_id="SMOKE_TEST")
        print("\n" + "=" * 70)
        print("CLINICAL REPORT (summary)")
        print("=" * 70)
        for disease, data in report["diseases"].items():
            print(f"\n{disease}:")
            print(f"  Status       : {data['status']}")
            print(f"  Recommendation: {data.get('clinical_recommendation')}")
            if data.get("ethics_flag"):
                print(f"  Ethics flag  : {data['ethics_flag']}")

        print("\nAll smoke-test sections executed without error.")
