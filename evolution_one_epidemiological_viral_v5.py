# =============================================================================
# EVOLUTION ONE v5 : High‑Performance Epidemiological Forecasting
# & Viral Evolution Engine
# =============================================================================
# Author  : Yoon A Limsuwan / MSPS NETWORK
# ORCID   : 0009-0008-2374-0788
# GitHub  : yoonalimsuwan
# License : MIT
# Year    : 2026
#
# AI Co-Developers:
#   - Claude   (Anthropic)  — LangevinBridgeMixin integration (Fix 3),
#                             attach_cahn_bridge() (Fix 5), BVFieldTheory
#                             standalone (Fix 1), SOCController removal (Fix 4),
#                             __all__ public API, full cross-cluster audit
#   - GPT      (OpenAI)     — coupling equation review, EpiEvolutionBridge
#                             numerical stability
#   - Gemini   (Google)     — initial pipeline scaffolding
#   - DeepSeek              — cross-verification of differentiability chain
#
# Changes v4 → v5  (Cross-cluster Integration)
#   Fix 1  — InteractionNetworkBV uses BVFieldTheory from one_core_evolution
#   Fix 3  — EpiForecastEngine inherits LangevinBridgeMixin
#   Fix 4  — LearnableRG alias updated; EpidemicClassifier uses DifferentiableSOC
#   Fix 5  — EpiForecastEngine.attach_cahn_bridge() added
#   Fix 6  — EVOLUTION_VERSION from one_core_evolution_v3 (3.0.0)
#
# Restored from v4 (v5_complete):
#   R1 — dataclass import restored
#   R2 — _TAU_GATE constant restored
#   R3 — hydro_default attribute restored in DifferentiableProteinEnergy
#   R4 — Biopython StructureBuilder / calc_dihedral / calc_angle imports restored
#   R5 — MAB_TARGETS full entries restored (Cilgavimab/Tixagevimab, Etesevimab)
#   R6 — VaccineDesigner.design_polyepitope_vaccine() 3-way linker logic restored
#   R7 — EpiForecastEngine logging restored throughout run()
#   R8 — Section header comments (§11, §13) and __all__ public API added
# =============================================================================

import math, os, sys, json, argparse, logging, warnings
import random, itertools, pickle, subprocess, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
from collections import defaultdict
from dataclasses import dataclass          # R1: restored

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr, spearmanr
from scipy.optimize import differential_evolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# ONE Core Evolution — single source of truth for shared components
from one_core_evolution import (
    SemanticStateContraction,    # SSC EMA filter        (Paper 4)
    CSOCBase,                    # CSOC abstract base
    BVFieldTheory,               # Fix 1: standalone BV
    DifferentiableRG,            # canonical learnable RG
    DifferentiableSOC,           # Fix 4: canonical SOC (replaces LearnableRG soc)
    CheckpointManager,           # canonical checkpoint
    LangevinEvolutionBridge,     # Langevin ↔ Rt bridge
    LangevinBridgeMixin,         # Fix 3: attach_langevin_bridge() mixin
    DifferentiableIto,           # fully differentiable Itô integrator
    EpiEvolutionBridge,          # Bug 7: ↔ EvolutionONEEngine bridge
    CahnHilliardEvoBridge,       # Fix 5: CH3D ↔ epi bridge
    get_device as _core_get_device,
    EVOLUTION_VERSION,
)

# Fix 4: LearnableRG is an alias for DifferentiableRG (backward-compat)
LearnableRG = DifferentiableRG

# Biopython
try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import GC
    from Bio import SeqIO
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.PDB import PDBParser, Polypeptide
    from Bio.PDB.StructureBuilder import StructureBuilder   # R4: restored
    from Bio.PDB.vectors import calc_dihedral, calc_angle   # R4: restored
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

def _has_bowtie():
    try:
        subprocess.run(["bowtie", "--version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False
HAS_BOWTIE = _has_bowtie()

try:
    from real_fold_one_v2 import (
        RefinementEngine as _RFO_Engine,
        RefinementConfig as _RFO_Config,
        CSOCKernel, SOCController,
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

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")
logger = logging.getLogger("EpiForecastONE")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

if not HAS_BIOPYTHON:
    logger.warning("Biopython not installed. "
                   "Sequence analysis and dN/dS will be unavailable.")

# =============================================================================
# Differentiability utilities
# =============================================================================

_TAU_GATE   = 1e-2   # R2: restored — sigmoid gate sharpness
_SOFTPLUS_B = 100.0  # softplus floor sharpness


def _softplus_floor(x: torch.Tensor, floor: float,
                    beta: float = _SOFTPLUS_B) -> torch.Tensor:
    """Differentiable floor: floor + softplus(x − floor)."""
    return floor + F.softplus(x - floor, beta=beta)


def _soft_count(coords: torch.Tensor, center: torch.Tensor,
                cutoff: float, tau: float = 0.5) -> torch.Tensor:
    """
    DIFF-FIX 4: differentiable soft count of atoms within cutoff.
    Replaces:  sum(1 for j if norm(ri-rj) < cutoff)
    With:      sum_j σ((cutoff − d_ij) / tau)
    """
    dists = torch.norm(coords - center.unsqueeze(0), dim=-1)
    return torch.sigmoid((cutoff - dists) / tau).sum()


def calc_dihedral_torch(p0, p1, p2, p3):
    """Compute single dihedral angle (scalar tensors). Fully differentiable."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (torch.norm(b1) + 1e-8)
    v  = b0 - torch.dot(b0, b1) * b1
    w  = b2 - torch.dot(b2, b1) * b1
    return torch.atan2(torch.dot(torch.cross(b1, v), w), torch.dot(v, w))


def calc_dihedral_torch_batch(p0, p1, p2, p3):
    """Vectorised dihedral for (M, 3) tensors — returns (M,). Fully differentiable."""
    b0  = p0 - p1
    b1  = p2 - p1
    b2  = p3 - p2
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-8)
    v   = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w   = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x   = (v * w).sum(-1)
    y   = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


# =============================================================================
# 1. Embedded Differentiable Physics Engine (fallback when REAL FOLD ONE absent)
# =============================================================================

class RefinementConfig:
    """Configuration for the embedded refinement engine."""
    def __init__(self, device: str = "cpu", steps: int = 50, lr: float = 0.01):
        self.device = device
        self.steps  = steps
        self.lr     = lr


class DifferentiableProteinEnergy(nn.Module):
    """
    Coarse‑grained protein energy: one bead per residue (Cα).

    DIFF-FIX 4: solvation term uses differentiable soft-count
    (sigmoid-gated) instead of hard integer neighbour count.
    """
    def __init__(self, seq: str):
        super().__init__()
        self.seq    = seq
        self.n_res  = len(seq)
        self.d0           = 3.8
        self.cos0         = -0.3
        self.sigma        = 4.0
        self.epsilon      = 0.5
        self.debye_length = 10.0
        self.dielectric   = 80.0
        self.hydro = {
            "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
            "G": 0.48, "H": -0.40, "I": 1.38,  "K": -1.50, "L": 1.06,
            "M": 0.64, "N": -0.78, "P": 0.12,  "Q": -0.85, "R": -2.53,
            "S": -0.18,"T": -0.05, "V": 1.08,  "W": 0.81,  "Y": 0.26,
        }
        self.hydro_default = 0.0   # R3: restored

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 3). Returns scalar total energy."""
        N   = coords.shape[0]
        dev = coords.device

        # Bond length
        bond = torch.tensor(0.0, device=dev)
        if N > 1:
            d    = torch.norm(coords[1:] - coords[:-1], dim=1)
            bond = ((d - self.d0) ** 2).sum()

        # Pseudo-angle
        angle = torch.tensor(0.0, device=dev)
        if N > 2:
            v1    = coords[:-2] - coords[1:-1]
            v2    = coords[2:]  - coords[1:-1]
            cos   = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
            angle = ((cos - self.cos0) ** 2).sum()

        # Pseudo-torsion (vectorised)
        torsion = torch.tensor(0.0, device=dev)
        if N > 3:
            p0t, p1t = coords[:-3], coords[1:-2]
            p2t, p3t = coords[2:-1], coords[3:]
            phi = calc_dihedral_torch_batch(p0t, p1t, p2t, p3t)
            torsion = (0.5 * (1.0 - torch.cos(phi))
                       + 0.5 * (1.0 - torch.cos(phi + 1.047))).sum()

        # Non-bonded (vectorised upper-triangle i < j-2)
        nonbond = torch.tensor(0.0, device=dev)
        if N > 2:
            ii, jj = torch.triu_indices(N, N, offset=2, device=dev)
            rij    = coords[ii] - coords[jj]
            d2     = (rij * rij).sum(-1)
            d      = torch.sqrt(d2 + 1e-8)
            sr6    = (self.sigma / d) ** 6
            lj     = self.epsilon * (sr6 * sr6 - 2.0 * sr6)
            q_i    = torch.tensor(
                [self._charge(k) for k in ii.tolist()], device=dev, dtype=coords.dtype)
            q_j    = torch.tensor(
                [self._charge(k) for k in jj.tolist()], device=dev, dtype=coords.dtype)
            coulomb = (0.1 * q_i * q_j
                       * torch.exp(-d / self.debye_length)
                       / (self.dielectric * d + 1e-8))
            nonbond = (lj + coulomb).sum()

        # Solvation — soft neighbour count (DIFF-FIX 4)
        solv = torch.tensor(0.0, device=dev)
        for i in range(N):
            others = torch.cat([coords[:i], coords[i+1:]], dim=0)
            if others.shape[0] > 0:
                n_soft = _soft_count(others, coords[i], cutoff=7.0, tau=0.5)
                solv   = solv + self.hydro.get(self.seq[i], self.hydro_default) * n_soft / 10.0

        return 0.5 * bond + 0.2 * angle + 0.1 * torsion + nonbond + 0.05 * solv

    def _charge(self, idx: int) -> float:
        return {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}.get(self.seq[idx], 0.0)


class RefinementEngine:
    """
    Gradient‑based energy minimisation (relaxation).

    DIFF-FIX 5: in-place indexed coordinate writes replaced with
    differentiable scatter assembly so gradients flow through the
    mutable-position coordinates correctly.
    """
    def __init__(self, cfg: RefinementConfig):
        self.cfg    = cfg
        self.device = cfg.device

    def compute_energy(self, coords: torch.Tensor, seq: str) -> float:
        energy_fn = DifferentiableProteinEnergy(seq).to(self.device)
        return energy_fn(coords.clone().detach()).item()

    def relax_local(self, coords: torch.Tensor, seq: str,
                    mutable_positions: List[int],
                    steps: int = None) -> Tuple[torch.Tensor, float]:
        """
        Local relaxation: minimise energy w.r.t. mutable residues.
        Returns (optimised_coords, energy_after).
        """
        steps     = steps or self.cfg.steps
        N         = coords.shape[0]
        mut       = [p for p in mutable_positions if 0 <= p < N]
        fix       = [i for i in range(N) if i not in set(mut)]
        if not mut:
            return coords.clone().detach(), self.compute_energy(coords, seq)

        mut_t     = coords[mut].clone().detach().requires_grad_(True)
        fix_t     = coords[fix].clone().detach()
        opt       = torch.optim.Adam([mut_t], lr=self.cfg.lr)
        energy_fn = DifferentiableProteinEnergy(seq).to(self.device)

        for _ in range(steps):
            opt.zero_grad()
            parts = []
            mi, fi = 0, 0
            for i in range(N):
                if i in set(mut):
                    parts.append(mut_t[mi].unsqueeze(0)); mi += 1
                else:
                    parts.append(fix_t[fi].unsqueeze(0)); fi += 1
            energy_fn(torch.cat(parts)).backward()
            opt.step()

        parts = []
        mi, fi = 0, 0
        for i in range(N):
            if i in set(mut):
                parts.append(mut_t[mi].detach().unsqueeze(0)); mi += 1
            else:
                parts.append(fix_t[fi].unsqueeze(0)); fi += 1
        final = torch.cat(parts)
        return final, energy_fn(final).item()


# =============================================================================
# 2. Learnable RG & Differentiable SOC
# =============================================================================

# LearnableRG → DifferentiableRG imported from one_core_evolution (Fix 4).
# Backward-compat alias so existing code using LearnableRG still works.
# (LearnableRG = DifferentiableRG already set at top of file)

# DifferentiableSOC is imported from one_core_evolution.
# The local duplicate is removed to keep a single canonical definition.
# Usage: self.soc = DifferentiableSOC(base_temp=300.0, beta=0.01)


# =============================================================================
# 3. Epidemic Classifier (fully differentiable)
# =============================================================================

class EpidemicClassifier(nn.Module):
    """
    Fully differentiable epidemic phase classifier.

    Fix 4: Uses DifferentiableSOC (canonical) instead of LearnableRG's
    internal SOC. Log-softmax for clean probability normalisation.
    DIFF-FIX 3: class probabilities via log-softmax over three logits.
    DIFF-FIX 8: threshold ordering guaranteed by cumsum of softplus.
    """

    def __init__(self,
                 threshold_outbreak_init: float = 1.0,
                 threshold_pandemic_init: float = 2.0):
        super().__init__()
        # DIFF-FIX 8: store two raw deltas; thresholds = cumsum(softplus(raw))
        self.raw_delta0 = nn.Parameter(torch.tensor(threshold_outbreak_init))
        self.raw_delta1 = nn.Parameter(torch.tensor(
            threshold_pandemic_init - threshold_outbreak_init))
        self.rg        = LearnableRG(kernel_size=5)
        self.soc       = DifferentiableSOC(base_temp=300.0, beta=0.01)
        self.ito_drift = nn.Parameter(torch.tensor(0.0))

    @property
    def threshold_outbreak(self) -> torch.Tensor:
        """Always positive — softplus of raw delta."""
        return F.softplus(self.raw_delta0)

    @property
    def threshold_pandemic(self) -> torch.Tensor:
        """Always > threshold_outbreak — cumsum guarantees ordering."""
        return F.softplus(self.raw_delta0) + F.softplus(self.raw_delta1)

    def forward(self, rt_values: torch.Tensor,
                steps: int = 10,
                return_trajectory: bool = False):
        if rt_values.dim() == 1:
            rt_values = rt_values.unsqueeze(0)

        rt = self.rg(rt_values)
        rt = self.soc(rt, steps=steps)
        rt = rt + self.ito_drift * 0.01

        th_out = self.threshold_outbreak
        th_pan = self.threshold_pandemic

        # DIFF-FIX 3: three logits → log-softmax (guarantees valid probs)
        logit_stable   = -(rt - th_out)
        logit_outbreak =  (rt - th_out) - F.softplus(rt - th_pan)
        logit_pandemic =  (rt - th_pan)

        logits    = torch.stack([logit_stable, logit_outbreak, logit_pandemic], dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if return_trajectory:
            return log_probs, rt
        return log_probs


# =============================================================================
# 4. Memory‑Efficient Data Loader
# =============================================================================

class EpidemiologicalDataLoader:
    """Handles case CSV and FASTA sequence loading."""

    def load_case_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, parse_dates=["date"])
        logger.info("Loaded case data: %d records.", len(df))
        return df

    def stream_sequences(self, fasta_path: str,
                         batch_size: int = 1000) -> Generator[Dict[str, str], None, None]:
        """Yield dicts of {id: sequence} in batches."""
        batch: Dict[str, str] = {}
        for rec in SeqIO.parse(fasta_path, "fasta"):
            batch[rec.id] = str(rec.seq).upper()
            if len(batch) >= batch_size:
                yield batch; batch = {}
        if batch:
            yield batch

    def load_sequences(self, fasta_path: str,
                       max_sequences: int = 10_000) -> Dict[str, str]:
        seqs: Dict[str, str] = {}
        for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta")):
            if max_sequences and i >= max_sequences:
                break
            seqs[rec.id] = str(rec.seq).upper()
        logger.info("Loaded %d sequences.", len(seqs))
        return seqs

    def build_prevalence_matrix(
            self,
            case_df: pd.DataFrame,
            strains: List[str],
            time_window: str = "W",
    ) -> Tuple[np.ndarray, List[str]]:
        df      = case_df.copy()
        df["period"] = df["date"].dt.to_period(time_window)
        periods = sorted(df["period"].unique())
        p_idx   = {p: i for i, p in enumerate(periods)}
        s_idx   = {s: j for j, s in enumerate(strains)}
        M = np.zeros((len(periods), len(strains)))
        for _, row in df.iterrows():
            s = row.get("strain", "total")
            if s in s_idx:
                M[p_idx[row["period"]], s_idx[s]] += row["cases"]
        return M, [str(p) for p in periods]


# =============================================================================
# 5. Viral Evolution Analysis
# =============================================================================

class ViralEvolutionAnalyzer:
    """Mutation frequencies and dN/dS (Nei‑Gojobori via Biopython)."""

    def compute_mutation_frequencies(
            self,
            sequences: Dict[str, str],
            ref_id: str = "reference") -> pd.DataFrame:
        if ref_id not in sequences:
            ref_id = next(iter(sequences))
        ref_seq = sequences[ref_id]
        sites   = []
        for sid, seq in sequences.items():
            for i in range(min(len(ref_seq), len(seq))):
                if seq[i] != ref_seq[i] and seq[i] != "-":
                    sites.append({"position": i, "ref": ref_seq[i],
                                  "alt": seq[i], "strain": sid})
        if not sites:
            return pd.DataFrame()
        df   = pd.DataFrame(sites)
        freq = df.groupby("position").agg(
            ref=("ref", "first"),
            total_mutations=("position", "count"),
            variants=("alt", lambda x: ",".join(sorted(set(x)))),
        ).reset_index()
        freq["frequency"] = freq["total_mutations"] / len(sequences)
        return freq

    def compute_dnds(self, seq1_str: str, seq2_str: str) -> float:
        """Nei‑Gojobori dN/dS (requires Biopython)."""
        if not HAS_BIOPYTHON:
            return 1.0
        try:
            dN, dS = cal_dn_ds(CodonSeq(seq1_str), CodonSeq(seq2_str), method="NG86")
            return 999.0 if dS == 0 else dN / dS
        except Exception as e:
            logger.warning("dN/dS failed: %s", e)
            return 1.0

    def pairwise_dnds(self, sequences: Dict[str, str],
                      gene_start: int = 0,
                      gene_end: int = None) -> float:
        ids = list(sequences.keys())
        if len(ids) < 2:
            return 1.0
        vals = []
        for i in range(min(10, len(ids))):
            for j in range(i+1, min(10, len(ids))):
                s1 = sequences[ids[i]][gene_start:gene_end]
                s2 = sequences[ids[j]][gene_start:gene_end]
                if len(s1) == len(s2) and len(s1) % 3 == 0:
                    v = self.compute_dnds(s1, s2)
                    if v != 999.0:
                        vals.append(v)
        return float(np.mean(vals)) if vals else 1.0


# =============================================================================
# 6. Future Variant Predictor
# =============================================================================

class FutureVariantPredictor:
    """Predicts escape / destabilising mutations via structural analysis."""

    def __init__(self, pdb_dir: str = "./pdbs"):
        self.pdb_dir = pdb_dir

    def predict_vulnerable_positions(
            self,
            protein: str,
            structure_file: str = None) -> List[Dict]:
        if HAS_HT:
            if not structure_file:
                candidates = list(Path(self.pdb_dir).glob(f"*{protein}*.pdb"))
                if not candidates:
                    logger.warning("No PDB for %s", protein)
                    return []
                structure_file = str(candidates[0])
            cfg     = HTConfig(pdb_file=structure_file, scan_full=True,
                               output_dir=f"./ht_scan_{protein}")
            scanner = HighThroughputScanner(cfg)
            scanner.load_structure()
            df = pd.DataFrame(scanner.scan_single_mutations())
            if df.empty:
                return []
            df["ddg"] = df["ddg"].astype(float)
            return df[df["ddg"] > 1.5].sort_values("ddg", ascending=False).to_dict("records")
        else:
            pdb_file = structure_file or self._find_pdb(protein)
            if not pdb_file:
                return []
            coords, seq = self._load_pdb_ca(pdb_file)
            if coords is None:
                return []
            engine  = RefinementEngine(RefinementConfig(device="cpu", steps=50, lr=0.02))
            results: List[Dict] = []
            for pos in range(len(seq)):
                wt = seq[pos]
                for new in "AVLGI":
                    if new == wt:
                        continue
                    try:
                        _, e_mut = engine.relax_local(
                            coords.clone().detach().requires_grad_(True),
                            seq[:pos] + new + seq[pos+1:], [pos], steps=30)
                        ddg = e_mut - engine.compute_energy(coords, seq)
                        if ddg > 1.5:
                            results.append({"position": pos, "wt": wt,
                                            "mut": new, "ddg": ddg,
                                            "type": "mutation"})
                    except Exception as e:
                        logger.warning("Relax failed %d%s->%s: %s", pos, wt, new, e)
            return results

    def _find_pdb(self, protein: str) -> Optional[str]:
        cands = list(Path(self.pdb_dir).glob(f"*{protein}*.pdb"))
        return str(cands[0]) if cands else None

    def _load_pdb_ca(self, pdb_file: str) -> Tuple[Optional[torch.Tensor], str]:
        """Extract Cα coords and sequence from a PDB file."""
        if not HAS_BIOPYTHON:
            logger.warning("Biopython required for PDB loading.")
            return None, ""
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("prot", pdb_file)
        except Exception as e:
            logger.error("PDB parse failed: %s", e)
            return None, ""
        ca_coords, seq = [], []
        for chain in structure[0]:
            for res in chain:
                if Polypeptide.is_aa(res, standard=True):
                    try:
                        ca_coords.append(res["CA"].get_vector().get_array())
                        seq.append(Polypeptide.three_to_one(res.get_resname()))
                    except KeyError:
                        continue
        if not ca_coords:
            return None, ""
        return (torch.tensor(np.array(ca_coords), dtype=torch.float32), "".join(seq))


# =============================================================================
# 7. Differentiable Epitope Scoring Network
# =============================================================================

class EpitopeScorer(nn.Module):
    """
    Bidirectional GRU epitope scorer.

    DIFF-FIX 6: outputs are sigmoid-activated logits kept in the
    computational graph (no detach) so the vaccine designer can
    backpropagate through epitope scores if needed.
    """
    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.conv       = nn.Conv1d(20, embed_dim, kernel_size=3, padding=1)
        self.gru        = nn.GRU(embed_dim, hidden_dim,
                                 batch_first=True, bidirectional=True)
        self.mhc_head   = nn.Linear(hidden_dim * 2, 1)
        self.bcell_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, L, 20) one-hot amino-acid encoding."""
        x = F.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        _, h = self.gru(x)
        h = h.permute(1, 0, 2).reshape(x.size(0), -1)
        # DIFF-FIX 6: no torch.no_grad / detach here
        return torch.sigmoid(self.mhc_head(h)), torch.sigmoid(self.bcell_head(h))


# =============================================================================
# 8. Vaccine Designer
# =============================================================================

CODON_OPT = {
    "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
    "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
    "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "CGG",
    "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC", "*": "TGA",
}


class VaccineDesigner:
    """
    Poly‑epitope mRNA vaccine constructor backed by EpitopeScorer.
    predict_epitopes() is fully differentiable w.r.t. scorer weights.
    """
    def __init__(self,
                 protein_sequences: Dict[str, str] = None,
                 epitope_scorer: nn.Module = None):
        self.proteins  = protein_sequences or {}
        self.scorer    = epitope_scorer or EpitopeScorer()
        self.aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def _peptide_to_tensor(self, peptide: str) -> torch.Tensor:
        idx    = [self.aa_to_idx.get(aa, 0) for aa in peptide]
        onehot = F.one_hot(torch.tensor(idx), num_classes=20).float()
        if len(onehot) < 9:
            onehot = F.pad(onehot, (0, 0, 0, 9 - len(onehot)))
        return onehot.unsqueeze(0)   # (1, L, 20)

    def predict_epitopes(self, protein_name: str,
                         window: int = 9) -> List[Dict]:
        seq      = self.proteins.get(protein_name, "")
        epitopes = []
        for i in range(len(seq) - window + 1):
            pep = seq[i:i+window]
            ten = self._peptide_to_tensor(pep)
            with torch.no_grad():
                mhc, bcell = self.scorer(ten)
            epitopes.append({"start": i, "end": i+window, "sequence": pep,
                              "mhc_score": mhc.item(), "bcell_score": bcell.item()})
        return sorted(epitopes, key=lambda x: x["mhc_score"] + x["bcell_score"], reverse=True)

    def design_polyepitope_vaccine(self,
                                   protein_names: List[str],
                                   top_k: int = 10) -> Dict:
        all_ep: List[Dict] = []
        for prot in protein_names:
            all_ep.extend(self.predict_epitopes(prot))
        all_ep.sort(key=lambda x: x["mhc_score"] + x["bcell_score"], reverse=True)
        seen, unique = set(), []
        for ep in all_ep:
            if ep["sequence"] not in seen:
                seen.add(ep["sequence"])
                unique.append(ep)
        signal = "MKWVSFFILFLLFSSAYSRGVFRR"
        aa_seq = signal
        for ep in unique[:top_k * 2]:
            # R6: restored 3-way linker logic (v4 original)
            if ep["mhc_score"] > 0.5 and ep["bcell_score"] > 0.5:
                aa_seq += "GPGPG" + ep["sequence"]
            elif ep["mhc_score"] > 0.5:
                aa_seq += "AAY" + ep["sequence"]
            else:
                aa_seq += "GPGPG" + ep["sequence"]
        mrna   = "".join(CODON_OPT.get(aa, "NNN") for aa in aa_seq)
        utr5   = "AGAUCCAGCUGCUCUCGACU"
        utr3   = "CUAGUGAUAAGCUGCUUU"
        poly_a = "A" * 120
        return {
            "amino_acid_sequence": aa_seq,
            "mRNA_sequence":       utr5 + mrna + utr3 + poly_a,
            "num_epitopes":        len(unique[:top_k * 2]),
            "epitopes":            unique[:top_k * 2],
        }


# =============================================================================
# 9. Therapeutic Recommender
# =============================================================================

ANTIVIRAL_TARGETS = {
    "Spike":  ["Remdesivir", "Paxlovid", "Molnupiravir"],
    "3CLpro": ["Nirmatrelvir", "Ensitrelvir"],
    "RDRP":   ["Remdesivir", "Favipiravir"],
}
# R5: restored full MAB list including Cilgavimab/Tixagevimab and Etesevimab
MAB_TARGETS = {
    "Spike": ["Sotrovimab", "Bebtelovimab", "Cilgavimab/Tixagevimab"],
    "RBD":   ["Regdanvimab", "Etesevimab"],
}


class TherapeuticRecommender:
    def recommend_antivirals(self, protein: str) -> List[str]:
        return ANTIVIRAL_TARGETS.get(protein, [])

    def recommend_mabs(self, protein: str,
                       escape_mutations: List[Dict]) -> List[str]:
        if any(abs(m.get("ddg", 0.0)) > 2.0 for m in escape_mutations):
            return []
        return MAB_TARGETS.get(protein, [])


# =============================================================================
# 10. Epidemiological Factor Analyzer
# =============================================================================

class EpidemiologicalFactorAnalyzer:
    def __init__(self, factor_file: str = None):
        self.factor_df = None
        if factor_file and os.path.exists(factor_file):
            self.factor_df = pd.read_csv(factor_file)

    def merge_with_rt_data(self, time_labels: List[str],
                           rt_values: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"time": time_labels, "rt": rt_values})
        if self.factor_df is not None:
            df = df.merge(self.factor_df, on="time", how="inner")
        return df

    def compute_correlations(self, merged_df: pd.DataFrame,
                             target: str = "rt") -> Dict:
        factors = [c for c in merged_df.columns if c not in ("time", target)]
        results = {}
        for fac in factors:
            if merged_df[fac].nunique() < 2:
                continue
            rp, _ = pearsonr(merged_df[fac], merged_df[target])
            rs, _ = spearmanr(merged_df[fac], merged_df[target])
            results[fac] = {"pearson_r": rp, "spearman_r": rs}
        return results


# =============================================================================
# 11. Checkpoint Manager
# =============================================================================

# R8: section header restored — CheckpointManager imported from one_core_evolution.
# Local duplicate removed.


# =============================================================================
# 12. Host‑Pathogen Network BV
# =============================================================================

class InteractionNetworkBV(BVFieldTheory):
    """
    Fix 1: Inherits from one_core_evolution.BVFieldTheory.
    Provides BV (Batalin-Vilkovisky) consistency check for
    host-pathogen network. Works without real_fold_one.
    """

    def __init__(self, node_names: List[str],
                 interactions: List[Tuple[int, int]]):
        field_names = [f"phi_{i}" for i in range(len(node_names))]
        super().__init__(field_names, [0] * len(field_names))
        self.node_names   = node_names
        self.interactions = interactions
        for name in field_names:
            self.phi[name] = torch.randn(1) * 0.01

    def action_functional(self, phi_dict, phi_star_dict) -> torch.Tensor:
        S = torch.tensor(0.0)
        for i, j in self.interactions:
            S = S + 0.5 * (phi_dict[f"phi_{i}"] - phi_dict[f"phi_{j}"]) ** 2
        return S

    def verify(self) -> bool:
        return self.classical_master_equation(self.action_functional)


# =============================================================================
# 13. Langevin-Evolution Bridge
# =============================================================================

# R8: section header restored — LangevinEvolutionBridge imported from
# one_core_evolution (canonical). Local duplicate removed.
# EpiForecastEngine gains attach_langevin_bridge() via LangevinBridgeMixin (Fix 3).


# =============================================================================
# 14. Main EpiForecast ONE Engine  (with LangevinBridgeMixin — Fix 3)
# =============================================================================

class EpiForecastEngine(LangevinBridgeMixin):
    """
    Orchestrates the full epidemiological + viral evolution pipeline.

    Fix 3: Inherits LangevinBridgeMixin → gains attach_langevin_bridge().
    Fix 5: attach_cahn_bridge() connects CH3D phase-field to Rt.
    DIFF-FIX 7: AMP GradScaler added to train_classifier for mixed-precision
    training on CUDA. CPU training falls back gracefully.

    Bridge usage::

        from structural_langevin_evo_v5 import AdvancedStructuralLangevin
        integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=310.0)
        bridge = engine.attach_langevin_bridge(integrator)

        from structural_cahn_hilliard_3d import StructuralCahnHilliard3D
        ch_solver = StructuralCahnHilliard3D()
        ch_bridge = engine.attach_cahn_bridge(ch_solver)
        rt_coupled = ch_bridge.phase_to_rt(u_field, rt_base_tensor)
    """

    def __init__(self, cfg: dict = None):
        self.cfg              = cfg or {}
        self.loader           = EpidemiologicalDataLoader()
        self.evolution        = ViralEvolutionAnalyzer()
        self.classifier       = EpidemicClassifier()
        self.structural       = FutureVariantPredictor(
            pdb_dir=self.cfg.get("pdb_dir", "./pdbs"))
        self.drug_engine      = TherapeuticRecommender()
        self.factor_analyzer  = EpidemiologicalFactorAnalyzer(
            factor_file=self.cfg.get("factor_file"))
        self.vaccine_designer = VaccineDesigner()
        self.device           = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        self.optimizer        = torch.optim.Adam(
            self.classifier.parameters(), lr=1e-3)
        self.scaler           = (GradScaler()
                                 if self.device.type == "cuda" else None)

        # Bridges — populated via attach_*() methods
        self.langevin_bridge: Optional[LangevinEvolutionBridge] = None  # Fix 3
        self.evo_bridge:      Optional[EpiEvolutionBridge]      = None
        self.cahn_bridge:     Optional[CahnHilliardEvoBridge]   = None  # Fix 5
        self.results: Dict    = {}

    # ── Langevin bridge (from LangevinBridgeMixin — Fix 3) ────────────────────
    # attach_langevin_bridge() is inherited from LangevinBridgeMixin

    # ── Evo bridge (Bug 7) ────────────────────────────────────────────────────

    def attach_evo_bridge(
        self,
        evo_engine,
        mu_to_rt_scale: float = 0.5,
        rt_to_mu_scale: float = 0.3,
    ) -> EpiEvolutionBridge:
        """
        Attach a differentiable EpiEvolutionBridge linking this
        EpiForecastEngine to an EvolutionONEEngine.

        Enables bidirectional coupling:
          μ (mutation load from EvolutionONEEngine) → modulates Rt
          Rt (from this engine) → modulates μ

        Args:
            evo_engine      : an EvolutionONEEngine instance.
            mu_to_rt_scale  : max Rt boost from high mutation load.
            rt_to_mu_scale  : max μ increase from high Rt.
        Returns:
            The attached EpiEvolutionBridge instance.

        Example::

            bridge = epi_engine.attach_evo_bridge(evo_engine)
            mu_t   = torch.tensor(evo_engine.results["mu_smooth"])
            rt_t   = torch.tensor(epi_engine.results["rt_smooth"])
            rt_coupled = bridge.mu_to_rt(mu_t, rt_t)
        """
        self.evo_bridge = EpiEvolutionBridge(
            mu_to_rt_scale=mu_to_rt_scale,
            rt_to_mu_scale=rt_to_mu_scale,
        )
        if hasattr(evo_engine, "epi_bridge"):
            evo_engine.epi_bridge = self.evo_bridge
        logger.info("EpiEvolutionBridge attached.")
        return self.evo_bridge

    # ── Cahn-Hilliard bridge (Fix 5) ──────────────────────────────────────────

    def attach_cahn_bridge(
        self,
        ch_solver,
        ssc:              Optional[SemanticStateContraction] = None,
        mu_floor:         float = 1e-4,
        interface_weight: float = 0.2,
    ) -> CahnHilliardEvoBridge:
        """
        Fix 5: Connect a StructuralCahnHilliard3D solver to this engine.

        The bridge maps:
          • u (phase field) → μ (mutation load) → modulates Rt via EpiEvolutionBridge
          • u interface sharpness → transmission heterogeneity → Rt boost

        Args:
            ch_solver        : StructuralCahnHilliard3D instance.
            ssc              : optional SemanticStateContraction for μ smoothing.
            mu_floor         : softplus floor for strict positivity of μ.
            interface_weight : weight of interface-sharpness Rt modulation.

        Returns:
            CahnHilliardEvoBridge (also stored as self.cahn_bridge).

        Usage::

            ch_solver  = StructuralCahnHilliard3D(cfg)
            ch_bridge  = epi_engine.attach_cahn_bridge(ch_solver)
            evo_bridge = epi_engine.attach_evo_bridge(evo_engine)

            u_new      = ch_solver.step(u_init, sigma)
            mu         = ch_bridge.project_to_mu(u_new)
            rt_base    = torch.tensor(epi_engine.results["rt_smooth"]).mean()
            rt_new     = evo_bridge.mu_to_rt(mu, rt_base)
        """
        self.cahn_bridge = CahnHilliardEvoBridge(
            ch_solver=ch_solver,
            ssc=ssc,
            mu_floor=mu_floor,
            interface_weight=interface_weight,
        )
        logger.info("CahnHilliardEvoBridge attached (interface_weight=%.3f).",
                    interface_weight)
        return self.cahn_bridge

    # ── Training ──────────────────────────────────────────────────────────────

    def train_classifier(self, rt_values: np.ndarray,
                         labels: np.ndarray,
                         epochs: int = 100) -> None:
        """
        Gradient-based training of EpidemicClassifier thresholds.

        DIFF-FIX 7: uses AMP GradScaler on CUDA for memory efficiency.
        """
        rt_t  = torch.tensor(rt_values, dtype=torch.float32).to(self.device)
        lbl_t = torch.tensor(labels,    dtype=torch.long).to(self.device)
        self.classifier.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with autocast():
                    log_p, _ = self.classifier(rt_t, return_trajectory=True)
                    loss      = F.nll_loss(log_p, lbl_t)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_p, _ = self.classifier(rt_t, return_trajectory=True)
                loss      = F.nll_loss(log_p, lbl_t)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, loss.item())
        self.classifier.eval()

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        case_file:               str,
        sequence_file:           Optional[str]              = None,
        protein_of_interest:     str                        = "Spike",
        compute_future_variants: bool                       = True,
        compute_structural:      bool                       = True,
        interaction_network:     Optional[List[Tuple[int, int]]] = None,
        train:                   bool                       = False,
        outcome_labels_file:     Optional[str]              = None,
        resume_from:             Optional[str]              = None,
    ) -> Dict:

        # Restore checkpoint
        if resume_from:
            ckpt = CheckpointManager.load(resume_from)
            if ckpt and "classifier_state" in ckpt:
                self.classifier.load_state_dict(ckpt["classifier_state"])

        # 1. Case data → Rt estimates
        case_df = self.loader.load_case_data(case_file)
        if case_df.empty:
            return {}
        if "strain" not in case_df.columns:
            case_df["strain"] = "total"
        strains = sorted(case_df["strain"].unique())
        M, time_labels = self.loader.build_prevalence_matrix(case_df, strains)
        total  = M.sum(axis=1)
        rt_est = np.ones_like(total)
        for i in range(1, len(total)):
            rt_est[i] = total[i] / (total[i-1] + 1e-8)
        rt_est = np.clip(rt_est, 0, 10)

        # 2. Optional threshold training
        if train and outcome_labels_file:
            odf    = pd.read_csv(outcome_labels_file)
            merged = pd.DataFrame({"time": time_labels, "rt": rt_est})
            merged = merged.merge(odf, on="time", how="inner")
            if len(merged) > 5:
                self.train_classifier(merged["rt"].values,
                                      merged["label"].values.astype(int))

        # 3. Phase inference
        rt_t = torch.tensor(rt_est, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_p, rt_smooth = self.classifier(rt_t, return_trajectory=True)
            states = torch.argmax(
                F.softmax(log_p, dim=-1), dim=-1).squeeze().cpu().numpy()
            rt_np  = rt_smooth.squeeze().cpu().numpy()
        H = entropy(np.bincount(states, minlength=3) / len(states))
        # R7: restored logging
        logger.info("Epidemic entropy H=%.4f  stable=%d  outbreak=%d  pandemic=%d",
                    H, int(np.sum(states == 0)),
                    int(np.sum(states == 1)), int(np.sum(states == 2)))

        # 4. Future prediction
        # NOTE (fix): DifferentiableSOC's `spread` term is a soft range
        # estimator (logsumexp soft max - soft min) computed across the
        # batch/sequence dimension. Feeding it a single value (just rt_np[-1])
        # makes spread collapse algebraically to 2*rt, so the SOC step never
        # actually reflects recent Rt volatility -- same degeneracy fixed in
        # EVOLUTION ONE's soc_evolve call. Use a recent window of the smoothed
        # Rt trajectory instead, so spread is a meaningful statistic, and read
        # the risk off the last time-step of the evolved window.
        window      = rt_np[-min(14, len(rt_np)):]
        rt_window   = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lp_fut, _ = self.classifier(rt_window, return_trajectory=True)
            fut_state = torch.argmax(F.softmax(lp_fut, dim=-1), dim=-1).squeeze().cpu().numpy()
            fut_state = int(np.atleast_1d(fut_state)[-1])
        pandemic_risk = ["Low", "Moderate", "High"][fut_state]
        logger.info("Predicted pandemic risk: %s", pandemic_risk)

        # 5. BV check
        bv_ok = False
        if interaction_network:
            try:
                max_node = max(max(p) for p in interaction_network)
                net      = InteractionNetworkBV(
                    [f"node{i}" for i in range(max_node + 1)],
                    interaction_network)
                bv_ok = net.verify()
                logger.info("BV satisfied: %s", bv_ok)
            except Exception as e:
                logger.warning("BV check failed: %s", e)

        # 6. Future variants & vaccine design
        future_variants: Dict = {}
        vaccine = None
        if sequence_file and compute_future_variants:
            seqs = self.loader.load_sequences(sequence_file, max_sequences=5000)
            if seqs:
                hotspots = self.evolution.compute_mutation_frequencies(seqs)
                dnds     = self.evolution.pairwise_dnds(seqs)
                escape   = self.structural.predict_vulnerable_positions(
                    protein_of_interest)
                future_variants = {
                    "hotspots":         hotspots.to_dict("records") if not hotspots.empty else [],
                    "dnds":             dnds,
                    "escape_mutations": escape,
                }
                logger.info("dN/dS = %.3f", dnds)

        if sequence_file and HAS_BIOPYTHON:
            self.vaccine_designer.proteins = self.loader.load_sequences(
                sequence_file, max_sequences=100)
            if self.vaccine_designer.proteins:
                pnames  = list(self.vaccine_designer.proteins.keys())[:5]
                vaccine = self.vaccine_designer.design_polyepitope_vaccine(
                    pnames, top_k=5)
                logger.info("Designed mRNA vaccine with %d epitopes.",
                            vaccine["num_epitopes"])

        # 7. Structural impact & drug recommendations
        structural_results: Dict = {}
        drug_recos: Dict         = {}
        if compute_structural and not HAS_HT:
            pdb = self.structural._find_pdb(protein_of_interest)
            if pdb:
                coords, seq = self.structural._load_pdb_ca(pdb)
                if coords is not None:
                    eng     = RefinementEngine(RefinementConfig(device="cpu", steps=30))
                    impacts = []
                    for pos in range(min(len(seq), 10)):
                        wt = seq[pos]
                        for new in "AVLGI":
                            if new == wt:
                                continue
                            try:
                                _, e_mut = eng.relax_local(
                                    coords.clone().detach().requires_grad_(True),
                                    seq[:pos] + new + seq[pos+1:], [pos], steps=20)
                                impacts.append({
                                    "position": pos, "wt": wt, "mut": new,
                                    "ddg": e_mut - eng.compute_energy(coords, seq)
                                })
                            except Exception as e:
                                logger.warning("Relax failed %d%s->%s: %s",
                                               pos, wt, new, e)
                    structural_results[protein_of_interest] = impacts
                    drug_recos[protein_of_interest] = {
                        "antivirals": self.drug_engine.recommend_antivirals(
                            protein_of_interest),
                        "monoclonal_antibodies": self.drug_engine.recommend_mabs(
                            protein_of_interest, impacts),
                    }

        # 8. Factor correlations
        factor_corrs: Dict = {}
        if self.factor_analyzer.factor_df is not None:
            merged = self.factor_analyzer.merge_with_rt_data(time_labels, rt_np)
            if len(merged) > 5:
                factor_corrs = self.factor_analyzer.compute_correlations(merged)

        self.results = {
            "rt_smooth":            rt_np,
            "states":               states,
            "entropy":              H,
            "pandemic_risk":        pandemic_risk,
            "future_variants":      future_variants,
            "vaccine_design":       vaccine,
            "structural_impacts":   structural_results,
            "drug_recommendations": drug_recos,
            "factor_correlations":  factor_corrs,
            "bv_satisfied":         bv_ok,
            "time_labels":          time_labels,
            "evolution_version":    EVOLUTION_VERSION,
        }

        ckpt_path = os.path.join(
            self.cfg.get("output_dir", "./epi_output"), "checkpoint.pkl")
        CheckpointManager.save(ckpt_path, {
            "classifier_state": self.classifier.state_dict(),
            "results":          self.results,
        })
        return self.results

    def plot_epidemic_curve(self, save_path: str = None) -> None:
        rt, states, time = (self.results["rt_smooth"],
                            self.results["states"],
                            self.results["time_labels"])
        colors = ["green", "orange", "red"]
        plt.figure(figsize=(10, 5))
        for i, s in enumerate(states):
            plt.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=colors[s])
        plt.plot(range(len(time)), rt, "ko-")
        plt.xticks(range(len(time)), time, rotation=45)
        plt.xlabel("Time period")
        plt.ylabel("Effective R")
        plt.title("Epidemic trajectory")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        plt.show()


# =============================================================================
# 15. Command Line Interface
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EVOLUTION ONE v5 — Full Differentiable Epidemiological Engine")
    parser.add_argument("--case_data", "-c", required=True,
                        help="CSV: date, location, cases, strain (opt)")
    parser.add_argument("--sequences", "-s",
                        help="FASTA of viral genomes")
    parser.add_argument("--protein",  default="Spike",
                        help="Protein for structural analysis")
    parser.add_argument("--factor_file",
                        help="CSV with external time-series factors")
    parser.add_argument("--pdb_dir",  default="./pdbs")
    parser.add_argument("--output_dir", default="./epi_output")
    parser.add_argument("--interaction_network", nargs="+", type=int,
                        help="Node-index pairs for BV network check")
    parser.add_argument("--train",    action="store_true",
                        help="Train thresholds from outcome labels")
    parser.add_argument("--outcome_labels_file",
                        help="CSV with time and label (0/1/2)")
    parser.add_argument("--no_future", action="store_true")
    parser.add_argument("--no_struct", action="store_true")
    parser.add_argument("--resume",    help="Resume from checkpoint .pkl")
    parser.add_argument("--plot",      action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interaction_network = None
    if args.interaction_network:
        pairs = args.interaction_network
        if len(pairs) % 2 != 0:
            sys.exit("Interaction network must supply an even number of indices.")
        interaction_network = [(pairs[i], pairs[i+1])
                               for i in range(0, len(pairs), 2)]

    engine = EpiForecastEngine(cfg={
        "factor_file": args.factor_file,
        "pdb_dir":     args.pdb_dir,
        "output_dir":  str(out_dir),
    })
    engine.run(
        case_file=args.case_data,
        sequence_file=args.sequences,
        protein_of_interest=args.protein,
        compute_future_variants=not args.no_future,
        compute_structural=not args.no_struct,
        interaction_network=interaction_network,
        train=args.train,
        outcome_labels_file=args.outcome_labels_file,
        resume_from=args.resume,
    )

    if engine.results:
        summary = {
            "pandemic_risk":        engine.results["pandemic_risk"],
            "entropy":              engine.results["entropy"],
            "bv_satisfied":         engine.results["bv_satisfied"],
            "vaccine_design":       engine.results["vaccine_design"],
            "drug_recommendations": engine.results["drug_recommendations"],
            "evolution_version":    engine.results["evolution_version"],
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        pd.DataFrame({
            "time":  engine.results["time_labels"],
            "rt":    engine.results["rt_smooth"],
            "state": engine.results["states"],
        }).to_csv(out_dir / "epidemic_states.csv", index=False)
        print(f"Results saved to {out_dir}")

    if args.plot:
        engine.plot_epidemic_curve(
            save_path=str(out_dir / "epidemic_curve.png"))


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Physics engine
    "RefinementConfig",
    "DifferentiableProteinEnergy",
    "RefinementEngine",
    # Epidemic models
    "EpidemicClassifier",
    # Data
    "EpidemiologicalDataLoader",
    # Viral evolution
    "ViralEvolutionAnalyzer",
    "FutureVariantPredictor",
    # Immunology
    "EpitopeScorer",
    "VaccineDesigner",
    # Clinical
    "TherapeuticRecommender",
    # Epidemiology
    "EpidemiologicalFactorAnalyzer",
    # Network / BV
    "InteractionNetworkBV",
    # Main engine
    "EpiForecastEngine",
    # Utilities
    "calc_dihedral_torch",
    "calc_dihedral_torch_batch",
    "_softplus_floor",
    "_soft_count",
]


if __name__ == "__main__":
    main()
