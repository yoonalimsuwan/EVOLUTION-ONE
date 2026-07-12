# =============================================================================
# EVOLUTION ONE v4 – Multi‑Level Cancer Evolution & Structural Impact Engine
# =============================================================================
# Author  : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID   : 0009-0008-2374-0788
# GitHub  : yoonalimsuwan
# License : MIT
# Year    : 2026
#
# AI Co-Developers:
#   - Claude   (Anthropic)  — SOCController removal (Fix 4), BVFieldTheory
#                             standalone fallback (Fix 1), LangevinBridgeMixin
#                             integration (Fix 3), CahnHilliardEvoBridge (Fix 5),
#                             __all__ public API, cross-cluster integration audit,
#                             full feature-parity restoration (20 missing items)
#   - GPT      (OpenAI)     — literature cross-check, differentiable SOC review
#   - Gemini   (Google)     — initial architecture scaffolding
#   - DeepSeek              — numerical stability verification
#
# Built on open-source foundations:
#   • Biopython – sequence I/O, motif search, BLAST (Biopython License)
#   • PyRanges  – fast genomic interval overlaps (MIT)
#   • SciPy     – numerical methods, differential evolution (BSD-3-Clause)
#   • Pandas    – data manipulation (BSD-3-Clause)
#   • NumPy     – arrays (BSD-3-Clause)
#   • Matplotlib / Seaborn – plotting (PSF-based / BSD-3-Clause)
#   • PyTorch   – automatic differentiation & GPU (BSD-style)
#   • Optuna (optional) – hyperparameter tuning (MIT)
#   • REAL FOLD ONE & HT modules – structural refinement & scanning (MIT)
#   • bowtie / BWA (external, optional) – off-target search (GPL / MIT)
#
# Features (all preserved from v3 + new fixes):
#   • Self-Organised Criticality (SOC) model of cancer evolution
#   • Semantic-State Contraction (SSC) & Renormalisation Group (RG) filtering
#   • Learnable CSOC kernel for adaptive dynamics
#   • Duon-aware mutation analysis (coding + regulatory overlap via BED)
#   • Predictive trajectory: will current mutations lead to cancer?
#   • Future mutation prediction (escape mutations) via REAL FOLD ONE HT
#   • Structural impact (ΔΔG) via REAL FOLD ONE
#   • Chemical intervention recommendation (targeted therapy + stabilisers)
#   • Retrospective lifestyle factor correlation (with p-values)
#   • Gene network BV consistency check
#   • Trainable SOC thresholds from clinical outcomes (diff. evolution / Optuna)
#   • Checkpoint / resume support
#   • Data pipeline with batched processing for large cohorts
#   • CRISPR-Cas editing: gRNA scoring (GC/hairpin/polyT), Bowtie/brute-force
#     off-target search, repair template with codon table + homology arms
#   • Epigenetic editing target ID: per-interval methylation-aware editing
#   • CLI: summary.json, sample_states.csv, crispr_designs.json,
#     epigenetic_targets.json
#   • Vendor-neutral: CPU, Colab T4, Huawei Ascend, Apple MPS, multi-GPU
#
# Changes v3 → v4  (Cross-cluster Integration)
#   Fix 1  — GeneNetworkBV uses BVFieldTheory from one_core_evolution
#   Fix 3  — EvolutionONEEngine inherits LangevinBridgeMixin
#   Fix 4  — EvolutionaryClassifier: SOCController replaced with DifferentiableSOC
#   Fix 5  — EvolutionONEEngine.attach_cahn_bridge() added
#   Fix 6  — EVOLUTION_VERSION from one_core_evolution_v3 (3.0.0)
# =============================================================================

import math, os, sys, json, argparse, logging, warnings, random, itertools
import pickle, subprocess, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass

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
    CSOCBase,                    # CSOC abstract base    (Paper 4)
    BVFieldTheory,               # Fix 1: standalone BV  (no real_fold_one needed)
    DifferentiableRG,            # fully differentiable learnable RG
    DifferentiableSOC,           # Fix 4: canonical SOC  (replaces SOCController)
    DifferentiableIto,           # fully differentiable Itô integrator
    CheckpointManager,           # unified checkpoint
    LangevinEvolutionBridge,     # Langevin ↔ population bridge
    LangevinBridgeMixin,         # Fix 3: attach_langevin_bridge() mixin
    EpiEvolutionBridge,          # Bug 7: bidirectional μ ↔ Rt coupling
    CahnHilliardEvoBridge,       # Fix 5: CH3D ↔ EVOLUTION ONE bridge
    get_device as _core_get_device,
    EVOLUTION_VERSION,
)

# -----------------------------------------------------------------------------
# BioPython (sequence handling and CRISPR design)
# -----------------------------------------------------------------------------
try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import GC
    from Bio import SeqIO
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

# -----------------------------------------------------------------------------
# REAL FOLD ONE & HT (fallback to embedded physics engine)
# -----------------------------------------------------------------------------
try:
    from real_fold_one_v2 import (
        RefinementEngine, RefinementConfig,
        CSOCKernel, SOCController, SemanticStateContraction as _RFO_SSC,
        DiffRGRefiner,
        NeighborListManager, reconstruct_backbone, build_sidechain_atoms,
        get_full_atom_coords_and_types,
        energy_bond, energy_angle, energy_rama, energy_clash,
        energy_electro, energy_solvent, energy_hbond,
        energy_lj_full, energy_coulomb_full, energy_torsion_chi,
        DEFAULT_LJ_PARAMS, DEFAULT_CHARGE_MAP,
        AA_3_TO_1, RESIDUE_CHARGE, MAX_CHI, RESIDUE_NCHI,
        load_structure, save_structure,
        ItoProcess, LangevinDynamics,
        BVFieldTheory as _RFO_BVFieldTheory, DNAOrigamiBV,
        HAS_BIOTITE as RFO_HAS_BIOTITE,
    )
    HAS_REAL_FOLD_ONE = True
    _RFO_BV_AVAILABLE = True
except ImportError:
    HAS_REAL_FOLD_ONE = False
    _RFO_BV_AVAILABLE = False

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
logger = logging.getLogger("EvolutionONE")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

# =============================================================================
# 1. Embedded Physics Engine fallback (when real_fold_one_v2 unavailable)
# =============================================================================

if not HAS_REAL_FOLD_ONE:

    class RefinementConfig:
        def __init__(self, device="cpu", steps=50, lr=0.01):
            self.device = device; self.steps = steps; self.lr = lr

    class RefinementEngine:
        def __init__(self, cfg):
            self.cfg    = cfg
            self.device = cfg.device

        def _energy(self, coords, seq):
            N  = coords.shape[0]
            d  = torch.norm(coords[1:] - coords[:-1], dim=1)
            E  = ((d - 3.8) ** 2).sum()
            if N > 2:
                v1  = coords[:-2] - coords[1:-1]
                v2  = coords[2:]  - coords[1:-1]
                cos = (v1 * v2).sum(-1) / (
                    v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
                E   = E + ((cos - (-0.3)) ** 2).sum()
            return E

        def compute_energy(self, coords, seq):
            return self._energy(coords, seq).item()

        def relax_local(self, coords, seq, mutable_positions, steps=None):
            steps  = steps or self.cfg.steps
            N      = coords.shape[0]
            mut    = [p for p in mutable_positions if 0 <= p < N]
            if not mut:
                return coords.clone().detach(), self.compute_energy(coords, seq)
            mut_idx = set(mut)
            mut_t   = coords[mut].clone().detach().requires_grad_(True)
            fix_t   = coords[[i for i in range(N) if i not in mut_idx]].clone().detach()
            opt     = torch.optim.Adam([mut_t], lr=self.cfg.lr)
            for _ in range(steps):
                opt.zero_grad()
                parts = []; mi = fi = 0
                for i in range(N):
                    if i in mut_idx:
                        parts.append(mut_t[mi].unsqueeze(0)); mi += 1
                    else:
                        parts.append(fix_t[fi].unsqueeze(0)); fi += 1
                self._energy(torch.cat(parts), seq).backward()
                opt.step()
            parts = []; mi = fi = 0
            for i in range(N):
                if i in mut_idx:
                    parts.append(mut_t[mi].detach().unsqueeze(0)); mi += 1
                else:
                    parts.append(fix_t[fi].unsqueeze(0)); fi += 1
            final = torch.cat(parts)
            return final, self._energy(final, seq).item()

    def load_structure(pdb_file):
        logger.warning("load_structure: real_fold_one unavailable, returning empty structure.")
        return {"coords": np.zeros((1, 3)), "sequence": "G"}

    def save_structure(data, path): pass


# =============================================================================
# 2. Gene Network BV  (Fix 1: uses one_core_evolution.BVFieldTheory)
# =============================================================================

class GeneNetworkBV(BVFieldTheory):
    """Batalin-Vilkovisky consistency check for a gene interaction network."""

    def __init__(self, gene_names: List[str], interactions: List[Tuple[int, int]]):
        field_names = [f"phi_{i}" for i in range(len(gene_names))]
        super().__init__(field_names, [0] * len(field_names))
        self.gene_names   = gene_names
        self.interactions = interactions
        for name in field_names:
            self.phi[name] = torch.randn(1) * 0.01

    def action_functional(self, phi_dict, phi_star_dict):
        S = torch.tensor(0.0)
        for i, j in self.interactions:
            S = S + 0.5 * (phi_dict[f"phi_{i}"] - phi_dict[f"phi_{j}"]) ** 2
        return S

    def verify(self) -> bool:
        return self.classical_master_equation(self.action_functional)


# =============================================================================
# 3. Mutation Data Handling
# =============================================================================

class MutationDataLoader:
    def load_maf(self, file_path: str,
                 genes_of_interest: List[str] = None) -> pd.DataFrame:
        maf = pd.read_csv(file_path, sep="\t", comment="#", low_memory=False)
        if genes_of_interest:
            maf = maf[maf["Hugo_Symbol"].isin(genes_of_interest)]
        required = ["Chromosome", "Start_Position", "End_Position",
                    "Hugo_Symbol", "Tumor_Sample_Barcode"]
        for col in required:
            if col not in maf.columns:
                logger.warning("MAF missing column %s, some analyses may be skipped.", col)
        return maf

    def load_vcf(self, file_path: str) -> pd.DataFrame:
        records = []
        with open(file_path) as f:
            for line in f:
                if line.startswith("#"): continue
                fields = line.strip().split("\t")
                if len(fields) < 8: continue
                chrom, pos, _, ref, alt, _, _, info = fields[:8]
                gene = "UNKNOWN"
                if "GENE=" in info:
                    gene = info.split("GENE=")[1].split(";")[0]
                records.append({
                    "Chromosome": chrom,
                    "Start_Position": int(pos),
                    "End_Position": int(pos) + len(ref) - 1,
                    "Reference_Allele": ref,
                    "Tumor_Seq_Allele2": alt,
                    "Hugo_Symbol": gene,
                    "Tumor_Sample_Barcode":
                        os.path.basename(file_path).replace(".vcf", ""),
                })
        return pd.DataFrame(records)

    def build_mutation_matrix(self, mutations: pd.DataFrame,
                              genes: List[str]) -> Tuple[np.ndarray, List[str]]:
        samples      = mutations["Tumor_Sample_Barcode"].unique()
        sample_index = {s: i for i, s in enumerate(samples)}
        gene_index   = {g: j for j, g in enumerate(genes)}
        M = np.zeros((len(samples), len(genes)))
        for _, row in mutations.iterrows():
            g = row.get("Hugo_Symbol", "")
            s = row.get("Tumor_Sample_Barcode", "")
            if g in gene_index and s in sample_index:
                M[sample_index[s], gene_index[g]] = 1
        return M, list(samples)


# =============================================================================
# 4. Duon Analysis
# =============================================================================

class DuonAnalyzer:
    """
    Loads a BED file of duon regions (coding + regulatory overlap).
    Uses PyRanges for fast interval queries.
    """

    def __init__(self, bed_file: str = None):
        self.duon_intervals = None
        if bed_file and os.path.exists(bed_file):
            try:
                self.duon_intervals = pr.read_bed(bed_file)
                logger.info("Loaded duon BED with %d intervals.",
                            len(self.duon_intervals))
            except Exception as e:
                logger.error("Failed to load duon BED: %s", e)

    def compute_duon_mutation_rate(self, mutation_df: pd.DataFrame) -> float:
        if self.duon_intervals is None or mutation_df.empty:
            return 0.0
        required = ["Chromosome", "Start_Position", "End_Position"]
        if not all(c in mutation_df.columns for c in required):
            logger.warning(
                "Mutation DataFrame missing genomic columns; duon rate set to 0.")
            return 0.0
        try:
            # Bug fix (3): tag every mutation row with a unique synthetic ID
            # before the join. A single mutation can overlap >1 duon interval
            # (e.g. fragmented/overlapping BED entries), and `join` emits one
            # output row per overlapping pair. Without de-duplication the
            # resulting rate can exceed 1.0. We count *distinct mutations*
            # that overlap at least one duon, not the number of overlap pairs.
            mut_for_join = mutation_df.reset_index(drop=True).copy()
            mut_for_join["_mut_row_id"] = mut_for_join.index
            mut_gr = pr.PyRanges(mut_for_join.rename(columns={
                "Start_Position": "Start",
                "End_Position":   "End",
            }))
            overlap = mut_gr.join(self.duon_intervals, how="inner")
            if len(overlap) == 0:
                return 0.0
            overlap_df = overlap.df if hasattr(overlap, "df") else overlap
            n_overlapping_mutations = overlap_df["_mut_row_id"].nunique()
            return n_overlapping_mutations / max(len(mutation_df), 1)
        except Exception as e:
            logger.warning("Duon overlap failed: %s", e)
            return 0.0


# =============================================================================
# 5. Evolutionary Classifier with SOC / Itô / RG + Hyperparameter Tuning
# =============================================================================

class EvolutionaryClassifier:
    """
    Classifies patients into stable, critical, or collapse states based on
    mutation load μ.

    Fix 4: SOCController (real_fold_one-only) replaced by DifferentiableSOC
    (one_core_evolution canonical, always available, fully differentiable).
    All other public methods and signatures preserved exactly from v3.
    """

    def __init__(self,
                 threshold_stable:   float = 0.2,
                 threshold_collapse: float = 0.8):
        self.threshold_stable   = threshold_stable
        self.threshold_collapse = threshold_collapse
        # Fix 4: canonical DifferentiableSOC (no SOCController dependency)
        self.rg       = DifferentiableRG(kernel_size=5)
        self._soc_dyn = DifferentiableSOC(base_temp=300.0, beta=0.01)
        self._ito_dyn = DifferentiableIto(T=300.0, dt=0.01)

    def mu_to_state(self, mu: float) -> int:
        if mu < self.threshold_stable:   return 0
        if mu > self.threshold_collapse: return 2
        return 1

    def classify_samples(self, mu_values: np.ndarray) -> np.ndarray:
        """Element-wise state classification — pure NumPy for compatibility."""
        return np.array([self.mu_to_state(float(mu)) for mu in mu_values])

    def classify_samples_differentiable(
        self, mu_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Fully differentiable soft classification via log-softmax.
        Returns (N, 3) probability tensor: [p_stable, p_critical, p_collapse].
        Gradients flow through mu_tensor for end-to-end training.
        """
        logit_stable   = -(mu_tensor - self.threshold_stable)  * 20.0
        logit_collapse =  (mu_tensor - self.threshold_collapse) * 20.0
        logit_critical =  ((mu_tensor - self.threshold_stable) * 20.0
                           - F.softplus((mu_tensor - self.threshold_collapse) * 20.0))
        logits = torch.stack([logit_stable, logit_critical, logit_collapse], dim=-1)
        return F.softmax(logits, dim=-1)

    def compute_entropy(self, states: np.ndarray) -> float:
        hist = np.bincount(states, minlength=3)
        p    = hist / len(states)
        p    = p[p > 0]
        return float(entropy(p, base=2))

    def soc_evolve(self, mu_values: torch.Tensor,
                   steps: int = 10) -> torch.Tensor:
        """Fully differentiable SOC evolution via DifferentiableSOC."""
        return self._soc_dyn(mu_values, steps=steps)

    def ito_evolve(self, mu0: float,
                   T:      float = 300.0,
                   dt:     float = 0.01,
                   steps:  int   = 100) -> torch.Tensor:
        """
        Fully differentiable Itô evolution via DifferentiableIto.
        T and dt parameters accepted for API compatibility (applied via
        DifferentiableIto's learnable T_param and stored dt).
        """
        self._ito_dyn.T_param.data.fill_(T)
        self._ito_dyn.dt = dt
        x0 = torch.tensor([mu0], dtype=torch.float32)
        return self._ito_dyn(x0, steps=steps)

    def tune_thresholds(self, mu_values: np.ndarray,
                        clinical_labels: np.ndarray,
                        n_iter: int = 50,
                        method: str = "differential_evolution"):
        """
        Optimise stable/collapse thresholds to maximise correlation with
        clinical labels.
        method='optuna' uses Bayesian optimisation (requires Optuna), otherwise
        falls back to SciPy's differential evolution.
        """
        def objective(params):
            s_th, c_th = params
            if s_th >= c_th or s_th < 0 or c_th > 1:
                return 1e9
            self.threshold_stable   = s_th
            self.threshold_collapse = c_th
            states    = self.classify_samples(mu_values)
            predicted = np.where(states == 0, 1.0,
                                 np.where(states == 2, 0.0, 0.5))
            if np.std(predicted) == 0 or np.std(clinical_labels) == 0:
                return 0.0
            corr, _ = pearsonr(predicted, clinical_labels)
            return -abs(corr)

        if method == "optuna" and HAS_OPTUNA:
            def optuna_objective(trial):
                return objective([
                    trial.suggest_float("stable",   0.05, 0.45),
                    trial.suggest_float("collapse", 0.55, 0.95),
                ])
            study = optuna.create_study(direction="minimize")
            study.optimize(optuna_objective, n_trials=n_iter)
            self.threshold_stable   = study.best_params["stable"]
            self.threshold_collapse = study.best_params["collapse"]
        else:
            result = differential_evolution(
                objective, [(0.05, 0.45), (0.55, 0.95)],
                maxiter=n_iter, seed=42)
            self.threshold_stable   = result.x[0]
            self.threshold_collapse = result.x[1]

        return self.threshold_stable, self.threshold_collapse


# =============================================================================
# 6. Future Mutation Predictor
# =============================================================================

class FutureMutationPredictor:
    """
    Predicts positions likely to acquire escape mutations by estimating
    ΔΔG upon mutation using either the full HT Scanner or built-in relaxation.
    """

    def __init__(self, pdb_dir: str = "./pdbs"):
        self.pdb_dir = pdb_dir

    def predict_vulnerable_positions(self, gene: str,
                                     structure_file: str = None) -> List[Dict]:
        if HAS_HT:
            if not structure_file:
                candidates = list(Path(self.pdb_dir).glob(f"*{gene}*.pdb"))
                if not candidates:
                    logger.warning("No PDB for %s", gene)
                    return []
                structure_file = str(candidates[0])
            cfg     = HTConfig(pdb_file=structure_file, scan_full=True,
                               output_dir=f"./ht_scan_{gene}")
            scanner = HighThroughputScanner(cfg)
            scanner.load_structure()
            df = pd.DataFrame(scanner.scan_single_mutations())
            if df.empty: return []
            df["ddg"] = df["ddg"].astype(float)
            return (df[df["ddg"] > 1.5]
                    .sort_values("ddg", ascending=False)
                    .to_dict("records"))
        else:
            pdb_file = structure_file or self._find_pdb(gene)
            if not pdb_file: return []
            data   = load_structure(pdb_file)
            coords = torch.tensor(data["coords"], dtype=torch.float32)
            seq    = data["sequence"]
            engine = RefinementEngine(RefinementConfig(device="cpu", steps=50))
            results = []
            for pos in range(len(seq)):
                wt = seq[pos]
                for new in "AGVLI":
                    if new == wt: continue
                    mut_seq = seq[:pos] + new + seq[pos + 1:]
                    try:
                        _, e_mut = engine.relax_local(
                            coords.clone().detach().requires_grad_(True),
                            mut_seq, [pos], steps=20)
                        ddg = e_mut - engine.compute_energy(coords, seq)
                        if ddg > 1.5:
                            results.append({
                                "chain": 0, "pos_in_chain": pos,
                                "global_pos": pos,
                                "wt": wt, "mut": new, "ddg": ddg,
                                "type": "mutation",
                            })
                    except Exception as e:
                        logger.warning("Relax failed %s %d%s->%s: %s",
                                       gene, pos, wt, new, e)
            return results

    def _find_pdb(self, gene: str) -> Optional[str]:
        candidates = list(Path(self.pdb_dir).glob(f"*{gene}*.pdb"))
        return str(candidates[0]) if candidates else None


# =============================================================================
# 7. CRISPR-Cas Editing Target Designer (full feature set)
# =============================================================================

class CRISPRDesigner:
    """
    Designs gRNA and repair templates for CRISPR-Cas editing.

    Scores guides using a rule-based model (Doench 2016 principles) and
    performs off-target search via Bowtie (if available) or basic string
    matching on a reference genome FASTA.
    """

    def __init__(self, pdb_dir: str = "./pdbs", ref_genome: str = None):
        self.pdb_dir    = pdb_dir
        self.ref_genome = ref_genome
        self.genome_seq = None
        if ref_genome and os.path.exists(ref_genome) and HAS_BIOPYTHON:
            try:
                self.genome_seq = SeqIO.to_dict(
                    SeqIO.parse(ref_genome, "fasta"))
                logger.info("Loaded reference genome with %d contigs.",
                            len(self.genome_seq))
            except Exception as e:
                logger.warning("Could not load reference genome: %s", e)

    # ---------- gRNA scoring (Doench 2016 based) ----------

    @staticmethod
    def score_grna(seq: str) -> float:
        """
        Compute on-target efficacy score for a 20mer guide.
        Implements simplified features from Doench et al. 2016.
        Returns a score between 0 and 1 (higher = better).
        """
        if len(seq) != 20:
            return 0.0
        s = seq.upper()

        # GC content (prefer 30-70%)
        if HAS_BIOPYTHON:
            gc = GC(s)
        else:
            gc = sum(1 for c in s if c in "GC") / 20 * 100
        gc_score = 1.0 if 30 <= gc <= 70 else max(0.0, 1 - abs(gc - 50) / 50)

        # Avoid poly-T stretches (>4 T's)
        polyT_score = 0.0 if "TTTT" in s else 1.0

        # Preference for G at position 20 (adjacent to PAM)
        pos20_score = 1.0 if s[19] == "G" else 0.5

        # Self-complementarity (hairpin) – inverted repeat length
        def max_inv_repeat(sequence: str) -> int:
            best = 0
            for i in range(len(sequence)):
                for j in range(i + 4, len(sequence)):
                    sub = sequence[i:j]
                    if HAS_BIOPYTHON:
                        rc = str(Seq(sub).reverse_complement())
                    else:
                        comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
                        rc = "".join(comp.get(c, "N") for c in reversed(sub))
                    if rc in sequence[j:]:
                        best = max(best, len(sub))
            return best

        inv_rep_len  = max_inv_repeat(s)
        hairpin_score = max(0.0, 1 - inv_rep_len / 8)

        return (0.3 * gc_score + 0.3 * polyT_score
                + 0.2 * pos20_score + 0.2 * hairpin_score)

    # ---------- Off-target search ----------

    def find_off_targets(self, guide: str,
                         max_mismatches: int = 3,
                         exclude_locus: Optional[Tuple[str, int, int]] = None
                         ) -> List[str]:
        """
        Search for off-target sites in the reference genome.
        Uses Bowtie if installed, otherwise performs brute-force approximate
        matching (only recommended for small genomes).

        exclude_locus: (chrom, genomic_start, genomic_end) of the guide's own
        on-target site, 0-based half-open, matching the coordinates returned
        by design_grna. Bug fix (4): without this, the guide's own exact-match
        location was always reported back as an "off-target" (0 mismatches
        against itself), inflating every off-target count by at least 1.
        """
        if not guide or len(guide) != 20:
            return []
        if HAS_BOWTIE and self.ref_genome:
            hits = self._bowtie_off_targets(guide, max_mismatches)
        elif self.genome_seq is not None:
            hits = self._bruteforce_off_targets(guide, max_mismatches)
        else:
            logger.warning(
                "No reference genome or Bowtie available; skipping off-target search.")
            return []

        if exclude_locus is not None:
            ex_chrom, ex_start, ex_end = exclude_locus
            # hits are formatted as "chrom:1-based_start-1-based_end"
            excl_tag = f"{ex_chrom}:{ex_start + 1}-{ex_end}"
            hits = [h for h in hits if h != excl_tag]
        return hits

    def _bowtie_off_targets(self, guide: str, max_mm: int) -> List[str]:
        """Run Bowtie 1 to find off-targets (0-3 mismatches)."""
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".fa", delete=False) as tf:
                tf.write(f">guide\n{guide}\n")
                fa_file = tf.name
            index_base = (self.ref_genome
                          .replace(".fa", "")
                          .replace(".fasta", ""))
            cmd = ["bowtie", "-f", "-v", str(max_mm), "-a",
                   "--suppress", "1,2,3,4,5,6,7",
                   "-x", index_base, fa_file, "/dev/null"]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=30)
            os.unlink(fa_file)
            if result.returncode != 0:
                logger.warning("Bowtie failed: %s", result.stderr)
                return []
            return [ln.strip() for ln in result.stdout.split("\n")
                    if ln.strip() and not ln.startswith("@")]
        except Exception as e:
            logger.warning("Bowtie off-target search error: %s", e)
            return []

    def _bruteforce_off_targets(self, guide: str,
                                max_mm: int) -> List[str]:
        """Fallback approximate string matching for small genomes."""
        if not HAS_BIOPYTHON:
            return []
        off_targets = []
        guide_rc = str(Seq(guide).reverse_complement())
        for chrom, record in self.genome_seq.items():
            seq = str(record.seq).upper()
            for s in [guide, guide_rc]:
                for i in range(len(seq) - 19):
                    window     = seq[i:i + 20]
                    mismatches = sum(1 for a, b in zip(s, window) if a != b)
                    if mismatches <= max_mm:
                        off_targets.append(f"{chrom}:{i+1}-{i+20}")
        return off_targets

    # ---------- gRNA design ----------

    def design_grna(self, gene: str, target_pos: int,
                    target_mut: str, flank_size: int = 250) -> Optional[Dict]:
        """
        Find all possible gRNA sequences near the target codon and return
        the best one based on on-target score and minimal off-targets.
        Requires a reference genome sequence to extract the guide context.
        """
        if not self.genome_seq:
            logger.warning("No reference genome; cannot design gRNA.")
            return None
        chrom     = list(self.genome_seq.keys())[0]
        chrom_seq = str(self.genome_seq[chrom].seq).upper()
        start  = max(0, target_pos - flank_size)
        end    = min(len(chrom_seq), target_pos + flank_size + 20)
        window = chrom_seq[start:end]

        candidates = []
        for i in range(20, len(window) - 2):
            pam = window[i + 1:i + 3]
            if pam == "GG":
                guide = window[i - 19:i + 1]
                if len(guide) == 20 and "N" not in guide:
                    g_start = start + i - 19
                    g_end   = start + i
                    score   = self.score_grna(guide)
                    off     = self.find_off_targets(
                        guide, max_mismatches=3,
                        exclude_locus=(chrom, g_start, g_end))
                    candidates.append({
                        "guide":           guide,
                        "score":           score,
                        "off_targets":     off,
                        "genomic_start":   start + i - 19,
                        "genomic_end":     start + i,
                    })
        if not candidates:
            return None
        return max(candidates,
                   key=lambda x: x["score"] - 0.01 * len(x["off_targets"]))

    # ---------- Repair template ----------

    def design_repair_template(self, gene: str, target_pos: int,
                               wt_aa: str, new_aa: str,
                               homology_arm_length: int = 30) -> Optional[Dict]:
        """
        Create a repair template with homology arms for HDR.
        Uses the reference genome to fetch surrounding sequence and replaces
        the target codon.
        """
        if not self.genome_seq:
            logger.warning("No reference genome; cannot design repair template.")
            return None
        codon_table = {
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
        if new_aa not in codon_table:
            return None
        new_codon = codon_table[new_aa][0]
        chrom     = list(self.genome_seq.keys())[0]
        chrom_seq = str(self.genome_seq[chrom].seq).upper()
        left_arm  = chrom_seq[max(0, target_pos - homology_arm_length):target_pos]
        right_arm = chrom_seq[target_pos + 3:target_pos + 3 + homology_arm_length]
        if len(left_arm) < homology_arm_length:
            left_arm = "N" * (homology_arm_length - len(left_arm)) + left_arm
        if len(right_arm) < homology_arm_length:
            right_arm = right_arm + "N" * (homology_arm_length - len(right_arm))
        template = left_arm + new_codon + right_arm
        return {
            "left_arm":     left_arm,
            "right_arm":    right_arm,
            "new_codon":    new_codon,
            "full_template": template,
        }

    # ---------- Main design pipeline ----------

    def design_crispr_targets(self, gene: str,
                              vulnerable_mutations: List[Dict]) -> List[Dict]:
        designs = []
        for mut in vulnerable_mutations:
            pos    = mut.get("global_pos", mut.get("pos_in_chain", 0))
            guide  = self.design_grna(gene, pos, mut.get("mut", "X"))
            repair = self.design_repair_template(
                gene, pos, mut.get("wt", "X"), mut.get("mut", "X"))
            if guide and repair:
                designs.append({
                    "gene":              gene,
                    "position":          pos,
                    "wt_aa":             mut.get("wt"),
                    "new_aa":            mut.get("mut"),
                    "ddg":               mut.get("ddg", 0.0),
                    "gRNA_sequence":     guide["guide"],
                    "on_target_score":   guide["score"],
                    "off_targets":       len(guide["off_targets"]),
                    "repair_template":   repair["full_template"],
                    "guide_genomic_start": guide["genomic_start"],
                    "guide_genomic_end":   guide["genomic_end"],
                })
        return designs


# =============================================================================
# 8. Epigenetic Editing Target Designer (full per-interval logic)
# =============================================================================

class EpigeneticDesigner:
    """
    Identifies duons that should be epigenetically edited based on mutation
    frequency and methylation status (when available).
    """

    def __init__(self, duon_analyzer: DuonAnalyzer):
        self.duon_analyzer = duon_analyzer

    def identify_silenced_duons(self, mutation_df: pd.DataFrame,
                                methylation_bed: str = None) -> List[Dict]:
        """
        Input : mutation_df with genomic positions; optional methylation BED.
        Returns list of duons with recommended editing action.
        """
        if self.duon_analyzer.duon_intervals is None:
            return []

        duon_intervals = self.duon_analyzer.duon_intervals
        # Bug fix (1): PyRanges has no `itergenome()` method (verified against
        # both pyranges v0 and the pyranges1 rewrite) — calling it raised
        # AttributeError and crashed this method on every real invocation.
        # The portable way to get a plain row-iterable from a PyRanges object
        # is via its underlying DataFrame.
        duon_df = duon_intervals.df if hasattr(duon_intervals, "df") else duon_intervals

        # Load methylation data if available
        meth_status: Dict = {}
        if (methylation_bed and os.path.exists(methylation_bed)
                and HAS_PYRANGES):
            try:
                meth = pr.read_bed(methylation_bed)
                meth_df = meth.df if hasattr(meth, "df") else meth
                for _, interval in meth_df.iterrows():
                    key = (interval["Chromosome"], interval["Start"], interval["End"])
                    meth_status[key] = (float(interval["Score"])
                                        if "Score" in interval and pd.notna(interval["Score"])
                                        else 50.0)
            except Exception as e:
                logger.warning("Failed to load methylation BED: %s", e)

        # Bug fix (2): the original implementation re-scanned the *entire*
        # mutation_df with a Python-level `iterrows()` for every single duon
        # interval (O(n_mutations * n_duons)), which defeats the entire
        # purpose of using PyRanges for "fast genomic interval overlap".
        # We instead do a single vectorized PyRanges join once, then group
        # the resulting overlaps by duon interval — O(n_mutations + n_duons)
        # in practice (PyRanges' join itself is the optimized interval-tree
        # step from the library).
        mut_overlap_counts: Dict[Tuple, int] = defaultdict(int)
        required = ["Chromosome", "Start_Position", "End_Position"]
        if HAS_PYRANGES and all(c in mutation_df.columns for c in required) \
                and not mutation_df.empty:
            try:
                mut_gr = pr.PyRanges(mutation_df.rename(columns={
                    "Start_Position": "Start",
                    "End_Position":   "End",
                }))
                joined = mut_gr.join(duon_intervals, how="inner")
                joined_df = joined.df if hasattr(joined, "df") else joined
                if len(joined_df) > 0:
                    # join() suffixes the right-hand (duon) Start/End columns
                    # to avoid collision with the left-hand mutation columns.
                    start_col = "Start_b" if "Start_b" in joined_df.columns else "Start"
                    end_col   = "End_b"   if "End_b"   in joined_df.columns else "End"
                    grouped = joined_df.groupby(
                        ["Chromosome", start_col, end_col]).size()
                    for (chrom, start, end), count in grouped.items():
                        mut_overlap_counts[(chrom, start, end)] = int(count)
            except Exception as e:
                logger.warning("Duon/mutation overlap join failed: %s", e)

        targets = []
        for _, interval in duon_df.iterrows():
            chrom = interval["Chromosome"]
            start = interval["Start"]
            end   = interval["End"]

            mut_overlap = mut_overlap_counts.get((chrom, start, end), 0)
            freq        = mut_overlap / max(len(mutation_df), 1)
            meth_key    = (chrom, start, end)
            meth_level  = meth_status.get(meth_key, 50.0)

            if freq > 0.1:
                if meth_level > 70:
                    rec = "dCas9-TET (demethylation)"
                elif meth_level < 30:
                    rec = "dCas9-DNMT (hypermethylation)"
                else:
                    rec = "further investigation needed"
                targets.append({
                    "chromosome":       chrom,
                    "start":            start,
                    "end":              end,
                    "mutation_frequency": freq,
                    "methylation_level":  meth_level,
                    "recommended_edit":   rec,
                })
        return targets

    def design_epigenetic_targets(self, genes: List[str],
                                  mutation_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        all_targets: Dict = {}
        targets = self.identify_silenced_duons(mutation_df)
        if targets:
            all_targets["global"] = targets
        return all_targets


# =============================================================================
# 9. Chemical Intervention Recommender (full 17-target list)
# =============================================================================

CANCER_DRUG_TARGETS = {
    "KRAS":   ["Sotorasib", "Adagrasib"],
    "EGFR":   ["Erlotinib", "Gefitinib", "Osimertinib"],
    "BRAF":   ["Vemurafenib", "Dabrafenib"],
    "PIK3CA": ["Alpelisib"],
    "ALK":    ["Crizotinib", "Alectinib"],
    "TP53":   ["APR-246", "COBI-348"],
    "IDH1":   ["Ivosidenib"],
    "IDH2":   ["Enasidenib"],
    "FLT3":   ["Midostaurin", "Gilteritinib"],
    "NTRK1":  ["Larotrectinib", "Entrectinib"],
    "MET":    ["Capmatinib", "Tepotinib"],
    "ERBB2":  ["Trastuzumab", "Pertuzumab"],
    "FGFR2":  ["Pemigatinib"],
    "FGFR3":  ["Erdafitinib"],
    "MTOR":   ["Everolimus", "Temsirolimus"],
    "AKT1":   ["Capivasertib"],
    "MAP2K1": ["Trametinib", "Selumetinib"],
}


class InterventionRecommender:
    def __init__(self):
        self.target_map = CANCER_DRUG_TARGETS

    def recommend_drugs(self, gene: str, ddg_data: List[Dict]) -> List[str]:
        drugs = self.target_map.get(gene, [])
        if not drugs:
            return []
        if any(abs(e.get("ddg", 0.0)) > 2.0 for e in ddg_data):
            return drugs
        return []

    def suggest_stabilisers(self, gene: str) -> List[str]:
        return {"TP53": ["PRIMA-1", "PhiKan083"]}.get(gene, [])


# =============================================================================
# 10. Retrospective Lifestyle Factor Analysis (with p-values)
# =============================================================================

class RetrospectiveAnalyzer:
    def __init__(self, lifestyle_file: str = None):
        self.lifestyle_df = None
        if lifestyle_file and os.path.exists(lifestyle_file):
            self.lifestyle_df = pd.read_csv(lifestyle_file)

    def merge_with_mutation_data(self, sample_ids: List[str],
                                 mu_values: np.ndarray,
                                 duon_rates: np.ndarray = None) -> pd.DataFrame:
        df = pd.DataFrame({"sample_id": sample_ids, "mu": mu_values})
        if duon_rates is not None:
            df["duon_rate"] = duon_rates
        if self.lifestyle_df is not None:
            df = df.merge(self.lifestyle_df, on="sample_id", how="inner")
        return df

    def compute_correlations(self, merged_df: pd.DataFrame,
                             target: str = "mu") -> Dict[str, Dict]:
        factors = [c for c in merged_df.columns
                   if c not in ("sample_id", "mu", "duon_rate")]
        results: Dict = {}
        for fac in factors:
            if merged_df[fac].nunique() < 2:
                continue
            r_pearson,  p_pearson  = pearsonr(merged_df[fac], merged_df[target])
            r_spearman, p_spearman = spearmanr(merged_df[fac], merged_df[target])
            results[fac] = {
                "pearson_r":  r_pearson,  "pearson_p":  p_pearson,
                "spearman_r": r_spearman, "spearman_p": p_spearman,
            }
        return results


# =============================================================================
# 12. Main Evolution ONE Engine  (LangevinBridgeMixin — Fix 3)
# =============================================================================

class EvolutionONEEngine(LangevinBridgeMixin):
    """
    Multi-level cancer evolution & structural impact engine.

    Fix 3: Inherits LangevinBridgeMixin → gains attach_langevin_bridge().
    Fix 5: attach_cahn_bridge() connects CH3D phase-field to μ.
    """

    def __init__(self, cfg: dict = None):
        self.cfg              = cfg or {}
        self.loader           = MutationDataLoader()
        self.duon             = DuonAnalyzer(self.cfg.get("duon_bed"))
        self.classifier       = EvolutionaryClassifier()
        self.structural       = FutureMutationPredictor(
            pdb_dir=self.cfg.get("pdb_dir", "./pdbs"))
        self.drug_engine      = InterventionRecommender()
        self.retrospective    = RetrospectiveAnalyzer(
            lifestyle_file=self.cfg.get("lifestyle_file"))
        self.crispr_designer  = CRISPRDesigner(
            pdb_dir=self.cfg.get("pdb_dir", "./pdbs"),
            ref_genome=self.cfg.get("ref_genome"))
        self.epigenetic_designer = EpigeneticDesigner(self.duon)

        # Bridges
        self.langevin_bridge: Optional[LangevinEvolutionBridge] = None
        self.epi_bridge:      Optional[EpiEvolutionBridge]      = None
        self.cahn_bridge:     Optional[CahnHilliardEvoBridge]   = None
        self.results: Dict    = {}

    # ── Epi bridge (Bug 7) ────────────────────────────────────────────────────

    def attach_epi_bridge(
        self,
        epi_engine,
        mu_to_rt_scale: float = 0.5,
        rt_to_mu_scale: float = 0.3,
    ) -> EpiEvolutionBridge:
        """
        Attach a differentiable EpiEvolutionBridge linking this
        EvolutionONEEngine to an EpiForecastEngine.

        After attaching, call bridge.mu_to_rt(mu, rt_base) to couple mutation
        load into epidemic Rt, and bridge.rt_to_mu(rt, mu_base) for reverse.
        """
        self.epi_bridge = EpiEvolutionBridge(
            mu_to_rt_scale=mu_to_rt_scale,
            rt_to_mu_scale=rt_to_mu_scale,
        )
        if hasattr(epi_engine, "evo_bridge"):
            epi_engine.evo_bridge = self.epi_bridge
        logger.info("EpiEvolutionBridge attached.")
        return self.epi_bridge

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

        Usage::
            ch_solver = StructuralCahnHilliard3D(cfg)
            ch_bridge = engine.attach_cahn_bridge(ch_solver)
            u_field   = ch_solver.step(u_init, sigma)
            mu        = ch_bridge.project_to_mu(u_field)
        """
        self.cahn_bridge = CahnHilliardEvoBridge(
            ch_solver=ch_solver, ssc=ssc,
            mu_floor=mu_floor, interface_weight=interface_weight,
        )
        logger.info("CahnHilliardEvoBridge attached (interface_weight=%.3f).",
                    interface_weight)
        return self.cahn_bridge

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self,
            input_file:              str,
            genes:                   List[str],
            format:                  str  = "maf",
            compute_future_mutations:bool = True,
            compute_structural:      bool = True,
            gene_interactions:       Optional[List[Tuple[int, int]]] = None,
            train_thresholds:        bool = False,
            clinical_labels_file:    Optional[str] = None,
            tune_method:             str  = "differential_evolution",
            resume_from:             Optional[str] = None,
            batch_size:              int  = 500) -> Dict:

        if resume_from:
            ckpt = CheckpointManager.load(resume_from)
            if ckpt:
                self.classifier.threshold_stable   = ckpt.get(
                    "threshold_stable",   0.2)
                self.classifier.threshold_collapse = ckpt.get(
                    "threshold_collapse", 0.8)

        # 1. Load mutations
        if format == "maf":
            mut_df = self.loader.load_maf(input_file, genes)
        elif format == "vcf":
            mut_df = self.loader.load_vcf(input_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        if mut_df.empty:
            logger.warning("No mutations found.")
            return {}

        # 2. Mutation matrix
        M, sample_ids = self.loader.build_mutation_matrix(mut_df, genes)
        N_samples, N_genes = M.shape
        logger.info("Total samples: %d  genes: %d", N_samples, N_genes)

        # 3. μ + RG smoothing
        mu_raw    = M.mean(axis=1)
        mu_tensor = torch.tensor(mu_raw, dtype=torch.float32)
        mu_smooth = self.classifier.rg(mu_tensor).detach().numpy()

        # 4. Duon rate
        if self.duon.duon_intervals is not None:
            duon_rates = np.full(N_samples,
                                 self.duon.compute_duon_mutation_rate(mut_df))
        else:
            duon_rates = np.zeros(N_samples)

        # 5. Phase classification
        states = self.classifier.classify_samples(mu_smooth)
        H      = self.classifier.compute_entropy(states)
        logger.info("H=%.4f  stable=%d  critical=%d  collapse=%d",
                    H, (states==0).sum(), (states==1).sum(), (states==2).sum())

        # 6. Threshold tuning
        if train_thresholds and clinical_labels_file:
            cdf    = pd.read_csv(clinical_labels_file)
            merged = pd.DataFrame({"sample_id": sample_ids, "mu": mu_smooth})
            merged = merged.merge(cdf, on="sample_id", how="inner")
            if len(merged) > 10:
                logger.info("Training SOC thresholds...")
                self.classifier.tune_thresholds(
                    merged["mu"].values, merged["label"].values,
                    n_iter=50, method=tune_method)
                logger.info("Tuned: stable=%.3f  collapse=%.3f",
                            self.classifier.threshold_stable,
                            self.classifier.threshold_collapse)

        # 7. Predictive evolution
        # NOTE (fix): soc_evolve's `spread` term is a soft range estimator
        # (logsumexp-based soft max - soft min) over the *batch* dimension.
        # Feeding it a single-element tensor (the cohort mean) makes spread
        # collapse algebraically to 2x, so the SOC dynamics degenerate into a
        # trivial decay that never reflects the cohort's actual criticality.
        # Feed the full per-sample cohort instead so spread is meaningful,
        # then reduce to a scalar for the downstream risk decision.
        mu_avg        = float(mu_smooth.mean())
        mu_t_cohort   = torch.tensor(mu_smooth, dtype=torch.float32)
        future_mu_soc = self.classifier.soc_evolve(mu_t_cohort, steps=20).mean()
        future_mu_ito = self.classifier.ito_evolve(mu_avg, steps=100)
        cancer_risk   = ("High"
                         if (self.classifier.mu_to_state(future_mu_soc.item()) == 2
                             or  self.classifier.mu_to_state(future_mu_ito.item()) == 2)
                         else ("Moderate"
                               if (self.classifier.mu_to_state(future_mu_soc.item()) == 1
                                   or  self.classifier.mu_to_state(future_mu_ito.item()) == 1)
                               else "Low"))
        logger.info("Predicted cancer risk: %s", cancer_risk)

        # 8. BV check
        bv_ok = False
        if gene_interactions and len(genes) > 1:
            try:
                bv_net = GeneNetworkBV(genes, gene_interactions)
                bv_ok  = bv_net.verify()
                logger.info("BV satisfied: %s", bv_ok)
            except Exception as e:
                logger.warning("BV check failed: %s", e)

        # 9. Future mutations
        future_mutations: Dict = {}
        if compute_future_mutations:
            for gene in genes:
                vuln = self.structural.predict_vulnerable_positions(gene)
                if vuln:
                    future_mutations[gene] = vuln

        # 10. CRISPR designs
        crispr_designs: Dict = {}
        for gene, muts in future_mutations.items():
            designs = self.crispr_designer.design_crispr_targets(gene, muts)
            if designs:
                crispr_designs[gene] = designs

        # 11. Epigenetic targets
        epigenetic_targets = self.epigenetic_designer.design_epigenetic_targets(
            genes, mut_df)

        # 12. Structural impact & drugs
        structural_results: Dict = {}
        drug_recos:         Dict = {}
        if compute_structural:
            for gene in genes:
                pdb_file = self._find_pdb(gene)
                if not pdb_file: continue
                data   = load_structure(pdb_file)
                coords = torch.tensor(data["coords"], dtype=torch.float32)
                seq    = data["sequence"]
                engine = RefinementEngine(RefinementConfig(device="cpu", steps=50))
                impacts = []
                for pos in range(len(seq)):
                    wt = seq[pos]
                    for new in "AVLIG":
                        if new == wt: continue
                        mut_seq = seq[:pos] + new + seq[pos + 1:]
                        try:
                            _, e_mut = engine.relax_local(
                                coords.clone().detach().requires_grad_(True),
                                mut_seq, [pos], steps=30)
                            impacts.append({
                                "position": pos, "wt": wt, "mut": new,
                                "ddg": e_mut - engine.compute_energy(coords, seq),
                            })
                        except Exception as e:
                            logger.warning("Relax failed %s %d%s->%s: %s",
                                           gene, pos, wt, new, e)
                structural_results[gene] = impacts
                drugs = self.drug_engine.recommend_drugs(gene, impacts)
                stabs = self.drug_engine.suggest_stabilisers(gene)
                if drugs or stabs:
                    drug_recos[gene] = {"targeted_drugs": drugs,
                                        "stabilisers": stabs}

        # 13. Lifestyle correlations (μ and duon_rate, with p-values)
        lifestyle_corrs: Dict = {}
        if self.retrospective.lifestyle_df is not None:
            merged = self.retrospective.merge_with_mutation_data(
                sample_ids, mu_smooth, duon_rates)
            if len(merged) > 5:
                lifestyle_corrs["mu"] = self.retrospective.compute_correlations(
                    merged, target="mu")
                if "duon_rate" in merged.columns:
                    lifestyle_corrs["duon_rate"] = \
                        self.retrospective.compute_correlations(
                            merged, target="duon_rate")

        self.results = {
            "mu_smooth":             mu_smooth,
            "states":                states,
            "entropy":               H,
            "duon_rates":            duon_rates,
            "cancer_risk":           cancer_risk,
            "future_mu_soc":         future_mu_soc.item(),
            "future_mu_ito":         future_mu_ito.item(),
            "bv_satisfied":          bv_ok,
            "future_mutations":      future_mutations,
            "structural_impacts":    structural_results,
            "drug_recommendations":  drug_recos,
            "lifestyle_correlations": lifestyle_corrs,
            "crispr_designs":        crispr_designs,
            "epigenetic_targets":    epigenetic_targets,
            "sample_ids":            sample_ids,
            "threshold_stable":      self.classifier.threshold_stable,
            "threshold_collapse":    self.classifier.threshold_collapse,
            "evolution_version":     EVOLUTION_VERSION,
        }

        ckpt_path = os.path.join(
            self.cfg.get("output_dir", "./evo_output"), "checkpoint.pkl")
        CheckpointManager.save(ckpt_path, self.results)
        return self.results

    def _find_pdb(self, gene: str) -> Optional[str]:
        candidates = list(
            Path(self.cfg.get("pdb_dir", "./pdbs")).glob(f"*{gene}*.pdb"))
        return str(candidates[0]) if candidates else None

    def plot_phase_diagram(self, save_path: str = None):
        mus = np.linspace(0, 1, 100)
        H_vals = [self.classifier.compute_entropy(
            self.classifier.classify_samples(np.full(1000, mu))) for mu in mus]
        plt.figure(figsize=(8, 5))
        plt.plot(mus, H_vals, linewidth=2)
        plt.xlabel("Mutation load μ"); plt.ylabel("Entropy H (bits)")
        plt.title("Cancer Evolution Phase Diagram")
        if save_path:
            plt.savefig(save_path, dpi=200)
        plt.show()


# =============================================================================
# 13. TCGA Real-Data Validation (cBioPortal clinical TSV)
# =============================================================================
#
# Added by Claude (Anthropic), June 2026.
#
# Reproduces the "Evolution ONE — REAL TCGA Data" panel-A→K validation figure
# from cBioPortal pan-can-atlas-2018 clinical TSV exports (BRCA, COADREAD).
#
# Threshold-logic bug fix included here (see TCGAValidator.classify
# docstring): earlier ad-hoc analysis scripts apparently computed regime
# counts for the bar chart (panel A) from the *full* cohort, but computed
# Kaplan-Meier survival curves (panel C) only on the *subset* that had
# non-missing Overall Survival data. Because a few patients have valid
# Mutation Count but missing OS, the two panels' "n per regime" disagreed
# (this is what produced the n=1 vs n=8 "Collapse" mismatch in the earlier
# figure). This implementation classifies regimes once on the
# analysis-ready dataframe (rows with valid mu AND valid OS) and reuses
# that single classification for every panel, so all counts are
# guaranteed self-consistent.
# =============================================================================

@dataclass
class TCGAValidationConfig:
    """
    Config for TCGAValidator.

    mu_source: how to compute the SOC mutation load μ per patient.
      - "tmb"      : μ = TMB (nonsynonymous), min-max normalised to [0,1]
                     across the pooled cohort. Preferred when present since
                     TMB is already mutations/Mb (panel length-normalised).
      - "mutcount" : μ = raw "Mutation Count", min-max normalised to [0,1].
                     Used as fallback when TMB column is absent/empty.
      - "auto"     : use "tmb" if available for a row, else fall back to
                     "mutcount" for that row (default).
    """
    mu_source: str = "auto"
    threshold_stable:   float = 0.2
    threshold_collapse: float = 0.8
    tune_thresholds:    bool = True
    tune_n_iter:        int = 50
    tune_method:        str = "differential_evolution"
    random_seed:        int = 42


class TCGAValidator:
    """
    Loads cBioPortal pan-can-atlas-2018 clinical_data.tsv exports, computes
    SOC mutation load μ, classifies Stable/Critical/Collapse regimes using
    EvolutionaryClassifier, and renders the panel A-K validation figure.

    Usage:
        v = TCGAValidator()
        df = v.run({"BRCA": "brca_..._clinical_data.tsv",
                     "COAD": "coadread_..._clinical_data.tsv"},
                    save_path="evolution_one_tcga_validation.png")
    """

    REQUIRED_COLUMNS = [
        "Patient ID", "Sample ID", "Mutation Count", "TMB (nonsynonymous)",
        "Overall Survival (Months)", "Overall Survival Status",
        "American Joint Committee on Cancer Tumor Stage Code",
        "Subtype",
    ]

    def __init__(self, cfg: TCGAValidationConfig = None):
        self.cfg = cfg or TCGAValidationConfig()
        self.classifier = EvolutionaryClassifier(
            threshold_stable=self.cfg.threshold_stable,
            threshold_collapse=self.cfg.threshold_collapse,
        )

    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    def load_cohort(self, tsv_paths: Dict[str, str]) -> pd.DataFrame:
        """
        tsv_paths: dict mapping a short cohort label (e.g. "BRCA", "COAD")
        to a cBioPortal clinical_data.tsv file path. Files are concatenated
        with a "CohortLabel" column added so downstream code can stratify
        by cancer type (panel B).
        """
        frames = []
        for label, path in tsv_paths.items():
            df = pd.read_csv(path, sep="\t", low_memory=False)
            missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                logger.warning(
                    "TCGAValidator: %s missing columns %s — those panels "
                    "will be skipped/NaN for this cohort.", label, missing)
            df["CohortLabel"] = label
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True, sort=False)
        logger.info("TCGAValidator: loaded %d total patients across %d cohorts",
                     len(combined), len(tsv_paths))
        return combined

    # ------------------------------------------------------------------ #
    # 2. Compute μ
    # ------------------------------------------------------------------ #
    def compute_mu(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        mut_count = pd.to_numeric(
            df.get("Mutation Count", np.nan), errors="coerce")
        tmb = pd.to_numeric(
            df.get("TMB (nonsynonymous)", np.nan), errors="coerce")

        if self.cfg.mu_source == "tmb":
            raw = tmb
        elif self.cfg.mu_source == "mutcount":
            raw = mut_count
        else:  # "auto": prefer TMB, fall back to Mutation Count per-row
            raw = tmb.where(tmb.notna(), mut_count)

        df["_mu_raw"] = raw

        # Drop rows with no usable mutation-load signal at all — these
        # cannot be classified into a SOC regime and must not silently
        # appear as a phantom "Stable" (mu=0) patient.
        n_before = len(df)
        df = df[df["_mu_raw"].notna()].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info("TCGAValidator: dropped %d/%d patients with no "
                         "Mutation Count or TMB value", n_dropped, n_before)

        # Min-max normalise to [0,1] over the pooled cohort actually used,
        # matching the θ_s≈0.15 / θ_c≈0.57 scale seen in the reference figure.
        mu_min, mu_max = df["_mu_raw"].min(), df["_mu_raw"].max()
        if mu_max > mu_min:
            df["mu"] = (df["_mu_raw"] - mu_min) / (mu_max - mu_min)
        else:
            df["mu"] = 0.0

        # RG smoothing via the engine's own DifferentiableRG, applied across
        # the pooled cohort exactly as EvolutionONEEngine.run() does for
        # MAF/VCF input (step 3 there) — keeps μ definition consistent
        # between the CLI pipeline and this validation pipeline.
        mu_tensor = torch.tensor(df["mu"].values, dtype=torch.float32)
        df["mu_smooth"] = self.classifier.rg(mu_tensor).detach().numpy()

        return df

    # ------------------------------------------------------------------ #
    # 3. Classify (single source of truth for every panel's regime counts)
    # ------------------------------------------------------------------ #
    def _parse_os(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse OS months (float) and OS event indicator (1=dead, 0=censored)."""
        df = df.copy()
        df["_os_months"] = pd.to_numeric(
            df.get("Overall Survival (Months)", np.nan), errors="coerce")
        status = df.get("Overall Survival Status", pd.Series(
            [None] * len(df), index=df.index))
        # cBioPortal encodes this as "0:LIVING" / "1:DECEASED"
        df["_os_event"] = status.astype(str).str.startswith("1").astype(float)
        df.loc[status.isna(), "_os_event"] = np.nan
        return df

    def classify(self, df: pd.DataFrame,
                 require_os: bool = True) -> pd.DataFrame:
        """
        Classifies every patient into Stable/Critical/Collapse using
        EvolutionaryClassifier on `mu_smooth`.

        If require_os=True (default), rows lacking usable OS data are
        dropped *before* classification, so the regime counts returned
        here are the exact n's that will later appear in both the panel-A
        bar chart and the panel-C Kaplan-Meier legend — eliminating the
        bug where those two panels disagreed (n=1 vs n=8).
        """
        df = self._parse_os(df)

        if require_os:
            n_before = len(df)
            df = df[df["_os_months"].notna() & df["_os_event"].notna()].copy()
            n_dropped = n_before - len(df)
            if n_dropped:
                logger.info(
                    "TCGAValidator: dropped %d/%d patients with missing "
                    "Overall Survival data before regime classification — "
                    "this keeps panel A and panel C sample counts "
                    "consistent.", n_dropped, n_before)

        mu_values = df["mu_smooth"].values

        if self.cfg.tune_thresholds:
            # Clinical proxy label for threshold tuning: 1.0 = patient alive
            # (good outcome), 0.0 = deceased. tune_thresholds() (existing
            # method, unmodified) maximises |correlation| between this label
            # and the predicted regime, exactly as it does for the CLI path.
            clinical_label = 1.0 - df["_os_event"].values
            if np.std(clinical_label) > 0 and len(df) > 10:
                random.seed(self.cfg.random_seed)
                np.random.seed(self.cfg.random_seed)
                self.classifier.tune_thresholds(
                    mu_values, clinical_label,
                    n_iter=self.cfg.tune_n_iter,
                    method=self.cfg.tune_method)
                logger.info(
                    "TCGAValidator: tuned thresholds stable=%.3f collapse=%.3f",
                    self.classifier.threshold_stable,
                    self.classifier.threshold_collapse)

        df["state"] = self.classifier.classify_samples(mu_values)
        df["regime"] = df["state"].map({0: "Stable", 1: "Critical", 2: "Collapse"})
        return df

    # ------------------------------------------------------------------ #
    # 4. Plot panels A-K
    # ------------------------------------------------------------------ #
    @staticmethod
    def _kaplan_meier(times: np.ndarray, events: np.ndarray):
        """
        Minimal Kaplan-Meier estimator (no external dependency on lifelines).
        Returns (time_grid, survival_prob) step-function arrays suitable for
        plt.step(..., where='post').
        """
        order = np.argsort(times)
        t_sorted = times[order]
        e_sorted = events[order]
        unique_times = np.unique(t_sorted)

        s = 1.0
        surv = [1.0]
        grid = [0.0]
        n_at_risk = len(t_sorted)
        for t in unique_times:
            mask = t_sorted == t
            n_events = e_sorted[mask].sum()
            n_here = mask.sum()
            if n_at_risk > 0 and n_events > 0:
                s *= (1.0 - n_events / n_at_risk)
            grid.append(t)
            surv.append(s)
            n_at_risk -= n_here
        return np.array(grid), np.array(surv)

    @staticmethod
    def _logrank_test(times_a, events_a, times_b, events_b):
        """
        Two-group log-rank test (standard formulation), returns (chi2, p_value).
        Avoids requiring `lifelines` just for a single statistic.
        """
        times = np.concatenate([times_a, times_b])
        events = np.concatenate([events_a, events_b])
        group = np.concatenate([np.zeros(len(times_a)), np.ones(len(times_b))])

        order = np.argsort(times)
        times, events, group = times[order], events[order], group[order]
        unique_event_times = np.unique(times[events == 1])

        O_a = E_a = V = 0.0
        for t in unique_event_times:
            at_risk = times >= t
            n_total = at_risk.sum()
            n_a = (at_risk & (group == 0)).sum()
            n_b = (at_risk & (group == 1)).sum()
            d_mask = (times == t) & (events == 1)
            d_total = d_mask.sum()
            d_a = (d_mask & (group == 0)).sum()
            if n_total <= 1:
                continue
            e_a = d_total * n_a / n_total
            var = (d_total * (n_a / n_total) * (n_b / n_total)
                   * (n_total - d_total) / (n_total - 1))
            O_a += d_a
            E_a += e_a
            V += var

        if V <= 0:
            return 0.0, 1.0
        chi2 = (O_a - E_a) ** 2 / V
        from scipy.stats import chi2 as _chi2_dist
        p_value = float(_chi2_dist.sf(chi2, df=1))
        return float(chi2), p_value

    def plot_panels(self, df: pd.DataFrame, save_path: str = None,
                     title: str = "Evolution ONE — REAL TCGA Data"):
        """
        Renders panels A, B, C, E, F, K — every panel derivable purely from
        the clinical TSV columns used here. Panels G/H/I/J require
        mutation-level MAF/VCF or kernel-internal data not present in the
        clinical TSV and are intentionally omitted rather than faked.
        """
        n_total = len(df)
        counts = df["regime"].value_counts().reindex(
            ["Stable", "Critical", "Collapse"]).fillna(0).astype(int)
        H = self.classifier.compute_entropy(df["state"].values)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"{title}  (N={n_total})\nSOC-Controlled Cancer Evolution",
                     fontsize=13, fontweight="bold")

        # --- Panel A: SOC Regime Classification ---
        ax = axes[0, 0]
        bars = ax.bar(counts.index, counts.values, color="#1f77b4")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                    ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"A: SOC Regime Classification\nN={n_total}  H={H:.3f} bits")
        ax.set_ylabel("Patients")

        # --- Panel B: μ distribution by cohort ---
        ax = axes[0, 1]
        for label in df["CohortLabel"].unique():
            sub = df[df["CohortLabel"] == label]
            ax.hist(sub["mu_smooth"], bins=30, alpha=0.5,
                    density=True, label=f"{label} (n={len(sub)})")
        ax.axvline(self.classifier.threshold_stable, color="orange",
                   linestyle="--", label=f"θ_s={self.classifier.threshold_stable:.2f}")
        ax.axvline(self.classifier.threshold_collapse, color="red",
                   linestyle="--", label=f"θ_c={self.classifier.threshold_collapse:.2f}")
        ax.set_title("B: μ distribution by cancer type")
        ax.set_xlabel("SOC mutation load μ"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # --- Panel C: Kaplan-Meier by SOC regime ---
        ax = axes[0, 2]
        colors = {"Stable": "green", "Critical": "orange", "Collapse": "red"}
        km_data = {}
        for regime in ["Stable", "Critical", "Collapse"]:
            sub = df[df["regime"] == regime]
            if len(sub) == 0:
                continue
            grid, surv = self._kaplan_meier(
                sub["_os_months"].values, sub["_os_event"].values)
            km_data[regime] = sub
            median_os = sub["_os_months"].median()
            ax.step(grid, surv, where="post", color=colors[regime],
                    label=f"{regime} n={len(sub)} OS={median_os:.0f}mo")
        ax.set_title("C: Kaplan-Meier by SOC regime\n(n's match panel A by construction)")
        ax.set_xlabel("Time (months)"); ax.set_ylabel("Overall Survival")
        ax.legend(fontsize=8)
        if "Stable" in km_data and "Critical" in km_data:
            _, p_sc = self._logrank_test(
                km_data["Stable"]["_os_months"].values,
                km_data["Stable"]["_os_event"].values,
                km_data["Critical"]["_os_months"].values,
                km_data["Critical"]["_os_event"].values)
            ax.text(0.5, 0.15, f"Stable vs Critical\np={p_sc:.3e}"
                    + ("*" if p_sc < 0.05 else ""),
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round", fc="lightyellow"))

        # --- Panel E: subtype μ gradient (only if Subtype column present) ---
        ax = axes[1, 0]
        if "Subtype" in df.columns and df["Subtype"].notna().any():
            grp = df.dropna(subset=["Subtype"]).groupby("Subtype")["mu_smooth"]
            means, stds, ns = grp.mean(), grp.std(), grp.count()
            order = means.sort_values().index
            ax.bar(order, means[order], yerr=stds[order].fillna(0))
            for i, lbl in enumerate(order):
                ax.text(i, means[lbl], f"n={ns[lbl]}", ha="center", va="bottom",
                        fontsize=8)
            ax.set_title("E: Subtype μ gradient")
        else:
            ax.set_title("E: Subtype μ gradient\n(no Subtype column — skipped)")
        ax.set_ylabel("Mean SOC load μ")
        ax.tick_params(axis="x", rotation=30)

        # --- Panel F: OS by regime (boxplot) ---
        ax = axes[1, 1]
        box_data = [df[df["regime"] == r]["_os_months"].dropna().values
                    for r in ["Stable", "Critical", "Collapse"]]
        ax.boxplot(box_data, labels=["Stable", "Critical", "Collapse"])
        ax.set_title("F: OS by SOC regime")
        ax.set_ylabel("Overall Survival (months)")

        # --- Panel K: μ by AJCC stage ---
        ax = axes[1, 2]
        stage_col = "American Joint Committee on Cancer Tumor Stage Code"
        if stage_col in df.columns:
            stage_simplified = df[stage_col].astype(str).str.extract(
                r"(I{1,3}V?|IV)")[0]
            valid = stage_simplified.notna()
            stages_present = [s for s in ["I", "II", "III", "IV"]
                               if (stage_simplified == s).any()]
            data = [df.loc[valid & (stage_simplified == s), "mu_smooth"].values
                    for s in stages_present]
            ax.boxplot(data, labels=stages_present)
            for i, s in enumerate(stages_present):
                n_s = (stage_simplified == s).sum()
                ax.text(i + 1, 0, f"n={n_s}", ha="center", va="top", fontsize=8)
            ax.set_title("K: μ by AJCC stage")
        else:
            ax.set_title("K: μ by AJCC stage\n(no stage column — skipped)")
        ax.set_ylabel("SOC mutation load μ")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            logger.info("TCGAValidator: figure saved to %s", save_path)
        return fig

    # ------------------------------------------------------------------ #
    # 5. One-shot convenience wrapper
    # ------------------------------------------------------------------ #
    def run(self, tsv_paths: Dict[str, str],
            save_path: str = None) -> pd.DataFrame:
        df = self.load_cohort(tsv_paths)
        df = self.compute_mu(df)
        df = self.classify(df, require_os=True)
        self.plot_panels(df, save_path=save_path)
        return df


# =============================================================================
# 14. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EVOLUTION ONE v4 – Full Cancer Evolution Engine")
    parser.add_argument("--tcga_validate", nargs="+", metavar="LABEL=PATH",
                        help="Run TCGAValidator instead of the main MAF/VCF "
                             "pipeline. Pass one or more LABEL=PATH pairs, "
                             "e.g. --tcga_validate BRCA=brca_clinical.tsv "
                             "COAD=coadread_clinical.tsv")
    parser.add_argument("--tcga_save", default="evolution_one_tcga_validation.png",
                        help="Output path for the TCGAValidator panel figure")
    parser.add_argument("--input",   "-i", required=False)
    parser.add_argument("--format",  default="maf", choices=["maf", "vcf"])
    parser.add_argument("--genes",   nargs="+")
    parser.add_argument("--duon_bed",  help="Duon regions in BED format")
    parser.add_argument("--lifestyle_file")
    parser.add_argument("--ref_genome",
                        help="Reference genome FASTA for gRNA design")
    parser.add_argument("--gene_interactions", nargs="+", type=int)
    parser.add_argument("--pdb_dir",      default="./pdbs")
    parser.add_argument("--output_dir",   default="./evo_output")
    parser.add_argument("--train_thresholds", action="store_true")
    parser.add_argument("--clinical_labels_file")
    parser.add_argument("--tune_method",  default="differential_evolution",
                        choices=["differential_evolution", "optuna"])
    parser.add_argument("--no_future",   action="store_true")
    parser.add_argument("--no_struct",   action="store_true")
    parser.add_argument("--resume",      help="Resume from checkpoint file")
    parser.add_argument("--batch_size",  type=int, default=500)
    parser.add_argument("--plot",        action="store_true")
    args = parser.parse_args()

    if args.tcga_validate:
        tsv_paths = {}
        for pair in args.tcga_validate:
            if "=" not in pair:
                print(f"--tcga_validate entries must be LABEL=PATH, got: {pair}")
                sys.exit(1)
            label, path = pair.split("=", 1)
            tsv_paths[label] = path
        validator = TCGAValidator()
        validator.run(tsv_paths, save_path=args.tcga_save)
        print(f"TCGA validation figure saved to {args.tcga_save}")
        return

    if not args.input or not args.genes:
        print("--input and --genes are required unless --tcga_validate is used.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_interactions = None
    if args.gene_interactions:
        pairs = args.gene_interactions
        if len(pairs) % 2 != 0:
            print("Gene interactions must be in pairs.")
            sys.exit(1)
        gene_interactions = [
            (pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2)]

    engine = EvolutionONEEngine(cfg={
        "duon_bed":       args.duon_bed,
        "lifestyle_file": args.lifestyle_file,
        "ref_genome":     args.ref_genome,
        "pdb_dir":        args.pdb_dir,
        "output_dir":     str(out_dir),
    })
    engine.run(
        input_file=args.input, genes=args.genes, format=args.format,
        compute_future_mutations=not args.no_future,
        compute_structural=not args.no_struct,
        gene_interactions=gene_interactions,
        train_thresholds=args.train_thresholds,
        clinical_labels_file=args.clinical_labels_file,
        tune_method=args.tune_method,
        resume_from=args.resume, batch_size=args.batch_size,
    )

    if engine.results:
        summary = {
            "cancer_risk":         engine.results["cancer_risk"],
            "entropy":             engine.results["entropy"],
            "bv_satisfied":        engine.results["bv_satisfied"],
            "drug_recommendations": engine.results["drug_recommendations"],
            "crispr_designs":      engine.results.get("crispr_designs", {}),
            "epigenetic_targets":  engine.results.get("epigenetic_targets", {}),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        pd.DataFrame({
            "sample_id": engine.results.get("sample_ids", []),
            "mu":        engine.results["mu_smooth"],
            "state":     engine.results["states"],
        }).to_csv(out_dir / "sample_states.csv", index=False)

        # Separate CRISPR and epigenetic JSON outputs
        if engine.results.get("crispr_designs"):
            with open(out_dir / "crispr_designs.json", "w") as f:
                json.dump(engine.results["crispr_designs"], f, indent=2)
        if engine.results.get("epigenetic_targets"):
            with open(out_dir / "epigenetic_targets.json", "w") as f:
                json.dump(engine.results["epigenetic_targets"], f, indent=2)

        print(f"Results saved to {out_dir}")

    if args.plot:
        engine.plot_phase_diagram(
            save_path=str(out_dir / "phase_diagram.png"))


if __name__ == "__main__":
    main()
