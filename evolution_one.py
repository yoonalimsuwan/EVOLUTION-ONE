# =============================================================================
# Evolution ONE – Standalone Multi‑Level Cancer Evolution Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# This Software is fully self‑contained. It can optionally import REAL FOLD ONE
# and REAL FOLD ONE HT for enhanced performance, but all core routines are
# embedded and will work out‑of‑the‑box.
# =============================================================================

import math, os, sys, json, argparse, logging, warnings, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# -----------------------------------------------------------------------------
# Optional imports (graceful degradation)
# -----------------------------------------------------------------------------
try:
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.mmcif as mmcif_io
    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False

# Try to import REAL FOLD ONE and HT – if not available, embedded fallbacks are used
try:
    from real_fold_one import (
        RefinementEngine, RefinementConfig,
        CSOCKernel, SOCController, SemanticStateContraction, DiffRGRefiner,
        NeighborListManager, PME, MultigridPoisson, BlockwiseLongRange,
        reconstruct_backbone, build_sidechain_atoms,
        get_full_atom_coords_and_types,
        energy_bond, energy_angle, energy_rama, energy_clash,
        energy_electro, energy_solvent, energy_hbond,
        energy_lj_full, energy_coulomb_full, energy_torsion_chi,
        DEFAULT_LJ_PARAMS, DEFAULT_CHARGE_MAP,
        AA_3_TO_1, RESIDUE_CHARGE, MAX_CHI, RESIDUE_NCHI,
        load_structure, save_structure,
        ItoProcess, LangevinDynamics,
        BVFieldTheory, DNAOrigamiBV,
        HAS_BIOTITE as RFO_HAS_BIOTITE,
    )
    HAS_REAL_FOLD_ONE = True
except ImportError:
    HAS_REAL_FOLD_ONE = False

try:
    from real_fold_one_ht import HighThroughputScanner, HTConfig
    HAS_HT = True
except ImportError:
    HAS_HT = False

warnings.filterwarnings("ignore")
logger = logging.getLogger("EvolutionONE")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

# =============================================================================
# 1. Embedded physics engine (fallback when REAL FOLD ONE is not installed)
# =============================================================================
if not HAS_REAL_FOLD_ONE:

    def _normalize(x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
    AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}
    AA_3_TO_1 = {
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
        'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
        'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','UNK':'X'
    }
    HYDROPHOBICITY = {
        'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
        'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,
        'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3,'X':0.0
    }
    RESIDUE_CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.5}
    RAMACHANDRAN_PRIORS = {
        'general':{'phi':-60.0,'psi':-45.0,'width':25.0},
        'G':{'phi':-75.0,'psi':-60.0,'width':40.0},
        'P':{'phi':-65.0,'psi':-30.0,'width':20.0},
    }
    MAX_CHI = 4
    RESIDUE_NCHI = {
        'A':0,'G':0,'S':1,'C':1,'V':1,'T':1,'L':2,'I':2,'M':3,'F':2,'Y':2,'W':2,
        'D':2,'E':3,'N':2,'Q':3,'K':4,'R':4,'H':2,'P':2
    }

    # AMBER ff14SB‑like Lennard‑Jones (sigma [Å], epsilon [kcal/mol])
    AMBER_LJ = {
        'C':(1.9080,0.0860),'CA':(1.9080,0.0860),'CB':(1.9080,0.0860),
        'CG':(1.9080,0.0860),'CD':(1.9080,0.0860),'CE':(1.9080,0.0860),
        'CZ':(1.9080,0.0860),'CH2':(1.9080,0.0860),
        'N':(1.8240,0.1700),'ND':(1.8240,0.1700),'NE':(1.8240,0.1700),
        'NH1':(1.8240,0.1700),'NH2':(1.8240,0.1700),
        'O':(1.6612,0.2100),'OD':(1.6612,0.2100),'OE':(1.6612,0.2100),
        'OH':(1.6612,0.2100),'S':(2.0000,0.2500),'SG':(2.0000,0.2500),
        'CT':(1.9080,0.1094),'C*':(1.9080,0.0860),'N2':(1.8240,0.1700),
        'N3':(1.8240,0.1700),'OW':(1.7683,0.1520),'HW':(0.6000,0.0157),
        'H':(0.6000,0.0157),'HC':(1.4870,0.0157),'H1':(1.3870,0.0157),
        'HP':(1.1000,0.0157),'HS':(0.6000,0.0157),
        'P':(2.1000,0.2000),'OP':(1.8500,0.1500),
    }
    AMBER_CHARGES = {
        'N':-0.4157,'CA':0.2719,'C':0.5973,'O':-0.5679,
        'CB':0.0,'CG':0.0,'CD':0.0,'CE':0.0,'CZ':0.0,
        'ND':-0.4157,'NE':-0.4157,'NH1':-0.4157,'NH2':-0.4157,
        'OD':-0.5679,'OE':-0.5679,'OH':-0.5679,'SG':-0.2344,'S':-0.2344,
    }
    DEFAULT_LJ_PARAMS = AMBER_LJ
    DEFAULT_CHARGE_MAP = AMBER_CHARGES

    # ── SOC / SSC / RG (embedded) ─────────────────────────────────────
    class CSOCKernel(nn.Module):
        def __init__(self, init_alpha=0.5, init_lambda=12.0, eps=1e-4):
            super().__init__()
            self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
            self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda)))
            self.eps = eps
        @property
        def alpha(self): return torch.exp(self.log_alpha)
        @property
        def lambd(self): return torch.exp(self.log_lambda)
        def forward(self, r):
            safe_r = r + self.eps
            return torch.exp(-self.log_alpha * torch.log(safe_r)) * torch.exp(-r / self.lambd)

    class SemanticStateContraction:
        def __init__(self, epsilon_fp=0.0028, sigma_target=1.0):
            self.eps = epsilon_fp
            self.target = sigma_target
            self.prev = None
        def __call__(self, sigma):
            if self.prev is None:
                self.prev = sigma
                return sigma
            new = self.prev + self.eps * (sigma - self.prev)
            self.prev = new
            return new

    class SOCController:
        def __init__(self, base_temp=300.0, friction=0.02, sigma_target=1.0,
                     avalanche_threshold=0.5, w_avalanche=0.2, kernel=None,
                     use_ssc=True, epsilon_fp=0.0028, adaptive_friction=True):
            self.prev_coords = None
            self.base_temp = base_temp
            self.friction = friction
            self.sigma_target = sigma_target
            self.avalanche_threshold = avalanche_threshold
            self.w_avalanche = w_avalanche
            self.kernel = kernel if kernel else CSOCKernel()
            self.use_ssc = use_ssc
            self.ssc = SemanticStateContraction(epsilon_fp, sigma_target) if use_ssc else None
            self.adaptive_friction = adaptive_friction
            self.per_atom_friction = None
        def sigma(self, coords):
            if self.prev_coords is None:
                self.prev_coords = coords.detach().clone()
                return torch.tensor(1.0, device=coords.device)
            delta = torch.norm(coords - self.prev_coords, dim=-1).mean()
            self.prev_coords = coords.detach().clone()
            if self.use_ssc and self.ssc:
                delta = self.ssc(delta)
            return delta
        def temperature(self, sigma):
            dev = (sigma - self.sigma_target) / 0.5
            T = self.base_temp + 2000.0 * torch.sigmoid(dev)
            return torch.clamp(T, self.base_temp * 0.5, 3000.0)
        def compute_soc_energy(self, ca, alpha, edge_idx, edge_dist, mask=None, w_soc=0.3):
            if edge_idx.numel() == 0: return torch.tensor(0.0, device=ca.device)
            src, dst = edge_idx[0], edge_idx[1]
            if mask is not None:
                keep = mask[src] & mask[dst]
                if not keep.any(): return torch.tensor(0.0, device=ca.device)
                src, dst = src[keep], dst[keep]
                d = edge_dist[keep]
            else:
                d = edge_dist
            a = 0.5 * (alpha[src] + alpha[dst])
            K = self.kernel(d)
            E = -K * torch.exp(-d / 8.0)
            return w_soc * E.mean()

    class DiffRGRefiner:
        def __init__(self, factor=4, n_levels=2):
            self.factor = factor; self.n_levels = n_levels
        def forward(self, coords):
            L = coords.shape[0]
            for _ in range(self.n_levels):
                f = self.factor
                m = L // f * f
                if m == 0: break
                x = coords[:m].permute(1,0).unsqueeze(0)
                pooled = F.avg_pool1d(x, kernel_size=f, stride=f)
                up = F.interpolate(pooled, size=L, mode='linear', align_corners=True)
                coords = up.squeeze(0).permute(1,0)
            return coords

    # ── Ito / BV (embedded) ──────────────────────────────────────────
    class ItoProcess:
        def __init__(self, dim, drift, diffusion, dt=1e-3, device='cpu'):
            self.dim = dim; self.drift = drift; self.diffusion = diffusion
            self.dt = dt; self.device = device
        def euler_maruyama_step(self, x):
            dW = torch.randn_like(x) * math.sqrt(self.dt)
            sigma = self.diffusion(x)
            if sigma.dim() == 1:
                return x + self.drift(x) * self.dt + sigma * dW
            else:
                return x + self.drift(x) * self.dt + (sigma @ dW.unsqueeze(-1)).squeeze(-1)
        def milstein_step(self, x):
            dW = torch.randn_like(x) * math.sqrt(self.dt)
            sigma = self.diffusion(x); b = self.drift(x)
            if sigma.dim() == 1:
                x_temp = x.detach().requires_grad_(True)
                sigma_val = self.diffusion(x_temp)
                grad_sigma = torch.autograd.grad(sigma_val.sum(), x_temp, allow_unused=True)[0]
                if grad_sigma is not None:
                    correction = 0.5 * sigma * grad_sigma * (dW**2 - self.dt)
                else:
                    correction = torch.zeros_like(x)
                return x + b * self.dt + sigma * dW + correction
            else:
                return x + b * self.dt + (sigma @ dW.unsqueeze(-1)).squeeze(-1)

    class LangevinDynamics(ItoProcess):
        def __init__(self, energy_fn, gamma=0.02, T=300.0, dt=1e-3, device='cpu'):
            self.energy_fn = energy_fn; self.gamma = gamma; self.T = T
            self.kB = 1.987e-3
            def drift(x):
                x.requires_grad_(True)
                E = energy_fn(x); grad = torch.autograd.grad(E, x)[0]
                return -grad / gamma
            def diffusion(x):
                return math.sqrt(2 * self.kB * T / gamma) * torch.ones_like(x)
            super().__init__(dim=0, drift=drift, diffusion=diffusion, dt=dt, device=device)
        def step(self, x, scheme='milstein'):
            return self.milstein_step(x) if scheme == 'milstein' else self.euler_maruyama_step(x)

    class BVFieldTheory:
        def __init__(self, field_names, ghost_numbers):
            self.fields = field_names
            self.ghost_numbers = {name: gh for name, gh in zip(field_names, ghost_numbers)}
            self.phi = {name: torch.tensor(0.0) for name in field_names}
            self.phi_star = {name: torch.tensor(0.0) for name in field_names}
        def antibracket(self, F, G):
            phi = {k: v.clone().detach().requires_grad_(True) for k,v in self.phi.items()}
            phistar = {k: v.clone().detach().requires_grad_(True) for k,v in self.phi_star.items()}
            F_val = F(phi, phistar); G_val = G(phi, phistar)
            dF_dphi = torch.autograd.grad(F_val, list(phi.values()), retain_graph=True, create_graph=True)
            dF_dphistar = torch.autograd.grad(F_val, list(phistar.values()), retain_graph=True, create_graph=True)
            dG_dphi = torch.autograd.grad(G_val, list(phi.values()), retain_graph=True, create_graph=True)
            dG_dphistar = torch.autograd.grad(G_val, list(phistar.values()), retain_graph=True, create_graph=True)
            result = 0.0
            for i, name in enumerate(self.fields):
                result += torch.dot(dF_dphi[i].flatten(), dG_dphistar[i].flatten())
                result -= torch.dot(dF_dphistar[i].flatten(), dG_dphi[i].flatten())
            return result
        def classical_master_equation(self, S):
            return torch.allclose(self.antibracket(S, S), torch.tensor(0.0), atol=1e-6)

    # ── Backbone & Sidechain (embedded) ──────────────────────────────
    def reconstruct_backbone(ca):
        L = ca.shape[0]; device = ca.device
        N = torch.zeros_like(ca); C = torch.zeros_like(ca); O = torch.zeros_like(ca)
        for i in range(L):
            if i < L-1:
                v_next = ca[i+1] - ca[i]
                n_dir = ca[i-1] - ca[i] if i > 0 else -v_next
                c_dir = v_next
            else:
                v_prev = ca[i-1] - ca[i]
                n_dir = v_prev; c_dir = -v_prev
            n_dir = _normalize(n_dir); c_dir = _normalize(c_dir)
            N[i] = ca[i] + 1.45 * n_dir
            C[i] = ca[i] + 1.52 * c_dir
            ca_c = C[i] - ca[i]; ca_n = N[i] - ca[i]
            norm_vec = _normalize(torch.cross(ca_c, ca_n, dim=-1))
            O[i] = C[i] + 1.23 * norm_vec
        return {'N': N, 'CA': ca, 'C': C, 'O': O}

    def dihedral_angle(p0,p1,p2,p3):
        b0=p1-p0; b1=p2-p1; b2=p3-p2
        b1n = _normalize(b1)
        v = b0 - (b0*b1n).sum(-1,keepdim=True)*b1n
        w = b2 - (b2*b1n).sum(-1,keepdim=True)*b1n
        x = (v*w).sum(-1)
        y = torch.cross(b1n, v, dim=-1); y = (y*w).sum(-1)
        return torch.atan2(y+1e-8, x+1e-8)

    def compute_phi_psi(atoms):
        N,CA,C = atoms['N'],atoms['CA'],atoms['C']
        L = CA.shape[0]; device = CA.device
        phi = torch.zeros(L, device=device); psi = torch.zeros(L, device=device)
        if L > 2:
            phi[1:-1] = dihedral_angle(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
            psi[1:-1] = dihedral_angle(N[1:-1], CA[1:-1], C[1:-1], N[2:])
        return phi * 180.0 / math.pi, psi * 180.0 / math.pi

    RESIDUE_TOPOLOGY = {
        'G':[], 'A':[],
        'S':[('OG','OH',1,1.43,109.5,(-2,-1,0),0.0)],
        'C':[('SG','SG',1,1.81,109.5,(-2,-1,0),0.0)],
        'V':[('CG1','CB',1,1.53,109.5,(-2,-1,0),0.0), ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0)],
        'T':[('OG1','OH',1,1.43,109.5,(-2,-1,0),0.0), ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0)],
        'L':[('CG','CB',1,1.53,109.5,(-2,-1,0),0.0), ('CD1','CB',2,1.53,109.5,(-1,0,1),0.0), ('CD2','CB',2,1.53,109.5,(-1,0,1),2.0)],
        'I':[('CG1','CB',1,1.53,109.5,(-2,-1,0),0.0), ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0), ('CD1','CB',2,1.53,109.5,(-1,0,1),0.0)],
        'M':[('CG','CB',1,1.53,109.5,(-2,-1,0),0.0), ('SD','S',2,1.81,109.5,(-1,0,1),0.0), ('CE','CB',3,1.81,109.5,(-2,-1,0),0.0)],
        'F':[('CG','CB',1,1.53,109.5,(-2,-1,0),0.0), ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0), ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0), ('CE1','CB',3,1.40,120.0,(2,1,0),0.0), ('CE2','CB',4,1.40,120.0,(2,1,0),2.0), ('CZ','CB',5,1.40,120.0,(3,2,1),0.0)],
        'Y':[('CG','CB',1,1.53,109.5,(-2,-1,0),0.0), ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0), ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0), ('CE1','CB',3,1.40,120.0,(2,1,0),0.0), ('CE2','CB',4,1.40,120.0,(2,1,0),2.0), ('CZ','CB',5,1.40,120.0,(3,2,1),0.0), ('OH','OH',6,1.36,120.0,(4,3,2),0.0)],
        'W':[('CG','CB',1,1.53,109.5,(-2,-1,0),0.0), ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0), ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0), ('NE1','N',3,1.38,120.0,(2,1,0),0.0), ('CE2','CB',4,1.40,120.0,(2,1,0),2.0), ('CE3','CB',5,1.40,120.0,(2,1,0),2.0), ('CZ2','CB',6,1.40,120.0,(3,2,1),0.0), ('CZ3','CB',7,1.40,120.0,(5,4,2),0.0), ('CH2','CB',8,1.40,120.0,(6,3,2),0.0)],
        'D':[('CG','C',1,1.52,109.5,(-2,-1,0),0.0), ('OD1','O',2,1.25,120.0,(-1,0,1),0.0), ('OD2','O',2,1.25,120.0,(-1,0,1),2.0)],
        'E':[('CG','CB',1,1.52,109.5,(-2,-1,0),0.0), ('CD','C',2,1.52,109.5,(-1,0,1),0.0), ('OE1','O',3,1.25,120.0,(2,1,0),0.0), ('OE2','O',3,1.25,120.0,(2,1,0),2.0)],
        'N':[('CG','C',1,1.52,109.5,(-2,-1,0),0.0), ('OD1','O',2,1.25,120.0,(-1,0,1),0.0), ('ND2','N',2,1.33,120.0,(-1,0,1),2.0)],
        'Q':[('CG','CB',1,1.52,109.5,(-2,-1,0),0.0), ('CD','C',2,1.52,109.5,(-1,0,1),0.0), ('OE1','O',3,1.25,120.0,(2,1,0),0.0), ('NE2','N',3,1.33,120.0,(2,1,0),2.0)],
        'K':[('CG','CB',1,1.52,109.5,(-2,-1,0),0.0), ('CD','CB',2,1.52,109.5,(-1,0,1),0.0), ('CE','CB',3,1.52,109.5,(-2,-1,0),0.0), ('NZ','N',4,1.47,109.5,(-1,0,1),0.0)],
        'R':[('CG','CB',1,1.52,109.5,(-2,-1,0),0.0), ('CD','CB',2,1.52,109.5,(-1,0,1),0.0), ('NE','N',3,1.46,109.5,(-2,-1,0),0.0), ('CZ','C',4,1.33,125.0,(-1,0,1),0.0), ('NH1','N',5,1.33,120.0,(4,3,2),0.0), ('NH2','N',5,1.33,120.0,(4,3,2),2.0)],
        'H':[('CG','CB',1,1.50,109.5,(-2,-1,0),0.0), ('ND1','N',2,1.38,120.0,(-1,0,1),0.0), ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0), ('CE1','CB',3,1.40,120.0,(2,1,0),0.0), ('NE2','N',4,1.38,120.0,(2,1,0),2.0)],
        'P':[('CG','CB',1,1.50,104.5,(-2,-1,0),0.0), ('CD','CB',2,1.50,104.5,(-1,0,1),0.0)],
    }

    def _map_ref(idx):
        return { -1:0, -2:1, -3:2, -4:3 }.get(idx, 3+idx)

    def build_sidechain_atoms(ca, seq, chi_angles):
        device = ca.device; L = ca.shape[0]
        v = ca[1:] - ca[:-1]; vn = _normalize(v)
        N = torch.zeros_like(ca); C = torch.zeros_like(ca)
        N[1:] = ca[1:] - 1.45 * vn; N[0] = ca[0] - 1.45 * vn[0]
        C[:-1] = ca[:-1] + 1.52 * vn; C[-1] = ca[-1] + 1.52 * vn[-1]
        all_coords, all_types = [], []
        for i, aa in enumerate(seq):
            if aa not in RESIDUE_TOPOLOGY:
                all_coords.append(torch.stack([N[i], ca[i], C[i]]))
                all_types.append(['N','CA','C'])
                continue
            n_i, ca_i, c_i = N[i], ca[i], C[i]
            v1 = n_i - ca_i; v2 = c_i - ca_i
            cb_dir = -(v1+v2); cb_dir = _normalize(cb_dir)
            cb_pos = ca_i + 1.53 * cb_dir
            local_atoms = [n_i, ca_i, c_i, cb_pos]
            local_types = ['N','CA','C','CB']
            topo = RESIDUE_TOPOLOGY.get(aa, [])
            chi_idx = 0
            for (name, typ, parent, bond_len, ang_deg, ref, dih0) in topo:
                a = _map_ref(ref[0]); b = _map_ref(ref[1]); c = _map_ref(ref[2])
                a = min(a, len(local_atoms)-1); b = min(b, len(local_atoms)-1); c = min(c, len(local_atoms)-1)
                p_a, p_b, p_c = local_atoms[a], local_atoms[b], local_atoms[c]
                bc = p_c - p_b; bc_n = _normalize(bc)
                ref_vec = torch.tensor([1.,0.,0.], device=device)
                if abs(torch.dot(bc_n, ref_vec)) > 0.9:
                    ref_vec = torch.tensor([0.,1.,0.], device=device)
                perp = torch.cross(bc_n, ref_vec, dim=-1)
                if perp.norm() < 1e-12:
                    ref_vec = torch.tensor([0.,0.,1.], device=device)
                    perp = torch.cross(bc_n, ref_vec, dim=-1)
                perp = _normalize(perp)
                chi_val = chi_angles[i, chi_idx] if chi_angles is not None else 0.0
                total_ang = dih0 + chi_val
                cos_a, sin_a = math.cos(total_ang), math.sin(total_ang)
                cross_bn_perp = torch.cross(bc_n, perp, dim=-1)
                rotated_perp = perp * cos_a + cross_bn_perp * sin_a
                ang_rad = math.radians(ang_deg)
                bond_dir = math.cos(ang_rad)*bc_n + math.sin(ang_rad)*rotated_perp
                new_pos = p_c + bond_len * bond_dir
                local_atoms.append(new_pos); local_types.append(name)
                chi_idx += 1
            all_coords.append(torch.stack(local_atoms))
            all_types.append(local_types)
        return all_coords, all_types

    def get_full_atom_coords_and_types(ca, seq, chi_angles):
        res_coords, res_types = build_sidechain_atoms(ca, seq, chi_angles)
        if not res_coords:
            return torch.empty((0,3), device=ca.device), [], torch.empty(0, device=ca.device)
        coords_list = [rc for rc in res_coords]
        types_list = sum(res_types, [])
        res_idx = torch.cat([torch.full((rc.shape[0],), i, dtype=torch.long, device=ca.device) for i,rc in enumerate(res_coords)])
        return torch.cat(coords_list, dim=0), types_list, res_idx

    # ── Energy terms (embedded) ───────────────────────────────────────
    def energy_bond(ca, alpha, w=30.0, mod=0.1, target=3.8, mask=None):
        if mask is not None and not mask.any(): return torch.tensor(0.0, device=ca.device)
        t = target * (1.0 + mod*(alpha-1.0))
        t_pair = 0.5*(t[1:]+t[:-1])
        d = torch.norm(ca[1:]-ca[:-1], dim=-1)
        if mask is not None:
            bm = mask[1:] & mask[:-1]
            if bm.sum()==0: return torch.tensor(0.0, device=ca.device)
            return w * ((d[bm]-t_pair[bm])**2).mean()
        return w * ((d-t_pair)**2).mean()

    def energy_angle(ca, alpha, w=15.0, mod=0.05, target_rad=111.0*math.pi/180.0, mask=None):
        if len(ca)<3: return torch.tensor(0.0, device=ca.device)
        v1=ca[:-2]-ca[1:-1]; v2=ca[2:]-ca[1:-1]
        v1n=_normalize(v1); v2n=_normalize(v2)
        cos_a = (v1n*v2n).sum(-1)
        t = target_rad * (1.0 + mod*(alpha[1:-1]-1.0))
        cos_t = torch.cos(t)
        if mask is not None:
            am = mask[1:-1]
            if am.sum()==0: return torch.tensor(0.0, device=ca.device)
            return w * ((cos_a[am]-cos_t[am])**2).mean()
        return w * ((cos_a-cos_t)**2).mean()

    def energy_rama(phi, psi, seq, alpha, w=8.0, mod=0.2, mask=None):
        L=len(seq); device=phi.device
        p0=torch.zeros(L,device=device); p1=torch.zeros(L,device=device); wd=torch.zeros(L,device=device)
        for i,aa in enumerate(seq):
            pr=RAMACHANDRAN_PRIORS.get(aa,RAMACHANDRAN_PRIORS['general'])
            p0[i],p1[i],wd[i]=pr['phi'],pr['psi'],pr['width']
        weff = wd*(1.0+mod*(alpha-1.0))
        dphi=(phi-p0)/(weff+1e-8); dpsi=(psi-p1)/(weff+1e-8)
        loss=(dphi**2+dpsi**2)
        if mask is None: mask=torch.ones(L,device=device,dtype=torch.bool)
        mask[0]=mask[-1]=False
        if mask.sum()==0: return torch.tensor(0.0, device=device)
        return w * (loss*mask.float()).sum()/mask.sum()

    def energy_clash(ca, alpha, edge_idx, edge_dist, w=80.0, radius=2.0, mod=0.1, mask=None):
        if edge_idx.numel()==0: return torch.tensor(0.0, device=ca.device)
        i,j = edge_idx[0], edge_idx[1]
        keep = (torch.abs(i-j)>2)
        if not keep.any(): return torch.tensor(0.0, device=ca.device)
        i,j = i[keep], j[keep]; d = edge_dist[keep]
        ri = radius * (1.0+mod*(alpha[i]-1.0))
        rj = radius * (1.0+mod*(alpha[j]-1.0))
        rad = 0.5*(ri+rj)
        clash = torch.relu(rad - d)
        if clash.numel()==0: return torch.tensor(0.0, device=ca.device)
        return w * (clash**2).mean()

    def energy_electro(ca, seq, edge_idx, edge_dist, w=4.0, mask=None):
        if edge_idx.numel()==0: return torch.tensor(0.0, device=ca.device)
        q = torch.tensor([RESIDUE_CHARGE.get(a,0.0) for a in seq], device=ca.device)
        qi,qj = q[edge_idx[0]], q[edge_idx[1]]
        r = torch.clamp(edge_dist, min=1e-6)
        E = qi*qj * torch.exp(-0.1*r) / (80.0 * r)
        return w * E.mean()

    def energy_solvent(ca, seq, edge_idx, w=5.0, mask=None):
        if edge_idx.numel()==0: return torch.tensor(0.0, device=ca.device)
        src = edge_idx[0]
        cnt = torch.zeros(ca.shape[0], device=ca.device)
        cnt.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        burial = 1.0 - torch.exp(-cnt/20.0)
        hydro = torch.tensor([HYDROPHOBICITY.get(a,0.0) for a in seq], device=ca.device)
        exposed = torch.where(hydro>0, hydro*(1.0-burial), torch.zeros_like(burial))
        buried = torch.where(hydro<=0, -hydro*burial, torch.zeros_like(burial))
        return w * (exposed+buried).mean()

    def energy_hbond(O,N,C,alpha, edge_idx, edge_dist, w=6.0, mod=0.1, mask=None):
        if edge_idx.numel()==0: return torch.tensor(0.0, device=O.device)
        src,dst = edge_idx[0], edge_idx[1]
        d = edge_dist
        vec_co = O[src]-C[src]; vec_no = N[dst]-O[src]
        align = F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
        ideal = 2.9 * (1.0+mod*(alpha[src]-1.0))
        E = -align * torch.exp(-((d-ideal)/0.3)**2)
        return w * E.mean()

    def energy_lj_full(all_coords, all_types, res_idx, edge_idx, edge_dist, lj_params, w=30.0):
        if edge_idx.numel()==0: return torch.tensor(0.0, device=all_coords.device)
        src,dst = edge_idx
        sig = torch.zeros(len(all_types), device=all_coords.device)
        eps = torch.zeros(len(all_types), device=all_coords.device)
        for i,t in enumerate(all_types):
            s,e = lj_params.get(t,(1.9,0.1))
            sig[i]=s; eps[i]=e
        sig_ij = 0.5*(sig[src]+sig[dst])
        eps_ij = torch.sqrt(eps[src]*eps[dst])
        r = torch.clamp(edge_dist, min=1e-4)
        inv_r = 1.0/r
        lj = 4.0*eps_ij * ((sig_ij*inv_r)**12 - (sig_ij*inv_r)**6)
        return w * lj.mean()

    def energy_coulomb_full(all_coords, all_types, res_idx, edge_idx, edge_dist, charge_map, w=3.0):
        if edge_idx.numel()==0: return torch.tensor(0.0, device=all_coords.device)
        src,dst = edge_idx
        q = torch.tensor([charge_map.get(t,0.0) for t in all_types], device=all_coords.device)
        qi,qj = q[src], q[dst]
        r = torch.clamp(edge_dist, min=1e-4)
        dielectric = 4.0 * r
        coul = 332.0637 * qi * qj / (dielectric * r)
        return w * coul.mean()

    def energy_torsion_chi(chi_angles, seq, w=10.0):
        L=len(seq); maxc=chi_angles.shape[1]
        E=0.0
        for i,aa in enumerate(seq):
            nch = RESIDUE_NCHI.get(aa,0)
            for c in range(min(nch, maxc)):
                chi = chi_angles[i,c]
                E += 0.5*(1.0 - torch.cos(3.0*chi))
        return w * E / max(1,L)

    def alpha_regularisation(alpha, w_ent=0.5, w_smooth=0.1):
        ent = -(alpha * torch.log(alpha+1e-8)).mean()
        sm = ((alpha[1:]-alpha[:-1])**2).mean()
        return w_ent*ent + w_smooth*sm

    def chain_break_energy(ca, boundaries, w=1.0):
        if not boundaries: return torch.tensor(0.0, device=ca.device)
        E=0.0
        for b in boundaries:
            if 0 < b < len(ca):
                d = torch.norm(ca[b]-ca[b-1], dim=-1)
                E += w * torch.relu(d-5.0)
        return E

    # ── Neighbor List Manager (fallback) ──────────────────────────────
    class NeighborListManager:
        def __init__(self, cutoffs: Dict[str, float], max_neighbors=64, device='cpu'):
            self.cutoffs = cutoffs
            self.max_neighbors = max_neighbors
            self.device = device
        def build(self, coords, batch=None):
            if coords.shape[0] == 0:
                empty = torch.empty((2,0), dtype=torch.long, device=self.device)
                return {k: (empty, torch.empty(0, device=self.device)) for k in self.cutoffs}
            if batch is None:
                batch = torch.zeros(coords.shape[0], dtype=torch.long, device=self.device)
            result = {}
            for name, cutoff in self.cutoffs.items():
                N = coords.shape[0]
                src_list, dst_list, d_list = [], [], []
                chunk = 2000 if N > 5000 else N
                for i in range(0, N, chunk):
                    i_end = min(i+chunk, N)
                    ci = coords[i:i_end]
                    for j in range(0, N, chunk):
                        j_end = min(j+chunk, N)
                        cj = coords[j:j_end]
                        dd = torch.cdist(ci, cj)
                        mask = (dd < cutoff) & (dd > 1e-6)
                        ii, jj = torch.where(mask)
                        if ii.numel() > 0:
                            src_list.append(ii + i)
                            dst_list.append(jj + j)
                            d_list.append(dd[ii, jj])
                if src_list:
                    edge = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
                    dist = torch.cat(d_list)
                else:
                    edge = torch.empty((2,0), dtype=torch.long, device=self.device)
                    dist = torch.empty(0, device=self.device)
                result[name] = (edge, dist)
            return result

    # ── Refinement Engine (embedded) ──────────────────────────────────
    @dataclass
    class RefinementConfig:
        w_bond: float = 30.0; w_angle: float = 15.0; w_rama: float = 8.0; w_clash: float = 80.0
        w_hbond: float = 6.0; w_electro: float = 4.0; w_solvent: float = 5.0; w_soc: float = 0.3
        w_alpha_entropy: float = 0.5; w_alpha_smooth: float = 0.1; w_chain_break: float = 1.0
        w_lj: float = 30.0; w_coulomb: float = 5.0; w_torsion: float = 10.0
        clash_radius: float = 2.0; angle_target_rad: float = 111.0*math.pi/180.0; bond_target: float = 3.8
        base_temp: float = 300.0; friction: float = 0.02; sigma_target: float = 1.0
        avalanche_threshold: float = 0.5; w_avalanche: float = 0.2
        cutoff: float = 12.0; max_neighbors: int = 64
        lr: float = 1e-4; steps: int = 600; rebuild_interval: int = 100
        use_amp: bool = True; grad_clip: float = 5.0; use_milstein: bool = False
        use_rg: bool = True; rg_factor: int = 4; rg_interval: int = 200
        use_ssc: bool = True; epsilon_fp: float = 0.0028
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        anneal_start: float = 1000.0; anneal_end: float = 300.0; anneal_cycles: int = 1

    class RefinementEngine:
        def __init__(self, cfg: Optional[RefinementConfig] = None):
            self.cfg = cfg or RefinementConfig()
            self.device = torch.device(self.cfg.device)
            self.kernel = CSOCKernel(init_alpha=0.5, init_lambda=12.0)
            self.soc = SOCController(
                base_temp=self.cfg.base_temp, friction=self.cfg.friction,
                sigma_target=self.cfg.sigma_target, avalanche_threshold=self.cfg.avalanche_threshold,
                w_avalanche=self.cfg.w_avalanche, kernel=self.kernel,
                use_ssc=self.cfg.use_ssc, epsilon_fp=self.cfg.epsilon_fp)
            self.rg = DiffRGRefiner(self.cfg.rg_factor) if self.cfg.use_rg else None
            self.scaler = GradScaler(enabled=self.cfg.use_amp and self.device.type == 'cuda')
            self.neighbor_mgr = NeighborListManager(
                cutoffs={'clash':2.5, 'lj':6.0, 'elec':self.cfg.cutoff},
                max_neighbors=self.cfg.max_neighbors,
                device=self.device)

        def _build_edges(self, coords, batch=None):
            return self.neighbor_mgr.build(coords, batch)

        def _total_energy(self, ca, seq, alpha, chi,
                          edge_dict, edge_hb, edge_hb_dist,
                          boundaries, chain_types, mask,
                          ligand_bridge=None):
            E = torch.tensor(0.0, device=ca.device)
            atoms = reconstruct_backbone(ca)
            phi, psi = compute_phi_psi(atoms)
            prot_mask = mask if mask is not None else torch.ones(ca.shape[0], dtype=torch.bool, device=ca.device)

            E += energy_bond(ca, alpha, self.cfg.w_bond, mod=0.1, target=self.cfg.bond_target, mask=prot_mask)
            E += energy_angle(ca, alpha, self.cfg.w_angle, mod=0.05, target_rad=self.cfg.angle_target_rad, mask=prot_mask)
            E += energy_rama(phi, psi, seq, alpha, self.cfg.w_rama, mod=0.2, mask=prot_mask)
            E += energy_clash(ca, alpha, edge_dict['clash'][0], edge_dict['clash'][1], self.cfg.w_clash, self.cfg.clash_radius, mod=0.1, mask=prot_mask)
            E += energy_hbond(atoms['O'], atoms['N'], atoms['C'], alpha, edge_hb, edge_hb_dist, self.cfg.w_hbond, mod=0.1, mask=prot_mask)
            elec_edge, elec_dist = edge_dict['elec']
            q_res = torch.tensor([RESIDUE_CHARGE.get(a,0.0) for a in seq], device=ca.device)
            E += energy_electro(ca, seq, elec_edge, elec_dist, self.cfg.w_electro, mask=prot_mask)
            E += energy_solvent(ca, seq, elec_edge, self.cfg.w_solvent, mask=prot_mask)
            E += self.soc.compute_soc_energy(ca, alpha, elec_edge, elec_dist, mask=prot_mask, w_soc=self.cfg.w_soc)
            E += alpha_regularisation(alpha, self.cfg.w_alpha_entropy, self.cfg.w_alpha_smooth)
            E += chain_break_energy(ca, boundaries, self.cfg.w_chain_break)

            if chi is not None:
                E += energy_torsion_chi(chi, seq, self.cfg.w_torsion)

            return E

        def compute_energy(self, coords, sequence, chain_types=None, mask=None, chi=None, alpha=None) -> float:
            coords = coords.view(-1, 3)
            L = len(sequence)
            if mask is None: mask = torch.ones(L, dtype=torch.bool, device=coords.device)
            if chi is None: chi = torch.zeros(L, MAX_CHI, device=coords.device)
            if alpha is None: alpha = torch.ones(L, device=coords.device)
            edge_dict = self._build_edges(coords)
            atoms = reconstruct_backbone(coords)
            edge_hb, edge_hb_dist = self._build_edges(atoms['O'])['lj']
            E = self._total_energy(coords, sequence, alpha, chi,
                                   edge_dict, edge_hb, edge_hb_dist,
                                   [], ['protein']*L, mask)
            return E.item()

        def relax_local(self, coords, sequence, positions, steps=30, window=3,
                        chain_types=None, mask=None, chi=None, alpha=None):
            coords = coords.view(-1, 3).detach().requires_grad_(True)
            L = len(sequence)
            if mask is None: mask = torch.ones(L, dtype=torch.bool, device=coords.device)
            if chi is None: chi = torch.zeros(L, MAX_CHI, device=coords.device)
            if alpha is None: alpha = torch.ones(L, device=coords.device)
            min_pos = min(positions); max_pos = max(positions)
            win_start = max(0, min_pos - window); win_end = min(L, max_pos + window + 1)
            opt = torch.optim.Adam([coords], lr=self.cfg.lr)
            best_E = float('inf'); best_coords = coords.clone()
            for _ in range(steps):
                opt.zero_grad()
                edge_dict = self._build_edges(coords)
                atoms = reconstruct_backbone(coords)
                edge_hb, edge_hb_dist = self._build_edges(atoms['O'])['lj']
                E = self._total_energy(coords, sequence, alpha, chi,
                                       edge_dict, edge_hb, edge_hb_dist,
                                       [], ['protein']*L, mask)
                E.backward()
                if coords.grad is not None:
                    grad_mask = torch.zeros(L, 3, device=coords.device)
                    grad_mask[win_start:win_end] = 1.0
                    coords.grad *= grad_mask
                opt.step()
                if E.item() < best_E:
                    best_E = E.item()
                    best_coords = coords.detach().clone()
            return best_coords, best_E

    def load_structure(filepath, chain=None):
        if not HAS_BIOTITE: raise ImportError("biotite required for PDB loading")
        if filepath.endswith('.pdb'):
            struct = pdb_io.PDBFile.read(filepath).get_structure(model=1)
        elif filepath.endswith('.cif') or filepath.endswith('.mmcif'):
            struct = mmcif_io.MMCIFFile.read(filepath).get_structure(model=1)
        else:
            raise ValueError("File must be .pdb or .cif/.mmcif")
        if chain is not None:
            struct = struct[struct.chain_id == chain]
        ca = struct[struct.atom_name == "CA"]
        coords = ca.coord.astype(np.float32)
        seq = [AA_3_TO_1.get(res.res_name, 'X') for res in ca.residues]
        return {'coords': coords, 'sequence': "".join(seq), 'chain_ids': list(ca.chain_id)}

# =============================================================================
# 2. Gene Network BV (uses REAL FOLD ONE's BVFieldTheory if available)
# =============================================================================
class GeneNetworkBV(BVFieldTheory):
    def __init__(self, gene_names: List[str], interactions: List[Tuple[int, int]]):
        field_names = [f"phi_{i}" for i in range(len(gene_names))]
        super().__init__(field_names, [0] * len(field_names))
        self.gene_names = gene_names
        self.interactions = interactions
        for i, name in enumerate(field_names):
            self.phi[name] = torch.randn(1) * 0.01
    def action_functional(self, phi_dict, phi_star_dict):
        S = torch.tensor(0.0)
        for i, j in self.interactions:
            S += 0.5 * (phi_dict[f"phi_{i}"] - phi_dict[f"phi_{j}"]) ** 2
        return S
    def verify(self) -> bool:
        return self.classical_master_equation(self.action_functional)

# =============================================================================
# 3. Mutation Data Handling
# =============================================================================
class MutationDataLoader:
    def load_maf(self, file_path: str, genes_of_interest: List[str] = None) -> pd.DataFrame:
        maf = pd.read_csv(file_path, sep='\t', comment='#', low_memory=False)
        if genes_of_interest:
            maf = maf[maf['Hugo_Symbol'].isin(genes_of_interest)]
        return maf

    def load_vcf(self, file_path: str) -> pd.DataFrame:
        records = []
        with open(file_path) as f:
            for line in f:
                if line.startswith('##'): continue
                if line.startswith('#'):
                    header = line.strip().split('\t')
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 8: continue
                chrom, pos, _, ref, alt, _, _, info = fields[:8]
                gene = 'UNKNOWN'
                if 'GENE=' in info:
                    gene = info.split('GENE=')[1].split(';')[0]
                records.append({
                    'Chromosome': chrom,
                    'Start_Position': int(pos),
                    'End_Position': int(pos) + len(ref) - 1,
                    'Reference_Allele': ref,
                    'Tumor_Seq_Allele2': alt,
                    'Hugo_Symbol': gene,
                    'Tumor_Sample_Barcode': os.path.basename(file_path).replace('.vcf', '')
                })
        return pd.DataFrame(records)

    def build_mutation_matrix(self, mutations: pd.DataFrame, genes: List[str]) -> Tuple[np.ndarray, List[str]]:
        samples = mutations['Tumor_Sample_Barcode'].unique()
        sample_index = {s: i for i, s in enumerate(samples)}
        gene_index = {g: j for j, g in enumerate(genes)}
        M = np.zeros((len(samples), len(genes)))
        for _, row in mutations.iterrows():
            gene = row.get('Hugo_Symbol', '')
            sample = row.get('Tumor_Sample_Barcode', '')
            if gene in gene_index and sample in sample_index:
                i = sample_index[sample]
                j = gene_index[gene]
                M[i, j] = 1
        return M, list(samples)

# =============================================================================
# 4. Duon Analysis
# =============================================================================
class DuonAnalyzer:
    def __init__(self, duon_file: str = None):
        self.duon_positions = set()
        if duon_file and os.path.exists(duon_file):
            with open(duon_file) as f:
                for line in f:
                    try: self.duon_positions.add(int(line.strip()))
                    except ValueError: continue
    def set_duon_positions(self, positions: List[int]):
        self.duon_positions = set(positions)
    def compute_duon_mutation_rate(self, codon_mutation_vector: np.ndarray) -> float:
        mutated_codon_indices = np.where(codon_mutation_vector > 0)[0]
        if len(mutated_codon_indices) == 0:
            return 0.0
        hits = sum(1 for idx in mutated_codon_indices if idx in self.duon_positions)
        return hits / len(mutated_codon_indices)

# =============================================================================
# 5. Evolutionary Classifier (SOC + Ito + RG)
# =============================================================================
class EvolutionaryClassifier:
    def __init__(self, threshold_stable=0.2, threshold_collapse=0.8):
        self.threshold_stable = threshold_stable
        self.threshold_collapse = threshold_collapse
        self.soc = SOCController(base_temp=300, friction=0.02, sigma_target=1.0)
        self.rg = DiffRGRefiner(factor=4, n_levels=2)

    def mu_to_state(self, mu: float) -> int:
        if mu < self.threshold_stable: return 0
        if mu > self.threshold_collapse: return 2
        return 1

    def classify_samples(self, mu_values: np.ndarray) -> np.ndarray:
        return np.array([self.mu_to_state(mu) for mu in mu_values])

    def compute_entropy(self, states: np.ndarray) -> float:
        hist = np.bincount(states, minlength=3)
        p = hist / len(states)
        p = p[p > 0]
        return entropy(p, base=2)

    def soc_evolve(self, mu_values: torch.Tensor, steps: int = 10) -> torch.Tensor:
        x = mu_values.clone().detach()
        for _ in range(steps):
            sigma = self.soc.sigma(x)
            T = self.soc.temperature(sigma)
            noise = torch.randn_like(x) * T * 0.01
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)
        return x

    def ito_evolve(self, mu0: float, T: float = 300.0, dt: float = 0.01, steps: int = 100) -> torch.Tensor:
        def energy_fn(x):
            return 0.5 * (x - 0.5)**2 + torch.sin(x * math.pi * 2) * 0.1
        ld = LangevinDynamics(energy_fn, T=T, dt=dt, device='cpu')
        x = torch.tensor([mu0])
        for _ in range(steps):
            x = ld.step(x, scheme='milstein')
        return x

# =============================================================================
# 6. Future Mutation Predictor (uses REAL FOLD ONE HT or embedded fallback)
# =============================================================================
class FutureMutationPredictor:
    def __init__(self, pdb_dir: str = './pdbs'):
        self.pdb_dir = pdb_dir

    def predict_vulnerable_positions(self, gene: str, structure_file: str = None) -> List[Dict]:
        # If REAL FOLD ONE HT is available, use it for full scan
        if HAS_HT:
            if not structure_file:
                candidates = list(Path(self.pdb_dir).glob(f"*{gene}*.pdb"))
                if not candidates:
                    logger.warning(f"No PDB for {gene}")
                    return []
                structure_file = str(candidates[0])
            cfg = HTConfig(pdb_file=structure_file, scan_full=True, output_dir=f"./ht_scan_{gene}")
            scanner = HighThroughputScanner(cfg)
            scanner.load_structure()
            results = scanner.scan_single_mutations()
            df = pd.DataFrame(results)
            if df.empty: return []
            df['ddg'] = df['ddg'].astype(float)
            destabilizing = df[df['ddg'] > 1.5].sort_values('ddg', ascending=False)
            return destabilizing.to_dict('records')
        else:
            # Embedded fallback: scan a few example mutations using embedded RefinementEngine
            pdb_file = structure_file or self._find_pdb(gene)
            if not pdb_file: return []
            data = load_structure(pdb_file)
            coords = torch.tensor(data['coords'], dtype=torch.float32)
            seq = data['sequence']
            # For demonstration, scan every 10th residue with a few substitutions
            cfg = RefinementConfig(device='cpu', steps=50)
            engine = RefinementEngine(cfg)
            results = []
            for pos in range(0, len(seq), 10):
                if pos >= len(seq): continue
                wt = seq[pos]
                # Try a few substitutions that are common
                for new in 'AGVLI':
                    if new == wt: continue
                    mut_seq = seq[:pos] + new + seq[pos+1:]
                    try:
                        _, e_mut = engine.relax_local(coords.clone().detach().requires_grad_(True), mut_seq, [pos], steps=20)
                        e_wt = engine.compute_energy(coords, seq)
                        ddg = e_mut - e_wt
                        if ddg > 1.5:
                            results.append({'chain':0, 'pos_in_chain':pos, 'global_pos':pos,
                                            'wt':wt, 'mut':new, 'ddg':ddg, 'type':'mutation'})
                    except Exception as e:
                        logger.warning(f"Relax failed for {gene} {pos}{wt}->{new}: {e}")
            return results

    def _find_pdb(self, gene: str) -> Optional[str]:
        pdb_dir = Path(self.pdb_dir)
        candidates = list(pdb_dir.glob(f"*{gene}*.pdb"))
        return str(candidates[0]) if candidates else None

# =============================================================================
# 7. Chemical Intervention Recommender
# =============================================================================
CANCER_DRUG_TARGETS = {
    'KRAS': ['Sotorasib', 'Adagrasib'],
    'EGFR': ['Erlotinib', 'Gefitinib', 'Osimertinib'],
    'BRAF': ['Vemurafenib', 'Dabrafenib'],
    'PIK3CA': ['Alpelisib'],
    'ALK': ['Crizotinib', 'Alectinib'],
    'TP53': ['APR-246', 'COBI-348'],
    'IDH1': ['Ivosidenib'],
    'IDH2': ['Enasidenib'],
    'FLT3': ['Midostaurin', 'Gilteritinib'],
    'NTRK1': ['Larotrectinib', 'Entrectinib'],
    'MET': ['Capmatinib', 'Tepotinib'],
    'ERBB2': ['Trastuzumab', 'Pertuzumab'],
    'FGFR2': ['Pemigatinib'],
    'FGFR3': ['Erdafitinib'],
    'MTOR': ['Everolimus', 'Temsirolimus'],
    'AKT1': ['Capivasertib'],
    'MAP2K1': ['Trametinib', 'Selumetinib'],
}

class InterventionRecommender:
    def __init__(self):
        self.target_map = CANCER_DRUG_TARGETS

    def recommend_drugs(self, gene: str, ddg_data: List[Dict]) -> List[str]:
        drugs = self.target_map.get(gene, [])
        if not drugs: return []
        if any(abs(entry.get('ddg',0.0)) > 2.0 for entry in ddg_data):
            return drugs
        return []

    def suggest_stabilisers(self, gene: str) -> List[str]:
        stabilisers = {
            'TP53': ['PRIMA-1', 'PhiKan083'],
        }
        return stabilisers.get(gene, [])

# =============================================================================
# 8. Retrospective Lifestyle Factor Analysis
# =============================================================================
class RetrospectiveAnalyzer:
    def __init__(self, lifestyle_file: str = None):
        self.lifestyle_df = None
        if lifestyle_file and os.path.exists(lifestyle_file):
            self.lifestyle_df = pd.read_csv(lifestyle_file)

    def merge_with_mutation_data(self, sample_ids: List[str], mu_values: np.ndarray,
                                 duon_rates: np.ndarray = None) -> pd.DataFrame:
        df = pd.DataFrame({'sample_id': sample_ids, 'mu': mu_values})
        if duon_rates is not None:
            df['duon_rate'] = duon_rates
        if self.lifestyle_df is not None:
            df = df.merge(self.lifestyle_df, on='sample_id', how='inner')
        return df

    def compute_correlations(self, merged_df: pd.DataFrame, target='mu') -> Dict[str, float]:
        factors = [col for col in merged_df.columns if col not in ('sample_id', 'mu', 'duon_rate')]
        results = {}
        for fac in factors:
            if merged_df[fac].nunique() < 2: continue
            r_pearson, p_pearson = pearsonr(merged_df[fac], merged_df[target])
            r_spearman, p_spearman = spearmanr(merged_df[fac], merged_df[target])
            results[fac] = {'pearson_r': r_pearson, 'pearson_p': p_pearson,
                            'spearman_r': r_spearman, 'spearman_p': p_spearman}
        return results

# =============================================================================
# 9. Main Evolution ONE Engine
# =============================================================================
class EvolutionONEEngine:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        self.loader = MutationDataLoader()
        self.duon = DuonAnalyzer(self.cfg.get('duon_file'))
        self.classifier = EvolutionaryClassifier()
        self.structural = FutureMutationPredictor(pdb_dir=self.cfg.get('pdb_dir', './pdbs'))
        self.drug_engine = InterventionRecommender()
        self.retrospective = RetrospectiveAnalyzer(lifestyle_file=self.cfg.get('lifestyle_file'))

    def run(self,
            input_file: str,
            genes: List[str],
            format: str = 'maf',
            compute_future_mutations: bool = True,
            compute_structural: bool = True,
            gene_interactions: List[Tuple[int, int]] = None) -> Dict:

        # 1. Load mutations
        if format == 'maf':
            mut_df = self.loader.load_maf(input_file, genes)
        elif format == 'vcf':
            mut_df = self.loader.load_vcf(input_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        if mut_df.empty:
            logger.warning("No mutations found for the specified genes.")
            return {}

        # 2. Build mutation matrix
        M, sample_ids = self.loader.build_mutation_matrix(mut_df, genes)
        N_samples, N_genes = M.shape
        mu_raw = M.mean(axis=1)

        # 3. Apply RG smoothing to mutation load
        mu_tensor = torch.tensor(mu_raw, dtype=torch.float32)
        mu_smooth = self.classifier.rg.forward(mu_tensor).numpy()

        # 4. Compute duon mutation rate per sample (if duon positions are provided)
        duon_rates = None
        if self.duon.duon_positions:
            codon_length = 100  # placeholder; real pipeline would map mutations to codons
            duon_rates = []
            for i in range(N_samples):
                n_mut = max(1, int(mu_raw[i] * codon_length))
                codon_muts = np.random.choice(codon_length, size=n_mut, replace=False)
                codon_vec = np.zeros(codon_length)
                codon_vec[codon_muts] = 1
                rate = self.duon.compute_duon_mutation_rate(codon_vec)
                duon_rates.append(rate)
            duon_rates = np.array(duon_rates)

        # 5. Classify current evolutionary regime
        states = self.classifier.classify_samples(mu_smooth)
        H = self.classifier.compute_entropy(states)
        logger.info(f"Current entropy H = {H:.4f}, "
                    f"stable={np.sum(states==0)}, critical={np.sum(states==1)}, collapse={np.sum(states==2)}")

        # 6. Predictive evolution: SOC + Ito
        mu_avg = mu_smooth.mean()
        mu_tensor_avg = torch.tensor(mu_avg, dtype=torch.float32).unsqueeze(0)
        future_mu_soc = self.classifier.soc_evolve(mu_tensor_avg, steps=20)
        future_mu_ito = self.classifier.ito_evolve(mu_avg, steps=100)
        future_state_soc = self.classifier.mu_to_state(future_mu_soc.item())
        future_state_ito = self.classifier.mu_to_state(future_mu_ito.item())
        cancer_risk = "High" if (future_state_soc == 2 or future_state_ito == 2) else \
                      ("Moderate" if (future_state_soc == 1 or future_state_ito == 1) else "Low")
        logger.info(f"Predicted future μ (SOC) = {future_mu_soc.item():.4f}, "
                    f"(Ito) = {future_mu_ito.item():.4f} → Cancer risk: {cancer_risk}")

        # 7. BV topological consistency for gene network
        bv_ok = False
        if gene_interactions and len(genes) > 1:
            try:
                bv_net = GeneNetworkBV(genes, gene_interactions)
                bv_ok = bv_net.verify()
                logger.info(f"BV master equation satisfied: {bv_ok}")
            except Exception as e:
                logger.warning(f"BV check failed: {e}")

        # 8. Future mutation scanning (using REAL FOLD ONE HT or embedded)
        future_mutations = {}
        if compute_future_mutations:
            for gene in genes:
                logger.info(f"Scanning future escape mutations for {gene} ...")
                vuln = self.structural.predict_vulnerable_positions(gene)
                if vuln:
                    future_mutations[gene] = vuln
                    logger.info(f"  → {len(vuln)} destabilising mutations identified")

        # 9. Structural impact of current mutations + drug recommendations
        structural_results = {}
        drug_recos = {}
        if compute_structural:
            for gene in genes:
                pdb_file = self._find_pdb(gene)
                if not pdb_file: continue
                data = load_structure(pdb_file)
                coords = torch.tensor(data['coords'], dtype=torch.float32)
                seq = data['sequence']
                # In practice, extract the exact amino-acid changes from the MAF.
                # Here we place two example mutations for demonstration.
                example_mutations = [(10, 'A', 'G'), (15, 'V', 'L')]
                cfg = RefinementConfig(device='cpu', steps=50)
                engine = RefinementEngine(cfg)
                impacts = []
                for pos, from_aa, to_aa in example_mutations:
                    if pos >= len(seq): continue
                    mut_seq = seq[:pos] + to_aa + seq[pos+1:]
                    try:
                        _, e_mut = engine.relax_local(
                            coords.clone().detach().requires_grad_(True), mut_seq,
                            [pos], steps=30)
                        e_wt = engine.compute_energy(coords, seq)
                        impacts.append({
                            'position': pos, 'wt': seq[pos], 'mut': to_aa,
                            'ddg': e_mut - e_wt
                        })
                    except Exception as e:
                        logger.warning(f"Relax failed for {gene} {pos}{from_aa}->{to_aa}: {e}")
                structural_results[gene] = impacts
                drugs = self.drug_engine.recommend_drugs(gene, impacts)
                stabilisers = self.drug_engine.suggest_stabilisers(gene)
                if drugs or stabilisers:
                    drug_recos[gene] = {'targeted_drugs': drugs, 'stabilisers': stabilisers}

        # 10. Retrospective lifestyle factor correlation
        lifestyle_corrs = {}
        if self.retrospective.lifestyle_df is not None and duon_rates is not None:
            merged = self.retrospective.merge_with_mutation_data(sample_ids, mu_smooth, duon_rates)
            if len(merged) > 5:
                lifestyle_corrs['mu'] = self.retrospective.compute_correlations(merged, target='mu')
                lifestyle_corrs['duon_rate'] = self.retrospective.compute_correlations(merged, target='duon_rate')

        self.results = {
            'mu_raw': mu_raw,
            'mu_smooth': mu_smooth,
            'states': states,
            'entropy': H,
            'duon_rates': duon_rates,
            'cancer_risk': cancer_risk,
            'future_mu_soc': future_mu_soc.item(),
            'future_mu_ito': future_mu_ito.item(),
            'bv_satisfied': bv_ok,
            'future_mutations': future_mutations,
            'structural_impacts': structural_results,
            'drug_recommendations': drug_recos,
            'lifestyle_correlations': lifestyle_corrs,
        }
        return self.results

    def _find_pdb(self, gene: str) -> Optional[str]:
        pdb_dir = Path(self.cfg.get('pdb_dir', './pdbs'))
        candidates = list(pdb_dir.glob(f"*{gene}*.pdb"))
        return str(candidates[0]) if candidates else None

    def plot_phase_diagram(self, save_path: str = None):
        mus = np.linspace(0, 1, 100)
        H_vals = [self.classifier.compute_entropy(
            self.classifier.classify_samples(np.full(1000, mu))) for mu in mus]
        plt.figure(figsize=(8, 5))
        plt.plot(mus, H_vals, linewidth=2)
        plt.xlabel('Mutation load μ')
        plt.ylabel('Entropy H (bits)')
        plt.title('Cancer Evolution Phase Diagram (SOC‑informed)')
        if save_path:
            plt.savefig(save_path, dpi=200)
        plt.show()

# =============================================================================
# 10. Command Line Interface
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evolution ONE – Standalone Cancer Evolution Engine")
    parser.add_argument('--input', '-i', required=True, help='Mutation file (MAF or VCF)')
    parser.add_argument('--format', default='maf', choices=['maf', 'vcf'])
    parser.add_argument('--genes', nargs='+', required=True, help='List of gene symbols')
    parser.add_argument('--duon_file', help='File with duon codon positions (one per line)')
    parser.add_argument('--lifestyle_file', help='CSV file with sample_id and environmental factors')
    parser.add_argument('--gene_interactions', nargs='+', type=int,
                        help='Pairs of gene indices, e.g. 0 1 1 2')
    parser.add_argument('--pdb_dir', default='./pdbs', help='Directory containing PDB files')
    parser.add_argument('--output_dir', default='./evo_output')
    parser.add_argument('--no_future', action='store_true', help='Skip future mutation scanning')
    parser.add_argument('--no_struct', action='store_true', help='Skip structural analysis')
    parser.add_argument('--plot', action='store_true', help='Generate phase diagram')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse gene interactions if provided
    gene_interactions = None
    if args.gene_interactions:
        pairs = args.gene_interactions
        if len(pairs) % 2 != 0:
            print("Gene interactions must be given in pairs.")
            sys.exit(1)
        gene_interactions = [(pairs[i], pairs[i+1]) for i in range(0, len(pairs), 2)]

    engine = EvolutionONEEngine(cfg={
        'duon_file': args.duon_file,
        'lifestyle_file': args.lifestyle_file,
        'pdb_dir': args.pdb_dir
    })
    engine.run(
        input_file=args.input,
        genes=args.genes,
        format=args.format,
        compute_future_mutations=not args.no_future,
        compute_structural=not args.no_struct,
        gene_interactions=gene_interactions
    )

    if engine.results:
        summary = {
            'cancer_risk': engine.results['cancer_risk'],
            'entropy': engine.results['entropy'],
            'bv_satisfied': engine.results['bv_satisfied'],
            'drug_recommendations': engine.results['drug_recommendations']
        }
        with open(out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        df = pd.DataFrame({
            'sample_id': engine.results.get('sample_ids', []),
            'mu': engine.results['mu_smooth'],
            'state': engine.results['states']
        })
        df.to_csv(out_dir / 'sample_states.csv', index=False)
        print(f"Results saved to {out_dir}")

    if args.plot:
        engine.plot_phase_diagram(save_path=str(out_dir / 'phase_diagram.png'))


if __name__ == "__main__":
    main()
