import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# =============================================================================
# 1. Configuration
# =============================================================================
@dataclass
class SGNOEvoConfig:
    node_in_dim: int = 20        # Amino acid / Voxel features
    hidden_dim: int = 128
    num_layers: int = 6
    dropout: float = 0.1
    cutoff_graph: float = 10.0   # Å for molecules / distance for patients
    cutoff_grid: float = 1.5     # Grid units for CH3D
    
    # Loss Weights
    lambda_evo: float = 1.0      # Weight for Mu/Rt prediction
    lambda_md: float = 0.5       # Weight for Langevin coords
    lambda_ch: float = 0.5       # Weight for Continuous CH3D

# =============================================================================
# 2. Shared FiLM-Modulated Backbone
# =============================================================================
class FiLMMessagePassing(nn.Module):
    """
    Message-passing layer using Feature-wise Linear Modulation (FiLM)
    Driven by the structural regime field sigma (σ).
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        # FiLM modulator: σ → (γ, β)
        self.film_gamma = nn.Linear(1, dim)
        self.film_beta  = nn.Linear(1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        # Message Aggregation
        msg_in = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.msg_mlp(msg_in)
        aggr = x.new_zeros(N, messages.size(1))
        aggr.index_add_(0, dst, messages)

        # FiLM Modulation by Structural Regime
        gamma = self.film_gamma(sigma)
        beta  = self.film_beta(sigma)
        modulated = gamma * aggr + beta

        # Update
        upd_in = torch.cat([x, modulated], dim=-1)
        x_new = self.upd_mlp(upd_in)
        return self.norm(x + x_new)

# =============================================================================
# 3. StructuralGNOEvolution (Main Model)
# =============================================================================
class StructuralGNOEvolution(nn.Module):
    def __init__(self, cfg: SGNOEvoConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # --- Shared Encoders ---
        self.node_embed = nn.Sequential(nn.Linear(cfg.node_in_dim, d), nn.LayerNorm(d))
        self.grid_embed = nn.Sequential(nn.Linear(4, d), nn.LayerNorm(d)) # For CH3D [u, x, y, z]

        # --- Shared Structural Backbone ---
        self.layers = nn.ModuleList([
            FiLMMessagePassing(d, cfg.dropout) for _ in range(cfg.num_layers)
        ])

        # --- Head 1: Evolution & Epidemic (Discrete/Population) ---
        self.evo_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 2) # Predicts [Next_Mu, Next_Rt]
        )
        self.classifier_head = nn.Linear(d, 3) # Predicts [Stable, Critical, Collapse]

        # --- Head 2: Structural Langevin (Atomistic) ---
        self.md_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 3) # Predicts Coordinate Displacement
        )

        # --- Head 3: CH3D Phase-Field (Continuous Grid) ---
        self.ch3d_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 1) # Predicts Delta_u
        )

    def _apply_backbone(self, x: torch.Tensor, edge_index: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, sigma)
        return x

    # --- Mode 1: EVOLUTION & EPIDEMIOLOGICAL ---
    def forward_evolution(self, feats: torch.Tensor, edge_index: torch.Tensor, sigma: torch.Tensor):
        """Couples EvolutionONE and EpiForecast via Mu and Rt."""
        x = self.node_embed(feats)
        x = self._apply_backbone(x, edge_index, sigma)
        
        mu_rt_pred = self.evo_head(x)          # (N, 2)
        logits = self.classifier_head(x)       # (N, 3)
        return mu_rt_pred, logits

    # --- Mode 2: STRUCTURAL LANGEVIN ---
    def forward_langevin(self, seq_feats: torch.Tensor, init_coords: torch.Tensor, edge_index: torch.Tensor, sigma: torch.Tensor):
        """Surrogate for BAOAB Langevin Integrator."""
        x = self.node_embed(seq_feats)
        x = self._apply_backbone(x, edge_index, sigma)
        
        displacements = self.md_head(x)
        return init_coords + displacements

    # --- Mode 3: CAHN-HILLIARD 3D ---
    def forward_ch3d(self, u_init: torch.Tensor, node_feats: torch.Tensor, edge_index: torch.Tensor, sigma_3d: torch.Tensor):
        """One-shot surrogate for Continuous Phase-field Evolution."""
        shape_3d = u_init.shape
        sigma_flat = sigma_3d.flatten().unsqueeze(-1)
        
        x = self.grid_embed(node_feats)
        x = self._apply_backbone(x, edge_index, sigma_flat)
        
        delta_u = self.ch3d_head(x).view(shape_3d)
        return u_init + delta_u

# =============================================================================
# 4. Multi-Objective Trainer (Conceptual Draft)
# =============================================================================
class SGNOEvolutionTrainer:
    def __init__(self, model: StructuralGNOEvolution, lr: float = 3e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step_all(self, batch_evo, batch_md, batch_ch):
        self.optimizer.zero_grad()
        total_loss = 0.0

        # 1. Evolution Loss (MSE for Mu/Rt + CrossEntropy for Class)
        mu_rt, logits = self.model.forward_evolution(batch_evo.feats, batch_evo.edge_index, batch_evo.sigma)
        loss_evo = F.mse_loss(mu_rt, batch_evo.true_mu_rt) + F.cross_entropy(logits, batch_evo.labels)
        total_loss += self.model.cfg.lambda_evo * loss_evo

        # 2. Langevin MD Loss (MSE of final coordinates)
        pred_coords = self.model.forward_langevin(batch_md.feats, batch_md.coords, batch_md.edge_index, batch_md.sigma)
        loss_md = F.mse_loss(pred_coords, batch_md.true_future_coords)
        total_loss += self.model.cfg.lambda_md * loss_md

        # 3. CH3D Loss (MSE of phase field)
        pred_u = self.model.forward_ch3d(batch_ch.u_init, batch_ch.feats, batch_ch.edge_index, batch_ch.sigma)
        loss_ch = F.mse_loss(pred_u, batch_ch.true_future_u)
        total_loss += self.model.cfg.lambda_ch * loss_ch

        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
