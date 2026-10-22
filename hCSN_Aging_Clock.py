"""HCSN Aging Clock — meta cloud (organ/system-level) + base cloud
(cell-level features), fused exactly like the original HCSN meta/base
design. Predicts biological age (regression).

Developer : PAI AND Yoon A Limsuwan 
Organization : MSPS NETWORK 
Licence : MIT
Year : 2026

"""
import torch
import torch.nn as nn
from model import RMSNorm, HCSNPass

class HCSNAgingClock(nn.Module):
    def __init__(self, d_genomics=5000, d_metabol=1000, d_model=256,
                 n_layers=4, n_heads=8):
        super().__init__()
        # base cloud: metabolomics (N_b ~ 100 features as tokens)
        self.base_enc = nn.Linear(d_metabol // 100, d_model)
        # meta cloud: genomics reduced to M 'organ-branch' tokens
        self.n_meta = 8
        self.meta_enc = nn.Sequential(
            nn.Linear(d_genomics, self.n_meta * d_model), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, n_layers)
        self.hcsn = HCSNPass(d_model)          # meta/base 'collapse' fusion
        self.outcome_to_branch = nn.Sequential(   # from original HCSN design
            nn.Linear(d_model, self.n_meta), nn.Softmax(-1))
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)       # predicted biological age

    def forward(self, geno, metab):             # (B,5000), (B,100,10)
        B = geno.shape[0]
        base = self.base_enc(metab)                       # (B,100,d)
        meta = self.meta_enc(geno).view(B, self.n_meta, -1)  # (B,8,d)
        # fuse: meta pooled-condition biases the base stream (as in HCSN v1)
        pooled = meta.mean(dim=1, keepdim=True)
        x = base + pooled
        x = self.backbone(x)
        x = self.hcsn(x)
        w = self.outcome_to_branch(x.mean(dim=1))         # (B,8)
        pooled_x = torch.einsum("bm,bnd->bnd", w, x) * 0.1 + x  # mild gate
        return self.head(self.norm(pooled_x.mean(dim=1))).squeeze(-1)

# --- smoke test ---
if __name__ == "__main__":
    torch.manual_seed(0)
    m = HCSNAgingClock()
    geno = torch.randn(4, 5000)
    metab = torch.randn(4, 100, 10)
    age = m(geno, metab)
    age.sum().backward()
    print("aging OK:", age.shape, "| params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
