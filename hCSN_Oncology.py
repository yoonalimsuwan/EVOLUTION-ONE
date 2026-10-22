"""HCSN Oncology — ViT-style pathology encoder + HCSNPass as periodic
'collapse' between transformer layers.
Domain meaning: meta cloud = whole-slide context, base cloud = patch
features; HMM outcome ≈ diagnostic 'measurement' of the slide.

Developers : PAI AND Yoon A Limsuwan 
Organization : MSPS NETWORK 
Licence : MIT
Year : 2026


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RMSNorm, HCSNPass

class PatchEncoder(nn.Module):
    def __init__(self, d_model=192, n_layers=6, n_heads=6, patch=16):
        super().__init__()
        self.embed = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, 400, d_model) * 0.02)  # 20x20 patches
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, n_layers)
        self.hcsn = HCSNPass(d_model)
        self.hcsn_every = 2
        self.head = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, img):                       # img: (B,3,320,320)
        f = self.embed(img).flatten(2).transpose(1, 2)   # (B,400,d)
        x = torch.cat([self.cls.expand(f.shape[0], -1, -1), f], dim=1)
        x = x + self.pos
        for i, blk in enumerate(self.backbone.layers):
            x = blk(x)
            if self.hcsn_every > 0 and (i + 1) % self.hcsn_every == 0:
                x = self.hcsn(x)
        return self.head(x[:, 0]).squeeze(-1)     # survival logit

# --- smoke test ---
if __name__ == "__main__":
    torch.manual_seed(0)
    m = PatchEncoder()
    img = torch.randn(2, 3, 320, 320)
    logit = m(img)
    logit.sum().backward()
    print("oncology OK:", logit.shape, "| params:", sum(p.numel() for p in m.parameters()))
