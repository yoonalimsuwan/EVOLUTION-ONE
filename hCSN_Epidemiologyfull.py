"""HCSN Epidemiology — SEIR compartmental model where transition rates
are learned functions (decoherence-inspired: population 'cloud' collapses
from susceptible to removed at a learned, time-varying rate).

NOT a language model — this is the honest architecture for this domain.

Developers   : PAI AND Yoon A Limsuwan
Organization : MSPS NETWORK
Licence      : MIT
Year         : 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSEIR(nn.Module):
    """S' = -beta S I ; E' = beta S I - sigma E
    I' = sigma E - gamma(t)*I ; R' = gamma(t)*I
    gamma(t) = gammas + net(features_t)    <- the 'decoherence rate'
    """

    def __init__(self, n_features=4, hidden=32):
        super().__init__()
        self.rate_net = nn.Sequential(          # learns gamma(t) from context
            nn.Linear(n_features, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Softplus())   # positive rate
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, S0, I0, T=100, features=None):
        # S0, I0: (B,) fractions; features: (B, T, n_features) e.g. mobility, season
        beta = F.softplus(self.log_beta)
        sigma = F.softplus(self.log_sigma)
        B = S0.shape[0]
        S, E, I, R = S0, torch.zeros_like(S0), I0, torch.zeros_like(S0)
        Ss, Is = [], []
        for t in range(T):
            gamma = self.rate_net(features[:, t, :]).squeeze(-1) if features is not None \
                    else torch.ones_like(S0) * 0.1
            new_E = beta * S * I
            new_I = sigma * E
            new_R = gamma * I
            S = S - new_E; E = E + new_E - new_I
            I = I + new_I - new_R; R = R + new_R
            Ss.append(S); Is.append(I)
        return torch.stack(Ss, 1), torch.stack(Is, 1)   # (B,T) each


# --- smoke test: fit a synthetic outbreak ---
if __name__ == "__main__":
    torch.manual_seed(0)
    m = LearnableSEIR()
    opt = torch.optim.AdamW(m.parameters(), lr=0.05)

    # simple direct integration for ground truth:
    S, I = 0.99, 0.01; gt = []
    for t in range(100):
        S = S - 0.3 * S * I; I = I + 0.3 * S * I - 0.2 * I
        gt.append(I)
    I_true = torch.tensor(gt).unsqueeze(0)

    for step in range(300):
        feat = torch.zeros(1, 100, 4)  # constant context
        _, I_pred = m(torch.tensor([0.99]), torch.tensor([0.01]), T=100, features=feat)
        loss = F.mse_loss(I_pred, I_true)
        opt.zero_grad(); loss.backward(); opt.step()

    print("epi OK — final fit MSE:", round(loss.item(), 6))
