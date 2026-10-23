"""
=============================================================================
SESI CELLULAR REGENERATION SYSTEM v2.0 - PART 2: LOSS & TRAINER
=============================================================================
"""
from __future__ import annotations

import math
import time
import copy
import contextlib
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = False

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .sesi_cellular_regen_models import CellularRegenConfig


def is_dist() -> bool: return dist.is_available() and dist.is_initialized()
def rank() -> int: return dist.get_rank() if is_dist() else 0
def world_size() -> int: return dist.get_world_size() if is_dist() else 1
def is_main() -> bool: return rank() == 0

def all_reduce_mean(x: Tensor) -> Tensor:
    if not is_dist(): return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / world_size()


class CellularRegenLoss(nn.Module):
    def __init__(self, cfg: CellularRegenConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        cfg = self.cfg
        dev = out["regen_score"].device
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}
        
        L["regen"] = F.mse_loss(out["regen_score"], batch.get("regen_target", zero)) if "regen_target" in batch else zero
        L["sobolev"] = -out["sobolev_regular"].mean() if "sobolev_regular" in out else zero
        L["fractal"] = out["fractal_dim"].pow(2).mean() if "fractal_dim" in out else zero
        L["hcsn"] = out["hcsn_probs"].mean() if "hcsn_probs" in out else zero
        L["yamanaka"] = out["reprogramming_rate"].mean() if "reprogramming_rate" in out else zero
        L["sptc"] = -out["phi_iit"].mean() if "phi_iit" in out else zero
        L["opmc"] = out["compression_ratio"].mean() if "compression_ratio" in out else zero
        L["sqct"] = out["collapse_rate"].mean() if "collapse_rate" in out else zero
        L["organoid"] = out["tissue_architecture"].mean() if "tissue_architecture" in out else zero
        
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg
        
        total = (cfg.w_regen * L["regen"] + cfg.w_sobolev * L["sobolev"] + cfg.w_fractal * L["fractal"] +
                 cfg.w_hcsn * L["hcsn"] + cfg.w_yamanaka * L["yamanaka"] + cfg.w_sptc * L["sptc"] +
                 cfg.w_opmc * L["opmc"] + cfg.w_sqct * L["sqct"] + cfg.w_organoid * L["organoid"] + cfg.w_reg * L["reg"])
        L["total"] = total
        return L


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is None:
                self.shadow[k] = v.detach().clone()
                continue
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)
    
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)
    
    def state_dict(self): return {k: v.clone() for k, v in self.shadow.items()}
    def load_state_dict(self, sd): self.shadow = {k: v.clone() for k, v in sd.items()}


class CellularRegenTrainer:
    def __init__(self, cfg: CellularRegenConfig, model: nn.Module, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        
        if cfg.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception:
                pass
        
        if is_dist():
            self.model = DDP(self.model, device_ids=[device.index] if device.type == "cuda" else None,
                             find_unused_parameters=False, gradient_as_bucket_view=True)
        
        self.loss_fn = CellularRegenLoss(cfg)
        amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.amp_dtype]
        self._amp_dtype = amp_dt
        self._amp_on = cfg.amp_dtype != "fp32" and device.type == "cuda"
        self.scaler = _GradScaler(enabled=(self._amp_on and amp_dt == torch.float16))
        
        fused = cfg.use_fused_adamw and device.type == "cuda"
        self.opt = torch.optim.AdamW([p for p in self._raw().parameters() if p.requires_grad],
                                     lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), fused=fused)
        self.ema = EMA(self._raw(), cfg.ema_decay)
        self.step = 0
    
    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP): m = m.module
        return m
    
    def lr_now(self) -> float:
        cfg = self.cfg
        if self.step < cfg.warmup_steps:
            return cfg.lr * (self.step + 1) / max(1, cfg.warmup_steps)
        prog = (self.step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        for g in self.opt.param_groups: g["lr"] = self.lr_now()
        if (self.step + 1) % cfg.grad_accum == 0: self.opt.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_on):
            out = self.model(batch["h_pooled"])
            losses = self.loss_fn(out, batch, self._raw())
            loss = losses["total"] / cfg.grad_accum
        
        self.scaler.scale(loss).backward()
        if (self.step + 1) % cfg.grad_accum == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_([p for p in self._raw().parameters() if p.requires_grad], cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update(self._raw())
        
        self.step += 1
        return {k: float(v) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1}
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tot: Dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_on):
                out = self.model(batch["h_pooled"])
                losses = self.loss_fn(out, batch, self._raw())
            bs = batch["h_pooled"].shape[0]
            for k, v in losses.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    tot[k] = tot.get(k, 0.0) + float(v) * bs
            n += bs
        for k in tot: tot[k] /= max(1, n)
        if is_dist():
            for k in list(tot.keys()):
                if k != "n": tot[k] = float(all_reduce_mean(torch.tensor(tot[k], device=self.device)))
        tot["n"] = n
        return tot
    
    def state_dict(self):
        return {"model": self._raw().state_dict(), "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(), "ema": self.ema.state_dict(), "step": self.step}
    
    def load_state_dict(self, sd):
        self._raw().load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        try: self.scaler.load_state_dict(sd["scaler"])
        except Exception: pass
        self.ema.load_state_dict(sd["ema"])
        self.step = sd["step"]
