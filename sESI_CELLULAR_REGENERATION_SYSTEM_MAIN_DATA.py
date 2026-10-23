"""
=============================================================================
SESI CELLULAR REGENERATION SYSTEM v2.0 - PART 3: MAIN & DATA
=============================================================================
"""
from __future__ import annotations

import os
import time
import logging
import argparse
import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from .sesi_cellular_regen_models import CellularRegenConfig, SESICellularRegenerationSystem
from .sesi_cellular_regen_loss import CellularRegenTrainer, is_dist, rank, world_size, is_main, all_reduce_mean

logger = logging.getLogger("sesi.cellular_regen_main")


class SyntheticRegenDataset(Dataset):
    def __init__(self, cfg: CellularRegenConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.h_pooled = torch.randn(n, cfg.agi_latent_dim, generator=g)
        self.regen_target = torch.rand(n, generator=g)
    
    def __len__(self): return self.h_pooled.shape[0]
    
    def __getitem__(self, i):
        return {"h_pooled": self.h_pooled[i], "regen_target": self.regen_target[i]}


def build_regen_loaders(cfg: CellularRegenConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tr = SyntheticRegenDataset(cfg, cfg.n_train, cfg.data_seed)
    va = SyntheticRegenDataset(cfg, cfg.n_val, cfg.data_seed + 1)
    te = SyntheticRegenDataset(cfg, cfg.n_test, cfg.data_seed + 2)
    
    tr_s = DistributedSampler(tr, shuffle=True, drop_last=True) if is_dist() else None
    va_s = DistributedSampler(va, shuffle=False) if is_dist() else None
    te_s = DistributedSampler(te, shuffle=False) if is_dist() else None
    
    common = dict(batch_size=cfg.batch_size, num_workers=2, pin_memory=(cfg.device != "cpu"), persistent_workers=False)
    return (
        DataLoader(tr, sampler=tr_s, shuffle=(tr_s is None), drop_last=True, **common),
        DataLoader(va, sampler=va_s, shuffle=False, **common),
        DataLoader(te, sampler=te_s, shuffle=False, **common),
    )


def setup_dist(backend: str = "nccl") -> Tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group(backend=backend)
            return torch.device("cuda", local), dist.get_rank(), dist.get_world_size()
        dist.init_process_group(backend="gloo")
        return torch.device("cpu"), dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    return torch.device("cpu"), 0, 1


def cleanup_dist() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_train(cfg: CellularRegenConfig, device: torch.device):
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_dl, val_dl, _ = build_regen_loaders(cfg)
    model = SESICellularRegenerationSystem(cfg)
    trainer = CellularRegenTrainer(cfg, model, device)
    
    if is_main():
        n_par = sum(p.numel() for p in trainer._raw().parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_par:,}")
    
    best = float("inf")
    t0 = time.time()
    epoch = 0
    while trainer.step < cfg.max_steps:
        s = getattr(train_dl, "sampler", None)
        if isinstance(s, DistributedSampler): s.set_epoch(epoch)
        
        for batch in train_dl:
            logs = trainer.train_step(batch)
            if is_main() and trainer.step % cfg.log_every == 0:
                logger.info(f"step {trainer.step:6d} | " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()) +
                            f" | lr={trainer.lr_now():.2e} | {time.time() - t0:.1f}s")
            
            if trainer.step % cfg.eval_every == 0 and trainer.step > 0:
                ev = trainer.evaluate(val_dl)
                if is_main():
                    logger.info("  [val] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items() if k != "n"))
                    if ev.get("total", float("inf")) < best:
                        best = ev["total"]
                        torch.save(trainer.state_dict(), os.path.join(cfg.out_dir, "best.pt"))
            
            if is_main() and trainer.step % cfg.ckpt_every == 0:
                torch.save(trainer.state_dict(), os.path.join(cfg.out_dir, f"ckpt_{trainer.step}.pt"))
            
            if trainer.step >= cfg.max_steps: break
        epoch += 1
    
    if is_main():
        torch.save(trainer.state_dict(), os.path.join(cfg.out_dir, "ckpt_final.pt"))
        logger.info(f"Training complete. Best val loss: {best:.4f}")


def main():
    parser = argparse.ArgumentParser("SESI Cellular Regeneration System")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    device, r, ws = setup_dist()
    seed_all(42)
    
    cfg = CellularRegenConfig()
    if args.steps is not None: cfg.max_steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.out: cfg.out_dir = args.out
    cfg.device = str(device)
    
    if args.mode == "train":
        run_train(cfg, device)
    
    cleanup_dist()


if __name__ == "__main__":
    main()
