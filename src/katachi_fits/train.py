"""Entrenamiento con la receta auditada de Katachi (plan §3.4) y reanudable.

Bucle por época (como `MaNGA_DR17_Regr.ipynb`, celdas 24 y 27):
  1. Entrenar las tres redes juntas, lote 32, orden aleatorio, aumentación.
     Pérdidas MSE independientes; tres Adam (1e-3, 1e-3, 1e-4).
  2. Pérdida D4000 en el conjunto retenido → `ReduceLROnPlateau` del optimizador de M*.
     (Katachi usaba Test; aquí Validation o la pérdida de entrenamiento, decisión D1.)
  3. `EarlyStopper(1e-3, 10)` con la pérdida D4000 media de entrenamiento.
  4. Se guarda el estado final (Katachi no restauraba la mejor época).

Un solo `backward()` sobre la suma de pérdidas da los mismos gradientes que tres
llamadas, porque las entradas encadenadas están desacopladas (`detach`).
"""
from __future__ import annotations

import io
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .chain import Chain
from .config import TARGETS, TrainConfig
from .io_guard import WriteGuard
from .transforms import Normalizer, augment, preprocess


class EarlyStopper:
    """Copia literal del de Katachi."""

    def __init__(self, precision=1e-3, patience=10):
        self.precision, self.patience = precision, patience
        self.badepochs = 0
        self.min_valid_loss = float("inf")

    def step(self, valid_loss):
        if valid_loss < self.min_valid_loss * (1 - self.precision):
            self.badepochs = 0
            self.min_valid_loss = valid_loss
        else:
            self.badepochs += 1
        return not (self.badepochs == self.patience)

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, d):
        vars(self).update(d)


def seed_everything(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GPUData:
    """Imágenes crudas (recorte central, float16) y etiquetas residentes en el dispositivo."""

    def __init__(self, x_raw, y, device: torch.device):
        x = x_raw if isinstance(x_raw, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x_raw))
        yy = y if isinstance(y, torch.Tensor) else torch.from_numpy(np.asarray(y, np.float32))
        self.x = x.to(device)
        self.y = yy.float().to(device)

    def __len__(self):
        return self.x.shape[0]


class Trainer:
    """Entrena la cadena con la receta de Katachi, guarda un punto de control por época y puede reanudar."""
    def __init__(self, cfg: TrainConfig, norm: Normalizer, out_px: int, run_dir: Path, guard: WriteGuard,
                 device: torch.device, log=print):
        self.cfg, self.norm, self.out_px = cfg, norm, out_px
        self.run_dir, self.guard, self.device, self.log = Path(run_dir), guard, device, log
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        seed_everything(cfg.seed)
        self.model = Chain(imagenet=True, ghost_splits=cfg.ghost_bn_splits).to(device)
        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        fused = cfg.fused_adam and device.type == "cuda"
        nets = (self.model.mass, self.model.sfr, self.model.d4000)
        self.opts = [torch.optim.Adam(n.parameters(), lr=lr, fused=fused) if fused else
                     torch.optim.Adam(n.parameters(), lr=lr) for n, lr in zip(nets, cfg.lr)]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opts[0], patience=cfg.scheduler_patience)
        self.stopper = EarlyStopper(cfg.stop_precision, cfg.stop_patience)
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed)
        self.perm_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
        self.val_gen = torch.Generator(device=device).manual_seed(cfg.seed + 2)
        self.epoch = 0
        self.history: list[dict] = []
        self.stopped = False   # detenido por EarlyStopper (no por max_epochs)
        self.forward = torch.compile(self.model) if cfg.compile else self.model
        self.mse = nn.MSELoss()

    # ---------- entrada ----------
    def _inputs(self, xr: torch.Tensor, train: bool, gen: torch.Generator | None = None) -> torch.Tensor:
        x = preprocess(xr, self.norm, self.out_px)
        if train or gen is not None:
            x = augment(x, gen if gen is not None else self.gen, self.cfg.rotation_degrees)
        if self.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    # ---------- época ----------
    def train_epoch(self, data: GPUData) -> dict:
        """Una época sobre Train con aumentación; devuelve la pérdida media de cada red."""
        self.model.train()
        n, bs = len(data), self.cfg.batch_size
        perm = torch.randperm(n, generator=self.perm_gen).to(self.device)
        sums = torch.zeros(3, device=self.device)
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = self._inputs(data.x[idx], train=True)
            y = data.y[idx]
            for o in self.opts:
                o.zero_grad(set_to_none=True)
            p = self.forward(x)
            losses = [self.mse(p[k], y[:, k:k + 1]) for k in range(3)]
            torch.stack(losses).sum().backward()
            for o in self.opts:
                o.step()
            sums += torch.stack([l.detach() for l in losses])
            nb += 1
        avg = (sums / nb).tolist()
        return {"train_loss_mstar": avg[0], "train_loss_sfr": avg[1], "train_loss_d4000": avg[2]}

    @torch.no_grad()
    def predict(self, data: GPUData, bs: int = 128, aug_gen: torch.Generator | None = None) -> np.ndarray:
        """Predicción en modo eval. Con `aug_gen`, aplica la aumentación de Katachi
        (su DataLoader de Test la tenía; así se calculaba la señal del scheduler)."""
        self.model.eval()
        out = []
        for i in range(0, len(data), bs):
            x = self._inputs(data.x[i:i + bs], train=False, gen=aug_gen)
            out.append(torch.cat(self.model(x), 1).float().cpu())
        return torch.cat(out).numpy()

    def fit(self, train: GPUData, held_out: GPUData | None, checkpoint_every: int = 1) -> pd.DataFrame:
        """Entrena hasta que se cumple la parada de Katachi o `max_epochs`; reanuda si hay punto de control."""
        if self.cfg.scheduler_on == "val" and held_out is None:
            raise ValueError("scheduler_on='val' requiere un conjunto Validation (D1-a)")
        self.resume()
        while not self.stopped and self.epoch < self.cfg.max_epochs:
            t0 = time.time()
            self.epoch += 1
            row = {"epoch": self.epoch, **self.train_epoch(train)}
            if held_out is not None:
                yv = held_out.y.cpu().numpy()
                pa = self.predict(held_out, aug_gen=self.val_gen)   # señal del scheduler (como Katachi)
                pv = self.predict(held_out)                          # métrica limpia, solo registro
                for k, t in enumerate(TARGETS):
                    row[f"val_loss_{t}"] = float(np.mean((pa[:, k] - yv[:, k]) ** 2))
                    row[f"val_rmse_{t}_sin_aum"] = float(np.sqrt(np.mean((pv[:, k] - yv[:, k]) ** 2)))
            sched_signal = row["val_loss_d4000"] if self.cfg.scheduler_on == "val" else row["train_loss_d4000"]
            self.scheduler.step(sched_signal)
            row.update(lr_mstar=self.opts[0].param_groups[0]["lr"], seconds=time.time() - t0)
            self.stopped = not self.stopper.step(row["train_loss_d4000"])
            self.history.append(row)
            if self.epoch % checkpoint_every == 0 or self.done:
                self.save()
            self.log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if self.done:
            self.save_final()
        return pd.DataFrame(self.history)

    @property
    def done(self) -> bool:
        """True cuando la parada temprana se activó o se llegó a `max_epochs`."""
        return self.stopped or self.epoch >= self.cfg.max_epochs

    # ---------- persistencia ----------
    def state(self) -> dict:
        """Todo lo necesario para reanudar exactamente: pesos, optimizadores, scheduler, parada y generadores aleatorios."""
        return {
            "epoch": self.epoch, "stopped": self.stopped, "history": self.history,
            "model": self.model.state_dict(),
            "opts": [o.state_dict() for o in self.opts],
            "scheduler": self.scheduler.state_dict(), "stopper": self.stopper.state_dict(),
            "gen": self.gen.get_state(), "perm_gen": self.perm_gen.get_state(), "val_gen": self.val_gen.get_state(),
            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def save(self) -> None:
        """Guarda el punto de control `last.pt` (reemplazo atómico) y `history.csv`."""
        self.guard.mkdir(self.run_dir)
        buf = io.BytesIO()
        torch.save(self.state(), buf)
        self.guard.write(self.run_dir / "last.pt", lambda t: Path(t).write_bytes(buf.getvalue()), allow_replace=True)
        self.guard.write(self.run_dir / "history.csv",
                         lambda t: pd.DataFrame(self.history).to_csv(t, index=False), allow_replace=True)

    def save_final(self) -> None:
        """Guarda los pesos de la última época en `final_model.pt` (una sola vez)."""
        target = self.run_dir / "final_model.pt"
        if not target.exists():
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
            self.guard.write(target, lambda t: Path(t).write_bytes(buf.getvalue()))

    def resume(self) -> bool:
        """Si existe `last.pt`, restaura el estado completo y devuelve True."""
        ck = self.run_dir / "last.pt"
        if not ck.exists():
            return False
        s = torch.load(ck, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        for o, so in zip(self.opts, s["opts"]):
            o.load_state_dict(so)
        self.scheduler.load_state_dict(s["scheduler"])
        self.stopper.load_state_dict(s["stopper"])
        self.gen.set_state(s["gen"].cpu())
        self.perm_gen.set_state(s["perm_gen"].cpu())
        self.val_gen.set_state(s["val_gen"].cpu())
        torch.set_rng_state(s["torch_rng"].cpu())
        np.random.set_state(s["np_rng"])
        random.setstate(s["py_rng"])
        if s.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda_rng"]])
        self.epoch, self.stopped, self.history = s["epoch"], s["stopped"], s["history"]
        self.log(f"[resume] reanudado en la época {self.epoch} (detenido={self.stopped})")
        return True

    def load_final(self) -> None:
        """Carga `final_model.pt`."""
        self.model.load_state_dict(torch.load(self.run_dir / "final_model.pt", map_location=self.device))
