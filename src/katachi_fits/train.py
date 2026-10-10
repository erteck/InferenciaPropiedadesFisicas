"""Entrenamiento de la cadena siguiendo el artículo de Katachi (§3.1.2–3.1.3), reanudable.

Receta (justificación en Documentación/Propuesta_congelado_Katachi.md):
  1. Las tres redes se entrenan juntas: lote 32, orden aleatorio, aumentación.
     Cada una tiene su propia pérdida (MSE) y su propio Adam (1e-3).
  2. Con congelado, solo se optimizan los parámetros entrenables (ver chain.py).
  3. Cada red tiene su propio ReduceLROnPlateau (paciencia 3), guiado por su
     pérdida de entrenamiento: el artículo no usa conjunto de validación.
  4. Cada red tiene su propia parada: sin mejora relativa > 1e-3 en 10 épocas en su
     pérdida de entrenamiento. Una red detenida queda fija (sin actualizar pesos ni
     BatchNorm) y sigue produciendo su predicción para la siguiente. El
     entrenamiento termina cuando las tres se detienen o al llegar a max_epochs.
  5. Validation solo se registra, para vigilar el sobreajuste; no toma decisiones.

Con `per_network=False` y `freeze_pretrained=False` se recupera lo que muestran los
pesos publicados (todo entrenable, planificador solo en la red de masa): sirve
como corrida de control.

Un solo `backward()` sobre la suma de las pérdidas activas da los mismos gradientes
que una llamada por red, porque las entradas encadenadas están desacopladas.

Mejoras opcionales (`MejorasConfig`, todas apagadas por defecto): preparación de la
imagen y aumentación por lote en GPU (`aumentacion.py`), planos estandarizados,
precisión mixta bf16, un programa rápido de tasa de aprendizaje (coseno con
calentamiento y número fijo de épocas) y promedio de las 8 simetrías al predecir.
Con todas apagadas, el entrenamiento es idéntico al original.
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

from .chain import Chain, apply_frozen_bn
from . import aumentacion as AU
from .config import TARGETS, MejorasConfig, TrainConfig
from .io_guard import WriteGuard
from .transforms import Normalizer, augment, band_permutation, network_input

NOMBRES = ("mstar", "sfr", "d4000")


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

    def __init__(self, x_raw, y, device: torch.device, sigma=None):
        x = x_raw if isinstance(x_raw, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x_raw))
        yy = y if isinstance(y, torch.Tensor) else torch.from_numpy(np.asarray(y, np.float32))
        self.x = x.to(device)
        self.y = yy.float().to(device)
        # σ del cielo de cada imagen (N, 3), solo si se usa la mejora `sigma_por_imagen`
        self.sigma = None if sigma is None else torch.as_tensor(sigma, dtype=torch.float32).to(device)

    def __len__(self):
        return self.x.shape[0]


class Trainer:
    """Entrena la cadena con la receta del artículo, guarda un punto de control por época y puede reanudar."""

    def __init__(self, cfg: TrainConfig, norm: Normalizer, out_px: int, run_dir: Path, guard: WriteGuard,
                 device: torch.device, log=print, mejoras: MejorasConfig | None = None,
                 estadisticas_entrada: tuple | None = None, planos: tuple | None = None):
        self.cfg, self.norm, self.out_px = cfg, norm, out_px
        self.run_dir, self.guard, self.device, self.log = Path(run_dir), guard, device, log
        self.mejoras = m = mejoras or MejorasConfig()
        # Camino original (transforms.py) si no se pide ninguna mejora de imagen o aumentación
        self.legado = not (m.margen_rotacion or m.rotacion_bilineal or m.estandarizar_imagen or m.sigma_por_imagen
                           or m.desplazamiento_px or m.ruido_max or m.psf_max_px)
        self.amp = m.precision_mixta and device.type == "cuda"
        self.perm = band_permutation(cfg.input_band_order)
        self.sigma_global = torch.as_tensor(np.asarray(norm.sigma, np.float32), device=device)
        self.escala = torch.as_tensor(np.asarray(norm.scale, np.float32), device=device)
        if m.estandarizar_imagen and estadisticas_entrada is None:
            raise ValueError("estandarizar_imagen requiere estadisticas_entrada (media y desviación por banda)")
        self.est = None if estadisticas_entrada is None else tuple(
            torch.as_tensor(np.asarray(v, np.float32), device=device).view(1, -1, 1, 1) for v in estadisticas_entrada)
        if m.margen_rotacion and m.desplazamiento_px > 5:
            raise ValueError("Con margen_rotacion, desplazamiento_px debe ser ≤ 5: el recorte de 440 px no cubre más")
        if m.estandarizar_planos and planos is None:
            raise ValueError("estandarizar_planos requiere planos = (medias, desviaciones) de log M* y log SFR")
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        seed_everything(cfg.seed)
        self.model = Chain(imagenet=True, ghost_splits=cfg.ghost_bn_splits,
                           freeze_first=cfg.first_trainable_param if cfg.freeze_pretrained else None,
                           freeze_bn_stats=cfg.freeze_bn_stats,
                           planos=planos if m.estandarizar_planos else None).to(device)
        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        fused = cfg.fused_adam and device.type == "cuda"
        self.opts = []
        for net, lr in zip(self.model.nets(), cfg.lr):
            params = [p for p in net.parameters() if p.requires_grad]
            self.opts.append(torch.optim.Adam(params, lr=lr, fused=True) if fused else torch.optim.Adam(params, lr=lr))
        n_sched = 3 if cfg.per_network else 1
        if m.programa_rapido:      # coseno con calentamiento lineal, por época, en las tres redes
            E, W = m.epocas_rapido, m.calentamiento_epocas
            f = lambda e: (e + 1) / W if e < W else 0.5 * (1 + math.cos(math.pi * (e - W) / max(1, E - W)))
            self.schedulers = [torch.optim.lr_scheduler.LambdaLR(o, f) for o in self.opts]
        else:
            self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(self.opts[k], patience=cfg.scheduler_patience)
                               for k in range(n_sched)]
        self.stoppers = [EarlyStopper(cfg.stop_precision, cfg.stop_patience) for _ in range(3 if cfg.per_network else 1)]
        self.net_stopped = [False, False, False]
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed)
        self.perm_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
        self.epoch = 0
        self.history: list[dict] = []
        self.forward = torch.compile(self.model) if cfg.compile else self.model
        self.mse = nn.MSELoss()

    # ---------- entrada y modos ----------
    def _inputs(self, xr: torch.Tensor, train: bool, sig: torch.Tensor | None = None,
                estandarizar: bool = True) -> torch.Tensor:
        if self.legado:
            x = network_input(xr, self.norm, self.out_px, self.cfg.input_band_order)
            if train:
                x = augment(x, self.gen, self.cfg.rotation_degrees)
        else:
            m = self.mejoras
            x = xr.float()
            sigma = sig if (m.sigma_por_imagen and sig is not None) else self.sigma_global
            if train and (m.ruido_max > 0 or m.psf_max_px > 0):
                x = AU.ruido_y_psf(x, sigma, self.gen, m.ruido_max, m.psf_max_px, m.prob_aumentacion)
            x = AU.estirar(x, sigma, self.norm.beta, self.escala)
            if train:
                x = AU.geometria(x, self.gen, self.out_px, self.cfg.rotation_degrees, m.rotacion_bilineal,
                                 m.desplazamiento_px)
            else:
                x = AU.recorte_central(x, self.out_px)
            if m.estandarizar_imagen and estandarizar:
                x = (x - self.est[0]) / self.est[1]
            x = x[:, self.perm]
        if self.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _set_train_mode(self) -> None:
        """Modo entrenamiento, salvo BatchNorm congeladas y redes ya detenidas."""
        self.model.train()
        for net, stopped in zip(self.model.nets(), self.net_stopped):
            if stopped:
                net.eval()
        apply_frozen_bn(self.model)

    @property
    def stopped(self) -> bool:
        return all(self.net_stopped)

    @property
    def done(self) -> bool:
        """True cuando las tres redes se detuvieron o se llegó a `max_epochs`."""
        if self.mejoras.programa_rapido:
            return self.epoch >= self.mejoras.epocas_rapido
        return self.stopped or self.epoch >= self.cfg.max_epochs

    # ---------- época ----------
    def train_epoch(self, data: GPUData) -> dict:
        """Una época sobre Train con aumentación; devuelve la pérdida media de cada red."""
        self._set_train_mode()
        n, bs = len(data), self.cfg.batch_size
        perm = torch.randperm(n, generator=self.perm_gen).to(self.device)
        sums = torch.zeros(3, device=self.device)
        nb = 0
        activas = [k for k in range(3) if not self.net_stopped[k]]
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = self._inputs(data.x[idx], train=True, sig=None if data.sigma is None else data.sigma[idx])
            y = data.y[idx]
            for k in activas:
                self.opts[k].zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                p = self.forward(x)
            losses = [self.mse(p[k].float(), y[:, k:k + 1]) for k in range(3)]
            if activas:
                torch.stack([losses[k] for k in activas]).sum().backward()
                for k in activas:
                    self.opts[k].step()
            sums += torch.stack([l.detach() for l in losses])
            nb += 1
        avg = (sums / nb).tolist()
        return {f"train_loss_{NOMBRES[k]}": avg[k] for k in range(3)}

    @torch.no_grad()
    def predict(self, data: GPUData, bs: int = 128, tta: bool = False) -> np.ndarray:
        """Predicción en modo evaluación, sin aumentación. Con `tta=True`, promedio de las
        8 simetrías exactas del cuadrado (giros de 90° y reflejos)."""
        self.model.eval()
        out = []
        for i in range(0, len(data), bs):
            sig = None if data.sigma is None else data.sigma[i:i + bs]
            x = self._inputs(data.x[i:i + bs], train=False, sig=sig)
            vistas = range(8) if tta else (0,)
            acc = 0
            for j in vistas:
                xv = AU.d4(x, j) if j else x
                if self.cfg.channels_last:
                    xv = xv.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    acc = acc + torch.cat(self.model(xv), 1).float()
            out.append((acc / len(vistas)).cpu())
        return torch.cat(out).numpy()

    def _step_schedulers_and_stoppers(self, row: dict) -> None:
        if self.mejoras.programa_rapido:      # número fijo de épocas: sin parada temprana
            for sc in self.schedulers:
                sc.step()
            return
        if self.cfg.per_network:
            for k in range(3):
                if self.net_stopped[k]:
                    continue
                signal = row[f"{'train' if self.cfg.scheduler_signal == 'train' else 'val'}_loss_{NOMBRES[k]}"]
                self.schedulers[k].step(signal)
                if not self.stoppers[k].step(row[f"train_loss_{NOMBRES[k]}"]):
                    self.net_stopped[k] = True
                    row[f"detenida_{NOMBRES[k]}"] = True
        else:   # control: como los pesos publicados (planificador en masa, parada por D4000)
            signal = row["val_loss_d4000" if self.cfg.scheduler_signal == "val" else "train_loss_d4000"]
            self.schedulers[0].step(signal)
            if not self.stoppers[0].step(row["train_loss_d4000"]):
                self.net_stopped = [True, True, True]

    def fit(self, train: GPUData, held_out: GPUData | None, checkpoint_every: int = 1) -> pd.DataFrame:
        """Entrena hasta que las tres redes se detienen o `max_epochs`; reanuda si hay punto de control."""
        if self.cfg.scheduler_signal == "val" and held_out is None:
            raise ValueError("scheduler_signal='val' requiere un conjunto Validation")
        self.resume()
        while not self.done:
            t0 = time.time()
            self.epoch += 1
            row = {"epoch": self.epoch, **self.train_epoch(train)}
            if held_out is not None:     # solo para vigilar el sobreajuste
                pv, yv = self.predict(held_out), held_out.y.cpu().numpy()
                for k, t in enumerate(TARGETS):
                    row[f"val_loss_{NOMBRES[k]}"] = float(np.mean((pv[:, k] - yv[:, k]) ** 2))
            self._step_schedulers_and_stoppers(row)
            for k in range(3):
                row[f"lr_{NOMBRES[k]}"] = self.opts[k].param_groups[0]["lr"]
            row["seconds"] = time.time() - t0
            self.history.append(row)
            if self.epoch % checkpoint_every == 0 or self.done:
                self.save()
            self.log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        self.save_final()
        return pd.DataFrame(self.history)

    # ---------- persistencia ----------
    def state(self) -> dict:
        """Todo lo necesario para reanudar exactamente: pesos, optimizadores, planificadores, paradas y generadores."""
        return {
            "epoch": self.epoch, "net_stopped": self.net_stopped, "history": self.history,
            "model": self.model.state_dict(),
            "opts": [o.state_dict() for o in self.opts],
            "schedulers": [s.state_dict() for s in self.schedulers],
            "stoppers": [s.state_dict() for s in self.stoppers],
            "gen": self.gen.get_state(), "perm_gen": self.perm_gen.get_state(),
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
        """Guarda los pesos finales en `final_model.pt` (una sola vez)."""
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
        for sc, ss in zip(self.schedulers, s["schedulers"]):
            sc.load_state_dict(ss)
        for st, ss in zip(self.stoppers, s["stoppers"]):
            st.load_state_dict(ss)
        self.gen.set_state(s["gen"].cpu())
        self.perm_gen.set_state(s["perm_gen"].cpu())
        torch.set_rng_state(s["torch_rng"].cpu())
        np.random.set_state(s["np_rng"])
        random.setstate(s["py_rng"])
        if s.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda_rng"]])
        self.epoch, self.net_stopped, self.history = s["epoch"], list(s["net_stopped"]), s["history"]
        self.log(f"[reanudación] época {self.epoch}, redes detenidas {self.net_stopped}")
        return True

    def load_final(self) -> None:
        """Carga `final_model.pt`."""
        self.model.load_state_dict(torch.load(self.run_dir / "final_model.pt", map_location=self.device))


def freezing_audit(chain: Chain, first_trainable: int, imagenet_state: dict) -> pd.DataFrame:
    """Compara una cadena entrenada con los pesos de ImageNet: para cada red, cuántos
    tensores congelados (parámetros y buffers de BatchNorm) siguen idénticos bit a bit
    y cuántos entrenables cambiaron. Es la prueba que muestra si el congelado fue real."""
    from .chain import frozen_param_names

    frozen = frozen_param_names(first_trainable)
    alias = {"conv1.conv_img.weight": "conv1.weight"}
    filas = []
    for name, net in zip(("masa", "SFR", "D4000"), chain.nets()):
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        bn_frozen = {n.rsplit(".", 1)[0] for n in frozen if "bn" in n or "downsample.1" in n}
        cong_ok = cong_tot = ent_cambio = ent_tot = 0
        for k, v in sd.items():
            ref_key = alias.get(k, k)
            capa = ref_key.rsplit(".", 1)[0]
            es_congelado = ref_key in frozen or (capa in bn_frozen and ref_key.split(".")[-1] in
                                                 ("running_mean", "running_var", "num_batches_tracked"))
            if es_congelado and ref_key in imagenet_state and imagenet_state[ref_key].shape == v.shape:
                cong_tot += 1
                cong_ok += int(torch.equal(v, imagenet_state[ref_key]))
            elif not es_congelado and v.dtype.is_floating_point and not k.endswith(("running_mean", "running_var")):
                ent_tot += 1
                ref = imagenet_state.get(ref_key)
                if ref is None or ref.shape != v.shape:     # capas nuevas (fc, canales extra): ¿dejaron su valor inicial?
                    cambiado = bool(v.abs().sum() > 0) if "conv_extra" in k else True
                else:
                    cambiado = not torch.equal(v, ref)
                ent_cambio += int(cambiado)
        filas.append({"red": name, "congelados idénticos a ImageNet": f"{cong_ok}/{cong_tot}",
                      "entrenables modificados": f"{ent_cambio}/{ent_tot}",
                      "parámetros entrenables": sum(p.numel() for p in net.parameters() if p.requires_grad)})
    return pd.DataFrame(filas)
