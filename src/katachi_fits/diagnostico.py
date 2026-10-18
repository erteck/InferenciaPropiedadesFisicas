"""Diagnósticos de los modelos de referencia (cuaderno `Notebooks/Avance3.24.ipynb`).

- `metricas`: métricas de regresión con el uso científico de cada una.
- `radio_mitad_luz`: radio que encierra la mitad del flujo en la banda r (curva de crecimiento).
- `predecir_con`: predicción de la cadena con una modificación de la entrada
  (oclusión de una banda o de un anillo) o de los planos que pasan de una red a la
  siguiente (permutación). Sirve para medir qué información usa la red.
- `mediana_movil`: mediana y percentiles 16–84 de un residuo en intervalos de otra variable.
"""
from __future__ import annotations

import numpy as np
import torch

from .chain import add_plane
from . import aumentacion as AU


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def metricas(y, p, umbral_catastrofico: float | None = 0.3) -> dict:
    """RMSE, sesgo, MAE, σ_NMAD, fracción de atípicos, errores catastróficos y R².

    - σ_NMAD = 1.4826 · mediana(|r − mediana(r)|): dispersión robusta a valores extremos.
    - atípicos: |r − mediana(r)| > 3 σ_NMAD.
    - catastróficos: |r| > `umbral_catastrofico` (0.3 dex = un factor 2), solo para
      las propiedades en logaritmo.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    r = p - y
    med = np.median(r)
    nmad = 1.4826 * np.median(np.abs(r - med))
    out = {
        "N": int(len(y)),
        "RMSE": float(np.sqrt(np.mean(r ** 2))),
        "sesgo": float(np.mean(r)),
        "MAE": float(np.mean(np.abs(r))),
        "sigma_NMAD": float(nmad),
        "atipicos": float(np.mean(np.abs(r - med) > 3 * nmad)),
        "R2": float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
    }
    if umbral_catastrofico is not None:
        out["catastroficos"] = float(np.mean(np.abs(r) > umbral_catastrofico))
    return out


def rmse_ic(y, p, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Intervalo del 95 % del RMSE por bootstrap de galaxias."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    i = np.random.default_rng(seed).integers(0, len(y), (n, len(y)))
    v = np.sqrt(((p[i] - y[i]) ** 2).mean(1))
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def delta_rmse(y, p_a, p_b, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """RMSE(a) − RMSE(b) sobre las mismas galaxias, con intervalo del 95 % por bootstrap pareado."""
    y, a, b = (np.asarray(v, float) for v in (y, p_a, p_b))
    i = np.random.default_rng(seed).integers(0, len(y), (n, len(y)))
    d = np.sqrt(((a[i] - y[i]) ** 2).mean(1)) - np.sqrt(((b[i] - y[i]) ** 2).mean(1))
    punto = float(np.sqrt(np.mean((a - y) ** 2)) - np.sqrt(np.mean((b - y) ** 2)))
    return punto, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def mediana_movil(x, r, bordes) -> dict:
    """Mediana y percentiles 16 y 84 de `r` en los intervalos de `x` definidos por `bordes`."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    centro, med, p16, p84, n = [], [], [], [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        s = (x >= lo) & (x < hi) & np.isfinite(r)
        if s.sum() >= 10:
            centro.append(np.median(x[s]))
            med.append(np.median(r[s]))
            p16.append(np.percentile(r[s], 16))
            p84.append(np.percentile(r[s], 84))
            n.append(int(s.sum()))
    return {k: np.array(v) for k, v in
            dict(x=centro, mediana=med, p16=p16, p84=p84, n=n).items()}


# ---------------------------------------------------------------------------
# Tamaño de cada galaxia
# ---------------------------------------------------------------------------
def radio_mitad_luz(x_raw: np.ndarray, banda: int = 1, r_max: float | None = None, r_cielo: float | None = None,
                    lote: int = 256) -> np.ndarray:
    """Radio (px) que encierra la mitad del flujo en `banda`, medido desde el centro.

    Curva de crecimiento en aperturas circulares hasta `r_max` (por defecto, 0.43 del
    lado), después de restar el nivel del cielo: la mediana del anillo exterior
    (r > `r_cielo`, por defecto 0.45 del lado). El ruido no se recorta, así que se
    promedia a cero en lugar de sumarse como flujo. Se usa para expresar distancias en
    unidades del tamaño de cada galaxia.
    """
    n, _, h, w = x_raw.shape
    r_max = r_max or 0.43 * min(h, w)
    r_cielo = r_cielo or 0.45 * min(h, w)
    yy, xx = np.indices((h, w))
    rr = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    dentro, fuera = rr <= r_max, rr > r_cielo
    orden = np.argsort(rr[dentro])
    r_ord = rr[dentro][orden]
    out = np.empty(n, np.float32)
    for i in range(0, n, lote):
        img = x_raw[i:i + lote, banda].astype(np.float32)
        img = img - np.median(img[:, fuera], axis=1)[:, None, None]
        c = np.cumsum(img[:, dentro][:, orden], axis=1)
        tot = c.max(axis=1, keepdims=True)
        k = np.argmax(c >= 0.5 * np.maximum(tot, 1e-12), axis=1)
        out[i:i + lote] = r_ord[k]
    return np.clip(out, 1.0, r_max)


def mascara_anillo(r50_px: torch.Tensor, lado: int, a: float, b: float) -> torch.Tensor:
    """Máscara (B, 1, H, W) con 1 en el anillo a ≤ r/R50 < b de cada galaxia."""
    yy, xx = torch.meshgrid(torch.arange(lado, device=r50_px.device), torch.arange(lado, device=r50_px.device),
                            indexing="ij")
    rr = torch.hypot(yy - (lado - 1) / 2, xx - (lado - 1) / 2).float()
    u = rr[None] / r50_px.view(-1, 1, 1).float()
    return ((u >= a) & (u < b)).unsqueeze(1)


# ---------------------------------------------------------------------------
# Predicción con modificaciones
# ---------------------------------------------------------------------------
@torch.no_grad()
def predecir_con(trainer, data, ocluir_banda: int | None = None, anillo: tuple[float, float] | None = None,
                 r50_px: np.ndarray | None = None, permutar: str | None = None, seed: int = 0,
                 bs: int = 128) -> np.ndarray:
    """Predicción de la cadena (sin TTA) con una modificación.

    - `ocluir_banda`: índice del canal de entrada (orden de la red) que se pone en 0,
      el nivel del cielo tras el estiramiento.
    - `anillo` = (a, b) con `r50_px`: se pone en 0 la región a ≤ r/R50 < b.
    - `permutar`: "masa→SFR" (el plano de masa que recibe la red de SFR, y con él el de
      la red de D4000), "masa→D4000" (solo el plano de masa de la red de D4000) o
      "SFR→D4000". La permutación mezcla los valores entre galaxias del conjunto, así
      que conserva su distribución pero rompe su relación con cada imagen.
    """
    model = trainer.model
    model.eval()
    dev = trainer.device
    perm = None
    if permutar is not None:
        if permutar not in ("masa→SFR", "masa→D4000", "SFR→D4000"):
            raise ValueError(permutar)
        perm = torch.from_numpy(np.random.default_rng(seed).permutation(len(data))).to(dev)
    preds = []
    # Primer paso: predicciones sin modificar (necesarias para permutar los planos)
    p_orig = trainer.predict(data, bs=bs) if perm is not None else None
    p_orig = None if p_orig is None else torch.from_numpy(p_orig).to(dev)
    for i in range(0, len(data), bs):
        sl = slice(i, i + bs)
        sig = None if data.sigma is None else data.sigma[sl]
        x = trainer._inputs(data.x[sl], train=False, sig=sig)
        if ocluir_banda is not None:
            x = x.clone()
            x[:, ocluir_banda] = 0
        if anillo is not None:
            m = mascara_anillo(torch.as_tensor(r50_px[sl], device=dev), x.shape[-1], *anillo)
            x = x * (~m)
        if trainer.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=trainer.amp):
            if perm is None:
                p = torch.cat(model(x), 1).float()
            else:
                idx = torch.arange(i, min(i + bs, len(data)), device=dev)
                p1 = model.mass(x).float()
                p1_mezcla = p_orig[perm[idx], 0:1]
                p1_sfr = p1_mezcla if permutar == "masa→SFR" else p1
                x4 = add_plane(x, model.plano(p1_sfr, 0))
                p2 = model.sfr(x4).float()
                p1_d4 = p1_mezcla if permutar in ("masa→SFR", "masa→D4000") else p1
                p2_d4 = p_orig[perm[idx], 1:2] if permutar == "SFR→D4000" else p2
                x5 = add_plane(add_plane(x, model.plano(p1_d4, 0)), model.plano(p2_d4, 1))
                p3 = model.d4000(x5).float()
                p = torch.cat([p1, p2, p3], 1)
        preds.append(p.cpu())
    return torch.cat(preds).numpy()


@torch.no_grad()
def entrada_red(trainer, x_raw: torch.Tensor) -> torch.Tensor:
    """Imagen tal como la recibe la red (estirada, recortada y en el orden de bandas de la red)."""
    return trainer._inputs(x_raw.to(trainer.device), train=False)


def vista_rgb(x_red: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.7) -> np.ndarray:
    """Imagen en color (H, W, 3) a partir de la entrada de la red (canales z, r, g → R, G, B)."""
    x = np.asarray(x_red, np.float32)
    lo, hi = np.percentile(x, [p_lo, p_hi])
    return np.clip((np.transpose(x, (1, 2, 0)) - lo) / max(hi - lo, 1e-6), 0, 1)


__all__ = ["metricas", "rmse_ic", "delta_rmse", "mediana_movil", "radio_mitad_luz", "mascara_anillo",
           "predecir_con", "entrada_red", "vista_rgb"]
