"""Pruebas de robustez de Katachi (§4.1.2).

- Ruido: σ = desviación estándar del borde de 5 px de cada imagen de entrada,
  100 iteraciones, media sobre galaxias de la desviación estándar por galaxia.
  Katachi: 0.023 (M*), 0.096 (SFR), 0.012 (D4000), 0.021 (t50).
- Degradación: reducir la resolución por un factor y volver al tamaño original.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import NOISE_BORDER_PX, NOISE_ITERATIONS


def border_std(x: torch.Tensor, width: int = NOISE_BORDER_PX) -> torch.Tensor:
    """Desviación estándar por imagen de los píxeles a ≤ width del borde (todas las bandas)."""
    m = torch.ones(x.shape[-2:], dtype=torch.bool, device=x.device)
    m[width:-width, width:-width] = False
    return x[..., m].flatten(1).std(dim=1)


@torch.no_grad()
def noise_bootstrap(predict: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, n_iter: int = NOISE_ITERATIONS,
                    seed: int = 0, sigma: torch.Tensor | None = None, bs: int = 128) -> np.ndarray:
    """Devuelve la desviación estándar por galaxia y salida, forma (N, n_salidas).
    `predict` recibe un lote ya preprocesado y devuelve (B, n_salidas)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    s = border_std(x) if sigma is None else sigma
    s = s.view(-1, 1, 1, 1) if s.ndim == 1 else s
    sums = sq = None
    for _ in range(n_iter):
        outs = []
        for i in range(0, x.shape[0], bs):
            xi = x[i:i + bs]
            si = s[i:i + bs] if s.shape[0] == x.shape[0] else s
            noisy = xi + torch.randn(xi.shape, generator=g, device=x.device) * si
            outs.append(predict(noisy).float())
        p = torch.cat(outs)
        sums = p if sums is None else sums + p
        sq = p ** 2 if sq is None else sq + p ** 2
    mean = sums / n_iter
    var = (sq / n_iter - mean ** 2).clamp_min(0) * n_iter / (n_iter - 1)
    return var.sqrt().cpu().numpy()


def degrade(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Pixelado como en Katachi: reducir la resolución y volver a ampliarla."""
    n = x.shape[-1]
    small = F.interpolate(x, size=(n // factor, n // factor), mode="area")
    return F.interpolate(small, size=(n, n), mode="nearest")
