"""Representaciones de la imagen para la extracción de características.

- Estampas para eigengalaxias (Uzeirbegovic, Geach & Kaviraj 2020): cada galaxia se
  centra, se rota para que su eje mayor quede horizontal y se reescala para que la
  estampa cubra ±2 R_P. Así el PCA describe la forma y el color de la luz, no el
  tamaño ni la orientación, que ya son características propias.
- Imagen RGB para Zoobot (Walmsley et al. 2023): asinh de Lupton et al. (2004) con
  z → rojo, r → verde y g → azul.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config as K


def _rejilla(n_in: int, semilado_px: float, angulo: float, n_out: int) -> np.ndarray:
    """Coordenadas (fila, col) en la imagen de entrada para cada píxel de la estampa."""
    c = (n_in - 1) / 2
    u = np.linspace(-semilado_px, semilado_px, n_out)
    U, V = np.meshgrid(u, u)               # U: eje mayor (horizontal), V: eje menor
    ca, sa = np.cos(angulo), np.sin(angulo)
    x = c + U * ca - V * sa
    y = c + U * sa + V * ca
    return np.stack([y, x])


def estampa(limpio: np.ndarray, rp_px: float, pa_rad: float, n_out: int = K.ESTAMPA_PX) -> np.ndarray:
    """Estampa (3, n_out, n_out) alineada, reescalada a ±2 R_P y con el brillo
    normalizado por el flujo de r (conserva los colores) y comprimido con asinh."""
    if not np.isfinite(rp_px):
        rp_px = 20.0
    pa = pa_rad if np.isfinite(pa_rad) else 0.0
    semilado = K.ESTAMPA_RADIOS * rp_px
    coords = _rejilla(limpio.shape[-1], semilado, pa, n_out)
    est = np.stack([ndimage.map_coordinates(b, coords, order=1, cval=0.0) for b in limpio]).astype(np.float32)
    escala = est[1].mean()
    if not np.isfinite(escala) or escala <= 0:
        return np.full_like(est, np.nan)
    return np.arcsinh(est / escala)


def rgb_lupton(limpio: np.ndarray, rp_px: float, n_out: int = 224, q: float = 8.0, stretch: float = 0.05) -> np.ndarray:
    """Imagen RGB uint8 (n_out, n_out, 3) con la galaxia ocupando ±2 R_P (mínimo 24 px)."""
    from astropy.visualization import make_lupton_rgb

    semilado = max(K.ESTAMPA_RADIOS * (rp_px if np.isfinite(rp_px) else 20.0), 24.0)
    coords = _rejilla(limpio.shape[-1], semilado, 0.0, n_out)
    g, r, z = (ndimage.map_coordinates(b, coords, order=1, cval=0.0) for b in limpio)
    return make_lupton_rgb(z, r, g, Q=q, stretch=stretch, minimum=0.0)
