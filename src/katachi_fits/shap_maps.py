"""Mapas SHAP y perfiles radiales, como en Katachi (plan §4, punto 6).

Auditado de `XAI_Methods.ipynb`, `manga_SHAP_radial_plots.ipynb` y
`Plotting_Notebook_update.ipynb`:
- `shap.GradientExplainer(model, lote_de_fondo)`, con un lote de 32 imágenes como fondo.
- Perfil: media sobre canales de `gaussian_filter(mapa·1e4, σ=4 px, 'nearest')`,
  luego promedio azimutal en anillos de 1 px desde el centro (127.5, 127.5).
- Eje en kpc: Katachi usa 0.5″/px (tamaño de spaxel de MaNGA) y D = cz/H0. Sus
  imágenes tienen en realidad 0.195″/px, así que su eje está estirado ×2.56. Para
  comparar E1 (0.262″/px) con B0 (0.195″/px) se usa la escala real de cada uno; la
  convención de Katachi solo se usa para reproducir sus figuras.
- Gradiente ∇ = perfil(20 kpc) − perfil(0 kpc), con el perfil interpolado a un eje
  común y el índice más cercano a 20 kpc.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from .config import SHAP_GRADIENT_KPC, SHAP_NSAMPLES, SHAP_SMOOTH_SIGMA_PX

C_KMS = 299792.0
H0 = 70.0
ARCSEC_PER_RAD = 206265.0


def kpc_per_px(z: float, arcsec_per_px: float) -> float:
    """Kiloparsecs por píxel a redshift z con D = cz/H0 (la aproximación de Katachi)."""
    d_mpc = C_KMS * z / H0
    return arcsec_per_px * d_mpc * 1e6 / ARCSEC_PER_RAD / 1e3


def azimuthal_average(image: np.ndarray) -> np.ndarray:
    """Copia del `azimuthalAverage` de Katachi (anillos enteros, centro geométrico)."""
    y, x = np.indices(image.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    ind = np.argsort(r.flat)
    r_sorted, i_sorted = r.flat[ind], image.flat[ind]
    r_int = r_sorted.astype(int)
    rind = np.where(r_int[1:] - r_int[:-1])[0]
    nr = rind[1:] - rind[:-1]
    csim = np.cumsum(i_sorted, dtype=float)
    return (csim[rind[1:]] - csim[rind[:-1]]) / nr


def radial_profile(shap_map_hwc: np.ndarray, sigma: float = SHAP_SMOOTH_SIGMA_PX) -> np.ndarray:
    """Perfil radial de un mapa SHAP como lo calcula Katachi (suavizado, media de bandas, anillos)."""
    temp = np.mean(ndimage.gaussian_filter(shap_map_hwc * 1e4, sigma, mode="nearest"), 2)
    return azimuthal_average(temp)


def gradient(profile: np.ndarray, z: float, arcsec_per_px: float, r_kpc: float = SHAP_GRADIENT_KPC) -> float:
    """∇ = perfil(r_kpc) − perfil(0); NaN si r_kpc cae fuera de la imagen."""
    xax = np.arange(len(profile)) * kpc_per_px(z, arcsec_per_px)
    if xax[-1] < r_kpc:
        return np.nan
    return float(np.interp(r_kpc, xax, profile) - profile[0])


def shap_maps(model: torch.nn.Module, background: torch.Tensor, x: torch.Tensor, batch: int = 8,
              nsamples: int = SHAP_NSAMPLES) -> np.ndarray:
    """Mapas SHAP (N, H, W, C) con GradientExplainer. nsamples=200 es el valor por
    defecto de shap, el que usó Katachi."""
    import shap

    model.eval()
    e = shap.GradientExplainer(model, background)
    out = []
    for i in range(0, x.shape[0], batch):
        sv = e.shap_values(x[i:i + batch], nsamples=nsamples)
        if isinstance(sv, list):  # shap < 0.45: lista por salida de (B, C, H, W)
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 5:          # shap >= 0.45: (B, C, H, W, salidas)
            sv = sv[..., 0]
        out.append(np.transpose(sv, (0, 2, 3, 1)))
    return np.concatenate(out)
