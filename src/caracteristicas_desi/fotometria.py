"""Fotometría sobre los recortes DESI: cielo, máscaras, curva de crecimiento y Petrosian.

Convenciones:
- El recorte es (3, N, N) en nanomaggies, bandas g, r, z, centrado en las
  coordenadas de MaNGA. El centro está en ((N−1)/2, (N−1)/2).
- Todas las aperturas son circulares, como las cantidades de Petrosian de SDSS
  (Blanton et al. 2001; Strauss et al. 2002) y el `PETRO_TH90` de NSA con el que
  se validan.
- El radio de Petrosian se mide en la banda r y la MISMA apertura se usa en las
  tres bandas, de modo que los colores comparan la misma región de la galaxia.
- La segmentación reutiliza el código del equipo (`src/data/mask_eval.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from scipy import ndimage

from src.data.mask_eval import background, threshold_mask

from . import config as K


@lru_cache(maxsize=8)
def mapa_radial(n: int) -> np.ndarray:
    """Distancia de cada píxel al centro del recorte, en píxeles."""
    c = (n - 1) / 2
    yy, xx = np.indices((n, n), dtype=np.float32)
    r = np.hypot(xx - c, yy - c)
    r.setflags(write=False)
    return r


def cielo(img: np.ndarray, r_min_px: float = K.CIELO_R_MIN_PX) -> tuple[float, float]:
    """Mediana y desviación del cielo con recorte sigma (3σ), en las esquinas del
    recorte. Los píxeles exactamente en 0 (sin cobertura) no cuentan."""
    r = mapa_radial(img.shape[0])
    v = img[(r > r_min_px) & (img != 0)]
    if v.size < 100:
        return 0.0, float("nan")
    _, med, std = sigma_clipped_stats(v, sigma=3.0, maxiters=5)
    return float(med), float(std)


def segmentar(img_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Máscara de la galaxia y de los vecinos, en la banda r.

    - Galaxia: umbral de 2σ sobre la imagen suavizada, componente que contiene el
      centro, sin huecos y crecida 2 px (`mask_eval.threshold_mask`, valores del equipo).
    - Vecinos: fuentes detectadas a 3σ sin suavizar que no son la central, crecidas
      3 px, y solo FUERA de la máscara de la galaxia, para no quitar luz propia
      (nudos de formación estelar del disco).
    Devuelve (galaxia, vecinos, número de fuentes vecinas).
    """
    n = img_r.shape[0]
    c = ((n - 1) / 2, (n - 1) / 2)
    gal = threshold_mask(img_r, c, K.SEG_NSIGMA, K.SEG_SUAVIZADO_PX, K.SEG_CRECER_PX)
    med, std = background(img_r)
    seg = detect_sources(img_r - med, K.VEC_NSIGMA * std, npixels=5)
    if seg is None:
        return gal, np.zeros_like(gal), 0
    lab = seg.data
    central = lab[int(round(c[1])), int(round(c[0]))]
    otros = (lab > 0) & (lab != central)
    n_fuentes = int(len(np.setdiff1d(np.unique(lab[otros]), [0])))
    vecinos = ndimage.binary_dilation(otros, iterations=K.VEC_CRECER_PX) & ~gal
    return gal, vecinos, n_fuentes


def rellenar_simetrico(img: np.ndarray, faltan: np.ndarray) -> np.ndarray:
    """Sustituye los píxeles marcados por su simétrico respecto al centro (rotación
    de 180°) cuando este es válido, y por 0 en otro caso. Aprovecha la simetría
    aproximada de las galaxias para no restar luz propia al quitar un vecino."""
    rot = img[::-1, ::-1]
    rot_ok = ~faltan[::-1, ::-1]
    out = img.copy()
    out[faltan & rot_ok] = rot[faltan & rot_ok]
    out[faltan & ~rot_ok] = 0.0
    return out


@dataclass
class Perfil:
    """Curva de crecimiento circular: flujo y área acumulados hasta cada radio."""
    radios: np.ndarray     # bordes, en píxeles
    flujo: np.ndarray      # flujo acumulado F(<R)
    area: np.ndarray       # número de píxeles con centro a r < R

    def F(self, R) -> np.ndarray:
        return np.interp(R, self.radios, self.flujo)

    def A(self, R) -> np.ndarray:
        return np.interp(R, self.radios, self.area)


def perfil(img: np.ndarray, paso: float = K.PASO_RADIAL_PX) -> Perfil:
    r = mapa_radial(img.shape[0])
    nb = int(np.ceil(r.max() / paso)) + 1
    k = (r / paso).astype(np.int32).ravel()
    s = np.bincount(k, weights=img.ravel().astype(np.float64), minlength=nb)
    a = np.bincount(k, minlength=nb).astype(np.float64)
    radios = np.arange(nb + 1) * paso
    return Perfil(radios, np.concatenate([[0.0], np.cumsum(s)]), np.concatenate([[0.0], np.cumsum(a)]))


def radio_petrosian(p: Perfil, r_max: float, eta: float = K.PETRO_ETA) -> float:
    """Radio donde el brillo superficial medio del anillo [0.8R, 1.25R] es `eta`
    veces el brillo medio dentro de R (definición de SDSS). NaN si no se alcanza
    antes de `r_max` (la galaxia es demasiado grande para el recorte)."""
    a_in, a_out = K.PETRO_ANILLO
    R = np.arange(2.0, r_max / a_out, K.PASO_RADIAL_PX)
    if R.size < 3:
        return float("nan")
    anillo = (p.F(a_out * R) - p.F(a_in * R)) / np.maximum(p.A(a_out * R) - p.A(a_in * R), 1)
    dentro = p.F(R) / np.maximum(p.A(R), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = anillo / dentro
    ok = np.isfinite(q) & (dentro > 0)
    debajo = np.where(ok & (q < eta))[0]
    if debajo.size == 0 or debajo[0] == 0:
        return float("nan")
    i = debajo[0]
    # Interpolación lineal entre el último radio por encima de eta y el primero por debajo.
    q0, q1 = q[i - 1], q[i]
    return float(R[i - 1] + (q0 - eta) / (q0 - q1) * (R[i] - R[i - 1]))


def radio_fraccion(p: Perfil, total: float, frac: float, r_lim: float) -> float:
    """Primer radio en el que F(<R) alcanza `frac` × `total` (sin pasar de r_lim)."""
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    m = p.radios <= r_lim
    R, F = p.radios[m], p.flujo[m]
    i = np.argmax(F >= frac * total)
    if F[i] < frac * total or i == 0:
        return float("nan")
    return float(R[i - 1] + (frac * total - F[i - 1]) / (F[i] - F[i - 1]) * (R[i] - R[i - 1]))


def magnitud(flujo: float) -> float:
    """Magnitud AB a partir del flujo en nanomaggies (NaN si el flujo no es positivo)."""
    return float(K.NANOMAGGIE_ZP - 2.5 * np.log10(flujo)) if flujo > 0 else float("nan")


@dataclass
class Medicion:
    """Resultados intermedios que necesitan la morfología, las estampas y las figuras."""
    limpio: np.ndarray                   # (3, N, N): cielo restado y vecinos rellenados
    sigma: np.ndarray                    # (3,) σ del cielo por banda
    mascara_gal: np.ndarray
    mascara_vec: np.ndarray
    perfiles: list
    rp_px: float
    r_ap_px: float                       # radio de la apertura de Petrosian, limitado al recorte
    valores: dict = field(default_factory=dict)


def medir_fotometria(cubo: np.ndarray, pixscale: float = K.PIXSCALE) -> Medicion:
    """Fotometría de Petrosian en g, r y z con la apertura medida en r."""
    cubo = np.asarray(cubo, dtype=np.float32)
    nb, n, _ = cubo.shape
    r_borde = (n - 1) / 2 - 1                         # mayor círculo completo dentro del recorte
    v: dict = {}

    med_sig = [cielo(cubo[b]) for b in range(nb)]
    sigma = np.array([s for _, s in med_sig])
    sin_datos = cubo == 0
    sub = cubo - np.array([m for m, _ in med_sig], np.float32)[:, None, None]
    sub[sin_datos] = 0.0

    gal, vec, n_fuentes = segmentar(sub[1])
    limpio = np.stack([rellenar_simetrico(sub[b], vec | sin_datos[b]) for b in range(nb)])
    perfiles = [perfil(limpio[b]) for b in range(nb)]

    rp = radio_petrosian(perfiles[1], r_borde)
    r_ap = min(K.PETRO_APERTURA * rp, r_borde) if np.isfinite(rp) else float("nan")
    v["petro_converge"] = bool(np.isfinite(rp))
    v["apertura_truncada"] = bool(np.isfinite(rp) and K.PETRO_APERTURA * rp > r_borde)
    v["R_P"] = rp * pixscale

    r = mapa_radial(n)
    en_ap = r < (r_ap if np.isfinite(r_ap) else 0)
    v["frac_vecinos"] = float(vec[en_ap].mean()) if en_ap.any() else float("nan")
    v["n_fuentes"] = n_fuentes
    v["frac_sin_datos"] = float(sin_datos.mean(axis=(1, 2)).max())

    flujos = {}
    for b, nombre in enumerate(K.BANDAS):
        v[f"sigma_cielo_{nombre}"] = float(sigma[b])
        f = float(perfiles[b].F(r_ap)) if np.isfinite(r_ap) else float("nan")
        flujos[nombre] = f
        v[f"flujo_{nombre}"] = f
        v[f"m_{nombre}"] = magnitud(f)

    # Radios en r (fracciones del flujo de Petrosian, como SDSS).
    pr = perfiles[1]
    for frac, nombre in ((0.2, "R20"), (0.5, "R50"), (0.8, "R80"), (0.9, "R90")):
        v[nombre] = radio_fraccion(pr, flujos["r"], frac, r_ap) * pixscale if np.isfinite(r_ap) else float("nan")

    # Tamaño de cada banda respecto de r: R50 de la banda con su propio flujo en la misma apertura.
    for b, nombre in ((0, "g"), (2, "z")):
        r50b = radio_fraccion(perfiles[b], flujos[nombre], 0.5, r_ap) if np.isfinite(r_ap) else float("nan")
        v[f"ratio_R50_{nombre}_r"] = r50b * pixscale / v["R50"] if v["R50"] > 0 else float("nan")

    # Colores interior (< R50) y exterior (R50–R90): gradiente de color.
    r50, r90 = v["R50"] / pixscale, v["R90"] / pixscale
    for b, nombre in enumerate(K.BANDAS):
        fin = float(perfiles[b].F(r50)) if np.isfinite(r50) else float("nan")
        fout = float(perfiles[b].F(r90) - perfiles[b].F(r50)) if np.isfinite(r90) else float("nan")
        v[f"m_in_{nombre}"] = magnitud(fin) if np.isfinite(fin) else float("nan")
        v[f"m_out_{nombre}"] = magnitud(fout) if np.isfinite(fout) else float("nan")

    return Medicion(limpio, sigma, gal, vec, perfiles, rp, r_ap, v)
