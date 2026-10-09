"""Morfología no paramétrica en la banda r.

Definiciones y parámetros de `statmorph` (Rodriguez-Gomez et al. 2019), que a su
vez siguen a Conselice (2003) para C, A y S, y a Lotz, Primack & Madau (2004) para
Gini y M20. Diferencias deliberadas, documentadas en el cuaderno:
- Aperturas circulares (también para la segmentación de Gini).
- El término de fondo de A y S se calcula analíticamente a partir de σ del cielo,
  suponiendo ruido gaussiano, en lugar de medirse en una caja de cielo.
- El centro de asimetría se busca en una rejilla de medio píxel (±1.5 px), donde la
  rotación de 180° es exacta (sin interpolación).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config as K
from .fotometria import Medicion, mapa_radial, radio_fraccion


def concentracion(m: Medicion) -> float:
    """C = 5 log10(R80 / R20), con los radios relativos al flujo dentro de 1.5 R_P."""
    if not np.isfinite(m.rp_px):
        return float("nan")
    p = m.perfiles[1]
    r_lim = min(K.CAS_EXTENSION * m.rp_px, p.radios[-1])
    total = float(p.F(r_lim))
    r20, r80 = radio_fraccion(p, total, 0.2, r_lim), radio_fraccion(p, total, 0.8, r_lim)
    return float(5 * np.log10(r80 / r20)) if r20 > 0 and r80 > 0 else float("nan")


def asimetria(img: np.ndarray, sigma: float, rp_px: float) -> tuple[float, float, float]:
    """A = Σ|I − I180| / Σ|I| − A_fondo, minimizada sobre el centro de rotación.

    A_fondo = área · E|ruido − ruido'| / Σ|I|, con E|·| = 2σ/√π para ruido gaussiano.
    Devuelve (A, dx, dy): asimetría y desplazamiento del centro óptimo en píxeles.
    """
    if not np.isfinite(rp_px):
        return float("nan"), float("nan"), float("nan")
    n = img.shape[0]
    c = (n - 1) / 2
    r_ap = min(K.CAS_EXTENSION * rp_px, c - 3)
    flip = img[::-1, ::-1]
    yy, xx = np.indices(img.shape, dtype=np.float32)
    mejor = (np.inf, 0, 0, None)
    s_max = K.ASIM_DESPLAZAMIENTO
    for sy in range(-s_max, s_max + 1):
        rot_y = np.roll(flip, sy, axis=0)
        for sx in range(-s_max, s_max + 1):
            rot = np.roll(rot_y, sx, axis=1)
            ap = np.hypot(xx - (c + sx / 2), yy - (c + sy / 2)) < r_ap
            den = np.abs(img[ap]).sum()
            if den <= 0:
                continue
            a = np.abs(img[ap] - rot[ap]).sum() / den
            if a < mejor[0]:
                mejor = (a, sx, sy, ap)
    a_raw, sx, sy, ap = mejor
    if ap is None:
        return float("nan"), float("nan"), float("nan")
    fondo = ap.sum() * (2 * sigma / np.sqrt(np.pi)) / np.abs(img[ap]).sum() if np.isfinite(sigma) else 0.0
    return float(a_raw - fondo), sx / 2, sy / 2


def suavidad(img: np.ndarray, sigma: float, rp_px: float) -> float:
    """S = [Σ(I − I_s)₊ − área · fondo] / Σ I en el anillo 0.25 R_P – 1.5 R_P, con I_s la
    imagen suavizada con una caja de 0.25 R_P (Conselice 2003; Lotz et al. 2004, ec. 11).

    Para ruido gaussiano, I − I_s tiene desviación σ·√(1 − 1/k²) y su parte positiva
    tiene media σ·√(1 − 1/k²)/√(2π); ese es el término de fondo por píxel.
    """
    if not np.isfinite(rp_px):
        return float("nan")
    k = int(K.CAS_FRACCION * rp_px)
    if k < 2:
        return float("nan")                     # la caja no suaviza: S no está definida
    r = mapa_radial(img.shape[0])
    anillo = (r >= K.CAS_FRACCION * rp_px) & (r <= min(K.CAS_EXTENSION * rp_px, r.max()))
    dif = np.clip(img - ndimage.uniform_filter(img, size=k), 0, None)
    total = img[anillo].sum()
    if total <= 0:
        return float("nan")
    fondo = sigma * np.sqrt(1 - 1 / k**2) / np.sqrt(2 * np.pi) if np.isfinite(sigma) else 0.0
    return float((dif[anillo].sum() - anillo.sum() * fondo) / total)


def segmentacion_gini(img: np.ndarray, rp_px: float) -> np.ndarray:
    """Píxeles de la imagen suavizada (σ = 0.2 R_P) por encima de su brillo medio en R_P,
    conectados al centro (Lotz et al. 2004; statmorph)."""
    sm = ndimage.gaussian_filter(img, K.GINI_FRACCION * rp_px, mode="constant")
    r = mapa_radial(img.shape[0])
    anillo = np.abs(r - rp_px) < 0.5
    if not anillo.any():
        return np.zeros(img.shape, bool)
    seg = sm >= sm[anillo].mean()
    lab, nlab = ndimage.label(seg, structure=np.ones((3, 3)))
    if nlab == 0:
        return seg
    c = img.shape[0] // 2
    objetivo = lab[c, c] or lab.flat[np.argmax(np.where(seg, sm, -np.inf))]
    return lab == objetivo


def gini(img: np.ndarray, seg: np.ndarray) -> float:
    """Coeficiente de Gini de |I| dentro de la segmentación (Lotz et al. 2004, ec. 6)."""
    x = np.sort(np.abs(img[seg]))
    n = x.size
    if n <= 1 or x.sum() == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * x) / ((n - 1) * x.sum()))


def m20(img: np.ndarray, seg: np.ndarray) -> float:
    """M20 = log10(Σ_{20 %} M_i / M_tot): segundo momento del 20 % más brillante del
    flujo respecto al total (Lotz et al. 2004, ec. 7–8)."""
    I = np.where(seg, img, 0.0).astype(np.float64)
    tot = I.sum()
    if tot <= 0:
        return float("nan")
    yy, xx = np.indices(I.shape)
    xc, yc = (I * xx).sum() / tot, (I * yy).sum() / tot
    d2 = (xx - xc) ** 2 + (yy - yc) ** 2
    m_tot = (I * d2).sum()
    v = I.ravel()
    orden = np.argsort(v)[::-1]
    acum = np.cumsum(v[orden])
    top = orden[: np.searchsorted(acum, 0.2 * tot) + 1]
    m_top = (v[top] * d2.ravel()[top]).sum()
    return float(np.log10(m_top / m_tot)) if m_top > 0 and m_tot > 0 else float("nan")


def forma(img: np.ndarray, seg: np.ndarray) -> tuple[float, float]:
    """Cociente de ejes b/a y ángulo de posición (rad) a partir de los momentos de
    segundo orden del flujo positivo dentro de la segmentación."""
    I = np.where(seg & (img > 0), img, 0.0).astype(np.float64)
    tot = I.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    yy, xx = np.indices(I.shape)
    xc, yc = (I * xx).sum() / tot, (I * yy).sum() / tot
    mxx = (I * (xx - xc) ** 2).sum() / tot
    myy = (I * (yy - yc) ** 2).sum() / tot
    mxy = (I * (xx - xc) * (yy - yc)).sum() / tot
    l1, l2 = np.linalg.eigvalsh([[mxx, mxy], [mxy, myy]])[::-1]
    return float(np.sqrt(max(l2, 0) / l1)) if l1 > 0 else float("nan"), float(0.5 * np.arctan2(2 * mxy, mxx - myy))


def medir_morfologia(m: Medicion) -> dict:
    """C, A, S, Gini, M20 y b/a en la banda r."""
    img, sig, rp = m.limpio[1], float(m.sigma[1]), m.rp_px
    v = {"C": concentracion(m)}
    v["A"], v["asim_dx"], v["asim_dy"] = asimetria(img, sig, rp)
    v["S"] = suavidad(img, sig, rp)
    if np.isfinite(rp):
        seg = segmentacion_gini(img, rp)
        v["gini"], v["m20"] = gini(img, seg), m20(img, seg)
        v["b_a"], v["pa_rad"] = forma(img, seg)
    else:
        v.update(gini=float("nan"), m20=float("nan"), b_a=float("nan"), pa_rad=float("nan"))
    return v
