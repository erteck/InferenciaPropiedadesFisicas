"""Métricas de calidad de una máscara de galaxia sobre un cubo DESI.

Responde a "¿la máscara capturó toda la luz de la galaxia?" con números:

light_fraction   flujo dentro de la máscara / flujo dentro de la apertura de
                 referencia (círculo de ``ref_factor`` × PETRO_TH90), por banda.
contamination    fracción del flujo dentro de la máscara que proviene de fuentes
                 detectadas que no son la galaxia central.
area_ratio       área de la máscara / área del círculo de radio PETRO_TH90.
r_mask_arcsec    radio equivalente de la máscara, sqrt(área / pi).
r_iso_arcsec     radio isofotal por banda al brillo superficial ``iso_sb``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from scipy import ndimage

from .desi import NANOMAGGIE_ZP, Cube, surface_brightness


def background(img: np.ndarray, border_frac: float = 0.1) -> tuple[float, float]:
    ny, nx = img.shape
    by, bx = int(ny * border_frac), int(nx * border_frac)
    ring = np.ones(img.shape, dtype=bool)
    ring[by:-by, bx:-bx] = False
    _, median, std = sigma_clipped_stats(img[ring], sigma=3.0)
    return float(median), float(std)


def radial_grid(shape: tuple[int, int], center: tuple[float, float]) -> np.ndarray:
    yy, xx = np.indices(shape)
    return np.hypot(xx - center[0], yy - center[1])


def circle(shape: tuple[int, int], center: tuple[float, float], radius_px: float) -> np.ndarray:
    return radial_grid(shape, center) <= radius_px


def keep_central_component(binary: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    labels, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(binary, dtype=bool)
    cx, cy = int(round(center[0])), int(round(center[1]))
    target = labels[cy, cx]
    if target == 0:
        dist = ndimage.distance_transform_edt(labels == 0, return_indices=True)[1]
        target = labels[dist[0][cy, cx], dist[1][cy, cx]]
    return ndimage.binary_fill_holes(labels == target)


def threshold_mask(img: np.ndarray, center: tuple[float, float], nsigma: float, smooth_px: float, grow_px: int) -> np.ndarray:
    med, _ = background(img)
    smooth = ndimage.gaussian_filter(img - med, smooth_px) if smooth_px > 0 else img - med
    med_s, std_s = background(smooth)
    mask = keep_central_component(smooth > med_s + nsigma * std_s, center)
    if grow_px > 0:
        mask = ndimage.binary_dilation(mask, iterations=grow_px)
    return mask


def surface_brightness_mask(img: np.ndarray, center: tuple[float, float], pixscale: float, iso_sb: float, smooth_px: float, grow_px: int) -> np.ndarray:
    med, _ = background(img)
    smooth = ndimage.gaussian_filter(img - med, smooth_px) if smooth_px > 0 else img - med
    limit = pixscale**2 * 10 ** (-0.4 * (iso_sb - NANOMAGGIE_ZP))
    mask = keep_central_component(smooth > limit, center)
    if grow_px > 0:
        mask = ndimage.binary_dilation(mask, iterations=grow_px)
    return mask


def curve_of_growth(img: np.ndarray, center: tuple[float, float], r_max_px: float, step_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r = radial_grid(img.shape, center)
    radii = np.arange(step_px, r_max_px + step_px, step_px)
    flux = np.array([img[r <= rr].sum() for rr in radii])
    return radii, flux


def isophotal_radius(img: np.ndarray, center: tuple[float, float], pixscale: float, iso_sb: float, r_max_px: float) -> float:
    r = radial_grid(img.shape, center)
    edges = np.arange(0, r_max_px + 2, 2.0)
    prof = np.array([img[(r >= a) & (r < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    sb = surface_brightness(np.clip(prof, 1e-12, None), pixscale)
    inside = np.where(sb < iso_sb)[0]
    if len(inside) == 0:
        return float("nan")
    last = inside[-1]
    return float(edges[last + 1] * pixscale)


def neighbor_mask(img: np.ndarray, center: tuple[float, float], nsigma: float = 3.0, grow_px: int = 3) -> np.ndarray:
    med, std = background(img)
    seg = detect_sources(img - med, nsigma * std, npixels=5)
    if seg is None:
        return np.zeros(img.shape, dtype=bool)
    cx, cy = int(round(center[0])), int(round(center[1]))
    central = seg.data[cy, cx]
    others = (seg.data > 0) & (seg.data != central)
    return ndimage.binary_dilation(others, iterations=grow_px)


def contamination_fraction(img: np.ndarray, mask: np.ndarray, center: tuple[float, float], nsigma: float = 3.0) -> float:
    med, std = background(img)
    seg = detect_sources(img - med, nsigma * std, npixels=5)
    if seg is None:
        return 0.0
    cx, cy = int(round(center[0])), int(round(center[1]))
    central = seg.data[cy, cx]
    others = (seg.data > 0) & (seg.data != central) & mask
    total = float(((img - med) * mask).sum())
    return float(((img - med) * others).sum() / total) if total > 0 else float("nan")


@dataclass
class MaskReport:
    mangaid: str
    mask_name: str
    petro_th90_arcsec: float
    r_mask_arcsec: float
    area_ratio: float
    light_fraction: dict[str, float] = field(default_factory=dict)
    contamination: dict[str, float] = field(default_factory=dict)
    r_iso_arcsec: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "mangaid": self.mangaid,
            "mask": self.mask_name,
            "petro_th90_arcsec": self.petro_th90_arcsec,
            "r_mask_arcsec": self.r_mask_arcsec,
            "area_ratio": self.area_ratio,
        }
        for b, v in self.light_fraction.items():
            row[f"light_frac_{b}"] = v
        for b, v in self.contamination.items():
            row[f"contam_{b}"] = v
        for b, v in self.r_iso_arcsec.items():
            row[f"r_iso_{b}_arcsec"] = v
        return row


def evaluate_mask(cube: Cube, mask: np.ndarray, mask_name: str, petro_th90: float, ref_factor: float, iso_sb: float) -> MaskReport:
    center = cube.center
    r90_px = cube.arcsec_to_px(petro_th90)
    ref = circle(cube.shape, center, ref_factor * r90_px)
    area_px = float(mask.sum())
    report = MaskReport(
        mangaid=cube.mangaid,
        mask_name=mask_name,
        petro_th90_arcsec=petro_th90,
        r_mask_arcsec=float(np.sqrt(area_px / np.pi) * cube.pixscale),
        area_ratio=area_px / (np.pi * r90_px**2),
    )
    neighbors = neighbor_mask(sum(b - background(b)[0] for b in cube.bands.values()), center)
    for band, img in cube.bands.items():
        med, _ = background(img)
        clean = np.where(neighbors, 0.0, img - med)
        ref_flux = float(clean[ref].sum())
        report.light_fraction[band] = float(clean[mask].sum() / ref_flux) if ref_flux > 0 else float("nan")
        report.contamination[band] = contamination_fraction(img, mask, center)
        report.r_iso_arcsec[band] = isophotal_radius(clean, center, cube.pixscale, iso_sb, ref_factor * r90_px)
    return report
