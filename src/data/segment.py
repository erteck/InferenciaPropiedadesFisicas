"""Etapa 4: segmentación de la galaxia objetivo.

Entrada  Datos/interim/cutouts/*.fits  (flujo lineal)
Salida   Datos/interim/masks/*.npy     (uint8, 1 = galaxia objetivo)

Método actual: el fondo se estima con sigma-clipping sobre el marco exterior
de la imagen, se umbraliza y se conserva la componente conexa que contiene el
centro. Se sustituirá por U-Net / contornos activos.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

from .common import INTERIM, ensure_dir, load_params, write_json


def border_pixels(img: np.ndarray, frac: float = 0.1) -> np.ndarray:
    ny, nx = img.shape
    by, bx = max(int(ny * frac), 1), max(int(nx * frac), 1)
    ring = np.ones_like(img, dtype=bool)
    ring[by:-by, bx:-bx] = False
    return img[ring]


def threshold_mask(img: np.ndarray, sigma: float, min_area_px: int) -> np.ndarray:
    img = np.nan_to_num(img.astype(np.float32))
    _, median, std = sigma_clipped_stats(border_pixels(img), sigma=3.0)
    binary = img > median + sigma * std
    labels, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(img, dtype=np.uint8)
    cy, cx = (s // 2 for s in img.shape)
    target = labels[cy, cx]
    if target == 0:
        sizes = ndimage.sum(binary, labels, range(1, n + 1))
        target = int(np.argmax(sizes)) + 1
    mask = labels == target
    if mask.sum() < min_area_px:
        return np.zeros_like(img, dtype=np.uint8)
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def main() -> None:
    p = load_params("segment")
    out_dir = ensure_dir(INTERIM / "masks")
    coverage = {}
    for f in sorted((INTERIM / "cutouts").glob("*.fits")):
        if p["method"] != "threshold":
            raise NotImplementedError(f"segmentación '{p['method']}' no implementada todavía")
        mask = threshold_mask(fits.getdata(f), float(p["sigma"]), int(p["min_area_px"]))
        np.save(out_dir / f"{f.stem}.npy", mask)
        coverage[f.stem] = float(mask.mean())
    write_json(INTERIM / "segment_summary.json", {"params": p, "mask_fraction": coverage})
    print(f"[segment] {len(coverage)} máscaras generadas")


if __name__ == "__main__":
    main()
