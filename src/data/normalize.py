"""Etapa 3: normalización de intensidades.

Entrada  Datos/interim/cutouts/*.fits
Salida   Datos/interim/normalized/*.npy  (float32 en [0, 1])
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from .common import INTERIM, ensure_dir, load_params, write_json


def normalize_array(img: np.ndarray, method: str, softening: float, clip_percentile: float) -> np.ndarray:
    x = np.nan_to_num(img.astype(np.float32))
    lo, hi = np.percentile(x, [100.0 - clip_percentile, clip_percentile])
    x = np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    if method == "linear":
        return x
    if method == "asinh":
        return np.arcsinh(x / softening) / np.arcsinh(1.0 / softening)
    if method == "log":
        return np.log1p(x / softening) / np.log1p(1.0 / softening)
    raise ValueError(f"método de normalización desconocido: {method}")


def main() -> None:
    p = load_params("normalize")
    out_dir = ensure_dir(INTERIM / "normalized")
    stats = {}
    for f in sorted((INTERIM / "cutouts").glob("*.fits")):
        img = fits.getdata(f)
        norm = normalize_array(img, p["method"], float(p["softening"]), float(p["clip_percentile"]))
        np.save(out_dir / f"{f.stem}.npy", norm)
        stats[f.stem] = {"mean": float(norm.mean()), "std": float(norm.std())}
    write_json(INTERIM / "normalize_summary.json", {"params": p, "images": stats})
    print(f"[normalize] {len(stats)} imágenes normalizadas con '{p['method']}'")


if __name__ == "__main__":
    main()
