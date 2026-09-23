"""Etapa 5: mapas de estructura interna (CAS: asimetría, grumosidad).

Entrada  Datos/interim/normalized/*.npy, Datos/interim/masks/*.npy
Salida   Datos/interim/structure_maps/*.npz  (un arreglo por mapa)
         Datos/interim/structure_summary.json (escalares por galaxia)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .common import INTERIM, ensure_dir, load_params, write_json


def asymmetry_map(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rotated = np.rot90(img, 2)
    return np.abs(img - rotated) * mask


def clumpiness_map(img: np.ndarray, mask: np.ndarray, smooth_sigma: float = 2.0) -> np.ndarray:
    smooth = ndimage.gaussian_filter(img, smooth_sigma)
    return np.clip(img - smooth, 0, None) * mask


MAPS = {"asymmetry": asymmetry_map, "clumpiness": clumpiness_map}


def main() -> None:
    p = load_params("structure")
    out_dir = ensure_dir(INTERIM / "structure_maps")
    scalars = {}
    for f in sorted((INTERIM / "normalized").glob("*.npy")):
        img = np.load(f)
        mask = np.load(INTERIM / "masks" / f.name).astype(bool)
        maps = {name: MAPS[name](img, mask) for name in p["maps"]}
        np.savez_compressed(out_dir / f"{f.stem}.npz", **maps)
        denom = max(float((img * mask).sum()), 1e-8)
        scalars[f.stem] = {name: float(m.sum() / denom) for name, m in maps.items()}
    write_json(INTERIM / "structure_summary.json", {"params": p, "scalars": scalars})
    print(f"[structure] {len(scalars)} galaxias, mapas: {', '.join(p['maps'])}")


if __name__ == "__main__":
    main()
