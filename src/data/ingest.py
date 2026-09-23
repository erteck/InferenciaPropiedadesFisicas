"""Etapa 1: construir el manifiesto de objetivos y reunir las imágenes crudas.

Salidas
-------
Datos/raw/manifest.csv   MaNGAID, RA, DEC y etiquetas de referencia por galaxia.
Datos/raw/fits/          Imágenes FITS crudas, una por objetivo.

La descarga de recortes DESI Legacy Survey todavía no está implementada; por
ahora la etapa copia las imágenes locales listadas en ``ingest.local_fits``
para poder ejecutar el pipeline de extremo a extremo.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from astropy.io import fits

from .common import RAW, ROOT, ensure_dir, load_params, write_json


def desi_cutout_url(ra: float, dec: float, bands: list[str], size_px: int, pixscale: float, layer: str) -> str:
    return (
        "https://www.legacysurvey.org/viewer/fits-cutout"
        f"?ra={ra:.6f}&dec={dec:.6f}&layer={layer}&pixscale={pixscale}"
        f"&size={size_px}&bands={''.join(bands)}"
    )


def build_manifest(catalog_csv: Path, labels_csv: Path, limit: int) -> pd.DataFrame:
    cat = pd.read_csv(catalog_csv).rename(columns={"MaNGAID": "MANGAID", "OBJRA": "RA", "OBJDEC": "DEC"})
    labels = pd.read_csv(labels_csv)
    df = cat.merge(labels, on="MANGAID", how="left").drop_duplicates("MANGAID")
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def is_image_fits(path: Path) -> bool:
    with fits.open(path) as hdul:
        return any(getattr(h, "is_image", False) and h.data is not None and h.data.ndim >= 2 for h in hdul)


def main() -> None:
    p = load_params("ingest")
    ensure_dir(RAW)
    fits_dir = ensure_dir(RAW / "fits")

    manifest = build_manifest(ROOT / p["catalog"], ROOT / p["labels"], int(p.get("limit", 0)))
    manifest["cutout_url"] = [
        desi_cutout_url(r.RA, r.DEC, p["bands"], p["size_px"], p["pixscale_arcsec"], p["survey"])
        for r in manifest.itertuples()
    ]
    manifest.to_csv(RAW / "manifest.csv", index=False)

    staged = []
    for src in p.get("local_fits", []):
        for f in sorted((ROOT / src).glob("*.fits")):
            if is_image_fits(f):
                shutil.copy2(f, fits_dir / f.name)
                staged.append(f.name)

    write_json(RAW / "ingest_summary.json", {"n_targets": len(manifest), "staged_fits": staged})
    print(f"[ingest] {len(manifest)} objetivos, {len(staged)} FITS locales copiados")


if __name__ == "__main__":
    main()
