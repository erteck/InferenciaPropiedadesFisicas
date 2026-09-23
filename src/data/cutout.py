"""Etapa 2: recorte uniforme centrado en la galaxia objetivo.

Entrada  Datos/raw/fits/*.fits
Salida   Datos/interim/cutouts/*.fits  (imagen recortada, mismo nombre)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D

from .common import INTERIM, RAW, ensure_dir, load_params, write_json


def first_image_hdu(hdul: fits.HDUList) -> fits.ImageHDU:
    for h in hdul:
        if getattr(h, "is_image", False) and h.data is not None and h.data.ndim >= 2:
            return h
    raise ValueError("el archivo no contiene una imagen 2D")


def pixel_radius(radius_arcsec: float, header: fits.Header, fallback_pixscale: float) -> int:
    pixscale = None
    for key in ("PIXSCALE", "CDELT2", "CD2_2"):
        if key in header and header[key]:
            pixscale = abs(float(header[key]))
            if key != "PIXSCALE":
                pixscale *= 3600.0
            break
    if pixscale is None:
        pixscale = fallback_pixscale
    return max(int(round(radius_arcsec / pixscale)), 8)


def cutout_image(data: np.ndarray, header: fits.Header, radius_arcsec: float, fallback_pixscale: float) -> np.ndarray:
    ny, nx = data.shape[-2:]
    r = min(pixel_radius(radius_arcsec, header, fallback_pixscale), ny // 2, nx // 2)
    center = (nx / 2.0, ny / 2.0)
    return Cutout2D(data.astype(np.float32), center, (2 * r, 2 * r), mode="trim").data


def main() -> None:
    p = load_params("cutout")
    ingest = load_params("ingest")
    out_dir = ensure_dir(INTERIM / "cutouts")
    processed = {}
    for f in sorted((RAW / "fits").glob("*.fits")):
        with fits.open(f) as hdul:
            hdu = first_image_hdu(hdul)
            cut = cutout_image(hdu.data, hdu.header, float(p["radius_arcsec"]), float(ingest["pixscale_arcsec"]))
            hdr = hdu.header.copy()
            hdr["HISTORY"] = f"cutout radius_arcsec={p['radius_arcsec']} center={p['center']}"
        fits.PrimaryHDU(cut, header=hdr).writeto(out_dir / f.name, overwrite=True)
        processed[f.name] = list(cut.shape)
    write_json(INTERIM / "cutout_summary.json", processed)
    print(f"[cutout] {len(processed)} imágenes recortadas")


if __name__ == "__main__":
    main()
