"""Lectura de cubos DESI Legacy Survey (g, r, z en un solo FITS) y catálogo maestro."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS

NANOMAGGIE_ZP = 22.5


@dataclass
class Cube:
    mangaid: str
    bands: dict[str, np.ndarray]
    header: fits.Header
    wcs: WCS
    pixscale: float

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.bands.values())).shape

    @property
    def center(self) -> tuple[float, float]:
        return (self.header["CRPIX1"] - 1.0, self.header["CRPIX2"] - 1.0)

    def arcsec_to_px(self, arcsec: float) -> float:
        return arcsec / self.pixscale


def load_cube(path: Path) -> Cube:
    with fits.open(path) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
    names = [header.get(f"BAND{i}", str(i)).strip() for i in range(data.shape[0])]
    wcs = WCS(header).celestial
    pixscale = float(abs(header["CD2_2"]) * 3600.0)
    mangaid = Path(path).name.replace("_grz.fits", "").replace(".fits", "")
    return Cube(mangaid, dict(zip(names, data)), header, wcs, pixscale)


def lupton_rgb(cube: Cube, stretch: float = 0.5, q: float = 8.0, minimum: float = 0.0) -> np.ndarray:
    return make_lupton_rgb(cube.bands["z"], cube.bands["r"], cube.bands["g"], stretch=stretch, Q=q, minimum=minimum)


def surface_brightness(flux_per_px: np.ndarray, pixscale: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return NANOMAGGIE_ZP - 2.5 * np.log10(flux_per_px / pixscale**2)


def load_catalog(path: Path) -> dict[str, dict]:
    table = Table.read(path)
    rows = {}
    for row in table:
        key = str(row["MANGAID"]).strip()
        rows[key] = {name: (str(row[name]).strip() if table[name].dtype.kind in "SU" else float(row[name])) for name in table.colnames}
    return rows
