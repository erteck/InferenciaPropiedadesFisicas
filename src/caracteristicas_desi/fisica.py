"""Variables físicas: extinción galáctica, distancias y cantidades intrínsecas.

Fuentes:
- Mapas de polvo de Schlegel, Finkbeiner & Davis (1998), consultados con `dustmaps`
  (Green 2018), y coeficientes A/E(B−V) de Legacy Surveys DR10, basados en
  Schlafly & Finkbeiner (2011).
- Cosmología ΛCDM plana con H0 = 73 km/s/Mpc y Ωm = 0.3, la que usa Pipe3D para
  las etiquetas (Sánchez et al. 2022), de modo que magnitudes y masas sean coherentes.
- Masa fotométrica de Ebrová et al. (2025), calibrada para Legacy Surveys, y la
  relación color–M/L de Bell et al. (2003).
- No se aplica corrección K: a la mediana de MaNGA (z ≈ 0.04) es de unas centésimas
  de magnitud en g y r; el redshift se conserva como variable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as K


def ebv_sfd(ra: np.ndarray, dec: np.ndarray, map_dir: Path) -> np.ndarray:
    """E(B−V) del mapa SFD en coordenadas ICRS (grados)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from dustmaps.sfd import SFDQuery

    q = SFDQuery(map_dir=str(map_dir))
    return np.asarray(q(SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg, frame="icrs")), float)


def cosmologia():
    from astropy.cosmology import FlatLambdaCDM

    return FlatLambdaCDM(H0=K.H0, Om0=K.OMEGA_M)


def distancias(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Módulo de distancia [mag] y escala física [kpc/″] para cada redshift (NaN si falta)."""
    z = np.asarray(z, float)
    dm = np.full(z.shape, np.nan)
    escala = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0)
    if ok.any():
        cos = cosmologia()
        dm[ok] = cos.distmod(z[ok]).value
        escala[ok] = cos.kpc_proper_per_arcmin(z[ok]).value / 60.0
    return dm, escala


def agregar_variables_fisicas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega magnitudes corregidas por extinción, colores, gradientes de color,
    brillo superficial, magnitudes absolutas, luminosidad, tamaños físicos y masas
    fotométricas. Requiere columnas `m_*`, `m_in_*`, `m_out_*`, `R50`, `ebv` y `z`."""
    d = df.copy()
    for b in K.BANDAS:
        a = K.EXTINCION[b] * d["ebv"]
        d[f"m_{b}0"] = d[f"m_{b}"] - a
        d[f"m_in_{b}0"] = d[f"m_in_{b}"] - a
        d[f"m_out_{b}0"] = d[f"m_out_{b}"] - a
        # Brillo superficial medio dentro de R50 (mitad del flujo en π R50²).
        d[f"mu50_{b}"] = d[f"m_{b}0"] + 2.5 * np.log10(2 * np.pi * d["R50"] ** 2)

    d["g_r"] = d["m_g0"] - d["m_r0"]
    d["r_z"] = d["m_r0"] - d["m_z0"]
    d["g_z"] = d["m_g0"] - d["m_z0"]
    for c1, c2 in (("g", "r"), ("r", "z")):
        d[f"{c1}_{c2}_in"] = d[f"m_in_{c1}0"] - d[f"m_in_{c2}0"]
        d[f"{c1}_{c2}_out"] = d[f"m_out_{c1}0"] - d[f"m_out_{c2}0"]
        d[f"delta_{c1}_{c2}"] = d[f"{c1}_{c2}_in"] - d[f"{c1}_{c2}_out"]

    d["C_sdss"] = d["R90"] / d["R50"]

    dm, escala = distancias(d["z"].to_numpy())
    d["DM"] = dm
    for b in K.BANDAS:
        d[f"M_{b}"] = d[f"m_{b}0"] - dm
    d["log_L_r"] = 0.4 * (K.M_SOL_R - d["M_r"])
    r50_kpc = d["R50"] * escala
    d["log_R50_kpc"] = np.log10(r50_kpc)
    area = np.log10(2 * np.pi * r50_kpc**2)
    d["log_SigmaL"] = d["log_L_r"] - area

    a, b, c = K.EBROVA
    d["logM_ebrova"] = a * d["M_g"] + b * d["M_r"] + c
    a_r, b_r = K.BELL_R
    d["logM_bell"] = d["log_L_r"] + a_r + b_r * d["g_r"]
    d["log_SigmaM"] = d["logM_ebrova"] - area
    return d
