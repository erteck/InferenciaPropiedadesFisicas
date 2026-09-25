"""Evaluar máscaras candidatas sobre los cubos DESI y producir tabla + figuras.

Salidas
-------
Datos/interim/mask_eval/metrics.csv        una fila por (galaxia, máscara)
Datos/interim/mask_eval/summary.json       promedios por máscara
Datos/interim/mask_eval/figures/*.png      RGB + contornos + curva de crecimiento
Datos/interim/mask_eval/figures/grid.png   las diez galaxias en una lámina
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import DATOS, INTERIM, ROOT, ensure_dir, load_params, write_json
from .desi import Cube, load_catalog, load_cube, lupton_rgb
from .mask_eval import background, circle, curve_of_growth, evaluate_mask, neighbor_mask, surface_brightness_mask, threshold_mask

BAND_COLORS = {"g": "#4daf4a", "r": "#e41a1c", "z": "#984ea3", "grz": "#ffffff"}
MASK_COLORS = {"thr_g": "#4daf4a", "thr_r": "#e41a1c", "thr_z": "#984ea3", "thr_grz": "#ffffff", "sb_r": "#00ffff"}


def candidate_masks(cube: Cube, p: dict, iso_sb: float) -> dict[str, np.ndarray]:
    masks = {}
    for band in ("g", "r", "z"):
        masks[f"thr_{band}"] = threshold_mask(cube.bands[band], cube.center, p["nsigma"], p["smooth_px"], p["grow_px"])
    combined = sum(cube.bands[b] - background(cube.bands[b])[0] for b in ("g", "r", "z"))
    masks["thr_grz"] = threshold_mask(combined, cube.center, p["nsigma"], p["smooth_px"], p["grow_px"])
    masks["sb_r"] = surface_brightness_mask(cube.bands["r"], cube.center, cube.pixscale, iso_sb, p["smooth_px"], p["grow_px"])
    return masks


def plot_galaxy(cube: Cube, masks: dict[str, np.ndarray], row: dict, petro_th90: float, ref_factor: float, out: Path) -> None:
    r90_px = cube.arcsec_to_px(petro_th90)
    half = int(min(ref_factor * r90_px * 1.3, cube.shape[0] // 2))
    cx, cy = (int(round(c)) for c in cube.center)
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    rgb = lupton_rgb(cube)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    axes[0].imshow(rgb[sl], origin="lower")
    axes[0].set_title(f"{cube.mangaid}  ({row.get('name', '')})\nR90 = {petro_th90:.1f}\"  T = {row.get('T_11', float('nan')):.0f}  log M* = {row.get('LogMassNSA', float('nan')):.2f}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(rgb[sl], origin="lower")
    for name, m in masks.items():
        axes[1].contour(m[sl], levels=[0.5], colors=[MASK_COLORS[name]], linewidths=1.2)
    for k, ls in ((1.0, "-"), (ref_factor, "--")):
        axes[1].contour(circle(cube.shape, cube.center, k * r90_px)[sl], levels=[0.5], colors=["yellow"], linewidths=0.8, linestyles=ls)
    axes[1].set_title("umbral g/r/z/grz = verde/rojo/morado/blanco; SB25(r) cian\nvecinos excluidos naranja; amarillo = R90 y 2·R90", fontsize=9)
    axes[1].axis("off")

    ax = axes[2]
    neighbors = neighbor_mask(sum(b - background(b)[0] for b in cube.bands.values()), cube.center)
    axes[1].contour(neighbors[sl], levels=[0.5], colors=["orange"], linewidths=0.6, linestyles="dotted")
    for band, img in cube.bands.items():
        sub = np.where(neighbors, 0.0, img - background(img)[0])
        radii, flux = curve_of_growth(sub, cube.center, ref_factor * r90_px)
        ax.plot(radii * cube.pixscale, flux / flux[-1], color=BAND_COLORS[band], label=f"banda {band}")
    for name, m in masks.items():
        r_eq = np.sqrt(m.sum() / np.pi) * cube.pixscale
        ax.axvline(r_eq, color=MASK_COLORS[name] if name != "thr_grz" else "k", ls=":", lw=1)
    ax.axvline(petro_th90, color="gold", ls="-", lw=1, label="PETRO_TH90")
    ax.axhline(0.9, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("radio [arcsec]")
    ax.set_ylabel(f"flujo acumulado / flujo en {ref_factor:g}·R90")
    ax.set_title("curva de crecimiento sin vecinos\nlíneas punteadas = radio equivalente de cada máscara", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_grid(cubes: list[Cube], best_masks: dict[str, np.ndarray], catalog: dict, ref_factor: float, out: Path) -> None:
    n = len(cubes)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    for ax, cube in zip(axes.flat, cubes):
        r90_px = cube.arcsec_to_px(catalog[cube.mangaid]["PETRO_TH90"])
        half = int(min(ref_factor * r90_px * 1.3, cube.shape[0] // 2))
        cx, cy = (int(round(c)) for c in cube.center)
        sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        ax.imshow(lupton_rgb(cube)[sl], origin="lower")
        ax.contour(best_masks[cube.mangaid][sl], levels=[0.5], colors=["white"], linewidths=1.0)
        ax.contour(circle(cube.shape, cube.center, r90_px)[sl], levels=[0.5], colors=["yellow"], linewidths=0.7)
        ax.set_title(cube.mangaid, fontsize=9)
        ax.axis("off")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Máscara por brillo superficial (r < 25 mag/arcsec², blanco) vs. PETRO_TH90 (amarillo)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    p = load_params("mask_eval")
    out_dir = ensure_dir(INTERIM / "mask_eval")
    fig_dir = ensure_dir(out_dir / "figures")
    catalog = load_catalog(ROOT / p["catalog"])
    cube_paths = sorted((ROOT / p["cubes"]).glob("*.fits"))

    records, cubes, grid_masks = [], [], {}
    for path in cube_paths:
        cube = load_cube(path)
        row = catalog.get(cube.mangaid)
        if row is None:
            print(f"[mask_eval] {cube.mangaid} no está en el catálogo, se omite")
            continue
        masks = candidate_masks(cube, p["threshold"], p["iso_sb"])
        for name, m in masks.items():
            rep = evaluate_mask(cube, m, name, row["PETRO_TH90"], p["ref_factor"], p["iso_sb"])
            records.append(rep.flat())
        plot_galaxy(cube, masks, row, row["PETRO_TH90"], p["ref_factor"], fig_dir / f"{cube.mangaid}.png")
        cubes.append(cube)
        grid_masks[cube.mangaid] = masks["sb_r"]
        print(f"[mask_eval] {cube.mangaid}: light_frac r = {records[-1]['light_frac_r']:.3f}, area_ratio = {records[-1]['area_ratio']:.2f}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")
    numeric = df.drop(columns=["mangaid"]).groupby("mask").agg(["mean", "std"])
    summary = {mask: {f"{col}_{stat}": float(v) for (col, stat), v in numeric.loc[mask].items()} for mask in numeric.index}
    write_json(out_dir / "summary.json", {"params": p, "n_galaxies": len(cubes), "by_mask": summary})
    if cubes:
        plot_grid(cubes, grid_masks, catalog, p["ref_factor"], fig_dir / "grid.png")
    print(f"[mask_eval] {len(cubes)} galaxias, {len(df)} filas -> {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
