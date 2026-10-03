"""Caché de los cubos FITS DESI (plan §3.3 y §3.5).

Para cada galaxia: leer el FITS → aplicar la variante → recortar el centro →
guardar en float16. Se guarda en bloques (`chunk_XXX.npz`) para poder reanudar;
nunca se reescribe un bloque existente. Cada variante tiene su propia carpeta.

- Variante "desi" (identidad): lee solo las filas centrales de cada banda (3
  lecturas contiguas en lugar de 7.7 MB). Bit a bit igual a `astropy` (test).
- Otras variantes: leen el cubo completo, porque el preprocesamiento necesita el
  cielo y los vecinos; corren en procesos separados (photutils es CPU).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .config import FitsConfig
from .io_guard import WriteGuard
from .variants import DESI, Variant

FITS_BLOCK = 2880
META_COLS = ("mangaid", "redshift", "PETRO_TH90", "log_mstar", "split")


@dataclass
class CubeHeader:
    """Lo que se necesita de la cabecera de un cubo para leerlo y validarlo."""
    data_loc: int
    naxis: tuple[int, int, int]
    bands: tuple[str, str, str]
    crpix: tuple[float, float]
    scale_arcsec: tuple[float, float]   # |CD1_1|, |CD2_2| en ″/px
    rotated: bool                       # CD1_2 o CD2_1 distintos de 0


def read_header(path: Path) -> CubeHeader:
    """Lee y valida la cabecera primaria sin leer los datos."""
    with open(path, "rb") as f:
        raw = b""
        while True:
            block = f.read(FITS_BLOCK)
            if len(block) < FITS_BLOCK:
                raise ValueError(f"Cabecera FITS truncada: {path}")
            raw += block
            if any(block[i:i + 8] == b"END     " for i in range(0, FITS_BLOCK, 80)):
                break
    h = fits.Header.fromstring(raw.decode("ascii"))
    if h.get("BITPIX") != -32 or h.get("NAXIS") != 3:
        raise ValueError(f"Formato inesperado (BITPIX={h.get('BITPIX')}, NAXIS={h.get('NAXIS')}): {path}")
    bands = tuple(str(h.get(f"BAND{i}", "")).strip().lower() for i in range(3))
    scale = (abs(float(h.get("CD1_1", 0))) * 3600, abs(float(h.get("CD2_2", 0))) * 3600)
    rotated = bool(h.get("CD1_2", 0) or h.get("CD2_1", 0))
    return CubeHeader(len(raw), (h["NAXIS1"], h["NAXIS2"], h["NAXIS3"]), bands, (h["CRPIX1"], h["CRPIX2"]),
                      scale, rotated)


def _validate(hdr: CubeHeader, cfg: FitsConfig, path) -> None:
    n = cfg.side_in
    if hdr.naxis != (n, n, 3):
        raise ValueError(f"Dimensiones {hdr.naxis} != {(n, n, 3)}: {path}")
    if hdr.bands != tuple(cfg.bands):
        raise ValueError(f"Orden de bandas {hdr.bands} != {cfg.bands}: {path}")
    if hdr.crpix != (n / 2 + 0.5, n / 2 + 0.5):
        raise ValueError(f"CRPIX {hdr.crpix} no centrado: {path}")
    if any(abs(s - cfg.pixscale_in) > 1e-6 for s in hdr.scale_arcsec) or hdr.rotated:
        raise ValueError(f"Escala/orientación WCS {hdr.scale_arcsec}″ (rotada={hdr.rotated}) != {cfg.pixscale_in}″: {path}")


def read_center(path: Path, cfg: FitsConfig) -> np.ndarray:
    """Recorte central (3, crop_px, crop_px) float32, bandas g,r,z, leyendo solo esas filas."""
    hdr = read_header(path)
    _validate(hdr, cfg, path)
    n, lo, k = cfg.side_in, cfg.crop_lo, cfg.crop_px
    out = np.empty((3, k, k), dtype=np.float32)
    with open(path, "rb", buffering=0) as f:
        for b in range(3):
            f.seek(hdr.data_loc + (b * n * n + lo * n) * 4)
            buf = f.read(k * n * 4)
            if len(buf) != k * n * 4:
                raise ValueError(f"FITS truncado: {path}")
            out[b] = np.frombuffer(buf, dtype=">f4").reshape(k, n)[:, lo:lo + k]
    if not np.isfinite(out).all():
        raise ValueError(f"Valores no finitos en el recorte: {path}")
    return out


def read_full(path: Path, cfg: FitsConfig | None = None) -> np.ndarray:
    """Cubo completo (3, 800, 800) en float32; si se da `cfg`, valida antes la cabecera."""
    if cfg is not None:
        _validate(read_header(path), cfg, path)
    with fits.open(path, mode="readonly", memmap=False) as h:
        return np.asarray(h[0].data, dtype=np.float32)


def crop_center(cube: np.ndarray, cfg: FitsConfig) -> np.ndarray:
    """Recorte central (3, crop_px, crop_px) de un cubo ya leído."""
    lo, k = cfg.crop_lo, cfg.crop_px
    return np.ascontiguousarray(cube[:, lo:lo + k, lo:lo + k])


def _one(args) -> tuple[np.ndarray, dict]:
    """Una galaxia: leer → variante → recortar. A nivel de módulo para usar procesos."""
    path, meta, cfg, variant = args
    if variant.is_identity:
        return read_center(Path(path), cfg), {"estado": "ok", "fallback": False}
    cube = read_full(Path(path), cfg)
    out, info = variant.apply(cube, meta)
    return crop_center(out, cfg), info


def cache_dir(root: Path, variant: Variant, cfg: FitsConfig, tag: str = "") -> Path:
    """Carpeta de la caché de una variante: `<raíz>/<clave de la variante>/recorte<px><tag>`."""
    return Path(root) / variant.key / f"recorte{cfg.crop_px}{tag}"


def build_cache(df: pd.DataFrame, path_col: str, out_dir: Path, guard: WriteGuard, cfg: FitsConfig,
                variant: Variant = DESI, chunk: int = 500, workers: int = 24, log=print) -> None:
    """Escribe chunk_XXX.npz con 'x' (N,3,k,k) en `cfg.storage_dtype`, 'mangaid',
    'frac_cero' (N,3) y 'fallback' (N,). Se reanuda saltando bloques existentes."""
    guard.mkdir(out_dir)
    manifest = Path(out_dir) / "variante.json"
    desc = {"variante": variant.manifest(), "crop_px": cfg.crop_px, "crop_lo": cfg.crop_lo,
            "storage_dtype": cfg.storage_dtype, "n_galaxias": len(df)}
    if manifest.exists():
        prev = json.loads(manifest.read_text())
        if prev != json.loads(json.dumps(desc, default=str)):
            raise ValueError(f"La caché en {out_dir} se creó con otra configuración:\n{prev}")
    else:
        guard.write_json(manifest, desc)

    ids = df["mangaid"].to_numpy()
    paths = df[path_col].to_numpy()
    metas = df[[c for c in META_COLS if c in df.columns]].to_dict("records")
    n_chunks = (len(ids) + chunk - 1) // chunk
    executor = cf.ThreadPoolExecutor if variant.is_identity else cf.ProcessPoolExecutor
    n_fallback = 0
    for c in range(n_chunks):
        target = Path(out_dir) / f"chunk_{c:03d}.npz"
        sl = slice(c * chunk, min((c + 1) * chunk, len(ids)))
        if target.exists():
            with np.load(target) as z:
                n_fallback += int(z["fallback"].sum())
            continue
        with executor(workers) as ex:
            futs = [ex.submit(_one, (p, m, cfg, variant)) for p, m in zip(paths[sl], metas[sl])]
            res, errores = [], []
            for p, i_, f in zip(paths[sl], ids[sl], futs):
                try:
                    res.append(f.result())
                except NotImplementedError:
                    raise
                except Exception as e:   # FITS ilegible, forma o escala inesperada, no finitos…
                    errores.append({"mangaid": i_, "ruta": p, "error": f"{type(e).__name__}: {e}"})
        if errores:
            f_err = Path(out_dir) / f"errores_lectura_bloque_{c:03d}.csv"
            if not f_err.exists():
                guard.write(f_err, lambda t: pd.DataFrame(errores).to_csv(t, index=False))
            raise RuntimeError(f"{len(errores)} FITS no pasaron los controles de lectura (detalle en {f_err}). "
                               f"Primer caso: {errores[0]}")
        x32 = np.stack([r[0] for r in res])
        # Los ceros (sin cobertura) se cuentan ANTES de pasar a float16: en float16 los
        # valores |x| < 3e-8 se redondean a 0 y contarían como ceros falsos.
        frac0 = (x32 == 0).mean(axis=(2, 3))
        x = x32.astype(cfg.storage_dtype)
        del x32
        infos = [r[1] for r in res]
        fb = np.array([bool(i.get("fallback")) for i in infos])
        n_fallback += int(fb.sum())
        if n_fallback > variant.max_fallback_frac * len(ids):
            raise RuntimeError(f"Fallaron {n_fallback} galaxias (> {variant.max_fallback_frac:.0%}); "
                               f"ejemplo: {next(i for i in infos if i.get('fallback'))}")
        info_json = np.array([json.dumps(i, default=str) for i in infos])

        def writer(tmp: Path, x=x, ids_c=ids[sl], frac0=frac0, fb=fb, info_json=info_json):
            with open(tmp, "wb") as f:
                np.savez(f, x=x, mangaid=ids_c.astype(str), frac_cero=frac0, fallback=fb, info=info_json)

        guard.write(target, writer)
        log(f"[cache {variant.key}] bloque {c + 1}/{n_chunks} ({sl.stop}/{len(ids)}) · respaldos: {n_fallback}")


def load_cache(out_dir: Path, dtype: str = "float16") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (x, mangaid, frac_cero, fallback). Reserva la memoria una sola vez."""
    files = sorted(Path(out_dir).glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"No hay caché en {out_dir}")
    sizes, shape = [], None
    for f in files:
        with np.load(f) as z:
            sizes.append(len(z["mangaid"]))
            shape = shape or z["x"].shape[1:]
    x = np.empty((sum(sizes), *shape), dtype=dtype)
    ids, f0, fb, i = [], [], [], 0
    for f, n in zip(files, sizes):
        with np.load(f) as z:
            x[i:i + n] = z["x"]
            ids.append(z["mangaid"]); f0.append(z["frac_cero"])
            fb.append(z["fallback"] if "fallback" in z else np.zeros(n, bool))
        i += n
    return x, np.concatenate(ids), np.concatenate(f0), np.concatenate(fb)


def sky_sigma(paths: list[Path], cfg: FitsConfig, threads: int = 16) -> np.ndarray:
    """σ_b del cielo por banda: MAD robusta del anillo r > sky_ring_px del campo de
    800 px de los cubos CRUDOS. Si la red recibe imágenes remuestreadas, se mide
    después del mismo remuestreo (el remuestreo cambia el ruido). Devuelve la
    mediana sobre galaxias, forma (3,). Se comparte entre variantes."""
    import torch
    import torch.nn.functional as F

    n = cfg.side_in
    yy, xx = np.indices((n, n))
    ring = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) > cfg.sky_ring_px
    factor = cfg.out_px / cfg.crop_px

    def one(p):
        a = read_full(Path(p))
        m = ring
        if cfg.resamples:
            a = F.interpolate(torch.from_numpy(a)[None], scale_factor=factor, mode="bilinear",
                              align_corners=False)[0].numpy()
            m = F.interpolate(torch.from_numpy(ring[None, None].astype(np.float32)), scale_factor=factor,
                              mode="nearest")[0, 0].numpy() > 0.5
        out = []
        for b in range(3):
            v = a[b][m]
            v = v[v != 0]
            med = np.median(v)
            out.append(1.4826 * np.median(np.abs(v - med)))
        return out

    with cf.ThreadPoolExecutor(threads) as ex:
        s = np.array(list(ex.map(one, paths)))
    return np.median(s, axis=0)
