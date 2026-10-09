"""Recorre la caché de recortes y mide las características de cada galaxia.

La caché (`resultados_katachi_fits/cache/desi/recorte300/chunk_XXX.npz`) la construye
el cuaderno de Katachi; aquí solo se lee. Cada bloque produce un archivo Parquet con
las características y un .npy con las estampas. Si Colab se desconecta, al volver a
ejecutar se saltan los bloques ya terminados.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .fotometria import medir_fotometria
from .imagenes import estampa
from .morfologia import medir_morfologia


def medir_galaxia(cubo: np.ndarray) -> tuple[dict, np.ndarray]:
    """Todas las características medibles en la imagen de una galaxia y su estampa.
    Si algo falla, devuelve `estado = "fallo"` con el motivo y valores ausentes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = medir_fotometria(cubo)
            mo = medir_morfologia(m)
            est = estampa(m.limpio, m.rp_px, mo.get("pa_rad", np.nan))
            return {**m.valores, **mo, "estado": "ok", "motivo": ""}, est
        except Exception as e:  # una galaxia problemática no detiene la medición
            return {"estado": "fallo", "motivo": f"{type(e).__name__}: {e}"[:200]}, None


def _guardar_npy(ruta: Path, arr: np.ndarray) -> None:
    with open(ruta, "wb") as f:     # con el archivo abierto, np.save no añade ".npy" al nombre temporal
        np.save(f, arr)


def _tarea(args):
    x, = args
    return medir_galaxia(np.asarray(x, dtype=np.float32))


def medir_cache(cache_dir: Path, out_dir: Path, guard, ids_validos=None, workers: int | None = None,
                n_estampa: int = 32, log=print) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Mide todas las galaxias de la caché. Devuelve (características, estampas, mangaid)."""
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    guard.mkdir(out_dir)
    bloques = sorted(cache_dir.glob("chunk_*.npz"))
    if not bloques:
        raise FileNotFoundError(f"No hay bloques de caché en {cache_dir}")
    workers = workers or max(1, (os.cpu_count() or 2))
    t0 = time.time()
    for i, b in enumerate(bloques):
        f_tab = out_dir / f"caracteristicas_{b.stem}.parquet"
        f_est = out_dir / f"estampas_{b.stem}.npy"
        if f_tab.exists() and f_est.exists():
            continue
        with np.load(b) as z:
            x, ids, frac0 = z["x"], z["mangaid"].astype(str), z["frac_cero"]
        sel = np.ones(len(ids), bool) if ids_validos is None else np.isin(ids, list(ids_validos))
        x, ids, frac0 = x[sel], ids[sel], frac0[sel]
        with cf.ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_tarea, [(xi,) for xi in x], chunksize=8))
        filas = []
        est = np.full((len(ids), 3, n_estampa, n_estampa), np.nan, np.float32)
        for k, (v, e) in enumerate(res):
            filas.append({"mangaid": ids[k], **v, "frac_cero_cache": float(frac0[k].max())})
            if e is not None:
                est[k] = e
        tab = pd.DataFrame(filas)
        guard.write(f_est, lambda t, est=est: _guardar_npy(t, est))
        guard.write(f_tab, lambda t, tab=tab: tab.to_parquet(t, index=False))
        log(f"[medición] bloque {i + 1}/{len(bloques)} · {len(ids)} galaxias · "
            f"fallos: {(tab.estado != 'ok').sum()} · {time.time() - t0:.0f} s")
    return cargar(out_dir)


def cargar(out_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Lee todos los bloques ya medidos."""
    out_dir = Path(out_dir)
    tabs = sorted(out_dir.glob("caracteristicas_chunk_*.parquet"))
    df = pd.concat([pd.read_parquet(t) for t in tabs], ignore_index=True)
    est = np.concatenate([np.load(out_dir / t.name.replace("caracteristicas_", "estampas_").replace(".parquet", ".npy"))
                          for t in tabs])
    return df, est, df["mangaid"].to_numpy()
