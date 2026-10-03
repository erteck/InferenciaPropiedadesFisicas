"""Pruebas del mecanismo de variantes (solo DESI vs DESI + preprocesamiento) y del
almacenamiento en float16."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.katachi_fits.cache import build_cache, cache_dir, load_cache, read_full
from src.katachi_fits.config import FitsConfig
from src.katachi_fits.io_guard import WriteGuard
from src.katachi_fits.variants import DESI, Variant, get

from .test_katachi_fits import SAMPLE_FITS, _write_cube


# Funciones de prueba a nivel de módulo (las variantes corren en procesos).
def sumar_uno(cube, meta, delta=1.0):
    return cube + delta, {"estado": "ok"}


def falla_si_impar(cube, meta):
    if int(meta["mangaid"].split("-")[1]) % 2:
        raise RuntimeError("sin componente central")
    return cube * 2, {"estado": "ok"}


def siempre_falla(cube, meta):
    return cube, {"estado": "fallo", "motivo": "prueba"}


def usa_el_borde(cube, meta):
    """Resta la mediana del borde del campo COMPLETO: demuestra que la variante ve 800 px."""
    borde = np.concatenate([cube[:, :50].reshape(3, -1), cube[:, -50:].reshape(3, -1)], 1)
    return cube - np.median(borde, 1)[:, None, None], {"estado": "ok"}


def _cohorte(tmp_path, n=4, offset_borde=0.0):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        d = rng.normal(scale=1e-3, size=(3, 800, 800)).astype(np.float32) + offset_borde
        p = tmp_path / f"g{i}.fits"
        _write_cube(p, d)
        rows.append({"mangaid": f"1-{i}", "ruta_local": str(p), "redshift": 0.03})
    return pd.DataFrame(rows)


def _guard(tmp_path):
    return WriteGuard(tmp_path / "res", [tmp_path / "ro"])


def test_variant_is_applied_on_full_cube_before_crop(tmp_path):
    df = _cohorte(tmp_path, offset_borde=0.5)
    v = Variant("borde", usa_el_borde)
    out = cache_dir(tmp_path / "res" / "cache", v, FitsConfig())
    build_cache(df, "ruta_local", out, _guard(tmp_path), FitsConfig(), variant=v, chunk=2, workers=2, log=lambda *_: None)
    x, _, _, fb = load_cache(out, dtype="float32")
    assert not fb.any()
    assert abs(float(np.median(x))) < 1e-3          # el offset de 0.5 se restó usando el borde fuera del recorte


def test_variant_vs_identity_same_ids_and_shapes(tmp_path):
    df = _cohorte(tmp_path)
    g = _guard(tmp_path)
    cfg = FitsConfig()
    v = Variant("mas_uno", sumar_uno, params={"delta": 1.0})
    for var in (DESI, v):
        build_cache(df, "ruta_local", cache_dir(tmp_path / "res" / "cache", var, cfg), g, cfg, variant=var,
                    chunk=3, workers=2, log=lambda *_: None)
    a, ia, *_ = load_cache(cache_dir(tmp_path / "res" / "cache", DESI, cfg), dtype="float32")
    b, ib, *_ = load_cache(cache_dir(tmp_path / "res" / "cache", v, cfg), dtype="float32")
    assert list(ia) == list(ib) and a.shape == b.shape
    np.testing.assert_allclose(b - a, 1.0, atol=1e-3)


def test_fallback_keeps_raw_cube_and_flags_it(tmp_path):
    df = _cohorte(tmp_path)
    cfg = FitsConfig()
    v = Variant("impar", falla_si_impar, max_fallback_frac=0.6)
    out = cache_dir(tmp_path / "res" / "cache", v, cfg)
    build_cache(df, "ruta_local", out, _guard(tmp_path), cfg, variant=v, chunk=4, workers=2, log=lambda *_: None)
    x, ids, _, fb = load_cache(out, dtype="float32")
    assert fb.tolist() == [False, True, False, True]
    raw1 = read_full(Path(df.ruta_local[1]))[:, 250:550, 250:550]
    np.testing.assert_allclose(x[1], raw1, atol=1e-5)                 # respaldo = crudo
    raw0 = read_full(Path(df.ruta_local[0]))[:, 250:550, 250:550]
    np.testing.assert_allclose(x[0], 2 * raw0, atol=1e-5)             # éxito = transformado
    with np.load(out / "chunk_000.npz") as z:
        assert "sin componente central" in str(z["info"][1])


def test_too_many_fallbacks_stops(tmp_path):
    df = _cohorte(tmp_path)
    v = Variant("mala", siempre_falla, max_fallback_frac=0.05)
    with pytest.raises(RuntimeError, match="Fallaron"):
        build_cache(df, "ruta_local", cache_dir(tmp_path / "res" / "cache", v, FitsConfig()), _guard(tmp_path),
                    FitsConfig(), variant=v, chunk=4, workers=2, log=lambda *_: None)


def test_unimplemented_team_variant_stops_instead_of_falling_back(tmp_path):
    df = _cohorte(tmp_path, n=1)
    v = get("desi_prep")
    with pytest.raises(NotImplementedError):
        build_cache(df, "ruta_local", cache_dir(tmp_path / "res" / "cache", v, FitsConfig()), _guard(tmp_path),
                    FitsConfig(), variant=v, chunk=1, workers=1, log=lambda *_: None)


def test_variant_key_changes_with_params_and_version():
    a = Variant("p", sumar_uno, params={"delta": 1.0})
    assert a.key != Variant("p", sumar_uno, params={"delta": 2.0}).key
    assert a.key != Variant("p", sumar_uno, params={"delta": 1.0}, version="2").key
    assert DESI.key == "desi"


@pytest.mark.skipif(not SAMPLE_FITS.exists(), reason="sin cubo de muestra")
def test_float16_error_is_far_below_katachi_8bit_step():
    from src.katachi_fits.cache import read_center

    a = read_center(SAMPLE_FITS, FitsConfig()).astype(np.float64)
    sig = np.array([1.0e-3, 2.7e-3, 9.9e-3])[:, None, None]
    s = np.array([5.8, 5.8, 5.2])[:, None, None]
    f = lambda x: np.arcsinh(x / (3 * sig)) / s
    err = np.abs(f(a.astype(np.float16).astype(np.float64)) - f(a)).max()
    assert err < (1 / 255) / 10                     # al menos 10× más fino que los PNG de Katachi
