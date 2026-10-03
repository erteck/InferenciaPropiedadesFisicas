"""Pruebas de equivalencia del plan §3.5 y de los componentes (datos sintéticos).

Las pruebas que usan archivos reales (cubo de muestra, scalars.cat, checkpoints de
Katachi) se omiten si los archivos no están; ver las variables de entorno abajo.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torchvision import transforms as T

from src.katachi_fits import catalog, evaluate, robustness, shap_maps
from src.katachi_fits.cache import read_center, read_full
from src.katachi_fits.chain import Chain, GhostBatchNorm2d, add_plane, katachi_chain, resnet50
from src.katachi_fits.config import FitsConfig, TrainConfig
from src.katachi_fits.io_guard import WriteGuard, WriteGuardError
from src.katachi_fits.train import EarlyStopper, GPUData, Trainer
from src.katachi_fits.transforms import Normalizer, augment, preprocess, resample

SAMPLE_FITS = Path(os.environ.get("KATACHI_SAMPLE_FITS", "/Users/erteck/Downloads/Catalogo maestro/1-53_c82ba629bfe9_800px_grz.fits"))
EXT = Path(os.environ.get("KATACHI_EXT", "/tmp"))


# ---------------------------------------------------------------- FITS / caché
def _write_cube(path: Path, data: np.ndarray, bands="grz", crpix=400.5, scale=0.262):
    from astropy.io import fits

    h = fits.Header()
    for i, b in enumerate(bands):
        h[f"BAND{i}"] = b
    h["CRPIX1"] = crpix
    h["CRPIX2"] = crpix
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -scale / 3600, 0.0, 0.0, scale / 3600
    fits.PrimaryHDU(data.astype(">f4"), header=h).writeto(path)


def test_read_center_matches_astropy(tmp_path):
    rng = np.random.default_rng(0)
    d = rng.normal(size=(3, 800, 800)).astype(np.float32)
    p = tmp_path / "c.fits"
    _write_cube(p, d)
    np.testing.assert_array_equal(read_center(p, FitsConfig()), d[:, 250:550, 250:550])


def test_read_center_rejects_bad_band_order(tmp_path):
    p = tmp_path / "c.fits"
    _write_cube(p, np.zeros((3, 800, 800), np.float32), bands="zrg")
    with pytest.raises(ValueError, match="bandas"):
        read_center(p, FitsConfig())


def test_read_center_rejects_truncated(tmp_path):
    p = tmp_path / "c.fits"
    _write_cube(p, np.zeros((3, 800, 800), np.float32))
    b = p.read_bytes()
    p.write_bytes(b[: len(b) // 2])
    with pytest.raises(ValueError):
        read_center(p, FitsConfig())


@pytest.mark.skipif(not SAMPLE_FITS.exists(), reason="sin cubo de muestra")
def test_read_center_real_sample_bitwise():
    full = read_full(SAMPLE_FITS)
    np.testing.assert_array_equal(read_center(SAMPLE_FITS, FitsConfig()), full[:, 250:550, 250:550])
    # también para un recorte de otro tamaño (el código no depende de 300)
    c = FitsConfig(crop_px=192, out_px=192)
    np.testing.assert_array_equal(read_center(SAMPLE_FITS, c), full[:, 304:496, 304:496])


def test_crop_is_centered_on_crpix():
    cfg = FitsConfig()
    # centro del recorte en coordenadas 0-based del cubo = CRPIX − 1 = 399.5, exacto
    assert cfg.crop_lo + (cfg.crop_px - 1) / 2 == 399.5
    assert cfg.field_arcsec == pytest.approx(78.6)
    assert cfg.pixscale_out == pytest.approx(0.262) and not cfg.resamples
    with pytest.raises(ValueError):
        FitsConfig(crop_px=191)                  # paridad distinta → no centrable en CRPIX


# ---------------------------------------------------------------- transformaciones
def test_resample_flux_scales_by_area_ratio():
    x = torch.zeros(1, 3, 191, 191)
    yy, xx = torch.meshgrid(torch.arange(191.0), torch.arange(191.0), indexing="ij")
    x[:] = torch.exp(-((xx - 95) ** 2 + (yy - 95) ** 2) / (2 * 15 ** 2))
    y = resample(x, 256)
    ratio = (y.sum() / x.sum()).item()
    assert ratio == pytest.approx((256 / 191) ** 2, rel=0.005)
    # centro conservado: el máximo queda en el centro 127.5 ± 0.5
    c = y[0, 0]
    iy, ix = np.unravel_index(int(c.argmax()), c.shape)
    assert abs(iy - 127.5) <= 0.5 and abs(ix - 127.5) <= 0.5


def test_preprocess_keeps_zero_at_zero():
    n = Normalizer(np.array([1e-3, 2e-3, 1e-2]), np.array([5.0, 5.0, 5.0]), 3.0)
    y = preprocess(torch.zeros(2, 3, 300, 300), n, 300)
    assert torch.all(y == 0) and y.shape == (2, 3, 300, 300)


def test_no_resampling_when_sizes_match():
    x = torch.rand(2, 3, 300, 300)
    assert resample(x, 300) is x


def test_augment_matches_torchvision_compose():
    """Mismo resultado que el Compose pickleado de Katachi para iguales decisiones."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    x = torch.rand(6, 3, 64, 64)
    g = torch.Generator().manual_seed(3)
    out = augment(x, g, 360.0)
    g2 = torch.Generator().manual_seed(3)
    u = torch.rand(6, 3, generator=g2)
    for i in range(6):
        xi = x[i]
        if u[i, 0] < 0.5:
            xi = TF.hflip(xi)
        if u[i, 1] < 0.5:
            xi = TF.vflip(xi)
        ref = T.RandomRotation((float(u[i, 2] * 360), float(u[i, 2] * 360)),
                               interpolation=InterpolationMode.NEAREST, fill=0)(xi)
        torch.testing.assert_close(out[i], ref, rtol=0, atol=0)


# ---------------------------------------------------------------- cadena
def test_chain_shapes_and_random_conv1():
    c = Chain(imagenet=False, ghost_splits=2)
    assert c.mass.conv1.in_channels == 3 and c.sfr.conv1.in_channels == 4 and c.d4000.conv1.in_channels == 5
    p = c(torch.rand(4, 3, 64, 64))
    assert all(t.shape == (4, 1) for t in p)


def test_add_plane_is_detached_constant():
    x = torch.rand(2, 3, 8, 8)
    v = torch.tensor([[1.5], [2.5]], requires_grad=True)
    y = add_plane(x, v)
    assert y.shape == (2, 4, 8, 8) and not y.requires_grad
    assert torch.all(y[0, 3] == 1.5) and torch.all(y[1, 3] == 2.5)


def _split_reference(bn: nn.BatchNorm2d, x: torch.Tensor):
    """Lo que hace nn.DataParallel con 2 réplicas: trocea, normaliza por mitad,
    y solo la réplica 0 actualiza los buffers originales."""
    parts = torch.chunk(x, 2, 0)
    r0 = bn
    r1 = copy.deepcopy(bn)
    return torch.cat([r0(parts[0]), r1(parts[1])], 0)


@pytest.mark.parametrize("n", [32, 20, 3])
def test_ghost_bn_equals_two_replica_split(n):
    torch.manual_seed(0)
    ref = nn.BatchNorm2d(8)
    ref.weight.data.uniform_(0.5, 1.5)
    ref.bias.data.uniform_(-0.5, 0.5)
    g = GhostBatchNorm2d.from_bn(copy.deepcopy(ref), 2)
    ref.train(); g.train()
    x = torch.randn(n, 8, 5, 5)
    torch.testing.assert_close(g(x), _split_reference(ref, x))
    torch.testing.assert_close(g.running_mean, ref.running_mean)
    torch.testing.assert_close(g.running_var, ref.running_var)
    assert int(g.num_batches_tracked) == int(ref.num_batches_tracked)


def test_single_backward_equals_three_backwards():
    torch.manual_seed(0)
    c1 = Chain(imagenet=False, ghost_splits=2)
    c2 = copy.deepcopy(c1)
    x, y = torch.rand(4, 3, 64, 64), torch.rand(4, 3)
    mse = nn.MSELoss()
    p = c1(x)
    torch.stack([mse(p[k], y[:, k:k + 1]) for k in range(3)]).sum().backward()
    p = c2(x)
    for k in range(3):
        mse(p[k], y[:, k:k + 1]).backward(retain_graph=True)
    for (n1, a), (_, b) in zip(c1.named_parameters(), c2.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7, msg=n1)


def test_early_stopper_is_katachi():
    s = EarlyStopper(1e-3, 3)
    seq = [1.0, 0.9995, 0.9992, 0.9991]   # ninguna mejora > 0.1 %
    assert [s.step(v) for v in seq] == [True, True, True, False]


# ---------------------------------------------------------------- entrenamiento y reanudación
def _tiny_trainer(tmp_path, seed=0):
    cfg = TrainConfig(seed=seed, batch_size=4, max_epochs=2, ghost_bn_splits=2,
                      fused_adam=False, channels_last=False, cudnn_benchmark=False)
    norm = Normalizer(np.ones(3) * 0.01, np.ones(3) * 5, 3.0)
    guard = WriteGuard(tmp_path, read_only=[tmp_path / "ro"])
    t = Trainer(cfg, norm, 32, tmp_path / "run", guard, torch.device("cpu"), log=lambda *_: None)
    return t


def test_training_resume_is_exact(tmp_path, monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.models, "resnet50",
                        lambda weights=None: torchvision.models.resnet.resnet50(weights=None))
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.01, size=(10, 3, 24, 24)).astype(np.float32)
    y = rng.normal(size=(10, 3)).astype(np.float32)
    tr, va = GPUData(x[:8], y[:8], torch.device("cpu")), GPUData(x[8:], y[8:], torch.device("cpu"))

    a = _tiny_trainer(tmp_path / "a")
    ha = a.fit(tr, va)

    b = _tiny_trainer(tmp_path / "b")
    b.cfg.max_epochs = 1
    b.fit(tr, va)
    b2 = _tiny_trainer(tmp_path / "b")   # nueva sesión: reanuda desde last.pt
    b2.cfg.max_epochs = 2
    hb = b2.fit(tr, va)
    for (n, p), (_, q) in zip(a.model.named_parameters(), b2.model.named_parameters()):
        torch.testing.assert_close(p, q, msg=n)
    np.testing.assert_allclose(ha["train_loss_d4000"], hb["train_loss_d4000"], rtol=1e-6)


# ---------------------------------------------------------------- escritura segura
def test_write_guard(tmp_path):
    ro = tmp_path / "catalogo"
    ro.mkdir()
    g = WriteGuard(tmp_path / "res", [ro])
    g.write_text(tmp_path / "res" / "a.txt", "x")
    with pytest.raises(WriteGuardError):
        g.write_text(tmp_path / "res" / "a.txt", "y")            # no sobrescribe
    with pytest.raises(WriteGuardError):
        g.write_text(ro / "b.txt", "x")                           # solo lectura
    with pytest.raises(WriteGuardError):
        g.write_text(tmp_path / "otro.txt", "x")                  # fuera de resultados
    with pytest.raises(WriteGuardError):
        WriteGuard(ro / "res", [ro])                              # resultados dentro de solo lectura
    g.write_text(tmp_path / "res" / "a.txt", "z", allow_replace=True)
    assert (tmp_path / "res" / "a.txt").read_text() == "z"


# ---------------------------------------------------------------- métricas
def test_metrics_match_sklearn():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rng = np.random.default_rng(1)
    y = rng.normal(10, 1, 500)
    p = y + rng.normal(0, 0.2, 500)
    m = evaluate.metrics(y, p)
    assert m["MSE"] == pytest.approx(mean_squared_error(y, p))
    assert m["MAD"] == pytest.approx(mean_absolute_error(y, p))
    assert m["R2"] == pytest.approx(r2_score(y, p))


def test_paired_comparison_verdicts():
    rng = np.random.default_rng(2)
    y = rng.normal(10, 1, 1000)
    ref = y + rng.normal(0, 0.25, 1000)
    better = y + rng.normal(0, 0.15, 1000)
    same = ref + rng.normal(0, 0.001, 1000)
    assert evaluate.paired_comparison(y, better, ref, "log_mstar", n=2000)["veredicto"] == "mejor"
    assert evaluate.paired_comparison(y, ref, better, "log_mstar", n=2000)["veredicto"] == "peor"
    assert evaluate.paired_comparison(y, same, ref, "log_mstar", n=2000)["veredicto"] == "equivalente"


def test_holm():
    assert evaluate.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


# ---------------------------------------------------------------- SHAP / robustez
def test_azimuthal_average_constant_and_length():
    prof = shap_maps.azimuthal_average(np.ones((256, 256)))
    assert len(prof) == 179 and np.allclose(prof, 1)        # mismo largo que Katachi (179)
    assert len(shap_maps.azimuthal_average(np.ones((300, 300)))) == 210


def test_border_std():
    x = torch.zeros(2, 3, 20, 20)
    x[0, :, :5, :] = 1.0
    s = robustness.border_std(x)
    assert s[0] > 0 and s[1] == 0


# ---------------------------------------------------------------- datos reales de Katachi
@pytest.mark.skipif(not (EXT / "scalars.cat").exists(), reason="sin scalars.cat")
def test_t50_reproduces_published():
    from src.katachi_fits.sfh import load_t50, predict_t50

    s = catalog.load_scalars(EXT / "scalars.cat")
    s = s[np.isfinite(s.t50_model) & np.isfinite(s.d4000) & np.isfinite(s.log_sfr)]
    t = predict_t50(load_t50(EXT / "t50.pt"), s.log_mstar, s.log_sfr, s.d4000)
    np.testing.assert_allclose(t, s.t50_model, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not (EXT / "images.cat").exists(), reason="sin images.cat")
def test_b0_predictions_reproduce_from_unaugmented_images():
    from src.katachi_fits.baseline import images_tensor, load_images_cat, predict_chain

    s = catalog.load_scalars(EXT / "scalars.cat")
    t = s[s.split == "Test"].head(32)
    x = images_tensor(load_images_cat(EXT / "images.cat"), t.mangaid)
    p = predict_chain(katachi_chain(EXT / "mass.pt", EXT / "sfr.pt", EXT / "d4000.pt"), x)
    np.testing.assert_allclose(p, t[["pred_mstar", "pred_sfr", "pred_d4000"]].to_numpy(), atol=2e-2)


def test_cache_roundtrip_and_resume(tmp_path):
    from src.katachi_fits.cache import build_cache, cache_dir, load_cache

    rng = np.random.default_rng(0)
    rows = []
    for i in range(5):
        d = rng.normal(scale=1e-3, size=(3, 800, 800)).astype(np.float32)
        d[0, 400, 400] = 0.0
        p = tmp_path / f"g{i}.fits"
        _write_cube(p, d)
        rows.append({"mangaid": f"1-{i}", "ruta_local": str(p)})
    df = pd.DataFrame(rows)
    g = WriteGuard(tmp_path / "res", [tmp_path / "ro"])
    out = tmp_path / "res" / "cache" / "desi"
    build_cache(df, "ruta_local", out, g, FitsConfig(), chunk=2, workers=2, log=lambda *_: None)
    build_cache(df, "ruta_local", out, g, FitsConfig(), chunk=2, workers=2, log=lambda *_: None)  # reanuda sin reescribir
    x, ids, f0, fb = load_cache(out)
    assert x.shape == (5, 3, 300, 300) and x.dtype == np.float16 and list(ids) == list(df.mangaid)
    ref = read_full(Path(rows[3]["ruta_local"]))[:, 250:550, 250:550]
    np.testing.assert_array_equal(x[3], ref.astype(np.float16))
    assert f0[0, 0] == pytest.approx(1 / 300 ** 2) and not fb.any()
    with pytest.raises(ValueError, match="otra configuración"):   # otra config en la misma carpeta
        build_cache(df, "ruta_local", out, g, FitsConfig(crop_px=200, out_px=200), chunk=2, workers=2, log=lambda *_: None)


def test_sky_sigma_recovers_noise_after_resampling(tmp_path):
    from src.katachi_fits.cache import sky_sigma

    rng = np.random.default_rng(1)
    paths = []
    for i in range(3):
        p = tmp_path / f"s{i}.fits"
        _write_cube(p, rng.normal(scale=[[[1e-3]], [[2e-3]], [[4e-3]]], size=(3, 800, 800)).astype(np.float32))
        paths.append(p)
    true = np.array([1e-3, 2e-3, 4e-3])
    s = sky_sigma(paths, FitsConfig(), threads=2)              # sin remuestreo: σ verdadero
    np.testing.assert_allclose(s, true, rtol=0.02)
    s2 = sky_sigma(paths, FitsConfig(crop_px=192, out_px=256), threads=2)   # con remuestreo: σ menor
    assert np.all(s2 < true) and np.all(s2 > 0.4 * true)
    assert s2[1] / s2[0] == pytest.approx(2, rel=0.05) and s2[2] / s2[0] == pytest.approx(4, rel=0.05)


def test_shap_maps_shape_and_profile():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    bg, x = torch.rand(8, 3, 32, 32), torch.rand(3, 3, 32, 32)
    s = shap_maps.shap_maps(m, bg, x, batch=2)
    assert s.shape == (3, 32, 32, 3)
    prof = shap_maps.radial_profile(s[0])
    assert prof.ndim == 1 and np.isfinite(prof).all()
    assert shap_maps.kpc_per_px(0.03, 0.195) == pytest.approx(0.1215, rel=0.01)


def test_binned_by_target_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(10, 1, 200)})
    df["p"] = df.y + rng.normal(0, 0.1, 200)
    r = evaluate.binned(df, "y", "p", "y", [6, 9, 10, 11, 14])
    assert len(r) == 4 and r.N.sum() == 200


def test_read_center_rejects_wrong_pixel_scale(tmp_path):
    p = tmp_path / "c.fits"
    _write_cube(p, np.zeros((3, 800, 800), np.float32), scale=0.5)
    with pytest.raises(ValueError, match="Escala"):
        read_center(p, FitsConfig())


def test_read_center_rejects_non_finite(tmp_path):
    d = np.zeros((3, 800, 800), np.float32)
    d[1, 400, 400] = np.nan
    p = tmp_path / "c.fits"
    _write_cube(p, d)
    with pytest.raises(ValueError, match="no finitos"):
        read_center(p, FitsConfig())


def test_cache_reports_bad_files_and_stops(tmp_path):
    from src.katachi_fits.cache import build_cache

    good, bad = tmp_path / "1-0_x.fits", tmp_path / "1-1_x.fits"
    _write_cube(good, np.zeros((3, 800, 800), np.float32))
    _write_cube(bad, np.zeros((3, 800, 800), np.float32), bands="zrg")
    df = pd.DataFrame({"mangaid": ["1-0", "1-1"], "ruta_local": [str(good), str(bad)]})
    g = WriteGuard(tmp_path / "res", [tmp_path / "ro"])
    with pytest.raises(RuntimeError, match="controles de lectura"):
        build_cache(df, "ruta_local", tmp_path / "res" / "c", g, FitsConfig(), chunk=2, workers=2, log=lambda *_: None)
    rep = pd.read_csv(tmp_path / "res" / "c" / "errores_lectura_bloque_000.csv")
    assert rep.mangaid.tolist() == ["1-1"] and "bandas" in rep.error[0]


def test_plan_rejects_path_of_another_galaxy(tmp_path):
    p = tmp_path / "plan.csv"
    pd.DataFrame({"mangaid": ["1-10", "1-11"], "ruta": ["/x/1-10_ab_800px_grz.fits", "/x/1-10_cd_800px_grz.fits"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="no corresponden"):
        catalog.load_plan(p)


def test_apply_qc_keeps_ok_and_records_reasons():
    df = pd.DataFrame({"mangaid": ["a", "b", "c"], "split": ["Train", "Train", "Test"],
                       "particion": ["Train", "Validation", "Test"], "QCFLAG": [0, 6, 1],
                       "grupo_qc": ["OK", "WARNING", "BAD"]})
    inc, exc = catalog.apply_qc(df, ("OK",))
    assert inc.mangaid.tolist() == ["a"]
    assert exc.mangaid.tolist() == ["b", "c"] and exc.particion.tolist() == ["Validation", "Test"]
    assert "Estrella en primer plano" in exc.motivo[0] and "Redshift incorrecto" in exc.motivo[1]


@pytest.mark.skipif(not (EXT / "pipe3d.fits").exists(), reason="sin VAC Pipe3D")
def test_team_counts_with_qc0():
    from src.katachi_fits import config as K

    plan = catalog.load_plan(Path(os.environ.get("KATACHI_PLAN_CSV",
        "/Users/erteck/Downloads/Catalogo maestro/drive-download-20261003T061903Z-1-001/plan.csv")))
    s = catalog.load_scalars(EXT / "scalars.cat")
    coh, _ = catalog.build_cohort(plan, s, catalog.load_pipe3d_qc(EXT / "pipe3d.fits", s))
    coh = catalog.add_validation(coh, K.VAL_FRACTION, K.VAL_SEED)
    inc, exc = catalog.apply_qc(coh, K.QC_GROUPS_INCLUDED)
    assert len(inc) == 9303 and (inc.split == "Test").sum() == 918 and len(exc) == 358


TEAM_CSV = Path(os.environ.get("KATACHI_TEAM_CATALOG",
                "/Users/erteck/Downloads/Catalogo maestro/catalogo_candidato_9661_galaxias.csv"))


@pytest.mark.skipif(not (TEAM_CSV.exists() and (EXT / "pipe3d.fits").exists()), reason="sin catálogo del equipo")
def test_team_catalog_matches_katachi_and_pipe3d_and_team_counts():
    from src.katachi_fits import config as K
    from src.katachi_fits.io_guard import sha256

    assert sha256(TEAM_CSV) == K.TEAM_CATALOG_SHA256
    team = catalog.load_team_catalog(TEAM_CSV)
    s = catalog.load_scalars(EXT / "scalars.cat")
    assert all(v == 0 for v in catalog.cross_check(team, s, catalog.load_pipe3d_qc(EXT / "pipe3d.fits", s)).values())
    plan = catalog.load_plan(Path("/Users/erteck/Downloads/Catalogo maestro/drive-download-20261003T061903Z-1-001/plan.csv"))
    coh, _ = catalog.build_cohort_team(plan, team, s)
    inc, exc = catalog.apply_qc(coh, K.QC_GROUPS_INCLUDED)
    assert inc.particion.value_counts().to_dict() == {"Train": 7123, "Validation": 1262, "Test": 918}
    assert len(exc) == 358 and int(inc.varias_obs_pipe3d.sum()) == 114
