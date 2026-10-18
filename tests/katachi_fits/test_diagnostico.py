"""Pruebas de diagnostico.py: métricas, radio de mitad de luz, oclusiones y permutación de planos."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torchvision

from src.katachi_fits import diagnostico as DG
from src.katachi_fits.config import MejorasConfig, TrainConfig
from src.katachi_fits.io_guard import WriteGuard
from src.katachi_fits.train import GPUData, Trainer
from src.katachi_fits.transforms import Normalizer

NORM = Normalizer(np.array([0.002, 0.004, 0.01]), np.array([5.7, 6.0, 5.5]), 3.0)


@pytest.fixture(autouse=True)
def _sin_imagenet(monkeypatch):
    monkeypatch.setattr(torchvision.models, "resnet50",
                        lambda weights=None: torchvision.models.resnet.resnet50(weights=None))


def _trainer(tmp_path, mejoras=None):
    cfg = TrainConfig(seed=0, batch_size=4, max_epochs=1, fused_adam=False, channels_last=False,
                      cudnn_benchmark=False, freeze_pretrained=False)
    g = WriteGuard(tmp_path, read_only=[tmp_path / "ro"])
    return Trainer(cfg, NORM, 32, tmp_path / "run", g, torch.device("cpu"), log=lambda *_: None,
                   mejoras=mejoras or MejorasConfig(programa_rapido=True, epocas_rapido=1))


def _datos(n=10, px=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.01, size=(n, 3, px, px)).astype(np.float32)
    x[:, :, px // 2 - 4:px // 2 + 4, px // 2 - 4:px // 2 + 4] += rng.uniform(0.5, 2, (n, 3, 1, 1))
    y = np.stack([10 + rng.normal(0, .5, n), rng.normal(0, .5, n), 1.5 + rng.normal(0, .1, n)], 1)
    return GPUData(x, y.astype(np.float32), torch.device("cpu"))


def test_metricas_casos_conocidos():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = DG.metricas(y, y + 0.1)
    assert m["RMSE"] == pytest.approx(0.1) and m["sesgo"] == pytest.approx(0.1)
    assert m["sigma_NMAD"] == pytest.approx(0.0) and m["catastroficos"] == 0
    m2 = DG.metricas(y, y + np.array([0, 0, 0, 1.0]))
    assert m2["catastroficos"] == pytest.approx(0.25)
    assert "catastroficos" not in DG.metricas(y, y, umbral_catastrofico=None)


def test_delta_rmse_signo_e_intervalo():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    bueno, malo = y + rng.normal(0, 0.1, 500), y + rng.normal(0, 0.5, 500)
    d, lo, hi = DG.delta_rmse(y, bueno, malo)
    assert d < 0 and hi < 0 and lo < d < hi


def test_radio_mitad_luz_de_una_gaussiana():
    px, s = 101, 6.0
    yy, xx = np.indices((px, px))
    g = np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / (2 * s ** 2))
    x = np.stack([g, g, g])[None].astype(np.float32)
    r50 = DG.radio_mitad_luz(x)[0]
    assert r50 == pytest.approx(s * np.sqrt(2 * np.log(2)), abs=0.6)       # 1.177 σ


def test_mascara_anillo_cubre_lo_pedido():
    m = DG.mascara_anillo(torch.tensor([5.0, 10.0]), 41, 0, 1)
    assert m.shape == (2, 1, 41, 41)
    assert m[0].sum() < m[1].sum()                           # galaxia más grande, núcleo más grande
    assert bool(m[0, 0, 20, 20]) and not bool(m[0, 0, 0, 0])


def test_sin_modificar_coincide_con_predict(tmp_path):
    t = _trainer(tmp_path)
    d = _datos()
    np.testing.assert_allclose(DG.predecir_con(t, d), t.predict(d), rtol=1e-5, atol=1e-5)


def test_ocluir_banda_y_anillo_cambian_la_prediccion(tmp_path):
    t = _trainer(tmp_path)
    d = _datos()
    base = t.predict(d)
    assert not np.allclose(DG.predecir_con(t, d, ocluir_banda=0), base)
    r50 = np.full(len(d), 4.0, np.float32)
    assert not np.allclose(DG.predecir_con(t, d, anillo=(0, 1), r50_px=r50), base)
    # Un anillo vacío (fuera de la imagen) no cambia nada
    np.testing.assert_allclose(DG.predecir_con(t, d, anillo=(100, 200), r50_px=r50), base, rtol=1e-5, atol=1e-5)


def test_permutaciones_solo_afectan_a_las_redes_siguientes(tmp_path):
    t = _trainer(tmp_path)
    d = _datos()
    base = t.predict(d)
    a = DG.predecir_con(t, d, permutar="masa→SFR", seed=1)
    np.testing.assert_allclose(a[:, 0], base[:, 0], rtol=1e-5, atol=1e-5)    # la masa no cambia
    b = DG.predecir_con(t, d, permutar="masa→D4000", seed=1)
    np.testing.assert_allclose(b[:, :2], base[:, :2], rtol=1e-5, atol=1e-5)  # masa y SFR no cambian
    c = DG.predecir_con(t, d, permutar="SFR→D4000", seed=1)
    np.testing.assert_allclose(c[:, :2], base[:, :2], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        DG.predecir_con(t, d, permutar="otra")


def test_mediana_movil_y_vista_rgb():
    x = np.linspace(0, 1, 200)
    r = np.where(x > 0.5, 1.0, 0.0)
    mm = DG.mediana_movil(x, r, [0, 0.25, 0.5, 0.75, 1.01])
    assert list(mm["mediana"]) == [0, 0, 1, 1]
    img = DG.vista_rgb(np.random.default_rng(0).normal(size=(3, 8, 8)))
    assert img.shape == (8, 8, 3) and img.min() >= 0 and img.max() <= 1
