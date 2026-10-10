"""Pruebas de las mejoras desactivables (aumentacion.py, opciones de Trainer y ejecutor).

Lo más importante: con todas las opciones apagadas, el entrenamiento es idéntico al
original; y cada opción hace exactamente lo que dice."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision

from src.katachi_fits import aumentacion as AU
from src.katachi_fits.chain import Chain, ChainTarget
from src.katachi_fits.config import MejorasConfig, TrainConfig
from src.katachi_fits.experimentos import Experimento, comparar_con_base
from src.katachi_fits.io_guard import WriteGuard
from src.katachi_fits.train import GPUData, Trainer
from src.katachi_fits.transforms import Normalizer, network_input

NORM = Normalizer(np.array([0.002, 0.004, 0.01]), np.array([5.7, 6.0, 5.5]), 3.0)


@pytest.fixture(autouse=True)
def _sin_imagenet(monkeypatch):
    """Pesos aleatorios: las pruebas no dependen de descargar ImageNet."""
    monkeypatch.setattr(torchvision.models, "resnet50",
                        lambda weights=None: torchvision.models.resnet.resnet50(weights=None))


def _datos(n=8, px=32, seed=0, sigma=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.01, size=(n, 3, px, px)).astype(np.float32)
    x[:, :, px // 2 - 3:px // 2 + 3, px // 2 - 3:px // 2 + 3] += 1.0       # una "galaxia" en el centro
    y = np.stack([10 + rng.normal(0, .5, n), rng.normal(0, .5, n), 1.5 + rng.normal(0, .1, n)], 1).astype(np.float32)
    s = np.full((n, 3), 0.01, np.float32) if sigma else None
    return GPUData(x, y, torch.device("cpu"), sigma=s)


def _trainer(tmp_path, mejoras=None, out_px=32, **kw):
    cfg = TrainConfig(seed=0, batch_size=4, max_epochs=2, fused_adam=False, channels_last=False,
                      cudnn_benchmark=False, freeze_pretrained=False, **kw)
    g = WriteGuard(tmp_path, read_only=[tmp_path / "ro"])
    est = (np.zeros(3), np.ones(3))
    planos = ([10.0, 0.0], [0.5, 0.5])
    return Trainer(cfg, NORM, out_px, tmp_path / "run", g, torch.device("cpu"), log=lambda *_: None,
                   mejoras=mejoras, estadisticas_entrada=est, planos=planos)


# ------------------------------------------------------------- todo apagado = original
def test_todo_apagado_es_identico_al_entrenamiento_original(tmp_path):
    a = _trainer(tmp_path / "a")                                   # sin mejoras
    b = _trainer(tmp_path / "b", mejoras=MejorasConfig())          # mejoras por defecto
    assert b.legado and b.mejoras.sufijo() == ""
    ha, hb = a.fit(_datos(), None), b.fit(_datos(), None)
    np.testing.assert_allclose(ha.train_loss_d4000, hb.train_loss_d4000, rtol=0, atol=0)
    for (n, p), (_, q) in zip(a.model.named_parameters(), b.model.named_parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0, msg=n)


def test_sin_aumentacion_el_camino_nuevo_coincide_con_network_input(tmp_path):
    t = _trainer(tmp_path, MejorasConfig(rotacion_bilineal=True))
    assert not t.legado
    x = _datos().x
    torch.testing.assert_close(t._inputs(x, train=False), network_input(x, NORM, 32, ("z", "r", "g")))


# ------------------------------------------------------------- geometría
def test_geometria_identidad_cuando_no_hay_giro_ni_reflejo():
    x = torch.rand(2, 3, 40, 40)
    g = torch.Generator().manual_seed(0)
    # Forzamos u = 0.9 (sin reflejo) y ángulo 0 con grados = 0
    y = AU.geometria(x, g, 40, 0.0, bilineal=True, desplazamiento_px=0)
    ref = x.clone()
    # La rejilla puede reflejar según el azar: comparamos con las 4 combinaciones posibles
    opciones = [ref, ref.flip(-1), ref.flip(-2), ref.flip(-1).flip(-2)]
    assert any(torch.allclose(y[i], o[i], atol=1e-5) for i in range(2) for o in opciones)


def test_con_margen_no_hay_esquinas_vacias():
    x = torch.rand(16, 3, 440, 440) + 0.5                            # imagen sin ceros
    y = AU.geometria(x, torch.Generator().manual_seed(1), 300, 360.0, bilineal=True, desplazamiento_px=5)
    assert y.shape == (16, 3, 300, 300)
    assert (y > 0).all(), "aparecieron píxeles de relleno: el margen no alcanza"


def test_sin_margen_si_hay_esquinas_vacias_a_45_grados():
    x = torch.ones(1, 3, 300, 300)
    g = torch.Generator().manual_seed(0)
    # Rotación fija de 45°: construimos theta a mano con la misma convención
    c = s = math.cos(math.pi / 4)
    theta = torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]])
    grid = torch.nn.functional.affine_grid(theta, (1, 3, 300, 300), align_corners=False)
    y = torch.nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    vacias = (y[0, 0] == 0).float().mean().item()
    assert 0.12 < vacias < 0.20                                      # ≈ 17 %, el defecto que corrige el margen


def test_d4_son_las_8_simetrias_distintas():
    x = torch.arange(16.).view(1, 1, 4, 4)
    vistas = {tuple(AU.d4(x, i).flatten().tolist()) for i in range(8)}
    assert len(vistas) == 8 and tuple(x.flatten().tolist()) in vistas


# ------------------------------------------------------------- aumentación física
def test_ruido_tiene_el_nivel_pedido():
    x = torch.zeros(400, 3, 32, 32)
    sig = torch.tensor([0.002, 0.004, 0.01])
    y = AU.ruido_y_psf(x, sig, torch.Generator().manual_seed(0), ruido_max=1.0, psf_max_px=0, prob=1.0)
    # k ~ U(0, 1)  →  E[k²] = 1/3  →  desviación del ruido = σ/√3
    np.testing.assert_allclose(y.std(dim=(0, 2, 3)).numpy(), (sig / math.sqrt(3)).numpy(), rtol=0.05)


def test_psf_conserva_el_flujo():
    x = torch.zeros(4, 3, 64, 64)
    x[:, :, 32, 32] = 1.0
    y = AU.ruido_y_psf(x, torch.ones(3), torch.Generator().manual_seed(0), 0.0, psf_max_px=2.0, prob=1.0)
    np.testing.assert_allclose(y.sum(dim=(2, 3)).numpy(), 1.0, rtol=1e-4)
    assert (y[:, :, 32, 32] < 1).all()                              # se desenfocó


def test_sin_entrenar_no_hay_aumentacion(tmp_path):
    t = _trainer(tmp_path, MejorasConfig(ruido_max=1.0, psf_max_px=1.0, desplazamiento_px=3))
    x = _datos().x
    torch.testing.assert_close(t._inputs(x, train=False), t._inputs(x, train=False))


def test_sigma_por_imagen_recupera_el_ruido():
    rng = np.random.default_rng(0)
    sig = np.array([0.002, 0.004, 0.01])
    x = torch.from_numpy((rng.normal(size=(5, 3, 100, 100)) * sig[None, :, None, None]).astype(np.float32))
    np.testing.assert_allclose(AU.sigma_por_imagen(x).numpy(), np.tile(sig, (5, 1)), rtol=0.08)


# ------------------------------------------------------------- estandarizaciones
def test_estandarizar_imagen_usa_las_estadisticas_dadas(tmp_path):
    t = _trainer(tmp_path, MejorasConfig(estandarizar_imagen=True))
    x = _datos().x
    base = network_input(x, NORM, 32, ("g", "r", "z"))
    out = t._inputs(x, train=False)
    torch.testing.assert_close(out, base[:, [2, 1, 0]])             # media 0 y desviación 1: sin cambio, solo orden


def test_planos_estandarizados_en_la_cadena_y_en_shap():
    c = Chain(imagenet=False, ghost_splits=1, planos=([10.0, -1.0], [0.5, 1.0])).eval()
    x = torch.rand(2, 3, 32, 32)
    p1, p2, p3 = c(x)
    torch.testing.assert_close(ChainTarget(c, 2)(x), p3)
    assert "plano_mu" in c.state_dict()
    torch.testing.assert_close(c.plano(torch.tensor([[10.5]]), 0), torch.tensor([[1.0]]))
    assert "plano_mu" not in Chain(imagenet=False, ghost_splits=1).state_dict()   # pesos antiguos siguen cargando


# ------------------------------------------------------------- entrenamiento con opciones
@pytest.mark.parametrize("mejoras,px", [
    (MejorasConfig(margen_rotacion=True, rotacion_bilineal=True, desplazamiento_px=2), 48),
    (MejorasConfig(ruido_max=1.0, psf_max_px=1.0, sigma_por_imagen=True), 32),
    (MejorasConfig(estandarizar_imagen=True, estandarizar_planos=True), 32),
])
def test_cada_opcion_entrena_sin_errores(tmp_path, mejoras, px):
    t = _trainer(tmp_path, mejoras, out_px=32)
    h = t.fit(_datos(px=px, sigma=True), None)
    assert len(h) == 2 and np.isfinite(h.train_loss_mstar).all()
    p = t.predict(_datos(px=px, sigma=True), tta=True)
    assert p.shape == (8, 3) and np.isfinite(p).all()


def test_programa_rapido_coseno_y_numero_fijo_de_epocas(tmp_path):
    t = _trainer(tmp_path, MejorasConfig(programa_rapido=True, epocas_rapido=4, calentamiento_epocas=1))
    t.cfg.max_epochs = 100
    h = t.fit(_datos(), None)
    assert len(h) == 4                                                # termina en epocas_rapido, sin parada temprana
    lr = h.lr_mstar.to_numpy()
    assert lr[0] == pytest.approx(1e-3) and lr[-1] < lr[1]            # sube con el calentamiento y luego baja


def test_tta_promedia_las_8_vistas(tmp_path):
    t = _trainer(tmp_path)
    d = _datos()
    p1, p8 = t.predict(d), t.predict(d, tta=True)
    assert p1.shape == p8.shape and not np.allclose(p1, p8)


# ------------------------------------------------------------- ejecutor
def test_identificador_refleja_las_opciones():
    assert Experimento("base").identificador(300) == "X_base_recorte300_completo_seed42"
    e = Experimento("ruido", MejorasConfig(ruido_max=1.0, programa_rapido=True))
    assert e.identificador(300) == "X_ruido_recorte300_completo_ruido1_rapido30_seed42"


def test_comparacion_contra_la_base(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 3))
    ids = [f"1-{i}" for i in range(200)]
    for nombre, ruido in (("a", 0.1), ("b", 0.3)):
        d = tmp_path / nombre; d.mkdir()
        df = pd.DataFrame({"mangaid": ids})
        for k, t in enumerate(("log_mstar", "log_sfr", "d4000")):
            df[f"pred_{t}"] = y[:, k] + rng.normal(0, ruido, 200)
        df.to_csv(d / "predicciones_validacion.csv", index=False)
    c = comparar_con_base((tmp_path / "a", tmp_path / "b"), y)
    assert c["veredicto_M*"] == "mejor" and c["dRMSE_M*"] < 0


def test_estadisticas_de_entrada_con_sigma_por_imagen():
    from src.katachi_fits.experimentos import estadisticas_entrada
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(50, 3, 40, 40)) * np.array([0.002, 0.004, 0.01])[None, :, None, None] * 3).astype(np.float32)
    sig = np.tile(np.array([0.006, 0.012, 0.03], np.float32), (50, 1))       # σ real de estas imágenes
    mu_g, sd_g = estadisticas_entrada(x, np.arange(50), NORM, 40)
    mu_s, sd_s = estadisticas_entrada(x, np.arange(50), NORM, 40, sigma_img=sig)
    assert not np.allclose(sd_g, sd_s)          # con otro σ, la escala de la entrada cambia
    assert np.all(np.abs(mu_s) < 0.01)          # el ruido puro queda centrado en 0


def test_desplazamiento_mayor_que_el_margen_se_rechaza(tmp_path):
    with pytest.raises(ValueError, match="desplazamiento_px"):
        _trainer(tmp_path, MejorasConfig(margen_rotacion=True, desplazamiento_px=6))


def test_tta_no_cambia_el_identificador():
    # TTA ya no es una opción: el ejecutor reporta siempre ambas versiones
    assert "tta" not in MejorasConfig(programa_rapido=True).sufijo()
