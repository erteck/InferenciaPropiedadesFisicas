"""Pruebas de las funciones adaptadas de los cuadernos del equipo (imagen_equipo.py)."""
from __future__ import annotations

import numpy as np
from astropy.visualization import PercentileInterval

from src.caracteristicas_desi import imagen_equipo as IE
from src.caracteristicas_desi.fotometria import segmentar

from .test_fe_desi import cubo, sersic

RUIDO = 0.002


def _galaxia_con_grumos(n_grumos: int, seed: int = 0) -> np.ndarray:
    img = sersic(1.0, 25, 400.0)
    rng = np.random.default_rng(seed)
    for _ in range(n_grumos):
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(25, 50)
        c = (149.5 + rad * np.cos(ang), 149.5 + rad * np.sin(ang))
        img = img + sersic(1.0, 1.5, 6.0, centro=c)
    return cubo(img, ruido=RUIDO, seed=seed)


def test_grumosidad_distingue_grumos_de_un_disco_liso():
    liso, grumoso = _galaxia_con_grumos(0), _galaxia_con_grumos(25)
    g_liso = IE.medir_equipo(liso.astype(np.float16))["grumosidad_g"]
    g_grum = IE.medir_equipo(grumoso.astype(np.float16))["grumosidad_g"]
    assert abs(g_liso) < 0.03                       # sin grumos, solo queda el ruido ya restado
    assert g_grum > g_liso + 0.03


def test_filtro_gaussiano_sesga_un_perfil_liso():
    # Motivo del cambio a la mediana: en un perfil que decae, el gaussiano deja la medida negativa.
    x = sersic(1.0, 25, 400.0) + np.random.default_rng(1).normal(0, RUIDO, (300, 300)).astype(np.float32)
    gal, vec, _ = segmentar(x)
    assert IE.grumosidad(x, gal, vec, filtro="gaussiano") < IE.grumosidad(x, gal, vec, filtro="mediana") - 0.02


def test_auditoria_detecta_banda_en_cero_y_no_finitos():
    c = _galaxia_con_grumos(0)
    assert not IE.medir_equipo(c)["banda_en_cero"]
    c0 = c.copy(); c0[2] = 0
    assert IE.medir_equipo(c0)["banda_en_cero"]
    cn = c.copy(); cn[0, 5, 5] = np.nan
    assert not IE.medir_equipo(cn)["finito"]
    assert IE.medir_equipo(c)["huella"] == IE.medir_equipo(c.copy())["huella"]


def test_normalizar_queda_en_cero_uno():
    img = cubo(sersic(4.0, 10, 300.0), ruido=RUIDO)[1]
    for est in ("lineal", "log", "asinh"):
        x = IE.normalizar(img, est, PercentileInterval(98))
        assert x.min() >= 0 and x.max() <= 1


def test_inverso_varianza_pesa_mas_la_banda_menos_ruidosa():
    rng = np.random.default_rng(0)
    c = np.stack([rng.normal(0, s, (300, 300)) for s in (0.001, 0.01, 0.01)]).astype(np.float32)
    c[0, 140:160, 140:160] += 1.0                    # solo la banda con menos ruido tiene señal
    col = IE.colapsos(c)
    centro = (slice(140, 160), slice(140, 160))
    assert col["inverso de la varianza"][centro].mean() > 0.9
    assert col["promedio sin cielo"][centro].mean() < 0.4


def test_mascara_gaussiana_separa_galaxia_central_de_estrella():
    img = sersic(1.0, 15, 300.0) + sersic(1.0, 1.0, 30.0, centro=(60.0, 60.0))
    img = img + np.random.default_rng(2).normal(0, RUIDO, img.shape).astype(np.float32)
    gal, otras = IE.mascara_gaussiana(img, RUIDO)
    assert gal[150, 150] and not gal[60, 60] and otras[60, 60]


def test_rotacion_arbitraria_rellena_y_giro_de_90_no():
    rot = IE.fraccion_relleno_rotacion(300, 45).set_index("ángulo")["fracción rellenada"]
    assert rot[0] == 0 and rot[90] == 0
    assert 0.15 < rot[45] < 0.19                    # 1 − 2(√2 − 1) ≈ 0.172


def test_asinh_p75_estandariza_con_entrenamiento():
    rng = np.random.default_rng(0)
    X = rng.lognormal(-5, 1, (8, 3, 50, 50)).astype(np.float32)
    A, st = IE.asinh_p75(X, X)
    assert np.allclose(A.mean(axis=(0, 2, 3)), 0, atol=0.05) and np.allclose(A.std(axis=(0, 2, 3)), 1, atol=0.05)
    assert st["s_b"].shape == (3,)
