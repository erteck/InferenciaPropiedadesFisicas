"""Pruebas de la ingeniería de características con galaxias sintéticas de valores conocidos.

Un perfil de Sérsic n = 1 (exponencial) circular tiene, en forma analítica:
    R50 = 1.678 h,  R90 = 3.890 h,  R_P(η = 0.2) = 2.16 R50 (Graham & Driver 2005),
y C = 5 log10(R80/R20) ≈ 2.7 dentro de 1.5 R_P. Un perfil de de Vaucouleurs (n = 4)
es más concentrado: C ≈ 5 con aperturas infinitas, menor dentro de 1.5 R_P.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaincinv

from src.caracteristicas_desi import config as K
from src.caracteristicas_desi.fisica import agregar_variables_fisicas, distancias
from src.caracteristicas_desi.fotometria import cielo, medir_fotometria, perfil, radio_fraccion, radio_petrosian
from src.caracteristicas_desi.imagenes import estampa
from src.caracteristicas_desi.morfologia import asimetria, gini, m20, medir_morfologia, suavidad
from src.caracteristicas_desi.pipeline import medir_galaxia
from src.caracteristicas_desi.transformaciones import TransformadorAsimetria

N = K.RECORTE_PX


def sersic(n_idx: float, re_px: float, flujo: float, b_a: float = 1.0, pa: float = 0.0, n: int = N,
           centro=None, sub: int = 4) -> np.ndarray:
    """Imagen de Sérsic con flujo total `flujo`, integrada con submuestreo sub × sub."""
    c = (n - 1) / 2 if centro is None else centro
    s = (np.arange(n * sub) + 0.5) / sub - 0.5
    X, Y = np.meshgrid(s, s)
    dx, dy = X - c[0] if isinstance(c, tuple) else X - c, Y - c[1] if isinstance(c, tuple) else Y - c
    ca, sa = np.cos(pa), np.sin(pa)
    u, v = dx * ca + dy * sa, -dx * sa + dy * ca
    r = np.hypot(u, v / b_a)
    bn = gammaincinv(2 * n_idx, 0.5)
    img = np.exp(-bn * ((r / re_px) ** (1 / n_idx) - 1))
    img = img.reshape(n, sub, n, sub).mean(axis=(1, 3))
    return (img / img.sum() * flujo).astype(np.float32)


def cubo(img: np.ndarray, colores=(0.6, 1.0, 1.4), ruido=0.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.stack([img * f for f in colores])
    if ruido:
        c = c + rng.normal(0, ruido, c.shape).astype(np.float32)
    return c


# --- Fotometría ---------------------------------------------------------------
def test_radios_de_un_exponencial():
    h = 6.0
    re_px = 1.678 * h
    img = sersic(1.0, re_px, 1000.0)
    p = perfil(img)
    rp = radio_petrosian(p, (N - 1) / 2 - 1)
    assert rp == pytest.approx(2.16 * re_px, rel=0.03)
    tot = p.F(2 * rp)
    assert radio_fraccion(p, tot, 0.5, 2 * rp) == pytest.approx(re_px, rel=0.03)
    assert radio_fraccion(p, tot, 0.9, 2 * rp) == pytest.approx(3.89 * h, rel=0.04)


def test_flujo_y_color_recuperados_con_ruido():
    img = sersic(1.0, 10.0, 500.0)
    m = medir_fotometria(cubo(img, colores=(0.5, 1.0, 1.5), ruido=0.01))
    v = m.valores
    # 2 R_P de un exponencial contiene ~99 % del flujo total.
    assert v["flujo_r"] == pytest.approx(500.0, rel=0.03)
    assert v["m_g"] - v["m_r"] == pytest.approx(-2.5 * np.log10(0.5), abs=0.02)
    assert v["R50"] == pytest.approx(10.0 * K.PIXSCALE, rel=0.05)
    assert not v["apertura_truncada"]


def test_cielo_se_mide_y_se_resta():
    rng = np.random.default_rng(1)
    img = rng.normal(0.05, 0.004, (N, N)).astype(np.float32)
    med, std = cielo(img)
    assert med == pytest.approx(0.05, abs=5e-4)
    assert std == pytest.approx(0.004, rel=0.05)


def test_vecino_no_contamina_el_flujo():
    gal = sersic(1.0, 8.0, 300.0)
    vecino = sersic(0.5, 2.0, 200.0, centro=((N - 1) / 2 + 45, (N - 1) / 2))
    solo = medir_fotometria(cubo(gal, ruido=0.005)).valores["flujo_r"]
    con = medir_fotometria(cubo(gal + vecino, ruido=0.005)).valores
    assert con["flujo_r"] == pytest.approx(solo, rel=0.03)    # sin máscara serían +200 (≈ +67 %)
    assert con["n_fuentes"] >= 1


def test_galaxia_mas_grande_que_el_recorte_se_marca():
    m = medir_fotometria(cubo(sersic(1.0, 70.0, 5000.0)))
    v = m.valores
    assert (not v["petro_converge"]) or v["apertura_truncada"]


# --- Morfología -----------------------------------------------------------------
def test_concentracion_disco_frente_a_esferoide():
    disco = medir_morfologia(medir_fotometria(cubo(sersic(1.0, 10.0, 500.0))))
    esfer = medir_morfologia(medir_fotometria(cubo(sersic(4.0, 10.0, 500.0))))
    assert disco["C"] == pytest.approx(2.7, abs=0.15)
    assert esfer["C"] > disco["C"] + 1.0
    assert esfer["gini"] > disco["gini"]
    assert esfer["m20"] < disco["m20"]          # luz brillante más concentrada en el centro


def test_asimetria_cero_en_simetrica_y_positiva_con_grumo():
    img = sersic(1.0, 10.0, 500.0)
    rp = 2.16 * 10.0
    a0, dx, dy = asimetria(img, 0.0, rp)
    assert abs(a0) < 0.01 and dx == 0 and dy == 0
    grumo = img.copy()
    c = N // 2
    grumo[c + 8:c + 12, c + 5:c + 9] += img.max() * 0.5
    a1, *_ = asimetria(grumo, 0.0, rp)
    assert a1 > 0.05


def test_asimetria_corrige_el_ruido():
    rng = np.random.default_rng(3)
    img = sersic(1.0, 10.0, 200.0)
    sig = 0.01
    ruidosa = img + rng.normal(0, sig, img.shape).astype(np.float32)
    a_sin_corr, *_ = asimetria(ruidosa, 0.0, 21.6)
    a_corr, *_ = asimetria(ruidosa, sig, 21.6)
    assert a_sin_corr > 0.05                      # el ruido simula asimetría
    assert abs(a_corr) < 0.03                     # y la corrección la elimina


def test_suavidad_detecta_grumos():
    rng = np.random.default_rng(4)
    img = sersic(1.0, 12.0, 800.0)
    liso = suavidad(img, 0.0, 26.0)
    grumos = img.copy()
    for _ in range(20):
        y, x = rng.integers(N // 2 - 25, N // 2 + 25, 2)
        grumos[y:y + 2, x:x + 2] += img[y, x] * 2
    # Un perfil liso no da exactamente 0: suavizar isofotas curvas deja residuos
    # positivos pequeños (también en statmorph). Lo relevante es el contraste.
    assert abs(liso) < 0.04
    assert suavidad(grumos, 0.0, 26.0) > liso + 0.03


def test_gini_casos_limite():
    seg = np.ones((10, 10), bool)
    assert gini(np.ones((10, 10)), seg) == pytest.approx(0.0, abs=1e-9)
    un_pixel = np.zeros((10, 10)); un_pixel[5, 5] = 1.0
    assert gini(un_pixel, seg) == pytest.approx(1.0, abs=1e-9)


def test_m20_rango_razonable():
    v = m20(sersic(1.0, 10.0, 500.0), np.ones((N, N), bool))
    assert -2.5 < v < -1.0


def test_b_a_de_una_eliptica():
    m = medir_fotometria(cubo(sersic(1.0, 12.0, 500.0, b_a=0.5, pa=0.6)))
    assert medir_morfologia(m)["b_a"] == pytest.approx(0.5, abs=0.06)


# --- Física ------------------------------------------------------------------------
def test_distancias_contra_valores_de_astropy():
    from astropy.cosmology import FlatLambdaCDM
    cos = FlatLambdaCDM(H0=73, Om0=0.3)
    dm, esc = distancias(np.array([0.03, np.nan]))
    assert dm[0] == pytest.approx(cos.distmod(0.03).value, abs=1e-6)
    assert esc[0] == pytest.approx(cos.kpc_proper_per_arcmin(0.03).value / 60, rel=1e-6)
    assert np.isnan(dm[1])


def test_variables_fisicas_y_extincion():
    df = pd.DataFrame({**{f"m_{b}": [16.0] for b in "grz"}, **{f"m_in_{b}": [17.0] for b in "grz"},
                       **{f"m_out_{b}": [17.5] for b in "grz"}, "R50": [5.0], "R90": [12.0], "ebv": [0.1], "z": [0.03]})
    d = agregar_variables_fisicas(df)
    assert d.m_g0[0] == pytest.approx(16.0 - 0.3214)
    assert d.g_r[0] == pytest.approx(-(3.214 - 2.165) * 0.1)
    a, b, c = K.EBROVA
    assert d.logM_ebrova[0] == pytest.approx(a * d.M_g[0] + b * d.M_r[0] + c)
    assert d.C_sdss[0] == pytest.approx(2.4)


# --- Estampas y robustez -------------------------------------------------------------
def test_estampa_alineada_y_finita():
    m = medir_fotometria(cubo(sersic(1.0, 10.0, 500.0, b_a=0.4, pa=1.0)))
    mo = medir_morfologia(m)
    e = estampa(m.limpio, m.rp_px, mo["pa_rad"])
    assert e.shape == (3, K.ESTAMPA_PX, K.ESTAMPA_PX) and np.isfinite(e).all()
    c = K.ESTAMPA_PX // 2
    # Tras alinear, la luz se extiende más a lo largo del eje horizontal.
    assert e[1, c, :].sum() > e[1, :, c].sum()


def test_medir_galaxia_no_falla_con_imagen_vacia():
    v, e = medir_galaxia(np.zeros((3, N, N), np.float32))
    assert v["estado"] in ("ok", "fallo")
    v, e = medir_galaxia(cubo(sersic(1.0, 8.0, 300.0), ruido=0.005))
    assert v["estado"] == "ok" and np.isfinite(v["m_r"])


# --- Transformaciones ---------------------------------------------------------------
def test_regla_de_asimetria():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"lognormal": rng.lognormal(0, 1, 2000), "normal": rng.normal(0, 1, 2000),
                      "negativa_sesgada": rng.exponential(1, 2000) - 0.5, "binaria": rng.integers(0, 2, 2000)})
    t = TransformadorAsimetria(umbral=1.0).fit(X)
    rep = t.reporte_
    assert rep.loc["lognormal", "transformacion"] in ("log10", "box-cox")
    assert rep.loc["normal", "transformacion"] == "ninguna"
    assert rep.loc["negativa_sesgada", "transformacion"] == "yeo-johnson"
    assert rep.loc["binaria", "transformacion"] == "ninguna"
    Y = t.transform(X)
    assert abs(Y["lognormal"].skew()) < 0.2 and abs(Y["negativa_sesgada"].skew()) < 0.3
    np.testing.assert_allclose(Y["normal"], X["normal"])


def test_transformacion_se_ajusta_en_train_y_se_aplica_igual():
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({"x": rng.lognormal(0, 1, 500)})
    va = pd.DataFrame({"x": [0.5, 1.0, np.nan]})
    t = TransformadorAsimetria().fit(tr)
    out = t.transform(va)
    assert np.isnan(out.x[2]) and np.isfinite(out.x[:2]).all()


def test_transformacion_nunca_devuelve_infinitos():
    tr = pd.DataFrame({"x": np.r_[np.zeros(990), np.linspace(1, 1e6, 10)]})
    out = TransformadorAsimetria().fit(tr).transform(pd.DataFrame({"x": [np.inf, 1e12, 0.0]}))
    assert not np.isinf(out.x).any()
