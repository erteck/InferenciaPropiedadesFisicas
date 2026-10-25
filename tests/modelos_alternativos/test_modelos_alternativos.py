"""Pruebas del paquete src/modelos_alternativos (sin descargar pesos: extractores con pesos aleatorios)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.modelos_alternativos import arquitecturas as A
from src.modelos_alternativos import datos as D
from src.modelos_alternativos import entrenamiento as E
from src.modelos_alternativos import evaluacion as V
from src.modelos_alternativos import interpretacion as I
from src.modelos_alternativos import representaciones as R

CPU = torch.device("cpu")
CNN = dict(nombre_extractor="convnext_atto", preentrenado=False)
VIT = dict(nombre_extractor="vit_small_patch16_dinov3", preentrenado=False, lado_entrada=32)


def _imagenes(n=12, lado=32, semilla=0):
    rng = np.random.default_rng(semilla)
    x = rng.normal(0, 0.01, (n, 3, lado, lado)).astype(np.float32)
    c = lado // 2
    x[:, :, c - 3:c + 3, c - 3:c + 3] += rng.uniform(0.5, 2, (n, 3, 1, 1)).astype(np.float32)
    return x.astype(np.float16)


def _conjunto(n=12, lado=32, k_escalares=1, semilla=0):
    rng = np.random.default_rng(semilla)
    esc = rng.normal(size=(n, k_escalares)) if k_escalares else None
    return D.ConjuntoEnGPU(_imagenes(n, lado, semilla), esc, rng.normal(size=(n, 3)), CPU)


def _preparador(imagenes, lado=32, **kw):
    return D.PreparadorEntrada(D.TransformacionGRZ.ajustar(imagenes, np.arange(len(imagenes))), lado=lado, **kw)


# ------------------------------------------------------------------ datos
def test_transformacion_grz_estandariza_entrenamiento():
    x = _imagenes(20)
    t = D.TransformacionGRZ.ajustar(x, np.arange(20), posiciones_por_imagen=64)
    y = t.aplicar(torch.from_numpy(x))
    assert torch.allclose(y.mean((0, 2, 3)), torch.zeros(3), atol=1e-3)
    assert torch.allclose(y.std((0, 2, 3)), torch.ones(3), atol=2e-2)
    assert (t.escala_asinh > 0).all()


def test_filtrar_muestra_excluye_banda_vacia_y_redshift_invalido():
    df = pd.DataFrame({"mangaid": list("abcd"), "particion": ["Train"] * 4, "log_mstar": [10, 10, 10, np.nan],
                       "log_sfr": [0, 0, 0, 0], "d4000": [1.5] * 4, "redshift": [0.03, -1, 0.03, 0.03]})
    ceros = np.array([[0, 0, 0], [0, 0, 0], [1.0, 0, 0], [0, 0, 0]])
    m, exc = D.filtrar_muestra(df, ceros)
    assert list(m.mangaid) == ["a"] and len(exc) == 3
    assert set(exc.motivo) == {"redshift no válido", "banda completa en cero", "etiqueta no finita"}


def test_estandarizador_es_reversible():
    v = np.random.default_rng(0).normal(5, 2, (50, 3))
    e = D.EstandarizadorEscalar.ajustar(v)
    np.testing.assert_allclose(e.invertir(e.transformar(v)), v, atol=1e-5)


def test_lupton_en_rango_y_orden_de_canales():
    x = torch.zeros(1, 3, 4, 4); x[0, 2] = 1.0          # solo banda z
    rgb = D.rgb_lupton(x)
    assert rgb.min() >= 0 and rgb.max() <= 1
    assert rgb[0, 0].mean() > 0 and rgb[0, 1:].abs().sum() == 0   # z va al canal rojo


def test_d4_son_ocho_simetrias_distintas_y_aumento_conserva_forma():
    x = torch.arange(16.).view(1, 1, 4, 4)
    vistas = {tuple(D.simetria_d4(x, k).flatten().tolist()) for k in range(8)}
    assert len(vistas) == 8
    g = torch.Generator().manual_seed(0)
    assert D.aumentar_d4(torch.randn(5, 3, 8, 8), g).shape == (5, 3, 8, 8)


def test_ajustar_lado_recorte_y_reescala():
    x = torch.randn(2, 3, 30, 30)
    assert D.ajustar_lado(x, 24, "recorte").shape[-1] == 24
    assert D.ajustar_lado(x, 16, "reescala").shape[-1] == 16


# ------------------------------------------------------------------ arquitecturas
@pytest.mark.parametrize("base,k", [(CNN, 1), (CNN, 0), (VIT, 2)])
def test_modelo_acepta_cualquier_numero_de_escalares(base, k):
    modelo = A.construir_modelo(A.EspecificacionModelo(**base), n_escalares=k, n_salidas=3)
    esc = torch.randn(2, k) if k else None
    assert modelo(torch.randn(2, 3, 32, 32), esc).shape == (2, 3)


@pytest.mark.parametrize("agregacion,factor", [("cls", 1), ("promedio", 1), ("cls_y_promedio", 2)])
def test_agregacion_de_tokens_vit(agregacion, factor):
    ext = A.crear_extractor(A.EspecificacionModelo(**VIT, agregacion_tokens=agregacion))
    assert ext.es_de_tokens and ext(torch.randn(2, 3, 32, 32)).shape == (2, ext.red.num_features * factor)


def test_estrategias_de_ajuste():
    def entrenables(estrategia, **kw):
        m = A.construir_modelo(A.EspecificacionModelo(**VIT, estrategia_ajuste=estrategia, **kw), 1, 3)
        return A.contar_parametros(m.extractor)["parametros_entrenables"], m
    completo, _ = entrenables("completo")
    congelado, _ = entrenables("congelado")
    dos_bloques, _ = entrenables("ultimos_bloques", bloques_abiertos=2)
    lora, m_lora = entrenables("lora", rango_lora=4)
    assert congelado == 0 < lora < dos_bloques < completo
    assert all(not p.requires_grad for n, p in m_lora.extractor.named_parameters() if "lora_" not in n)
    m = A.construir_modelo(A.EspecificacionModelo(**CNN, estrategia_ajuste="ultimos_bloques", bloques_abiertos=1), 1, 3)
    assert 0 < A.contar_parametros(m.extractor)["parametros_entrenables"] < A.contar_parametros(m.extractor)["parametros_totales"]


def test_lora_empieza_identica_al_modelo_original():
    torch.manual_seed(0)
    base = A.construir_modelo(A.EspecificacionModelo(**VIT), 1, 3).eval()
    con_lora = A.construir_modelo(A.EspecificacionModelo(**VIT, estrategia_ajuste="lora"), 1, 3)
    con_lora.extractor.red.load_state_dict(base.extractor.red.state_dict(), strict=False)
    for b, l in zip(base.extractor.red.blocks, con_lora.extractor.red.blocks):
        l.attn.qkv.base.load_state_dict(b.attn.qkv.state_dict()); l.attn.proj.base.load_state_dict(b.attn.proj.state_dict())
    con_lora.cabeza.load_state_dict(base.cabeza.state_dict()); con_lora.eval()
    x, z = torch.randn(2, 3, 32, 32), torch.randn(2, 1)
    torch.testing.assert_close(base(x, z), con_lora(x, z), rtol=1e-5, atol=1e-5)


def test_grupos_de_parametros_cubren_solo_entrenables_y_aplican_decaimiento_por_capa():
    m = A.construir_modelo(A.EspecificacionModelo(**CNN), 1, 3)
    grupos = A.grupos_de_parametros(m, 1e-4, 1e-3, 0.05, decaimiento_por_capa=0.7)
    ids = [id(p) for g in grupos for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in m.parameters() if p.requires_grad)
    tasas_extractor = [g["lr"] for g in grupos if g["grupo"] == "extractor"]
    assert min(tasas_extractor) < max(tasas_extractor) <= 1e-4
    assert {g["lr"] for g in grupos if g["grupo"] == "cabeza"} == {1e-3}


# ------------------------------------------------------------------ entrenamiento y evaluación
def test_entrenar_reduce_perdida_y_guarda_mejor_epoca():
    tr, va = _conjunto(16), _conjunto(8, semilla=1)
    prep = _preparador(_imagenes(16))
    m = A.construir_modelo(A.EspecificacionModelo(**CNN), 1, 3)
    cfg = E.ConfiguracionEntrenamiento(epocas=3, epocas_calentamiento=1, tamano_lote=8, precision_mixta=False,
                                       canales_ultimos=False, tasa_extractor=1e-3, decaimiento_por_capa=1.0)
    epocas_vistas = []
    r = E.entrenar(m, prep, tr, va, cfg, al_terminar_epoca=lambda e, c: epocas_vistas.append(e), registro=lambda *_: None)
    assert epocas_vistas == [1, 2, 3] and 1 <= r.mejor_epoca <= 3
    assert r.mejor_criterio_validacion == pytest.approx(r.historial.criterio_validacion.min())
    pred = E.predecir(m, prep, va, precision_mixta=False)
    assert E.mse_medio_estandarizado(pred, va.etiquetas.numpy()) == pytest.approx(r.mejor_criterio_validacion, rel=1e-4)


def test_tta_promedia_las_ocho_vistas():
    va, prep = _conjunto(4), _preparador(_imagenes(4))
    m = A.construir_modelo(A.EspecificacionModelo(**CNN), 1, 3)
    con = E.predecir(m, prep, va, tta=True, precision_mixta=False)
    manual = np.mean([m.eval()(D.simetria_d4(prep(va.imagenes), k), va.escalares).detach().numpy() for k in range(8)], 0)
    np.testing.assert_allclose(con, manual, rtol=1e-4, atol=1e-5)


def test_corrida_guardada_se_recupera(tmp_path):
    from src.katachi_fits.io_guard import WriteGuard
    tr, va, prep = _conjunto(8), _conjunto(8, semilla=1), _preparador(_imagenes(8))
    espec = A.EspecificacionModelo(**CNN)
    m = A.construir_modelo(espec, 1, 3)
    cfg = E.ConfiguracionEntrenamiento(epocas=1, tamano_lote=8, precision_mixta=False, canales_ultimos=False)
    r = E.entrenar(m, prep, tr, va, cfg, registro=lambda *_: None)
    carpeta = tmp_path / "corrida"
    E.guardar_corrida(WriteGuard(tmp_path, [tmp_path / "ro"]), carpeta, r, {"especificacion": espec.a_dict()})
    assert E.corrida_terminada(carpeta)
    m2, resumen, hist = E.cargar_corrida(carpeta, A.construir_modelo(espec, 1, 3))
    np.testing.assert_allclose(E.predecir(m2, prep, va, precision_mixta=False), E.predecir(m, prep, va, precision_mixta=False),
                               rtol=1e-5, atol=1e-6)
    assert resumen["mejor_epoca"] == 1 and len(hist) == 1


def test_metricas_y_mse_normalizado():
    y = np.array([[0., 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    p = y + 0.5
    m = V.metricas(y[:, 0], p[:, 0], desviacion_entrenamiento=0.5)
    assert m["RMSE"] == pytest.approx(0.5) and m["sesgo"] == pytest.approx(0.5) and m["MSE_normalizado"] == pytest.approx(1.0)
    assert V.mse_normalizado_medio(y, p, [0.5, 1.0, 0.25]) == pytest.approx(np.mean([1, 0.25, 4]))
    d, lo, hi = V.diferencia_pareada_rmse(y[:, 0], y[:, 0] + 0.1, p[:, 0], n_remuestreos=200)
    assert d < 0 and V.veredicto(lo, hi, 0.02) == "mejor"


# ------------------------------------------------------------------ representaciones, búsqueda e interpretación
def test_cabeza_sobre_representaciones_aprende():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(200, 16)); z = rng.normal(size=(200, 1))
    y = np.c_[r[:, :1] + z, r[:, 1:2], -r[:, 2:3]]
    cfg = R.ConfiguracionCabeza(epocas=60, tamano_lote=64, unidades_ocultas=32, tasa_aprendizaje=3e-3)
    res = R.entrenar_cabeza(r[:150], z[:150], y[:150], r[150:], z[150:], y[150:], cfg, dispositivo=CPU)
    assert res.mejor_criterio_validacion < 0.5 * E.mse_medio_estandarizado(np.zeros_like(y[150:]), y[150:])
    assert res.predecir(r[150:], z[150:]).shape == (50, 3)


def test_extraer_representaciones_dimension():
    va, prep = _conjunto(5), _preparador(_imagenes(5))
    ext = A.crear_extractor(A.EspecificacionModelo(**VIT))
    assert R.extraer_representaciones(ext, prep, va, tamano_lote=2).shape == (5, ext.dimension)


def test_busqueda_reanuda_y_poda(tmp_path):
    from src.modelos_alternativos import busqueda as B
    proponer = lambda t: {"x": t.suggest_float("x", -2, 2)}
    def evaluar(valores, al_terminar_epoca):
        for epoca in range(1, 6):
            al_terminar_epoca(epoca, (valores["x"] - 0.5) ** 2 + 1 / epoca)
        return (valores["x"] - 0.5) ** 2
    respaldo = tmp_path / "estudio.db"
    e1 = B.ejecutar_busqueda("prueba_unitaria_busqueda", respaldo, proponer, evaluar, 6, registro=lambda *_: None)
    n_primera = len(e1.get_trials(deepcopy=False))
    e2 = B.ejecutar_busqueda("prueba_unitaria_busqueda", respaldo, proponer, evaluar, 8, registro=lambda *_: None)
    assert respaldo.exists() and n_primera == 6 and len(e2.trials) == 8
    assert any(t.state.name == "PRUNED" for t in e2.trials)          # la poda se activa con estas curvas
    nuevo = B.ejecutar_busqueda("prueba_unitaria_busqueda", tmp_path / "otro.db", proponer, evaluar, 2, registro=lambda *_: None)
    assert len(nuevo.trials) == 2                                      # sin respaldo, el estudio empieza de cero
    tabla = B.tabla_pruebas(e2)
    assert {"prueba", "estado", "criterio_validacion", "x"} <= set(tabla.columns)


def test_oclusion_cambia_prediccion_y_anillo_vacio_no():
    va, prep = _conjunto(6, k_escalares=2), _preparador(_imagenes(6))
    m = A.construir_modelo(A.EspecificacionModelo(**CNN), 2, 3)
    base = E.predecir(m, prep, va, precision_mixta=False)
    r50 = np.full(6, 4.0, np.float32)
    assert not np.allclose(I.predecir_con_modificacion(m, prep, va, banda=0, precision_mixta=False), base)
    assert not np.allclose(I.predecir_con_modificacion(m, prep, va, anillo=(0, 1), radio_mitad_luz_px=r50,
                                                       precision_mixta=False), base)
    np.testing.assert_allclose(I.predecir_con_modificacion(m, prep, va, anillo=(100, 200), radio_mitad_luz_px=r50,
                                                           precision_mixta=False), base, rtol=1e-5, atol=1e-6)
    permutada = I.predecir_con_modificacion(m, prep, va, escalar_permutado=1, semilla=3, precision_mixta=False)
    assert permutada.shape == base.shape
    # La oclusión se hace en flujo: con la composición de Lupton también funciona, y una banda en 0 sí cambia la entrada
    lupton = D.PreparadorEntrada(prep.transformacion_grz, "rgb_lupton", lado=32)
    assert not np.allclose(I.predecir_con_modificacion(m, lupton, va, banda=2, precision_mixta=False),
                           I.predecir_con_modificacion(m, lupton, va, precision_mixta=False))


def _contexto_sintetico(tmp_path, n=24):
    from src.katachi_fits.io_guard import WriteGuard
    from src.modelos_alternativos.contexto import ContextoDatos
    rng = np.random.default_rng(0)
    muestra = pd.DataFrame({"mangaid": [f"g{i}" for i in range(n)],
                            "particion": ["Train"] * (n // 2) + ["Validation"] * (n // 4) + ["Test"] * (n - n // 2 - n // 4),
                            "log_mstar": rng.normal(10, .5, n), "log_sfr": rng.normal(0, 1, n), "d4000": rng.normal(1.5, .2, n),
                            "redshift": rng.uniform(.01, .1, n)})
    imagenes = _imagenes(n)
    idx = D.indices_por_particion(muestra)
    return ContextoDatos(None, WriteGuard(tmp_path, [tmp_path / "ro"]), muestra, pd.DataFrame(), imagenes, idx,
                         D.OBJETIVOS, ("redshift",), D.TransformacionGRZ.ajustar(imagenes, idx["Train"], 64),
                         D.EstandarizadorEscalar.ajustar(muestra.iloc[idx["Train"]][list(D.OBJETIVOS)].to_numpy()),
                         D.EstandarizadorEscalar.ajustar(muestra.iloc[idx["Train"]][["redshift"]].to_numpy()))


def test_ejecutar_corrida_entrena_guarda_y_reutiliza(tmp_path):
    from src.modelos_alternativos import experimento as X
    ctx = _contexto_sintetico(tmp_path)
    espec = A.EspecificacionModelo(**CNN)
    cfg = E.ConfiguracionEntrenamiento(epocas=2, tamano_lote=8, precision_mixta=False, canales_ultimos=False)
    entrada = {"representacion": "grz_estandarizado", "lado": 32}
    r1 = X.ejecutar_corrida(ctx, tmp_path / "modelo", "prueba", espec, cfg, entrada, CPU, registro=lambda *_: None)
    r2 = X.ejecutar_corrida(ctx, tmp_path / "modelo", "prueba", espec, cfg, entrada, CPU, registro=lambda *_: None)
    assert not r1.reutilizada and r2.reutilizada and r1.identificador == r2.identificador
    p1 = X.predicciones_originales(ctx, r1, "Validation", CPU)
    np.testing.assert_allclose(p1, X.predicciones_originales(ctx, r2, "Validation", CPU), rtol=1e-5, atol=1e-6)
    assert p1.shape == (len(ctx.indices["Validation"]), 3) and abs(p1[:, 0].mean() - 10) < 3   # escala original
    otra = X.identificador_corrida("prueba", espec, E.ConfiguracionEntrenamiento(semilla=7), entrada)
    assert otra != r1.identificador
    reg = X.registrar_eleccion_antes_de_prueba(ctx, tmp_path / "registro.json", {"modelo": "x"}, p1, registro=lambda *_: None)
    assert X.registrar_eleccion_antes_de_prueba(ctx, tmp_path / "registro.json", {"modelo": "x"}, p1,
                                                registro=lambda *_: None)["fecha_utc"] == reg["fecha_utc"]
    tabla = X.tabla_corrida(ctx, p1, "Validation", modelo="prueba")
    assert set(tabla.objetivo) == set(D.OBJETIVOS) and "MSE_normalizado" in tabla


def test_cabeza_plegada_reproduce_la_cabeza_sobre_representaciones():
    rng = np.random.default_rng(1)
    r = rng.normal(3, 2, size=(80, 8)); z = rng.normal(size=(80, 1)); y = rng.normal(size=(80, 3))
    res = R.entrenar_cabeza(r[:60], z[:60], y[:60], r[60:], z[60:], y[60:], R.ConfiguracionCabeza(epocas=5, tamano_lote=16), CPU)
    plegada = R.plegar_estandarizacion(res).eval()
    directa = plegada(torch.cat([torch.as_tensor(r[60:], dtype=torch.float32), torch.as_tensor(z[60:], dtype=torch.float32)], 1))
    np.testing.assert_allclose(directa.detach().numpy(), res.predecir(r[60:], z[60:]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("agregacion", ["cls", "cls_y_promedio"])
def test_modelo_congelado_equivale_a_cabeza_sobre_representaciones(tmp_path, agregacion):
    from src.modelos_alternativos import experimento as X
    ctx = _contexto_sintetico(tmp_path)
    espec = A.EspecificacionModelo(**VIT, agregacion_tokens=agregacion, estrategia_ajuste="congelado")
    entrada = {"representacion": "grz_estandarizado", "lado": 32}
    torch.manual_seed(0)
    reps = X.representaciones_en_cache(ctx, tmp_path / "m", espec, entrada, ("Train", "Validation"), CPU, registro=lambda *_: None)
    assert (tmp_path / "m" / "representaciones").exists()
    de_disco = X.representaciones_en_cache(ctx, tmp_path / "m", espec, entrada, ("Train", "Validation"), CPU, registro=lambda *_: None)
    np.testing.assert_array_equal(de_disco["Validation"], reps["Validation"])        # la caché se lee de vuelta completa
    cfg = R.ConfiguracionCabeza(epocas=5, tamano_lote=8)
    torch.manual_seed(0)   # mismos pesos aleatorios del extractor que en la extracción
    corrida = X.ejecutar_cabeza_congelada(ctx, tmp_path / "m", "congelado", espec, cfg, entrada, reps, CPU, registro=lambda *_: None)
    validacion = ctx.conjunto("Validation", CPU)
    pred_modelo = E.predecir(corrida.modelo, corrida.preparador, validacion, precision_mixta=False)
    propias = R.extraer_representaciones(corrida.modelo.extractor, corrida.preparador, validacion)   # vectores del extractor del modelo
    res = R.cargar_cabeza(corrida.carpeta / "cabeza.pt", propias.shape[1] + 1, 3, cfg)
    np.testing.assert_allclose(pred_modelo, res.predecir(propias, validacion.escalares.numpy()), rtol=1e-4, atol=1e-5)
    solo_cabeza = X.ejecutar_cabeza_congelada(ctx, tmp_path / "m", "busqueda", espec, cfg, entrada, reps, CPU, guardar=False,
                                              construir_modelo_imagen=False, registro=lambda *_: None)
    assert solo_cabeza.modelo is None and np.isfinite(solo_cabeza.resumen["mejor_criterio_validacion"])
    otra = X.ejecutar_cabeza_congelada(ctx, tmp_path / "m", "congelado", espec, cfg, entrada, reps, CPU, registro=lambda *_: None)
    assert otra.reutilizada and corrida.resumen["parametros_entrenables"] < corrida.resumen["parametros_totales"]
