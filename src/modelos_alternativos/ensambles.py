"""Ensambles de modelos base a partir de sus predicciones.

Todo trabaja con arreglos `P` de forma (miembros, galaxias, propiedades) y con las etiquetas `y` de
forma (galaxias, propiedades). Un miembro puede ser una red, o el promedio de las semillas de una
familia (capa homogénea), y la combinación de familias distintas es la capa heterogénea.

- **Mezcla convexa** (*blending*): promedio ponderado con pesos no negativos que suman uno, aprendidos en
  el conjunto de reserva y regularizados hacia el promedio uniforme (Breiman 1996).
- **Meta-aprendiz** (*stacking*): un modelo que recibe las predicciones de todos los miembros y
  predice la **corrección del promedio uniforme**. Así la regularización lo acerca al promedio y no a
  una constante (Wolpert 1992).
- **Validación cruzada anidada**: el ciclo interno elige los hiperparámetros y el externo mide el
  desempeño en galaxias que no intervinieron en esa elección (Cawley y Talbot 2010).
- **Diversidad**: descomposición de Krogh y Vedelsby (1995), error del ensamble = error medio de los
  miembros − ambigüedad (desacuerdo entre ellos).

La métrica principal es la del protocolo del equipo: el promedio entre propiedades de (RMSE/σ_entrenamiento)².
"""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TIPOS_META = ("promedio", "mezcla_convexa", "lineal_positivo", "elastica_positiva", "ridge_correccion", "bosque_residual")


# ---------------------------------------------------------------------------
# Métrica principal
# ---------------------------------------------------------------------------
def perdida_por_galaxia(y, p, sigma) -> np.ndarray:
    """Error cuadrático normalizado de cada galaxia, promediado entre propiedades: (n,)."""
    return np.mean(((np.asarray(p, float) - y) / sigma) ** 2, axis=1)


def criterio(y, p, sigma) -> float:
    """MSE normalizado medio (menor es mejor); equivale a promediar (RMSE/σ)² entre propiedades."""
    return float(np.mean(((np.asarray(p, float) - y) / sigma) ** 2))


def criterio_por_propiedad(y, p, sigma) -> np.ndarray:
    return np.mean(((np.asarray(p, float) - y) / sigma) ** 2, axis=0)


# ---------------------------------------------------------------------------
# Mezcla convexa
# ---------------------------------------------------------------------------
def pesos_mezcla_convexa(P, y, sigma, regularizacion: float = 0.0, alcance: str = "por_objetivo") -> np.ndarray:
    """Pesos (miembros, propiedades) no negativos con suma uno en cada propiedad.

    Minimizan el error normalizado más `regularizacion` · ‖w − uniforme‖². Con `alcance="global"` las
    tres propiedades comparten los mismos pesos.
    """
    P = np.asarray(P, float)
    M, n, J = P.shape
    uniforme = np.full(M, 1.0 / M)

    def resolver(propiedades) -> np.ndarray:
        A = [P[:, :, j].T / sigma[j] for j in propiedades]
        b = [y[:, j] / sigma[j] for j in propiedades]
        Q = sum(a.T @ a for a in A) / (n * len(A)) + regularizacion * np.eye(M)
        c = -(sum(a.T @ bb for a, bb in zip(A, b)) / (n * len(A)) + regularizacion * uniforme)
        resultado = minimize(lambda w: w @ Q @ w + 2 * c @ w, uniforme, jac=lambda w: 2 * (Q @ w + c), method="SLSQP",
                             bounds=[(0.0, 1.0)] * M, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                                                     "jac": lambda w: np.ones(M)}],
                             options={"ftol": 1e-12, "maxiter": 300})
        w = np.clip(resultado.x, 0.0, None)
        return w / w.sum()

    if alcance == "global":
        return np.repeat(resolver(range(J))[:, None], J, axis=1)
    return np.stack([resolver([j]) for j in range(J)], axis=1)


def aplicar_pesos(P, W) -> np.ndarray:
    """Combina los miembros con pesos (miembros, propiedades): (galaxias, propiedades)."""
    return np.einsum("mnj,mj->nj", np.asarray(P, float), np.asarray(W, float))


# ---------------------------------------------------------------------------
# Ajuste y predicción de cualquier configuración
# ---------------------------------------------------------------------------
def _entradas(P, z, config, estadisticas=None, propiedad=None):
    """Matriz estandarizada de entradas del meta-aprendiz: predicciones (+ redshift, + desacuerdo)."""
    M, n, J = P.shape
    bloques = [P[:, :, propiedad].T if propiedad is not None else P.transpose(1, 0, 2).reshape(n, M * J)]
    if config.get("usar_z") and z is not None:
        bloques.append(np.asarray(z, float)[:, None])
    if config.get("usar_desacuerdo"):
        d = P.std(0)
        bloques.append(d[:, [propiedad]] if propiedad is not None else d)
    X = np.hstack(bloques)
    if estadisticas is None:
        estadisticas = (X.mean(0), X.std(0) + 1e-12)
    return (X - estadisticas[0]) / estadisticas[1], estadisticas


def ajustar_predecir(config: dict, P_ajuste, y_ajuste, z_ajuste, P_prediccion, z_prediccion, sigma) -> np.ndarray:
    """Ajusta la configuración con las galaxias de ajuste y predice las galaxias de predicción: (n_pred, propiedades)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

    tipo = config["tipo"]
    P_ajuste, P_prediccion = np.asarray(P_ajuste, float), np.asarray(P_prediccion, float)
    M, n_aj, J = P_ajuste.shape
    base_aj, base_pr = P_ajuste.mean(0), P_prediccion.mean(0)
    if tipo == "promedio":
        return base_pr
    if tipo == "mezcla_convexa":
        W = pesos_mezcla_convexa(P_ajuste, y_ajuste, sigma, config.get("lambda", 0.0), config.get("alcance", "por_objetivo"))
        return aplicar_pesos(P_prediccion, W)
    salida = np.empty((P_prediccion.shape[1], J))
    if tipo in ("lineal_positivo", "elastica_positiva"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(J):
                modelo = (LinearRegression(positive=True) if tipo == "lineal_positivo"
                          else ElasticNet(alpha=config["alpha"], l1_ratio=config["l1_ratio"], positive=True, max_iter=20000))
                modelo.fit(P_ajuste[:, :, j].T / sigma[j], y_ajuste[:, j] / sigma[j])
                salida[:, j] = modelo.predict(P_prediccion[:, :, j].T / sigma[j]) * sigma[j]
        return salida
    residuo = (y_ajuste - base_aj) / sigma                 # corrección que debe aprender el meta-aprendiz
    if tipo == "ridge_correccion":
        if config.get("entradas", "todas") == "misma_propiedad":
            for j in range(J):
                X, est = _entradas(P_ajuste, z_ajuste, config, propiedad=j)
                Xp, _ = _entradas(P_prediccion, z_prediccion, config, est, propiedad=j)
                salida[:, j] = Ridge(alpha=config["alpha"]).fit(X, residuo[:, j]).predict(Xp)
        else:
            X, est = _entradas(P_ajuste, z_ajuste, config)
            Xp, _ = _entradas(P_prediccion, z_prediccion, config, est)
            salida[:] = Ridge(alpha=config["alpha"]).fit(X, residuo).predict(Xp)
        return base_pr + salida * sigma
    if tipo == "bosque_residual":
        X, est = _entradas(P_ajuste, z_ajuste, config)
        Xp, _ = _entradas(P_prediccion, z_prediccion, config, est)
        for j in range(J):
            modelo = HistGradientBoostingRegressor(max_depth=int(config["profundidad"]), learning_rate=float(config["tasa"]),
                                                   max_iter=int(config["iteraciones"]), min_samples_leaf=30,
                                                   l2_regularization=1.0, random_state=0)
            salida[:, j] = modelo.fit(X, residuo[:, j]).predict(Xp)
        return base_pr + salida * sigma
    raise ValueError(f"Tipo de combinación desconocido: {tipo}")


# ---------------------------------------------------------------------------
# Validación cruzada y búsqueda
# ---------------------------------------------------------------------------
def pliegues(n: int, k: int, semilla: int, repeticion: int = 0) -> list:
    orden = np.random.default_rng(semilla + 1000 * repeticion).permutation(n)
    return np.array_split(orden, k)


def _recorte(z, idx):
    return None if z is None else np.asarray(z)[idx]


def validar_configuracion(config: dict, P, y, z, sigma, k: int = 5, semilla: int = 0, repeticion: int = 0):
    """Predicciones fuera de pliegue de una configuración y su criterio."""
    oof = np.empty_like(y, dtype=float)
    for prueba in pliegues(len(y), k, semilla, repeticion):
        ajuste = np.setdiff1d(np.arange(len(y)), prueba)
        oof[prueba] = ajustar_predecir(config, P[:, ajuste], y[ajuste], _recorte(z, ajuste), P[:, prueba], _recorte(z, prueba), sigma)
    return oof, criterio(y, oof, sigma)


def buscar_configuracion(P, y, z, sigma, proponer=None, candidatos=None, pliegues_internos: int = 4, n_pruebas: int = 40,
                         semilla: int = 0):
    """Elige la mejor configuración por validación cruzada interna.

    Con `candidatos` (lista de configuraciones) prueba todas; con `proponer(prueba) -> configuración`
    usa Optuna (TPE) durante `n_pruebas` pruebas. Devuelve la mejor configuración y la tabla de pruebas.
    """
    filas, configs = [], []

    def evaluar(config):
        _, valor = validar_configuracion(config, P, y, z, sigma, pliegues_internos, semilla)
        configs.append(config)
        filas.append({**{k: v for k, v in config.items()}, "criterio": valor})
        return valor

    if candidatos is not None:
        for config in candidatos:
            evaluar(config)
    else:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=semilla))
        estudio.optimize(lambda prueba: evaluar(proponer(prueba)), n_trials=n_pruebas)
    tabla = pd.DataFrame(filas)
    return dict(configs[int(tabla.criterio.idxmin())]), tabla.sort_values("criterio").reset_index(drop=True)


def validacion_anidada(P, y, z, sigma, proponer=None, candidatos=None, pliegues_externos: int = 5, repeticiones: int = 2,
                       pliegues_internos: int = 4, n_pruebas: int = 30, semilla: int = 0, n_trabajos: int = 1) -> dict:
    """Desempeño honesto de una familia de combinaciones: la búsqueda interna nunca ve el pliegue externo.

    Devuelve las predicciones fuera de pliegue de cada repetición, su criterio y las configuraciones elegidas.
    """
    from joblib import Parallel, delayed

    def una(rep, k, prueba):
        ajuste = np.setdiff1d(np.arange(len(y)), prueba)
        config, _ = buscar_configuracion(P[:, ajuste], y[ajuste], _recorte(z, ajuste), sigma, proponer, candidatos,
                                         pliegues_internos, n_pruebas, semilla + 100 * rep + k)
        pred = ajustar_predecir(config, P[:, ajuste], y[ajuste], _recorte(z, ajuste), P[:, prueba], _recorte(z, prueba), sigma)
        return rep, k, prueba, pred, config

    tareas = [(rep, k, prueba) for rep in range(repeticiones)
              for k, prueba in enumerate(pliegues(len(y), pliegues_externos, semilla, rep))]
    resultados = Parallel(n_jobs=n_trabajos)(delayed(una)(*t) for t in tareas)
    oof = [np.empty_like(y, dtype=float) for _ in range(repeticiones)]
    elegidas = []
    for rep, k, prueba, pred, config in resultados:
        oof[rep][prueba] = pred
        elegidas.append({"repeticion": rep, "pliegue": k, **config})
    criterios = [criterio(y, o, sigma) for o in oof]
    return {"oof_por_repeticion": oof, "oof": oof[0], "criterios": criterios, "criterio_medio": float(np.mean(criterios)),
            "criterio_desviacion": float(np.std(criterios, ddof=1)) if repeticiones > 1 else float("nan"),
            "configuraciones": pd.DataFrame(elegidas)}


# ---------------------------------------------------------------------------
# Diversidad e incertidumbre
# ---------------------------------------------------------------------------
def correlacion_residuos(P, y) -> np.ndarray:
    """Correlación entre los residuos de los miembros, por propiedad: (propiedades, miembros, miembros)."""
    R = np.asarray(P, float) - y[None]
    return np.stack([np.corrcoef(R[:, :, j]) for j in range(R.shape[2])])


def descomposicion_ambiguedad(P, y, sigma, W=None, nombres_objetivos=None) -> pd.DataFrame:
    """Error del ensamble = error medio de los miembros − ambigüedad (Krogh y Vedelsby 1995), en unidades de σ².

    Es exacta con pesos convexos. Una ambigüedad grande frente al error medio indica que combinar ayuda.
    """
    P = np.asarray(P, float)
    M, n, J = P.shape
    W = np.full((M, J), 1.0 / M) if W is None else np.asarray(W, float)
    combinado = aplicar_pesos(P, W)
    filas = []
    for j in range(J):
        error_miembros = float(np.sum(W[:, j] * np.mean((P[:, :, j] - y[None, :, j]) ** 2, axis=1))) / sigma[j] ** 2
        ambiguedad = float(np.sum(W[:, j] * np.mean((P[:, :, j] - combinado[None, :, j]) ** 2, axis=1))) / sigma[j] ** 2
        filas.append({"propiedad": nombres_objetivos[j] if nombres_objetivos else j, "error medio de los miembros": error_miembros,
                      "ambigüedad": ambiguedad, "error del ensamble": float(np.mean((combinado[:, j] - y[:, j]) ** 2)) / sigma[j] ** 2})
    return pd.DataFrame(filas)


def desacuerdo_ponderado(P, W=None) -> np.ndarray:
    """Desviación ponderada de los miembros respecto al ensamble, por galaxia y propiedad: (n, propiedades)."""
    P = np.asarray(P, float)
    M, _, J = P.shape
    W = np.full((M, J), 1.0 / M) if W is None else np.asarray(W, float)
    combinado = aplicar_pesos(P, W)
    return np.sqrt(np.einsum("mnj,mj->nj", (P - combinado[None]) ** 2, W))


# ---------------------------------------------------------------------------
# Comparación y selección
# ---------------------------------------------------------------------------
def diferencia_pareada_criterio(y, p_a, p_b, sigma, n_remuestreos: int = 2000, semilla: int = 0):
    """criterio(a) − criterio(b) con intervalo del 95 % por bootstrap pareado de galaxias (negativo = mejor a)."""
    d = perdida_por_galaxia(y, p_a, sigma) - perdida_por_galaxia(y, p_b, sigma)
    idx = np.random.default_rng(semilla).integers(0, len(d), (n_remuestreos, len(d)))
    medias = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))


def regla_un_error_estandar(perdidas: dict, complejidad: dict):
    """Entre los modelos que no se distinguen del mejor en más de un error estándar, elige el más simple.

    `perdidas` = {modelo: pérdida por galaxia, sobre las mismas galaxias}; `complejidad` = {modelo: número que crece
    con el costo (p. ej., redes por galaxia)}. Como todos los modelos se evalúan en las mismas galaxias, el error
    estándar es el de la diferencia pareada con el mejor: descuenta la variación común (galaxias difíciles para
    todos) y mide solo lo que separa a los dos modelos.
    """
    medias = {k: float(np.mean(v)) for k, v in perdidas.items()}
    mejor = min(medias, key=medias.get)
    n = len(perdidas[mejor])
    diferencia = {k: float(np.mean(np.asarray(v) - perdidas[mejor])) for k, v in perdidas.items()}
    error = {k: float(np.std(np.asarray(v) - perdidas[mejor], ddof=1) / np.sqrt(n)) for k, v in perdidas.items()}
    dentro = [k for k in medias if k == mejor or diferencia[k] <= error[k]]
    elegido = min(dentro, key=lambda k: (complejidad[k], medias[k]))
    tabla = pd.DataFrame({"criterio": medias, "diferencia con el mejor": diferencia, "error estándar de la diferencia": error,
                          "complejidad": complejidad}).sort_values("criterio")
    tabla["dentro de 1 error estándar"] = tabla.index.isin(dentro)
    error_sin_parear = float(np.std(perdidas[mejor], ddof=1) / np.sqrt(n))
    return elegido, tabla, {"mejor": mejor, "error_estandar_sin_parear": error_sin_parear}


def horas_gpu_por_millon(ms_por_galaxia: float) -> float:
    """Horas de GPU para predecir un millón de galaxias (extrapolación a partir de milisegundos por galaxia)."""
    return float(ms_por_galaxia) * 1e6 / 3.6e6


def pendiente_de_calibracion(y, p) -> float:
    """Pendiente de la regresión de la predicción sobre la referencia; menor que 1 = compresión hacia el centro."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.cov(p, y)[0, 1] / np.var(y, ddof=1))


def registrar_eleccion(ruta, eleccion: dict, predicciones_validacion, registro=print) -> dict:
    """Guarda la elección del modelo final junto con una huella de sus predicciones de validación, antes de abrir la prueba.

    Si ya existe un registro, lo conserva y avisa cuando la elección cambió: en ese caso la evaluación en
    prueba ya no es independiente de las decisiones tomadas.
    """
    huella = hashlib.sha256(np.round(np.asarray(predicciones_validacion, float), 6).tobytes()).hexdigest()
    ruta = Path(ruta)
    if ruta.exists():
        previo = json.loads(ruta.read_text())
        registro(f"La prueba se abrió por primera vez el {previo['fecha_utc']}")
        if previo.get("huella_predicciones_validacion") != huella:
            registro("AVISO: el modelo elegido cambió desde la primera apertura; esta evaluación no es una prueba independiente.")
        return previo
    nuevo = {**eleccion, "huella_predicciones_validacion": huella, "fecha_utc": datetime.now(timezone.utc).isoformat()}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(nuevo, indent=2, ensure_ascii=False, default=str))
    registro("Elección registrada antes de abrir la prueba.")
    return nuevo
