"""Métricas de evaluación de los modelos alternativos, independientes del modelo.

Protocolo común del equipo:
- MAE, RMSE, sesgo (media de predicción − etiqueta) y R², en la escala original de cada propiedad.
- MSE normalizado por propiedad: (RMSE / σ_entrenamiento)²; el **criterio de selección** es su
  promedio entre propiedades, que permite combinar errores en dex con errores adimensionales.

Además: σ_NMAD (dispersión robusta), intervalos por bootstrap y diferencias pareadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metricas(etiqueta, prediccion, desviacion_entrenamiento: float | None = None) -> dict:
    y, p = np.asarray(etiqueta, float), np.asarray(prediccion, float)
    validas = np.isfinite(y) & np.isfinite(p)
    y, p = y[validas], p[validas]
    residuo = p - y
    rmse = float(np.sqrt(np.mean(residuo ** 2)))
    mediana = np.median(residuo)
    salida = {"n": int(len(y)), "MAE": float(np.mean(np.abs(residuo))), "RMSE": rmse, "sesgo": float(residuo.mean()),
              "sigma_NMAD": float(1.4826 * np.median(np.abs(residuo - mediana))),
              "R2": float(1 - np.sum(residuo ** 2) / np.sum((y - y.mean()) ** 2))}
    if desviacion_entrenamiento:
        salida["MSE_normalizado"] = (rmse / desviacion_entrenamiento) ** 2
    return salida


def tabla_metricas(etiquetas: np.ndarray, predicciones: np.ndarray, nombres_objetivos, desviaciones_entrenamiento,
                   **columnas_fijas) -> pd.DataFrame:
    """Una fila por propiedad, más columnas fijas (modelo, partición, semilla…)."""
    filas = [{**columnas_fijas, "objetivo": nombre, **metricas(etiquetas[:, k], predicciones[:, k], desviaciones_entrenamiento[k])}
             for k, nombre in enumerate(nombres_objetivos)]
    return pd.DataFrame(filas)


def mse_normalizado_medio(etiquetas: np.ndarray, predicciones: np.ndarray, desviaciones_entrenamiento) -> float:
    """Criterio de selección del protocolo común: promedio de (RMSE / σ_entrenamiento)² entre propiedades."""
    e, p = np.asarray(etiquetas, float), np.asarray(predicciones, float)
    rmse = np.sqrt(np.nanmean((p - e) ** 2, axis=0))
    return float(np.mean((rmse / np.asarray(desviaciones_entrenamiento, float)) ** 2))


def intervalo_rmse(etiqueta, prediccion, n_remuestreos: int = 2000, semilla: int = 0) -> tuple[float, float]:
    y, p = np.asarray(etiqueta, float), np.asarray(prediccion, float)
    idx = np.random.default_rng(semilla).integers(0, len(y), (n_remuestreos, len(y)))
    valores = np.sqrt(((p[idx] - y[idx]) ** 2).mean(1))
    return float(np.quantile(valores, 0.025)), float(np.quantile(valores, 0.975))


def diferencia_pareada_rmse(etiqueta, prediccion_a, prediccion_b, n_remuestreos: int = 2000,
                            semilla: int = 0) -> tuple[float, float, float]:
    """RMSE(a) − RMSE(b) sobre las mismas galaxias y su intervalo del 95 % (bootstrap pareado)."""
    y, a, b = (np.asarray(v, float) for v in (etiqueta, prediccion_a, prediccion_b))
    idx = np.random.default_rng(semilla).integers(0, len(y), (n_remuestreos, len(y)))
    d = np.sqrt(((a[idx] - y[idx]) ** 2).mean(1)) - np.sqrt(((b[idx] - y[idx]) ** 2).mean(1))
    punto = float(np.sqrt(np.mean((a - y) ** 2)) - np.sqrt(np.mean((b - y) ** 2)))
    return punto, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def veredicto(limite_inferior: float, limite_superior: float, margen: float) -> str:
    """Regla de decisión del proyecto para una diferencia (negativo = mejor el primero)."""
    if limite_superior < 0:
        return "mejor"
    if limite_inferior > 0:
        return "peor"
    if -margen < limite_inferior and limite_superior < margen:
        return "equivalente"
    return "no concluyente"


def metricas_por_grupo(etiquetas: np.ndarray, predicciones: np.ndarray, grupos, nombres_objetivos,
                       minimo_por_grupo: int = 10) -> pd.DataFrame:
    """RMSE y sesgo de cada propiedad dentro de cada grupo (tipo de galaxia, intervalo de masa…)."""
    grupos = np.asarray(grupos)
    filas = []
    for g in pd.unique(grupos):
        sel = grupos == g
        if sel.sum() < minimo_por_grupo:
            continue
        for k, nombre in enumerate(nombres_objetivos):
            m = metricas(etiquetas[sel, k], predicciones[sel, k])
            filas.append({"grupo": g, "objetivo": nombre, "n": m["n"], "RMSE": m["RMSE"], "sesgo": m["sesgo"]})
    return pd.DataFrame(filas)
