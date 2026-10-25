"""Búsqueda de hiperparámetros genérica con Optuna (Akiba et al. 2019).

- Muestreo bayesiano TPE con semilla fija.
- Poda por *successive halving* asíncrono (ASHA): las pruebas que van peor que la mayoría en la
  misma época se detienen antes, y el presupuesto se concentra en las prometedoras.
- El estudio se guarda en SQLite: si la sesión se desconecta, la búsqueda continúa donde quedó.

El cuaderno define dos funciones: `proponer(trial) -> dict` (los valores de una prueba) y
`evaluar(valores, al_terminar_epoca) -> float` (entrena y devuelve el criterio de validación).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def ejecutar_busqueda(nombre_estudio: str, ruta_respaldo: Path, proponer: Callable, evaluar: Callable,
                      n_pruebas: int, semilla: int = 42, epocas_minimas_antes_de_podar: int = 3,
                      factor_reduccion: int = 3, guardar_respaldo: Callable[[Path, Path], None] | None = None,
                      registro=print):
    """Ejecuta (o reanuda) un estudio hasta completar `n_pruebas` pruebas terminadas o podadas.

    SQLite no es confiable sobre Google Drive, así que el estudio vive en un archivo local y, al
    terminar cada prueba, se copia a `ruta_respaldo` con `guardar_respaldo(origen, destino)`. Al
    empezar, si existe el respaldo, se recupera.
    """
    import shutil
    import tempfile
    import uuid

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ruta_local = Path(tempfile.gettempdir()) / f"{nombre_estudio}_{uuid.uuid4().hex[:8]}.db"
    ruta_respaldo = Path(ruta_respaldo)
    # El respaldo en Drive manda: se recupera si existe; si no, se empieza un estudio nuevo.
    if ruta_respaldo.exists():
        shutil.copy2(ruta_respaldo, ruta_local)
    respaldar = guardar_respaldo or (lambda origen, destino: shutil.copy2(origen, destino))
    estudio = optuna.create_study(
        study_name=nombre_estudio, storage=f"sqlite:///{ruta_local}", load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=semilla, multivariate=True),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=epocas_minimas_antes_de_podar,
                                                      reduction_factor=factor_reduccion))
    hechas = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in estudio.trials)

    def objetivo(trial):
        valores = proponer(trial)

        def al_terminar_epoca(epoca, criterio):
            trial.report(criterio, epoca)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return evaluar(valores, al_terminar_epoca)

    pendientes = max(0, n_pruebas - hechas)
    registro(f"Estudio {nombre_estudio}: {hechas} pruebas hechas, {pendientes} por hacer")
    def al_terminar_prueba(estudio_, prueba):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()           # libera la memoria de la prueba anterior, también si se podó
        except ImportError:
            pass
        if prueba.state == optuna.trial.TrialState.FAIL:
            registro(f"Prueba {prueba.number} falló: {prueba.system_attrs.get('fail_reason', 'sin detalle')[:300]}")
        respaldar(ruta_local, ruta_respaldo)

    if pendientes:
        estudio.optimize(objetivo, n_trials=pendientes, gc_after_trial=True,
                         catch=(RuntimeError,),   # una prueba que se queda sin memoria no detiene la búsqueda
                         callbacks=[al_terminar_prueba])
    respaldar(ruta_local, ruta_respaldo)
    estados = [t.state for t in estudio.trials]
    if optuna.trial.TrialState.COMPLETE not in estados:
        fallidas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.FAIL]
        detalle = fallidas[-1].system_attrs.get("fail_reason", "") if fallidas else ""
        raise RuntimeError(f"Ninguna prueba del estudio {nombre_estudio} terminó. Último error: {detalle[:500]}")
    return estudio


def tabla_pruebas(estudio) -> pd.DataFrame:
    """Una fila por prueba: estado, criterio de validación, épocas usadas y valores probados."""
    filas = []
    for t in estudio.trials:
        filas.append({"prueba": t.number, "estado": t.state.name.lower(), "criterio_validacion": t.value,
                      "epocas_usadas": (max(t.intermediate_values) if t.intermediate_values else 0),
                      "minutos": (t.duration.total_seconds() / 60 if t.duration else float("nan")), **t.params})
    return pd.DataFrame(filas).sort_values("criterio_validacion", na_position="last").reset_index(drop=True)


def importancia_hiperparametros(estudio) -> pd.Series:
    """Importancia de cada hiperparámetro (fANOVA de Optuna) sobre las pruebas terminadas."""
    import optuna
    try:
        return pd.Series(optuna.importance.get_param_importances(estudio)).sort_values(ascending=False)
    except (RuntimeError, ValueError):            # muy pocas pruebas terminadas
        return pd.Series(dtype=float)
