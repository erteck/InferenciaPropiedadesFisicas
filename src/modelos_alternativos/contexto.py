"""Contexto de datos compartido por los cuadernos de modelos alternativos.

`preparar_contexto` hace, una sola vez y con las galaxias de entrenamiento, todo lo que los
modelos necesitan: muestra, imágenes, transformación GRZ, estandarizadores y los tres conjuntos en
la GPU. Las propiedades a predecir y las variables escalares auxiliares son parámetros.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.katachi_fits import catalog as catalogo
from src.katachi_fits import config as config_katachi
from src.katachi_fits.cache import build_cache, cache_dir, load_cache
from src.katachi_fits.config import FitsConfig, Paths
from src.katachi_fits.io_guard import WriteGuard
from src.katachi_fits.variants import DESI

from .datos import (OBJETIVOS, PARTICIONES, ConjuntoEnGPU, EstandarizadorEscalar, TransformacionGRZ, filtrar_muestra,
                    indices_por_particion)


@dataclass
class ContextoDatos:
    rutas: Paths
    guardia: WriteGuard
    muestra: pd.DataFrame
    excluidas: pd.DataFrame
    imagenes: np.ndarray                         # (N, 3, 300, 300) float16, orden g, r, z
    indices: dict
    objetivos: tuple
    variables_escalares: tuple
    transformacion_grz: TransformacionGRZ
    estandarizador_etiquetas: EstandarizadorEscalar
    estandarizador_escalares: EstandarizadorEscalar | None
    conjuntos: dict = field(default_factory=dict)  # partición → ConjuntoEnGPU

    @property
    def desviacion_etiquetas_entrenamiento(self) -> np.ndarray:
        return np.asarray(self.estandarizador_etiquetas.desviacion, float)

    def etiquetas(self, particion: str) -> np.ndarray:
        return self.muestra.iloc[self.indices[particion]][list(self.objetivos)].to_numpy(float)

    def a_escala_original(self, prediccion_estandarizada: np.ndarray) -> np.ndarray:
        return self.estandarizador_etiquetas.invertir(prediccion_estandarizada)

    def conjunto(self, particion: str, dispositivo) -> ConjuntoEnGPU:
        """Crea (o reutiliza) el conjunto de una partición en el dispositivo."""
        if particion not in self.conjuntos:
            idx = self.indices[particion]
            escalares = None
            if self.variables_escalares:
                escalares = self.estandarizador_escalares.transformar(
                    self.muestra.iloc[idx][list(self.variables_escalares)].to_numpy(float))
            etiquetas = self.estandarizador_etiquetas.transformar(self.etiquetas(particion))
            self.conjuntos[particion] = ConjuntoEnGPU(self.imagenes[idx], escalares, etiquetas, dispositivo)
        return self.conjuntos[particion]

    def resumen(self) -> dict:
        return {"galaxias": len(self.muestra), **{p: int(len(i)) for p, i in self.indices.items()},
                "excluidas": int(len(self.excluidas)), "objetivos": list(self.objetivos),
                "variables_escalares": list(self.variables_escalares),
                "transformacion_grz": self.transformacion_grz.a_dict(),
                "estandarizador_etiquetas": self.estandarizador_etiquetas.a_dict(),
                "estandarizador_escalares": None if self.estandarizador_escalares is None
                else self.estandarizador_escalares.a_dict()}


def preparar_contexto(base_drive: Path, carpeta_resultados: str, objetivos=OBJETIVOS, variables_escalares=("redshift",),
                      limite_galaxias: int = 0, registro=print) -> ContextoDatos:
    """Lee el catálogo y la caché de imágenes de Drive y ajusta las transformaciones con entrenamiento.

    Escribe solo dentro de `base_drive / carpeta_resultados`; el catálogo y los FITS son de solo lectura.
    """
    base_drive = Path(base_drive)
    descarga = base_drive / config_katachi.DOWNLOAD_DIRNAME
    plan_csv = next(descarga.glob("**/plan.csv"))
    catalogo_equipo = next(p for n in config_katachi.TEAM_CATALOG_NAMES for p in descarga.glob(f"**/{n}"))
    rutas = Paths(base=base_drive, plan_csv=plan_csv, inventario_csv=plan_csv.parent / "inventario.csv",
                  fits_dir=descarga / "fits", results=base_drive / config_katachi.RESULTS_DIRNAME,
                  team_catalog=catalogo_equipo)
    guardia = WriteGuard(base_drive / carpeta_resultados, rutas.read_only)
    guardia.mkdir(base_drive / carpeta_resultados)

    plan = catalogo.load_plan(plan_csv, rutas.fits_dir)
    equipo = catalogo.load_team_catalog(catalogo_equipo)
    scalars = catalogo.load_scalars(rutas.external / "scalars.cat")
    cohorte, _ = catalogo.build_cohort_team(plan, equipo, scalars)
    if limite_galaxias:            # solo para pruebas locales reducidas
        cohorte = (cohorte.groupby("particion", group_keys=False)
                   .apply(lambda d: d.head(max(limite_galaxias // 3, 12) if d.name != "Train" else limite_galaxias))
                   .reset_index(drop=True))
    cohorte, _ = catalogo.apply_qc(cohorte, config_katachi.QC_GROUPS_INCLUDED)
    cohorte = cohorte.reset_index(drop=True)

    configuracion_fits = FitsConfig()
    etiqueta_cache = f"_lim{limite_galaxias}" if limite_galaxias else ""
    carpeta_cache = cache_dir(rutas.cache, DESI, configuracion_fits, etiqueta_cache)
    if not any(carpeta_cache.glob("chunk_*.npz")):
        guardia_cache = WriteGuard(rutas.results, rutas.read_only)
        build_cache(cohorte, "ruta_local", carpeta_cache, guardia_cache, configuracion_fits, variant=DESI, chunk=500, workers=24)
    imagenes, ids, fraccion_ceros, _ = load_cache(carpeta_cache, dtype=configuracion_fits.storage_dtype)
    if list(ids) != list(cohorte.mangaid):
        raise ValueError("El orden de la caché de imágenes no coincide con la muestra")

    muestra, excluidas = filtrar_muestra(cohorte, fraccion_ceros)
    imagenes = imagenes[muestra.index.to_numpy()]
    muestra = muestra.reset_index(drop=True)
    muestra["log_ssfr"] = muestra["log_sfr"] - muestra["log_mstar"]
    indices = indices_por_particion(muestra)
    registro(f"Muestra: {len(muestra)} galaxias ({', '.join(f'{p} {len(indices[p])}' for p in PARTICIONES)}) · "
             f"excluidas: {len(excluidas)}")

    entrenamiento = indices["Train"]
    transformacion = TransformacionGRZ.ajustar(imagenes, entrenamiento)
    estandarizador_etiquetas = EstandarizadorEscalar.ajustar(muestra.iloc[entrenamiento][list(objetivos)].to_numpy(float))
    estandarizador_escalares = (EstandarizadorEscalar.ajustar(muestra.iloc[entrenamiento][list(variables_escalares)].to_numpy(float))
                                if variables_escalares else None)
    return ContextoDatos(rutas, guardia, muestra, excluidas, imagenes, indices, tuple(objetivos), tuple(variables_escalares),
                         transformacion, estandarizador_etiquetas, estandarizador_escalares)
