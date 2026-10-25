"""Contexto de datos compartido por los cuadernos de modelos alternativos.

`preparar_contexto` hace, una sola vez y con las galaxias de entrenamiento, todo lo que los
modelos necesitan: muestra, imágenes, transformación GRZ y estandarización de las etiquetas.

Las variables escalares que acompañan a la imagen no se fijan aquí: cada corrida elige las suyas
(por ejemplo, ninguna, o un redshift predicho desde la imagen). Cada columna escalar tiene su propio
estandarizador ajustado con entrenamiento, y las columnas nuevas se agregan con `agregar_columna`.
Las imágenes de cada partición se suben a la GPU una sola vez y las comparten todas las corridas.
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
    rutas: Paths | None
    guardia: WriteGuard
    muestra: pd.DataFrame
    excluidas: pd.DataFrame
    imagenes: np.ndarray                         # (N, 3, 300, 300) float16, orden g, r, z
    indices: dict
    objetivos: tuple
    transformacion_grz: TransformacionGRZ
    estandarizador_etiquetas: EstandarizadorEscalar
    estandarizadores_escalares: dict = field(default_factory=dict)   # columna → EstandarizadorEscalar
    _imagenes_gpu: dict = field(default_factory=dict, repr=False)    # (partición, dispositivo) → tensor
    _conjuntos: dict = field(default_factory=dict, repr=False)       # (partición, dispositivo, variables) → ConjuntoEnGPU

    @property
    def desviacion_etiquetas_entrenamiento(self) -> np.ndarray:
        return np.asarray(self.estandarizador_etiquetas.desviacion, float)

    def etiquetas(self, particion: str) -> np.ndarray:
        return self.muestra.iloc[self.indices[particion]][list(self.objetivos)].to_numpy(float)

    def a_escala_original(self, prediccion_estandarizada: np.ndarray) -> np.ndarray:
        return self.estandarizador_etiquetas.invertir(prediccion_estandarizada)

    # ------------------------------------------------------------------ variables escalares
    def estandarizador_de(self, columna: str) -> EstandarizadorEscalar:
        """Estandarizador de una columna, ajustado con las galaxias de entrenamiento (se crea la primera vez)."""
        if columna not in self.estandarizadores_escalares:
            valores = self.muestra.iloc[self.indices["Train"]][columna].to_numpy(float)
            if not np.isfinite(valores).all():
                raise ValueError(f"La columna {columna} tiene valores no finitos en entrenamiento")
            self.estandarizadores_escalares[columna] = EstandarizadorEscalar.ajustar(valores)
        return self.estandarizadores_escalares[columna]

    def escalares_estandarizados(self, particion: str, variables) -> np.ndarray:
        idx = self.indices[particion]
        if not variables:
            return np.zeros((len(idx), 0), np.float32)
        columnas = [self.estandarizador_de(v).transformar(self.muestra.iloc[idx][v].to_numpy(float)) for v in variables]
        return np.concatenate(columnas, 1).astype(np.float32)

    def agregar_columna(self, nombre: str, valores_por_particion: dict) -> None:
        """Agrega una columna a la muestra (p. ej., una variable predicha desde la imagen).

        `valores_por_particion` = {partición: arreglo en el orden de `indices[partición]`}. Si la
        columna ya existía, se reemplaza y su estandarizador se vuelve a ajustar con entrenamiento.
        """
        columna = np.full(len(self.muestra), np.nan)
        for particion, valores in valores_por_particion.items():
            valores = np.asarray(valores, float).ravel()
            if len(valores) != len(self.indices[particion]):
                raise ValueError(f"{nombre}: {len(valores)} valores para {len(self.indices[particion])} galaxias de {particion}")
            columna[self.indices[particion]] = valores
        self.muestra[nombre] = columna
        self.estandarizadores_escalares.pop(nombre, None)
        self._conjuntos = {k: v for k, v in self._conjuntos.items() if nombre not in k[2]}

    # ------------------------------------------------------------------ conjuntos en la GPU
    def conjunto(self, particion: str, dispositivo, variables_escalares=()) -> ConjuntoEnGPU:
        """Conjunto de una partición con las variables escalares pedidas; las imágenes se comparten."""
        variables = tuple(variables_escalares)
        clave = (particion, str(dispositivo), variables)
        if clave not in self._conjuntos:
            clave_imagenes = (particion, str(dispositivo))
            if clave_imagenes not in self._imagenes_gpu:
                self._imagenes_gpu[clave_imagenes] = torch.from_numpy(
                    np.ascontiguousarray(self.imagenes[self.indices[particion]])).to(dispositivo)
            etiquetas = self.estandarizador_etiquetas.transformar(self.etiquetas(particion))
            self._conjuntos[clave] = ConjuntoEnGPU(self._imagenes_gpu[clave_imagenes],
                                                   self.escalares_estandarizados(particion, variables), etiquetas, dispositivo)
        return self._conjuntos[clave]

    def resumen(self) -> dict:
        return {"galaxias": len(self.muestra), **{p: int(len(i)) for p, i in self.indices.items()},
                "excluidas": int(len(self.excluidas)), "objetivos": list(self.objetivos),
                "transformacion_grz": self.transformacion_grz.a_dict(),
                "estandarizador_etiquetas": self.estandarizador_etiquetas.a_dict(),
                "estandarizadores_escalares": {k: v.a_dict() for k, v in self.estandarizadores_escalares.items()}}


def preparar_contexto(base_drive: Path, carpeta_resultados: str, objetivos=OBJETIVOS, limite_galaxias: int = 0,
                      registro=print) -> ContextoDatos:
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
    return ContextoDatos(rutas, guardia, muestra, excluidas, imagenes, indices, tuple(objetivos), transformacion,
                         estandarizador_etiquetas)
