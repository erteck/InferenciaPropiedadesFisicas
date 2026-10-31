"""Predicciones de los modelos base en un formato común, para combinarlas sin volver a usar las redes.

Cada red exporta un archivo `pred_<familia>_s<semilla>.csv` con las galaxias de validación y de prueba:

    mangaid, particion, redshift, y_<propiedad>…, pred_<propiedad>…, familia, semilla, tta

Los archivos son pequeños (unos kilobytes), así que el cuaderno de ensambles se ejecuta sin GPU, sin
Drive y sin los pesos de las redes. Las galaxias se alinean siempre por `mangaid`, nunca por posición.
Junto a las predicciones se guardan dos archivos más: `tiempos_modelos.csv` (costo de cada red) y
`escala_train.json` (desviación de cada propiedad en entrenamiento, que define la métrica principal).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OBJETIVOS = ("log_mstar", "log_sfr", "d4000")
PARTICIONES_EXPORTADAS = ("Validation", "Test")


def nombre_archivo(familia: str, semilla: int) -> str:
    return f"pred_{familia}_s{semilla}.csv"


def tabla_predicciones(muestra: pd.DataFrame, indices: dict, familia: str, semilla: int, predicciones_por_particion: dict,
                       tta: bool, objetivos=OBJETIVOS) -> pd.DataFrame:
    """Tabla con las predicciones (escala original) de una red; `indices[partición]` ordena las filas de `muestra`."""
    partes = []
    for particion, prediccion in predicciones_por_particion.items():
        filas = muestra.iloc[indices[particion]]
        prediccion = np.asarray(prediccion, float)
        if prediccion.shape != (len(filas), len(objetivos)):
            raise ValueError(f"{familia} s{semilla} {particion}: forma {prediccion.shape}, se esperaba {(len(filas), len(objetivos))}")
        tabla = pd.DataFrame({"mangaid": filas.mangaid.to_numpy(), "particion": particion, "redshift": filas.redshift.to_numpy(float),
                              **{f"y_{t}": filas[t].to_numpy(float) for t in objetivos},
                              **{f"pred_{t}": prediccion[:, k] for k, t in enumerate(objetivos)}})
        partes.append(tabla)
    salida = pd.concat(partes, ignore_index=True)
    salida["familia"], salida["semilla"], salida["tta"] = familia, int(semilla), bool(tta)
    return salida


def guardar_tabla(tabla: pd.DataFrame, carpeta: Path, familia: str, semilla: int, escribir=None) -> Path:
    """Escribe el CSV de una red. `escribir(ruta, función)` permite usar una puerta de escritura protegida."""
    ruta = Path(carpeta) / nombre_archivo(familia, semilla)
    if escribir is None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(ruta, index=False)
    else:
        escribir(ruta, lambda f: tabla.to_csv(f, index=False))
    return ruta


def leer_carpeta(carpeta: Path) -> pd.DataFrame:
    """Une todos los `pred_*.csv` de una carpeta."""
    archivos = sorted(Path(carpeta).glob("pred_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos pred_*.csv en {carpeta}")
    return pd.concat([pd.read_csv(a, dtype={"mangaid": str}) for a in archivos], ignore_index=True)


def leer_escala(carpeta: Path) -> np.ndarray:
    datos = json.loads((Path(carpeta) / "escala_train.json").read_text())
    return np.array([datos[t] for t in OBJETIVOS], float)


@dataclass
class ConjuntoPredicciones:
    """Predicciones alineadas por galaxia de varias familias de modelos sobre una partición."""
    particion: str
    ids: np.ndarray                  # (n,)
    y: np.ndarray                    # (n, 3)
    redshift: np.ndarray             # (n,)
    pred: dict                       # familia → (n_semillas, n, 3)
    semillas: dict                   # familia → lista de semillas, en el orden de `pred`
    objetivos: tuple = OBJETIVOS

    @property
    def familias(self) -> list:
        return list(self.pred)

    def media(self, familia: str) -> np.ndarray:
        """Promedio de las semillas de una familia (ensamble homogéneo con pesos iguales): (n, 3)."""
        return self.pred[familia].mean(0)

    def medias(self, familias=None) -> np.ndarray:
        """Matriz (familias, n, 3) con el promedio de semillas de cada familia."""
        return np.stack([self.media(f) for f in (familias or self.familias)])

    def desacuerdo_semillas(self, familia: str) -> np.ndarray:
        """Desviación entre las semillas de una familia, por galaxia y propiedad: (n, 3)."""
        return self.pred[familia].std(0, ddof=1) if len(self.semillas[familia]) > 1 else np.zeros_like(self.y)

    def subconjunto(self, familias) -> "ConjuntoPredicciones":
        return ConjuntoPredicciones(self.particion, self.ids, self.y, self.redshift, {f: self.pred[f] for f in familias},
                                    {f: self.semillas[f] for f in familias}, self.objetivos)


def construir_conjunto(tabla: pd.DataFrame, particion: str, familias=None, objetivos=OBJETIVOS, tolerancia_etiquetas: float = 1e-6,
                       minimo_semillas: int = 1, registro=print) -> ConjuntoPredicciones:
    """Alinea por `mangaid` las predicciones de las familias pedidas en una partición.

    Se queda con las galaxias que tienen predicción en todas las redes y comprueba que todas las
    tablas traen las mismas etiquetas y el mismo redshift para cada galaxia.
    """
    t = tabla[tabla.particion == particion]
    familias = list(familias) if familias else sorted(t.familia.unique())
    faltan = [f for f in familias if f not in set(t.familia)]
    if faltan:
        raise ValueError(f"No hay predicciones de {particion} para: {faltan}")
    por_red = {(f, int(s)): g.set_index("mangaid") for (f, s), g in t[t.familia.isin(familias)].groupby(["familia", "semilla"])}
    ids = None
    for g in por_red.values():
        ids = g.index if ids is None else ids.intersection(g.index)
    descartadas = max(len(g) for g in por_red.values()) - len(ids)
    if descartadas:
        registro(f"{particion}: {descartadas} galaxias sin predicción en alguna red; se usan {len(ids)}")
    ids = np.array(sorted(ids))
    referencia = next(iter(por_red.values())).loc[ids]
    y = referencia[[f"y_{o}" for o in objetivos]].to_numpy(float)
    redshift = referencia["redshift"].to_numpy(float)
    pred, semillas = {}, {}
    for f in familias:
        semillas_f = sorted(s for (ff, s) in por_red if ff == f)
        if len(semillas_f) < minimo_semillas:
            raise ValueError(f"{f}: {len(semillas_f)} semillas, se piden al menos {minimo_semillas}")
        matrices = []
        for s in semillas_f:
            g = por_red[(f, s)].loc[ids]
            if np.abs(g[[f"y_{o}" for o in objetivos]].to_numpy(float) - y).max() > tolerancia_etiquetas:
                raise ValueError(f"{f} s{s}: las etiquetas no coinciden con las de las demás redes")
            if np.abs(g["redshift"].to_numpy(float) - redshift).max() > tolerancia_etiquetas:
                raise ValueError(f"{f} s{s}: el redshift no coincide con el de las demás redes")
            matrices.append(g[[f"pred_{o}" for o in objetivos]].to_numpy(float))
        pred[f], semillas[f] = np.stack(matrices), semillas_f
    return ConjuntoPredicciones(particion, ids, y, redshift, pred, semillas, tuple(objetivos))
