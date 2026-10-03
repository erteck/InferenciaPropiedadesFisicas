"""Variantes de entrada: "solo DESI" y, más adelante, "DESI + preprocesamiento".

Una variante es una función que recibe el cubo COMPLETO de 800×800 px (para que
el preprocesamiento pueda usar el cielo y los vecinos) y devuelve un cubo del
mismo tamaño. Después se recorta el centro y se guarda en una caché propia de la
variante. Todo lo demás (cohorte, particiones, normalización, receta, semilla,
evaluación) es idéntico entre variantes; así la diferencia E2 − E1 mide solo el
efecto del preprocesamiento.

Cómo añadir el preprocesamiento del equipo (cuando esté listo como .py):

    def preprocesamiento_equipo(cubo, meta, **params):
        # cubo: np.ndarray (3, 800, 800) float32, bandas g, r, z, flujo en nanomaggies
        # meta: dict con mangaid, redshift, PETRO_TH90, ... de la cohorte
        ...
        return cubo_limpio, {"estado": "ok"}      # o {"estado": "fallo", "motivo": "..."}

    register(Variant("desi_prep", preprocesamiento_equipo, params={"nsigma": 2.0}, version="1"))

Reglas:
- Si la función lanza una excepción o devuelve estado "fallo" para una galaxia, se
  usa el cubo crudo de esa galaxia y se marca `fallback=True` (la comparación sigue
  siendo pareada). Si fallan más del `max_fallback_frac`, la construcción se detiene.
- `NotImplementedError` nunca se trata como fallo de una galaxia: siempre detiene todo.
- La función debe estar definida a nivel de módulo (se ejecuta en procesos separados).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

PrepFn = Callable[..., tuple[np.ndarray, dict]]


@dataclass(frozen=True)
class Variant:
    """Una forma de preparar la imagen antes del recorte (identidad o preprocesamiento)."""
    name: str
    fn: PrepFn | None = None          # None → identidad (solo DESI, lectura rápida)
    params: dict = field(default_factory=dict)
    version: str = "1"                 # subir al cambiar el código de `fn`
    max_fallback_frac: float = 0.05
    description: str = ""

    @property
    def is_identity(self) -> bool:
        return self.fn is None

    @property
    def key(self) -> str:
        """Nombre de carpeta: nombre + hash de parámetros y versión. Cambiar un
        parámetro genera una caché nueva; nunca se mezclan cachés."""
        if self.is_identity:
            return self.name
        blob = json.dumps({"params": self.params, "version": self.version}, sort_keys=True, default=str)
        return f"{self.name}_{hashlib.sha256(blob.encode()).hexdigest()[:8]}"

    def manifest(self) -> dict:
        return {"name": self.name, "key": self.key, "version": self.version, "params": self.params,
                "fn": None if self.fn is None else f"{self.fn.__module__}.{self.fn.__qualname__}",
                "max_fallback_frac": self.max_fallback_frac, "description": self.description}

    def apply(self, cube: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
        """Aplica la variante con la política de respaldo documentada arriba."""
        if self.is_identity:
            return cube, {"estado": "ok", "fallback": False}
        try:
            out, info = self.fn(cube.copy(), meta, **self.params)
        except NotImplementedError:
            raise
        except Exception as e:  # fallo de esta galaxia → cubo crudo
            return cube, {"estado": "fallo", "fallback": True, "motivo": f"{type(e).__name__}: {e}"[:300]}
        info = dict(info or {})
        out = np.asarray(out, dtype=np.float32)
        if info.get("estado") == "fallo" or out.shape != cube.shape or not np.isfinite(out).all():
            motivo = info.get("motivo") or ("forma distinta" if out.shape != cube.shape else "valores no finitos")
            return cube, {**info, "estado": "fallo", "fallback": True, "motivo": motivo}
        return out, {**info, "estado": "ok", "fallback": False}


DESI = Variant("desi", None, description="Cubos DESI DR10 grz sin preprocesamiento (E1).")

REGISTRY: dict[str, Variant] = {DESI.name: DESI}


def register(v: Variant) -> Variant:
    """Registra una variante para poder seleccionarla por nombre desde el cuaderno."""
    REGISTRY[v.name] = v
    return v


def get(name: str) -> Variant:
    """Devuelve la variante registrada con ese nombre."""
    if name not in REGISTRY:
        raise KeyError(f"Variante desconocida '{name}'. Registradas: {sorted(REGISTRY)}")
    return REGISTRY[name]


# -----------------------------------------------------------------------------
# Lugar para el preprocesamiento del equipo (Bloque 0, Preprocesamiento_v1.1).
# Se registra para que el cuaderno lo reconozca, pero detiene la ejecución hasta
# que se implemente; así nunca se entrena por error con datos "sin preprocesar"
# creyendo que lo estaban.
# -----------------------------------------------------------------------------
def preprocesamiento_equipo(cube: np.ndarray, meta: dict, **params) -> tuple[np.ndarray, dict]:
    raise NotImplementedError(
        "Pegar aquí el preprocesamiento del equipo (cielo, máscara, limpieza de estrellas). "
        "Contrato: recibe (3, 800, 800) g,r,z en nanomaggies y devuelve el mismo tamaño.")


register(Variant("desi_prep", preprocesamiento_equipo, params={}, version="0",
                 description="DESI + preprocesamiento del equipo (pendiente de implementar)."))


# -----------------------------------------------------------------------------
# EJEMPLO del contrato (no es el preprocesamiento del equipo; no usar como resultado).
# Resta a cada banda la mediana del cielo medida en el anillo exterior del campo de
# 800 px. Sirve para probar de punta a punta el camino E2 vs E1.
# -----------------------------------------------------------------------------
def ejemplo_restar_cielo(cube: np.ndarray, meta: dict, r_min_px: float = 300.0) -> tuple[np.ndarray, dict]:
    n = cube.shape[-1]
    yy, xx = np.indices((n, n))
    anillo = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) > r_min_px
    cielo = np.array([np.median(b[anillo & (b != 0)]) for b in cube])
    return cube - cielo[:, None, None] * (cube != 0), {"estado": "ok", "cielo": cielo.tolist()}


register(Variant("ejemplo_cielo", ejemplo_restar_cielo, params={"r_min_px": 300.0}, version="1",
                 description="EJEMPLO: resta la mediana del cielo del anillo exterior. Solo para probar E2."))
