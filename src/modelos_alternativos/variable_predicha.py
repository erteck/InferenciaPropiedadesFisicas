"""Etapa 1 de un modelo en dos etapas: predecir una variable del catálogo a partir de la imagen.

En la práctica solo se dispone de la imagen. Una variable que no se observa al predecir (como el
redshift) no puede ser una entrada del modelo; sí se puede **estimar desde la imagen** y usar esa
estimación como entrada de una segunda etapa. Para que la segunda etapa aprenda con estimaciones
del mismo nivel de error que tendrá con galaxias nuevas:

- **Entrenamiento: predicciones fuera de partición** (*cross-fitting*; Wolpert 1992). Las galaxias de
  entrenamiento se dividen en K partes; cada parte se predice con un modelo entrenado con las otras
  K − 1. Ninguna galaxia recibe una estimación de un modelo que la vio.
- **Validación y prueba:** un modelo entrenado con todas las galaxias de entrenamiento.
- **Selección de la época:** con una fracción interna de las galaxias de entrenamiento de cada
  modelo, nunca con la parte que se predice ni con validación.
- Solo se usan las imágenes y la variable de entrenamiento; la variable real de validación y
  prueba no interviene en ningún ajuste.

El modelo de la etapa 1 es un extractor congelado más una cabeza (rápido: K + 1 cabezas en minutos).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .arquitecturas import EspecificacionModelo
from .contexto import ContextoDatos
from .datos import EstandarizadorEscalar
from .experimento import es_extractor_de_tokens, representaciones_en_cache, seleccionar_agregacion
from .representaciones import ConfiguracionCabeza, entrenar_cabeza

TRANSFORMACIONES = {
    "identidad": (lambda v: v, lambda v: v),
    "log10": (np.log10, lambda v: 10.0 ** v),
}


def _ajustar_y_predecir(r_ajuste, y_ajuste, r_predecir, config: ConfiguracionCabeza, fraccion_interna: float, semilla: int,
                        dispositivo) -> tuple[np.ndarray, int]:
    """Entrena una cabeza con (r_ajuste, y_ajuste), eligiendo la época con una fracción interna, y predice r_predecir."""
    generador = np.random.default_rng(semilla)
    orden = generador.permutation(len(r_ajuste))
    n_interna = max(1, int(round(fraccion_interna * len(r_ajuste))))
    interna, ajuste = orden[:n_interna], orden[n_interna:]
    escala = EstandarizadorEscalar.ajustar(y_ajuste[ajuste])
    vacio = lambda n: np.zeros((n, 0), np.float32)
    resultado = entrenar_cabeza(r_ajuste[ajuste], vacio(len(ajuste)), escala.transformar(y_ajuste[ajuste]),
                                r_ajuste[interna], vacio(len(interna)), escala.transformar(y_ajuste[interna]),
                                replace(config, semilla=semilla), dispositivo)
    prediccion = escala.invertir(resultado.predecir(r_predecir, vacio(len(r_predecir)))).ravel()
    return prediccion, resultado.mejor_epoca


def metricas_redshift(z_real, z_predicho) -> dict:
    """Métricas habituales de redshift fotométrico: Δz/(1+z), su dispersión robusta y la fracción de atípicos."""
    z_real, z_predicho = np.asarray(z_real, float), np.asarray(z_predicho, float)
    dz = (z_predicho - z_real) / (1 + z_real)
    mediana = np.median(dz)
    nmad = 1.4826 * np.median(np.abs(dz - mediana))
    relativo = np.abs(z_predicho - z_real) / z_real
    return {"n": int(len(z_real)), "sesgo_dz": float(mediana), "sigma_NMAD_dz": float(nmad),
            "atipicos_dz_mayor_0.05": float(np.mean(np.abs(dz) > 0.05)),
            "error_relativo_mediano": float(np.median(relativo)),
            "RMSE_log10z": float(np.sqrt(np.mean((np.log10(z_predicho) - np.log10(z_real)) ** 2))),
            "spearman": float(pd.Series(z_real).corr(pd.Series(z_predicho), method="spearman"))}


def predecir_variable_desde_imagen(contexto: ContextoDatos, carpeta: Path, columna_objetivo: str, nombre_columna: str,
                                   espec: EspecificacionModelo, entrada: dict, config_cabeza: ConfiguracionCabeza, dispositivo,
                                   n_particiones: int = 5, fraccion_interna: float = 0.1, transformacion: str = "log10",
                                   semilla: int = 42, registro=print) -> pd.DataFrame:
    """Estima `columna_objetivo` desde la imagen y la agrega a la muestra como `nombre_columna`.

    Devuelve una tabla por galaxia con la predicción y su origen (fuera de partición o modelo
    completo). El resultado se guarda en `carpeta` y se reutiliza si ya existe con la misma
    configuración, de modo que todos los cuadernos usan exactamente la misma estimación.
    """
    if transformacion not in TRANSFORMACIONES:
        raise ValueError(f"transformacion debe ser una de {tuple(TRANSFORMACIONES)}")
    directa, inversa = TRANSFORMACIONES[transformacion]
    descripcion = {"columna_objetivo": columna_objetivo, "espec": espec.a_dict(), "entrada": entrada, "cabeza": config_cabeza.a_dict(),
                   "n_particiones": n_particiones, "fraccion_interna": fraccion_interna, "transformacion": transformacion,
                   "semilla": semilla, "galaxias": int(len(contexto.muestra))}
    carpeta = Path(carpeta)
    ruta_tabla, ruta_descripcion = carpeta / f"{nombre_columna}.csv", carpeta / f"{nombre_columna}_configuracion.json"
    if ruta_tabla.exists() and ruta_descripcion.exists() and json.loads(ruta_descripcion.read_text()) == json.loads(json.dumps(descripcion)):
        tabla = pd.read_csv(ruta_tabla, dtype={"mangaid": str}).set_index("mangaid")
        registro(f"[{nombre_columna}] se reutiliza la estimación guardada")
    else:
        reps = representaciones_en_cache(contexto, carpeta, espec, entrada, ("Train", "Validation", "Test"), dispositivo, registro=registro)
        tokens = es_extractor_de_tokens(espec.nombre_extractor)
        r = {p: seleccionar_agregacion(reps[p], espec.agregacion_tokens, tokens) for p in ("Train", "Validation", "Test")}
        y_entrenamiento = directa(contexto.muestra.iloc[contexto.indices["Train"]][columna_objetivo].to_numpy(float))
        n = len(y_entrenamiento)
        pliegue = np.random.default_rng(semilla).permutation(np.arange(n) % n_particiones)
        fuera_de_particion = np.full(n, np.nan)
        for k in range(n_particiones):
            prediccion, epoca = _ajustar_y_predecir(r["Train"][pliegue != k], y_entrenamiento[pliegue != k], r["Train"][pliegue == k],
                                                    config_cabeza, fraccion_interna, semilla + k, dispositivo)
            fuera_de_particion[pliegue == k] = prediccion
            registro(f"[{nombre_columna}] partición {k + 1}/{n_particiones}: mejor época {epoca}")
        completo, epoca = _ajustar_y_predecir(r["Train"], y_entrenamiento, np.concatenate([r["Validation"], r["Test"]]),
                                              config_cabeza, fraccion_interna, semilla + n_particiones, dispositivo)
        n_val = len(r["Validation"])
        filas = []
        for particion, valores, origen in (("Train", fuera_de_particion, "fuera de partición"),
                                           ("Validation", completo[:n_val], "modelo con todo entrenamiento"),
                                           ("Test", completo[n_val:], "modelo con todo entrenamiento")):
            ids = contexto.muestra.mangaid.iloc[contexto.indices[particion]].to_numpy()
            filas.append(pd.DataFrame({"mangaid": ids, "particion": particion, nombre_columna: inversa(valores), "origen": origen,
                                       **({"pliegue": pliegue} if particion == "Train" else {})}))
        tabla = pd.concat(filas).set_index("mangaid")
        contexto.guardia.mkdir(carpeta)
        contexto.guardia.write(ruta_tabla, lambda f: tabla.reset_index().to_csv(f, index=False), allow_replace=True)
        contexto.guardia.write_json(ruta_descripcion, descripcion, allow_replace=True)
    contexto.agregar_columna(nombre_columna, {p: tabla.loc[contexto.muestra.mangaid.iloc[contexto.indices[p]], nombre_columna].to_numpy(float)
                                              for p in ("Train", "Validation", "Test")})
    return tabla
