"""Experimentos de alto nivel, independientes del modelo: corridas reutilizables y registro antes de la prueba.

Un cuaderno de modelo solo declara qué quiere entrenar (`EspecificacionModelo`,
`ConfiguracionEntrenamiento` y la representación de entrada) y llama a `ejecutar_corrida`. Si la
corrida ya terminó en una sesión anterior, se recupera de Drive en lugar de entrenarse de nuevo.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .arquitecturas import EspecificacionModelo, construir_modelo, contar_parametros
from .contexto import ContextoDatos
from .datos import PreparadorEntrada
from .entrenamiento import (ConfiguracionEntrenamiento, cargar_corrida, corrida_terminada, entrenar, guardar_corrida,
                            predecir)
from .evaluacion import tabla_metricas


def crear_preparador(contexto: ContextoDatos, entrada: dict, media_rgb=None, desviacion_rgb=None) -> PreparadorEntrada:
    """`entrada` = {"representacion": ..., "lado": ..., "modo_lado": ...}."""
    return PreparadorEntrada(contexto.transformacion_grz, entrada.get("representacion", "grz_estandarizado"),
                             entrada.get("lado", 300), entrada.get("modo_lado", "recorte"), media_rgb, desviacion_rgb)


def normalizacion_rgb_del_extractor(nombre_extractor: str) -> tuple:
    import timm
    cfg = timm.models.get_pretrained_cfg(nombre_extractor)
    return tuple(cfg.mean), tuple(cfg.std)


def identificador_corrida(prefijo: str, espec: EspecificacionModelo, config: ConfiguracionEntrenamiento, entrada: dict) -> str:
    """Nombre legible más una huella corta de toda la configuración: dos configuraciones distintas no comparten carpeta."""
    contenido = json.dumps({"espec": espec.a_dict(), "config": config.a_dict(), "entrada": entrada}, sort_keys=True)
    return f"{prefijo}_semilla{config.semilla}_{hashlib.sha1(contenido.encode()).hexdigest()[:8]}"


@dataclass
class ResultadoCorrida:
    identificador: str
    carpeta: Path
    modelo: torch.nn.Module
    preparador: PreparadorEntrada
    resumen: dict
    historial: pd.DataFrame
    reutilizada: bool


def ejecutar_corrida(contexto: ContextoDatos, carpeta_modelo: Path, prefijo: str, espec: EspecificacionModelo,
                     config: ConfiguracionEntrenamiento, entrada: dict, dispositivo, al_terminar_epoca=None,
                     guardar: bool = True, registro=print) -> ResultadoCorrida:
    """Entrena una corrida o la recupera si ya terminó. Guarda pesos, historial y resumen en Drive."""
    media_rgb, desviacion_rgb = (normalizacion_rgb_del_extractor(espec.nombre_extractor)
                                 if entrada.get("representacion") == "rgb_lupton" else (None, None))
    preparador = crear_preparador(contexto, entrada, media_rgb, desviacion_rgb)
    n_escalares, n_salidas = len(contexto.variables_escalares), len(contexto.objetivos)
    identificador = identificador_corrida(prefijo, espec, config, entrada)
    carpeta = Path(carpeta_modelo) / "corridas" / identificador
    if guardar and corrida_terminada(carpeta):
        modelo, resumen, historial = cargar_corrida(carpeta, construir_modelo(espec, n_escalares, n_salidas))
        registro(f"[{identificador}] ya terminada · se recupera (mejor época {resumen['mejor_epoca']})")
        return ResultadoCorrida(identificador, carpeta, modelo.to(dispositivo), preparador, resumen, historial, True)
    modelo = construir_modelo(espec, n_escalares, n_salidas)
    parametros = contar_parametros(modelo)
    registro(f"[{identificador}] {parametros['parametros_entrenables'] / 1e6:.1f} M de parámetros entrenables "
             f"de {parametros['parametros_totales'] / 1e6:.1f} M")
    resultado = entrenar(modelo, preparador, contexto.conjunto("Train", dispositivo), contexto.conjunto("Validation", dispositivo),
                         config, al_terminar_epoca=al_terminar_epoca, registro=registro)
    resumen = {"identificador": identificador, "especificacion": espec.a_dict(), "configuracion": config.a_dict(),
               "entrada": entrada, **parametros, "mejor_epoca": resultado.mejor_epoca,
               "mejor_criterio_validacion": resultado.mejor_criterio_validacion,
               "minutos_entrenamiento": resultado.segundos / 60, "memoria_maxima_gb": resultado.memoria_maxima_gb,
               "fecha_utc": datetime.now(timezone.utc).isoformat()}
    if guardar:
        guardar_corrida(contexto.guardia, carpeta, resultado, resumen)
    return ResultadoCorrida(identificador, carpeta, modelo, preparador, resumen, resultado.historial, False)


def predicciones_originales(contexto: ContextoDatos, corrida: ResultadoCorrida, particion: str, dispositivo,
                            tta: bool = False) -> np.ndarray:
    """Predicciones de una partición en la escala original de cada propiedad."""
    canales_ultimos = corrida.resumen.get("configuracion", {}).get("canales_ultimos", False)
    estandarizadas = predecir(corrida.modelo, corrida.preparador, contexto.conjunto(particion, dispositivo), tta=tta,
                              canales_ultimos=canales_ultimos)
    return contexto.a_escala_original(estandarizadas)


def tiempo_inferencia_ms(contexto: ContextoDatos, corrida: ResultadoCorrida, dispositivo, n_imagenes: int = 256) -> float:
    """Milisegundos por galaxia al predecir sin TTA (después de un lote de calentamiento)."""
    conjunto = contexto.conjunto("Validation", dispositivo).subconjunto(n_imagenes)
    canales_ultimos = corrida.resumen.get("configuracion", {}).get("canales_ultimos", False)
    predecir(corrida.modelo, corrida.preparador, conjunto, tamano_lote=64, canales_ultimos=canales_ultimos)   # calentamiento
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    inicio = time.time()
    predecir(corrida.modelo, corrida.preparador, conjunto, tamano_lote=64, canales_ultimos=canales_ultimos)
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - inicio) / len(conjunto)


def tabla_corrida(contexto: ContextoDatos, prediccion: np.ndarray, particion: str, **columnas) -> pd.DataFrame:
    return tabla_metricas(contexto.etiquetas(particion), prediccion, contexto.objetivos,
                          contexto.desviacion_etiquetas_entrenamiento, particion=particion, **columnas)


def guardar_predicciones(contexto: ContextoDatos, carpeta: Path, particion: str, prediccion: np.ndarray, nombre: str) -> Path:
    idx = contexto.indices[particion]
    tabla = pd.DataFrame({"mangaid": contexto.muestra.mangaid.iloc[idx].to_numpy(), "particion": particion,
                          **{f"pred_{t}": prediccion[:, k] for k, t in enumerate(contexto.objetivos)}})
    ruta = Path(carpeta) / f"{nombre}_{particion}.csv"
    contexto.guardia.write(ruta, lambda f: tabla.to_csv(f, index=False), allow_replace=True)
    return ruta


def registrar_eleccion_antes_de_prueba(contexto: ContextoDatos, ruta: Path, eleccion: dict,
                                       predicciones_validacion: np.ndarray, registro=print) -> dict:
    """Guarda la elección (modelo, configuración y huella de sus predicciones de validación) antes de abrir la prueba.

    Si ya existe un registro, lo devuelve y avisa si la elección cambió: en ese caso la evaluación
    en prueba ya no es independiente.
    """
    huella = hashlib.sha256(np.round(np.asarray(predicciones_validacion, float), 6).tobytes()).hexdigest()
    registro_nuevo = {**eleccion, "huella_predicciones_validacion": huella, "fecha_utc": datetime.now(timezone.utc).isoformat()}
    ruta = Path(ruta)
    if ruta.exists():
        previo = json.loads(ruta.read_text())
        registro(f"La prueba se abrió por primera vez el {previo['fecha_utc']}")
        if previo.get("huella_predicciones_validacion") != huella:
            registro("AVISO: el modelo elegido cambió desde la primera apertura; esta evaluación no es una prueba independiente.")
        return previo
    contexto.guardia.write_json(ruta, registro_nuevo)
    registro(f"Elección registrada antes de abrir la prueba: {ruta}")
    return registro_nuevo


# ---------------------------------------------------------------------------
# Modelos con extractor congelado (representaciones fijas + cabeza)
# ---------------------------------------------------------------------------
def representaciones_en_cache(contexto: ContextoDatos, carpeta_modelo: Path, espec: EspecificacionModelo, entrada: dict,
                              particiones, dispositivo, registro=print) -> dict:
    """Vectores del extractor congelado por partición, calculados una vez y guardados en Drive.

    Se guardan con la agregación `cls_y_promedio` cuando el extractor es de tokens, para poder
    elegir después CLS, promedio o ambos sin volver a extraer.
    """
    from .arquitecturas import crear_extractor
    from .representaciones import extraer_representaciones

    espec_extraccion = EspecificacionModelo(**{**espec.a_dict(), "capas_lora": tuple(espec.capas_lora),
                                               "agregacion_tokens": "cls_y_promedio", "estrategia_ajuste": "congelado"})
    nombre = f"{espec.nombre_extractor.replace('/', '_')}_{entrada.get('representacion')}_{entrada.get('lado')}{entrada.get('modo_lado', 'recorte')}"
    carpeta = Path(carpeta_modelo) / "representaciones"
    salida, extractor = {}, None
    for particion in particiones:
        ruta = carpeta / f"{nombre}_{particion}.npy"
        if ruta.exists():
            salida[particion] = np.load(ruta)
            continue
        if extractor is None:
            extractor = crear_extractor(espec_extraccion).to(dispositivo)
            media, desviacion = (normalizacion_rgb_del_extractor(espec.nombre_extractor)
                                 if entrada.get("representacion") == "rgb_lupton" else (None, None))
            preparador = crear_preparador(contexto, entrada, media, desviacion)
        inicio = time.time()
        salida[particion] = extraer_representaciones(extractor, preparador, contexto.conjunto(particion, dispositivo))
        contexto.guardia.mkdir(carpeta)
        contexto.guardia.write(ruta, lambda f, a=salida[particion]: _guardar_npy(f, a), allow_replace=True)
        registro(f"[{nombre}] {particion}: {salida[particion].shape} en {time.time() - inicio:.0f} s")
    del extractor
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()
    return salida


def _guardar_npy(ruta, arreglo) -> None:
    """np.save sobre un archivo abierto: con una ruta sin sufijo .npy, numpy escribiría en otro archivo."""
    with open(ruta, "wb") as archivo:
        np.save(archivo, arreglo)


def es_extractor_de_tokens(nombre_extractor: str) -> bool:
    """Los extractores de tokens (ViT) guardan [CLS ; promedio]; las CNN, un solo vector."""
    arquitectura = nombre_extractor.split(".")[0]
    return "vit" in arquitectura or "eva" in arquitectura


def seleccionar_agregacion(representaciones: np.ndarray, agregacion: str, es_de_tokens: bool = True) -> np.ndarray:
    """De un vector [CLS ; promedio] guardado, devuelve la parte que corresponde a la agregación pedida."""
    if not es_de_tokens or agregacion == "cls_y_promedio":
        return representaciones
    mitad = representaciones.shape[1] // 2
    return representaciones[:, :mitad] if agregacion == "cls" else representaciones[:, mitad:]


def ejecutar_cabeza_congelada(contexto: ContextoDatos, carpeta_modelo: Path, prefijo: str, espec: EspecificacionModelo,
                              config_cabeza, entrada: dict, representaciones: dict, dispositivo,
                              guardar: bool = True, construir_modelo_imagen: bool = True, registro=print) -> ResultadoCorrida:
    """Entrena (o recupera) la cabeza sobre representaciones congeladas y la convierte en un modelo de imagen.

    El resultado es un `ResultadoCorrida` como el de `ejecutar_corrida`, así que el resto del
    cuaderno (comparación, diagnóstico, oclusión y prueba) lo trata igual que a un modelo ajustado.
    Con `construir_modelo_imagen=False` solo se entrena la cabeza (útil en una búsqueda: no se
    carga el extractor) y `modelo` queda en `None`.
    """
    from .arquitecturas import ModeloRegresionImagen, crear_extractor
    from .representaciones import cargar_cabeza, entrenar_cabeza, guardar_cabeza, plegar_estandarizacion

    espec = EspecificacionModelo(**{**espec.a_dict(), "capas_lora": tuple(espec.capas_lora), "estrategia_ajuste": "congelado"})
    contenido = json.dumps({"espec": espec.a_dict(), "cabeza": config_cabeza.a_dict(), "entrada": entrada}, sort_keys=True)
    identificador = f"{prefijo}_semilla{config_cabeza.semilla}_{hashlib.sha1(contenido.encode()).hexdigest()[:8]}"
    carpeta = Path(carpeta_modelo) / "corridas" / identificador
    escalares = {p: (contexto.estandarizador_escalares.transformar(
        contexto.muestra.iloc[contexto.indices[p]][list(contexto.variables_escalares)].to_numpy(float))
        if contexto.variables_escalares else np.zeros((len(contexto.indices[p]), 0), np.float32)) for p in ("Train", "Validation")}
    etiquetas = {p: contexto.estandarizador_etiquetas.transformar(contexto.etiquetas(p)) for p in ("Train", "Validation")}
    es_de_tokens = es_extractor_de_tokens(espec.nombre_extractor)
    r = {p: seleccionar_agregacion(representaciones[p], espec.agregacion_tokens, es_de_tokens) for p in ("Train", "Validation")}
    dimension = r["Train"].shape[1] + len(contexto.variables_escalares)
    reutilizada = guardar and (carpeta / "cabeza.pt").exists()
    if reutilizada:
        resultado = cargar_cabeza(carpeta / "cabeza.pt", dimension, len(contexto.objetivos), config_cabeza)
        registro(f"[{identificador}] ya terminada · se recupera")
    else:
        resultado = entrenar_cabeza(r["Train"], escalares["Train"], etiquetas["Train"], r["Validation"], escalares["Validation"],
                                    etiquetas["Validation"], config_cabeza, dispositivo)
    modelo, parametros = None, {"parametros_totales": None}
    if construir_modelo_imagen:
        modelo = ModeloRegresionImagen(crear_extractor(espec), len(contexto.variables_escalares), len(contexto.objetivos),
                                       config_cabeza.unidades_ocultas, config_cabeza.dropout)
        modelo.cabeza = plegar_estandarizacion(resultado)
        modelo.to(dispositivo).eval()
        parametros = contar_parametros(modelo)
    media, desviacion = (normalizacion_rgb_del_extractor(espec.nombre_extractor)
                         if entrada.get("representacion") == "rgb_lupton" else (None, None))
    resumen = {"identificador": identificador, "especificacion": espec.a_dict(),
               "configuracion": {**config_cabeza.a_dict(), "canales_ultimos": False}, "configuracion_cabeza": config_cabeza.a_dict(),
               "entrada": entrada, **parametros, "parametros_entrenables": int(sum(p.numel() for p in resultado.cabeza.parameters())),
               "mejor_epoca": resultado.mejor_epoca, "mejor_criterio_validacion": resultado.mejor_criterio_validacion,
               "minutos_entrenamiento": resultado.segundos / 60}
    if guardar and not reutilizada:
        guardar_cabeza(contexto.guardia, carpeta / "cabeza.pt", resultado, resumen)
        contexto.guardia.write_json(carpeta / "resumen.json", resumen, allow_replace=True)
    historial = pd.DataFrame({"epoca": [resultado.mejor_epoca], "criterio_validacion": [resultado.mejor_criterio_validacion]})
    return ResultadoCorrida(identificador, carpeta, modelo, crear_preparador(contexto, entrada, media, desviacion), resumen, historial,
                            reutilizada)
