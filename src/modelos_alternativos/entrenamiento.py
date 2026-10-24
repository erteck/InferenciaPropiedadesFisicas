"""Entrenamiento genérico de los modelos alternativos (Avance 4).

El bucle no conoce la arquitectura ni las variables: recibe un modelo `f(imagen, escalares)`, un
preparador de entrada y dos `ConjuntoEnGPU`. Las etiquetas ya están estandarizadas, así que el
promedio del MSE estandarizado entre salidas es exactamente el MSE normalizado del protocolo común.

- Optimizador AdamW con grupos de parámetros (tasa del extractor con decaimiento por capa y tasa
  propia de la cabeza).
- Tasa con calentamiento lineal y descenso en coseno, por paso.
- Pérdida Huber o MSE, promediada entre salidas.
- Se guarda el estado de la época con menor criterio de validación (regla del protocolo común).
- `al_terminar_epoca(epoca, valor)` permite que una búsqueda de hiperparámetros detenga pruebas
  poco prometedoras (poda).
- Las corridas se guardan en una carpeta propia y se reutilizan si ya terminaron.
"""
from __future__ import annotations

import copy
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .arquitecturas import grupos_de_parametros
from .datos import ConjuntoEnGPU, PreparadorEntrada, simetria_d4


@dataclass
class ConfiguracionEntrenamiento:
    epocas: int = 30
    epocas_calentamiento: int = 2
    tamano_lote: int = 32
    tasa_extractor: float = 1e-4
    tasa_cabeza: float = 1e-3
    decaimiento_pesos: float = 0.05
    decaimiento_por_capa: float = 0.8          # 1.0 = misma tasa en todo el extractor
    perdida: str = "huber"                     # "huber" o "mse"
    delta_huber: float = 1.0
    recorte_gradiente: float = 5.0
    aumentar: bool = True                      # simetrías D4 en entrenamiento
    precision_mixta: bool = True               # bfloat16 en GPU
    canales_ultimos: bool = True               # formato channels_last (ayuda a las CNN)
    checkpoint_gradientes: bool = False        # ahorra memoria en extractores grandes
    semilla: int = 42

    def a_dict(self) -> dict:
        return asdict(self)


def fijar_semillas(semilla: int) -> None:
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla); torch.cuda.manual_seed_all(semilla)


def crear_perdida(config: ConfiguracionEntrenamiento) -> nn.Module:
    if config.perdida == "huber":
        return nn.HuberLoss(delta=config.delta_huber)
    if config.perdida == "mse":
        return nn.MSELoss()
    raise ValueError(config.perdida)


def mse_medio_estandarizado(prediccion: np.ndarray, etiqueta: np.ndarray) -> float:
    """Promedio entre salidas del MSE en unidades estandarizadas (= MSE normalizado medio)."""
    return float(np.mean(np.mean((np.asarray(prediccion) - np.asarray(etiqueta)) ** 2, axis=0)))


def _programa_tasa(optimizador, pasos_totales: int, pasos_calentamiento: int):
    def factor(paso):
        if paso < pasos_calentamiento:
            return (paso + 1) / max(1, pasos_calentamiento)
        avance = (paso - pasos_calentamiento) / max(1, pasos_totales - pasos_calentamiento)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, avance)))
    return torch.optim.lr_scheduler.LambdaLR(optimizador, factor)


@torch.no_grad()
def predecir(modelo: nn.Module, preparador: PreparadorEntrada, conjunto: ConjuntoEnGPU, tta: bool = False,
             tamano_lote: int = 128, precision_mixta: bool = True, canales_ultimos: bool = False) -> np.ndarray:
    """Predicción en unidades estandarizadas. Con `tta`, promedio de las 8 simetrías del cuadrado."""
    modelo.eval()
    dispositivo = conjunto.imagenes.device
    usar_bf16 = precision_mixta and dispositivo.type == "cuda"
    salidas = []
    for i in range(0, len(conjunto), tamano_lote):
        x = preparador(conjunto.imagenes[i:i + tamano_lote])
        escalares = conjunto.escalares[i:i + tamano_lote]
        acumulado = 0
        vistas = range(8) if tta else (0,)
        for k in vistas:
            xv = simetria_d4(x, k) if k else x
            if canales_ultimos:
                xv = xv.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=usar_bf16):
                acumulado = acumulado + modelo(xv, escalares).float()
        salidas.append((acumulado / len(vistas)).cpu())
    return torch.cat(salidas).numpy()


@dataclass
class ResultadoEntrenamiento:
    historial: pd.DataFrame
    mejor_epoca: int
    mejor_criterio_validacion: float
    segundos: float
    memoria_maxima_gb: float = float("nan")
    estado_mejor: dict = field(default_factory=dict, repr=False)


def entrenar(modelo: nn.Module, preparador: PreparadorEntrada, entrenamiento: ConjuntoEnGPU, validacion: ConjuntoEnGPU,
             config: ConfiguracionEntrenamiento, al_terminar_epoca: Callable[[int, float], None] | None = None,
             registro: Callable[[str], None] = print) -> ResultadoEntrenamiento:
    """Entrena y deja en `modelo` los pesos de la época con menor criterio de validación."""
    fijar_semillas(config.semilla)
    dispositivo = entrenamiento.imagenes.device
    modelo.to(dispositivo)
    if config.canales_ultimos:
        modelo.to(memory_format=torch.channels_last)
    if config.checkpoint_gradientes and hasattr(modelo.extractor.red, "set_grad_checkpointing"):
        modelo.extractor.red.set_grad_checkpointing(True)
    grupos = grupos_de_parametros(modelo, config.tasa_extractor, config.tasa_cabeza, config.decaimiento_pesos,
                                  config.decaimiento_por_capa)
    optimizador = torch.optim.AdamW(grupos, fused=dispositivo.type == "cuda")
    pasos_por_epoca = math.ceil(len(entrenamiento) / config.tamano_lote)
    programa = _programa_tasa(optimizador, config.epocas * pasos_por_epoca, config.epocas_calentamiento * pasos_por_epoca)
    perdida = crear_perdida(config)
    generador_orden = torch.Generator(device="cpu").manual_seed(config.semilla)
    generador_aumento = torch.Generator(device=dispositivo).manual_seed(config.semilla + 1)
    usar_bf16 = config.precision_mixta and dispositivo.type == "cuda"
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    historial, mejor = [], (math.inf, 0, None)
    inicio = time.time()
    for epoca in range(1, config.epocas + 1):
        modelo.train()
        t0, suma_perdida, lotes = time.time(), 0.0, 0
        orden = torch.randperm(len(entrenamiento), generator=generador_orden).to(dispositivo)
        for i in range(0, len(entrenamiento), config.tamano_lote):
            idx = orden[i:i + config.tamano_lote]
            x = preparador(entrenamiento.imagenes[idx], aumentar=config.aumentar, generador=generador_aumento)
            if config.canales_ultimos:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=usar_bf16):
                prediccion = modelo(x, entrenamiento.escalares[idx])
            valor = perdida(prediccion.float(), entrenamiento.etiquetas[idx])
            optimizador.zero_grad(set_to_none=True)
            valor.backward()
            if config.recorte_gradiente:
                torch.nn.utils.clip_grad_norm_([p for g in grupos for p in g["params"]], config.recorte_gradiente)
            optimizador.step(); programa.step()
            suma_perdida += float(valor.detach()); lotes += 1
        pred_validacion = predecir(modelo, preparador, validacion, precision_mixta=config.precision_mixta,
                                   canales_ultimos=config.canales_ultimos)
        criterio = mse_medio_estandarizado(pred_validacion, validacion.etiquetas.cpu().numpy())
        fila = {"epoca": epoca, "perdida_entrenamiento": suma_perdida / lotes, "criterio_validacion": criterio,
                **{f"mse_validacion_salida{k}": float(np.mean((pred_validacion[:, k] - validacion.etiquetas[:, k].cpu().numpy()) ** 2))
                   for k in range(pred_validacion.shape[1])},
                "tasa_cabeza": optimizador.param_groups[-1]["lr"], "segundos": time.time() - t0}
        historial.append(fila)
        if criterio < mejor[0]:
            mejor = (criterio, epoca, {k: v.detach().to("cpu", copy=True) for k, v in modelo.state_dict().items()})
        registro(f"época {epoca:3d} · pérdida {fila['perdida_entrenamiento']:.4f} · validación {criterio:.4f}"
                 f"{'  ← mejor' if mejor[1] == epoca else ''} · {fila['segundos']:.0f} s")
        if al_terminar_epoca is not None:
            al_terminar_epoca(epoca, criterio)          # puede lanzar una excepción para podar la prueba
    modelo.load_state_dict(mejor[2])
    memoria = torch.cuda.max_memory_allocated() / 1e9 if dispositivo.type == "cuda" else float("nan")
    return ResultadoEntrenamiento(pd.DataFrame(historial), mejor[1], mejor[0], time.time() - inicio, memoria, mejor[2])


# ---------------------------------------------------------------------------
# Corridas guardadas
# ---------------------------------------------------------------------------
def guardar_corrida(guardia, carpeta: Path, resultado: ResultadoEntrenamiento, descripcion: dict) -> None:
    """Guarda pesos de la mejor época, historial y un resumen JSON. `guardia` es un WriteGuard."""
    guardia.mkdir(carpeta)
    buffer = io.BytesIO(); torch.save(resultado.estado_mejor, buffer)
    guardia.write(carpeta / "mejor_modelo.pt", lambda f: Path(f).write_bytes(buffer.getvalue()), allow_replace=True)
    guardia.write(carpeta / "historial.csv", lambda f: resultado.historial.to_csv(f, index=False), allow_replace=True)
    resumen = {**descripcion, "mejor_epoca": resultado.mejor_epoca,
               "mejor_criterio_validacion": resultado.mejor_criterio_validacion,
               "minutos_entrenamiento": resultado.segundos / 60, "memoria_maxima_gb": resultado.memoria_maxima_gb}
    guardia.write(carpeta / "resumen.json", lambda f: Path(f).write_text(json.dumps(resumen, indent=2, ensure_ascii=False)),
                  allow_replace=True)


def corrida_terminada(carpeta: Path) -> bool:
    return (Path(carpeta) / "resumen.json").exists() and (Path(carpeta) / "mejor_modelo.pt").exists()


def cargar_corrida(carpeta: Path, modelo: nn.Module) -> tuple[nn.Module, dict, pd.DataFrame]:
    carpeta = Path(carpeta)
    modelo.load_state_dict(torch.load(carpeta / "mejor_modelo.pt", map_location="cpu"))
    return modelo, json.loads((carpeta / "resumen.json").read_text()), pd.read_csv(carpeta / "historial.csv")
