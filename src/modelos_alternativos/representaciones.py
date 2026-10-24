"""Representaciones congeladas: extraer una vez y entrenar solo la cabeza (sondeo, *probing*).

Es la forma en que los autores de los modelos fundacionales evalúan sus representaciones: el
extractor no cambia y encima se entrena un modelo ligero. Como el extractor es fijo, sus vectores
se calculan una sola vez y cada configuración de la cabeza se entrena en segundos.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .arquitecturas import CabezaMultisalida, Extractor
from .datos import ConjuntoEnGPU, PreparadorEntrada, simetria_d4
from .entrenamiento import fijar_semillas, mse_medio_estandarizado


@torch.no_grad()
def extraer_representaciones(extractor: Extractor, preparador: PreparadorEntrada, conjunto: ConjuntoEnGPU,
                             tamano_lote: int = 64, promediar_simetrias: bool = False) -> np.ndarray:
    """Vector del extractor para cada imagen (N, d). Con `promediar_simetrias`, promedio de las 8 vistas D4."""
    extractor.eval()
    dispositivo = conjunto.imagenes.device
    vectores = []
    for i in range(0, len(conjunto), tamano_lote):
        x = preparador(conjunto.imagenes[i:i + tamano_lote])
        vistas = range(8) if promediar_simetrias else (0,)
        acumulado = 0
        for k in vistas:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dispositivo.type == "cuda"):
                acumulado = acumulado + extractor(simetria_d4(x, k) if k else x).float()
        vectores.append((acumulado / len(vistas)).cpu())
    return torch.cat(vectores).numpy()


@dataclass
class ConfiguracionCabeza:
    unidades_ocultas: int = 256
    dropout: float = 0.1
    tasa_aprendizaje: float = 1e-3
    decaimiento_pesos: float = 1e-2
    epocas: int = 200
    tamano_lote: int = 256
    perdida: str = "huber"
    delta_huber: float = 1.0
    estandarizar_representaciones: bool = True
    semilla: int = 42

    def a_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultadoCabeza:
    cabeza: nn.Module
    media: torch.Tensor
    desviacion: torch.Tensor
    mejor_epoca: int
    mejor_criterio_validacion: float
    segundos: float

    @torch.no_grad()
    def predecir(self, representaciones: np.ndarray, escalares: np.ndarray) -> np.ndarray:
        self.cabeza.eval()
        dispositivo = self.media.device
        r = (torch.as_tensor(representaciones, dtype=torch.float32, device=dispositivo) - self.media) / self.desviacion
        e = torch.as_tensor(np.asarray(escalares, np.float32).reshape(len(r), -1), device=dispositivo)
        return self.cabeza(torch.cat([r, e], 1)).cpu().numpy()


def entrenar_cabeza(representaciones_entrenamiento: np.ndarray, escalares_entrenamiento: np.ndarray,
                    etiquetas_entrenamiento: np.ndarray, representaciones_validacion: np.ndarray,
                    escalares_validacion: np.ndarray, etiquetas_validacion: np.ndarray, config: ConfiguracionCabeza,
                    dispositivo=None) -> ResultadoCabeza:
    """Entrena la cabeza multisalida sobre vectores fijos; conserva la época con menor criterio de validación."""
    fijar_semillas(config.semilla)
    dispositivo = dispositivo or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dispositivo)
    r_tr, r_va = t(representaciones_entrenamiento), t(representaciones_validacion)
    n_tr, n_va = len(r_tr), len(r_va)
    e_tr, e_va = t(escalares_entrenamiento).reshape(n_tr, -1), t(escalares_validacion).reshape(n_va, -1)
    y_tr, y_va = t(etiquetas_entrenamiento), t(etiquetas_validacion)
    if config.estandarizar_representaciones:
        media, desviacion = r_tr.mean(0), r_tr.std(0).clamp_min(1e-6)
    else:
        media, desviacion = torch.zeros(r_tr.shape[1], device=dispositivo), torch.ones(r_tr.shape[1], device=dispositivo)
    x_tr = torch.cat([(r_tr - media) / desviacion, e_tr], 1)
    x_va = torch.cat([(r_va - media) / desviacion, e_va], 1)
    cabeza = CabezaMultisalida(x_tr.shape[1], y_tr.shape[1], config.unidades_ocultas, config.dropout).to(dispositivo)
    optimizador = torch.optim.AdamW(cabeza.parameters(), lr=config.tasa_aprendizaje, weight_decay=config.decaimiento_pesos)
    pasos = config.epocas * math.ceil(n_tr / config.tamano_lote)
    programa = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, pasos)
    perdida = nn.HuberLoss(delta=config.delta_huber) if config.perdida == "huber" else nn.MSELoss()
    generador = torch.Generator(device="cpu").manual_seed(config.semilla)
    mejor, inicio = (math.inf, 0, None), time.time()
    for epoca in range(1, config.epocas + 1):
        cabeza.train()
        orden = torch.randperm(n_tr, generator=generador).to(dispositivo)
        for i in range(0, n_tr, config.tamano_lote):
            idx = orden[i:i + config.tamano_lote]
            valor = perdida(cabeza(x_tr[idx]), y_tr[idx])
            optimizador.zero_grad(set_to_none=True); valor.backward(); optimizador.step(); programa.step()
        cabeza.eval()
        with torch.no_grad():
            criterio = mse_medio_estandarizado(cabeza(x_va).cpu().numpy(), y_va.cpu().numpy())
        if criterio < mejor[0]:
            mejor = (criterio, epoca, {k: v.detach().clone() for k, v in cabeza.state_dict().items()})
    cabeza.load_state_dict(mejor[2])
    return ResultadoCabeza(cabeza, media, desviacion, mejor[1], mejor[0], time.time() - inicio)


class CabezaConEstandarizacion(nn.Module):
    """Aplica (r − media) / desviación a la parte de representación del vector y después la cabeza entrenada.

    Así, el extractor congelado y la cabeza forman un modelo de imagen ordinario que reproduce
    exactamente lo aprendido sobre las representaciones guardadas. (Incorporar la estandarización
    a los pesos de la primera capa sería equivalente en teoría, pero pierde precisión cuando alguna
    componente del vector casi no varía.)
    """

    def __init__(self, resultado: "ResultadoCabeza"):
        super().__init__()
        self.cabeza = copy.deepcopy(resultado.cabeza)
        self.register_buffer("media", resultado.media.detach().clone())
        self.register_buffer("desviacion", resultado.desviacion.detach().clone())

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        d = self.media.numel()
        representacion = (vector[:, :d].float() - self.media) / self.desviacion
        return self.cabeza(torch.cat([representacion, vector[:, d:].float()], 1))


def plegar_estandarizacion(resultado: "ResultadoCabeza") -> CabezaConEstandarizacion:
    """Cabeza que recibe la representación sin estandarizar (ver `CabezaConEstandarizacion`)."""
    return CabezaConEstandarizacion(resultado).cpu()


def guardar_cabeza(guardia, ruta, resultado: ResultadoCabeza, descripcion: dict) -> None:
    import io
    import json
    from pathlib import Path
    buffer = io.BytesIO()
    torch.save({"cabeza": resultado.cabeza.state_dict(), "media": resultado.media.cpu(), "desviacion": resultado.desviacion.cpu(),
                "mejor_epoca": resultado.mejor_epoca, "mejor_criterio_validacion": resultado.mejor_criterio_validacion,
                "segundos": resultado.segundos, "descripcion": descripcion}, buffer)
    guardia.write(Path(ruta), lambda f: Path(f).write_bytes(buffer.getvalue()), allow_replace=True)


def cargar_cabeza(ruta, dimension_entrada: int, n_salidas: int, config: ConfiguracionCabeza) -> ResultadoCabeza:
    estado = torch.load(ruta, map_location="cpu")
    cabeza = CabezaMultisalida(dimension_entrada, n_salidas, config.unidades_ocultas, config.dropout)
    cabeza.load_state_dict(estado["cabeza"])
    return ResultadoCabeza(cabeza, estado["media"], estado["desviacion"], estado["mejor_epoca"],
                           estado["mejor_criterio_validacion"], estado["segundos"])
