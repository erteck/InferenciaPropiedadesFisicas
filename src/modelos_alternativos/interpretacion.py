"""Interpretación por oclusión, válida para cualquier modelo `f(imagen, escalares)` y cualquier representación.

Las modificaciones se hacen sobre la imagen en flujo físico, antes de prepararla para la red. Los
coadds de DESI tienen el cielo restado, así que un flujo de 0 corresponde al nivel del cielo:

- Oclusión de una banda: su flujo se pone en 0 (la banda deja de aportar información).
- Oclusión de una región: el flujo de las tres bandas se pone en 0 dentro de un anillo
  a ≤ r/R50 < b, con el radio de mitad de luz de cada galaxia medido en la misma imagen.
- Permutación de una variable escalar: se revuelven sus valores entre galaxias.

Después, la imagen modificada pasa por la misma preparación que usa el modelo (estandarización o
composición en color, recorte o reescalado), de modo que la prueba es igual para todos los modelos.
La importancia de cada elemento es el aumento del error cuando el modelo deja de recibirlo.
"""
from __future__ import annotations

import numpy as np
import torch

from .datos import ConjuntoEnGPU, PreparadorEntrada


def mascara_anillo(radio_mitad_luz_px: torch.Tensor, lado: int, interior: float, exterior: float) -> torch.Tensor:
    ejes = torch.arange(lado, device=radio_mitad_luz_px.device)
    yy, xx = torch.meshgrid(ejes, ejes, indexing="ij")
    radio = torch.hypot(yy - (lado - 1) / 2, xx - (lado - 1) / 2).float()
    u = radio[None] / radio_mitad_luz_px.view(-1, 1, 1).float()
    return ((u >= interior) & (u < exterior)).unsqueeze(1)


@torch.no_grad()
def predecir_con_modificacion(modelo, preparador: PreparadorEntrada, conjunto: ConjuntoEnGPU, banda: int | None = None,
                              anillo: tuple[float, float] | None = None, radio_mitad_luz_px: np.ndarray | None = None,
                              escalar_permutado: int | None = None, semilla: int = 0, tamano_lote: int = 128,
                              precision_mixta: bool = True) -> np.ndarray:
    """Predicción (unidades estandarizadas) con una sola modificación de la entrada.

    `banda` es el índice de la banda en la imagen cruda (0 = g, 1 = r, 2 = z). `radio_mitad_luz_px`
    está en píxeles de la imagen cruda.
    """
    modelo.eval()
    dispositivo = conjunto.imagenes.device
    escalares = conjunto.escalares
    if escalar_permutado is not None:
        orden = torch.from_numpy(np.random.default_rng(semilla).permutation(len(conjunto))).to(dispositivo)
        escalares = escalares.clone(); escalares[:, escalar_permutado] = escalares[orden, escalar_permutado]
    salidas = []
    for i in range(0, len(conjunto), tamano_lote):
        crudas = conjunto.imagenes[i:i + tamano_lote]
        if banda is not None or anillo is not None:
            crudas = crudas.clone()
        if banda is not None:
            crudas[:, banda] = 0
        if anillo is not None:
            radios = torch.as_tensor(radio_mitad_luz_px[i:i + tamano_lote], device=dispositivo)
            crudas = crudas * ~mascara_anillo(radios, crudas.shape[-1], *anillo)
        x = preparador(crudas)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision_mixta and dispositivo.type == "cuda"):
            salidas.append(modelo(x, escalares[i:i + tamano_lote]).float().cpu())
    return torch.cat(salidas).numpy()
