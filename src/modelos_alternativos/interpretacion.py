"""Interpretación por oclusión, válida para cualquier modelo `f(imagen, escalares)`.

- Oclusión de un canal: se pone en 0 un canal de la entrada ya preparada (el nivel del cielo en la
  representación GRZ estandarizada corresponde a un valor cercano a 0).
- Oclusión de una región: se ponen en 0 todos los canales dentro de un anillo a ≤ r/R50 < b, con el
  radio de mitad de luz de cada galaxia.
- Permutación de una variable escalar: se revuelven sus valores entre galaxias.

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
def predecir_con_modificacion(modelo, preparador: PreparadorEntrada, conjunto: ConjuntoEnGPU, canal: int | None = None,
                              anillo: tuple[float, float] | None = None, radio_mitad_luz_px: np.ndarray | None = None,
                              escalar_permutado: int | None = None, semilla: int = 0, tamano_lote: int = 128,
                              precision_mixta: bool = True) -> np.ndarray:
    """Predicción (unidades estandarizadas) con una sola modificación de la entrada."""
    modelo.eval()
    dispositivo = conjunto.imagenes.device
    escala_radio = None
    escalares = conjunto.escalares
    if escalar_permutado is not None:
        orden = torch.from_numpy(np.random.default_rng(semilla).permutation(len(conjunto))).to(dispositivo)
        escalares = escalares.clone(); escalares[:, escalar_permutado] = escalares[orden, escalar_permutado]
    salidas = []
    for i in range(0, len(conjunto), tamano_lote):
        x = preparador(conjunto.imagenes[i:i + tamano_lote])
        if canal is not None:
            x = x.clone(); x[:, canal] = 0
        if anillo is not None:
            if escala_radio is None:     # el radio se midió en la imagen de 300 px; se lleva al lado de la entrada
                escala_radio = (x.shape[-1] / conjunto.imagenes.shape[-1]) if preparador.modo_lado == "reescala" else 1.0
            radios = torch.as_tensor(radio_mitad_luz_px[i:i + tamano_lote] * escala_radio, device=dispositivo)
            x = x * ~mascara_anillo(radios, x.shape[-1], *anillo)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision_mixta and dispositivo.type == "cuda"):
            salidas.append(modelo(x, escalares[i:i + tamano_lote]).float().cpu())
    return torch.cat(salidas).numpy()
