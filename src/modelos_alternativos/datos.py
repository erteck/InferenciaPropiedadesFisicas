"""Datos comunes de los modelos alternativos (Avance 4).

Reproduce el protocolo de comparación del equipo (cuaderno `ResNet50_multitarea_..._mod1`):

- Muestra: galaxias con QCFLAG = 0, etiquetas finitas y redshift finito y positivo, sin
  imágenes con una banda completamente en cero en el recorte central.
- Imagen GRZ: `asinh(f / s_b)` con `s_b` = percentil 75 de |f| en 512 posiciones por imagen de
  entrenamiento, y después estandarización por banda con la media y la desviación de
  entrenamiento.
- Etiquetas y redshift: estandarizados con entrenamiento.
- Aumentación: las 8 simetrías del cuadrado (giros de 90° y reflejos), sin interpolación.

Todo se ajusta con las galaxias de entrenamiento y se aplica sin cambios a validación y prueba.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

OBJETIVOS = ("log_mstar", "log_sfr", "d4000")
PARTICIONES = ("Train", "Validation", "Test")
ORDEN_BANDAS_CACHE = ("g", "r", "z")


# ---------------------------------------------------------------------------
# Muestra
# ---------------------------------------------------------------------------
def filtrar_muestra(cohorte: pd.DataFrame, fraccion_ceros: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica los requisitos del protocolo común. Devuelve (muestra, excluidas con su motivo).

    `fraccion_ceros` (N, 3) es la fracción de píxeles en cero de cada banda en el recorte central,
    en el mismo orden que `cohorte`.
    """
    motivos = pd.Series("", index=cohorte.index, dtype=object)
    etiquetas_finitas = np.isfinite(cohorte[list(OBJETIVOS)].to_numpy(float)).all(1)
    redshift_valido = np.isfinite(cohorte["redshift"].to_numpy(float)) & (cohorte["redshift"].to_numpy(float) > 0)
    banda_vacia = (np.asarray(fraccion_ceros) >= 0.999).any(1)
    motivos[~etiquetas_finitas] += "etiqueta no finita; "
    motivos[~redshift_valido] += "redshift no válido; "
    motivos[banda_vacia] += "banda completa en cero; "
    conservar = etiquetas_finitas & redshift_valido & ~banda_vacia
    excluidas = cohorte.loc[~conservar, ["mangaid", "particion"]].assign(motivo=motivos[~conservar].str.rstrip("; "))
    return cohorte.loc[conservar].copy(), excluidas.reset_index(drop=True)


def indices_por_particion(muestra: pd.DataFrame) -> dict[str, np.ndarray]:
    return {p: np.where(muestra["particion"].to_numpy() == p)[0] for p in PARTICIONES}


# ---------------------------------------------------------------------------
# Transformaciones ajustadas con entrenamiento
# ---------------------------------------------------------------------------
@dataclass
class EstandarizadorEscalar:
    """(x − media) / desviación por columna, con estadísticas de entrenamiento (desviación poblacional)."""
    media: np.ndarray
    desviacion: np.ndarray

    @classmethod
    def ajustar(cls, valores_entrenamiento: np.ndarray) -> "EstandarizadorEscalar":
        v = np.asarray(valores_entrenamiento, float)
        v = v[:, None] if v.ndim == 1 else v
        return cls(v.mean(0), v.std(0))

    def transformar(self, valores: np.ndarray) -> np.ndarray:
        v = np.asarray(valores, float)
        v = v[:, None] if v.ndim == 1 else v
        return ((v - self.media) / self.desviacion).astype(np.float32)

    def invertir(self, valores_estandarizados: np.ndarray) -> np.ndarray:
        return np.asarray(valores_estandarizados, float) * self.desviacion + self.media

    def a_dict(self) -> dict:
        return {"media": self.media.tolist(), "desviacion": self.desviacion.tolist()}


@dataclass
class TransformacionGRZ:
    """asinh(f / s_b) seguido de estandarización por banda; parámetros de entrenamiento."""
    escala_asinh: np.ndarray        # (3,) s_b: percentil 75 de |f|
    media: np.ndarray               # (3,) tras asinh
    desviacion: np.ndarray          # (3,) tras asinh

    @classmethod
    def ajustar(cls, imagenes: np.ndarray, indices_entrenamiento: np.ndarray, posiciones_por_imagen: int = 512,
                semilla: int = 0, lote: int = 256) -> "TransformacionGRZ":
        """`imagenes` (N, 3, H, W) en flujo físico (orden g, r, z)."""
        generador = np.random.default_rng(semilla)
        n_pixeles = imagenes.shape[2] * imagenes.shape[3]
        muestras = []
        for i in indices_entrenamiento:
            pos = generador.choice(n_pixeles, posiciones_por_imagen, replace=False)
            muestras.append(np.abs(imagenes[i].reshape(3, -1)[:, pos].astype(np.float32)))
        escala = np.percentile(np.concatenate(muestras, 1), 75, axis=1)
        suma = np.zeros(3); suma_cuadrados = np.zeros(3); total = 0
        for k in range(0, len(indices_entrenamiento), lote):
            x = np.arcsinh(imagenes[indices_entrenamiento[k:k + lote]].astype(np.float32) / escala[None, :, None, None])
            suma += x.sum((0, 2, 3)); suma_cuadrados += (x.astype(np.float64) ** 2).sum((0, 2, 3)); total += x[:, 0].size
        media = suma / total
        return cls(escala, media, np.sqrt(suma_cuadrados / total - media ** 2))

    def tensores(self, dispositivo) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forma = (1, 3, 1, 1)
        return tuple(torch.as_tensor(np.asarray(v, np.float32), device=dispositivo).view(forma)
                     for v in (self.escala_asinh, self.media, self.desviacion))

    def aplicar(self, x: torch.Tensor) -> torch.Tensor:
        escala, media, desviacion = self.tensores(x.device)
        return (torch.asinh(x.float() / escala) - media) / desviacion

    def a_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in vars(self).items()}


def rgb_lupton(x_grz: torch.Tensor, q: float = 8.0, estiramiento: float = 0.5) -> torch.Tensor:
    """Imagen en color de Lupton et al. (2004) con R = z, G = r, B = g, en [0, 1].

    Es la misma composición de `astropy.visualization.make_lupton_rgb` (con `minimum=0`), escrita
    en PyTorch para aplicarla por lotes en la GPU.
    """
    x = x_grz.float().clamp_min(0)
    rojo, verde, azul = x[:, 2], x[:, 1], x[:, 0]
    intensidad = (rojo + verde + azul) / 3
    factor = torch.asinh(q * intensidad / estiramiento) / q / intensidad.clamp_min(1e-12)
    rgb = torch.stack([rojo, verde, azul], 1) * factor.unsqueeze(1)
    maximo = rgb.amax(1, keepdim=True).clamp_min(1.0)        # recorta conservando el color, como astropy
    return (rgb / maximo).clamp(0, 1)


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def recorte_central(x: torch.Tensor, lado: int) -> torch.Tensor:
    alto, ancho = x.shape[-2:]
    if lado >= alto:
        return x
    i, j = (alto - lado) // 2, (ancho - lado) // 2
    return x[..., i:i + lado, j:j + lado]


def ajustar_lado(x: torch.Tensor, lado: int, modo: str = "recorte") -> torch.Tensor:
    """Lleva la imagen a `lado` píxeles: recorte central o reescalado bilineal del campo completo."""
    if x.shape[-1] == lado:
        return x
    if modo == "recorte":
        return recorte_central(x, lado)
    if modo == "reescala":
        return F.interpolate(x, size=(lado, lado), mode="bilinear", align_corners=False, antialias=True)
    raise ValueError(modo)


def simetria_d4(x: torch.Tensor, indice: int) -> torch.Tensor:
    """Una de las 8 simetrías exactas del cuadrado: 0–3 giros de 90°, 4–7 los mismos con reflejo."""
    if indice >= 4:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, indice % 4, dims=(-2, -1))


def aumentar_d4(x: torch.Tensor, generador: torch.Generator) -> torch.Tensor:
    """Aplica una simetría elegida al azar a cada imagen del lote."""
    eleccion = torch.randint(0, 8, (x.shape[0],), generator=generador, device=generador.device).to(x.device)
    salida = torch.empty_like(x)
    for k in range(8):
        sel = eleccion == k
        if sel.any():
            salida[sel] = simetria_d4(x[sel], k)
    return salida


# ---------------------------------------------------------------------------
# Conjunto en memoria de la GPU
# ---------------------------------------------------------------------------
class ConjuntoEnGPU:
    """Una partición en el dispositivo: imágenes crudas (float16), variables escalares y etiquetas.

    `escalares` (N, k) son las variables auxiliares ya estandarizadas que acompañan a la imagen
    (por ejemplo, el redshift). Con k = 0 el modelo usa solo la imagen.
    """

    def __init__(self, imagenes, escalares: np.ndarray | None, etiquetas: np.ndarray, dispositivo):
        """`imagenes` puede ser un arreglo de NumPy o un tensor que ya está en el dispositivo (se reutiliza sin copiar)."""
        n = imagenes.shape[0]
        if isinstance(imagenes, torch.Tensor):
            self.imagenes = imagenes.to(dispositivo)
        else:
            self.imagenes = torch.from_numpy(np.ascontiguousarray(imagenes)).to(dispositivo)
        escalares = np.zeros((n, 0), np.float32) if escalares is None else np.asarray(escalares, np.float32).reshape(n, -1)
        self.escalares = torch.as_tensor(escalares, device=dispositivo)
        self.etiquetas = torch.as_tensor(np.asarray(etiquetas, np.float32).reshape(n, -1), device=dispositivo)

    def __len__(self) -> int:
        return self.imagenes.shape[0]

    @property
    def n_escalares(self) -> int:
        return self.escalares.shape[1]

    def subconjunto(self, n: int) -> "ConjuntoEnGPU":
        """Las primeras `n` galaxias, sin copiar memoria."""
        sub = object.__new__(ConjuntoEnGPU)
        sub.imagenes, sub.escalares, sub.etiquetas = self.imagenes[:n], self.escalares[:n], self.etiquetas[:n]
        return sub


class PreparadorEntrada:
    """Convierte imágenes crudas en la entrada de la red: representación, tamaño y aumentación."""

    def __init__(self, transformacion_grz: TransformacionGRZ, representacion: str = "grz_estandarizado",
                 lado: int = 300, modo_lado: str = "recorte", media_rgb=None, desviacion_rgb=None):
        if representacion not in ("grz_estandarizado", "rgb_lupton"):
            raise ValueError(representacion)
        self.transformacion_grz, self.representacion = transformacion_grz, representacion
        self.lado, self.modo_lado = lado, modo_lado
        self.media_rgb = None if media_rgb is None else torch.tensor(media_rgb).view(1, 3, 1, 1)
        self.desviacion_rgb = None if desviacion_rgb is None else torch.tensor(desviacion_rgb).view(1, 3, 1, 1)

    def __call__(self, imagenes_crudas: torch.Tensor, aumentar: bool = False,
                 generador: torch.Generator | None = None) -> torch.Tensor:
        if self.representacion == "grz_estandarizado":
            x = self.transformacion_grz.aplicar(imagenes_crudas)
        else:
            x = rgb_lupton(imagenes_crudas)
            if self.media_rgb is not None:
                x = (x - self.media_rgb.to(x.device)) / self.desviacion_rgb.to(x.device)
        x = ajustar_lado(x, self.lado, self.modo_lado)
        if aumentar:
            x = aumentar_d4(x, generador)
        return x.contiguous()

    def a_dict(self) -> dict:
        return {"representacion": self.representacion, "lado": self.lado, "modo_lado": self.modo_lado}
