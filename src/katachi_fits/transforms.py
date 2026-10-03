"""Transformaciones en GPU (plan §3.3 y §3.4).

Orden: [remuestreo, solo si OUTPUT_PX ≠ CROP_PX] → asinh(x/(β σ_b)) → /s_b →
volteos → rotación. Con el recorte de 300 px a escala nativa no hay remuestreo.
La rotación usa exactamente `torchvision.transforms.functional.rotate` con
NEAREST y fill=0, como el `RandomRotation((0,360))` de Katachi. Como el vecino
más cercano conmuta con operaciones por píxel, aplicar el estiramiento antes o
después de rotar da lo mismo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import FitsConfig


@dataclass
class Normalizer:
    """Constantes de la transformación de brillo: σ del cielo y escala por banda, y β."""
    sigma: np.ndarray   # (3,) σ del cielo por banda, tras remuestreo
    scale: np.ndarray   # (3,) s_b, percentil 99.9 de Train tras el estiramiento
    beta: float

    def to_json(self) -> str:
        return json.dumps({"sigma": self.sigma.tolist(), "scale": self.scale.tolist(), "beta": self.beta})

    @staticmethod
    def from_json(s: str) -> "Normalizer":
        d = json.loads(s)
        return Normalizer(np.asarray(d["sigma"], float), np.asarray(d["scale"], float), float(d["beta"]))


def resample(x: torch.Tensor, out_px: int) -> torch.Tensor:
    """Bilineal: conserva el brillo superficial por píxel; el flujo total escala
    por (out_px/in_px)². Si el tamaño ya es out_px, no hace nada."""
    if x.shape[-1] == out_px and x.shape[-2] == out_px:
        return x
    return F.interpolate(x, size=(out_px, out_px), mode="bilinear", align_corners=False)


def stretch(x: torch.Tensor, sigma, beta: float) -> torch.Tensor:
    """asinh(x / (β σ_b)) por banda."""
    s = torch.as_tensor(np.asarray(sigma, np.float32), device=x.device).view(1, -1, 1, 1)
    return torch.asinh(x / (beta * s))


def fit_scale(x_raw: torch.Tensor, cfg: FitsConfig, sigma, n_max: int = 1000, seed: int = 0) -> np.ndarray:
    """s_b = percentil `scale_percentile` de los píxeles estirados de Train."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x_raw.shape[0], generator=g)[:n_max]
    out = []
    for b in range(0, len(idx), 100):
        y = stretch(resample(x_raw[idx[b:b + 100]].float(), cfg.out_px), sigma, cfg.beta)
        out.append(y.transpose(0, 1).reshape(3, -1).cpu())
    y = torch.cat(out, 1)
    q = cfg.scale_percentile / 100
    # torch.quantile tiene un límite de tamaño; se usa numpy.
    return np.array([np.quantile(y[b].numpy(), q) for b in range(3)])


def preprocess(x_raw: torch.Tensor, norm: Normalizer, out_px: int) -> torch.Tensor:
    """Imagen cruda → entrada de la red: [remuestreo] → asinh(x / βσ) → / s_b."""
    s = torch.as_tensor(norm.scale.astype(np.float32), device=x_raw.device).view(1, -1, 1, 1)
    return stretch(resample(x_raw.float(), out_px), norm.sigma, norm.beta) / s


def augment(x: torch.Tensor, gen: torch.Generator, degrees: float = 360.0) -> torch.Tensor:
    """RandomHorizontalFlip(0.5) → RandomVerticalFlip(0.5) → RandomRotation((0,deg),
    NEAREST, fill=0), por muestra, como el Compose pickleado de Katachi."""
    n = x.shape[0]
    u = torch.rand(n, 3, generator=gen, device=gen.device)
    hflip = u[:, 0] < 0.5
    vflip = u[:, 1] < 0.5
    angles = (u[:, 2] * degrees).tolist()
    out = torch.empty_like(x)
    for i in range(n):
        xi = x[i]
        if hflip[i]:
            xi = TF.hflip(xi)
        if vflip[i]:
            xi = TF.vflip(xi)
        out[i] = TF.rotate(xi, angles[i], interpolation=InterpolationMode.NEAREST, fill=0.0)
    return out
