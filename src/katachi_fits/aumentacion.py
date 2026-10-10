"""Preparación de la imagen y aumentación por lote en GPU (mejoras desactivables).

Se usa solo cuando alguna opción de `MejorasConfig` está encendida; con todas
apagadas, `Trainer` sigue el camino original de `transforms.py` sin cambios.

Orden de las operaciones (cada una justificada en
Documentación/Propuesta_mejoras_preprocesamiento_DESI.md):

    flujo crudo (B, 3, N, N), N = 300 o el recorte con margen
      → [solo entrenamiento] ruido gaussiano con el σ del cielo y desenfoque de PSF,
        ambos en flujo, antes del estiramiento, para que se parezcan al ruido real
      → asinh(x / (β σ)) / s_b, con σ global o por imagen
      → geometría: [solo entrenamiento] reflejos, rotación y desplazamiento en una
        sola transformación afín; siempre, recorte central a 300 px
      → [opcional] estandarización por banda (media 0, desviación 1)
      → orden de bandas de la red (z, r, g)

La geometría va después del estiramiento y antes de estandarizar, para que el
relleno de las esquinas (cuando no hay margen) valga 0, es decir, "sin flujo".
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sigma_por_imagen(x_raw: torch.Tensor, r_frac: float = 0.45, chunk: int = 256) -> torch.Tensor:
    """σ del cielo de cada imagen y banda: 1.4826 × MAD de los píxeles a más de
    `r_frac` × lado del centro, sin contar los ceros exactos (zonas sin cobertura).
    Devuelve (N, 3) en float32."""
    n, b, h, w = x_raw.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = torch.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    anillo = (r > r_frac * min(h, w)).reshape(-1)
    out = torch.empty(n, b)
    for i in range(0, n, chunk):
        v = x_raw[i:i + chunk].float().reshape(-1, b, h * w)[:, :, anillo]
        v = torch.where(v == 0, torch.nan, v)
        med = v.nanmedian(dim=2, keepdim=True).values
        out[i:i + chunk] = 1.4826 * (v - med).abs().nanmedian(dim=2).values
    return out.nan_to_num(nan=float(out.nanmedian())).clamp_min(1e-6)


def ruido_y_psf(x: torch.Tensor, sigma: torch.Tensor, gen: torch.Generator, ruido_max: float,
                psf_max_px: float, prob: float) -> torch.Tensor:
    """Aumentación física en flujo. Con probabilidad `prob` por imagen:
    - suma ruido gaussiano k·σ·N(0, 1), con k ~ U(0, ruido_max);
    - convoluciona cada banda con una gaussiana σ_psf ~ U(0, psf_max_px).
    `sigma` es (B, 3) o (3,)."""
    bsz, nb, h, w = x.shape
    dev = x.device
    sig = sigma.to(dev, torch.float32).reshape(-1, nb) if sigma.dim() > 1 else sigma.to(dev).view(1, nb).expand(bsz, nb)
    if ruido_max > 0:
        activo = (torch.rand(bsz, generator=gen, device=dev) < prob).float()
        k = torch.rand(bsz, generator=gen, device=dev) * ruido_max * activo
        x = x + (k[:, None] * sig)[:, :, None, None] * torch.randn(x.shape, generator=gen, device=dev)
    if psf_max_px > 0:
        activo = (torch.rand(bsz, generator=gen, device=dev) < prob).float()
        s = (torch.rand(bsz, generator=gen, device=dev) * psf_max_px * activo).clamp_min(1e-3)
        rad = max(1, math.ceil(3 * psf_max_px))
        t = torch.arange(-rad, rad + 1, device=dev, dtype=torch.float32)
        ker = torch.exp(-0.5 * (t[None] / s[:, None]) ** 2)
        ker = (ker / ker.sum(1, keepdim=True)).repeat_interleave(nb, 0)          # (B·3, k)
        z = x.reshape(1, bsz * nb, h, w)
        z = F.conv2d(F.pad(z, (rad, rad, 0, 0), mode="reflect"), ker[:, None, None, :], groups=bsz * nb)
        z = F.conv2d(F.pad(z, (0, 0, rad, rad), mode="reflect"), ker[:, None, :, None], groups=bsz * nb)
        x = z.reshape(bsz, nb, h, w)
    return x


def estirar(x: torch.Tensor, sigma: torch.Tensor, beta: float, escala: torch.Tensor) -> torch.Tensor:
    """asinh(x / (β σ)) / s_b. `sigma` es (B, 3) por imagen o (3,) global."""
    nb = x.shape[1]
    s = sigma.to(x.device, torch.float32)
    s = s.view(-1, nb, 1, 1) if s.dim() > 1 else s.view(1, nb, 1, 1)
    return torch.asinh(x / (beta * s)) / escala.to(x.device, torch.float32).view(1, nb, 1, 1)


def recorte_central(x: torch.Tensor, out_px: int) -> torch.Tensor:
    n = x.shape[-1]
    if n == out_px:
        return x
    lo = (n - out_px) // 2
    return x[..., lo:lo + out_px, lo:lo + out_px]


def geometria(x: torch.Tensor, gen: torch.Generator, out_px: int, grados: float, bilineal: bool,
              desplazamiento_px: int) -> torch.Tensor:
    """Reflejos (p = 0.5 cada uno), rotación U(0, grados) y desplazamiento U(−d, d) px,
    en una sola transformación afín que muestrea directamente la región central de
    `out_px` px. Si la entrada tiene margen (N ≥ out_px·√2 + desplazamiento), la
    imagen girada no tiene esquinas vacías."""
    bsz, _, n, _ = x.shape
    dev = x.device
    u = torch.rand(bsz, 5, generator=gen, device=dev)
    fx = torch.where(u[:, 0] < 0.5, -1.0, 1.0)
    fy = torch.where(u[:, 1] < 0.5, -1.0, 1.0)
    ang = u[:, 2] * math.radians(grados)
    esc = out_px / n                                   # el recorte de salida en coordenadas normalizadas de la entrada
    c, s = torch.cos(ang), torch.sin(ang)
    tx = (u[:, 3] * 2 - 1) * desplazamiento_px * 2 / n
    ty = (u[:, 4] * 2 - 1) * desplazamiento_px * 2 / n
    theta = torch.stack([torch.stack([c * fx * esc, -s * fy * esc, tx], 1),
                         torch.stack([s * fx * esc, c * fy * esc, ty], 1)], 1)
    grid = F.affine_grid(theta, (bsz, x.shape[1], out_px, out_px), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear" if bilineal else "nearest", padding_mode="zeros",
                         align_corners=False)


D4 = [(k, f) for f in (False, True) for k in range(4)]


def d4(x: torch.Tensor, i: int) -> torch.Tensor:
    """La i-ésima de las 8 simetrías exactas del cuadrado (giros de 90° y reflejos)."""
    k, f = D4[i]
    y = torch.flip(x, dims=(-1,)) if f else x
    return torch.rot90(y, k, dims=(-2, -1)) if k else y
