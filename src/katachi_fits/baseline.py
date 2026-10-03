"""Línea base B0: resultados publicados de Katachi (plan §3.2 y §2.1).

- `pred_mstar/pred_sfr/pred_d4000` de `scalars.cat`. Se verificó que salen de las
  imágenes de `images.cat` SIN aumentación pasadas por la cadena publicada
  (|Δ| ~1e-4); por eso E1 también se evalúa sin aumentación.
- Para SHAP, Katachi usó redes solo-imagen (masa = primera de la cadena;
  `sfr_solo.pytorch`; `d4000.pytorch`), con imágenes `ToTensor()` en [0, 1].
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import functional as TF


def images_tensor(images_df, ids) -> torch.Tensor:
    """Imágenes RGB de Katachi como tensor (N,3,256,256) en [0,1], en el orden de `ids`."""
    m = images_df.set_index(images_df["mangaid"].astype(str).str.strip())["images"]
    return torch.stack([TF.to_tensor(m.loc[i]) for i in ids])


def load_images_cat(path):
    """Carga `images.cat` de Katachi: un DataFrame con `mangaid` e `images` (PIL RGB 256×256)."""
    import hickle

    d = hickle.load(str(path))
    d["mangaid"] = d["mangaid"].astype(str).str.strip()
    return d


@torch.no_grad()
def predict_chain(chain, x: torch.Tensor, bs: int = 64) -> np.ndarray:
    """Predicciones (N, 3) de una cadena en modo evaluación, por lotes, sin aumentación."""
    chain.eval()
    dev = next(chain.parameters()).device
    out = [torch.cat(chain(x[i:i + bs].to(dev)), 1).cpu() for i in range(0, x.shape[0], bs)]
    return torch.cat(out).numpy()
