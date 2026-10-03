"""t50 y historias de formación estelar, como en Katachi (plan §3.6).

Verificado contra `scalars.cat`:
- `t50_model = 13.6 × T50Net(D4000, log SFR − log M*)` con las etiquetas Pipe3D.
- `sfh = dense_basis.tuple_to_sfh([log M*, log SFR, 1, t50_model / cosmo.age(z)], z)`
  reproduce exactamente la columna `sfh` publicada.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .chain import T50Net, load_katachi_state
from .config import T50_SCALE_GYR

T50_SCALE = T50_SCALE_GYR


def load_t50(path: Path) -> T50Net:
    """Carga la red D4000→t50 publicada por Katachi."""
    m = T50Net()
    m.load_state_dict(load_katachi_state(path), strict=True)
    return m.eval()


@torch.no_grad()
def predict_t50(net: T50Net, log_mstar, log_sfr, d4000) -> np.ndarray:
    """t50 en Gyr a partir de M*, SFR y D4000 (la red usa D4000 y sSFR)."""
    x = np.stack([np.asarray(d4000, float), np.asarray(log_sfr, float) - np.asarray(log_mstar, float)], 1)
    return T50_SCALE * net(torch.tensor(x, dtype=torch.float32)).squeeze(1).numpy()


def sfh(log_mstar, log_sfr, t50_gyr, z):
    """Una SFH con dense_basis. Devuelve (sfr(t), t_cosmico[Gyr])."""
    import dense_basis as db

    age = db.cosmo.age(float(z)).value
    q = float(t50_gyr) / age
    q = min(max(q, 1e-3), 0.99)
    s, t = db.tuple_to_sfh(np.array([float(log_mstar), float(log_sfr), 1.0, q]), float(z))
    return np.asarray(s), np.asarray(t)


def mass_sfr_at_lookback(sfr_t: np.ndarray, t: np.ndarray, lookback_gyr: float, log_mstar_now: float):
    """M* y SFR a un tiempo de retroceso, integrando la SFH (como la Fig. 8 de Katachi).
    La masa se escala para que la integral total coincida con M* observada."""
    t_obs = t.max()
    tt = t_obs - lookback_gyr
    if tt <= t.min():
        return np.nan, np.nan
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (sfr_t[1:] + sfr_t[:-1]) * np.diff(t))])
    frac = np.interp(tt, t, cum) / cum[-1] if cum[-1] > 0 else np.nan
    m = log_mstar_now + np.log10(frac) if frac and frac > 0 else np.nan
    sfr_then = np.interp(tt, t, sfr_t)
    return m, (np.log10(sfr_then) if sfr_then > 0 else np.nan)
