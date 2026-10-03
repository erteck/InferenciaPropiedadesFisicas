"""Métricas y comparaciones (plan §1 y §4).

Métricas de Katachi (Fig. 3): R², MSE, MAD (desviación absoluta media) y RMSE.
Añadidas: sesgo, σ_NMAD, fracción de atípicos, pendiente residuo–verdad, IC por
bootstrap y comparación pareada E1 vs B0 con la regla de decisión fijada en §1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import EQUIVALENCE_MARGINS, N_BOOTSTRAP

MARGINS = EQUIVALENCE_MARGINS


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Métricas de regresión de un objetivo (Katachi: R², MSE, MAD, RMSE; más sesgo, σ_NMAD, atípicos y pendiente)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    r = p - y
    med = np.median(r)
    nmad = 1.4826 * np.median(np.abs(r - med))
    return {
        "N": int(len(y)),
        "RMSE": float(np.sqrt(np.mean(r ** 2))),
        "MSE": float(np.mean(r ** 2)),
        "MAD": float(np.mean(np.abs(r))),
        "R2": float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
        "sesgo": float(np.mean(r)),
        "sigma_NMAD": float(nmad),
        "frac_atipicos_3sNMAD": float(np.mean(np.abs(r - med) > 3 * nmad)),
        "pendiente_residuo": float(np.polyfit(y, r, 1)[0]),
        "pearson": float(stats.pearsonr(y, p)[0]),
        "spearman": float(stats.spearmanr(y, p)[0]),
    }


def bootstrap_ci(y, p, stat="RMSE", n=N_BOOTSTRAP, seed=0, alpha=0.05) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap sobre galaxias para RMSE, σ_NMAD o sesgo."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    r = p[idx] - y[idx]
    if stat == "RMSE":
        v = np.sqrt(np.mean(r ** 2, axis=1))
    elif stat == "sigma_NMAD":
        med = np.median(r, axis=1, keepdims=True)
        v = 1.4826 * np.median(np.abs(r - med), axis=1)
    elif stat == "sesgo":
        v = r.mean(axis=1)
    else:
        raise ValueError(stat)
    return tuple(np.quantile(v, [alpha / 2, 1 - alpha / 2]).tolist())


def paired_comparison(y, p_new, p_ref, target: str, n=N_BOOTSTRAP, seed=0) -> dict:
    """ΔRMSE = RMSE(nuevo) − RMSE(referencia) con bootstrap pareado + Wilcoxon sobre
    errores cuadráticos por galaxia. Regla de decisión del plan §1."""
    y, a, b = (np.asarray(v, float) for v in (y, p_new, p_ref))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d_rmse = np.sqrt(np.mean((a[idx] - y[idx]) ** 2, 1)) - np.sqrt(np.mean((b[idx] - y[idx]) ** 2, 1))

    def nmad(r):
        med = np.median(r, axis=1, keepdims=True)
        return 1.4826 * np.median(np.abs(r - med), axis=1)

    d_nmad = nmad(a[idx] - y[idx]) - nmad(b[idx] - y[idx])
    point = float(np.sqrt(np.mean((a - y) ** 2)) - np.sqrt(np.mean((b - y) ** 2)))
    lo, hi = np.quantile(d_rmse, [0.025, 0.975])
    # TOST al 5 %: el IC del 90 % dentro de ±margen
    lo90, hi90 = np.quantile(d_rmse, [0.05, 0.95])
    m = MARGINS[target]
    w = stats.wilcoxon((a - y) ** 2, (b - y) ** 2)
    if hi < 0:
        verdict = "mejor"
    elif lo > 0:
        verdict = "peor"
    elif -m < lo90 and hi90 < m:
        verdict = "equivalente"
    else:
        verdict = "no concluyente"
    return {
        "objetivo": target, "N": int(len(y)),
        "dRMSE": point, "dRMSE_IC95": (float(lo), float(hi)), "dRMSE_IC90": (float(lo90), float(hi90)),
        "dNMAD": float(np.median(d_nmad)), "dNMAD_IC95": tuple(np.quantile(d_nmad, [0.025, 0.975]).tolist()),
        "wilcoxon_p": float(w.pvalue), "margen": m, "veredicto": verdict,
    }


def holm(pvals: list[float]) -> list[float]:
    """Corrección de Holm–Bonferroni para varias pruebas."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def binned(df: pd.DataFrame, y: str, p: str, by: str, bins) -> pd.DataFrame:
    """Sesgo, RMSE y σ_NMAD del residuo dentro de intervalos de otra variable."""
    d = df[list(dict.fromkeys([y, p, by]))].dropna().copy()   # by puede ser igual a y
    d["bin"] = pd.cut(d[by], bins) if not isinstance(bins, str) else d[by]
    rows = []
    for b, g in d.groupby("bin", observed=True):
        if len(g) >= 5:
            m = metrics(g[y], g[p])
            rows.append({"bin": str(b), "N": m["N"], "RMSE": m["RMSE"], "sesgo": m["sesgo"], "sigma_NMAD": m["sigma_NMAD"]})
    return pd.DataFrame(rows)
