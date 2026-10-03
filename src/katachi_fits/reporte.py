"""Figuras y tablas del cuaderno.

Aquí vive el código de presentación para que las celdas del cuaderno sean cortas
y se lean como pasos del análisis. Ninguna función de este módulo entrena,
modifica datos ni escribe archivos: solo calcula tablas y dibuja.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

from . import evaluate as EV
from . import robustness as RB
from . import shap_maps as SM
from .config import TARGETS

ETIQUETAS = {"log_mstar": "log M* [M⊙]", "log_sfr": "log SFR [M⊙/año]", "d4000": "D4000"}
NOMBRES_RED = ("M*", "SFR", "D4000")


def rgb_desde_grz(x: np.ndarray) -> np.ndarray:
    """Imagen (3, H, W) en orden g, r, z → (H, W, 3) para mostrar (R=z, G=r, B=g).
    Solo para visualizar; la red recibe g, r, z en ese orden."""
    return np.clip(np.stack([x[2], x[1], x[0]], -1), 0, 1)


# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------
def galeria_sdss_desi(sdss: np.ndarray, desi: np.ndarray, titulos: list[str], campo_desi: float, px_desi: int):
    """Las mismas galaxias en la imagen SDSS de Katachi (arriba) y en la entrada de
    nuestra red (abajo)."""
    n = len(titulos)
    fig, axs = plt.subplots(2, n, figsize=(2.7 * n, 5.6), squeeze=False)
    for j in range(n):
        axs[0, j].imshow(np.transpose(sdss[j], (1, 2, 0)), origin="upper")
        axs[1, j].imshow(rgb_desde_grz(desi[j]), origin="lower")
        axs[0, j].set_title(titulos[j], fontsize=8)
    for a in axs.flat:
        a.set_xticks([]); a.set_yticks([])
    axs[0, 0].set_ylabel("SDSS gri\n(Katachi, 50″)")
    axs[1, 0].set_ylabel(f"DESI grz\n(nuestra, {campo_desi:.1f}″)")
    fig.suptitle(f"Mismas galaxias. Abajo: {px_desi} px a escala nativa de DESI.", fontsize=10)
    plt.tight_layout(); plt.show()


def histogramas_entrada(crudo: np.ndarray, entrada: np.ndarray, sigma: np.ndarray):
    """Distribución de píxeles antes (flujo / σ del cielo) y después de la
    transformación (lo que recibe la red)."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
    for b, nb in enumerate("grz"):
        axs[0].hist(crudo[:, b].ravel() / sigma[b], bins=200, range=(-5, 50), histtype="step", label=nb)
        axs[1].hist(entrada[:, b].ravel(), bins=200, range=(-0.3, 1.2), histtype="step", label=nb)
    axs[0].set(xlabel="flujo / σ_b", ylabel="píxeles", yscale="log", title="Antes: flujo en unidades de ruido del cielo")
    axs[1].set(xlabel="valor de entrada", yscale="log", title="Después: asinh(x / βσ) / s  (Katachi usaba [0, 1])")
    axs[0].legend(); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def curvas_aprendizaje(historia: pd.DataFrame):
    """Pérdida por época de cada red (Train con aumentación; Validation sin ella) y su
    tasa de aprendizaje. La línea vertical marca la época en que se detuvo cada red."""
    fig, axs = plt.subplots(2, 3, figsize=(16, 6.5), sharex=True)
    for k, (t, n) in enumerate(zip(TARGETS, ("mstar", "sfr", "d4000"))):
        ax = axs[0, k]
        ax.plot(historia.epoch, historia[f"train_loss_{n}"], label="Train (con aumentación)")
        if f"val_loss_{n}" in historia:
            ax.plot(historia.epoch, historia[f"val_loss_{n}"], label="Validation (solo vigilancia)")
        ax.set(yscale="log", ylabel="MSE", title=ETIQUETAS[t])
        axs[1, k].plot(historia.epoch, historia[f"lr_{n}"])
        axs[1, k].set(yscale="log", xlabel="época", ylabel="tasa de aprendizaje")
        if f"detenida_{n}" in historia and historia[f"detenida_{n}"].fillna(False).astype(bool).any():
            e = int(historia.loc[historia[f"detenida_{n}"].fillna(False).astype(bool), "epoch"].iloc[0])
            for a in axs[:, k]:
                a.axvline(e, color="gray", ls=":", label=f"se detiene (época {e})")
        axs[0, k].legend(fontsize=7)
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def tabla_metricas(df: pd.DataFrame, modelos: dict[str, dict[str, str]], n_boot: int) -> pd.DataFrame:
    """Una fila por modelo y objetivo: RMSE con IC 95 % (bootstrap), R², MAD, sesgo,
    σ_NMAD, fracción de atípicos y pendiente del residuo.
    `modelos` = {nombre: {objetivo: columna_de_predicción}}."""
    filas = []
    for t in TARGETS:
        for nombre, cols in modelos.items():
            m = EV.metrics(df[t], df[cols[t]])
            lo, hi = EV.bootstrap_ci(df[t], df[cols[t]], "RMSE", n=n_boot)
            filas.append({"modelo": nombre, "objetivo": t, **m, "RMSE_IC95": f"[{lo:.3f}, {hi:.3f}]"})
    cols = ["modelo", "objetivo", "N", "RMSE", "RMSE_IC95", "R2", "MAD", "sesgo", "sigma_NMAD",
            "frac_atipicos_3sNMAD", "pendiente_residuo"]
    return pd.DataFrame(filas)[cols]


def figura3(paneles: dict[str, pd.DataFrame], pred: dict[str, str], titulo: str):
    """Figura 3 de Katachi: predicción contra valor verdadero, un panel por partición,
    con R², MSE, MAD y RMSE como en el artículo. `paneles` = {nombre: DataFrame}."""
    fig, axs = plt.subplots(3, len(paneles), figsize=(5 * len(paneles), 14), squeeze=False)
    for i, t in enumerate(TARGETS):
        for j, (nombre, d) in enumerate(paneles.items()):
            ax = axs[i, j]
            ok = np.isfinite(d[pred[t]])
            if ok.sum() == 0:
                ax.set_visible(False); continue
            y, yp = d.loc[ok, t], d.loc[ok, pred[t]]
            m = EV.metrics(y, yp)
            ax.hexbin(y, yp, gridsize=60, bins="log", cmap="viridis", mincnt=1)
            lo, hi = np.percentile(np.r_[y, yp], [0.5, 99.5])
            ax.plot([lo, hi], [lo, hi], "r--", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set(xlabel=f"{ETIQUETAS[t]} verdadero", ylabel="predicho", title=f"{nombre} (N={m['N']})")
            ax.text(0.03, 0.97, f"R²={m['R2']:.3f}\nMSE={m['MSE']:.3f}\nMAD={m['MAD']:.3f}\nRMSE={m['RMSE']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="w", alpha=0.8))
    fig.suptitle(titulo); plt.tight_layout(); plt.show()


def comparacion_pareada(df: pd.DataFrame, nuevo: dict[str, str], ref: dict[str, str], n_boot: int) -> pd.DataFrame:
    """ΔRMSE = RMSE(nuevo) − RMSE(referencia) sobre las mismas galaxias, con su IC y
    el veredicto de la regla del §1; Wilcoxon con corrección de Holm."""
    comp = pd.DataFrame([EV.paired_comparison(df[t], df[nuevo[t]], df[ref[t]], t, n=n_boot) for t in TARGETS])
    comp["wilcoxon_p_holm"] = EV.holm(comp.wilcoxon_p.tolist())
    return comp[["objetivo", "N", "dRMSE", "dRMSE_IC95", "dRMSE_IC90", "margen", "veredicto",
                 "dNMAD", "dNMAD_IC95", "wilcoxon_p", "wilcoxon_p_holm"]]


def residuos_por_intervalos(df: pd.DataFrame, modelos: dict[str, dict[str, str]], intervalos: dict[str, list]):
    """Sesgo ± σ_NMAD del residuo en intervalos de cada variable (masa, redshift…)."""
    fig, axs = plt.subplots(3, len(intervalos), figsize=(4.3 * len(intervalos), 10), squeeze=False)
    for i, t in enumerate(TARGETS):
        for j, (var, bordes) in enumerate(intervalos.items()):
            ax = axs[i, j]
            for k, (nombre, cols) in enumerate(modelos.items()):
                r = EV.binned(df, t, cols[t], var, bordes)
                if len(r):
                    ax.errorbar(np.arange(len(r)) + 0.1 * k, r.sesgo, yerr=r.sigma_NMAD, fmt="os"[k % 2],
                                capsize=3, label=nombre)
                    ax.set_xticks(range(len(r))); ax.set_xticklabels(r.bin, rotation=30, fontsize=7)
            ax.axhline(0, c="k", lw=0.8); ax.set_title(f"{t} según {var}", fontsize=9)
            if j == 0:
                ax.set_ylabel("sesgo ± σ_NMAD")
    axs[0, 0].legend(); plt.tight_layout(); plt.show()


def rmse_por_grupo(df: pd.DataFrame, grupo: str, modelos: dict[str, dict[str, str]]) -> pd.DataFrame:
    """RMSE de cada modelo dentro de cada valor de una variable categórica."""
    filas = []
    for v, d in df.groupby(grupo):
        for t in TARGETS:
            fila = {grupo: v, "objetivo": t, "N": len(d)}
            for nombre, cols in modelos.items():
                fila[f"RMSE {nombre}"] = EV.metrics(d[t], d[cols[t]])["RMSE"] if len(d) > 2 else np.nan
            filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# t50 e historias de formación estelar
# ---------------------------------------------------------------------------
def figura4_t50(df: pd.DataFrame, versiones: list[tuple[str, str, str, str]]):
    """t50 sobre el plano M*–SFR. `versiones` = [(col_t50, col_masa, col_sfr, título)]."""
    fig, axs = plt.subplots(1, len(versiones), figsize=(5.5 * len(versiones), 4.3), squeeze=False)
    for ax, (v, mx, sx, lab) in zip(axs[0], versiones):
        sc = ax.scatter(df[mx], df[sx], c=df[v], s=6, cmap="viridis", vmin=2, vmax=12)
        ax.set(xlabel="log M*", ylabel="log SFR", title=f"t50 · {lab}")
    plt.colorbar(sc, ax=axs[0], label="t50 [Gyr, tiempo cósmico]"); plt.show()


def plano_en_el_pasado(df: pd.DataFrame, col_m: str, col_sfr: str, col_t50: str, lookback: list[float]) -> np.ndarray:
    """Para cada galaxia, reconstruye su historia con dense_basis y devuelve (M*, SFR)
    hace `lookback` Gyr. Forma (N, len(lookback), 2)."""
    from .sfh import mass_sfr_at_lookback, sfh

    out = np.full((len(df), len(lookback), 2), np.nan)
    for i, r in enumerate(df.itertuples()):
        s, t = sfh(getattr(r, col_m), getattr(r, col_sfr), getattr(r, col_t50), r.redshift)
        for k, lb in enumerate(lookback):
            out[i, k] = mass_sfr_at_lookback(s, t, lb, getattr(r, col_m))
    return out


# ---------------------------------------------------------------------------
# Robustez
# ---------------------------------------------------------------------------
def cadena_con_t50(chain, t50_net, escala_gyr: float):
    """Función imagen → (M*, SFR, D4000, t50 [Gyr], log t50), como la cadena completa de Katachi."""
    def f(x):
        p1, p2, p3 = chain(x)
        t = escala_gyr * t50_net(torch.cat([p3, p2 - p1], 1))
        return torch.cat([p1, p2, p3, t, torch.log10(t.clamp_min(1e-3))], 1)
    return f


def degradacion(modelos: dict, df: pd.DataFrame, factores=(1, 2, 4, 8), lote: int = 128) -> pd.DataFrame:
    """RMSE al reducir la resolución por cada factor y volver al tamaño original.
    `modelos` = {nombre: (función, imágenes ya preprocesadas)}."""
    filas = []
    for nombre, (f, x) in modelos.items():
        for fac in factores:
            with torch.no_grad():
                p = torch.cat([f(RB.degrade(x[i:i + lote], fac) if fac > 1 else x[i:i + lote])
                               for i in range(0, len(x), lote)]).cpu().numpy()
            for k, t in enumerate(TARGETS):
                filas.append({"modelo": nombre, "factor": fac, "objetivo": t, "RMSE": EV.metrics(df[t], p[:, k])["RMSE"]})
    return pd.DataFrame(filas)


def figura_degradacion(deg: pd.DataFrame):
    """RMSE contra el factor de degradación de la imagen, por objetivo y modelo."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.5))
    for k, t in enumerate(TARGETS):
        for nombre, d in deg[deg.objetivo == t].groupby("modelo"):
            axs[k].plot(d.factor, d.RMSE, "o-", label=nombre)
        axs[k].set(xscale="log", xticks=[1, 2, 4, 8], xticklabels=["1", "2", "4", "8"],
                   xlabel="factor de degradación", ylabel="RMSE", title=ETIQUETAS[t])
    axs[0].legend(); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def mapas_shap(redes: dict, x: torch.Tensor, i_fondo, i_explicar, nsamples: int) -> dict[str, np.ndarray]:
    """Mapas SHAP (N, H, W, C) por red. El fondo y las galaxias explicadas son
    conjuntos distintos."""
    return {n: SM.shap_maps(m, x[i_fondo], x[i_explicar], batch=8, nsamples=nsamples) for n, m in redes.items()}


def gradientes(mapas: dict[str, np.ndarray], z: np.ndarray, escala: float, escala_katachi: float | None,
               index) -> pd.DataFrame:
    """Perfil radial y gradiente ∇ = perfil(20 kpc) − perfil(0) de cada mapa, con la
    escala real del modelo y, si se indica, con la convención de Katachi (0.5″/px)."""
    out = {}
    for n, m in mapas.items():
        prof = [SM.radial_profile(mi) for mi in m]
        out[f"grad_{n}_real"] = [SM.gradient(p, zi, escala) for p, zi in zip(prof, z)]
        out[f"grad_{n}_katachi"] = [SM.gradient(p, zi, escala_katachi) if escala_katachi else np.nan
                                    for p, zi in zip(prof, z)]
        out[f"perfil_{n}"] = prof
    return pd.DataFrame(out, index=index)


def figura5_mapas(x: np.ndarray, sh: pd.DataFrame, shap: dict[str, dict[str, np.ndarray]], n: int = 5):
    """Imagen y mapas SHAP (media sobre bandas) de `n` galaxias."""
    sel = np.arange(min(n, len(sh)))
    filas = 1 + 3 * len(shap)
    fig, axs = plt.subplots(filas, len(sel), figsize=(3 * len(sel), 2.7 * filas), squeeze=False)
    for j, i in enumerate(sel):
        axs[0, j].imshow(rgb_desde_grz(x[i]), origin="lower")
        axs[0, j].set_title(f"{sh.mangaid.iloc[i]}\nM*={sh.log_mstar.iloc[i]:.2f}", fontsize=8)
        r = 1
        for k, d in shap.items():
            for nr in NOMBRES_RED:
                m = d[nr][i].mean(-1); v = np.abs(m).max() or 1
                axs[r, j].imshow(m, origin="lower", cmap="bwr", vmin=-v, vmax=v)
                if j == 0:
                    axs[r, j].set_ylabel(f"{k}\n{nr}", fontsize=7)
                r += 1
    for a in axs.flat:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()


def figura6_perfiles(sh: pd.DataFrame, gr: dict[str, pd.DataFrame], escalas: dict[str, float], r_max_kpc: float = 25):
    """Perfil radial mediano por tercil de masa, en kpc reales."""
    eje = np.linspace(0, r_max_kpc, 126)
    ter = pd.qcut(sh.log_mstar, 3, labels=["bajo", "medio", "alto"])
    fig, axs = plt.subplots(len(gr), 3, figsize=(15, 3.6 * len(gr)), squeeze=False)
    for r, (k, g) in enumerate(gr.items()):
        for c, n in enumerate(NOMBRES_RED):
            for lab in ["bajo", "medio", "alto"]:
                ps = [np.interp(eje, np.arange(len(p)) * SM.kpc_per_px(z, escalas[k]), p, right=np.nan)
                      for p, z, tt in zip(g[f"perfil_{n}"], sh.redshift, ter) if tt == lab]
                if ps:
                    axs[r, c].plot(eje, np.nanmedian(ps, 0), label=f"M* {lab}")
            axs[r, c].axhline(0, c="k", lw=0.6)
            axs[r, c].set(xlabel="radio [kpc]", ylabel=f"SHAP ({n})", title=k, yscale="symlog")
    axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.show()


def figura7_gradientes(sh: pd.DataFrame, paneles: list[tuple[str, pd.DataFrame, str]]):
    """Gradientes sobre el plano M*–SFR. `paneles` = [(título, tabla, 'real'|'katachi')]."""
    fig, axs = plt.subplots(len(paneles), 3, figsize=(15, 4 * len(paneles)), squeeze=False)
    for r, (k, g, conv) in enumerate(paneles):
        for c, n in enumerate(NOMBRES_RED):
            v = np.asarray(g[f"grad_{n}_{conv}"], float)
            lim = (np.nanpercentile(np.abs(v), 95) if np.isfinite(v).any() else 1) or 1
            sc = axs[r, c].scatter(sh.log_mstar, sh.log_sfr, c=v, cmap="Spectral_r", vmin=-lim, vmax=lim, s=12)
            plt.colorbar(sc, ax=axs[r, c])
            axs[r, c].set(xlabel="log M*", ylabel="log SFR", title=f"{k} · ∇{n}")
    plt.tight_layout(); plt.show()


def comparar_gradientes(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """¿Usan dos modelos las mismas regiones? KS sobre las distribuciones de ∇ y
    Spearman galaxia por galaxia, con la escala real de cada modelo."""
    filas = []
    for n in NOMBRES_RED:
        x, y = np.asarray(a[f"grad_{n}_real"], float), np.asarray(b[f"grad_{n}_real"], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 5:
            filas.append({"objetivo": n, "N": int(ok.sum()), "KS p": stats.ks_2samp(x[ok], y[ok]).pvalue,
                          "Spearman ρ": stats.spearmanr(x[ok], y[ok])[0]})
    return pd.DataFrame(filas)


def figura8_pasado(planos: dict[str, np.ndarray], pos, color: np.ndarray, lookback, etiqueta_color: str):
    """Plano M*–SFR hace `lookback` Gyr, coloreado por un gradiente SHAP."""
    lim = (np.nanpercentile(np.abs(color), 95) if np.isfinite(color).any() else 1) or 1
    fig, axs = plt.subplots(len(planos), len(lookback), figsize=(15, 4 * len(planos)), squeeze=False)
    for r, (lab, P) in enumerate(planos.items()):
        for c, lb in enumerate(lookback):
            sc = axs[r, c].scatter(P[pos, c, 0], P[pos, c, 1], c=color, cmap="Spectral_r", vmin=-lim, vmax=lim, s=12)
            axs[r, c].set(xlabel="log M*", ylabel="log SFR", title=f"{lab} · hace {lb:g} Gyr")
    plt.colorbar(sc, ax=axs[:, -1], label=etiqueta_color); plt.show()


def figura_shap_por_banda(shap: dict[str, dict[str, np.ndarray]], escalas: dict[str, float], orden=("z", "r", "g")):
    """Perfil radial del mapa de M* separado por canal de entrada (Katachi §4.2.3).
    Katachi recibía R = i, G = r, B = g; nuestra red, `orden`."""
    fig, axs = plt.subplots(1, len(shap), figsize=(5.5 * len(shap), 3.6), squeeze=False)
    for a, (k, d) in zip(axs[0], shap.items()):
        bandas = list(orden) if k.startswith("E") else ["i", "r", "g"]   # orden de entrada de cada red
        for b in range(3):
            prof = np.array([SM.azimuthal_average(m[..., b]) for m in d["M*"]])
            a.plot(np.arange(prof.shape[1]) * escalas[k], np.median(prof, 0), label=bandas[b])
        a.set(xlabel="radio [″]", ylabel="SHAP (M*) mediano", title=k); a.legend()
    plt.tight_layout(); plt.show()


def controles_shap(red, fondo: torch.Tensor, x: torch.Tensor, nsamples: int) -> dict:
    """Controles de Katachi: (a) imágenes de ruido deberían dar mapas débiles;
    (b) rotar la imagen 90° debería rotar el mapa igual."""
    ruido = torch.randn_like(x) * x.std() + x.mean()
    s_real = SM.shap_maps(red, fondo, x, nsamples=nsamples)
    s_ruido = SM.shap_maps(red, fondo, ruido, nsamples=nsamples)
    s_rot = SM.shap_maps(red, fondo, torch.rot90(x, 1, (2, 3)), nsamples=nsamples)
    back = np.rot90(s_rot, -1, axes=(1, 2))
    rho = [stats.pearsonr(a.mean(-1).ravel(), b.mean(-1).ravel())[0] for a, b in zip(s_real, back)]
    return {"|SHAP| medio, imágenes reales": float(np.abs(s_real).mean()),
            "|SHAP| medio, ruido": float(np.abs(s_ruido).mean()),
            "correlación mapa vs mapa rotado (mediana)": float(np.nanmedian(rho))}
