"""Cohorte del experimento (plan §3.1).

Fuente de verdad (decisión del equipo): el catálogo maestro del equipo
(`catalogo_candidatas_original.csv`, versión 20260925T190214_499654Z) para
etiquetas, errores, QCFLAG y `split_proyecto`. Rutas FITS: `plan.csv`.
`scalars.cat` (Katachi) y la VAC Pipe3D se usan como verificación cruzada y para
las predicciones publicadas de Katachi (B0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CUT_LOG_MSTAR, CUT_LOG_SFR, TARGETS

CUTS = {"log_mstar": CUT_LOG_MSTAR, "log_sfr": CUT_LOG_SFR}
QC_GROUP = {0: "OK", 1: "BAD", 2: "BAD", 3: "WARNING", 4: "WARNING", 5: "WARNING", 6: "WARNING", 7: "WARNING"}
QC_MEANING = {0: "Controles superados", 1: "Redshift incorrecto", 2: "Baja S/N o campo vacío",
              3: "Ajuste y/o AGN intenso", 4: "Masa discrepante con NSA", 5: "Redshift discrepante con NSA",
              6: "Estrella en primer plano", 7: "Sistema en fusión"}
SCALAR_COLS = ["mangaid", "split", "log_mstar", "log_sfr", "d4000", "redshift",
               "pred_mstar", "pred_sfr", "pred_d4000", "t50_model", "t50_Pipe3D", "Av", "Z"]


def _norm_id(s: pd.Series) -> pd.Series:
    return s.map(lambda x: x.decode() if isinstance(x, bytes) else x).astype(str).str.strip()


def load_plan(path: Path, fits_dir: Path | None = None) -> pd.DataFrame:
    """Lee `plan.csv` (mangaid → FITS) y verifica que cada archivo corresponda a su galaxia."""
    plan = pd.read_csv(path, dtype={"mangaid": str})
    plan["mangaid"] = _norm_id(plan["mangaid"])
    if plan["mangaid"].duplicated().any():
        raise ValueError("plan.csv tiene mangaid duplicados")
    plan["archivo"] = plan["ruta"].map(lambda r: Path(r).name)
    # El nombre del archivo empieza con el mangaid; se verifica para no asociar una
    # imagen a la galaxia equivocada (la asociación se hace siempre por mangaid).
    mal = plan[~plan.apply(lambda r: r["archivo"].startswith(r["mangaid"] + "_"), axis=1)]
    if len(mal):
        raise ValueError(f"{len(mal)} rutas no corresponden a su mangaid, p. ej. {mal[['mangaid', 'archivo']].head(3).values.tolist()}")
    if not plan["archivo"].is_unique:
        raise ValueError("plan.csv asigna el mismo FITS a más de una galaxia")
    if fits_dir is not None:
        plan["ruta_local"] = plan["archivo"].map(lambda a: str(Path(fits_dir) / a))
    return plan


def load_scalars(path: Path) -> pd.DataFrame:
    """Lee `scalars.cat` de Katachi: etiquetas Pipe3D, partición de Katachi y sus predicciones."""
    import hickle

    s = hickle.load(str(path))
    s = s[[c for c in SCALAR_COLS if c in s.columns]].copy()
    s["mangaid"] = _norm_id(s["mangaid"])
    for c in s.columns:
        if c not in ("mangaid", "split"):
            s[c] = pd.to_numeric(s[c], errors="coerce")
    return s


def load_pipe3d_qc(path: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """QCFLAG y errores de Pipe3D. La VAC repite 139 mangaid (varios plateifu);
    se elige la fila cuyas etiquetas coinciden con las de Katachi."""
    from astropy.table import Table

    t = Table.read(path, hdu=1)
    cols = ["mangaid", "plateifu", "QCFLAG", "log_Mass", "log_SFR_Ha", "D4000_Re_fit1",
            "e_log_Mass", "e_log_SFR_Ha", "e_D4000_Re_fit"]
    v = t[cols].to_pandas()
    v["mangaid"] = _norm_id(v["mangaid"])
    v["plateifu"] = _norm_id(v["plateifu"])
    m = labels[["mangaid", "log_mstar", "log_sfr", "d4000"]].merge(v, on="mangaid", how="left")
    m["dist"] = (np.abs(m.log_Mass - m.log_mstar) + np.abs(m.log_SFR_Ha - m.log_sfr)
                 + np.abs(m.D4000_Re_fit1 - m.d4000)).fillna(np.inf)
    m = m.sort_values("dist").drop_duplicates("mangaid")
    out = m[["mangaid", "plateifu", "QCFLAG", "e_log_Mass", "e_log_SFR_Ha", "e_D4000_Re_fit", "dist"]].rename(
        columns={"e_log_Mass": "error_log_mstar", "e_log_SFR_Ha": "error_log_sfr",
                 "e_D4000_Re_fit": "error_d4000", "dist": "pipe3d_match_dist"})
    out["grupo_qc"] = out["QCFLAG"].map(QC_GROUP).fillna("DESCONOCIDO")
    return out


TEAM_COLS = ["mangaid", "split_proyecto", "split_katachi", "QCFLAG_original", "categoria_QCFLAG",
             "log_mstar", "log_sfr", "d4000", "redshift", "error_log_mstar", "error_log_sfr", "error_d4000",
             "plateifu_pipe3d", "observaciones_en_pipe3d", "estado_identidad", "FoV", "version_catalogo"]


def load_team_catalog(path: Path) -> pd.DataFrame:
    """Lee el catálogo maestro del equipo y deja las columnas que usa el experimento."""
    t = pd.read_csv(path, dtype={"mangaid": str})
    t["mangaid"] = _norm_id(t["mangaid"])
    if t["mangaid"].duplicated().any():
        raise ValueError("El catálogo del equipo tiene mangaid duplicados")
    falt = [c for c in TEAM_COLS if c not in t.columns]
    if falt:
        raise ValueError(f"Faltan columnas en el catálogo del equipo: {falt}")
    t = t[TEAM_COLS].rename(columns={"plateifu_pipe3d": "plateifu", "QCFLAG_original": "QCFLAG"})
    t["grupo_qc"] = t["QCFLAG"].map(QC_GROUP).fillna("DESCONOCIDO")
    t["varias_obs_pipe3d"] = t["observaciones_en_pipe3d"] > 1
    return t


def cross_check(team: pd.DataFrame, scalars: pd.DataFrame, qc_vac: pd.DataFrame | None = None, tol: float = 1e-9) -> dict:
    """Comprueba que el catálogo del equipo coincide con scalars.cat (etiquetas, split de
    Katachi) y con la VAC Pipe3D (QCFLAG, plate-IFU, errores). Devuelve el número de
    diferencias por campo; todas deben ser 0."""
    out = {}
    j = team.merge(scalars, on="mangaid", how="left", suffixes=("", "_k"), validate="one_to_one")
    out["sin_scalars"] = int(j["split"].isna().sum())
    for c in ("log_mstar", "log_sfr", "d4000", "redshift"):
        out[f"{c}_vs_scalars"] = int((np.abs(j[c] - j[f"{c}_k"]) > tol).sum())
    out["split_katachi_vs_scalars"] = int((j["split_katachi"] != j["split"]).sum())
    if qc_vac is not None:
        v = team.merge(qc_vac, on="mangaid", how="left", suffixes=("", "_vac"), validate="one_to_one")
        out["QCFLAG_vs_vac"] = int((v["QCFLAG"] != v["QCFLAG_vac"]).sum())
        out["plateifu_vs_vac"] = int((v["plateifu"] != v["plateifu_vac"]).sum())
        for c in ("error_log_mstar", "error_log_sfr", "error_d4000"):
            out[f"{c}_vs_vac"] = int((np.abs(v[c] - v[f"{c}_vac"]) > tol).sum())
    return out


def build_cohort_team(plan: pd.DataFrame, team: pd.DataFrame, scalars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cohorte desde el catálogo del equipo. `particion` = split_proyecto. Se añaden
    las predicciones publicadas de Katachi (B0) y sus columnas auxiliares."""
    rep: dict = {"plan": len(plan), "catalogo_equipo": len(team)}
    if set(plan["mangaid"]) != set(team["mangaid"]):
        raise ValueError("plan.csv y el catálogo del equipo no tienen las mismas galaxias")
    aux = scalars[[c for c in ["mangaid", "pred_mstar", "pred_sfr", "pred_d4000", "t50_model", "t50_Pipe3D", "Av", "Z"]
                   if c in scalars.columns]]
    df = plan.merge(team, on="mangaid", how="left", validate="one_to_one").merge(aux, on="mangaid", how="left",
                                                                                validate="one_to_one")
    df["split"] = df["split_katachi"]
    df["particion"] = df["split_proyecto"]
    df["pasa_cortes"] = katachi_cuts(df)
    rep["fuera_de_cortes"] = int((~df["pasa_cortes"]).sum())
    if rep["fuera_de_cortes"]:
        raise ValueError(f"{rep['fuera_de_cortes']} galaxias del catálogo no pasan los cortes de Katachi")
    bad = df[(df["particion"] == "Test") != (df["split"] == "Test")]
    if len(bad):
        raise ValueError(f"split_proyecto Test no coincide con el Test de Katachi en {len(bad)} galaxias")
    rep["cohorte"] = len(df)
    rep["por_particion"] = df["particion"].value_counts().to_dict()
    rep["qc"] = pd.crosstab(df["particion"], df["grupo_qc"]).to_dict()
    return df, rep


def katachi_cuts(df: pd.DataFrame) -> pd.Series:
    """Cortes del artículo: 8 < log M* < 12, −4 < log SFR < 2 y D4000 finito."""
    ok = np.isfinite(df["d4000"])
    for c, (lo, hi) in CUTS.items():
        ok &= (df[c] > lo) & (df[c] < hi)
    return ok


def build_cohort(plan: pd.DataFrame, scalars: pd.DataFrame, qc: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Une plan + scalars (+QC), aplica los cortes y devuelve la cohorte y un reporte."""
    rep: dict = {"plan": len(plan)}
    df = plan.merge(scalars, on="mangaid", how="left", validate="one_to_one", indicator=True)
    rep["sin_scalars"] = int((df["_merge"] != "both").sum())
    df = df.drop(columns="_merge")
    if qc is not None:
        df = df.merge(qc, on="mangaid", how="left", validate="one_to_one")
    df["pasa_cortes"] = katachi_cuts(df)
    rep["fuera_de_cortes"] = int((~df["pasa_cortes"]).sum())
    rep["split_katachi"] = df["split"].value_counts(dropna=False).to_dict()
    df = df[df["pasa_cortes"] & df["split"].isin(["Train", "Test"])].reset_index(drop=True)
    rep["cohorte"] = len(df)
    rep["por_split"] = df["split"].value_counts().to_dict()
    if "grupo_qc" in df:
        rep["qc"] = pd.crosstab(df["split"], df["grupo_qc"]).to_dict()
    return df, rep


def add_validation(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Separa Validation de Train, estratificado por decil de masa (D1-a).
    `particion` ∈ {Train, Validation, Test}. fraction=0 → sin Validation (D1-b)."""
    df = df.copy()
    df["particion"] = df["split"]
    if fraction <= 0:
        return df
    tr = df.index[df["split"] == "Train"]
    deciles = pd.qcut(df.loc[tr, "log_mstar"], 10, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    val = []
    for d in sorted(deciles.unique()):
        idx = np.sort(tr[deciles.values == d])
        k = int(round(fraction * len(idx)))
        val.extend(rng.choice(idx, size=k, replace=False))
    df.loc[val, "particion"] = "Validation"
    return df


def apply_qc(df: pd.DataFrame, groups) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decisión del equipo: conservar solo los grupos QC indicados. Devuelve
    (incluidas, excluidas_con_motivo). La partición de origen se conserva."""
    keep = df["grupo_qc"].isin(groups)
    exc = df.loc[~keep, ["mangaid", "split", "particion", "QCFLAG", "grupo_qc"]].copy()
    exc["motivo"] = exc.apply(lambda r: f"QCFLAG={int(r.QCFLAG) if pd.notna(r.QCFLAG) else 'NA'} "
                                        f"({r.grupo_qc}: {QC_MEANING.get(r.QCFLAG, 'sin dato')})", axis=1)
    return df.loc[keep].reset_index(drop=True), exc.reset_index(drop=True)


def check_targets(df: pd.DataFrame) -> None:
    """Falla si alguna etiqueta del experimento no es un número finito."""
    for t in TARGETS:
        if not np.isfinite(df[t]).all():
            raise ValueError(f"Etiquetas no finitas en {t}")
