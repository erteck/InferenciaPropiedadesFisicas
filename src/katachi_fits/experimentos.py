"""Ejecutor de experimentos: corre una lista de configuraciones una tras otra.

Uso (desde el cuaderno `Notebooks/Experimentos_DESI.ipynb`):

    ctx = preparar_datos(...)              # cohorte, caché, σ y s_b (una vez)
    tabla = correr_lista(EXPERIMENTOS, ctx, ...)

Para cada experimento:
  1. construye su identificador a partir de las opciones encendidas;
  2. si ya terminó (existe `resultado.json`), lo salta;
  3. si quedó a medias, lo reanuda desde `last.pt`;
  4. entrena, predice Validation (con y sin TTA) y guarda métricas, predicciones y
     la comparación pareada contra la línea base del lote;
  5. agrega una fila a `resultados_experimentos.csv` y, si se pide, la registra en
     Weights & Biases.

El conjunto de prueba nunca se usa aquí.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import evaluate as EV
from .aumentacion import sigma_por_imagen
from .cache import build_cache, cache_dir, load_cache
from .config import TARGETS, Experiment, FitsConfig, MejorasConfig, TrainConfig
from .train import GPUData, Trainer
from .transforms import Normalizer
from .variants import DESI

NOMBRES = {"log_mstar": "M*", "log_sfr": "SFR", "d4000": "D4000"}


@dataclass
class Experimento:
    """Una corrida: nombre corto, mejoras y cambios a la receta base."""
    nombre: str
    mejoras: MejorasConfig = field(default_factory=MejorasConfig)
    semilla: int = 42
    congelar: bool = False            # la corrida 2 entrena todas las capas
    descripcion: str = ""

    def identificador(self, crop_px: int) -> str:
        modo = "congelado144" if self.congelar else "completo"
        return f"X_{self.nombre}_recorte{crop_px}_{modo}{self.mejoras.sufijo()}_seed{self.semilla}"


@dataclass
class Contexto:
    """Datos compartidos por todas las corridas de un lote."""
    cohorte: pd.DataFrame
    idx: dict
    y: np.ndarray
    norm: Normalizer
    fits: FitsConfig
    x300: np.ndarray                  # caché de 300 px
    x_margen: np.ndarray | None       # caché con margen (440 px), si se construyó
    sigma_img: dict                   # {"300": (N,3), "margen": (N,3)} σ por imagen
    est_entrada: dict                 # media y desviación por banda tras asinh / s_b (Train),
                                      # por (caché, σ por imagen): ("300", False), ("margen", True)…
    planos: tuple                     # media y desviación de log M* y log SFR (Train)


# ---------------------------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------------------------
def _cache_margen(cohorte, paths, guard, fcfg: FitsConfig, margen_px: int, tag: str, log):
    cfg_m = replace(fcfg, crop_px=margen_px, out_px=margen_px)
    d = cache_dir(paths.cache, DESI, cfg_m, tag)
    build_cache(cohorte, "ruta_local", d, guard, cfg_m, variant=DESI, chunk=500, workers=24, log=log)
    x, ids, _, _ = load_cache(d, dtype=fcfg.storage_dtype)
    assert list(ids) == list(cohorte.mangaid), "El orden de la caché con margen no coincide con la muestra"
    return x


def estadisticas_entrada(x_raw: np.ndarray, idx_tr: np.ndarray, norm: Normalizer, out_px: int,
                         n: int = 1000, seed: int = 0, sigma_img: np.ndarray | None = None) -> tuple:
    """Media y desviación por banda de la entrada de la red (asinh / s_b) en Train.
    Con `sigma_img` (N, 3) se usa el σ de cada imagen, igual que en la corrida."""
    from .aumentacion import estirar, recorte_central

    sel = np.sort(np.random.default_rng(seed).choice(idx_tr, min(n, len(idx_tr)), replace=False))
    sig_g = torch.as_tensor(np.asarray(norm.sigma, np.float32))
    esc = torch.as_tensor(np.asarray(norm.scale, np.float32))
    acc = []
    for i in range(0, len(sel), 100):
        x = recorte_central(torch.from_numpy(x_raw[sel[i:i + 100]].astype(np.float32)), out_px)
        sig = sig_g if sigma_img is None else torch.from_numpy(sigma_img[sel[i:i + 100]].astype(np.float32))
        acc.append(estirar(x, sig, norm.beta, esc).transpose(0, 1).reshape(3, -1))
    v = torch.cat(acc, 1)
    return v.mean(1).numpy(), v.std(1).numpy()


def preparar_datos(cohorte: pd.DataFrame, x300: np.ndarray, norm: Normalizer, fcfg: FitsConfig, paths, guard,
                   experimentos: list[Experimento], tag: str = "", log=print) -> Contexto:
    """Construye lo que necesitan las corridas: caché con margen (si alguna la usa),
    σ por imagen (si alguna lo usa) y las estadísticas de Train para estandarizar."""
    idx = {p: np.where(cohorte.particion == p)[0] for p in ("Train", "Validation", "Test")}
    y = cohorte[list(TARGETS)].to_numpy(np.float32)
    usa_margen = any(e.mejoras.margen_rotacion for e in experimentos)
    usa_sigma = any(e.mejoras.sigma_por_imagen for e in experimentos)
    x_m = _cache_margen(cohorte, paths, guard, fcfg, MejorasConfig(margen_rotacion=True).margen_px, tag, log) \
        if usa_margen else None
    sig = {}
    if usa_sigma:
        t0 = time.time()
        sig["300"] = sigma_por_imagen(torch.from_numpy(x300)).numpy()
        if x_m is not None:
            sig["margen"] = sigma_por_imagen(torch.from_numpy(x_m)).numpy()
        log(f"[datos] σ por imagen calculado en {time.time() - t0:.0f} s")
    # Estadísticas para estandarizar la imagen, calculadas igual que las ve cada corrida:
    # con la misma caché (el centro de 300 px) y el mismo σ (global o de cada imagen).
    est = {}
    for e in experimentos:
        clave = ("margen" if e.mejoras.margen_rotacion else "300", e.mejoras.sigma_por_imagen)
        if clave not in est:
            xs = x_m if clave[0] == "margen" else x300
            est[clave] = estadisticas_entrada(xs, idx["Train"], norm, fcfg.out_px,
                                              sigma_img=sig[clave[0]] if clave[1] else None)
    tr = cohorte.iloc[idx["Train"]]
    planos = ([float(tr.log_mstar.mean()), float(tr.log_sfr.mean())],
              [float(tr.log_mstar.std()), float(tr.log_sfr.std())])
    log(f"[datos] Train {len(idx['Train'])} · Validation {len(idx['Validation'])} · "
        f"caché con margen: {'sí' if x_m is not None else 'no'} · σ por imagen: {'sí' if usa_sigma else 'no'}")
    return Contexto(cohorte, idx, y, norm, fcfg, x300, x_m, sig, est, planos)


# ---------------------------------------------------------------------------
# Una corrida
# ---------------------------------------------------------------------------
def _metricas(y: np.ndarray, p: np.ndarray, sufijo: str = "") -> dict:
    fila = {}
    for k, t in enumerate(TARGETS):
        m = EV.metrics(y[:, k], p[:, k])
        lo, hi = EV.bootstrap_ci(y[:, k], p[:, k], n=1000)
        n = NOMBRES[t]
        fila.update({f"RMSE_{n}{sufijo}": m["RMSE"], f"IC_{n}{sufijo}": f"[{lo:.3f}, {hi:.3f}]",
                     f"sesgo_{n}{sufijo}": m["sesgo"], f"NMAD_{n}{sufijo}": m["sigma_NMAD"], f"R2_{n}{sufijo}": m["R2"]})
    return fila


def correr(exp: Experimento, ctx: Contexto, paths, guard, device, base: TrainConfig | None = None,
           max_epochs: int | None = None, log=print) -> dict:
    """Entrena y evalúa un experimento. Devuelve la fila de resultados."""
    m = exp.mejoras
    run_id = exp.identificador(ctx.fits.crop_px)
    run_dir = guard.mkdir(paths.run_dir(run_id))
    f_res = run_dir / "resultado.json"
    if f_res.exists():
        log(f"[{exp.nombre}] ya terminado · se reutiliza {run_id}")
        return json.loads(f_res.read_text())

    tcfg = replace(base or TrainConfig(), seed=exp.semilla, freeze_pretrained=exp.congelar)
    if max_epochs is not None:
        tcfg = replace(tcfg, max_epochs=max_epochs)
        if m.programa_rapido:
            m = replace(m, epocas_rapido=min(m.epocas_rapido, max_epochs))
    cfg_json = json.loads(Experiment(paths=paths, fits=ctx.fits, train=tcfg, mejoras=m).to_json())
    cfg_json.update(nombre=exp.nombre, descripcion=exp.descripcion, normalizador=json.loads(ctx.norm.to_json()))
    f_cfg = run_dir / "config.json"
    if not f_cfg.exists():
        guard.write_json(f_cfg, cfg_json)

    if m.margen_rotacion and ctx.x_margen is None:
        raise ValueError("Este experimento usa margen_rotacion: inclúyelo en la lista que recibe preparar_datos")
    x = ctx.x_margen if m.margen_rotacion else ctx.x300
    sig = ctx.sigma_img.get("margen" if m.margen_rotacion else "300") if m.sigma_por_imagen else None
    sel = lambda p: (x[ctx.idx[p]], ctx.y[ctx.idx[p]], None if sig is None else sig[ctx.idx[p]])
    est = ctx.est_entrada[("margen" if m.margen_rotacion else "300", m.sigma_por_imagen)]
    trainer = Trainer(tcfg, ctx.norm, ctx.fits.out_px, run_dir, guard, device, log=log, mejoras=m,
                      estadisticas_entrada=est, planos=ctx.planos)
    TR = GPUData(*sel("Train")[:2], device, sigma=sel("Train")[2])
    VA = GPUData(*sel("Validation")[:2], device, sigma=sel("Validation")[2])

    t0 = time.time()
    hist = trainer.fit(TR, VA)
    t_ent = time.time() - t0
    del TR
    if device.type == "cuda":
        torch.cuda.empty_cache()

    y_va = ctx.y[ctx.idx["Validation"]]
    p = trainer.predict(VA)
    p_tta = trainer.predict(VA, tta=True)
    pred = pd.DataFrame({"mangaid": ctx.cohorte.mangaid.iloc[ctx.idx["Validation"]].to_numpy()})
    for k, t in enumerate(TARGETS):
        pred[f"pred_{t}"], pred[f"pred_tta_{t}"] = p[:, k], p_tta[:, k]
    guard.write(run_dir / "predicciones_validacion.csv", lambda f: pred.to_csv(f, index=False), allow_replace=True)

    fila = {"experimento": exp.nombre, "descripcion": exp.descripcion, "id": run_id, "semilla": exp.semilla,
            "epocas": int(len(hist)), "s_por_epoca": float(hist.seconds.mean()), "min_entrenamiento": t_ent / 60,
            **_metricas(y_va, p), **_metricas(y_va, p_tta, "_tta"),
            "opciones": json.dumps(asdict(m), ensure_ascii=False)}
    guard.write_json(f_res, fila)
    return fila


# ---------------------------------------------------------------------------
# La lista
# ---------------------------------------------------------------------------
def comparar_con_base(run_dirs: tuple[Path, Path], y_va: np.ndarray) -> dict:
    """Comparación pareada (bootstrap) de un experimento contra la línea base del lote,
    ambos sin TTA, sobre las mismas galaxias de Validation."""
    pe, pb = (pd.read_csv(d / "predicciones_validacion.csv") for d in run_dirs)
    assert (pe.mangaid.values == pb.mangaid.values).all(), "Las galaxias de Validation no coinciden"
    out = {}
    for k, t in enumerate(TARGETS):
        c = EV.paired_comparison(y_va[:, k], pe[f"pred_{t}"].to_numpy(), pb[f"pred_{t}"].to_numpy(), t, n=2000)
        n = NOMBRES[t]
        out[f"dRMSE_{n}"] = c["dRMSE"]
        out[f"dRMSE_IC_{n}"] = f"[{c['dRMSE_IC95'][0]:+.3f}, {c['dRMSE_IC95'][1]:+.3f}]"
        out[f"veredicto_{n}"] = c["veredicto"]
    return out


def correr_lista(experimentos: list[Experimento], ctx: Contexto, paths, guard, device, base: TrainConfig | None = None,
                 max_epochs: int | None = None, wandb_proyecto: str | None = None, log=print) -> pd.DataFrame:
    """Corre todos los experimentos en orden y devuelve la tabla de resultados.
    El primero de la lista es la línea base contra la que se comparan los demás."""
    f_tabla = paths.results / "experimentos" / "resultados_experimentos.csv"
    guard.mkdir(f_tabla.parent)
    filas = []
    y_va = ctx.y[ctx.idx["Validation"]]
    dir_base = None
    for i, exp in enumerate(experimentos):
        log(f"\n===== {i + 1}/{len(experimentos)} · {exp.nombre}: {exp.descripcion}")
        run = None
        ya_terminado = (paths.run_dir(exp.identificador(ctx.fits.crop_px)) / "resultado.json").exists()
        if wandb_proyecto and not ya_terminado:      # no se vuelve a registrar lo que ya se subió
            import wandb
            run = wandb.init(project=wandb_proyecto, name=exp.nombre, reinit=True,
                             config={"descripcion": exp.descripcion, "semilla": exp.semilla, **asdict(exp.mejoras)})
        fila = correr(exp, ctx, paths, guard, device, base=base, max_epochs=max_epochs, log=log)
        d = paths.run_dir(fila["id"])
        if dir_base is None:
            dir_base = d
        else:
            fila.update(comparar_con_base((d, dir_base), y_va))
        if run is not None:
            hist = pd.read_csv(d / "history.csv")
            for _, h in hist.iterrows():                       # curvas por época
                run.log({c: h[c] for c in hist.columns if c != "epoch" and pd.api.types.is_number(h[c])},
                        step=int(h.epoch))
            run.summary.update({k: v for k, v in fila.items() if isinstance(v, (int, float, str))})
            run.finish()
        filas.append(fila)
        tabla = pd.DataFrame(filas)
        guard.write(f_tabla, lambda f, t=tabla: t.to_csv(f, index=False), allow_replace=True)
    return pd.DataFrame(filas)
