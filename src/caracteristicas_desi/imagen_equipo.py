"""Preparación de la imagen y canales derivados, adaptados de los cuadernos del equipo.

Origen:
- `Preprocesamiento_v1.3.1.ipynb` (José Florencio Maguey Peralta): fondo del cielo con
  `Background2D`, colapso multibanda, comparación de estiramientos con entropía, saturación y
  ancho de la CDF, máscara gaussiana con estrellas, contorno activo y canal de grumosidad.
- `DESI_Entrenamiento_Mejorado_Jupyter.ipynb`: representaciones para la red (asinh con el
  percentil 75 y estandarización con entrenamiento; RGB de Lupton con normalización ImageNet)
  y auditoría de las entradas (finitud, banda en cero, duplicados exactos).

Cambios respecto de los originales, para que funcionen con nuestros datos (cubos g, r, z de
DESI DR10, en nanomaggies) y en todas las galaxias:
- 3 bandas (g, r, z) en lugar de 4 (g, r, i, z), y recortes de 300 px o cubos de 800 px.
- El tamaño de caja de `Background2D` se calcula con el tamaño de la galaxia (PETRO_TH90), y
  la galaxia y las estrellas se enmascaran antes de estimar el fondo (sección 5.2 del cuaderno).
- El centro se toma de la forma del arreglo y no de una variable global del encabezado.
- La entropía se calcula en bits (base 2) y la saturación como porcentaje de píxeles de la
  galaxia en el valor máximo.
- La grumosidad excluye el núcleo y resta la contribución esperada del ruido, como en la
  suavidad S de Conselice (2003), para poder compararla entre galaxias.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import (AsinhStretch, AsymmetricPercentileInterval, ImageNormalize, LinearStretch,
                                   LogStretch, MinMaxInterval, PercentileInterval, make_lupton_rgb)
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, detect_threshold
from scipy import ndimage

from . import config as K
from .fotometria import cielo, mapa_radial, rellenar_simetrico, segmentar

# --- Constantes ------------------------------------------------------------------
GRUM_SIGMA_PX = 8.0          # suavizado del canal de grumosidad (valor del Preprocesamiento: σ = 8 px ≈ 2.1″)
GRUM_NUCLEO_PX = 2 * GRUM_SIGMA_PX   # radio central excluido: el núcleo brillante no es "grumo"
GRUM_MEDIANA_PX = 17         # lado del filtro de mediana (≈ 2 σ + 1, la misma escala que el gaussiano)
LOG_A = 100.0                # LogStretch(a = 100): valor ganador del Preprocesamiento
ASINH_NSIGMA = 3.0           # AsinhStretch: transición lineal → logarítmica a 3σ del cielo
R_CENTRAL_PX = 6             # radio de la región central para la prueba de orden de brillo (≈ 1.6″)
IMAGENET_MEDIA = np.array([0.485, 0.456, 0.406])   # canales R, G, B (torchvision)
IMAGENET_DESV = np.array([0.229, 0.224, 0.225])


# --- 5.2 Fondo del cielo y máscaras ---------------------------------------------------
def caja_fondo(r90_arcsec: float, n: int, pixscale: float = K.PIXSCALE) -> int:
    """Tamaño de caja de Background2D: cuatro veces R90, entre 32 px y n/4.

    El Preprocesamiento usaba el eje mayor de la galaxia / 5. Con una caja más pequeña que la
    galaxia, el halo se confunde con cielo y se resta luz propia (lo medimos en la sección 5.2).
    """
    r90_px = r90_arcsec / pixscale if np.isfinite(r90_arcsec) and r90_arcsec > 0 else 20.0
    return int(np.clip(4 * r90_px, 32, n // 4))


def fondo_2d(img: np.ndarray, caja: int, mascara: np.ndarray | None = None) -> Background2D:
    """Mapa del cielo con mediana por cajas y recorte sigma (3σ), como en el Preprocesamiento."""
    sc = SigmaClip(sigma=3.0)
    filtro = 3
    return Background2D(img, (caja, caja), filter_size=(filtro, filtro), sigma_clip=sc,
                        bkg_estimator=MedianBackground(sigma_clip=sc), mask=mascara)


def mascara_gaussiana(img: np.ndarray, rms, nsigma: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Máscara de la galaxia central y de las demás fuentes (estrellas y vecinas).

    Suaviza con un núcleo gaussiano de 3 px, detecta fuentes por encima de `nsigma` veces el
    ruido y separa la fuente más cercana al centro del resto.
    """
    suave = convolve(img, Gaussian2DKernel(x_stddev=3, y_stddev=3))
    umbral = detect_threshold(suave, nsigma, error=np.broadcast_to(rms, img.shape))   # argumentos por posición:
    seg = detect_sources(suave, umbral, 5)          # photutils cambió sus nombres entre versiones
    if seg is None:
        vacia = np.zeros(img.shape, bool)
        return vacia, vacia
    lab = seg.data
    cy, cx = (np.array(img.shape) - 1) / 2
    etiquetas = np.setdiff1d(np.unique(lab), [0])
    yy, xx = np.indices(img.shape)
    dist = [np.hypot(yy[lab == e].mean() - cy, xx[lab == e].mean() - cx) for e in etiquetas]
    central = etiquetas[int(np.argmin(dist))]
    galaxia = ndimage.binary_fill_holes(lab == central)
    return galaxia, (lab > 0) & ~galaxia


def contorno_activo(img_norm: np.ndarray, mascara: np.ndarray, iteraciones: int = 10) -> np.ndarray:
    """Ajusta el borde de la máscara a los bordes de la imagen (contorno activo geodésico).

    Usa el gradiente gaussiano inverso y `morphological_geodesic_active_contour` de scikit-image
    (Caselles et al. 1997; Márquez-Neila et al. 2014), con los parámetros del Preprocesamiento.
    """
    from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
    g = inverse_gaussian_gradient(img_norm)
    c = morphological_geodesic_active_contour(g, num_iter=iteraciones, init_level_set=mascara.astype(np.int8),
                                              threshold=0.1, balloon=0.2, smoothing=2)
    return ndimage.binary_fill_holes(c.astype(bool))


# --- 5.3 Colapso multibanda y estiramiento --------------------------------------------
def colapsos(cubo: np.ndarray) -> dict[str, np.ndarray]:
    """Las tres formas de combinar las bandas del Preprocesamiento.

    - promedio sin cielo: promedio de las bandas después de restar la mediana del cielo;
    - promedio crudo: promedio sin restar el cielo (grupo de control);
    - inverso de la varianza: cada banda pesa 1/σ², de modo que la banda con menos ruido pesa más.
    """
    med_sig = [cielo(cubo[b]) for b in range(cubo.shape[0])]
    sub = cubo - np.array([m for m, _ in med_sig], np.float32)[:, None, None]
    w = np.array([1 / s ** 2 if np.isfinite(s) and s > 0 else 0.0 for _, s in med_sig])
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    return {"promedio sin cielo": sub.mean(0), "promedio crudo": cubo.mean(0),
            "inverso de la varianza": np.tensordot(w, sub, axes=1)}


def _intervalos():
    return {"min-max": MinMaxInterval(), "percentil 98": PercentileInterval(98),
            "percentil asimétrico 1–98": AsymmetricPercentileInterval(1, 98)}


def normalizar(img: np.ndarray, estiramiento: str, intervalo) -> np.ndarray:
    """Aplica un intervalo calculado en la propia imagen y un estiramiento; devuelve valores en [0, 1]."""
    vmin, vmax = intervalo.get_limits(img)
    if estiramiento == "lineal":
        st = LinearStretch()
    elif estiramiento == "log":
        st = LogStretch(a=LOG_A)
    else:
        _, _, s = sigma_clipped_stats(img, sigma=3, maxiters=5)
        a = ASINH_NSIGMA * s / (vmax - vmin) if vmax > vmin else 0.1
        st = AsinhStretch(a=float(np.clip(a, 1e-4, 1.0)))
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=st, clip=True)
    return np.asarray(norm(img), dtype=np.float32)


def metricas_imagen(x01: np.ndarray, galaxia: np.ndarray) -> dict:
    """Métricas del Preprocesamiento sobre una imagen normalizada en [0, 1].

    - entropía de Shannon en bits de la imagen en 8 bits (máximo 8): variedad de tonos;
    - saturación: % de píxeles de la galaxia en el valor máximo (núcleo "quemado");
    - ancho de la CDF: niveles de gris entre los percentiles 15 y 85 (contraste del cuerpo);
    - brillo central: valor medio en los 6 px centrales, para la prueba de orden de brillo.
    """
    from skimage.measure import shannon_entropy
    u8 = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    cdf = np.cumsum(np.bincount(u8.ravel(), minlength=256)) / u8.size
    r = mapa_radial(x01.shape[0])
    return {"entropía [bits]": float(shannon_entropy(u8, base=2)),
            "saturación [%]": float(np.mean(u8[galaxia] >= 255) * 100) if galaxia.any() else np.nan,
            "ancho CDF": int(np.argmax(cdf >= 0.85) - np.argmax(cdf >= 0.15)),
            "brillo central": float(x01[r < R_CENTRAL_PX].mean())}


def rejilla_normalizaciones(X: np.ndarray, ids, norm_katachi=None) -> pd.DataFrame:
    """Evalúa las 27 combinaciones (3 colapsos × 3 estiramientos × 3 intervalos) en cada galaxia.

    Si se da `norm_katachi` (σ, s_b, β), agrega como referencia la transformación global de la red
    (la misma para todas las galaxias), promediada sobre las bandas.
    """
    filas = []
    for x, m in zip(X, ids):
        x = np.asarray(x, np.float32)
        col = colapsos(x)
        galaxia, _, _ = segmentar(col["promedio sin cielo"])
        for nombre_c, img in col.items():
            for nombre_i, intervalo in _intervalos().items():
                for est in ("lineal", "log", "asinh"):
                    filas.append({"mangaid": m, "colapso": nombre_c, "estiramiento": est, "intervalo": nombre_i,
                                  **metricas_imagen(normalizar(img, est, intervalo), galaxia)})
        r = mapa_radial(x.shape[-1])
        filas.append({"mangaid": m, "colapso": "referencia", "estiramiento": "flujo crudo", "intervalo": "ninguno",
                      "brillo central": float(col["promedio sin cielo"][r < R_CENTRAL_PX].mean())})
        if norm_katachi is not None:
            sig, esc, beta = norm_katachi
            k = (np.arcsinh(x / (beta * sig[:, None, None])) / esc[:, None, None]).mean(0)
            filas.append({"mangaid": m, "colapso": "referencia", "estiramiento": "global de la red (Katachi)",
                          "intervalo": "el mismo para todas", **metricas_imagen(np.clip(k, 0, 1), galaxia)})
    return pd.DataFrame(filas)


# --- 5.4 Representaciones de Entrenamiento (M05) ---------------------------------------
def asinh_p75(X: np.ndarray, X_entrenamiento: np.ndarray, n_max: int = 2_000_000, seed: int = 0):
    """asinh(f / s_b), con s_b el percentil 75 de |f| en entrenamiento, y después media 0 y
    desviación 1 por banda con estadísticos de entrenamiento. Devuelve (X transformado, estadísticos)."""
    rng = np.random.default_rng(seed)
    s, mu, sd = [], [], []
    for b in range(X.shape[1]):
        v = X_entrenamiento[:, b].ravel()
        v = v[rng.choice(v.size, min(n_max, v.size), replace=False)]
        s_b = float(np.percentile(np.abs(v), 75))
        a = np.arcsinh(v / s_b)
        s.append(s_b); mu.append(float(a.mean())); sd.append(float(a.std()))
    s, mu, sd = (np.array(t, np.float32)[None, :, None, None] for t in (s, mu, sd))
    return (np.arcsinh(X / s) - mu) / sd, {"s_b": s.ravel(), "media": mu.ravel(), "desviación": sd.ravel()}


def rgb_lupton_imagenet(X: np.ndarray, q: float = 8.0, stretch: float = 0.5) -> np.ndarray:
    """RGB de Lupton (R = z, G = r, B = g; Lupton et al. 2004) normalizado como ImageNet.
    Se devuelve en el orden de bandas g, r, z para compararlo con las demás representaciones."""
    out = np.empty_like(X, dtype=np.float32)
    for k, x in enumerate(X):
        rgb = make_lupton_rgb(x[2], x[1], x[0], Q=q, stretch=stretch).astype(np.float32) / 255
        rgb = (rgb - IMAGENET_MEDIA) / IMAGENET_DESV          # (N, N, 3) en orden R, G, B = z, r, g
        out[k] = rgb[..., ::-1].transpose(2, 0, 1)            # → g, r, z
    return out


def fraccion_relleno_rotacion(n: int = 300, paso_grados: int = 5) -> pd.DataFrame:
    """Fracción de píxeles que una rotación rellena con 0, según el ángulo."""
    uno = np.ones((n, n), np.float32)
    filas = []
    for ang in range(0, 360, paso_grados):
        rot = ndimage.rotate(uno, ang, reshape=False, order=0, cval=0.0)
        filas.append({"ángulo": ang, "fracción rellenada": float((rot == 0).mean())})
    return pd.DataFrame(filas)


# --- 4.1 y 6.7 Auditoría de entradas y grumosidad (todas las galaxias) -----------------
def _residuo_positivo(img: np.ndarray, filtro: str, sel: np.ndarray) -> np.ndarray:
    """max(I − I_suave, 0), calculado solo en la caja que contiene `sel` (más rápido)."""
    ys, xs = np.nonzero(sel)
    m = GRUM_MEDIANA_PX if filtro == "mediana" else int(3 * GRUM_SIGMA_PX)
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, img.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, img.shape[1])
    sub = img[y0:y1, x0:x1]
    if filtro == "mediana":
        suave = ndimage.median_filter(sub, size=GRUM_MEDIANA_PX, mode="nearest")
    else:
        suave = ndimage.gaussian_filter(sub, GRUM_SIGMA_PX, mode="nearest", truncate=2.5)
    out = np.zeros_like(img)
    out[y0:y1, x0:x1] = np.clip(sub - suave, 0, None)
    return out


def grumosidad(img: np.ndarray, galaxia: np.ndarray, otras: np.ndarray | None = None,
               filtro: str = "mediana") -> float:
    """Fracción de la luz de la galaxia en estructuras pequeñas (canal de grumosidad).

    Preprocesamiento: grumos = max(I − I_suave, 0), con I_suave un filtro gaussiano de σ = 8 px.
    Adaptaciones, medidas en la sección 6.7 del cuaderno:
    - `filtro="mediana"` (por omisión): I_suave es un filtro de mediana de 17 px. El filtro
      gaussiano, en un perfil que decae, deja I_suave por encima de I en las afueras, y entonces
      la medida termina describiendo la forma del perfil y no los grumos. La mediana no tiene ese
      sesgo en perfiles que decaen y elimina igualmente los grumos pequeños.
    - Se excluye el núcleo (r < 16 px), cuyo pico siempre deja residuo.
    - Se resta lo que aporta el ruido, medido con el mismo filtro en el cielo de las esquinas, como
      el término de fondo de la suavidad S (Conselice 2003).
    """
    r = mapa_radial(img.shape[0])
    region = galaxia & (r >= GRUM_NUCLEO_PX)
    cielo_ = (r > K.CIELO_R_MIN_PX) & ~galaxia & (img != 0)
    if otras is not None:
        cielo_ &= ~otras
    if region.sum() < 20 or cielo_.sum() < 100:
        return float("nan")
    total = float(np.clip(img[region], 0, None).sum())
    alta_gal = _residuo_positivo(img, filtro, region)
    # Cielo: una esquina basta y es mucho más barata que filtrar todo el recorte.
    n = img.shape[0]
    esquina = np.zeros_like(cielo_)
    esquina[: n // 5, : n // 5] = True
    sky = cielo_ & esquina if (cielo_ & esquina).sum() >= 100 else cielo_
    alta_cielo = _residuo_positivo(img, filtro, sky)
    ruido = region.sum() * float(alta_cielo[sky].mean())
    return (float(alta_gal[region].sum()) - ruido) / total if total > 0 else float("nan")


def medir_equipo(x16: np.ndarray) -> dict:
    """Auditoría de la entrada (Entrenamiento, M04) y grumosidad en g y r (Preprocesamiento)."""
    v = {"huella": hashlib.sha1(np.ascontiguousarray(x16).tobytes()).hexdigest()[:16]}
    x = np.asarray(x16, np.float32)
    v["finito"] = bool(np.isfinite(x).all())
    v["banda_en_cero"] = bool(any((x[b] == 0).all() for b in range(x.shape[0])))
    if not v["finito"] or v["banda_en_cero"]:
        return {**v, "grumosidad_g": np.nan, "grumosidad_r": np.nan, "grumosidad_gauss_r": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            med_sig = [cielo(x[b]) for b in range(3)]
            sin_datos = x == 0
            sub = x - np.array([m for m, _ in med_sig], np.float32)[:, None, None]
            sub[sin_datos] = 0.0
            gal, vec, _ = segmentar(sub[1])
            for b, nombre in ((0, "g"), (1, "r")):
                limpio = rellenar_simetrico(sub[b], vec | sin_datos[b])
                v[f"grumosidad_{nombre}"] = grumosidad(limpio, gal, vec)
                if nombre == "r":      # definición original (filtro gaussiano), solo como diagnóstico
                    v["grumosidad_gauss_r"] = grumosidad(limpio, gal, vec, filtro="gaussiano")
        except Exception:      # una galaxia problemática no detiene el recorrido
            v["grumosidad_g"] = v["grumosidad_r"] = v["grumosidad_gauss_r"] = np.nan
    return v


def _tarea(x):
    return medir_equipo(x)


def recorrer_cache_equipo(cache_dir: Path, out_dir: Path, guard, ids_validos=None, workers: int | None = None,
                          log=print) -> pd.DataFrame:
    """Un recorrido por la caché con la auditoría y la grumosidad; un Parquet por bloque (reanudable)."""
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    guard.mkdir(out_dir)
    workers = workers or max(1, os.cpu_count() or 2)
    t0 = time.time()
    bloques = sorted(cache_dir.glob("chunk_*.npz"))
    for i, b in enumerate(bloques):
        f = out_dir / f"equipo_{b.stem}.parquet"
        if f.exists():
            continue
        with np.load(b) as z:
            x, ids = z["x"], z["mangaid"].astype(str)
        sel = np.ones(len(ids), bool) if ids_validos is None else np.isin(ids, list(ids_validos))
        with cf.ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_tarea, list(x[sel]), chunksize=8))
        tab = pd.DataFrame([{"mangaid": m, **r} for m, r in zip(ids[sel], res)])
        guard.write(f, lambda t, tab=tab: tab.to_parquet(t, index=False))
        log(f"[equipo] bloque {i + 1}/{len(bloques)} · {sel.sum()} galaxias · {time.time() - t0:.0f} s")
    return pd.concat([pd.read_parquet(t) for t in sorted(out_dir.glob("equipo_chunk_*.parquet"))], ignore_index=True)
