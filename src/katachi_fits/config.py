"""Configuración del experimento.

TODOS los valores importantes están en el bloque de CONSTANTES de abajo.
Cambiar una constante aquí la cambia en todo el código. El cuaderno muestra en su
§2 las que normalmente se ajustan (recorte, β, semilla, variante, D1).

La receta de entrenamiento viene de la auditoría del código, los checkpoints y
los DataLoaders publicados de Katachi (plan §3.4): no cambiarla sin documentarlo.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# =============================================================================
# CONSTANTES
# =============================================================================

# --- Rutas (Google Drive) ------------------------------------------------------
# Estructura esperada en "Mi unidad":
#   descarga_22fb149046f162f3/plan.csv, inventario.csv, configuracion.json
#   descarga_22fb149046f162f3/fits/*.fits
#   descarga_22fb149046f162f3/catalogo_candidato_9661_galaxias.csv
#   resultados_katachi_fits/            ← se crea; ÚNICA carpeta donde se escribe
DRIVE_BASE = Path("/content/drive/MyDrive")
DOWNLOAD_DIRNAME = "descarga_22fb149046f162f3"     # carpeta del descargador (plan.csv, fits/, catálogo)
RESULTS_DIRNAME = "resultados_katachi_fits"
# Catálogo maestro del equipo (versión 20260925T190214_499654Z). Es el catálogo que
# leyó el descargador: su SHA-256 coincide con `catalogo_sha256` de configuracion.json.
TEAM_CATALOG_NAMES = ("catalogo_candidato_9661_galaxias.csv", "catalogo_candidatas_original.csv")
TEAM_CATALOG_SHA256 = "b7fbb3be6a4da4d69b5feb8a0fe740c9eff59c7e800dc8705ccd7ef82439a8ae"

# --- Imágenes FITS (variable experimental, plan §3.3) ------------------------
FITS_SIDE_PX = 800              # lado de los cubos descargados
DESI_PIXSCALE_ARCSEC = 0.262    # escala nativa de DESI DR10
CROP_PX = 300                   # recorte central (decisión del equipo): 300 px = 78.6″
OUTPUT_PX = 300                 # tamaño de entrada a la red; = CROP_PX → sin remuestreo
BANDS = ("g", "r", "z")         # orden en el FITS (BAND0..2); se verifica en cada archivo
ASINH_BETA = 3.0                # estiramiento asinh(x / (β σ_b))
SCALE_PERCENTILE = 99.9         # s_b: percentil de Train tras el estiramiento
SKY_RING_PX = 300.0             # σ_b se mide en r > 300 px del campo de 800 px
N_SKY_GALAXIES = 500            # galaxias de Train para medir σ_b
ZERO_FLAG_FRACTION = 0.05       # > 5 % de ceros en el recorte → "cobertura incompleta"
STORAGE_DTYPE = "float16"       # error máx. en la entrada ~1e-4 (40× menor que 1/255)

# --- Cortes de la muestra ----------------------------------------------------
CUT_LOG_MSTAR = (8.0, 12.0)     # Katachi: 8 < log M* < 12
CUT_LOG_SFR = (-4.0, 2.0)       # Katachi: −4 < log SFR < 2  (y D4000 finito)
# Decisión del equipo: solo QCFLAG = 0 (grupo OK de Pipe3D). WARNING y BAD quedan
# fuera del experimento, registrados en exclusiones.csv. Katachi no usaba este filtro.
QC_GROUPS_INCLUDED = ("OK",)

# --- Receta de entrenamiento: el artículo de Katachi §3.1.2–3.1.3 -------------
# Justificación de cada valor: Documentación/Propuesta_congelado_Katachi.md (D1–D9).
SEED = 42
BATCH_SIZE = 32
LR_MSTAR, LR_SFR, LR_D4000 = 1e-3, 1e-3, 1e-3   # D6: "a learning rate of 10⁻³ for all three networks"
SCHEDULER_PATIENCE = 3          # D7: ReduceLROnPlateau, uno por red
SCHEDULER_SIGNAL = "train"      # D7: pérdida de entrenamiento de cada red (el artículo no usa validación)
PER_NETWORK = True              # D7–D8: planificador y parada independientes por red
STOP_PRECISION = 1e-3           # D8: mejora relativa mínima (EarlyStopper de los autores)
STOP_PATIENCE = 10
# Congelado (D1–D3). FREEZE_PRETRAINED = False reproduce lo que muestran los pesos
# publicados (todo entrenable): sirve como corrida de control.
FREEZE_PRETRAINED = True
FIRST_TRAINABLE_PARAM = 144     # D1: umbral del código de los autores → últimas 5 conv + lineal
FREEZE_BN_STATS = True          # D2: BatchNorm congelada en modo evaluación
# D4: orden de canales a la entrada de la red. Los filtros congelados de ImageNet
# esperan R, G, B; Katachi usaba R = i, G = r, B = g (la banda más roja en R).
INPUT_BAND_ORDER = ("z", "r", "g")
MAX_EPOCHS = 1000
GHOST_BN_SPLITS = 2             # Katachi entrenó con nn.DataParallel en 2 GPUs
ROTATION_DEGREES = 360.0        # RandomRotation((0, 360)), vecino más cercano, relleno 0

# --- Particiones --------------------------------------------------------------
# Se usa `split_proyecto` del catálogo del equipo tal como está (Train 7,390 /
# Validation 1,302 / Test 969; con QCFLAG = 0: 7,123 / 1,262 / 918). Test = Test de Katachi.
# VAL_FRACTION solo se usa si no hay catálogo del equipo (respaldo, no recomendado).
VAL_FRACTION = 0.15
VAL_SEED = 42

# --- Evaluación (plan §1 y §4) -----------------------------------------------
EQUIVALENCE_MARGINS = {"log_mstar": 0.02, "log_sfr": 0.02, "d4000": 0.01}
N_BOOTSTRAP = 10_000
NOISE_ITERATIONS = 100          # bootstrap de ruido de Katachi
NOISE_BORDER_PX = 5             # σ del ruido = desviación estándar del borde de 5 px
SSFR_CUT = -10.8                # SFR también se reporta con sSFR > −10.8
N_SHAP_GALAXIES = 200
SHAP_BACKGROUND = 32            # Katachi usó un lote de 32 imágenes como fondo
SHAP_NSAMPLES = 200             # valor por defecto de shap.GradientExplainer
SHAP_SMOOTH_SIGMA_PX = 4.0
SHAP_GRADIENT_KPC = 20.0        # ∇ = perfil(20 kpc) − perfil(0)
KATACHI_KPC_ARCSEC_PER_PX = 0.5 # convención (errónea) de Katachi para pasar a kpc
KATACHI_PIXSCALE_ARCSEC = 0.195 # escala real de las imágenes de Katachi (50″ / 256 px)
T50_SCALE_GYR = 13.6            # salida de la red D4000→t50 de Katachi

# =============================================================================

KATACHI_URL = "https://www.astr.tohoku.ac.jp/~juanpabloalfonzo/Katachi_MaNGA_Catalogues"
PIPE3D_URL = "https://data.sdss.org/sas/dr17/env/MANGA_PIPE3D/v3_1_1/3.1.1/SDSS17Pipe3D_v3_1_1.fits"

KATACHI_FILES = {
    "scalars.cat": f"{KATACHI_URL}/scalars.cat",
    "images.cat": f"{KATACHI_URL}/images.cat",
    "Mass_ResNet50_log_t50_chain_90_10.pytorch": f"{KATACHI_URL}/models/Mass_ResNet50_log_t50_chain_90_10.pytorch",
    "SFR_ResNet50_log_t50_chain_90_10.pytorch": f"{KATACHI_URL}/models/SFR_ResNet50_log_t50_chain_90_10.pytorch",
    "d4000_d4000_chain.pytorch": f"{KATACHI_URL}/models/d4000_d4000_chain.pytorch",
    "d400_to_t50.pytorch": f"{KATACHI_URL}/models/d400_to_t50.pytorch",
    # Redes solo-imagen que Katachi usó para sus mapas SHAP de SFR y D4000.
    "sfr_solo.pytorch": f"{KATACHI_URL}/models/sfr_solo.pytorch",
    "d4000.pytorch": f"{KATACHI_URL}/models/d4000.pytorch",
    "SDSS17Pipe3D_v3_1_1.fits": PIPE3D_URL,
}

TARGETS = ("log_mstar", "log_sfr", "d4000")


@dataclass
class Paths:
    """Rutas del proyecto. Catálogo y FITS son de solo lectura."""

    base: Path = DRIVE_BASE
    plan_csv: Path | None = None
    inventario_csv: Path | None = None
    team_catalog: Path | None = None
    fits_dir: Path | None = None
    results: Path | None = None
    local: Path = Path("/content/katachi_local")

    def __post_init__(self) -> None:
        self.base = Path(self.base)
        self.plan_csv = Path(self.plan_csv or self.base / DOWNLOAD_DIRNAME / "plan.csv")
        self.inventario_csv = Path(self.inventario_csv or self.plan_csv.parent / "inventario.csv")
        self.fits_dir = Path(self.fits_dir or self.base / DOWNLOAD_DIRNAME / "fits")
        self.results = Path(self.results or self.base / RESULTS_DIRNAME)
        self.local = Path(self.local)
        if self.team_catalog is not None:
            self.team_catalog = Path(self.team_catalog)

    @property
    def read_only(self) -> list[Path]:
        """Carpetas y archivos que el código nunca debe modificar."""
        ro = [self.plan_csv, self.inventario_csv, self.fits_dir, self.fits_dir.parent]
        # Solo el archivo: su carpeta (la raíz del proyecto en Drive) contiene la carpeta de resultados.
        return ro + ([self.team_catalog] if self.team_catalog else [])

    @property
    def external(self) -> Path:
        return self.results / "externos"

    @property
    def cache(self) -> Path:
        return self.results / "cache"

    def run_dir(self, run_id: str) -> Path:
        return self.results / "runs" / run_id


@dataclass
class FitsConfig:
    """Paso FITS → tensor (plan §3.3). Valores por defecto = CONSTANTES."""

    side_in: int = FITS_SIDE_PX
    crop_px: int = CROP_PX
    out_px: int = OUTPUT_PX
    pixscale_in: float = DESI_PIXSCALE_ARCSEC
    beta: float = ASINH_BETA
    scale_percentile: float = SCALE_PERCENTILE
    sky_ring_px: float = SKY_RING_PX
    n_sky_galaxies: int = N_SKY_GALAXIES
    zero_flag_frac: float = ZERO_FLAG_FRACTION
    bands: tuple[str, ...] = BANDS
    storage_dtype: str = STORAGE_DTYPE

    def __post_init__(self) -> None:
        if not 0 < self.crop_px <= self.side_in:
            raise ValueError(f"crop_px={self.crop_px} fuera de rango")
        if (self.side_in - self.crop_px) % 2:
            raise ValueError("crop_px debe tener la misma paridad que side_in para quedar centrado en CRPIX")

    @property
    def crop_lo(self) -> int:
        """Primer píxel del recorte; con lados de igual paridad queda centrado
        exactamente en CRPIX = 400.5 (1-based) = 399.5 (0-based)."""
        return (self.side_in - self.crop_px) // 2

    @property
    def pixscale_out(self) -> float:
        return self.crop_px * self.pixscale_in / self.out_px

    @property
    def field_arcsec(self) -> float:
        return self.crop_px * self.pixscale_in

    @property
    def resamples(self) -> bool:
        return self.out_px != self.crop_px


@dataclass
class TrainConfig:
    """Receta auditada de Katachi (plan §3.4). Valores por defecto = CONSTANTES."""

    seed: int = SEED
    batch_size: int = BATCH_SIZE
    lr: tuple[float, float, float] = (LR_MSTAR, LR_SFR, LR_D4000)
    scheduler_patience: int = SCHEDULER_PATIENCE
    scheduler_signal: str = SCHEDULER_SIGNAL
    per_network: bool = PER_NETWORK
    freeze_pretrained: bool = FREEZE_PRETRAINED
    first_trainable_param: int = FIRST_TRAINABLE_PARAM
    freeze_bn_stats: bool = FREEZE_BN_STATS
    input_band_order: tuple[str, ...] = INPUT_BAND_ORDER
    stop_precision: float = STOP_PRECISION
    stop_patience: int = STOP_PATIENCE
    max_epochs: int = MAX_EPOCHS
    ghost_bn_splits: int = GHOST_BN_SPLITS
    rotation_degrees: float = ROTATION_DEGREES
    val_fraction: float = VAL_FRACTION
    val_seed: int = VAL_SEED
    # Optimizaciones de velocidad que no cambian el método (plan §3.5).
    fused_adam: bool = True
    channels_last: bool = True
    compile: bool = False
    cudnn_benchmark: bool = True


@dataclass
class MejorasConfig:
    """Mejoras de preparación de la imagen, aumentación y velocidad.

    Todas vienen apagadas: con los valores por defecto se reproduce la corrida 2
    (todas las capas entrenables). Justificación de cada una en
    Documentación/Propuesta_mejoras_preprocesamiento_DESI.md.
    """
    # --- Preparación de la imagen ---
    margen_rotacion: bool = False      # caché con margen: girar sin esquinas vacías y recortar después
    rotacion_bilineal: bool = False    # interpolación bilineal en lugar de vecino más cercano
    estandarizar_imagen: bool = False  # media 0 y desviación 1 por banda (estadísticas de entrenamiento)
    estandarizar_planos: bool = False  # planos de masa y SFR estandarizados en la cadena
    sigma_por_imagen: bool = False     # asinh con el σ del cielo de cada imagen, no uno global
    # --- Aumentación (solo al entrenar) ---
    desplazamiento_px: int = 0         # desplazamiento aleatorio de hasta ±N píxeles (máx. 5 con margen)
    # La predicción con TTA (promedio de las 8 simetrías) no es una opción: el ejecutor de
    # experimentos siempre reporta las dos versiones, con y sin TTA.
    ruido_max: float = 0.0             # ruido gaussiano añadido: k·σ_cielo, k ~ U(0, ruido_max)
    psf_max_px: float = 0.0            # desenfoque gaussiano σ ~ U(0, psf_max_px) píxeles
    prob_aumentacion: float = 0.5      # probabilidad de aplicar ruido y desenfoque a cada imagen
    # --- Velocidad y programa de entrenamiento ---
    precision_mixta: bool = False      # autocast en bfloat16 (solo GPU)
    programa_rapido: bool = False      # coseno con calentamiento y número fijo de épocas
    epocas_rapido: int = 30
    calentamiento_epocas: int = 2

    @property
    def margen_px(self) -> int:
        """Lado del recorte guardado en caché cuando hay margen de rotación."""
        # 300·√2 = 424.3 px cubre cualquier giro; +2·5·√2 permite desplazar hasta 5 px.
        # 440 tiene la misma paridad que 800, así que el recorte queda centrado en CRPIX.
        return 440 if self.margen_rotacion else 0

    def sufijo(self) -> str:
        """Sufijo del identificador de la corrida: solo las opciones encendidas."""
        partes = []
        if self.margen_rotacion: partes.append("margen")
        if self.rotacion_bilineal: partes.append("bilin")
        if self.estandarizar_imagen: partes.append("stdimg")
        if self.estandarizar_planos: partes.append("stdplanos")
        if self.sigma_por_imagen: partes.append("sigimg")
        if self.desplazamiento_px: partes.append(f"desp{self.desplazamiento_px}")
        if self.ruido_max: partes.append(f"ruido{self.ruido_max:g}")
        if self.psf_max_px: partes.append(f"psf{self.psf_max_px:g}")
        if self.precision_mixta: partes.append("bf16")
        if self.programa_rapido: partes.append(f"rapido{self.epocas_rapido}")
        return ("_" + "_".join(partes)) if partes else ""


@dataclass
class Experiment:
    """Agrupa rutas, configuración FITS y receta; se guarda como `config.json` de cada corrida."""
    paths: Paths = field(default_factory=Paths)
    fits: FitsConfig = field(default_factory=FitsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mejoras: MejorasConfig = field(default_factory=MejorasConfig)

    def to_json(self) -> str:
        def conv(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o

        return json.dumps(asdict(self), default=conv, indent=2, ensure_ascii=False)
