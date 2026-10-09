"""Constantes del Avance 2. Cada valor indica de dónde sale."""
from __future__ import annotations

# --- Imagen -----------------------------------------------------------------
BANDAS = ("g", "r", "z")             # orden en los FITS y en la caché
PIXSCALE = 0.262                     # ″/px, escala nativa de DESI DR10
RECORTE_PX = 300                     # recorte de la caché de Katachi (78.6″)
NANOMAGGIE_ZP = 22.5                 # m = 22.5 − 2.5 log10(flujo [nanomaggies])

# --- Cielo y segmentación ---------------------------------------------------
# Cielo residual: píxeles a más de este radio del centro (esquinas del recorte).
CIELO_R_MIN_PX = 127.0
# Máscara de la galaxia: umbral sobre la banda r suavizada (valores del equipo,
# params.yaml → mask_eval.threshold; "segmentar en r" es la conclusión de
# Notebooks/Evaluacion_mascaras_DESI.ipynb).
SEG_NSIGMA = 2.0
SEG_SUAVIZADO_PX = 2.0
SEG_CRECER_PX = 2
# Vecinos: fuentes detectadas a 3σ sin suavizar (src/data/mask_eval.neighbor_mask).
VEC_NSIGMA = 3.0
VEC_CRECER_PX = 3

# --- Petrosian (convención SDSS: Blanton et al. 2001; Strauss et al. 2002) ---
PETRO_ETA = 0.2                      # cociente de brillo superficial que define R_P
PETRO_ANILLO = (0.8, 1.25)           # anillo de 0.8 R a 1.25 R
PETRO_APERTURA = 2.0                 # flujo de Petrosian: dentro de 2 R_P
PASO_RADIAL_PX = 0.5                 # resolución de la curva de crecimiento

# --- Morfología (definiciones de statmorph; Rodriguez-Gomez et al. 2019) ------
CAS_EXTENSION = 1.5                  # C, A y S se miden dentro de 1.5 R_P
CAS_FRACCION = 0.25                  # S: caja de suavizado de 0.25 R_P; anillo interior
GINI_FRACCION = 0.2                  # Gini/M20: suavizado de 0.2 R_P para la segmentación
ASIM_DESPLAZAMIENTO = 3              # búsqueda del centro de asimetría: ±1.5 px en pasos de 0.5

# --- Física -------------------------------------------------------------------
# A_b / E(B−V) de Legacy Surveys DR10 (Schlafly & Finkbeiner 2011, calculados por
# E. Schlafly para DECam; BASS y MzLS se tratan en el sistema de DECam).
EXTINCION = {"g": 3.214, "r": 2.165, "z": 1.211}
# Cosmología de Pipe3D DR17 (Sánchez et al. 2022, nota 5): ΛCDM plana.
H0, OMEGA_M = 73.0, 0.3
# Magnitud absoluta del Sol en r (AB, SDSS; Willmer 2018).
M_SOL_R = 4.65
# Masa fotométrica calibrada para Legacy Surveys (Ebrová et al. 2025, ec. 1; IMF de Salpeter).
EBROVA = (0.673, -1.108, 0.996)
# M/L en r a partir de g − r (Bell et al. 2003, tabla 7; IMF "diet Salpeter").
BELL_R = (-0.306, 1.097)
# Frontera norte (BASS/MzLS) / sur (DECam) de Legacy Surveys.
DEC_NORTE = 32.375

# --- Clases para pruebas de clasificación -------------------------------------
CORTE_SSFR = -10.8                   # formadora / apagada (corte de Katachi)
CORTE_D4000 = 1.5                    # población joven / vieja

# --- Extracción ------------------------------------------------------------------
ESTAMPA_PX = 32                      # lado de las estampas para eigengalaxias
ESTAMPA_RADIOS = 2.0                 # la estampa cubre ±2 R_P

# --- Mapas SFD de E(B−V) (Schlegel, Finkbeiner & Davis 1998) --------------------
# Copia pública de los mapas originales (la misma que usa el paquete `sfdmap`).
SFD_URLS = {
    "SFD_dust_4096_ngp.fits": "https://raw.githubusercontent.com/kbarbary/sfddata/master/SFD_dust_4096_ngp.fits",
    "SFD_dust_4096_sgp.fits": "https://raw.githubusercontent.com/kbarbary/sfddata/master/SFD_dust_4096_sgp.fits",
}
