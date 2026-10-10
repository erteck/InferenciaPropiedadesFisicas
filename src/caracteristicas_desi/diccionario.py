"""Diccionario de características: definición, unidades, motivo físico y referencia.

Los grupos definen los conjuntos que se comparan en el cuaderno:
    aparente    lo que se mide en la imagen sin conocer la distancia
    intrinseca  lo que requiere el redshift (distancia)
    estructura  tamaño y distribución de la luz
    contexto    condiciones de observación (posibles fuentes de sesgo)
"""
from __future__ import annotations

import pandas as pd

_B = {"g": "g", "r": "r", "z": "z"}

FILAS = [
    # --- aparente ---------------------------------------------------------------
    *[(f"m_{b}0", "aparente", f"Magnitud de Petrosian en {b} (apertura 2 R_P medida en r), corregida por extinción galáctica",
       "mag", "Brillo total de la galaxia en esa banda", "Blanton et al. 2001; Schlafly & Finkbeiner 2011")
      for b in _B],
    ("g_r", "aparente", "m_g0 − m_r0", "mag", "Color: edad de la población estelar y polvo; principal indicador de M/L",
     "Bell et al. 2003"),
    ("r_z", "aparente", "m_r0 − m_z0", "mag", "Color más sensible a estrellas viejas y al polvo", "Bell et al. 2003"),
    ("g_z", "aparente", "m_g0 − m_z0", "mag", "Color de base larga (combinación lineal de g−r y r−z)", "—"),
    *[(f"{c}_in", "aparente", f"Color {c.replace('_', '−')} dentro de R50", "mag", "Color del núcleo (bulbo)", "—")
      for c in ("g_r", "r_z")],
    *[(f"{c}_out", "aparente", f"Color {c.replace('_', '−')} en el anillo R50–R90", "mag", "Color del disco", "—")
      for c in ("g_r", "r_z")],
    *[(f"delta_{c}", "aparente", f"{c}_in − {c}_out", "mag",
       "Gradiente de color: núcleo más rojo que el disco indica apagado desde dentro (inside-out)", "—")
      for c in ("g_r", "r_z")],
    *[(f"mu50_{b}", "aparente", f"Brillo superficial medio dentro de R50 en {b}: m + 2.5 log10(2π R50²)",
       "mag/arcsec²", "Separa discos difusos de sistemas compactos", "Blanton et al. 2001") for b in _B],
    # --- intrínseca ----------------------------------------------------------------
    ("z", "intrinseca", "Redshift espectroscópico de MaNGA", "—", "Distancia; entra en todas las variables intrínsecas", "—"),
    ("DM", "intrinseca", "Módulo de distancia, ΛCDM plana H0 = 73, Ωm = 0.3", "mag",
     "Convierte lo aparente en intrínseco", "Sánchez et al. 2022 (cosmología de Pipe3D)"),
    *[(f"M_{b}", "intrinseca", f"Magnitud absoluta en {b}: m_{b}0 − DM (sin corrección K)", "mag", "Luminosidad real", "—")
      for b in _B],
    ("log_L_r", "intrinseca", "log10 de la luminosidad en r: 0.4 (4.65 − M_r)", "log L☉", "Luminosidad en unidades solares",
     "Willmer 2018"),
    ("log_R50_kpc", "intrinseca", "log10 de R50 en kpc", "log kpc", "Tamaño físico", "—"),
    ("log_SigmaL", "intrinseca", "log L_r − log(2π R50²)", "log L☉/kpc²", "Densidad de luminosidad", "—"),
    ("logM_ebrova", "intrinseca", "0.673 M_g − 1.108 M_r + 0.996", "log M☉",
     "Masa fotométrica calibrada para Legacy Surveys (dispersión 25 %)", "Ebrová et al. 2025"),
    ("logM_bell", "intrinseca", "log L_r − 0.306 + 1.097 (g − r)", "log M☉", "Masa por la relación color–M/L",
     "Bell et al. 2003"),
    ("log_SigmaM", "intrinseca", "logM_ebrova − log(2π R50²)", "log M☉/kpc²",
     "Densidad superficial de masa, ligada al apagado de la formación estelar", "—"),
    # --- estructura -------------------------------------------------------------------
    ("R_P", "estructura", "Radio de Petrosian en r (η = 0.2)", "arcsec", "Escala de tamaño independiente de la profundidad",
     "Blanton et al. 2001"),
    *[(n, "estructura", f"Radio que contiene el {p} % del flujo de Petrosian en r", "arcsec", "Tamaño aparente",
       "Blanton et al. 2001") for n, p in (("R20", 20), ("R50", 50), ("R80", 80), ("R90", 90))],
    ("C", "estructura", "5 log10(R80/R20), con el flujo dentro de 1.5 R_P", "—", "Concentración: bulbo frente a disco",
     "Conselice 2003"),
    ("C_sdss", "estructura", "R90/R50", "—", "Concentración de SDSS; separa tempranas (> 2.6) de tardías",
     "Strateva et al. 2001"),
    ("A", "estructura", "Asimetría de rotación de 180°, con el término de ruido restado", "—",
     "Perturbaciones, interacciones y formación estelar irregular", "Conselice 2003"),
    ("S", "estructura", "Suavidad: luz en estructuras de escala < 0.25 R_P", "—", "Formación estelar grumosa (regiones HII)",
     "Conselice 2003"),
    ("gini", "estructura", "Coeficiente de Gini de la distribución de flujo", "—",
     "Cuán concentrada está la luz en pocos píxeles", "Lotz et al. 2004"),
    ("m20", "estructura", "Segundo momento del 20 % más brillante del flujo, relativo al total", "—",
     "Luz brillante concentrada en el centro (bulbo) o repartida (nudos, fusiones)", "Lotz et al. 2004"),
    ("b_a", "estructura", "Cociente de ejes a partir de los momentos de segundo orden", "—",
     "Inclinación: modula el polvo y el color observado", "—"),
    ("ratio_R50_g_r", "estructura", "R50 en g / R50 en r", "—", "Gradiente de color expresado como tamaño",
     "—"),
    ("ratio_R50_z_r", "estructura", "R50 en z / R50 en r", "—", "Gradiente de color expresado como tamaño",
     "—"),
    *[(f"grumosidad_{b}", "estructura",
       f"Fracción de la luz en {b} en estructuras de menos de 17 px (4.5″): Σ max(I − mediana₁₇(I), 0) / Σ I, "
       "fuera del núcleo (r ≥ 16 px) y con el ruido del cielo restado", "—",
       "Grumos de formación estelar reciente (cúmulos jóvenes en los brazos)",
       "Preprocesamiento del equipo; Conselice 2003") for b in ("g", "r")],
    # --- contexto ------------------------------------------------------------------
    ("region_norte", "contexto", "1 si Dec ≥ 32.375° (BASS/MzLS), 0 si es DECam", "—",
     "Otro telescopio, otros filtros y otra profundidad", "Dey et al. 2019"),
    ("ebv", "contexto", "E(B−V) del mapa SFD en la posición de la galaxia", "mag", "Polvo de la Vía Láctea",
     "Schlegel et al. 1998"),
    *[(f"sigma_cielo_{b}", "contexto", f"σ del cielo residual en {b}", "nmgy/px", "Profundidad local de la imagen", "—")
      for b in _B],
    ("frac_vecinos", "contexto", "Fracción de la apertura ocupada por fuentes vecinas", "—",
     "Contaminación por vecinos", "—"),
    ("n_fuentes", "contexto", "Número de fuentes vecinas detectadas en el recorte", "—", "Densidad del entorno proyectado",
     "—"),
    ("frac_sin_datos", "contexto", "Fracción máxima de píxeles sin datos entre las tres bandas", "—", "Cobertura incompleta",
     "—"),
    ("apertura_truncada", "contexto", "1 si 2 R_P no cabe en el recorte de 78.6″", "—",
     "La galaxia es más grande que el campo", "—"),
]

DICCIONARIO = pd.DataFrame(FILAS, columns=["variable", "grupo", "definicion", "unidades", "motivo", "referencia"])


def grupo(nombre: str) -> list[str]:
    return DICCIONARIO.loc[DICCIONARIO.grupo == nombre, "variable"].tolist()
