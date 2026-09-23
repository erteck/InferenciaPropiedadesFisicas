# Inferencia de propiedades físicas de galaxias a partir de imágenes multibanda mediante aprendizaje supervisado interpretable

**Instituto Tecnológico y de Estudios Superiores de Monterrey**
Escuela de Ingeniería y Ciencias — Maestría en Inteligencia Artificial Aplicada
Proyecto Integrador

Repositorio: <https://github.com/erteck/InferenciaPropiedadesFisicas>

---

## Descripción del proyecto

Desarrollamos y evaluamos un prototipo de aprendizaje supervisado para
estimar la **masa estelar**, la **tasa de formación estelar (SFR)** y el
**índice D4000** (usado como indicador de las características de la
población estelar) a partir de **imágenes multibanda del DESI Legacy
Survey**. Tomamos como punto de partida el framework **Katachi**, cuya
arquitectura adaptamos y reentrenamos con estas imágenes, utilizando como
etiquetas de referencia las estimaciones derivadas del análisis
espectroscópico de **MaNGA** mediante **Pipe3D**.

Sobre esa línea base, incorporamos una etapa de **preprocesamiento y
segmentación** orientada a aislar la galaxia objetivo, conservar su luz
detectable y reducir la contaminación de fuentes vecinas. También
evaluamos **representaciones de las estructuras internas** de la galaxia
como entradas adicionales del modelo. Mediante experimentos controlados
determinamos si estos componentes y las alternativas de arquitectura
mejoran la predicción. El propósito final es apoyar la caracterización de
galaxias a partir de imágenes y evaluar la coherencia física de las
predicciones mediante herramientas de interpretabilidad y revisión
científica.

## Objetivo

Predecir **masa estelar**, **SFR** e **índice D4000** mediante regresión
supervisada con múltiples salidas continuas, a partir de imágenes
multibanda del DESI Legacy Survey y usando etiquetas de referencia de
MaNGA/Pipe3D. El procesamiento de imágenes, la segmentación y el análisis
de estructuras internas se incorporan como componentes de apoyo a la
tarea predictiva principal.

**Dominio de aplicación:** modelos predictivos (aprendizaje supervisado).

**Sector industrial:** Sector 54 — Servicios profesionales, científicos y
técnicos. Clase SCIAN México 2023 **54171**: Servicios de investigación
científica y desarrollo en ciencias naturales y exactas, ingeniería y
ciencias de la vida.

**Institución beneficiaria:** Instituto de Astronomía de la Universidad
Nacional Autónoma de México (IA-UNAM), Ciudad Universitaria, Ciudad de
México.

## Integrantes

| Integrante                        | Matrícula   |
| --------------------------------- | ----------- |
| Luz Copelia Minutti Pérez         | A01796921   |
| José Florencio Maguey Peralta     | A01796727   |
| Erick Alberto Bustos Cruz         | A01378966   |

### Asesor del Proyecto Integrador — ITESM

- **Dr. Iván Reyes Amezcua** — Investigador Posdoctoral, Profesor titular
  de MLOps (MNA). <reyes.ivan@tec.mx>

### Asesores científicos — Instituto de Astronomía, UNAM

- **Dr. Héctor Hernández Toledo** — Investigador titular B, IA-UNAM.
  <hector@astro.unam.mx>
- **Dr. José Antonio Vázquez Mata** — Investigador posdoctoral, IA-UNAM.
  <jvazquez@astro.unam.mx>

## Estructura del repositorio

```
InferenciaPropiedadesFisicas/
├── README.md              ← Este archivo
├── requirements.txt       ← Dependencias (pip); environment.yml para conda
├── dvc.yaml               ← Pipeline de preprocesamiento (etapas DVC)
├── params.yaml            ← Parámetros de cada etapa
├── dvc.lock               ← Hashes de datos por versión (generado por DVC)
├── src/
│   └── data/              ← Código de las etapas del pipeline
├── Datos/                 ← Datos utilizados en el proyecto (ver Datos/README.md)
│   ├── color_morp.csv
│   ├── listado_Katachi_MaNGA_DR17_coordenadas_oficiales.csv
│   ├── metricas_cross_target.csv
│   ├── UNET/              ← Imágenes FITS para experimentos con U-Net
│   │   ├── m51.fits
│   │   └── SSDS_M51.fits
│   ├── raw/               ← Manifiesto e imágenes crudas (DVC)
│   ├── interim/           ← Recortes, normalizados, máscaras, mapas (DVC)
│   ├── processed/         ← Cubos Zarr train/val/test (DVC)
│   └── splits/            ← Particiones fijas (DVC)
├── Notebooks/             ← Análisis, experimentación y desarrollo
│   ├── Katachi_Comparacion_Masas_Pipe3D_Taylor_NSA (1).ipynb
│   ├── Katachi_Entrenamiento_Taylor_NoTrain.ipynb
│   ├── Katachi_Entrenamiento_Taylor_Trained.ipynb
│   ├── Katachi_Masa_Morfologia_Color_Analisis_Completo.ipynb
│   ├── Metricas masa análisis.ipynb
│   ├── _build_taylor_notebook.py
│   ├── UNET/              ← Notebooks de experimentos con U-Net
│   └── framework-katachi/ ← Copia del framework Katachi (notebooks + módulos)
└── Documentación/         ← Documentos y materiales de apoyo
    ├── UNET.pdf
    ├── UNET Attention.pdf
    ├── pspooling.pdf
    ├── Papers/            ← Artículos y referencias bibliográficas
    └── Entregables/       ← Entregables por semana
        ├── Semana_1/
        ├── Semana_2/
        ├── Semana_8/
        └── Semana_9/
```

### Descripción de cada carpeta

- **Datos/** — Conjuntos de datos utilizados en el proyecto: catálogos en
  CSV (coordenadas oficiales del listado Katachi–MaNGA DR17, métricas
  cruzadas, color y morfología) e imágenes en formato FITS empleadas en los
  experimentos con U-Net. Los datos derivados del preprocesamiento
  (`raw/`, `interim/`, `processed/`, `splits/`) se versionan con DVC y no
  entran en git; cada commit o tag `dataset/vX.Y` identifica una versión
  exacta del dataset. El flujo completo está en `Datos/README.md`.
- **src/data/** — Etapas del pipeline de datos (`ingest`, `cutout`,
  `normalize`, `segment`, `structure`, `split`), orquestadas por `dvc.yaml`
  y parametrizadas en `params.yaml`.
- **Notebooks/** — Notebooks de Jupyter para análisis, experimentación y
  desarrollo del proyecto. Incluye los notebooks principales del análisis
  Katachi/Taylor/Pipe3D, la subcarpeta `UNET/` con experimentos de
  segmentación de imágenes y `framework-katachi/` con una copia local del
  framework (notebooks y módulos `.py` asociados).
- **Documentación/** — Documentos y materiales relevantes para el
  desarrollo y seguimiento del proyecto: PDFs propios (UNET, UNET
  Attention, pspooling), la carpeta `Papers/` con referencias bibliográficas
  y `Entregables/` con las subcarpetas de las semanas 1, 2, 8 y 9 donde se
  depositarán los entregables comprometidos.

> La estructura podrá evolucionar conforme avance el proyecto; esta versión
> constituye la organización inicial clara y funcional del repositorio.

## Plan de entregables

Las semanas corresponden al calendario del curso.

| Sem. | Entregable de referencia del curso    | Entregable comprometido para el proyecto |
| :--: | ------------------------------------- | ---------------------------------------- |
| 1    | Planteamiento del proyecto            | Planteamiento del proyecto: problema, beneficiario, objetivos y alcance. |
| 2    | Avance 0. Propuesta de proyecto       | Avance 0. Protocolo experimental y responsabilidades del equipo. |
| 3    | Avance 1. Análisis exploratorio de datos | Catálogo maestro MaNGA/Pipe3D asociado con imágenes DESI; diccionario de variables, análisis de etiquetas, cobertura, duplicados y calidad; piloto de recortes. |
| 4    | Avance 2. Ingeniería de características | Preparación de RGB y cubos de flujo; regla documentada de recorte y normalización; prototipo de máscaras y mapas internos con evaluación inicial de conservación de luz y contaminación. |
| 5    | Avance 3. Baseline                    | Línea base Katachi–DESI sin segmentación. Particiones fijas, entrenamiento reproducible y métricas de regresión por propiedad; documentación de diferencias frente a la reproducción con SDSS. |
| 6    | Avance 4. Modelos alternativos        | Comparación controlada de entradas con segmentación y estructuras internas. |
| 7    | Avance 5. Modelo final                | Selección mediante validación y evaluación; análisis de errores, variabilidad e interpretabilidad; pesos, código de inferencia y documentación de limitaciones. |
| 8    | Avance 6. Producto de difusión        | Producto de difusión. |
| 9    | Avance 7. Resumen ejecutivo           | Resumen ejecutivo: resultados, utilidad para el IA-UNAM, recursos necesarios y recomendaciones de continuidad. |
| 10–11 | Presentación final                   | Presentación final: exposición y demostración del prototipo, comparación experimental y entrega del código y documentación reproducible. |

En este repositorio se resguardan explícitamente los entregables de las
**semanas 1, 2, 8 y 9** en `Documentación/Entregables/`. El resto de los
avances se irán documentando conforme se produzcan.

---

*Fecha de entrega del planteamiento: 18 de septiembre de 2026.*
