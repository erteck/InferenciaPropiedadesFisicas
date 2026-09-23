# Datos y versionado con DVC

Los catálogos CSV y las imágenes FITS de prueba en esta carpeta se guardan
directamente en git. Todo lo que genera el pipeline de preprocesamiento
(`Datos/raw/`, `Datos/interim/`, `Datos/processed/`, `Datos/splits/`) se
versiona con [DVC](https://dvc.org): git guarda solo los hashes en `dvc.lock`
y los bytes viven en el remoto de DVC.

## Estructura

```
Datos/
├── *.csv                      Catálogos de entrada (git)
├── UNET/*.fits                Imágenes de prueba (git)
├── raw/                       manifest.csv + fits/ crudos          (DVC)
├── interim/                   cutouts/ normalized/ masks/ structure_maps/ (DVC)
├── processed/                 train.zarr val.zarr test.zarr        (DVC)
└── splits/                    train.csv val.csv test.csv           (DVC)
```

Los archivos `*_summary.json` de cada carpeta son métricas de la etapa y sí
entran en git, para poder comparar versiones con `git diff` sin descargar datos.

## Pipeline

Definido en `dvc.yaml`; los parámetros están en `params.yaml`; el código en
`src/data/`.

| Etapa       | Entrada                      | Salida                          |
| ----------- | ---------------------------- | ------------------------------- |
| `ingest`    | catálogos CSV, FITS locales  | `raw/manifest.csv`, `raw/fits/` |
| `cutout`    | `raw/fits/`                  | `interim/cutouts/`              |
| `normalize` | `interim/cutouts/`           | `interim/normalized/`           |
| `segment`   | `interim/cutouts/`           | `interim/masks/`                |
| `structure` | normalized + masks           | `interim/structure_maps/`       |
| `split`     | manifest + interim           | `processed/*.zarr`, `splits/`   |

`dvc repro` ejecuta solo las etapas cuyas dependencias o parámetros cambiaron.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

o con conda: `conda env create -f environment.yml && conda activate ipf`.

## Flujo de trabajo

Cambiar el preprocesamiento y guardar una versión:

```bash
vim params.yaml                         # p. ej. normalize.method: log
dvc repro
git add params.yaml dvc.lock Datos
git commit -m "feat(data): Probar normalización log"
dvc push
```

Marcar una versión importante (baseline, entregable):

```bash
git tag -a dataset/v0.3-baseline -m "Dataset del baseline, semana 5"
git push origin main --tags
```

Recuperar una versión exacta:

```bash
git checkout dataset/v0.3-baseline
dvc pull
```

Ver qué cambió entre dos versiones sin descargar nada:

```bash
git diff dataset/v0.1-asinh dataset/v0.2-log -- params.yaml Datos/*/*_summary.json
dvc params diff dataset/v0.1-asinh dataset/v0.2-log
```

Probar una variante sin ensuciar el historial:

```bash
dvc exp run -S normalize.method=lupton -S cutout.radius_arcsec=60
dvc exp show
```

## Remoto

El remoto por defecto (`local_test`) es una carpeta local en
`.dvc_remote_local/` en la raíz del repo, ignorada por git. Sirve para
probar el flujo en una sola máquina. Para compartir datos entre el equipo hay
que apuntar a un remoto común:

```bash
# Servidor SSH (p. ej. IA-UNAM)
dvc remote add -d iaunam ssh://usuario@host/ruta/dvcstore

# Backblaze B2 / Cloudflare R2 (compatibles con S3)
dvc remote add -d b2 s3://bucket/dvcstore
dvc remote modify b2 endpointurl https://s3.us-west-004.backblazeb2.com
dvc remote modify --local b2 access_key_id ...
dvc remote modify --local b2 secret_access_key ...
```

Al cambiar de remoto basta con `dvc push`; el historial de git no cambia.
Las credenciales van siempre con `--local` para que no entren en git.

## Versiones publicadas

| Tag                   | Cambio                          |
| --------------------- | ------------------------------- |
| `dataset/v0.1-asinh`  | Pipeline inicial, M51, asinh    |
| `dataset/v0.2-log`    | Normalización log               |
