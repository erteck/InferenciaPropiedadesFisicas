"""Modelos alternativos del Avance 4: código común e independiente de la variante.

- `datos`: transformaciones ajustadas con entrenamiento, aumentación y conjuntos en la GPU.
- `contexto`: prepara la muestra y los conjuntos desde Drive.
- `arquitecturas`: especificación declarativa del modelo, extractor de timm, cabeza y estrategias de ajuste.
- `entrenamiento`: bucle de entrenamiento, predicción y corridas guardadas.
- `representaciones`: extracción congelada y entrenamiento de la cabeza sobre vectores fijos.
- `busqueda`: búsqueda de hiperparámetros con Optuna.
- `evaluacion`: métricas del protocolo común e intervalos.
- `interpretacion`: oclusión de canales y regiones.
"""
