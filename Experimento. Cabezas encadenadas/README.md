# Experimento: cabezas encadenadas

Experimento posterior al modelo final del Avance 5. Prueba si encadenar las cabezas de regresión como Katachi (la masa estimada entra a la cabeza de la SFR, y ambas a la de D4000) mejora a los tres miembros del ensamble y a la mezcla convexa.

| Archivo | Contenido | Dónde se ejecuta |
|---|---|---|
| `Cabezas_Encadenadas_DINOv3_ConvNeXt_EfficientNet.ipynb` | Entrena DINOv3 con LoRA, ConvNeXt-Tiny y EfficientNet-B3 con cabezas en cadena (tres semillas, misma configuración del Avance 5), los compara con sus versiones de cabezas independientes, reconstruye la mezcla convexa y aplica la regla de selección | Colab con GPU A100 (unos 110 minutos de entrenamiento) |

## Interruptores

| Variable | Valores | Efecto |
|---|---|---|
| `ENCADENAR` | `True` (por defecto) o `False` | `True` entrena las cabezas en cadena; `False` recupera las redes del Avance 5 sin entrenar y debe reproducir sus predicciones |
| `CON_GRADIENTE` | `False` (por defecto) o `True` | Si el error de la SFR y del D4000 también corrige a las cabezas anteriores de la cadena |

En el código, la opción es `EspecificacionModelo(encadenar_salidas=True, cadena_con_gradiente=False)` en `src/modelos_alternativos/arquitecturas.py`. Con la opción apagada, el modelo y la huella de cada corrida son idénticos a los del Avance 5.

## Salidas

Las corridas se guardan en Drive junto a las del Avance 5 (`resultados_modelos_alternativos/avance5_modelos_base/corridas/`, con el sufijo `_cadena`). Las predicciones, el resumen de los ensambles y el registro de la elección se guardan en `resultados_modelos_alternativos/experimento_cabezas_encadenadas/` y se descargan en un zip al terminar.
