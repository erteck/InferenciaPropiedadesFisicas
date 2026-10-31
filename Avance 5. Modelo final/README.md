# Avance 5. Modelo final

En esta etapa del proyecto exploramos modelos de ensamble para mejorar la predicción de propiedades físicas de galaxias a partir de sus imágenes. Partimos de los modelos individuales con mejor desempeño en la fase anterior y evaluamos distintas formas de combinar sus predicciones.

El trabajo incluye ensambles homogéneos y heterogéneos, optimización de hiperparámetros y una comparación con los modelos individuales. Analizamos el desempeño mediante métricas de regresión y tiempos de entrenamiento para determinar si las combinaciones ofrecen mejoras que justifiquen su complejidad computacional.

La selección del modelo final considera su precisión y capacidad de generalización en datos no vistos. Complementamos la evaluación con gráficos de predicciones y residuos para identificar sesgos, reconocer los rangos con mayores errores y discutir las limitaciones del modelo.

En esta carpeta se documentan los experimentos, las configuraciones y los resultados correspondientes a esta etapa.

## Archivos

| Archivo | Contenido | Dónde se ejecuta |
|---|---|---|
| `Ensambles_Homogeneos_ResNet50_D_E_vf.ipynb` | Ensambles homogéneos de ResNet50 (variantes D y E, tres semillas cada una) | Servidor con GPU |
| `Modelos_Base_DINOv3_ConvNeXt_EfficientNet.ipynb` | Entrenamiento de los miembros nuevos (DINOv3 y ConvNeXt con redshift, EfficientNet-B3) y exportación de las predicciones de todas las redes | Colab con GPU A100 |
| `Avance5.24.ipynb` | Entregable: ensambles heterogéneos, comparación, selección del modelo final y evaluación en prueba | Sin GPU |
| `predicciones/` | Predicciones de validación y de prueba de cada red, un archivo `pred_<familia>_s<semilla>.csv` por red | Datos de entrada de `Avance5.24.ipynb` |
