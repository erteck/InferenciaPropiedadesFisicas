"""Ingeniería de características sobre imágenes DESI (Avance 2).

Módulos:
    config           constantes (todas las decisiones numéricas en un solo lugar)
    fotometria       cielo, segmentación, curva de crecimiento, Petrosian, radios
    morfologia       C, A, S, Gini, M20 y forma
    fisica           extinción, distancias y variables intrínsecas
    imagenes         estampas para eigengalaxias y RGB para Zoobot
    pipeline         recorre la caché y guarda resultados reanudables
    transformaciones regla de asimetría (log, Box-Cox, Yeo-Johnson)
    diccionario      definición, unidades, motivo y referencia de cada característica
"""
