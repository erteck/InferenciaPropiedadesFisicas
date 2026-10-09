# Plan del Avance 2: ingeniería de características para imágenes DESI

**Estado:** propuesta para revisión. No se ha escrito código ni el cuaderno.
**Entregable:** `Notebooks/Avance2.24.ipynb` (en español), que se ejecuta de principio a fin en Colab.
**Fase de CRISP-ML(Q):** 2, *Data Engineering (Data Preparation)*.

---

## 1. Qué queremos lograr

Nuestros datos crudos son cubos FITS de DESI (g, r, z, en nanomaggies) y un catálogo con redshift y etiquetas de Pipe3D. Un modelo tabular no puede usar una imagen de 90,000 píxeles directamente, y una CNN usa todos esos píxeles sin saber nada de física. En este avance convertimos las imágenes en **un conjunto pequeño de variables con significado físico**, las transformamos y escalamos para que los modelos converjan y no las dominen sus colas, y seleccionamos y extraemos las más útiles.

Cada decisión se valida con una prueba. La pregunta central del cuaderno es:

> ¿Cuánto de la masa estelar, la SFR y el D4000 se puede predecir con unas decenas de características bien construidas, comparado con la CNN que usa la imagen completa?

**Referencias que ya tenemos (galaxias de validación):** la CNN de la segunda corrida obtiene RMSE de 0.224 dex en masa, 0.631 dex en SFR y 0.075 en D4000. Katachi publicado obtiene 0.222, 0.668 y 0.100 en sus galaxias de prueba.

---

## 2. Datos y reglas

| Elemento | Decisión |
|---|---|
| Muestra | Las 9,303 galaxias con QCFLAG = 0 (decisión del equipo): 7,123 de entrenamiento, 1,262 de validación y 918 de prueba |
| Imágenes | La caché de recortes de 300 × 300 px (78.6″) que ya construyó el cuaderno de Katachi en Drive. Si no existe, se reconstruye con el mismo código (~15 min) |
| Ajuste de todo | Transformaciones, escaladores, selección, PCA y FA **se ajustan solo con entrenamiento** |
| Evaluación | Solo validación. **Las galaxias de prueba no se usan**, igual que en el resto del proyecto |
| Etiquetas | log M\*, log SFR y D4000 de Pipe3D, sin transformar (sección 5.4) |

### Variables que **no** entran al modelo, y por qué

| Variable | Fuente | Motivo |
|---|---|---|
| `LogMass`, `log_SFR_Ha`, `log_SFR_ssp`, `log_age_mean_LW`, `log_ZH_mean_LW`, `vel_sigma_Re` | Pipe3D (`Datos/inferencia.fits`) | Salen del mismo ajuste espectral que las etiquetas: sería fuga de información |
| `LogMassNSA`, `LogMassTaylor`, `nsa_sersic_mass` | NSA / Taylor | Son otras estimaciones de la masa, casi la respuesta. **Solo se usan para validar** nuestro estimador fotométrico de masa |
| `error_log_mstar`, `error_log_sfr`, `error_d4000` | Pipe3D | Son metadatos de la etiqueta, no información de la galaxia |
| `QCFLAG` | Pipe3D | Es constante (0) en la muestra de modelado |
| `mangaid`, RA, Dec en bruto | Catálogo | Son identificadores. Las coordenadas crudas permitirían memorizar zonas del cielo; la información útil de la posición (región y extinción) se construye aparte |

---

## 3. Investigación: qué se usa con imágenes de galaxias

| Técnica | Fuente | Cómo la usamos |
|---|---|---|
| Fotometría en apertura de Petrosian (η = 0.2) y radios R20, R50, R80, R90 | Estándar de SDSS (Blanton et al. 2001) | Base de todas las magnitudes y tamaños |
| Concentración C = 5 log(R80/R20), asimetría A y suavidad S (CAS) | Conselice 2003 | Estructura: bulbo frente a disco, perturbaciones, formación estelar grumosa |
| Gini y M20 | Lotz et al. 2004; implementación de referencia en `statmorph` (Rodriguez-Gomez et al. 2019) | Distribución de la luz; separa tempranas de tardías y fusiones |
| R90/R50, con corte de 2.6 entre tempranas y tardías | Strateva et al. 2001 | Concentración comparable con SDSS |
| Relación color–masa/luminosidad | Bell et al. 2003: log(M/L_r) = −0.306 + 1.097 (g − r) | Estimador físico de masa |
| Masa fotométrica **calibrada para Legacy Surveys** | Ebrová et al. 2025 (A&A 704, A232): log M\* = 0.673 M_g − 1.108 M_r + 0.996, dispersión 25 % | Estimador de masa a partir de nuestras propias bandas |
| Coeficientes de extinción galáctica A/E(B−V) = 3.214, 2.165, 1.211 (g, r, z) | [Legacy Surveys DR10](https://www.legacysurvey.org/dr10/catalogs/); los mismos para BASS/MzLS | Corrección de magnitudes y colores |
| Tamaños relativos entre bandas (R50,g / R50,r) | Zhang et al. 2025 (arXiv:2509.23901) | Gradientes de color que aportan información que el redshift no explica |
| Eigengalaxias (PCA sobre imágenes) | Uzeirbegovic et al. 2020 (MNRAS 498, 4021) | Extracción de características desde los píxeles |
| Representaciones preentrenadas de Zoobot (ConvNeXt-Nano, entrenado con GZ Evo, que incluye DESI) | Walmsley et al.; [modelos preentrenados de Zoobot](https://zoobot.readthedocs.io/en/latest/pretrained_models.html) | Extracción moderna desde los píxeles, *opcional* (necesita GPU) |
| Ruido, región y profundidad como fuentes de sesgo | Ye et al. 2025; DR10 (norte BASS/MzLS frente a sur DECam) | Diagnóstico de sesgos por región |
| Fases y tareas de *Data Preparation* | [CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml) | Estructura de las conclusiones |

---

## 4. Construcción de características (rúbrica: *Construcción*, 30 pts)

Todas se miden sobre el recorte de 300 px. La segmentación **reutiliza el código del equipo** (`src/data/mask_eval.py`): umbral en la banda r para la máscara de la galaxia y detección de vecinos para excluirlos. Así se respeta la evaluación de máscaras ya hecha (en la banda r se captura 0.999 de la luz con 2 % de contaminación). El cielo residual se mide en el borde del recorte con *sigma clipping*.

### 4.1 Fotometría (por banda: g, r, z)

| Característica | Definición | Por qué |
|---|---|---|
| `m_b` | 22.5 − 2.5 log10(F_b), con F_b el flujo dentro de 2 R_P(r) y los vecinos enmascarados | Magnitud aparente. El logaritmo es la transformación natural del flujo: los errores son multiplicativos |
| `m_b0` | m_b − A_b E(B−V), con E(B−V) del mapa SFD (`dustmaps`) | Quita el enrojecimiento de la Vía Láctea, que no es propiedad de la galaxia |
| `g_r`, `r_z`, `g_z` | Diferencias de `m_b0` | El color es el mejor indicador de la población estelar (edad, polvo) y de M/L |
| `g_r_in`, `g_r_out`, `delta_g_r` (y lo mismo con r − z) | Color dentro de R50 y en el anillo R50–R90; diferencia de ambos | Gradiente de color: un núcleo viejo con disco joven predice D4000 y SFR |
| `mu50_b` | m_b0 + 2.5 log10(2π R50²) [mag/arcsec²] | Brillo superficial: separa discos difusos de sistemas compactos |

### 4.2 Tamaño y estructura (banda r, salvo que se indique)

| Característica | Definición | Por qué |
|---|---|---|
| `R_P`, `R20`, `R50`, `R80`, `R90` | Radios de Petrosian y de la curva de crecimiento, en aperturas elípticas | Tamaño aparente |
| `C`, `C_sdss` | 5 log10(R80/R20); R90/R50 | Bulbo frente a disco |
| `b_a` | Cociente de ejes de los momentos de segundo orden | Inclinación: modula el polvo y el color observado |
| `A`, `S` | Asimetría (rotación de 180°, con el término de fondo restado) y suavidad | Perturbaciones y formación estelar grumosa |
| `gini`, `m20` | Lotz et al. 2004, sobre la máscara de la galaxia | Distribución de la luz |
| `ratio_R50_g_r`, `ratio_R50_z_r` | R50 de la banda / R50 de r | Gradiente de color expresado como tamaño |
| `excede_campo` | 2 R90 > 78.6″ | Avisa cuándo la galaxia no cabe en el recorte |

El ángulo de posición **no** se incluye: la orientación en el cielo no tiene relación física con las propiedades. Es un ejemplo de variable que se descarta por conocimiento del dominio.

### 4.3 Variables que dependen de la distancia (usan el redshift)

| Característica | Definición | Por qué |
|---|---|---|
| `DM` | Módulo de distancia a partir de z (cosmología ΛCDM plana) | Convierte lo aparente en intrínseco. Ataca directamente el sesgo con la distancia que vimos en la CNN |
| `M_b` | m_b0 − DM | Magnitud absoluta: la luminosidad real |
| `log_L_r` | 0.4 (4.65 − M_r), con M_⊙,r = 4.65 (Willmer 2018) | Luminosidad en unidades solares |
| `R50_kpc`, `log_SigmaL` | R50 × escala física; log L_r − log(2π R50²) | Tamaño físico y densidad de luminosidad |
| `logM_ebrova` | 0.673 M_g − 1.108 M_r + 0.996 | Estimador de masa calibrado para Legacy Surveys |
| `logM_bell` | log L_r − 0.306 + 1.097 (g − r) | Segundo estimador de masa, independiente del anterior |
| `log_SigmaM` | logM − log(2π R50,kpc²) | Densidad superficial de masa, ligada al apagado de la formación estelar |

**Lo que no se aplica en v1:** la corrección K, que es pequeña a z ≈ 0.03 (la mediana de MaNGA). Se documenta como limitación; el redshift se conserva como variable para que los modelos absorban lo que quede. La cosmología se fija como constante y se documenta (pendiente 8.5).

### 4.4 Contexto observacional

| Característica | Por qué |
|---|---|
| `region` (norte BASS/MzLS o sur DECam) | Otros filtros, otra profundidad y otra PSF (sección 3.0 de la propuesta de mejoras) |
| `ebv` | La extinción también aumenta el ruido |
| `sigma_cielo_b` | Profundidad local de la imagen |
| `frac_sin_datos` | Zonas sin datos en el recorte |
| `z` | Se conserva: entra en todas las variables de la sección 4.3 |

### 4.5 Variable externa: `T_11` (tipo morfológico)

`T_11` viene de un catálogo de morfología basado en imágenes SDSS (`Datos/inferencia.fits`). No sale de nuestras imágenes, así que se trata como **conjunto aparte**: sirve para mostrar la codificación ordinal frente a one-hot (sección 5.1) y para medir cuánto aporta, sin mezclarla con el conjunto principal.

### 4.6 Validación de las mediciones (calidad de datos de CRISP-ML(Q))

Antes de usar las características se comprueba que miden lo que dicen medir:

- **Nuestro R90 frente a `PETRO_TH90` de NSA:** se espera una relación cercana a 1.
- **Nuestras C y A frente a las del catálogo.**
- **`logM_ebrova` frente a la masa de NSA y la de Pipe3D**, en entrenamiento: la correlación y el desplazamiento por la IMF.
- **Pruebas automáticas con galaxias sintéticas de Sérsic** de radio y flujo conocidos.

---

## 5. Codificación, discretización y transformaciones (rúbrica: *Normalización*, 30 pts)

### 5.1 Codificación

| Variable | Técnica | Por qué |
|---|---|---|
| `region` | One-hot binaria (una columna 0/1) | Dos categorías sin orden |
| `T_11` | **Ordinal** (orden de la secuencia de Hubble) frente a **one-hot** de clases agrupadas (E, S0, Sa–Sb, Sc–Sd, Irr) | Se comparan las dos con el modelo. El orden es físico, así que se espera que la ordinal baste con menos columnas |
| `excede_campo` | Binaria | — |

### 5.2 Discretización (binning)

| Qué | Técnica | Para qué |
|---|---|---|
| Redshift | 5 intervalos por cuantiles, con los límites ajustados en entrenamiento | Análisis estratificado y detección de sesgo con la distancia. **No** entra al modelo como intervalo, porque se perdería información; esa decisión se justifica |
| Clases de las etiquetas | Formadora / apagada con log sSFR = −10.8 (el corte de Katachi); población joven / vieja con D4000 = 1.5 | Necesarias para chi-cuadrado y ANOVA de clasificación. **Nunca** son características |
| Características para chi-cuadrado | `KBinsDiscretizer` (cuantiles, 10 intervalos) | Chi-cuadrado exige conteos categóricos |

### 5.3 Transformaciones contra la asimetría ("características sesgadas")

**Regla, fijada antes de mirar resultados:** se calcula la asimetría (*skewness*) de cada característica en entrenamiento; si |asimetría| > 1, se transforma.
- **Variables positivas** (flujos, radios, luminosidades): logaritmo, que tiene sentido físico, o Box-Cox con λ ajustado en entrenamiento. Se elige la que deje la asimetría más cerca de 0.
- **Variables que pueden ser negativas** (A, gradientes de color): Yeo-Johnson.

**Evidencia en el cuaderno:** una tabla de asimetría y curtosis antes y después, y gráficas Q-Q de las cuatro variables más asimétricas.

### 5.4 Etiquetas

No se transforman. log M\* y log SFR ya son logaritmos, y D4000 tiene una distribución bimodal que ninguna transformación monótona vuelve normal. Además, así se mantiene la comparación con Katachi y la CNN en las mismas unidades.

### 5.5 Escalamiento

| Escalador | Dónde | Por qué |
|---|---|---|
| `StandardScaler` | Por defecto, para modelos lineales, PCA, FA y redes | Media 0 y desviación 1: todas las variables pesan igual y el problema queda bien condicionado |
| `RobustScaler` | Se compara para A, S y M20, que tienen colas | Mediana y rango intercuartílico: es menos sensible a valores extremos |
| `MinMaxScaler` | Antes de `VarianceThreshold` y de chi-cuadrado | Deja la varianza comparable y los valores no negativos |

Todo se encadena en un `Pipeline` / `ColumnTransformer` de scikit-learn ajustado en entrenamiento y guardado con `joblib`, para que el próximo avance lo reutilice sin fugas.

### 5.6 Prueba de convergencia (objetivo 2.4)

Se mide el efecto del escalamiento sobre la optimización:
- **Número de condición** de XᵀX con datos crudos, escalados y transformados más escalados.
- **Curvas de pérdida y número de iteraciones** de `SGDRegressor` y `MLPRegressor` en los tres casos, más el RMSE final en validación.

Se espera que el escalamiento reduzca las iteraciones en órdenes de magnitud; el cuaderno lo cuantifica.

### 5.7 Sesgos por subgrupo

Para cada característica se compara su distribución:
- **Entre el norte y el sur:** diferencia estandarizada de medias y prueba de Kolmogorov-Smirnov.
- **Con el redshift, dentro de intervalos de masa:** C y Gini bajan con la distancia por la PSF.
- **Entre entrenamiento y validación**, para confirmar que las particiones son comparables.

Las características con sesgo fuerte se marcan, y se prueba si la variable `region` lo corrige.

### 5.8 Normalización a nivel de píxel (para la CNN)

Se documenta, con histogramas, asimetría y curtosis, por qué el flujo crudo no es una entrada adecuada y qué hacen el asinh y la estandarización por banda. Esto conecta con las opciones 3.5 y 3.6 de `Propuesta_mejoras_preprocesamiento_DESI.md`. **No** se entrena ninguna CNN en este avance.

---

## 6. Selección y extracción (rúbrica: *Selección / extracción*, 30 pts)

Todo se ajusta con entrenamiento y se confirma con validación.

### 6.1 Métodos de filtro

| Paso | Método | Qué decide |
|---|---|---|
| 1 | `VarianceThreshold` sobre variables en escala MinMax | Elimina las casi constantes |
| 2 | Correlación de Spearman, con agrupamiento jerárquico para \|ρ\| > 0.95 | De cada grupo redundante (p. ej. R80 y R90, o magnitudes de un mismo tipo) se queda la de mayor información mutua con las etiquetas |
| 3 | ANOVA F lineal (`f_regression`) e **información mutua** (`mutual_info_regression`) para las tres etiquetas continuas | Relevancia lineal y no lineal |
| 4 | **Chi-cuadrado** y ANOVA de clasificación (`f_classif`) contra las clases de la sección 5.2 | Relevancia para separar galaxias formadoras de apagadas y jóvenes de viejas |
| 5 | Ranking combinado + **curva de validación**: RMSE en validación frente a las k mejores (k = 3, 5, 8, 12, 16, 24, todas) | Se elige el menor k cuyo RMSE está a menos de un error estándar del mínimo. Así la selección se apoya en una prueba |

### 6.2 Extracción

| Método | Detalle | Qué decide |
|---|---|---|
| **PCA** sobre las características estandarizadas | Gráfica de sedimentación, varianza acumulada (se retiene la que llegue al 95 %) y cargas | Se interpretan los componentes (tamaño/luminosidad, color, concentración…) y se compara el RMSE con PCs frente a características |
| **Análisis factorial** con rotación varimax | Pruebas previas de KMO y Bartlett (`factor_analyzer`), y número de factores por análisis paralelo | Factores latentes interpretables |
| **Eigengalaxias** | PCA incremental sobre imágenes reducidas a 75 × 75 px (promedio de 4 × 4), con el brillo transformado, alineadas por su eje mayor y normalizadas en flujo | Morfología en espacio de imagen con unas decenas de números |
| **Zoobot + PCA** *(opcional, necesita GPU)* | Representaciones de ConvNeXt-Nano preentrenado, reducidas con PCA | La alternativa moderna de extracción. Hay cambio de dominio: Zoobot se entrenó con imágenes de color, no con FITS |

### 6.3 Comparación final

Se usan dos modelos fijos y sencillos, sin optimizar hiperparámetros, porque el objetivo es comparar características, no modelos: **Ridge** (lineal, sensible al escalamiento) y **HistGradientBoosting** (no lineal).

| Conjunto | Contenido |
|---|---|
| S0 | Solo el redshift (referencia ingenua) |
| S1 | Fotometría aparente |
| S2 | S1 + variables con distancia (sección 4.3) |
| S3 | S2 + estructura (todas las construidas) |
| S4 | S3 seleccionadas (sección 6.1) |
| S5 | PCA de S3 |
| S6 | Factores de FA |
| S7 | Eigengalaxias |
| S8 | Zoobot + PCA (opcional) |
| S9 | S4 + la mejor extracción de imagen |
| S_ext | S4 + `T_11` (con cada codificación) |

**Se reporta:** RMSE en validación con intervalo de confianza por bootstrap para cada etiqueta, número de variables, tiempo de entrenamiento y memoria, comparados con la CNN y con Katachi.

---

## 7. Conclusiones (rúbrica: *Conclusiones*, 10 pts)

Se organizan con las tareas de *Data Preparation* de CRISP-ML(Q):

- **Selección de datos:** QCFLAG, particiones y prueba cerrada.
- **Limpieza:** cielo residual, vecinos, extinción y validación de las mediciones.
- **Construcción de características:** las de la sección 4, con su motivo físico.
- **Estandarización:** las transformaciones, escaladores y pipeline de la sección 5.

Para cada tarea se indica qué **requisito** cumplía, qué **riesgo** atacaba (fuga de información, sesgo, asimetría, mala convergencia) y qué **método de calidad** lo controló, como pide la metodología. Se cierra con lo que pasa a la fase 3 (*Model Engineering*): el pipeline guardado, la lista de características seleccionadas y la propuesta de un modelo híbrido CNN + características.

---

## 8. Implementación

### 8.1 Archivos

| Archivo | Contenido |
|---|---|
| `src/caracteristicas_desi/fotometria.py` | Cielo, segmentación (reutiliza `src/data/mask_eval.py`), aperturas elípticas, curva de crecimiento y Petrosian |
| `src/caracteristicas_desi/morfologia.py` | C, A, S, Gini y M20 |
| `src/caracteristicas_desi/fisica.py` | Extinción, cosmología, magnitudes absolutas y estimadores de masa |
| `src/caracteristicas_desi/imagenes.py` | Reducción y alineación para eigengalaxias; entrada para Zoobot |
| `src/caracteristicas_desi/pipeline.py` | Recorre los bloques de la caché en paralelo y guarda resultados reanudables |
| `tests/caracteristicas_desi/` | Galaxias sintéticas de Sérsic con valores conocidos, casos límite de Gini y asimetría, y valores de referencia de extinción y distancia |
| `Notebooks/Avance2.24.ipynb` | El entregable. Las decisiones de ingeniería (transformación, codificación, escalamiento, selección, extracción y modelos) se escriben **visibles en el cuaderno** con scikit-learn; solo la medición de píxeles vive en `src/caracteristicas_desi/` |

**Salidas en Drive** (`MyDrive/resultados_fe/`): `caracteristicas_v1.parquet`, `diccionario_caracteristicas.csv` (nombre, fórmula, unidades, significado y fuente), `pipeline_fe.joblib`, `seleccion.json` y las figuras.

### 8.2 Tiempo estimado en Colab

| Paso | Tiempo |
|---|---|
| Cargar la caché desde Drive (5 GB) | 3–5 min |
| Medir características (9,303 galaxias, en paralelo) | 5–15 min |
| Eigengalaxias | ~3 min |
| Zoobot (opcional, GPU) | ~5 min |
| Selección, extracción y modelos | ~10 min |
| **Total** | **~30–45 min** |

### 8.3 Requisitos previos

- Hacer commit y push de los cambios de rutas de Drive pendientes (`config.py` y el cuaderno de Katachi), porque el cuaderno nuevo clona el repositorio.
- Que la caché `resultados_katachi_fits/cache/desi/recorte300` exista en Drive. Si no existe, el cuaderno la construye.

### 8.4 Cómo seguimos

1. Reviso contigo este plan y ajusto lo que decidas.
2. Implemento el código y las pruebas, y corro el cuaderno en local con 120 galaxias, para que llegue a Colab sin errores.
3. Lo corres en Colab.
4. Con tus salidas, escribo las celdas de análisis y las conclusiones, y ajusto lo que los resultados indiquen.

### 8.5 Decisiones que necesito de ti

1. **Nombre:** ¿`Avance2.24.ipynb` en `Notebooks/`? (El Avance 1 es `Avance1_Equipo24.ipynb`.)
2. **Variables con redshift:** las recomiendo; son las de mayor valor físico. Implican que el modelo usa imagen + redshift espectroscópico, que MaNGA siempre tiene.
3. **`T_11`:** ¿como conjunto aparte (lo que propongo), dentro del principal, o fuera?
4. **Zoobot:** ¿lo incluimos como sección opcional? Agrega dependencias y necesita GPU.
5. **Cosmología:** propongo usar la misma que Pipe3D para que las magnitudes sean coherentes con las etiquetas. La confirmo en su documentación al implementar; si no la encuentro, uso Planck 2018 y lo documento. 
