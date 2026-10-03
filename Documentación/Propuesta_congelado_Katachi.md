# Propuesta: entrenamiento con capas congeladas, como en el artículo de Katachi

**Estado:** aprobada e implementada (`src/katachi_fits/chain.py`, `train.py`, `config.py`; pruebas en `tests/katachi_fits/test_freezing.py`; documentada en la sección 7.1 del cuaderno).
**Alcance:** cómo reentrenar la cadena de Katachi (masa → SFR → D4000) con los FITS de DESI siguiendo el congelado del artículo (Alfonzo et al. 2024, §3.1.2 y §3.1.3), de forma defendible y verificable.

---

## 1. Por qué hace falta esta propuesta

El artículo y los artefactos publicados de Katachi se contradicen:

| Fuente | Qué dice sobre el congelado |
|---|---|
| Artículo, §3.1.3 | "All the layers of this pretrained model remained frozen, and we trained the final convolutional and linear layers of the model using the process that was described in Section 3.1.2." |
| Código (`MaNGA_DR17_Regr.ipynb`) | Intenta congelar los primeros 144 de 161 tensores, pero el bucle recorre una variable equivocada (`model`) y tiene un error de sintaxis (`if i < 144:,`). No se guardó ninguna salida. |
| Pesos publicados (7 modelos) | Ningún tensor coincide con ImageNet V1 ni V2, y ningún par de modelos comparte un tensor. Todas las capas se entrenaron. |

El equipo decidió que **el artículo manda**. Pero el artículo deja sin resolver varios detalles que, al congelar, sí cambian el resultado: cuántas capas se entrenan, qué pasa con BatchNorm, cómo se agregan los canales de masa y SFR, qué pérdida guía la tasa de aprendizaje y cuándo parar. Este documento fija cada uno con su justificación.

## 2. Regla para decidir

Cuando las fuentes no coinciden, se aplica este orden:

1. **El texto del artículo**, cuando es explícito.
2. **El código de los autores**, cuando el artículo es ambiguo, porque es la única forma en que ellos mismos convirtieron sus palabras en operaciones.
3. **La práctica documentada de transferencia de aprendizaje**, cuando ninguno de los dos dice nada. Las referencias son la guía de Keras de F. Chollet, la documentación de PyTorch y la literatura citada abajo.
4. **Nuestras restricciones:** las imágenes son FITS y no PNG, y Test se abre una sola vez.

Cada decisión queda en una constante de `config.py`, y cada desviación de Katachi tiene una prueba automática.

## 3. Decisiones

### D1. Qué capas se entrenan

**Opciones.**
- **(a)** Solo la última capa convolucional (`layer4.2.conv3`) y la lineal: 1.05 M parámetros, 4 %.
- **(b)** Desde el tensor 144: las últimas 5 capas convolucionales (`layer4.1.conv2` a `layer4.2.conv3`) y la lineal: 7.9 M, 34 %.
- **(c)** Toda la etapa `layer4`: unos 15 M, 64 %.

**Decisión: (b), umbral 144.**

**Argumento.** El artículo dice "final convolutional and linear layers", en plural, pero no da un número. La única regla concreta de los autores es `i < 144`. El bucle imprime el índice y la forma de cada tensor, lo que sugiere que eligieron el umbral mirando la arquitectura, no al azar. Contra (b) está el comentario del código, "Except the Last Conv Layer", que va en singular y apoya (a). Pero un comentario en lenguaje natural es una evidencia más débil que el número que ejecuta la regla. (c) no aparece en ninguna fuente de Katachi.

**Riesgo.** Si la intención real era (a), damos al modelo más capacidad de la que tenía. Para medirlo, el umbral queda en la constante `FIRST_TRAINABLE_PARAM = 144`, y (a), con valor 156, es la prueba de sensibilidad prioritaria si hay cómputo (§5).

### D2. BatchNorm en la parte congelada

**Opciones.**
- **(a)** `requires_grad = False` y nada más: los pesos de BatchNorm quedan fijos, pero su media y su varianza se siguen actualizando con cada lote. Es el comportamiento por defecto de PyTorch en modo entrenamiento, y el que habría tenido el código de Katachi.
- **(b)** Además, poner esas BatchNorm en modo evaluación: estadísticas de ImageNet, sin cambios.

**Decisión: (b).**

**Argumento.**
1. El artículo dice que las capas "remained frozen". Con (a), las capas congeladas cambian su salida durante el entrenamiento, así que no estarían congeladas en ningún sentido operativo.
2. La guía de transferencia de aprendizaje de Keras ata explícitamente el congelado de BatchNorm al modo inferencia: "When you set `bn_layer.trainable = False`, the BatchNormalization layer will run in inference mode, and will not update its mean & variance statistics". También advierte que actualizarlas "will suddenly destroy what the model has learned".
3. En la comunidad de PyTorch, la respuesta habitual a "cómo congelar BatchNorm" es ponerla en modo evaluación (discuss.pytorch.org, "Proper way of freezing BatchNorm running statistics"). arXiv 2102.05543 resume que se recomienda mantener BatchNorm en modo inferencia al ajustar.
4. Hay además un problema propio de este caso. Katachi entrenó en dos GPU, así que cada BatchNorm promediaba sobre mitades de 16 imágenes. Con (a), las estadísticas de 15 M de parámetros congelados derivarían con ese ruido.

**Contraargumento y respuesta.** Las estadísticas de ImageNet no corresponden a imágenes astronómicas, que son en su mayoría cielo cercano a cero. Pero las capas entrenables (D1) tienen sus propias BatchNorm en modo normal, y esas sí se adaptan a la nueva distribución. Es el diseño estándar: características fijas y una "cabeza" que se adapta. La opción (a) queda en una constante (`FREEZE_BN_STATS = True`) como prueba de sensibilidad.

**Verificación.** Los buffers `running_mean`, `running_var` y `num_batches_tracked` de la parte congelada deben quedar idénticos bit a bit a los de ImageNet al terminar.

### D3. Primera capa de las redes de SFR y D4000 (4 y 5 canales)

**Opciones.**
- **(a)** Como en el código de Katachi: una capa nueva aleatoria de 4 o 5 canales, congelada. Quedaría aleatoria para siempre.
- **(b)** Igual que (a), pero entrenable.
- **(c)** Copiar los pesos de ImageNet en los 3 canales de imagen y congelarlos, e iniciar en cero los canales de masa (y SFR) y entrenarlos.

**Decisión: (c).**

**Argumento.**
1. **La primera capa forma parte del modelo preentrenado.** La regla del artículo ("all the layers of this pretrained model remained frozen") se aplica a sus pesos de imagen. Los canales de masa y SFR no existen en ImageNet: son capas nuevas, y la práctica estándar es entrenar lo nuevo.
2. **(a) y (b) contradicen el propósito del congelado.** `layer1`, que está congelada, espera exactamente las salidas de la `conv1` de ImageNet. Una `conv1` aleatoria le entregaría otra cosa, y las características preentrenadas dejarían de valer. El artículo justifica el congelado precisamente por aprovechar esas características ("leverage its knowledge of general object features"). La guía de Keras advierte de lo mismo: mezclar capas aleatorias entrenables con capas preentrenadas produce gradientes grandes que "will destroy your pre-trained features".
3. **(c) es la práctica estándar para añadir canales a una red preentrenada:** copiar los pesos RGB e iniciar los canales nuevos en cero o con valores pequeños (foros de PyTorch y fast.ai, Stack Overflow). Al iniciar en cero, al principio la red de SFR se comporta exactamente como la ResNet50 preentrenada, y aprende a usar la masa sin estropear nada.
4. **Por qué Katachi pudo usar una capa aleatoria.** Sus pesos publicados muestran que entrenó todas las capas, así que la red pudo reaprender `conv1` y lo que sigue. Con congelado eso ya no es posible. Su elección de capa aleatoria era coherente con lo que hicieron, no con lo que escribieron.

**Implementación.** Se suman dos convoluciones: `conv_imagen(x[:, :3])`, congelada y con pesos de ImageNet, más `conv_extra(x[:, 3:])`, entrenable e iniciada en cero. El resultado es matemáticamente igual a una sola `conv1` de 4 o 5 canales, pero el congelado parcial queda explícito y es verificable.

**Verificación.** Los pesos de imagen de la primera capa deben quedar idénticos a ImageNet; los de los canales extra, distintos de cero.

### D4. Orden de las bandas

**Opciones.** **(a)** g, r, z (orden del FITS, el que pidió el documento del equipo). **(b)** z, r, g.

**Decisión: (b), z, r, g.**

**Argumento.** Con la primera capa congelada, sus filtros son los de ImageNet: el canal 0 aprendió sobre rojo, el 1 sobre verde y el 2 sobre azul. Las imágenes Lupton de Katachi ponían R = i, G = r y B = g, es decir, la banda de onda más larga en el canal rojo (Lupton et al. 2004). Con z, r, g mantenemos esa correspondencia. Con g, r, z, el filtro "rojo" recibiría la banda más azul y los filtros de color de ImageNet verían los colores invertidos.

Cuando el documento del equipo pidió "documentar el orden como g, r, z", la red se iba a reentrenar completa y el orden daba igual. Con capas congeladas ya no da igual. El cambio no cuesta nada: es un reordenamiento al cargar.

**Verificación.** Una prueba que confirme que el canal 0 de la entrada es la banda z del FITS (según `BAND2` en el encabezado).

### D5. Escala de entrada

**Decisión.** Mantener `asinh(flujo / (β·σ)) / s`, con valores aproximadamente entre 0 y 1 y el cielo en 0. **Sin** la normalización de ImageNet.

**Argumento.** Katachi usó `ToTensor()` (valores de 0 a 1, sin normalización de ImageNet), y el artículo no dice nada al respecto. Mantener la misma escala que Katachi es lo fiel. Que el cielo valga 0 reproduce el fondo negro de sus PNG, y además hace que el relleno 0 de la rotación aleatoria signifique "sin luz".

**Riesgo.** Con características congeladas, la escala de entrada influye más que con la red completa. Se anota como limitación y como prueba de sensibilidad futura (normalizar con la media y la desviación de ImageNet).

### D6. Optimización

**Decisión.** Adam con tasa **1e-3 en las tres redes**, solo sobre los parámetros entrenables, error cuadrático medio y lotes de 32.

**Argumento.** El artículo es explícito: "a learning rate of 10⁻³ for all three networks and trained with batch sizes of 32". El 1e-4 de la red de D4000 en el código contradice al artículo y, por la regla del §2, se descarta.

### D7. Reducción de la tasa de aprendizaje

**Decisión.** Un `ReduceLROnPlateau` por red, con paciencia 3 y los demás argumentos por defecto de PyTorch, guiado por la **pérdida de entrenamiento de esa red**.

**Argumento.**
1. El artículo usa una partición 90/10 sin conjunto de validación, y evalúa en Test solo "después de entrenar". Si la tasa de aprendizaje dependiera de un conjunto retenido, añadiríamos un paso que el artículo no tiene.
2. "Each network is trained independently… without knowing the loss function value of any of the other networks": cada red con su propio planificador y su propia pérdida.
3. El código usa la pérdida de **Test** para el planificador. Eso filtra Test hacia el entrenamiento y no se puede imitar.
4. El artículo nombra la función de PyTorch y solo especifica la paciencia (3), así que los demás argumentos van por defecto: factor 0.1 y umbral relativo de 1e-4.

### D8. Parada

**Decisión.** Una parada por red. Una red se detiene cuando su pérdida de entrenamiento no mejora más de 0.1 % (relativo) en 10 épocas consecutivas. Desde ese momento:
- sus pesos y su BatchNorm quedan fijos;
- sigue produciendo su predicción para la red siguiente.

El entrenamiento termina cuando las tres redes se detienen, o al llegar a 1000 épocas. Los pesos finales de cada red son los del momento en que se detuvo. No se elige "la mejor época".

**Argumento.**
1. El artículo: "if the training loss had not been reduced by more than 10⁻³ in 10 epochs, the training was stopped". La regla es sobre la pérdida de entrenamiento y, como cada red es independiente, se aplica a cada una.
2. El artículo no dice si el 10⁻³ es absoluto o relativo. El `EarlyStopper` de los autores es relativo (`loss < min·(1 − 1e-3)`) y es la única forma concreta disponible.
3. Elegir la mejor época con Validation sería un criterio de selección que el artículo no tiene.

### D9. Qué datos se usan en cada paso

**Decisión.**

| Conjunto | Galaxias | Uso |
|---|---|---|
| Train | 7,123 | Entrenar |
| Validation | 1,262 | Solo vigilar el sobreajuste: ninguna regla depende de ella |
| Test | 918 | Una sola vez, al final |

**Argumento.** El artículo comprueba el sobreajuste comparando el error de entrenamiento con el de un conjunto no visto. Nosotros no podemos usar Test para eso sin gastarlo, y Validation cumple ese papel. La alternativa, entrenar con Train + Validation (8,385, más cerca del "90 %" del artículo), daría un 18 % más de datos pero dejaría sin un control independiente del sobreajuste. Además, la partición del equipo (`split_proyecto`) ya separa Validation. Se mantiene.

### D10. Lo que no cambia

| Elemento | Se mantiene así | Por qué |
|---|---|---|
| Aumentación | Reflejos horizontal y vertical (p = 0.5) y rotación de 0 a 360°, vecino más cercano, relleno 0 | Artículo §3.1.4 y DataLoaders publicados |
| Entradas de la cadena | Masa y SFR **predichas** y desacopladas, como plano constante | Artículo §3.1.1–3.1.2 |
| BatchNorm de las capas entrenables | En dos mitades del lote | Reproduce las dos GPU de Katachi; el artículo no lo contradice |
| Evaluación | Sin aumentación | Reproduce las predicciones publicadas; lo confirma el cuaderno de reproducción del equipo |

## 4. Cómo se demuestra que se hizo bien

Es la misma prueba que reveló que los pesos de Katachi no estaban congelados, ahora aplicada a los nuestros.

1. **Prueba automática**, antes de entrenar en Colab. Se dan varios pasos de entrenamiento y se comprueba:
   - **Lo congelado sigue idéntico bit a bit a ImageNet:** los 144 tensores congelados, sus buffers de BatchNorm y los pesos de imagen de la primera capa.
   - **Lo entrenable sí cambió:** los 17 tensores finales y los canales nuevos.
   - **Detenido = fijo:** una red detenida ya no cambia.
2. **En el cuaderno, sobre los pesos finales guardados:** una tabla por red con los parámetros congelados, los entrenables y la comparación contra ImageNet. Cualquiera puede volver a ejecutarla.
3. **Registro:** la época en que se detuvo cada red y la evolución de su tasa de aprendizaje.

## 5. Qué cambia en la interpretación

- **B0 no es directamente comparable.** Los pesos publicados de Katachi se entrenaron sin congelar, así que nuestra corrida frente a B0 mezcla dos efectos: las imágenes y el congelado. Lo diremos explícitamente.
- **Corridas adicionales**, si hay cómputo y en orden de prioridad. Cada una cambia una sola constante:
  1. **Control sin congelar**, igual en todo lo demás. Permite separar los efectos: congelado frente a sin congelar es el efecto del congelado, y sin congelar frente a B0 es el efecto de las imágenes.
  2. **Umbral 156** (solo la última convolucional), para la ambigüedad de D1.
  3. **BatchNorm actualizándose** en la parte congelada, para D2.
- **Costo.** En la red de masa, la retropropagación solo recorre las capas finales, así que es bastante más rápida. En las de SFR y D4000, el gradiente de los canales nuevos de la primera capa obliga a recorrer toda la red, pero sin calcular gradientes de los pesos congelados. En conjunto se espera menos tiempo que el entrenamiento completo; lo medirá la prueba de velocidad del cuaderno.
- **Riesgo científico.** Las características de ImageNet congeladas pueden rendir menos en imágenes astronómicas que una red reentrenada por completo. Si pasa, es un resultado del método del artículo, no un error de implementación. El control sin congelar lo cuantificaría.

## 6. Resumen de constantes (`config.py`)

| Constante | Valor | Decisión |
|---|---|---|
| `FREEZE_PRETRAINED` | `True` | D1 |
| `FIRST_TRAINABLE_PARAM` | `144` | D1 |
| `FREEZE_BN_STATS` | `True` | D2 |
| `EXTRA_CHANNELS_INIT` | `"zeros"` (con los pesos de imagen de ImageNet congelados) | D3 |
| `INPUT_BAND_ORDER` | `("z", "r", "g")` | D4 |
| `LR_MSTAR`, `LR_SFR`, `LR_D4000` | `1e-3`, `1e-3`, `1e-3` | D6 |
| `SCHEDULER_SIGNAL` | `"train"` (por red) | D7 |
| `STOP_SCOPE` | `"per_network"` | D8 |
| Validation | solo para vigilar | D9 |

## Referencias

- Alfonzo, J. P., et al. 2024, ApJ 967, 152, §3.1.1–3.1.4.
- Código y artefactos de Katachi: <https://github.com/juanpabloalfonzo/Katachi>; pesos en `astr.tohoku.ac.jp/~juanpabloalfonzo/Katachi_MaNGA_Catalogues/`.
- Chollet, F. *Transfer learning & fine-tuning* (guía de Keras): <https://keras.io/guides/transfer_learning/>.
- Foros de PyTorch: "Proper way of freezing BatchNorm running statistics"; "How to modify the input channels of a ResNet model".
- arXiv:2102.05543, sobre el papel de los parámetros de BatchNorm en la transferencia de aprendizaje.
- Lupton, R., et al. 2004, PASP 116, 133, sobre la composición RGB.
