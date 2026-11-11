# Metricas de IA y validacion

## Principio de validacion

Cada modulo debe tener metricas propias y trazabilidad. La demo de lazo cerrado puede funcionar con heuristicas, pero una afirmacion doctoral requiere etiquetas, splits, baselines y comparacion reproducible.

No se debe afirmar que `Quad_Cat` de VREED equivale a estres o carga cognitiva. `Quad_Cat` es una etiqueta afectiva por cuadrantes; puede servir como baseline afectivo, no como verdad terreno directa para `Bajo Estres / Carga Media / Alta Sobrecarga`.

## Linea base local actual

Dataset ocular local:

- 312 muestras.
- 4 clases balanceadas: 78 por clase.
- 48 features oculares.

Resultados existentes:

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| LogReg | 0.317 | 0.319 |
| SVM_RBF | 0.270 | 0.252 |
| RandomForest | 0.286 | 0.285 |
| MLP_Deep | 0.365 | 0.367 |

Lectura doctoral:

- Este baseline es bajo.
- Sirve para comparacion inicial y para justificar mejora metodologica.
- No debe presentarse como clasificador robusto de carga cognitiva.

## Metricas de reconstruccion 3D

### Preparacion de datos RGB

| Metrica | Como se mide | Criterio inicial |
|---|---|---:|
| Frames extraidos | conteo por nivel | 120-180 para primer entrenamiento ligero |
| Resolucion efectiva | ancho/alto tras resize | 1280 px ancho inicial |
| Blur | varianza de Laplaciano por frame | filtrar percentil bajo |
| Redundancia | diferencia temporal/angular entre frames | muestreo uniforme |
| Holdout | frames no usados en training | 10%-15% |

### COLMAP / SfM

| Metrica | Como se mide | Criterio inicial |
|---|---|---:|
| Imagenes registradas | registradas / extraidas | >= 80% |
| Error reproyeccion medio | COLMAP model analyzer | <= 2.0 px inicial |
| Puntos 3D sparse | conteo COLMAP | reportar, no fijar aun |
| Cobertura circular | inspeccion de camaras | trayectoria cerrada sin saltos grandes |
| Intrinsecos | modelo y parametros estimados | reportar por nivel |

Si COLMAP registra menos de 80%:

- Reducir frames borrosos.
- Usar `sequential_matcher`.
- Probar `exhaustive_matcher` en subset.
- Agregar mascaras para fondo si la roca ocupa poca imagen.

### 3D Gaussian Splatting

| Metrica | Como se mide | Criterio inicial |
|---|---|---:|
| PSNR holdout | render vs imagen real | >= 22 dB demo, >= 25 dB deseable |
| SSIM holdout | render vs imagen real | >= 0.75 demo, >= 0.85 deseable |
| LPIPS holdout | render vs imagen real | <= 0.25 demo, <= 0.18 deseable |
| Tiempo de entrenamiento | minutos por nivel | reportar |
| Pico VRAM | NVIDIA tools / logs | < 5.8 GB |
| Numero de gaussianas | salida del trainer | reportar |
| Tamano del modelo | MB | reportar raw y comprimido |
| Importacion en Blender | PLY/mesh/splat cargado sin errores | obligatorio para demo Blender |
| FPS/fluidez en Blender | viewport o render preview | reportar |

Los umbrales son criterios iniciales de aceptacion tecnica, no resultados esperados garantizados.

### Validacion con ToF

La carpeta `captures/ToF` permite validar geometria si se resuelve la correspondencia espacial con RGB.

| Metrica | Como se mide | Requisito |
|---|---|---|
| ICP fitness | registro 3DGS/COLMAP contra PLY ToF | requiere escala/correspondencia |
| ICP RMSE | distancia residual tras alineacion | reportar en unidades ToF |
| Chamfer distance | nubes muestreadas | reportar media y p95 |
| Cobertura de profundidad | puntos validos por mapa | usar `confidence.tiff` |
| Consistencia de silueta | mascara render vs rango/ToF | IoU si hay segmentacion |

Sin calibracion extrinseca/intrinseca RGB-ToF, ToF debe tratarse como validacion cualitativa o de escala aproximada, no como verdad geometrica absoluta.

## Metricas del detector ocular

| Metrica | Como se mide | Criterio inicial |
|---|---|---:|
| FPS de captura | frames procesados por segundo | >= 20 fps |
| Latencia por frame | tiempo captura->features | <= 50 ms |
| Tasa de deteccion valida | frames con pupila/gaze valido | >= 85% en condiciones controladas |
| Dropout ocular | ventanas sin suficientes muestras | <= 15% |
| Estabilidad de mirada | desviacion en punto fijo | reportar RMSE px |
| Blink rate | parpadeos/minuto | plausible fisiologicamente |

Si se usa EyeTheia:

- Reportar error de mirada en pixeles durante calibracion.
- Usar calibracion de 9 o 13 puntos.
- Guardar modelo/base usada: `itracker_baseline.tar` o `itracker_mpiiface.tar`.

## Metricas del clasificador de 3 estados

Estados objetivo:

- `LOW`: bajo estres / baja carga.
- `MEDIUM`: carga media.
- `HIGH`: alta sobrecarga.

Etiquetas aceptables para tesis:

- NASA-TLX o escala validada por bloque.
- CLARE u otro dataset con labels de carga cognitiva.
- Protocolo propio con consentimiento, tarea controlada y labels por ventana/bloque.

Metricas minimas:

| Metrica | Motivo |
|---|---|
| F1 macro | robustez ante clases no balanceadas |
| Balanced accuracy | evita sobrepremiar clase dominante |
| Matriz de confusion | muestra errores criticos `HIGH` vs `LOW` |
| Precision/recall por clase | sensibilidad a sobrecarga |
| AUROC OvR | separabilidad probabilistica |
| Brier score / ECE | calibracion de probabilidades |
| Latencia de inferencia | uso en lazo cerrado |

Splits recomendados:

- Holdout estratificado solo para prueba rapida.
- K-fold estratificado para baseline.
- Leave-One-Subject-Out si hay multiples sujetos; este es el split mas defendible para generalizacion humana.

Criterios iniciales:

- Superar baseline local VREED `F1_macro=0.367`.
- Para demo: `F1_macro >= 0.55` si hay labels reales.
- Para tesis: priorizar LOSO y reportar intervalo de confianza; no fijar umbral sin conocer dataset final.

### Intervalos calculados el 2026-06-18

Se calcularon intervalos descriptivos desde:

```text
outputs\emotion_benchmark\run_20260616_034800\fold_metrics.csv
```

Como los 25 folds de `RepeatedStratifiedKFold` comparten datos de
entrenamiento, no se trataron como 25 replicas independientes. El intervalo
principal usa las cinco medias por repeticion completa y una distribucion t de
Student:

| Target/modalidad | F1 macro | IC 95% por repeticiones |
|---|---:|---:|
| `quad4`, Eye+ECG+GSR | 0,722051 | [0,699038; 0,745064] |
| `stress3`, Eye+ECG+GSR | 0,734171 | [0,712492; 0,755851] |
| `stress3`, webcam ocular 6 features | 0,427370 | [0,410027; 0,444713] |

Artefactos:

```text
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals.csv
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals.md
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals.png
```

Estos intervalos describen variabilidad del protocolo repetido. No reemplazan
un intervalo entre participantes ni una validacion LOSO.

## Metricas del lazo cerrado

| Metrica | Como se mide | Criterio inicial |
|---|---|---:|
| Latencia sensor->estado | timestamp webcam a prediccion | <= 150 ms |
| Latencia estado->renderer | envio UDP a cambio visual | <= 50 ms |
| Latencia total | sensor a escena | <= 200 ms |
| Perdida UDP | mensajes enviados vs recibidos | < 1% localhost |
| Estabilidad de estado | cambios/minuto | sin oscilacion rapida |
| Tiempo de reaccion visual | cambio de luz/fog tras `HIGH` | <= 250 ms |
| Trazabilidad | logs sincronizados | obligatorio |

### Integracion minima medida el 2026-06-18

Prueba real entre un proceso Python y Blender 4.4.3 en localhost:

```text
mensajes enviados: 30
mensajes reconocidos: 30
perdida: 0%
latencia ida-vuelta media: 1,2108 ms
latencia ida-vuelta p95: 2,1593 ms
envio -> recepcion Blender media: 0,1899 ms
aplicacion de propiedad en Blender media: 0,0891 ms
```

Fuente:

```text
outputs\closed_loop\latency_blender_20260618_valid2\latency_summary.json
outputs\closed_loop\latency_blender_20260618_valid2\latency_messages.csv
```

Alcance: valida transporte UDP, ACK y actualizacion de propiedades dentro de
Blender. No incluye adquisicion de webcam, inferencia del modelo ni tiempo de
refresco visual del viewport.

Se ejecuto ademas una prueba con el modelo ocular real y 30 muestras VREED:

```text
inferencia media: 1,2126 ms
inferencia p95: 1,5794 ms
modelo -> Blender -> ACK media: 2,5758 ms
modelo -> Blender -> ACK p95: 2,9458 ms
perdida: 0%
```

Fuente:

```text
outputs\closed_loop\model_blender_latency_20260618\
```

Esta segunda prueba incluye prediccion, transporte y aplicacion. Sigue
excluyendo captura webcam, extraccion live de features y refresco visual.

Reglas de seguridad del demo:

- Usar histeresis de 2 a 3 ventanas.
- Mantener override manual.
- Loguear todo comando enviado y recibido.
- No ocultar completamente la escena: minimo de luminosidad 35%-40%.
- En Blender, medir tambien el tiempo UDP->actualizacion de propiedad (`World`, luz o volumen), porque el viewport puede actualizar mas lento que el script.

## Artefactos que deben guardarse

Por cada corrida:

- Configuracion de extraccion de frames.
- Lista de imagenes usadas en train/holdout.
- Parametros COLMAP.
- Parametros 3DGS.
- Metricas PSNR/SSIM/LPIPS.
- Capturas render vs ground truth.
- Archivo `.blend` usado en la demo.
- Script Blender de escucha UDP.
- Versiones de librerias.
- Modelo ocular y scaler.
- Matriz de confusion y reporte de clasificacion.
- Logs UDP con timestamps.
- Video corto del comportamiento adaptativo.
