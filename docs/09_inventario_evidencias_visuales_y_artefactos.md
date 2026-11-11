# Inventario de evidencias visuales y artefactos

Fecha de inventario: 2026-06-18.

Este documento indica que archivos existen realmente, que muestran y como
pueden utilizarse en el informe doctoral. Las rutas fueron verificadas en disco.

## Resumen rapido

Evidencia visual preparada en `outputs`:

- 15 imagenes listas para inspeccion.
- 3 hojas de contacto de los videos RGB.
- 2 figuras de segmentacion.
- 7 previews de reconstruccion 3D.
- 2 matrices de confusion del benchmark afectivo.
- 1 grafico de F1 con intervalos de confianza.

Datos visuales e insumos:

- 480 fotogramas RGB seleccionados: 160 por nivel.
- 270 imagenes ToF `range_color.png`.
- 810 imagenes ToF TIFF: 270 confidence, 270 intensity y 270 range_gray.
- 160 mascaras GrabCut experimentales para `level_01`.

Escenas Blender:

- 4 escenas locales de COLMAP sparse/ToF.
- 12 escenas principales en `D:\emotion3dgs_results`.
- Archivos `.blend1` adicionales son respaldos automaticos de Blender.

## Figuras listas para el informe

### Adquisicion RGB

| Archivo | Contenido | Uso recomendado |
|---|---|---|
| `outputs\reports\level_01_contact_sheet.jpg` | Muestra circular del nivel inferior; dos rocas y entorno de captura | Metodologia de adquisicion y cobertura multivista |
| `outputs\reports\level_02_contact_sheet.jpg` | Muestra circular del nivel medio; tres rocas | Metodologia de adquisicion y cobertura multivista |
| `outputs\reports\level_03_contact_sheet.jpg` | Muestra circular del nivel superior; tres rocas y vista mas cenital | Metodologia de adquisicion y cobertura multivista |

Las tres hojas tienen resolucion `1280x540`. Son aptas para mostrar que se
capturaron multiples puntos de vista alrededor de cada nivel.

### Reconstruccion 3D y Blender

| Archivo | Resolucion | Que muestra | Interpretacion |
|---|---:|---|---|
| `outputs\reports\level_01_3dgs_7000_rgb_points_preview.png` | 960x540 | Centros gaussianos convertidos a puntos RGB visibles en Blender | Evidencia diagnostica de 3DGS; no es el render fotorealista de splats |
| `outputs\reports\level_01_colmap_dense_mesh_depth11_preview.png` | 960x540 | Malla densa inicial con roca, piso y fondo | Resultado de reconstruccion completa de escena |
| `outputs\reports\level_01_colmap_dense_mesh_depth11_COLOR_DEMO_preview.png` | 960x540 | Variante oscura de la malla densa | Preview secundario; no usar como figura principal |
| `outputs\reports\level_01_colmap_dense_mesh_depth11_VERTEX_COLOR_EMISSION_preview.png` | 960x540 | Malla densa con color por vertice y shader Emission | Mejor figura actual para mostrar color reconstruido |
| `outputs\reports\level_01_colmap_dense_mesh_depth11_no_plane_t008_preview.png` | 960x540 | Resultado tras retirar un plano dominante | Diagnostico del efecto de limpieza geometrica |
| `outputs\reports\level_01_colmap_dense_mesh_depth11_no_2planes_t008_preview.png` | 960x540 | Resultado tras retirar dos planos | Diagnostico; muestra perdida de continuidad |
| `outputs\reports\level_01_colmap_dense_mesh_depth11_clean_main_component_preview.png` | 960x540 | Componente principal despues de RANSAC y filtrado voxel | Evidencia de que el aislamiento automatico queda fragmentado |

La figura principal recomendada para color es:

```text
outputs\reports\level_01_colmap_dense_mesh_depth11_VERTEX_COLOR_EMISSION_preview.png
```

Advertencia: el material `Emission` permite observar el color sin depender de
la iluminacion, pero no sirve para evaluar una respuesta fisica a cambios de
luz. Para el lazo cerrado debe usarse una variante con material iluminable.

### Segmentacion experimental

| Archivo | Resolucion | Que muestra | Uso recomendado |
|---|---:|---|---|
| `outputs\reports\level_01_masks_threshold_test.jpg` | 1280x684 | Prueba de mascara basada en umbral | Metodologia/resultado negativo |
| `outputs\reports\level_01_masks_grabcut_contact_sheet.jpg` | 1280x684 | Superposicion de mascaras GrabCut | Limitaciones de segmentacion automatica |

Estas figuras deben presentarse como experimentos descartados o diagnosticos.
No deben describirse como segmentacion validada de la roca.

### Clasificacion afectiva

| Archivo | Resolucion | Que muestra | Uso recomendado |
|---|---:|---|---|
| `outputs\emotion_benchmark\run_20260616_034800\confusion_quad4_best.png` | 1210x1056 | Matriz de confusion de cuatro cuadrantes, Eye+ECG+GSR, HistGradientBoosting | Resultados del benchmark afectivo |
| `outputs\emotion_benchmark\run_20260616_034800\confusion_stress3_best.png` | 1210x1056 | Matriz de confusion low/medium/high, Eye+ECG+GSR, HistGradientBoosting | Figura principal del clasificador operacional |

La matriz `stress3` es la figura principal recomendada para explicar el modulo
psicofisiologico.

Figura adicional:

```text
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals.png
```

Muestra F1 macro e IC 95% calculado sobre las cinco medias de repeticion del
protocolo de validacion cruzada.

## Colecciones de imagenes e insumos

### Fotogramas RGB

```text
data\rocks\level_01\images_1280\  -> 160 JPG
data\rocks\level_02\images_1280\  -> 160 JPG
data\rocks\level_03\images_1280\  -> 160 JPG
```

Total: 480 JPG de `1280x720`. Se descartaron aproximadamente los dos primeros
y dos ultimos segundos de cada video durante la seleccion.

El manifiesto de trazabilidad se encuentra en:

```text
outputs\reports\frame_manifest.csv
```

### Capturas ToF

Directorio:

```text
captures\ToF
```

Contenido verificado:

| Tipo | Cantidad |
|---|---:|
| `range_color.png` | 270 |
| `confidence.tiff` | 270 |
| `intensity.tiff` | 270 |
| `range_gray.tiff` | 270 |
| `points_color.ply` | 270 |
| `points_intensity.ply` | 270 |
| `.pfs` | 6 |

Estas capturas existen como insumo, pero todavia no hay una figura final de
fusion RGB-ToF ni una reconstruccion ToF calibrada para el informe.

### Mascaras GrabCut

```text
D:\emotion3dgs_results\level_01_dense_colmap\masks_grabcut\
```

Cantidad: 160 mascaras PNG. Son experimentales y no fueron aceptadas como
entrada final porque incluyeron demasiado piso/fondo.

## Escenas Blender y render asociado

### COLMAP sparse y ToF

| Escena Blender | Render formal disponible | Funcion |
|---|---|---|
| `data\rocks\level_01\blender\level_01_sparse_colmap.blend` | No | Nube sparse del nivel inferior |
| `data\rocks\level_02\blender\level_02_sparse_colmap.blend` | No | Nube sparse del nivel medio |
| `data\rocks\level_03\blender\level_03_sparse_colmap.blend` | No | Nube sparse del nivel superior |
| `data\rocks\level_01\blender\tof_smoke_test.blend` | No | Prueba de importacion de una nube ToF |

`level_01_sparse_colmap.blend1` es un respaldo automatico, no una escena
experimental diferente.

### 3D Gaussian Splatting

| Escena Blender | Render formal disponible | Funcion |
|---|---|---|
| `D:\emotion3dgs_results\level_01_smoke_cuda\level_01_3dgs_smoke_100_steps.blend` | No | Smoke test de 100 pasos |
| `D:\emotion3dgs_results\level_01_train_3000\level_01_3dgs_3000_steps.blend` | No | Entrenamiento de 3000 pasos |
| `D:\emotion3dgs_results\level_01_train_7000\level_01_3dgs_7000_steps.blend` | No | PLY gaussiano nativo a 7000 pasos |
| `D:\emotion3dgs_results\level_01_train_7000\level_01_3dgs_7000_rgb_points.blend` | Si | Conversion a puntos RGB para inspeccion |
| `D:\emotion3dgs_results\level_01_train_7000\level_01_3dgs_7000_rgb_alpha075_points.blend` | No | Variante filtrada por alpha |

Render asociado a la escena RGB:

```text
outputs\reports\level_01_3dgs_7000_rgb_points_preview.png
```

### Reconstruccion densa COLMAP

| Escena Blender | Render formal disponible | Funcion |
|---|---|---|
| `D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11.blend` | Si | Malla densa base |
| `D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_FULL_SCENE_DEMO.blend` | Usa la evidencia visual de la malla base | Demo navegable con contexto |
| `D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_COLOR_DEMO.blend` | Si | Variante de demostracion de color/escena |
| `D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_VERTEX_COLOR_EMISSION.blend` | Si | Mejor escena actual para inspeccion del color |
| `D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_no_plane_t008.blend` | Si | Limpieza de un plano |
| `D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_no_2planes_t008.blend` | Si | Limpieza de dos planos |
| `D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_clean_main_component.blend` | Si | Componente principal filtrado |

Los `.blend1` en `D:` son respaldos automaticos. No deben contarse como
resultados independientes.

## Estadisticas verificadas

### COLMAP sparse

| Nivel | Imagenes | Puntos 3D | Observaciones | Track medio | Obs./imagen | Error medio |
|---|---:|---:|---:|---:|---:|---:|
| Nivel 1 inferior | 160/160 | 14.413 | 81.705 | 5,668841 | 510,656 | 0,981951 px |
| Nivel 2 medio | 160/160 | 15.945 | 104.580 | 6,558796 | 653,625 | 0,977803 px |
| Nivel 3 superior | 160/160 | 12.541 | 155.784 | 12,421976 | 973,650 | 0,874171 px |

Fuente:

```text
outputs\reports\colmap_summary.csv
```

### Entrenamiento 3DGS

| Paso | Gaussianas | Memoria reportada | Tiempo ellipse |
|---:|---:|---:|---:|
| 99 | 14.413 | 0,249636 | 5,494 s |
| 999 | 27.794 | 0,263191 | 36,916 s |
| 2.999 | 168.164 | 0,406942 | 114,446 s |
| 6.999 | 373.024 | 0,610796 | 284,573 s |

Archivos fuente:

```text
D:\emotion3dgs_results\level_01_smoke_cuda\stats\train_step0099_rank0.json
D:\emotion3dgs_results\level_01_train_3000\stats\train_step0999_rank0.json
D:\emotion3dgs_results\level_01_train_3000\stats\train_step2999_rank0.json
D:\emotion3dgs_results\level_01_train_7000\stats\train_step6999_rank0.json
```

Conversion del PLY gaussiano a puntos RGB:

```text
input_points: 373024
output_points: 353914
min_alpha: 0.01
alpha_mean: 0.466199
rgb_mean_r: 0.369512
rgb_mean_g: 0.372860
rgb_mean_b: 0.363847
```

### COLMAP denso

```text
depth_maps: 320
normal_maps: 320
fused_geometric points: 1.168.467
meshed_poisson_depth11 vertices: 467.737
meshed_poisson_depth11 faces: 915.009
```

### Segmentacion y limpieza

Cobertura GrabCut:

```text
media: 0,7365
minima: 0,4274
maxima: 0,8124
```

Limpieza RANSAC/componentes:

```text
planos eliminados: 673.205 puntos
restantes despues de planos: 495.262 puntos
componente voxel principal: 436.292 puntos
voxel_size: 0,25
malla filtrada: 789.713 vertices y 1.507.969 caras
```

### Benchmark afectivo

Protocolo:

```text
RepeatedStratifiedKFold: 5 folds x 5 repeticiones
312 muestras VREED
```

Mejor modelo `quad4`:

```text
modalidad: Eye + ECG + GSR, 75 features
modelo: HistGradientBoosting
accuracy: 0,725540
balanced accuracy: 0,725042
F1 macro: 0,722051
```

Mejor modelo `stress3`:

```text
modalidad: Eye + ECG + GSR, 75 features
modelo: HistGradientBoosting
accuracy: 0,738423
balanced accuracy: 0,733763
F1 macro: 0,734171
```

Rendimiento `stress3` por clase:

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| low | 0,68 | 0,66 | 0,67 |
| medium | 0,74 | 0,75 | 0,75 |
| high | 0,80 | 0,79 | 0,79 |

Modelo compatible con webcam:

```text
modalidad: 6 features oculares
modelo: regresion logistica balanceada
accuracy: 0,447506
balanced accuracy: 0,439232
F1 macro: 0,427370
```

Archivos fuente:

```text
outputs\emotion_benchmark\run_20260616_034800\summary_metrics.csv
outputs\emotion_benchmark\run_20260616_034800\fold_metrics.csv
outputs\emotion_benchmark\run_20260616_034800\classification_report_quad4_best.txt
outputs\emotion_benchmark\run_20260616_034800\classification_report_stress3_best.txt
```

## Mapa sugerido para el informe

1. Adquisicion multivista: usar las tres hojas de contacto.
2. Calidad SfM: usar la tabla COLMAP; actualmente no existe grafico generado.
3. Evolucion 3DGS: usar la tabla de pasos y el preview RGB como diagnostico.
4. Reconstruccion navegable: usar el preview de malla densa.
5. Preservacion de color: usar `VERTEX_COLOR_EMISSION_preview.png`.
6. Limitacion de segmentacion: usar la hoja GrabCut.
7. Efecto de limpieza: comparar malla base con `clean_main_component`.
8. Clasificacion afectiva: usar ambas matrices de confusion.
9. Estado operacional: destacar `confusion_stress3_best.png`.
10. Incertidumbre del clasificador: usar `f1_confidence_intervals.png`.
11. Lazo cerrado: existe una medicion tecnica UDP/Blender, pero todavia no hay
    una captura o video formal de dificultad visible `easy/normal/hard`.

Logs de integracion disponibles:

```text
outputs\closed_loop\latency_blender_20260618_valid2\
outputs\closed_loop\model_blender_latency_20260618\
```

## Advertencias para Claude

- No describir el preview de puntos RGB como render 3DGS fotorealista.
- No afirmar que la segmentacion GrabCut fue exitosa.
- No afirmar que la reconstruccion densa contiene solo las rocas.
- No usar la escena Emission para demostrar respuesta fisica a iluminacion.
- No afirmar que existe fusion RGB-ToF calibrada; solo existen los datos y una
  prueba de importacion ToF.
- Se puede afirmar que transporte UDP y actualizacion de propiedades fueron
  validados con 30 mensajes y 0% perdida. No afirmar que el lazo completo
  webcam-inferencia-dificultad visible esta validado.
- Los estados `low/medium/high` son un mapeo operacional desde arousal y
  valencia, no diagnosticos clinicos.
- El soporte 1560 mostrado en las matrices corresponde a predicciones
  agregadas de la validacion repetida, no a 1560 participantes independientes.
- Los archivos preprocesados si tienen IDs por participante, pero los CSV de
  features no conservan el mapeo verificable de las 312 filas a los 26 sujetos.
  Por ello LOSO sigue pendiente hasta reconstruir features o recuperar el mapa.
