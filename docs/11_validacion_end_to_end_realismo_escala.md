# Validacion integrada, realismo 3DGS y escala metrica

Fecha de implementacion: 2026-06-19.

Este documento separa resultados ya medidos de la corrida formal que requiere
participacion frente a la webcam.

## 1. Lazo cerrado end-to-end

### Implementacion

Flujo instrumentado:

```text
webcam
-> deteccion de ojos y pupila
-> ventana temporal de 2 s
-> seis features oculares compatibles con VREED
-> modelo stress3 live-compatible
-> UDP localhost
-> dificultad aplicada en Blender
-> ACK
-> metricas y log de eventos
```

Scripts nuevos:

```text
tools\live_closed_loop_end_to_end.py
tools\blender_udp_difficulty_controller_experiment.py
tools\prepare_closed_loop_difficulty_scene.py
tools\render_closed_loop_difficulty_states.py
tools\start_closed_loop_experiment.ps1
```

La implementacion no modifica la geometria, los materiales ni los archivos
Blender previos. La escena experimental es una copia:

```text
D:\emotion3dgs_results\closed_loop_e2e\level_01_closed_loop_difficulty_v4.blend
```

La version `v4` usa una pose real de COLMAP (`level_01_0088.jpg`) para evitar
la orientacion arbitraria del viewport. Los indicadores estan anclados a la
camara y no modifican ni rotan la reconstruccion.

Indicadores adaptativos:

| Dificultad | Estado recibido | Indicadores visibles |
|---|---|---|
| Facil, nivel 0 | `high` | 3 anillos verdes |
| Normal, nivel 1 | `medium` | 5 anillos amarillos |
| Dificil, nivel 2 | `low` | 7 anillos rojos |

Estos anillos hacen visible la dificultad aplicada. No son blancos de
calibracion ocular ni es obligatorio mirarlos durante una prueba de latencia.

Previews de control `v4`:

```text
D:\emotion3dgs_results\closed_loop_e2e\state_previews_v4\difficulty_easy.png
D:\emotion3dgs_results\closed_loop_e2e\state_previews_v4\difficulty_normal.png
D:\emotion3dgs_results\closed_loop_e2e\state_previews_v4\difficulty_hard.png
```

### Camara de telefono

Se creo un entorno independiente para no modificar los entornos de COLMAP,
3DGS ni Blender:

```text
D:\emotion_eye_env312
Python 3.12.0
MediaPipe 0.10.21
OpenCV 4.11.0
scikit-learn 1.9.0
NumPy 1.26.4
```

El runtime conserva dos backends seleccionables:

- `haar`: detector historico Haar mas contorno oscuro;
- `mediapipe`: Face Mesh refinado de 478 landmarks, incluidos ambos iris.

El lanzador usa `mediapipe` por defecto. El backend Haar permanece disponible
para comparacion y para no romper las pruebas anteriores.

El runtime acepta una camara virtual por indice o un stream de red:

```powershell
# Camara virtual del telefono expuesta como indice 1
.\tools\start_closed_loop_experiment.ps1 -DurationSeconds 60 -CameraSource 1 -Width 1280 -Height 720 -NoMirror

# Stream de red compatible con OpenCV
.\tools\start_closed_loop_experiment.ps1 -DurationSeconds 60 -CameraSource "http://IP:PUERTO/video" -Width 1280 -Height 720 -NoMirror

# Comparacion explicita con el metodo anterior
.\tools\start_closed_loop_experiment.ps1 -DurationSeconds 60 -CameraSource 1 -EyeDetector haar
```

Antes del lazo cerrado se valida la fuente:

```powershell
D:\emotion_eye_env312\Scripts\python.exe .\tools\preview_video_source.py --source 1 --width 1280 --height 720

D:\emotion_eye_env312\Scripts\python.exe .\tools\evaluate_eye_detection_source.py `
  --source 1 --width 1280 --height 720 --duration-seconds 15 `
  --detector mediapipe `
  --output-dir D:\emotion3dgs_results\closed_loop_e2e\camera_tests\android_mediapipe_eye_quality
```

La camara trasera puede usarse si el telefono queda montado junto al monitor
con la lente dirigida al rostro. La mejora debe demostrarse repitiendo
resolucion efectiva, FPS y porcentaje de frames con deteccion ocular valida.

La fuente virtual puede abrirse y entregar cuadros sin transmitir video. El
caso observado muestra fondo negro y un icono blanco de telefono. El runtime
ahora lo registra como `camera_placeholder_stream` y aborta la corrida, en vez
de confundirlo con ausencia de rostro. Para reactivarlo:

1. Desbloquear Android.
2. Mantener activo `Enlace a Windows`.
3. Aceptar en el telefono el uso de la camara por el PC.
4. Confirmar imagen real con `preview_video_source.py`.

Al iniciar el lazo, el runtime espera hasta 30 segundos dentro del mismo
proceso y muestra `Active la camara conectada en Android`. El reloj formal de
la sesion comienza solamente despues de recibir video real. Esto evita que la
autorizacion de Phone Link consuma parte de la duracion experimental.

Prueba estatica MediaPipe sobre un frame Android real:

```text
rostro detectado: si
ojos detectados: 2
iris detectados: 2
evidencia:
outputs\camera_detection_quality\android_face_mediapipe_annotated.jpg
```

Las marcas verdes de esa evidencia corresponden a landmarks de iris, no a una
segmentacion por intensidad. Esta prueba valida localizacion ocular en un
fotograma; la tasa temporal debe obtenerse con la fuente Android transmitiendo
video real durante toda la ventana.

La webcam integrada fue probada tambien a `1280x720`:

```text
resolucion reportada: 1280x720
FPS efectivo: 28,884
fallos de lectura: 0
```

El frame de control mostro que la limitacion principal no era solamente la
resolucion nominal: el rostro estaba lateral, relativamente pequeno y con
zonas sobreexpuestas. La camara del telefono debe ubicarse cerca del monitor,
frontal al rostro y con iluminacion difusa para que el ojo ocupe mas pixeles.

### Validacion automatizada previa

Se enviaron seis estados alternados a la copia experimental:

```text
low -> medium -> high -> low -> medium -> high
```

Resultados:

| Metrica | Resultado |
|---|---:|
| Mensajes enviados | 6 |
| ACK recibidos | 6 |
| Perdida UDP | 0% |
| RTT medio | 1,8303 ms |
| RTT p95 | 2,6978 ms |
| RTT maximo | 2,8224 ms |
| Aplicacion Blender media | 0,3458 ms |
| Aplicacion Blender maxima | 0,5586 ms |
| Objetos modificados por evento | 11 a 14 |

Evidencia:

```text
D:\emotion3dgs_results\closed_loop_e2e\automated_validation\
```

Esta prueba valida transporte, aplicacion y cambio visual. No incluye webcam ni
inferencia.

### Piloto webcam ejecutado

Se ejecuto una corrida piloto real de 60 segundos:

```text
D:\emotion3dgs_results\closed_loop_e2e\run_20260619_160330\
```

Resultados de integracion y estabilidad:

| Metrica | Resultado |
|---|---:|
| Duracion continua | 60,327 s |
| Termino solicitado completado | Si |
| Frames capturados | 1.776 |
| FPS efectivo | 29,439 |
| Deteccion ocular valida | 83,727% |
| Fallos de camara | 0 |
| Mensajes enviados / ACK | 110 / 110 |
| Perdida UDP | 0% |
| Comandos aplicados en Blender | 110 |
| Ultimo frame -> aplicacion media | 20,469 ms |
| Ultimo frame -> aplicacion p95 | 30,685 ms |
| Ultimo frame -> aplicacion maxima | 36,433 ms |
| Modelo -> ACK media | 9,153 ms |
| Blender apply media | 0,428 ms |

La prueba valida funcionamiento continuo de:

```text
camara -> extraccion live -> inferencia -> UDP -> Blender -> ACK
```

Sin embargo, no valida sensibilidad multestado. Las 110 predicciones fueron
`low`, con probabilidad 1,0. La auditoria posterior identifico incompatibilidad
de dominio y unidades entre las features live y las de entrenamiento:

- VREED expresa duraciones en milisegundos; el runtime entregaba segundos.
- `Num_of_Fixations` y `Num_of_Saccade` son tasas por segundo en VREED; el
  runtime entregaba conteos crudos.
- El detector webcam produjo amplitudes y frecuencias de sacada muy superiores
  a la distribucion VREED.

Por tanto, esta corrida debe citarse como prueba end-to-end de integracion,
latencia y estabilidad. No debe citarse como validacion de reconocimiento de
carga cognitiva ni de adaptacion psicofisiologica multestado.

Adicionalmente se corrigio el runtime posterior al piloto para descartar
muestras oculares antiguas. La metrica de ventana temporal del piloto no debe
usarse; las metricas desde el ultimo frame, modelo, UDP y Blender si son
validas.

Para repetir el piloto de 60 segundos:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_closed_loop_experiment.ps1 -DurationSeconds 60
```

### Corrida formal pendiente

La corrida doctoral de 10 minutos se ejecutara solo despues de resolver o
sustituir de forma explicita la transformacion live de features VREED:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_closed_loop_experiment.ps1 -DurationSeconds 600
```

Durante la corrida:

1. Mantener el rostro visible y con iluminacion estable.
2. Mirar naturalmente entre distintos puntos de la pantalla.
3. No cerrar Blender ni la ventana de webcam.
4. Presionar `Q` solo para interrumpir la prueba.

Cada corrida crea:

```text
D:\emotion3dgs_results\closed_loop_e2e\run_YYYYMMDD_HHMMSS\
  end_to_end_events.csv
  end_to_end_summary.json
  blender_difficulty_events.csv
```

Metricas registradas:

- duracion continua y motivo de termino;
- frames capturados y FPS efectivo;
- porcentaje de frames con deteccion ocular valida;
- latencia de lectura de webcam;
- latencia de procesamiento ocular;
- latencia de features e inferencia;
- UDP hasta recepcion Blender;
- aplicacion de dificultad en Blender;
- ultimo frame hasta aplicacion y ACK;
- retardo incluyendo la ventana temporal;
- mensajes, ACK y perdida;
- comandos aplicados, transiciones y cambios visuales.

La latencia de procesamiento y el retardo de decision se reportan por separado.
La ventana temporal de 2 segundos no debe ocultarse dentro de una unica cifra.

### Corrida Android + MediaPipe completada

Se completo una nueva corrida end-to-end de 60 segundos con la camara Android,
MediaPipe Face Mesh refinado y la escena Blender `v4`:

```text
D:\emotion3dgs_results\closed_loop_e2e\run_20260619_194050\
```

Resultados principales:

| Metrica | Resultado |
|---|---:|
| Duracion continua | 60,090 s |
| Deteccion ocular durante video real | 998/998, 100% |
| Mensajes / ACK | 105 / 105 |
| Perdida UDP | 0% |
| Adaptaciones Blender | 105 |
| Transiciones de dificultad | 11 |
| Ultimo frame -> Blender media | 83,681 ms |
| Ultimo frame -> Blender p95 | 98,544 ms |
| Ultimo frame -> Blender maxima | 501,757 ms |
| Blender apply media | 0,959 ms |
| Retardo medio incluyendo ventana | 2.015,153 ms |

El informe completo, con configuracion, desglose de latencia, outlier y limites
de interpretacion, esta en:

```text
docs\12_resultado_lazo_cerrado_android_mediapipe.md
```

## 2. Intervalos de confianza del F1

Metodo: IC 95% t de Student sobre las cinco medias de repeticion completa de la
validacion cruzada `5 folds x 5 repeticiones`.

| Configuracion | F1 macro | IC 95% |
|---|---:|---:|
| `quad4`, Eye+ECG+GSR, HistGradientBoosting | 0,722051 | [0,699038; 0,745064] |
| `stress3`, Eye+ECG+GSR, HistGradientBoosting | 0,734171 | [0,712492; 0,755851] |
| `stress3`, webcam ocular, LogReg | 0,427370 | [0,410027; 0,444713] |

Estos intervalos describen variabilidad del protocolo de CV. No son intervalos
de generalizacion entre sujetos y no reemplazan LOSO.

## 3. Realismo visual 3DGS

El entrenamiento 3DGS uso:

```text
160 imagenes totales
test_every = 8
140 imagenes de entrenamiento
20 vistas held-out
```

El checkpoint de 7.000 pasos fue evaluado sin reentrenamiento:

```text
D:\emotion3dgs_results\level_01_train_7000\ckpts\ckpt_6999_rank0.pt
```

Resultado oficial calculado por `gsplat` sobre tensores flotantes:

| Metrica | Resultado |
|---|---:|
| Vistas held-out | 20 |
| PSNR medio | 29,7772 dB |
| SSIM medio | 0,913685 |
| LPIPS medio | 0,297730 |
| Tiempo de render medio | 0,05132 s/imagen |
| Gaussianas | 373.024 |

Fuente:

```text
D:\emotion3dgs_results\level_01_eval_holdout_7000\stats\val_step6999.json
```

Metricas por vista, recalculadas desde los PNG guardados:

| Metrica | Media | Desviacion | Minimo | Maximo |
|---|---:|---:|---:|---:|
| PSNR | 29,7513 dB | 1,9562 | 25,8043 | 32,9126 |
| SSIM | 0,912667 | 0,021914 | 0,852349 | 0,945197 |

Artefactos:

```text
D:\emotion3dgs_results\level_01_eval_holdout_7000\renders\
D:\emotion3dgs_results\level_01_eval_holdout_7000\report\holdout_per_view_metrics.csv
D:\emotion3dgs_results\level_01_eval_holdout_7000\report\holdout_per_view_summary.json
D:\emotion3dgs_results\level_01_eval_holdout_7000\report\holdout_real_vs_3dgs_contact_sheet.jpg
```

Limitacion metodologica: las 20 imagenes held-out no participaron en la
optimizacion 3DGS, pero sus poses se obtuvieron de la reconstruccion COLMAP
calculada con las 160 imagenes. Deben describirse como vistas retenidas para
evaluar sintesis visual, no como adquisicion totalmente independiente.

## 4. Escala metrica

Los archivos de configuracion Basler declaran:

```text
Scan3dDistanceUnit = Millimeter
```

Por tanto:

```text
1 unidad ToF = 1 mm = 0,001 m
```

Dimensiones axis-aligned de las superficies visibles segmentadas:

| Roca | X | Y | Z |
|---|---:|---:|---:|
| 1 | 156,70 mm | 169,09 mm | 85,91 mm |
| 2 | 202,76 mm | 175,03 mm | 89,57 mm |
| 3 | 162,10 mm | 171,90 mm | 77,82 mm |

Estas medidas estan en el sistema de coordenadas de la Basler y cubren solo la
superficie visible desde la vista ToF fija. No deben presentarse como
dimensiones completas ni como ground truth del 3DGS.

La escala ToF todavia no se ha transferido a COLMAP/3DGS porque no existe una
calibracion extrinseca RGB-ToF verificada. Esa transferencia requiere registro
3D, transformacion de similitud y error residual documentado.
