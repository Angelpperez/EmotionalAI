# Resultado del lazo cerrado con Android y MediaPipe

Fecha de ejecucion: 19 de junio de 2026.

## Objetivo

Validar una integracion continua:

```text
camara Android
-> localizacion facial y de iris
-> extraccion temporal de variables oculares
-> modelo stress3
-> UDP local
-> cambio de dificultad en Blender
-> ACK y registro de latencia
```

Esta prueba demuestra integracion, transporte, respuesta visual y estabilidad.
No constituye por si sola una validacion clinica ni psicofisiologica del estado
de carga cognitiva.

## Configuracion

| Componente | Configuracion |
|---|---|
| Fuente | Camara Android conectada a Windows, indice OpenCV 1 |
| Resolucion | 1280 x 720 |
| Espejo | Desactivado |
| Detector | MediaPipe Face Mesh con 478 landmarks e iris refinados |
| MediaPipe | 0.10.21 |
| OpenCV | 4.11.0 |
| Python | 3.12.0, entorno `D:\emotion_eye_env312` |
| Modelo | `eye_legacy6_live_compatible__LogReg_balanced.joblib` |
| Ventana temporal | 2 segundos |
| Intervalo UDP | 0,5 segundos |
| Puerto de esta corrida | 5056 |
| Escena | `level_01_closed_loop_difficulty_v4.blend` |

El puerto 5056 se utilizo para no cerrar ni modificar una sesion Blender
anterior que mantenia ocupado el puerto 5055.

## Validacion ocular previa

Corrida de control:

```text
D:\emotion3dgs_results\closed_loop_e2e\camera_tests\
  android_mediapipe_eye_quality_active\
```

| Metrica | Resultado |
|---|---:|
| Duracion | 15,013 s |
| Cuadros recibidos | 418 |
| Cuadros de estado de Phone Link | 81 |
| Cuadros con video real | 337 |
| Rostro detectado sobre video real | 337/337, 100% |
| Ambos ojos sobre video real | 337/337, 100% |
| Ambos iris sobre video real | 337/337, 100% |
| Deteccion global incluyendo transicion | 80,622% |
| Procesamiento MediaPipe medio | 22,828 ms |
| Procesamiento MediaPipe maximo | 34,721 ms |

El porcentaje global no representa fallos de MediaPipe: los 81 cuadros
restantes contienen el icono de estado de Phone Link antes de transmitir
video. El runtime identifica esos cuadros y no los usa como observaciones
oculares.

Evidencia visual:

```text
D:\emotion3dgs_results\closed_loop_e2e\camera_tests\
  android_mediapipe_eye_quality_active\best_annotated_frame.jpg
```

El fotograma es de `1280 x 720` y muestra el cuadro facial, los contornos
oculares y los centros de ambos iris detectados por MediaPipe.

## Evidencia visual de los estados adaptativos

Existen tres renderizados de control de la misma escena Blender `v4`:

| Estado visual | Archivo | Resolucion | Contenido |
|---|---|---:|---|
| Facil | `state_previews_v4\difficulty_easy.png` | 1280 x 720 | texto `DIFICULTAD: FACIL` y 3 anillos verdes |
| Normal | `state_previews_v4\difficulty_normal.png` | 1280 x 720 | texto `DIFICULTAD: NORMAL` y 5 anillos amarillos |
| Dificil | `state_previews_v4\difficulty_hard.png` | 1280 x 720 | texto `DIFICULTAD: DIFICIL` y 7 anillos rojos |

Rutas completas:

```text
D:\emotion3dgs_results\closed_loop_e2e\state_previews_v4\
  difficulty_easy.png
  difficulty_normal.png
  difficulty_hard.png
```

Estas figuras verifican visualmente el resultado de aplicar cada uno de los
tres niveles del controlador sobre la escena utilizada. Fueron generadas como
renderizados controlados antes de la corrida y no son capturas temporales
extraidas de los 60 segundos de `run_20260619_194050`.

La evidencia especifica de que la corrida produjo cambios en tiempo real esta
en los registros sincronizados:

```text
end_to_end_events.csv
blender_difficulty_events.csv
```

Ambos contienen 105 eventos concordantes, 11 transiciones entre `hard` y
`easy`, y las marcas temporales de recepcion y aplicacion. La corrida no guardo
video ni fotogramas de Blender automaticamente. Por tanto, para el informe se
deben presentar conjuntamente:

1. el fotograma MediaPipe como evidencia de adquisicion ocular;
2. los tres renderizados como evidencia del efecto visual de cada comando;
3. los CSV y sus metricas como evidencia de que los comandos se aplicaron
   durante la sesion end-to-end.

## Corrida end-to-end

Artefactos:

```text
D:\emotion3dgs_results\closed_loop_e2e\run_20260619_194050\
  end_to_end_summary.json
  end_to_end_events.csv
  blender_difficulty_events.csv
```

### Estabilidad y eventos

| Metrica | Resultado |
|---|---:|
| Duracion solicitada | 60,000 s |
| Duracion continua medida | 60,090 s |
| Finalizacion | Duracion completada |
| Cuadros procesados durante la sesion | 998 |
| Fallos de lectura | 0 |
| Cuadros con deteccion ocular valida | 998/998, 100% |
| FPS efectivo con MediaPipe | 16,608 |
| Mensajes enviados | 105 |
| ACK recibidos | 105 |
| Perdida UDP | 0% |
| Adaptaciones aplicadas en Blender | 105 |
| Transiciones de dificultad | 11 |
| Eventos con cambio visual | 12 |

Antes de iniciar el reloj se descartaron 75 cuadros de estado de Phone Link.
La duracion formal comenzo solamente cuando la fuente entrego video real.

### Estados observados

| Estado suavizado | Eventos |
|---|---:|
| `low` | 62 |
| `high` | 43 |
| `medium` | 0 |

El sistema produjo dos estados y ejecuto transiciones visibles. Sin embargo,
la ausencia de `medium` y las probabilidades frecuentemente saturadas exigen
calibracion y validacion externa antes de interpretar estos estados como carga
cognitiva real.

### Regla de adaptacion y cambio visual

El controlador usa una regla compensatoria: cuando estima menor carga aumenta
la dificultad, y cuando estima mayor carga la reduce.

| Estado estimado | Interpretacion operacional | Dificultad aplicada | Nivel | Indicador visible |
|---|---|---|---:|---|
| `low` | baja carga o capacidad disponible | `hard` | 2 | 7 anillos rojos |
| `medium` | carga intermedia | `normal` | 1 | 5 anillos ambar |
| `high` | carga elevada o posible sobrecarga | `easy` | 0 | 3 anillos verdes |

En cada comando, Blender:

1. actualiza el texto superior a `DIFICULTAD: DIFICIL`, `NORMAL` o `FACIL`;
2. muestra solamente los anillos correspondientes al nivel activo;
3. oculta los indicadores de los otros dos niveles;
4. registra el estado, nivel, cantidad de objetos modificados y tiempo de
   aplicacion;
5. devuelve un ACK al proceso de inferencia.

La roca, su geometria, sus colores, sus materiales, la camara y la iluminacion
no se modifican. Los anillos constituyen una visualizacion controlada del
comando adaptativo. Todavia no representan una tarea minera validada: no
alteran objetivos, rutas, ayudas, oclusiones ni limites de tiempo.

### Metricas de adaptacion observada

| Dificultad | Tiempo activo | Porcentaje de la sesion | Comandos |
|---|---:|---:|---:|
| `normal` | 1,923 s | 3,200% | estado inicial, sin prediccion `medium` |
| `hard` | 34,621 s | 57,615% | 62 |
| `easy` | 23,547 s | 39,185% | 43 |
| **Total** | **60,090 s** | **100%** | **105 comandos inferidos** |

La dificultad cambio 11 veces entre `hard` y `easy`. Incluyendo la primera
aplicacion desde el estado inicial `normal`, hubo 12 eventos con cambio visual.
Estos eventos modificaron 122 estados de visibilidad de objetos:

```text
primera aplicacion normal -> hard: 12 objetos
cada una de las 11 transiciones hard <-> easy: 10 objetos
total: 12 + (11 x 10) = 122
```

No hubo comandos `medium`, por lo que el nivel `normal` solo permanecio activo
durante la acumulacion inicial de la ventana temporal.

La aplicacion visual dentro de Blender fue rapida:

```text
media:   0,959 ms
mediana: 0,885 ms
p95:     1,407 ms
maximo:  3,137 ms
```

Por tanto, la contribucion de Blender a la latencia fue pequena. La mayor parte
del tiempo correspondio a localizacion ocular, inferencia y transporte.

### Latencias

| Segmento | Media | Mediana | P95 | Maximo |
|---|---:|---:|---:|---:|
| Lectura de cuadro | 5,039 ms | 4,892 ms | 6,752 ms | 21,339 ms |
| MediaPipe ocular | 23,702 ms | 23,405 ms | 27,042 ms | 48,859 ms |
| Features | 0,418 ms | 0,411 ms | 0,579 ms | 0,744 ms |
| Inferencia | 27,433 ms | 26,584 ms | 33,756 ms | 36,963 ms |
| UDP hasta recepcion Blender | 15,479 ms | 9,525 ms | 28,417 ms | 436,229 ms |
| Aplicacion en Blender | 0,959 ms | 0,885 ms | 1,407 ms | 3,137 ms |
| RTT UDP | 18,131 ms | 11,930 ms | 31,171 ms | 438,617 ms |
| Ultimo cuadro hasta cambio Blender | 83,681 ms | 78,084 ms | 98,544 ms | 501,757 ms |
| Ultimo cuadro hasta ACK | 85,373 ms | 79,629 ms | 100,372 ms | 503,341 ms |
| Modelo hasta ACK | 45,695 ms | 39,382 ms | 61,903 ms | 465,139 ms |

El maximo de 501,757 ms se concentra en el evento 5. En ese evento, el
transporte hasta Blender tardo 436,229 ms, mientras la aplicacion Blender tardo
0,804 ms. Por tanto, el pico no fue producido por el cambio de escena.

### Retardo de decision

La latencia de software anterior no incluye el tiempo necesario para acumular
la ventana ocular. El retardo desde la observacion mas antigua de la ventana
hasta la aplicacion fue:

```text
media:   2.015,153 ms
mediana: 2.011,667 ms
p95:     2.040,056 ms
maximo:  2.461,854 ms
```

La latencia de procesamiento y el retardo de decision deben reportarse por
separado.

## Interpretacion valida

La corrida demuestra:

1. Integracion funcional de todos los modulos durante 60 segundos.
2. Localizacion continua de ambos iris mediante landmarks.
3. Comunicacion UDP sin perdida en 105 eventos.
4. Aplicacion efectiva de 105 comandos en Blender.
5. Latencia software menor a 100 ms en el percentil 95 desde el ultimo cuadro
   hasta el cambio de dificultad.

No demuestra todavia:

1. Exactitud psicofisiologica de los estados `low`, `medium` y `high`.
2. Generalizacion a sujetos no vistos.
3. Equivalencia validada entre las variables MediaPipe y las variables VREED.
4. Rendimiento durante una sesion formal de diez minutos.
