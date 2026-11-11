# Factibilidad de las criticas de validacion

Fecha: 2026-06-18.

## Respuesta ejecutiva

| Pregunta | Factible | Estado |
|---|---|---|
| Calcular IC del F1 desde `fold_metrics.csv` | Si | Completado |
| Modular dificultad en vez de iluminacion | Si | Controlador implementado; falta tarea visual etiquetada |
| Integracion minima con latencia | Si | Completada en Blender 4.4.3 |
| VREED con IDs para LOSO | Parcial | IDs existen en `.dat`; no estan mapeados en los CSV de features |

## 1. Intervalo de confianza del F1

Script:

```text
tools\calculate_f1_confidence_intervals.py
```

Metodo: IC 95% t de Student sobre las cinco medias por repeticion completa de
la validacion `5 folds x 5 repeticiones`. Los 25 folds no se usan como replicas
independientes porque comparten datos de entrenamiento.

Resultados:

| Resultado | F1 macro | IC 95% |
|---|---:|---:|
| `quad4`, Eye+ECG+GSR, HistGradientBoosting | 0,722051 | [0,699038; 0,745064] |
| `stress3`, Eye+ECG+GSR, HistGradientBoosting | 0,734171 | [0,712492; 0,755851] |
| `stress3`, webcam 6 features, LogReg | 0,427370 | [0,410027; 0,444713] |

Salidas:

```text
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals.csv
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals_selected.csv
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals.md
outputs\emotion_benchmark\run_20260616_034800\f1_confidence_intervals.png
```

Limitacion: es un intervalo de variabilidad del protocolo de CV, no de
generalizacion entre sujetos.

Fuente tecnica:

```text
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
```

## 2. Controlador de dificultad

Nuevo controlador:

```text
tools\blender_udp_difficulty_controller.py
```

Mapeo:

```text
low    -> hard   -> nivel 2
medium -> normal -> nivel 1
high   -> easy   -> nivel 0
```

El controlador:

- no modifica la roca;
- no toca geometria, materiales, vertex colors, luces ni mundo;
- guarda dificultad como propiedades de escena;
- solo altera visibilidad de objetos etiquetados explicitamente;
- responde con ACK para medir latencia;
- registra eventos en `outputs\closed_loop\difficulty_events.csv`.

Etiquetas soportadas:

```python
obj["closed_loop_min_difficulty"] = 0
obj["closed_loop_max_difficulty"] = 2
```

Estado real: la infraestructura esta implementada y validada. La escena actual
no contiene una tarea formal ni objetos etiquetados, por lo que falta definir
que significa dificultad en el experimento minero: obstaculos, cantidad de
objetivos, ayudas, tiempo limite o complejidad de ruta. Esa definicion debe
quedar en el protocolo experimental antes de generar una demostracion visual.

Fundamento externo:

- ComputationalMR adapta elementos de una interfaz segun carga y capacidad
  cognitiva:
  https://github.com/eth-ait/ComputationalMR
- Marco de VR adaptativa con HRV/eye tracking y ajuste de dificultad:
  https://arxiv.org/html/2504.06461v1

## 3. Integracion minima y latencia

Se ejecuto una prueba real:

```text
Python udp_latency_probe
-> UDP localhost
-> Blender 4.4.3
-> aplicacion de dificultad
-> ACK
-> Python
```

Escena usada:

```text
data\rocks\level_01\blender\level_01_sparse_colmap.blend
```

Resultados:

| Metrica | Resultado |
|---|---:|
| Mensajes enviados | 30 |
| ACK recibidos | 30 |
| Perdida | 0% |
| RTT media | 1,2108 ms |
| RTT mediana | 1,0650 ms |
| RTT p95 | 2,1593 ms |
| RTT maxima | 2,5794 ms |
| Envio a recepcion Blender media | 0,1899 ms |
| Aplicacion en Blender media | 0,0891 ms |
| Aplicacion p95 | 0,1183 ms |

Salidas:

```text
outputs\closed_loop\latency_blender_20260618_valid2\latency_messages.csv
outputs\closed_loop\latency_blender_20260618_valid2\latency_summary.json
outputs\closed_loop\latency_blender_20260618_valid2\blender_stdout.log
```

Alcance validado:

- transporte UDP local;
- recepcion dentro de Blender;
- mapeo `state -> difficulty`;
- escritura de propiedades;
- ACK y perdida.

Pendiente:

- latencia de adquisicion de webcam;
- latencia de extraccion de features;
- latencia de inferencia;
- refresco visible del viewport;
- estabilidad durante una sesion larga;
- prueba con objetos de tarea etiquetados.

### Integracion con el modelo real

Tambien se cargaron el artefacto `stress3` live-compatible y 30 muestras
oculares reales del CSV de VREED:

```text
tools\model_to_blender_latency_probe.py
```

Resultados:

| Metrica | Resultado |
|---|---:|
| Predicciones reconocidas | 30/30 |
| Distribucion | low=11, medium=8, high=11 |
| Inferencia media | 1,2126 ms |
| Inferencia p95 | 1,5794 ms |
| UDP ida-vuelta media | 1,3336 ms |
| Modelo -> Blender -> ACK media | 2,5758 ms |
| Modelo -> Blender -> ACK p95 | 2,9458 ms |
| Perdida | 0% |

Salidas:

```text
outputs\closed_loop\model_blender_latency_20260618\
```

Esta es una integracion minima modelo-renderer. No es aun una medicion
sensor-a-renderer porque utiliza features VREED ya extraidas.

## 4. IDs de sujeto y LOSO

Auditoria:

```text
tools\audit_vreed_subject_ids.py
outputs\vreed_subject_audit\vreed_subject_completeness.csv
outputs\vreed_subject_audit\summary.json
```

Hallazgos:

```text
34 IDs: 101-134
34 archivos eye tracking por participante
34 archivos ECG/GSR por participante
408 cuestionarios post-exposicion = 34 x 12
32 participantes con 12 ensayos eye presentes
31 estructuralmente completos en eye + fisiologia
312 filas de features = 26 x 12
```

Los CSV usados por el benchmark:

```text
EyeTracking_FeaturesExtracted.csv
ECG_FeaturesExtracted.csv
GSR_FeaturesExtracted.csv
```

no contienen `subject_id` ni `trial_id`. La publicacion externa confirma que
los 408 ensayos se redujeron a 312 ensayos de 26 participantes por calidad y
problemas tecnicos, pero no identifica el mapa fila -> participante.

Conclusion:

```text
No es metodologicamente valido ejecutar LOSO sobre los CSV actuales suponiendo
que cada bloque consecutivo de 12 filas corresponde a un ID determinado.
```

LOSO es factible mediante uno de estos caminos:

1. reconstruir las features desde los `.dat` por participante;
2. solicitar a los autores el mapeo de las 312 filas a los 26 IDs;
3. obtener una version del dataset con columnas de sujeto/ensayo.

Con los grupos verificados, se usara:

```text
sklearn.model_selection.LeaveOneGroupOut
```

Fuentes:

- VREED, Kent Academic Repository:
  https://kar.kent.ac.uk/91242/
- Confirmacion de 312 ensayos y 26 participantes:
  https://www.mdpi.com/2076-3417/14/14/6042
- API oficial `LeaveOneGroupOut`:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html

## Prioridad recomendada

1. Definir una tarea minera y sus variables de dificultad observables.
2. Etiquetar objetos de ayuda/obstaculo en una copia de la escena Blender.
3. Ejecutar webcam -> inferencia -> dificultad y registrar latencia total.
4. Reconstruir features VREED por participante para ejecutar LOSO.
5. Comparar CV estratificada frente a LOSO y recalcular intervalos.
