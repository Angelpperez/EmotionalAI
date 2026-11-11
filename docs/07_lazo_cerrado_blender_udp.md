# Lazo cerrado Blender UDP

Objetivo: demostrar que el entorno 3D responde en tiempo real a un estado psicofisiologico simplificado.

Actualizacion 2026-06-18: el controlador principal recomendado pasa a modular
dificultad operacional, no iluminacion. Los controladores de luz se conservan
como prototipos historicos y no fueron eliminados.

## Archivos

```text
tools/blender_udp_environment_controller.py
tools/blender_udp_difficulty_controller.py
tools/send_eye_state_udp.py
tools/udp_latency_probe.py
tools/model_to_blender_latency_probe.py
```

Controlador recomendado:

```text
tools/blender_udp_difficulty_controller.py
```

## Estados de dificultad

| Estado psicofisiologico | Dificultad | Nivel |
|---|---|---:|
| `low` | `hard` | 2 |
| `medium` | `normal` | 1 |
| `high` | `easy` | 0 |

La regla reduce la exigencia cuando se detecta alta sobrecarga y la aumenta
cuando la carga es baja. El controlador no modifica la roca, sus materiales,
colores, luces o geometria.

Las propiedades escritas en la escena son:

```text
closed_loop_state
closed_loop_label
closed_loop_difficulty
closed_loop_difficulty_level
closed_loop_difficulty_normalized
closed_loop_last_update
```

Para que un objeto participe en la dificultad debe etiquetarse explicitamente:

```python
obj["closed_loop_min_difficulty"] = 2
obj["closed_loop_max_difficulty"] = 2
```

Los objetos sin estas propiedades no se alteran. Esto permite añadir
obstaculos, ayudas, objetivos o senales sin tocar la reconstruccion.

## Estados soportados

| Estado UDP | Interpretacion | Efecto en Blender |
|---|---|---|
| `low` | Bajo Estres | Luz alta, sin niebla |
| `medium` | Carga Media | Luz reducida, niebla leve |
| `high` | Alta Sobrecarga | Luz baja, niebla densa |

Esta tabla corresponde al prototipo historico de iluminacion, no al
controlador recomendado para la evaluacion doctoral.

Tambien acepta alias: `bajo`, `medio`, `alto`, `0`, `1`, `2`.

## Uso en Blender

Abrir el `.blend` de la roca, luego ejecutar el script desde Blender:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.4\blender.exe" `
  "data\rocks\level_01\blender\level_01_sparse_colmap.blend" `
  --python "tools\blender_udp_environment_controller.py"
```

El script crea o reutiliza:

```text
ClosedLoop_Key_Light
ClosedLoop_Fog_Volume
ClosedLoop_Fog_Material
```

Escucha por UDP en:

```text
127.0.0.1:5055
```

## Prueba desde otra terminal

Enviar un estado:

```powershell
python tools\send_eye_state_udp.py --state low
python tools\send_eye_state_udp.py --state medium
python tools\send_eye_state_udp.py --state high
```

Probar secuencia automatica:

```powershell
python tools\send_eye_state_udp.py --sequence --cycles 3 --interval 2
```

## Contrato de mensaje

JSON recomendado:

```json
{"state":"high","score":0.9,"timestamp":1780000000.0,"source":"eye_inference"}
```

Texto plano tambien funciona:

```text
high
```

## Integracion con inferencia ocular

El modulo ocular debe enviar uno de estos tres estados cada vez que la ventana temporal se actualice:

```text
low     -> Bajo Estres
medium  -> Carga Media
high    -> Alta Sobrecarga
```

El mismo mensaje es compatible con el controlador de dificultad. No es
necesario modificar `live_vreed_eye_state_udp.py`.

## Resultado de integracion minima

El 2026-06-18 se ejecuto Blender 4.4.3 en modo background con la escena sparse
de `level_01`. La sonda envio 30 estados alternando `low`, `medium` y `high`.

```text
30/30 ACK
0% perdida UDP
RTT media: 1,2108 ms
RTT p95: 2,1593 ms
envio -> recepcion Blender: 0,1899 ms media
aplicacion de dificultad: 0,0891 ms media
```

Artefactos:

```text
outputs\closed_loop\latency_blender_20260618_valid2\latency_messages.csv
outputs\closed_loop\latency_blender_20260618_valid2\latency_summary.json
outputs\closed_loop\latency_blender_20260618_valid2\blender_stdout.log
```

La prueba valida el transporte y el cambio de propiedades de escena. Como la
escena usada no contiene objetos etiquetados de tarea, aun falta una prueba
visual con obstaculos/ayudas y una medicion webcam -> inferencia -> Blender.

Prueba adicional con el modelo real `stress3` live-compatible:

```text
30 muestras oculares VREED -> modelo -> Blender -> ACK
predicciones: low=11, medium=8, high=11
inferencia media: 1,2126 ms
modelo -> ACK media: 2,5758 ms
modelo -> ACK p95: 2,9458 ms
perdida: 0%
```

Artefactos:

```text
outputs\closed_loop\model_blender_latency_20260618\
```

## Fundamento externo

- ComputationalMR implementa adaptacion de interfaces MR segun carga
  cognitiva y expone restricciones de capacidad/carga:
  https://github.com/eth-ait/ComputationalMR
- Un marco reciente de VR adaptativa propone ajustar dificultad en tiempo real
  a partir de HRV y eye tracking:
  https://arxiv.org/html/2504.06461v1
- Los timers usados para el listener siguen la API oficial de Blender:
  https://docs.blender.org/api/current/bpy.app.timers.html

Para rigor doctoral, el mapeo final debe reportar:

1. fuente del modelo o baseline;
2. ventana temporal usada;
3. variables: coordenadas de pupila, velocidad/sacadas, blink rate;
4. matriz de confusion o F1 macro en validacion;
5. latencia media entre inferencia y cambio visual en Blender.
