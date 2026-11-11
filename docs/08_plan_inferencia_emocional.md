# Plan de Inferencia Emocional y Carga Psicofisiologica

## Objetivo

Construir un modulo reproducible que estime un estado operacional para el lazo
cerrado con Blender:

```text
low    = bajo estres / baja carga
medium = carga media o estado ambiguo/intermedio
high   = alta sobrecarga / estres elevado
```

El objetivo del sistema no es diagnosticar emociones clinicas. El objetivo es
producir una senal psicofisiologica trazable para adaptar la luminosidad/niebla
del entorno 3D.

## Estado del repositorio

El repositorio tiene tres lineas de trabajo:

```text
EmotionRecognition\CLIP.py
EmotionRecognition\clipfinal.py
```

Baseline exploratorio zero-shot. Sirve para comparacion, pero no debe ser el
modelo principal porque CLIP no esta especializado en expresion facial fina y
el detector Haar es fragil.

```text
EmotionRecognition\EyeTracking.py
EmotionRecognition\EyeTracking1.py
EmotionRecognition\EyeTracking2.py
```

Prototipos de extraccion ocular por webcam. Son utiles porque calculan pupila,
angulo de mirada y features temporales live-compatible.

```text
EmotionRecognition\VREED.py
EmotionRecognition\vreedpipeline.py
EmotionRecognition\pipelinevreed.py
```

Trabajo mas defendible: clasificacion basada en VREED. La version previa uso
solo 6 features oculares y obtuvo F1 macro aproximado de 0.367 con MLP.

## Dataset local VREED

Archivos usados:

```text
EmotionRecognition\archive\04 Eye Tracking Data\02 Eye Tracking Data (Features Extracted)\EyeTracking_FeaturesExtracted.csv
EmotionRecognition\archive\05 ECG-GSR Data\02 ECG-GSR (Features Extracted)\ECG_FeaturesExtracted.csv
EmotionRecognition\archive\05 ECG-GSR Data\02 ECG-GSR (Features Extracted)\GSR_FeaturesExtracted.csv
EmotionRecognition\archive\02 Stimuli Selection\02 Final Videos Description.xlsx
```

Dimensiones observadas:

```text
Eye-tracking: 312 filas, 50 columnas
ECG:          312 filas, 19 columnas
GSR:          312 filas, 9 columnas
Quad_Cat:     4 clases balanceadas, 78 muestras por clase
```

La planilla de descripcion de videos mapea `Quad_Cat` a arousal/valence:

```text
Quad 0 = high arousal, positive valence
Quad 1 = low arousal, positive valence
Quad 2 = low arousal, negative valence
Quad 3 = high arousal, negative valence
```

## Mapeo operacional de 3 estados

Para el lazo cerrado se define:

| Quad_Cat | Interpretacion circumplex | Estado |
|---:|---|---|
| 0 | alta activacion, valencia positiva | medium |
| 1 | baja activacion, valencia positiva | low |
| 2 | baja activacion, valencia negativa | medium |
| 3 | alta activacion, valencia negativa | high |

Justificacion: `high` se reserva para alta activacion con valencia negativa,
que es el caso mas compatible con estres/sobrecarga. Los cuadrantes mixtos se
tratan como `medium` hasta tener etiquetas directas de carga cognitiva.

## Benchmark implementado

Script:

```text
tools\benchmark_vreed_affective_states.py
```

Ejecuta:

```text
D:\emotion3dgs_env\Scripts\python.exe tools\benchmark_vreed_affective_states.py
```

Modalidades evaluadas:

```text
eye_legacy6_live_compatible
eye_all49
ecg_all18
gsr_all8
physio_ecg_gsr26
multimodal_eye_ecg_gsr75
```

Modelos:

```text
LogReg_balanced
SVM_RBF_balanced
RandomForest_balanced
ExtraTrees_balanced
HistGradientBoosting
MLP_small
```

Metricas:

```text
accuracy
balanced_accuracy
F1 macro
matriz de confusion
classification report
```

Protocolo actual:

```text
RepeatedStratifiedKFold, 5 folds x 5 repeats
```

Limitacion actualizada: los CSV de features extraidas no incluyen `subject_id`
ni `trial_id`, por lo que no se puede ejecutar Leave-One-Subject-Out
directamente sobre el benchmark actual.

Sin embargo, el repositorio local si contiene:

```text
34 archivos eye tracking preprocesados, uno por ID 101-134
34 archivos ECG/GSR preprocesados, uno por ID 101-134
perfil y cuestionarios con columna ID
orden aleatorizado de los 12 videos por participante
```

Auditoria:

```text
outputs\vreed_subject_audit\vreed_subject_completeness.csv
outputs\vreed_subject_audit\summary.json
```

Resultados de la auditoria:

```text
34 participantes con archivos individuales
32 con 12 ensayos oculares presentes
31 estructuralmente completos en eye + baseline/12 segmentos fisiologicos
312 filas en los CSV de features = 26 participantes x 12 ensayos
```

La literatura confirma que las 408 interacciones iniciales se redujeron a 312
ensayos de 26 participantes por calidad y problemas tecnicos. El dataset
publicado no incluye en los CSV el mapeo verificable entre esas filas y los 26
IDs. Por rigor, no se asignaran IDs suponiendo un orden.

Para ejecutar LOSO correctamente se debe:

1. reconstruir las features desde los `.dat` por participante, replicando el
   extractor publicado; o
2. obtener de los autores el mapeo de filas a participantes.

Referencia externa:

```text
https://www.mdpi.com/2076-3417/14/14/6042
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
```

## Runtime live implementado

Script:

```text
tools\live_vreed_eye_state_udp.py
```

Usa solo el modelo `stress3` entrenado con:

```text
eye_legacy6_live_compatible
```

porque esas seis features son las que el prototipo webcam actual puede calcular
en tiempo real.

Ejecucion:

```text
D:\emotion3dgs_env\Scripts\python.exe tools\live_vreed_eye_state_udp.py
```

Salida UDP:

```json
{"state":"low|medium|high","score":0.0,"source":"live_vreed_eye_state_udp"}
```

Log:

```text
outputs\emotion_runtime\live_vreed_eye_state_log.csv
```

## Repositorios/modelos externos para comparacion

Opciones recomendadas:

- EmotiEffLib: biblioteca ligera para reconocimiento de emocion/engagement en
  fotos y videos, con backends PyTorch/ONNX.
- HSEmotionONNX: modelo ONNX de reconocimiento facial de emociones, licencia
  Apache-2.0.
- Py-Feat: toolbox academico para landmarks, Action Units, emociones y analisis
  reproducible.
- OpenFace / OpenFace 3.0: landmarks, head pose, Action Units y gaze. Buena
  referencia metodologica para extractor facial/ocular.

Decision actual:

```text
No reemplazar el nucleo VREED todavia.
Primero fortalecer VREED y usar modelos faciales como canal auxiliar/comparativo.
```

## Siguiente paso experimental

1. Ejecutar benchmark VREED.
2. Elegir modelo `stress3` live-compatible.
3. Abrir Blender con la escena 3D.
4. Ejecutar `live_vreed_eye_state_udp.py`.
5. Guardar logs de:
   - features oculares
   - prediccion
   - score/probabilidades
   - comando UDP
   - condicion visual en Blender

6. Reemplazar la adaptacion luminosa por dificultad operacional:
   - `low` -> `hard`
   - `medium` -> `normal`
   - `high` -> `easy`
7. Etiquetar objetos de tarea en Blender y medir el cambio visual.

## Fuentes externas usadas para seleccion tecnica

- EmotiEffLib: https://github.com/sb-ai-lab/EmotiEffLib
- HSEmotionONNX: https://github.com/av-savchenko/hsemotion-onnx
- Py-Feat: https://py-feat.org/ y https://github.com/cosanlab/py-feat
- OpenFace: https://github.com/TadasBaltrusaitis/OpenFace
- OpenFace 3.0 paper: https://arxiv.org/html/2506.02891v1
