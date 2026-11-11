# Validación técnica de ViT-FER en AffectNetFiltered

Implementación y ejecución del [protocolo doctoral](PLAN_EXPERIMENTO_DOCTORAL.md).
El perfil principal evalúa `trpakov/vit-face-expression` sin fine-tuning y genera
métricas, predicciones, embeddings, attention rollout y material para PPT.
OpenFace y SVC están prohibidos en este perfil y esperan revisión del investigador.

## Resultado real del 11 de noviembre de 2026

Corrida: `outputs/run_20261111T150020Z_dec91d26/`.

| Métrica | OUR RESULT |
|---|---:|
| Imágenes evaluadas | 3263 |
| Accuracy | 31.9338% |
| Macro-F1 | 27.8783% |
| Balanced Accuracy | 31.7338% |
| UAR | 31.7338% |

Referencia del [paper, tabla 4](https://arxiv.org/html/2409.08907v1): 43.9%.
Delta descriptivo: **-11.9662 puntos porcentuales**. Son datasets/preprocesamientos
diferentes; el delta no demuestra superioridad ni degradación experimental directa.
Nombre del experimento: **Technical validation on AffectNetFiltered**.

La auditoría encontró 39853 imágenes legibles, JPEG, 224×224, escala de grises,
sin imágenes corruptas. Entre las siete clases había 3369 candidatas de validation.
Se excluyeron 32 por etiquetas fuente contradictorias en contenido idéntico y
otras 74 por contenido también presente en train. `contempt` se excluye como clase
de salida y se conserva para detectar contradicciones de anotación.

| Clase | N evaluado | F1 |
|---|---:|---:|
| neutral | 483 | 22.6268% |
| fear | 444 | 43.7191% |
| anger | 472 | 31.5789% |
| happiness | 491 | 54.0385% |
| sadness | 462 | 29.8232% |
| disgust | 455 | 2.1645% |
| surprise | 456 | 11.1969% |

Los originales permanecen intactos. La validación suministrada no está acreditada
como la partición oficial del AffectNet original. Los parámetros de CLAHE/blur y
la independencia por identidad no se conocen. Las métricas describen concordancia
con etiquetas del dataset, no estados psicológicos de las personas.

## Entorno aislado

Se validó Python **3.11.18**, PyTorch **2.6.0+cu126**, torchvision
**0.21.0+cu126**, transformers **4.49.0**, NumPy **1.26.4** y las dependencias
exactas de [requirements_freeze.txt](requirements_freeze.txt). Hardware utilizado:
NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB de VRAM. Los metadatos registran CPU,
RAM, driver, CUDA, flags deterministas, rutas y versiones efectivas.

Por falta de espacio en C:, `.venv311` es una junction al entorno en
`D:\EmotionRecognitionRuntime\affectnet_replication\.venv311`. Python portátil,
cachés, descargas y temporales también se mantienen en D:. Se accede al entorno
desde la ruta habitual del proyecto; no se mezclaron paquetes con `env/`.

Para reconstruir el entorno en Windows:

```powershell
Set-Location 'C:\Users\angel\source\repos\EmotionRecognition\foundational models\affectnet_replication'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_environment.ps1
```

El script instala el bootstrap uv dentro de `.python/`, utiliza Python portátil
sin registro de Windows ni cambios de PATH y crea `.venv311`. Reutiliza el entorno
si ya existe y corresponde a Python 3.11; no borra entornos ni archivos. Puede
recibir `-RuntimeRoot` para elegir otra ubicación de almacenamiento. La caché del
modelo se configura por separado en el perfil JSON.

## Ejecutar

```powershell
# Inspección completa y selección; no carga un modelo.
.\.venv311\Scripts\python.exe run_experiment.py --profile affectnetfiltered-vit --inspect-only

# Auditoría, inferencia y todos los recursos científicos.
.\.venv311\Scripts\python.exe run_experiment.py --profile affectnetfiltered-vit --data-root 'D:\DATASETS\archive\AffectNetPrepared'
```

El perfil está en [configs/affectnetfiltered_vit.json](configs/affectnetfiltered_vit.json).
Declara las particiones reales, source→canonical mapping, política, revisión del
modelo y N/distribución esperados. Si cambian las clases, las particiones o la
distribución depurada, la ejecución se detiene antes de inferencia para revisar
el protocolo. Los nombres numéricos nunca se reinterpretan silenciosamente.

Cada ejecución crea un run ID nuevo. `outputs/latest_run.json` señala la última.
El modelo y su processor quedan fijados al commit
`ef0bc6fc34241b6587e7e009e7711357be28c024`. Los archivos se descargan con
`local_dir`, compatible con Windows sin permisos para crear symlinks. Se usan
safetensors y `trust_remote_code=False`.

El piloto de 32 imágenes sólo verifica infraestructura. La evaluación final
usa FP32, backend eager, batch inicial 16 y seed 42. Los cambios de batch por
OOM quedan registrados. El CLS guardado es la entrada real del clasificador,
tras la normalización final; su reconstrucción de logits se comprueba.

Para regenerar figuras y reportes desde una inferencia terminada:

```powershell
.\.venv311\Scripts\python.exe run_experiment.py --profile affectnetfiltered-vit --render-run '.\outputs\run_20261111T150020Z_dec91d26'
```

Este comando usa el perfil/manifiesto congelados y los arrays guardados. No
repite la evaluación completa. Reutiliza mapas de atención ya calculados; si
faltan, sólo vuelve a procesar los ejemplos seleccionados. Archiva la versión
del código de postprocesamiento sin sobrescribir el código original ejecutado.

## Recursos generados

Los PNG usan 300 dpi, fondo blanco y matplotlib. El resumen tiene 3840×2160
píxeles; el GIF de inferencia, 1280×720 y 3.5 segundos por ejemplo.

| Archivo en `figures/` | Contenido |
|---|---|
| `dataset_examples.png` | Cuatro imágenes reales por clase del conjunto evaluado |
| `confusion_matrix_vit.png` | Matriz absoluta |
| `confusion_matrix_vit_normalized.png` | Matriz normalizada por etiqueta verdadera |
| `f1_per_class_vit.png` | F1 de las siete clases |
| `class_distribution.png` | Support final por clase |
| `prediction_examples.png` | Aciertos de alta/baja confidence y errores de alta confidence |
| `paper_vs_current_accuracy.png` | Referencia contextual y resultado propio, con advertencia |
| `experiment_summary_16x9.png` | Resumen de método y métricas para PPT |
| `top_confusions.png` | Top 3 por tasa normalizada y ejemplos reales |
| `confidence_correct_vs_wrong.png` | Confidence de aciertos/errores |
| `reliability_diagram.png` | Reliability y ECE top-label de 10 bins |
| `vit_embedding_projection.png` | Proyección PCA del CLS; UMAP no estaba instalado |
| `vit_attention_examples.png` | Rollout verificado en dos aciertos y dos errores |
| `vit_inference_demo.gif` | Tres aciertos y dos errores, con probabilidades auténticas |
| `confusion_matrix_build.gif` | Acumulación de predicciones hasta la matriz final |

Las reglas de selección y sample IDs están en
`error_analysis/selected_examples.csv`. Las fotografías se leen sin aplicar
filtros; sólo los mapas de atención añaden overlays explicativos. La atención
es interpretativa, no causal. Una proyección 2D no demuestra estados psicológicos.

## Artefactos y trazabilidad

```text
outputs/run_<UTC>_<id>/
  dataset_inspection.json
  protocol_snapshot.json
  experiment_metadata.json
  requirements_freeze.txt
  manifests/{all_images.csv,evaluation_manifest.csv,exclusions.csv,duplicate_groups.json}
  predictions/{predictions_vit.csv,logits_vit.npy,probabilities_vit.npy,prediction_failures.csv}
  metrics/{metrics_vit.json,classification_report_vit.csv,calibration_vit.json,...}
  embeddings/{vit_cls_embeddings.npy,vit_cls_labels.csv,classifier_parameters.npz,...}
  attention/{<sample_id>.npy,metadata.json,attention_validation.csv}
  error_analysis/{summary.json,selected_examples.csv,panels.json,...}
  figures/{13 PNG,2 GIF,figure_manifest.json,...}
  source_code/             archivos originales ejecutados y snapshots posteriores
  logs/                    progreso, piloto, comandos, errores y verificaciones
  comparison.csv
  report.md
  ppt_results.txt
  artifact_manifest.json
```

`predictions_vit.csv` contiene source label, canonical true label, predicted label,
confidence, correctness, siete probabilidades, sample ID, índice y hashes. Los
arrays comparten exactamente su orden. JSON/arrays usan fracciones [0,1]; las
tablas y figuras rotulan porcentajes. `delta_accuracy_pp` usa puntos porcentuales.
La tabla del paper deja N/R en las métricas que no reporta.

`artifact_manifest.json` registra SHA256 y tamaño de los entregables; `source_code/`
preserva el código ejecutado aunque después se modifique el workspace. Para PPT,
usar `ppt_results.txt`, el resumen 16:9 y las figuras individuales.

## Comprobar

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
.\.venv311\Scripts\python.exe scripts\verify_run.py --run '.\outputs\run_20261111T150020Z_dec91d26' --report '.\outputs\run_20261111T150020Z_dec91d26\logs\verification.json'
```

El verificador recalcula métricas desde predicciones, valida logits/probabilidades,
reconstruye logits desde CLS, comprueba aislamiento de train, tamaño/dpi de figuras,
frames del GIF, hashes de modelos y fuentes archivadas. Por defecto también
comprueba los hashes de **todos los originales auditados**, sin modificarlos.
`--skip-originals` omite sólo esa última lectura.

Las pruebas temporales verifican software; sus fixtures no son resultados del
experimento. El flujo histórico de AffectNet original/OpenFace/SVC se conserva
en [README_LEGACY.md](README_LEGACY.md) y requiere el perfil explícito
`--profile legacy-comparison`. No forma parte de esta validación ViT.
