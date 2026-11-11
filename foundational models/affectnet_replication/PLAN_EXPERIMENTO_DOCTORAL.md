# Plan experimental: validación técnica de ViT-FER en AffectNetFiltered

**Fecha:** 11 de noviembre de 2026, America/Santiago.  
**Estado actual:** implementado y ejecutado; verificación independiente completada.  
**Versión del protocolo:** 1.0, fijada antes de observar predicciones.  
**Nombre del experimento:** `Technical validation on AffectNetFiltered`.  
**Proyecto:** `C:\Users\angel\source\repos\EmotionRecognition\foundational models\affectnet_replication`.  
**Datos:** `D:\DATASETS\archive\AffectNetPrepared`.  
**Alcance del documento original:** decisiones, evidencia inicial, implementación prevista y criterios de aceptación fijados antes de inferencia. La implementación se realizó después de la instrucción «implementa». El protocolo metodológico 1.0 se conserva; los tiempos futuros de las secciones de planificación describen ese diseño previo.

**Cierre de implementación:** corrida `run_20261111T150020Z_dec91d26`, N=3263; Accuracy=31.9338%, Macro-F1=27.8783%, Balanced Accuracy/UAR=31.7338%. Se generaron y verificaron 13 PNG y 2 GIF. Pasaron 25 pruebas del código y la comprobación de hashes de los 39853 originales. Véase [EJECUCION_AFFECTNETFILTERED.md](EJECUCION_AFFECTNETFILTERED.md) para el cierre y [README.md](README.md) para comandos operativos. Una copia exacta del plan anterior a la ejecución se conserva dentro del directorio de fuentes de la corrida.

Este documento integra los dos requisitos adjuntos: primero obtener métricas reales de ViT-FER y después producir material científico para la presentación. OpenFace y SVC quedan fuera de esta iteración. Su ejecución futura requiere que el investigador revise primero los resultados de ViT, conforme a su instrucción explícita.

## 1. Pregunta científica y alcance de las conclusiones

Evaluar cómo clasifica un ViT-FER preentrenado las siete etiquetas de expresión facial de la copia local de AffectNetFiltered, sin adaptar sus pesos. El resultado medirá concordancia con las etiquetas del dataset; no medirá estados psicológicos internos de las personas.

Las preguntas son:

1. ¿Qué Accuracy, Macro-F1, Balanced Accuracy y UAR obtiene sobre la partición de evaluación congelada?
2. ¿Qué clases y pares dirigidos de clases concentran los errores?
3. ¿Cómo se relaciona el máximo softmax con la corrección y la calibración empírica?
4. ¿Qué estructura muestran los embeddings y qué patrones presenta el attention rollout, cuando su extracción pueda verificarse?

La comparación con Schuller et al. es contextual. No se plantea una prueba de superioridad ni una estimación del efecto causal de CLAHE, blur o escala de grises. Para eso harían falta las mismas imágenes originales, variantes controladas y un diseño adicional.

**Términos del informe:** OUR RESULT y PAPER REFERENCE. La cifra 43.9% es una referencia publicada, nunca un resultado de esta ejecución. Se conservará el título `Technical validation on AffectNetFiltered`; no se declarará reproducción exacta. El modelo es un clasificador ViT especializado mediante entrenamiento previo en FER2013; usarlo sin adaptación a este dataset no equivale a clasificación abierta de emociones ni a un sistema clínicamente validado. [S1, S2]

## 2. Datos encontrados: evidencia local

### 2.1 Estructura observada

```text
D:\DATASETS\
  archive.zip
  archive\
    AffectNetPrepared\
      train_class_Final\
        anger/ contempt/ disgust/ fear/ happy/ neutral/ sad/ surprise/
      val_class_Final\
        anger/ contempt/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

Se identificaron dos carpetas de partición y ocho clases por carpeta. Los nombres son observaciones del filesystem. No se inferirá el origen oficial de las particiones a partir de sus nombres.

### 2.2 Conteos observados antes de depurar duplicados

| Etiqueta canónica | Carpeta fuente | Train | Validation disponible | Diferencia de validation respecto a 500 |
|---|---|---:|---:|---:|
| neutral | neutral | 4803 | 486 | -14 |
| fear | fear | 4738 | 481 | -19 |
| anger | anger | 4787 | 484 | -16 |
| happiness | happy | 4848 | 491 | -9 |
| sadness | sad | 4741 | 477 | -23 |
| disgust | disgust | 3660 | 480 | -20 |
| surprise | surprise | 4756 | 470 | -30 |
| **Total principal: siete clases** | | **32333** | **3369** | **-131** |
| Excluida: contempt | contempt | 3664 | 487 | No participa |
| **Total descargado: ocho clases** | | **35997** | **3856** | No aplica |

**Total de archivos de imagen:** 39853. **Extensión observada:** `.jpg`.
El paquete ZIP local ocupa 295918159 bytes. Estos son conteos de archivos, no predicciones ni estimaciones de rendimiento.

La evaluación tiene un máximo inicial de **3369 candidatos** de las siete clases. El N definitivo se fijará tras legibilidad, conflictos de etiqueta y duplicados. No se completarán las clases hasta 500 mediante copias, aumentos o imágenes de train. El paper describe 3500 imágenes, 500 por clase: la copia local ya difiere en tamaño y composición. [S1]

### 2.3 Integridad, formato y duplicados

La auditoría de planificación está separada de las corridas de modelos:

- [Auditoría JSON](outputs/planning_20261111/dataset_inspection.json).
- [Manifiesto por imagen](outputs/planning_20261111/dataset_manifest_audit.csv).
- [Grupos de duplicados](outputs/planning_20261111/duplicate_groups.json).
- [Script exacto de auditoría](outputs/planning_20261111/audit_dataset.py).

**Auditoría completa terminada:** 39853/39853 imágenes decodificadas, **0 ilegibles**, todas `224×224`, modo `L`, formato JPEG. No es una comprobación por muestreo. Se recorrieron las ocho clases, incluida contempt para auditar conflictos. La auditoría no ejecutó ViT, OpenFace ni SVC.

| Hallazgo sobre contenido idéntico | Ocho clases descargadas | Siete clases principales |
|---|---:|---:|
| Grupos de duplicados | 495 | 452 |
| Imágenes pertenecientes a esos grupos | 1130 | 1042 |
| Copias adicionales respecto a una por grupo | 635 | 590 |
| Grupos presentes en ambas particiones | 106 | 97 |
| Grupos con etiquetas fuente distintas | 149 | 129 |

**Un grupo no equivale a una imagen.** Estos totales describen el conjunto completo, no solamente validation. Los grupos por bytes y por píxeles coinciden también en sus miembros, no sólo en sus conteos. Hay imágenes idénticas con etiquetas incompatibles; esta ambigüedad anotacional se tratará antes de evaluar, sin elegir la etiqueta que favorezca al clasificador.

La auditoría verifica cabecera y decodificación completa con Pillow, registra dimensiones y modo, calcula SHA256 del archivo y SHA256 de los píxeles RGB decodificados más sus dimensiones. La segunda comprobación detecta contenido idéntico pese a diferencias de metadatos/compresión que no alteren los píxeles. No realiza reconocimiento de identidad ni demuestra independencia por sujeto; tampoco detecta todas las imágenes semejantes o versiones transformadas.

El modo `L` quedó confirmado en el recorrido exhaustivo, no sólo en las muestras iniciales. Convertir `L` a RGB replica la intensidad en tres canales; no recupera color perdido.

### 2.4 Tamaño previsto al aplicar la política de calidad

Se calculó una previsualización de las reglas de la sección 3.1 sobre el manifiesto auditado. Se revisan conflictos entre las **ocho etiquetas fuente**, aunque contempt se excluya del experimento. El informe de esta operación no contiene inferencia ni métricas del modelo:

- [Previsualización JSON](outputs/planning_20261111/exclusion_policy_preview.json).
- [Candidatos finales propuestos](outputs/planning_20261111/evaluation_candidate_plan.csv).
- [Exclusiones propuestas por imagen](outputs/planning_20261111/evaluation_exclusions_plan.csv).

| Clase | Candidatas de validation | Conflictos de etiqueta | Contenido también en train, tras quitar conflictos | N propuesto para evaluar |
|---|---:|---:|---:|---:|
| neutral | 486 | 2 | 1 | 483 |
| fear | 481 | 8 | 29 | 444 |
| anger | 484 | 4 | 8 | 472 |
| happiness | 491 | 0 | 0 | 491 |
| sadness | 477 | 6 | 9 | 462 |
| disgust | 480 | 4 | 21 | 455 |
| surprise | 470 | 8 | 6 | 456 |
| **Total** | **3369** | **32** | **74** | **3263** |

Flujo observado para las siete clases: **3369 candidatas → 3369 legibles → 3337 sin conflictos → 3263 sin solapamiento con train → 3263 tras deduplicación interna**. No quedan copias internas adicionales después de esos filtros. Los motivos de las columnas son disjuntos por el orden de la política; no se restan dos veces imágenes afectadas por más de un problema.

Por tanto, **N=3263 es el tamaño previsto para la ejecución principal con esta política**, no una cantidad ya evaluada. Se confirmará al generar y congelar el manifiesto del run. Son 237 imágenes menos que las 3500 de referencia, además de la diferencia de preprocesamiento. Se conservarán visibles los conteos originales y el criterio de selección: una depuración puede cambiar la dificultad del conjunto y no demuestra mejora de la distribución ni generalización.

SHA256 del manifiesto auditado: `01e4df9339cac34eb1849a50a83cd22eb62984f6fe52a99aca0ec671303e7415`. SHA256 del manifiesto de selección propuesto: `c70b14ecf2ee3959bb451a74fa29296e40f79614e77249c06ed7967d3718e3e9`. Una modificación de archivos o política obliga a regenerar ambos y revisar N.

### 2.5 Procedencia y preprocesamiento: qué está confirmado

El usuario identifica la descarga como AffectNetFiltered. El catálogo primario de Kaggle contiene `tanzeekhan/affectnetfiltered`, describe CLAHE y Gaussian Blur y declara un tamaño de 295918159 bytes, coincidente con el ZIP local. Esa coincidencia apoya la identificación, pero no demuestra igualdad de versión ni integridad frente a un checksum remoto. [S3]

SHA256 del ZIP local: `c52a0e5948dc9aba6b9ce81b9c4cc1d4405faf3c6f41849ec9ec3da797826bd0`. Su directorio central contiene 39853 imágenes y ningún README, CSV, JSON o PDF de procedencia. La configuración fijada y los metadatos remotos consultados se conservan en [external_sources_snapshot.json](outputs/planning_20261111/external_sources_snapshot.json).

Se debe registrar URL de descarga, versión del dataset, fecha y licencia declarada de la fuente. El contenido local y los píxeles no permiten reconstruir por sí solos parámetros de CLAHE/blur ni demostrar que se aplicaron uniformemente. La frase metodológica será: **preprocesamiento descrito por la distribución; parámetros e identidad respecto al AffectNet original no verificados**. No se volverán a aplicar CLAHE, blur, realce, recorte facial o augmentations en este experimento.

## 3. Protocolo que se congela antes de observar resultados

| Decisión | Especificación |
|---|---|
| Experimento | `Technical validation on AffectNetFiltered` |
| Modelo | `trpakov/vit-face-expression`, off-the-shelf |
| Pesos | Sin entrenamiento, fine-tuning, calibración ajustada ni modificación |
| Clases | Exactamente las siete de la tabla; `contempt` excluida |
| Fuente de evaluación | `val_class_Final`, con exclusiones auditadas |
| Función de train | Auditoría de separación; no se usa para ajustar ViT |
| Partición derivada | No necesaria con la estructura actual |
| Semilla | 42 para Python random, NumPy y torch |
| Preprocesamiento añadido | Decodificación RGB y processor del mismo commit del modelo |
| Predicción | `argmax` de los siete logits; softmax sin temperatura ajustada |
| Métrica primaria | Accuracy |
| Métricas secundarias | Macro-F1, Balanced Accuracy, precision/recall macro, UAR y métricas por clase |
| Selección de ejemplos | Reglas deterministas de las secciones 8–9, registradas en CSV |
| Selección de hiperparámetros | Ninguna sobre esta evaluación |
| OpenFace / SVC | Etapas diferidas, sin instalación ni ejecución en esta iteración |

La partición se denominará **provided validation split** o **validación suministrada por AffectNetFiltered**. La palabra «oficial» requiere documentación que establezca su correspondencia con la partición original. Si otra copia futura careciera de validation, un split estratificado por grupos de duplicados, seed 42 y proporción explícita se registraría como `derived validation split` en otro experimento; no reemplazará silenciosamente esta corrida.

### 3.1 Política de exclusión y denominadores

Aplicar estas reglas en orden, antes de inferencia, manteniendo originales intactos:

1. Registrar todos los archivos; excluir `desktop.ini` y otros no gráficos del conteo de imágenes.
2. Excluir `contempt` mediante mapping explícito a `null`, registrando conteos por partición.
3. Excluir imágenes no decodificables, con ruta, excepción y motivo. No repararlas silenciosamente.
4. Si un grupo idéntico por bytes o píxeles tiene etiquetas fuente incompatibles entre cualquiera de las ocho clases descargadas, poner sus miembros candidatos en cuarentena para este análisis. La exclusión de contempt como clase de salida no elimina su utilidad para detectar una contradicción de anotación. Registrar todas las filas del grupo, sin decidir la etiqueta por la predicción.
5. Excluir de evaluación los miembros de grupos idénticos presentes tanto en train como en validation. Conservar el dato original y el motivo en el manifiesto. Esta política de separación también protege un futuro análisis entrenado; con ViT congelado, el solapamiento local por sí solo no demuestra que sus pesos hayan visto esas imágenes.
6. Dentro de validation, conservar un único representante por contenido idéntico y etiqueta: la menor ruta relativa en orden lexicográfico. Las copias restantes quedan excluidas, no borradas.
7. Conservar en el manifiesto los motivos primario y secundarios de exclusión; el conteo de attrition utiliza el primer motivo para evitar restas dobles.

Publicar el flujo `N_candidatos → N_legibles → N_sin_conflictos → N_sin_solapamiento → N_final`, por clase y total. Los conteos originales de la sección 2 permanecen visibles. Si aparece una exclusión que vacía una clase, el informe principal de siete clases queda bloqueado y se describe el problema; no se presentará una métrica de seis clases con el mismo nombre.

La presencia de duplicados o una diferencia de distribución se documenta aunque no cambie la accuracy. Cualquier cambio posterior de esta política crea una nueva versión del protocolo y un nuevo run ID.

## 4. Entorno, modelo y ejecución reproducible

### 4.1 Entorno aislado

El launcher inspeccionado ofrece Python 3.13.5 y Python 3.12; no lista Python 3.11. La auditoría de datos utiliza el Pillow del entorno existente y lo registra; eso no constituye el entorno de inferencia.

Crear `.venv311` dentro de este proyecto con Python 3.11. No instalar paquetes en `env/`, en el Python global ni en el entorno de otros experimentos. Registrar la ubicación exacta de `sys.executable`. El `.gitignore` deberá excluir `.venv311/` antes de crearlo.

La pareja actualmente declarada `torch==2.6.0` / `torchvision==0.21.0` y `transformers==4.49.0` es un punto de partida, no un entorno ya validado. Comprobar disponibilidad de wheels Python 3.11 y la variante CUDA con las instrucciones oficiales; si se necesita otra versión, justificar y congelar la combinación realmente funcional. No cambiar a una versión supuesta de PyTorch por novedad. Añadir `tqdm` al entorno base; UMAP será opcional. [S4]

Comandos disponibles una vez instalado Python 3.11:

```powershell
Set-Location 'C:\Users\angel\source\repos\EmotionRecognition\foundational models\affectnet_replication'
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
# Incorporar tqdm al fichero de requisitos antes de la instalación definitiva.
.\.venv311\Scripts\python.exe -m pip check
.\.venv311\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements_freeze.txt
```

Guardar también el comando de instalación de los wheels y su índice: `pip freeze` no captura por sí solo driver, índice ni todas las condiciones del hardware. Probar imports de torch, torchvision, transformers, sklearn, NumPy, pandas, matplotlib, Pillow y tqdm dentro de `.venv311`. Medir CUDA disponible, nombre de GPU, driver, VRAM y CPU; no asumir que el hardware informado en una corrida previa sigue siendo el activo.

### 4.2 Revisión y mapping del modelo

El Hub consultado durante la planificación devuelve el commit:

```text
ef0bc6fc34241b6587e7e009e7711357be28c024
```

Es el commit candidato para congelar, no evidencia de pesos descargados o inferencia ejecutada. Cargar processor, configuración y pesos del mismo commit y registrar hashes de archivos locales. Si ese commit deja de ser accesible, registrar el bloqueo; no sustituirlo por `main` sin actualizar el protocolo.

| ID del modelo observado en config | Etiqueta del modelo | Etiqueta canónica |
|---:|---|---|
| 0 | angry | anger |
| 1 | disgust | disgust |
| 2 | fear | fear |
| 3 | happy | happiness |
| 4 | neutral | neutral |
| 5 | sad | sadness |
| 6 | surprise | surprise |

Este mapping procede de la configuración publicada, no del orden de las carpetas. En ejecución se imprimirá y archivará `model.config.id2label` y se comprobará la biyección con nuestras siete clases; cualquier discrepancia no resuelta detiene la inferencia. La revisión científica previa queda documentada por esta tabla y se verifica automáticamente contra el runtime. [S2]

El processor consultado declara tamaño 224, resize activo, normalización por canal con media y desviación 0.5 y `resample=2`. Se cargará con `AutoImageProcessor` en vez de reconstruirlo a mano; se guardará `processor.to_dict()` completo. [S2]

### 4.3 Ejecución y control de fallos

Usar `model.eval()`, `torch.inference_mode()`, FP32 como referencia y procesamiento determinista. Registrar flags de TF32, cuDNN, algoritmos deterministas y `CUBLAS_WORKSPACE_CONFIG`. La semilla no garantiza identidad entre todo hardware y toda versión de librería.

Realizar un piloto de 32 imágenes seleccionadas de forma determinista del manifiesto final para comprobar memoria, mapping y contratos. Las predicciones del piloto no se publican como evaluación final. Iniciar con batch 16; si hay OOM reducir 16 → 8 → 4 → 1 y registrar el cambio. No cambiar imágenes ni pesos para resolver memoria. El piloto medirá imágenes/segundo y permitirá estimar tiempo restante; este documento no promete duraciones sin medición.

La corrida final recorre el manifiesto congelado una vez. Guardar por batch progreso, filas completadas y fallos. Si una imagen que pasó auditoría falla durante inferencia, registrar el error y marcar la corrida incompleta. Sólo una exclusión documentada en un nuevo manifiesto permite producir una corrida final con otro N. No convertir silenciosamente errores de infraestructura en exclusiones favorables.

La extracción principal guardará logits, probabilidades y CLS antes del clasificador; las figuras posteriores reutilizan esos resultados. Atención se calcula aparte sólo para las imágenes seleccionadas y sin tocar los pesos. Una figura opcional que falle se marca `skipped` o `failed` con motivo, preservando métricas ya válidas.

## 5. Revisión del código actual y cambios necesarios

| Archivo / área | Disponible | Cambio necesario para este protocolo |
|---|---|---|
| `config.py` | Clases, seed y referencias del paper | Perfil explícito AffectNetFiltered con nombre experimental y particiones observadas |
| `src/inspect_dataset.py` | Inspección, manifest CSV, SHA256 y legibilidad | Admitir particiones explícitas; inventario exhaustivo; contempt excluida; hashes de píxeles, conflictos y attrition |
| `run_experiment.py` | Corridas separadas, metadatos y estados | Modo exclusivo ViT; prohibir instalación/ejecución OpenFace/SVC en ese modo; informe de dos experimentos |
| `src/vit_inference.py` | Inferencia RGB + processor; mapping y máximo softmax | Conservar source label, correctness, siete probabilidades, logits, sample ID y CLS; progreso persistente |
| `src/metrics.py` | Métricas agregadas/por clase y matrices de confusión | CSV de classification report; tabla específica; ECE, análisis de errores y gráficos solicitados |
| Nuevos módulos pequeños | Pendientes | `error_analysis.py`, `visualizations.py`, `embeddings.py`, `attention.py`, con responsabilidades separadas |
| `requirements.txt` | Versiones candidatas | Validar en 3.11, añadir tqdm y producir freeze real |
| `.gitignore` | Entorno anterior, datos y outputs | Añadir `.venv311/`; mantener datos y pesos fuera de Git |
| `README.md` | Flujo original AffectNet/OpenFace/SVC | Dar prioridad al perfil actual y sus comandos reales; preservar la documentación histórica |
| `tests/` | Pruebas de protocolo previo | Cubrir particiones reales, exclusión de contempt, probabilidades, CLS y consistencia de artefactos |

**Problema concreto:** hoy `read_folders()` reconoce train/training y val/valid/validation; no reconoce `train_class_Final` ni `val_class_Final`. Además, el runner completo intenta la fase OpenFace tras ViT. Por ello, indicar solamente la nueva ruta al comando anterior no implementa este protocolo.

Se propone un perfil `affectnetfiltered-vit` que cargue particiones y mapping explícitos desde `configs/affectnetfiltered_vit.json`. El perfil debe decidir las etapas permitidas de forma centralizada; no basta con ocultar OpenFace en el reporte.

**Interfaz propuesta, aún no implementada; no ejecutar como si ya existiera:**

```powershell
.\.venv311\Scripts\python.exe run_experiment.py --profile affectnetfiltered-vit --data-root 'D:\DATASETS\archive\AffectNetPrepared'
```

Implementar y probar primero esa interfaz, o documentar la interfaz equivalente finalmente elegida. Todos los comandos publicados como operativos deberán corresponder a flags existentes. La planificación no modifica los pipelines actuales.

## 6. Contratos de datos y estructura de salida

Cada corrida final usará una carpeta nueva; los outputs de planificación no se reutilizan como predicciones. El origen de todas las tablas y figuras será el mismo run ID y manifiesto.

```text
outputs/run_<UTC>_<id>/
  dataset_inspection.json
  protocol_snapshot.json
  experiment_metadata.json
  requirements_freeze.txt
  manifests/
    all_images.csv
    evaluation_manifest.csv
    exclusions.csv
  metrics/
    metrics_vit.json
    classification_report_vit.csv
    confusion_matrix_vit.csv
    confusion_matrix_vit_normalized.csv
    calibration_vit.json
    reliability_bins.csv
  predictions/
    predictions_vit.csv
    logits_vit.npy
    probabilities_vit.npy
    prediction_failures.csv
  embeddings/
    vit_cls_embeddings.npy
    vit_cls_labels.csv
    projection_coordinates.csv
    projection_metadata.json
  error_analysis/
    summary.json
    selected_examples.csv
    false_positives/
    false_negatives/
    correct_high_confidence/
  figures/
    <recursos definidos en sección 8>
    figure_manifest.json
  comparison.csv
  report.md
  ppt_results.txt
  logs/
    execution.log
    progress.json
  artifact_manifest.json
```

Los artefactos condicionales ausentes se listarán con motivo; no se crean matrices, embeddings ni mapas de atención ficticios para completar la estructura.

### 6.1 Identidad y alineación de filas

Cada muestra tendrá `sample_id` estable calculado a partir de la partición y ruta relativa; SHA256 del archivo y de píxeles se guardan como campos separados. La ruta no sustituye el hash de contenido ni viceversa.

`predictions_vit.csv` incluirá los campos requeridos:

```text
sample_id, filepath, source_label, canonical_true_label,
predicted_label, confidence, correct
```

Añadir `split`, `file_sha256`, `rgb_sha256`, `prediction_index` y las siete columnas `p_neutral, p_fear, p_anger, p_happiness, p_sadness, p_disgust, p_surprise`. Mantener temporalmente `true_label` como alias validado de `canonical_true_label` si lo necesita el módulo existente; nunca permitir divergencias entre ambos.

Logits, probabilidades y embeddings tendrán exactamente el mismo orden de filas que el CSV. `prediction_index` enlaza cada fila con los arrays; `vit_cls_labels.csv` repetirá ID, índice, etiqueta y ruta. Los arrays serán numéricos y podrán leerse con `allow_pickle=False`. Evitar redondeos destinados a la presentación en los datos base.

### 6.2 Metadatos mínimos

Registrar Python y ejecutable; versiones de torch, torchvision, transformers, sklearn, NumPy, pandas, matplotlib, Pillow, tqdm y UMAP si se usa; CUDA de torch, driver, GPU, VRAM, CPU y RAM; semillas; timestamps UTC y zona de presentación; comandos y argumentos; fuente y revisión del modelo; processor completo; perfil experimental; hashes del código, ZIP, manifiestos y pesos; conteos y exclusiones; N intentado/completado/fallido; batch size, precisión numérica y backend de atención; tiempos por fase; estados y errores.

Guardar el estado de Git y el hash de los archivos ejecutados: un commit no basta cuando el repositorio tiene cambios locales. `artifact_manifest.json` contendrá tamaño, SHA256, tipo y archivo fuente de cada entregable; las rutas del informe final serán absolutas y comprobadas.

## 7. Métricas, referencia y análisis de errores

### 7.1 Definiciones fijadas

El orden de toda matriz y array será `neutral, fear, anger, happiness, sadness, disgust, surprise`. Las filas de la matriz representan etiquetas verdaderas; las columnas, predichas.

- **Accuracy:** aciertos / N final.
- **Macro Precision / Recall / F1:** promedio no ponderado de los valores por clase, calculados con las siete etiquetas explícitas y política de divisiones por cero documentada.
- **UAR:** promedio de los siete recalls.
- **Balanced Accuracy:** definición de scikit-learn. Con soporte positivo en las siete clases, debe coincidir con UAR dentro de tolerancia numérica.
- **Normalización:** cada fila de la matriz se divide por el support de la clase verdadera, no por el total de predicciones de una columna.
- **Unidades:** fracciones [0,1] en JSON y arrays; porcentajes rotulados en tablas y figuras. El delta de accuracy se expresa en puntos porcentuales.

Los JSON incluirán N, support, clase ausente si la hubiera, orden, partición y hashes de los inputs. El CSV por clase tendrá precision, recall, F1 y support, más filas macro/weighted claramente identificadas.

### 7.2 Tabla con el paper

| Experiment | Dataset | Model | Accuracy | Macro-F1 | Balanced Accuracy | UAR |
|---|---|---|---|---|---|---|
| Schuller et al. reference | AffectNet original, subconjunto del paper | ViT-FER | 43.9% | N/R | N/R | N/R |
| Our technical validation | AffectNetFiltered, provided validation split depurado | ViT-FER | Pendiente | Pendiente | Pendiente | Pendiente |

`N/R` significa no reportado para esa comparación; `Pendiente` significa aún no ejecutado. No completar el paper con resultados de FER2013 de la model card ni con valores propios. [S1]

Calcular `delta_accuracy_pp = 100 × our_accuracy - 43.9`. Un delta positivo o negativo es una diferencia descriptiva entre resultados de conjuntos diferentes; no prueba mejora o degradación del método.

Como análisis adicional posterior a los entregables prioritarios, se pueden calcular intervalos descriptivos de incertidumbre mediante bootstrap estratificado por clase, seed 42 y 2000 remuestreos del manifiesto final depurado. Guardar método, unidad, réplicas y percentiles 2.5/97.5. No inventar intervalos para el paper, contrastar significación contra 43.9% ni interpretarlos como evidencia de generalización por sujeto. Si no hay independencia por identidad, señalar esa limitación.

### 7.3 Errores y selección reproducible

Publicar dos rankings diferentes, siempre dirigidos `TRUE → PRED`:

1. Los dos mayores conteos off-diagonal, con su conteo y porcentaje de fila, para la salida resumida.
2. Las tres mayores tasas off-diagonal de la matriz normalizada, con su conteo y support, para `top_confusions.png`.

Resolver empates por el orden canónico, primero TRUE y después PRED. Guardar también las dos clases con peor F1 y todas las clases empatadas en el mayor recall. No reemplazar tasas por conteos sin cambiar el rótulo.

Para errores multiclase, cada error es FP para la clase predicha y FN para la verdadera. Las colecciones «10 FP» y «10 FN» pueden compartir imágenes; no son veinte observaciones independientes. Elegir hasta diez errores diferentes por colección, equilibrando pares de confusión cuando sea posible y con desempate reproducible, y guardar una fila por uso en `selected_examples.csv`. Si hay menos de diez, mostrar la cantidad existente. Copiar bytes originales; nombrar las copias con sample ID para evitar colisiones.

## 8. Catálogo de figuras y prioridades

Reglas comunes: matplotlib, sin seaborn; fondo blanco; tipografía legible en diapositiva; colores suaves y estables por clase; PNG a 300 dpi y dimensiones explícitas; textos y colores consistentes. Las figuras analíticas de conteos o pipeline se derivan del manifiesto/protocolo; las que describen rendimiento se derivan de predicciones reales. No se necesita una predicción para mostrar que una imagen pertenece a una clase del dataset.

Las imágenes del dataset se mantienen intactas en disco. El layout puede escalarlas para visualización sin filtros cosméticos, recorte que oculte información ni alteración de contenido. Sólo los overlays interpretativos añaden contenido sobre la imagen. La visualización del dato original se diferenciará del tensor normalizado que recibe el modelo.

| Prioridad | Archivo en `figures/` | Pregunta que responde | Fuente y regla verificable |
|---|---|---|---|
| 1 | `dataset_examples.png` | ¿Qué imágenes se evaluaron? | 7 columnas × 4 muestras de validation final; seed 42 sobre rutas ordenadas; cabeceras canónicas; pie con N realmente evaluado |
| 1 | `confusion_matrix_vit_normalized.png` | ¿Dónde se concentra el error relativo por clase? | Matriz normalizada por TRUE; etiquetas y escala [0,1] o % coherentes |
| 1 | `f1_per_class_vit.png` | ¿Qué clases funcionan mejor/peor? | F1 real de las siete clases; support visible o en pie |
| 1 | `prediction_examples.png` | ¿Cómo difieren aciertos y errores según confidence? | Tres filas definidas en sección 8.1; TRUE, PRED y p sin modificar |
| 1 | `confusion_matrix_vit.png` | ¿Cuántas imágenes hay en cada confusión? | Matriz absoluta, suma=N |
| 1 | `class_distribution.png` | ¿Qué soporte tiene cada clase evaluada? | Conteos del manifiesto final; identificar partición y exclusión de contempt |
| 1 | `paper_vs_current_accuracy.png` | ¿Qué referencia contextual acompaña al resultado? | Dos barras reales: 43.9% publicado y nuestra accuracy; datasets y limitación explícitos |
| 2 | `experiment_summary_16x9.png` | ¿Cuál fue el experimento y su resultado? | Layout 16:9 con dataset, N, modelo, sin fine-tuning, cuatro métricas y referencia contextual |
| 2 | `top_confusions.png` | ¿Qué pares tienen mayor tasa de confusión? | Top 3 normalizado; tasa, conteo/support y hasta 3 ejemplos reales por par |
| 2 | `vit_inference_demo.gif` | ¿Cómo produce una predicción el pipeline? | 5–8 muestras, al menos 3 aciertos y 2 errores cuando existan; siete probabilidades auténticas |
| 3 | `vit_embedding_projection.png` | ¿Cómo se solapan las etiquetas en la representación? | Un punto por muestra evaluada; método y parámetros registrados |
| 3 | `vit_attention_examples.png` | ¿Qué distribución de atención se observa? | 2 aciertos y 2 errores; original/overlay, TRUE/PRED/p; extracción validada |
| 3 | `confidence_correct_vs_wrong.png` | ¿El clasificador se muestra confiado cuando falla? | Histogramas con mismos bins y normalización explícita, N de cada grupo y medias |
| 3 | `reliability_diagram.png` | ¿Cómo se corresponde confidence con frecuencia de acierto? | Bins, conteos y ECE sin recalibración ajustada |
| Opcional | `confusion_matrix_build.gif` | ¿Cómo se acumula la matriz? | Acumulación de predicciones en orden registrado; último frame idéntico a la matriz final |

Los rótulos de comparación incluirán **“Different dataset preprocessing; not a direct replication.”**. El resumen 16:9 incluirá **“Technical validation — different dataset preprocessing; not a direct replication.”**. Para PNG 16:9 se fija como salida objetivo 3840×2160, equivalente a 12.8×7.2 pulgadas a 300 dpi; comprobar tamaño después de guardar, sin recorte automático que cambie la proporción.

### 8.1 Reglas de ejemplos y GIF

Para `prediction_examples.png`, seleccionar hasta cinco ejemplos únicos por fila:

- Aciertos de mayor confidence, orden descendente.
- Aciertos de menor confidence, orden ascendente, excluyendo los usados en la primera fila.
- Errores de mayor confidence, orden descendente.

Desempatar por sample ID. «Alta» y «baja» describen posiciones relativas de ese grupo, no probabilidades calibradas ni umbrales elegidos después de ver las imágenes. Se informará la selección por extremos, por lo que estos paneles no estiman frecuencias poblacionales.

En `dataset_examples.png`, usar selección aleatoria seed 42 por clase entre las muestras realmente evaluadas. Si alguna clase tiene menos de cuatro, mostrar las existentes con celdas vacías rotuladas; no duplicar imágenes para rellenar.

En `top_confusions.png`, elegir hasta tres muestras del par mediante seed 42 sobre sample IDs ordenados, sin elegir por apariencia. Si hay menos, indicar el N disponible.

Para el GIF principal, escoger tres aciertos y dos errores con un muestreo determinista que favorezca cubrir clases/pares distintos; añadir hasta tres muestras sólo si aportan cobertura. Mantener cada ejemplo al menos 3 segundos y registrar duración, orden e IDs. Usar 1280×720 y tipografía grande; barras en orden canónico, eje fijo [0,1], clase predicha resaltada, TRUE/PRED/confidence y correcto/error mediante símbolos soportados por la fuente o texto equivalente. La paleta y compresión del GIF no cambiarán los valores escritos.

Si faltan tres aciertos o dos errores, usar los disponibles y documentar la imposibilidad del requisito; nunca fabricar errores. La atención es un panel adicional sólo si ya está validada y no retrasa este GIF. Las selecciones y reglas quedan archivadas en `selected_examples.csv` y `figure_manifest.json`.

## 9. Análisis complementarios: embedding, atención y calibración

### 9.1 CLS justo antes del clasificador

Extraer exactamente el tensor que recibe la capa clasificadora del checkpoint cargado, tras la normalización final del encoder. Una implementación viable es un hook de entrada de `model.classifier` durante la inferencia principal, retirado al terminar. Verificar la arquitectura real y el tipo de capa antes de usar el hook.

No asumir que `outputs.hidden_states[-1][:,0,:]` es idéntico al CLS posnormalización: comprobarlo contra el código de la versión fijada. La prueba decisiva es que aplicar el clasificador a los embeddings guardados reproduzca los logits originales, con tolerancias FP32 declaradas (`atol=1e-5`, `rtol=1e-4` como criterio inicial). Si falla, corregir la extracción antes de publicar la proyección.

Guardar una matriz `N×D` float32; D se obtiene del tensor real, no se impone como 768 aunque la configuración consultada declare ese tamaño. Ninguna etiqueta se utiliza para aprender la proyección no supervisada.

Si UMAP ya forma parte del entorno congelado y funciona, usar `n_components=2`, `n_neighbors=15`, `min_dist=0.1`, `metric='euclidean'`, `random_state=42`, sin búsqueda de parámetros por apariencia. Si no está disponible, usar PCA de dos componentes sobre embeddings centrados, sin escalar dimensiones por desviación, y registrar varianza explicada. Ante un fallo de UMAP, declarar el fallo y el fallback, no etiquetar PCA como UMAP. Guardar coordenadas y método; no extraer conclusiones psicológicas ni usar separación visual como métrica de clasificación. [S5]

### 9.2 Attention rollout

Verificar en el runtime que `output_attentions=True` devuelve tensores reales. Si el backend optimizado no los proporciona, probar `attn_implementation='eager'` con el mismo commit y los mismos pesos, conforme al soporte de la versión fijada. Comprobar que logits y predicción concuerdan con la inferencia base dentro de tolerancia documentada; no dar por hecho que cambiar backend conserva exactamente los números. [S4]

Para cada capa l, promediar cabezas `A_l`, añadir la identidad y normalizar filas: `A'_l = row_normalize(A_l + I)`. Acumular `R_0=I`, `R_l=A'_l @ R_(l-1)`. Extraer la fila CLS hacia los tokens de imagen, excluyendo CLS, y reconstruir la rejilla según número real de patches. No utilizar una rejilla fija si las dimensiones del runtime no coinciden. Documentar interpolación al tamaño original, normalización del mapa y alpha del overlay. Este rollout sigue una aproximación publicada de flujo de atención, no una explicación causal. [S6]

Elegir dos aciertos y dos errores mediante una regla reproducible, con IDs en el manifiesto. Guardar los mapas numéricos además del PNG. Validar shape de capas/cabezas/tokens, finitud, no negatividad y filas de atención que sumen aproximadamente uno; rechazar tensores incoherentes con motivo. Un mapa realmente uniforme se informa como tal: no dividir por un rango cero ni fabricar contraste para que parezca informativo. Una normalización visual por imagen permite ver patrones, pero no comparar intensidad absoluta entre personas.

Añadir **“Attention visualization is interpretative, not causal.”**. No atribuir el resultado a una región como causa probada, no afirmar activación de emociones internas y no sustituir un mapa no disponible por una saliencia inventada. Si falla la extracción fiable, omitir esta figura y registrar la causa.

### 9.3 Confidence y calibración

Confidence es `max(softmax(logits))` sin calibración aprendida. Calcular media de confidence en aciertos y en errores, con N por grupo; si un grupo está vacío, su media es `null`, no cero.

Para reliability diagram y ECE top-label usar 10 bins iguales, con bordes [0,0.1,…,1], intervalos cerrados a la izquierda y abiertos a la derecha, salvo el último que incluye 1. Para cada bin registrar n, confidence media y accuracy; bins vacíos se marcan sin estimación. Definir:

```text
ECE = suma_b (n_b / N) * abs(accuracy_b - mean_confidence_b)
```

Comprobar que los bins particionan exactamente N y que las probabilidades por muestra son finitas, están en [0,1] y suman uno dentro de tolerancia. Guardar ECE como fracción y rotular su porcentaje si se muestra. ECE depende de bins y muestra: se presenta como diagnóstico empírico, no como prueba de «conciencia» de incertidumbre. No ajustar temperatura con esta evaluación final. [S7]

## 10. Secuencia de implementación y puntos de control

| Paso | Trabajo | Evidencia para pasar al siguiente |
|---|---|---|
| P0 | Auditoría local de esta planificación | Conteos, legibilidad, hashes y evidencia en sección 2 |
| P1 | Adaptar particiones, mapping y perfil exclusivo ViT | Manifest reproducible; contempt excluida; OpenFace/SVC inaccesibles desde el perfil |
| P2 | Crear y validar `.venv311` | Ejecutable 3.11, imports, pip check, freeze y prueba CUDA/CPU |
| P3 | Congelar protocolo, evaluación y commit del modelo | JSON con hashes, mapping verificado, N por clase y exclusiones |
| P4 | Piloto de infraestructura | Siete logits/probabilidades, CLS válido, memoria y tiempos medidos |
| P5 | Inferencia final y métricas | Cobertura completa del manifiesto, CSV/arrays/JSON coherentes, classification report y matrices |
| P6 | Figuras prioridad 1 | Archivos verificables y selección registrada; luego prioridad 2 |
| P7 | Resumen, tabla y GIF de prioridad 2 | Artefactos reales listos para presentación y `ppt_results.txt` |
| P8 | Análisis de prioridad 3 | Embedding, atención y calibración; cada omisión documentada |
| P9 | Cierre y entrega para revisión del investigador | Informe con rutas absolutas y límites; detenerse antes de OpenFace/SVC |

Si existe presión de tiempo, entregar P5–P7 y marcar P8 pendiente con detalle. No sacrificar métricas completas ni coherencia del N por producir una animación. La animación acumulativa de confusión se realiza sólo después de los entregables requeridos y con tiempo disponible.

La revisión solicitada antes de OpenFace/SVC es un punto de control del usuario. No es necesario pedir aprobación para cada cambio reversible del pipeline ViT; sí se conserva el límite explícito de no continuar a esas otras fases.

## 11. Criterios de aceptación y pruebas útiles

### Datos y protocolo

- [x] La raíz absoluta existe y se registran ambas particiones reales y sus ocho carpetas.
- [x] El total observado cuadra con la suma por clase y por extensión.
- [x] Todas las imágenes que integran la evaluación se decodifican y tienen hashes; los fallos están registrados.
- [x] `happy→happiness`, `sad→sadness`, `contempt→excluida`; siete etiquetas efectivas.
- [x] No hay contenido idéntico entre train y evaluación final; no hay duplicados repetidos ni conflictos sin resolver en evaluación.
- [x] N y support finales se congelan antes de inferencia; se diferencia partición suministrada de validación oficial.

### Modelo y resultados

- [x] Python 3.11 y dependencias efectivas documentados en un entorno independiente.
- [x] Processor y pesos provienen del mismo commit; id2label registrado y compatible.
- [x] El perfil de este experimento no llama instaladores ni funciones OpenFace/SVC.
- [x] Cada sample ID evaluado aparece una vez; no hay muestras extra, pérdidas de filas ni resultados del piloto mezclados.
- [x] Logits y probabilidades tienen shape `N×7`; argmax y confidence coinciden con el CSV, con tolerancia justificada.
- [x] Los embeddings tienen N filas y reproducen los logits al pasar por el clasificador.
- [x] La matriz suma N; su diagonal/N reproduce Accuracy; filas reproducen support y recall.
- [x] UAR y Balanced Accuracy coinciden con las siete clases presentes; Macro-F1 coincide con la media de siete F1.
- [x] Las métricas se pueden recalcular desde los artefactos guardados sin volver a ejecutar el modelo.
- [x] La tabla tiene dos experimentos, referencias N/R y delta descriptivo, sin afirmaciones comparativas directas.

### Material para tesis/PPT

- [x] Figuras de prioridad 1 y 2 abren, contienen N correcto, rótulos legibles, unidades y limitaciones.
- [x] PNG declaran 300 dpi; el resumen mantiene 16:9 y 3840×2160.
- [x] Ejemplos y copias enlazan con IDs reales; no se infieren atributos sensibles ni estados internos.
- [x] GIF usa probabilidades originales; existe mezcla de aciertos/errores si los datos lo permiten.
- [x] Atención, si se publica, procede de tensores verificables y tiene advertencia interpretativa.
- [x] ECE/bins y proyección son reproducibles y su método aparece en la figura y en metadatos.
- [x] Cualquier recurso opcional omitido tiene un estado y motivo; no se presenta como generado.
- [x] `artifact_manifest.json`, `report.md` y `ppt_results.txt` enlazan archivos existentes del mismo run ID.

Las pruebas automatizadas usarán fixtures temporales para verificar el contrato, sin convertir sus scores en resultados científicos. Probar específicamente los nombres `train_class_Final`/`val_class_Final`, corrupción, contempt, duplicados por píxeles con bytes distintos, conflictos de etiqueta, mapping de logits no posicional, límites de bins ECE, top confusiones normalizadas frente a conteos y alineación de arrays. Un test del perfil comprobará que intentar entrar en OpenFace/SVC falla antes de ejecutar cualquier proceso. El piloto real valida imports y modelo; las pruebas con fixtures no lo sustituyen.

## 12. Riesgos concretos y respuestas

| Situación | Respuesta prevista | Consecuencia para el informe |
|---|---|---|
| No hay Python 3.11 | Instalar un intérprete local y crear `.venv311`; conservar entornos previos | Inferencia pendiente hasta tener entorno verificable |
| CUDA/wheel incompatible | Validar pareja torch/vision; permitir CPU con registro explícito | Cambia el tiempo, no las imágenes ni pesos |
| Memoria insuficiente | Reducir batch; atención en batch 1 sólo para seleccionadas | Registrar configuración real |
| Imagen corrupta o contenido conflictivo | Manifiesto de exclusiones y nuevo N antes de evaluar | Reportar attrition por clase |
| Duplicados entre particiones | Aplicar política predefinida sin borrar originales | Previsualización auditada: 3263 de 3369 candidatas; confirmar al congelar |
| Procedencia incompleta | Conservar ZIP/hash y citar distribución; marcar parámetros no verificados | Validación técnica sin equivalencia con AffectNet original |
| Hub inaccesible | Reutilizar sólo caché del commit exacto si está íntegra; si falta, registrar bloqueo | Ninguna predicción inventada |
| Atención no fiable | Omitir figura con motivo y conservar métricas | Limitación técnica explícita |
| UMAP ausente | PCA documentado, sin instalar extras que retrasen lo principal | Título y metadatos indican PCA |
| Pocos aciertos/errores para un panel | Mostrar cantidad disponible y dejar constancia | Sin duplicaciones ni errores fabricados |

## 13. Entrega para presentación y cierre de esta iteración

`ppt_results.txt` contendrá sólo datos reales de la corrida: Dataset, Model, N, Classes, Fine-tuning, Accuracy, Macro-F1, Balanced Accuracy, UAR, Worst class, Best class, Largest confusion, Paper reference y Methodological limitation. Se definirá «Best/Worst class» por F1, y se añadirá mejor recall como dato separado si procede. Los empates quedarán visibles.

La salida final de la implementación seguirá los doce puntos pedidos: ruta; distribución; N evaluado; mapping; cuatro métricas; siete F1; dos principales confusiones; referencia 43.9%; accuracy propia; delta con advertencia; rutas absolutas de todos los recursos generados; errores y límites. Las métricas pendientes no se rellenan en este plan.

**Condición de cierre:** resultados ViT y recursos requeridos o sus omisiones justificadas, disponibles para revisión. El siguiente paso científico posterior será decidir, con esa evidencia, si conviene avanzar a OpenFace/SVC o preparar una reproducción con el conjunto original; esa decisión no forma parte de esta ejecución ViT.

## 14. Fuentes y evidencia

Consulta de fuentes técnicas: 11 de noviembre de 2026. Se archivarán respuestas/configuraciones consultadas cuando se congele el run.

- **S1.** Schuller et al. (2024), [Affective Computing Has Changed: The Foundation Model Disruption, arXiv v1](https://arxiv.org/html/2409.08907v1), sección 2.1, tablas 3–4. Fuente de las siete clases, tamaño de referencia y 43.9%.
- **S2.** Autor del checkpoint, [model card](https://huggingface.co/trpakov/vit-face-expression), [config del commit candidato](https://huggingface.co/trpakov/vit-face-expression/blob/ef0bc6fc34241b6587e7e009e7711357be28c024/config.json) y [processor del mismo commit](https://huggingface.co/trpakov/vit-face-expression/blob/ef0bc6fc34241b6587e7e009e7711357be28c024/preprocessor_config.json). Configuración comprobada mediante HTTP; commit resuelto por la API del Hub durante la planificación.
- **S3.** Distribuidor, [AffectNetFiltered en Kaggle](https://www.kaggle.com/datasets/tanzeekhan/affectnetfiltered) y [catálogo API](https://www.kaggle.com/api/v1/datasets/list?search=affectnetfiltered). La API devuelve descripción breve, identificador y tamaño coincidente; queda pendiente confirmar la versión exacta de la descarga del usuario.
- **S4.** Documentación primaria: [ViT en Transformers 4.49.0](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/vit) y [versiones anteriores de PyTorch](https://pytorch.org/get-started/previous-versions/). El soporte documentado de attentions se comprobará además en el runtime real.
- **S5.** [Documentación de UMAP](https://umap-learn.readthedocs.io/en/latest/parameters.html) y [PCA en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Los parámetros del análisis se fijan en este protocolo; no se seleccionan por apariencia.
- **S6.** Abnar y Zuidema (2020), [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/). Base metodológica del rollout y de sus límites como explicación.
- **S7.** Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). Contexto metodológico para calibración; este experimento no ajusta temperature scaling.

Los JSON y CSV locales enlazados en sección 2 son evidencia de esta máquina, no fuentes bibliográficas. Están bajo `outputs/`, excluido de Git; las cifras esenciales se incluyen aquí para que el plan siga siendo interpretable cuando se versiona sin los artefactos locales.
