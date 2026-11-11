# AffectNet: AU + SVC frente a ViT-FER

Proyecto aislado en `foundational models/affectnet_replication` para la comparación
visual de Schuller et al. (2024),
[arXiv:2409.08907v1, sección 2.1 y tablas 3–4](https://arxiv.org/html/2409.08907v1).
No se copian métricas del paper como resultados propios. Las etapas sin datos o
dependencias quedan documentadas con valores nulos y motivo.

**Plan activo (11 de noviembre de 2026):**
[PLAN_EXPERIMENTO_DOCTORAL.md](PLAN_EXPERIMENTO_DOCTORAL.md).
La copia local está en `D:\DATASETS\archive\AffectNetPrepared`, con carpetas
`train_class_Final` y `val_class_Final`. El plan actual se titula
`Technical validation on AffectNetFiltered` y prioriza sólo ViT-FER. Detalla
las adaptaciones pendientes antes de usar el runner anterior con estos datos,
los controles metodológicos y los recursos para la presentación. Los comandos
del flujo completo documentado abajo corresponden al proyecto previo.

## Datos encontrados en esta máquina

La inspección inicial del repositorio encontró VREED en
`EmotionRecognition/archive/` y `archive.zip` (cuestionarios, eye tracking,
ECG/GSR), capturas ToF en `captures/ToF/` e imágenes de reconstrucción de rocas
en `data/`. No encontró AffectNet, ni `FaceLandmarkImg` en el repositorio o PATH.
VREED no se transforma en AffectNet y las capturas de rocas no se usan como
expresiones faciales. Cada corrida vuelve a inspeccionar los archivos reales y
guarda un inventario, en vez de tomar esta descripción como entrada.

El entorno existente `env/` utiliza Python 3.13.5; también está instalado Python
3.12. No se encontró Python 3.11 en el launcher. Hay PyTorch, torchvision y GPU
CUDA en el entorno existente, pero no transformers. La ejecución inicial usa
ese entorno sólo para inspección, metadatos y esquema del pipeline. No valida
las dependencias objetivo ni constituye una corrida de los modelos.

## Preparar el entorno objetivo

Ejecutar desde PowerShell en esta carpeta (las comillas preservan el espacio en
`foundational models`):

```powershell
Set-Location 'C:\Users\angel\source\repos\EmotionRecognition\foundational models\affectnet_replication'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si falta Python 3.11, instalarlo desde
[python.org](https://www.python.org/downloads/release/python-3119/) y comprobar
`py -0p`. No es necesario cambiar el Python del resto del repositorio. Para
usar CUDA, elegir los wheels de **las mismas versiones** de torch/torchvision
según las [instrucciones de PyTorch](https://pytorch.org/get-started/previous-versions/#v260).
El programa registra las versiones efectivamente utilizadas y la GPU.

## Inspección antes de ejecutar modelos

```powershell
.\.venv\Scripts\python.exe run_experiment.py --profile legacy-comparison --inspect-only
.\.venv\Scripts\python.exe run_experiment.py --profile legacy-comparison --data-root 'D:\Datasets\AffectNet' --inspect-only
```

La autodetección de raíz sólo se acepta si hay una única candidata con imágenes
y nombre AffectNet. Si hay varias, indicar `--data-root`. Los formatos admitidos
son:

* `train/<etiqueta_semántica>/...` y `validation/<etiqueta_semántica>/...`.
  Se detectan las particiones por nombres inequívocos train/training y
  validation/val/valid. Nunca se ordenan carpetas alfabéticamente para asignar IDs.
* Un CSV con columnas filepath, label y split, usando `--manifest`.
* CSV separados usando `--train-csv` y `--validation-csv`; no necesitan columna
  split. Las rutas se resuelven respecto a `--data-root`.

Se detectan columnas sólo si hay una correspondencia inequívoca entre los
nombres conocidos y el archivo real. Para nombres distintos, usar
`--filepath-column`, `--label-column` y `--split-column`. Por ejemplo, los CSV
nativos con `subDirectory_filePath` y `expression` se pueden seleccionar
explícitamente. Los CSV se leen como texto para conservar los IDs originales.

Las siete clases canónicas son, en este orden:
`neutral, fear, anger, happiness, sadness, disgust, surprise`.
Se aceptan los sinónimos angry/happy/sad. Los números **siempre** requieren un
JSON explícito `--label-map`, verificado frente a la documentación de la copia
local. Un valor `null` excluye una etiqueta adicional, con registro de las filas
excluidas. Un formato distinto, por ejemplo anotaciones NPY, se registra como
parte del inventario pero no se interpreta automáticamente: preparar primero
un manifiesto verificado, conservando su documentación de procedencia.

La inspección produce `class_name`, `class_id` original, ID canónico separado,
`n_images` y `n_valid` por partición, verifica decodificación RGB de cada imagen
y calcula SHA256 para detectar fuga entre train y validation. Compara con las
500 imágenes válidas por clase (3500) y con el train de la tabla 3. Documenta las
diferencias **antes** de llamar a los modelos. Coincidir en tamaño no demuestra
que sean exactamente las mismas imágenes del paper. Si no hay datos, los
conteos de clase son desconocidos (`null`), no ceros observados.

## Ejecutar el experimento

```powershell
.\.venv\Scripts\python.exe run_experiment.py --profile legacy-comparison --data-root 'D:\Datasets\AffectNet'
```

Para CSV y etiquetas numéricas:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --profile legacy-comparison --data-root 'D:\Datasets\AffectNet' --train-csv 'D:\Datasets\AffectNet\training.csv' --validation-csv 'D:\Datasets\AffectNet\validation.csv' --filepath-column subDirectory_filePath --label-column expression --label-map '.\data\verified_label_map.json'
```

`verified_label_map.json` lo debe proporcionar quien conozca la codificación de
esa distribución: el proyecto no crea un mapping numérico supuesto.

ViT utiliza [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression),
su image processor, imágenes RGB y evaluación sin fine-tuning. Antes de predecir
comprueba la biyección de `model.config.id2label` con las siete clases. Registra
configuración, processor y commit HF resuelto; para repetirlo exactamente usar
`--model-revision <commit_registrado>`. La primera ejecución con datos descarga
los pesos desde Hugging Face. `confidence` es el máximo softmax, no una
probabilidad calibrada.

ViT guarda `metrics/metrics_vit.json`, todas las predicciones, fallos y matrices
de confusión absoluta y normalizada por etiqueta verdadera. Las métricas JSON
se expresan en [0,1]; la tabla comparativa y figuras usan porcentajes. UAR y
macro recall usan siempre siete clases; balanced accuracy usa la definición de
scikit-learn sobre clases con soporte. Una clase ausente se advierte y aparece
en el JSON; no se interpreta la matriz normalizada de esa clase como evidencia
de rendimiento.

## OpenFace en Windows

Se usa OpenFace **2.x**, proyecto `TadasBaltrusaitis/OpenFace`; no el paquete
homónimo de reconocimiento de identidad. Se busca `FaceLandmarkImg.exe` en el
repositorio, PATH, `OPENFACE_HOME`, `C:\OpenFace` y
`C:\Program Files\OpenFace`. Puede indicarse `--openface-exe`.

Las instrucciones oficiales están en
[Windows Installation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation)
y [Command line arguments](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments).
Para preparar los binarios oficiales 2.2.0 dentro de este proyecto:

```powershell
New-Item -ItemType Directory -Path '.\vendor' -Force
Invoke-WebRequest -Uri 'https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_v2.2.0_win_x64.zip' -OutFile '.\vendor\OpenFace_v2.2.0_win_x64.zip'
Expand-Archive -LiteralPath '.\vendor\OpenFace_v2.2.0_win_x64.zip' -DestinationPath '.\vendor\OpenFace-2.2.0'
Get-ChildItem -LiteralPath '.\vendor\OpenFace-2.2.0' -Recurse -Filter 'FaceLandmarkImg.exe'
Get-ChildItem -LiteralPath '.\vendor\OpenFace-2.2.0' -Recurse -Filter 'download_models.ps1'
```

Instalar el [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
si falta el runtime requerido por los binarios. En la carpeta extraída que
contenga el ejecutable, descargar los modelos CE-CLM siguiendo
[Model download](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download):

```powershell
# Desde la carpeta que contenga el script oficial:
powershell -NoProfile -ExecutionPolicy Bypass -File '.\download_models.ps1'
# Desde affectnet_replication, pasando la ruta obtenida arriba:
.\.venv\Scripts\python.exe run_experiment.py --profile legacy-comparison --data-root 'D:\Datasets\AffectNet' --openface-exe 'RUTA_COMPLETA\FaceLandmarkImg.exe'
```

El script usa imágenes independientes (`FaceLandmarkImg -f <imagen> -aus`),
no seguimiento de vídeo. Mantiene una carpeta de salida única por imagen,
guarda logs, exige `success=1` si la salida incluye ese campo, selecciona el rostro de mayor confianza y valida
las columnas AU realmente presentes: presencia binaria `_c`, intensidad `_r`
en [0,5], sin NaN. Descarta y registra los fallos, sin imputarlos como ceros.
Comprobará si aparecen las 35 variables del paper, pero usará y documentará el
esquema real. Cambios de esquema interrumpen la fase para evitar mezclar vectores.
No se promete equivalencia con la versión histórica del paper: registra el hash
del ejecutable. El CSV estático oficial 2.2.0 incluye `face, confidence` pero no
`success`: en ese formato, una fila verifica una salida del detector de rostros,
no el éxito del ajuste de landmarks. Esa limitación queda registrada; las
imágenes sin filas de rostro se descartan. No se presupone un umbral adicional
de confianza, que el paper tampoco especifica. Véase el
[serializador oficial](https://github.com/TadasBaltrusaitis/OpenFace/blob/OpenFace_2.2.0/lib/local/Utilities/src/RecorderCSV.cpp).
La invocación por imagen prioriza trazabilidad y puede ser lenta
en el train completo.

## SVC y separación de datos

Se entrenan SVC linear/RBF, con y sin `class_weight='balanced'`, buscando
exactamente `C=[0.01,0.1,1,10,100]`. La selección maximiza accuracy mediante
StratifiedGroupKFold **sólo en train**, agrupando por SHA256 para que una imagen
idéntica no aparezca en ambos lados de un fold. Se usan hasta cinco folds,
reducidos según grupos por clase. Si no pueden conservar las siete clases,
la fase se detiene con un motivo explícito. Luego se reajusta en todo train y
se evalúa validation una sola vez. Se guardan las asignaciones internas, scores
de búsqueda, mejor C, modelos joblib y predicciones. Los SVC no calculan
probabilidades; su campo confidence queda vacío.

Por defecto no se escalan AUs; el paper no especifica completamente escalado,
gamma RBF ni el protocolo de selección interna. Se registra `gamma='scale'`.
`--scale-features` activa StandardScaler ajustado dentro de cada fold, y queda
documentado como variante metodológica. No se declara reproducción exacta.
La búsqueda exacta de SVC sobre cientos de miles de imágenes puede ser costosa;
`--n-jobs 1` evita multiplicar el consumo de memoria por defecto. No se muestrea
el dataset ni se sustituye SVC por una aproximación silenciosamente.

## Resultados y figuras

Cada corrida crea `outputs/run_<UTC>_<id>/` y actualiza `outputs/latest_run.json`.
No reutiliza métricas antiguas ni mezcla resultados entre corridas.

```text
outputs/run_.../
  inspection/{repository_inventory.json,dataset_report.json,class_distribution.csv,...}
  metrics/{metrics_vit.json,metrics_svc_*.json,cv_svc_*.json,openface_summary.json,...}
  predictions/{predictions_vit.csv,predictions_svc_*.csv,*failures.csv,...}
  figures/{comparison_accuracy.png,comparison_macro_f1.png,
           confusion_matrix_vit.png,confusion_matrix_vit_normalized.png,
           pipeline_comparison.png,...}
  openface_features.csv
  models/*.joblib
  common_subset/comparison.csv
  comparison.csv
  comparison.md
  experiment_metadata.json
  report.md
```

Los archivos de fases no ejecutadas no se fabrican. `comparison.csv` conserva
los cinco modelos con columnas OUR RESULT, PAPER RESULT y delta de accuracy en
puntos porcentuales; resultados ausentes quedan vacíos. Las figuras PNG usan
300 dpi, fondo blanco y colores suaves. El diagrama se puede crear sin datos;
los gráficos de accuracy/F1 sólo se crean si al menos un modelo fue evaluado,
marcando cada modelo pendiente. Las matrices sólo se crean desde predicciones
reales.

La tabla principal informa la población de cada modelo: ViT sobre validation
legible y SVC sobre imágenes aceptadas por OpenFace. También se evalúan ambos
sobre la intersección de éxitos en `common_subset/` cuando están disponibles;
esa tabla usa el mismo conjunto de imágenes. Los descartes se documentan y no
se oculta la diferencia de poblaciones.

Los metadatos incluyen Python, paquetes, CUDA/GPU, semillas 42 (random, NumPy,
torch), argumentos, hashes del código y manifiesto, imágenes procesadas y
fallidas, tiempos y estados. Los estados bloqueados por falta de datos producen
exit code 2, fallos de modelos exit code 3 y una corrida completa exit code 0.
`--inspect-only` termina con 0 aunque el informe detecte datos pendientes.

## Verificar el código

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas usan fixtures temporales, nunca generan resultados experimentales
ni descargan pesos. Comprueban mapping, cálculo de métricas, detección de fuga,
conteos, rechazo de codificaciones ambiguas y selección interna sin imágenes
compartidas. La ejecución inicial sin AffectNet verifica el recorrido bloqueado,
los metadatos y el diagrama; no demuestra inferencia ViT u OpenFace funcional.

