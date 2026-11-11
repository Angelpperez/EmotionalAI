# Registro de ejecución: Technical validation on AffectNetFiltered

Fecha: 11 de noviembre de 2026. Run ID: `run_20261111T150020Z_dec91d26`.

Implementación del [protocolo 1.0](PLAN_EXPERIMENTO_DOCTORAL.md) y su evaluación
real. OpenFace y SVC quedan pendientes de revisión del investigador.

## Datos y población evaluada

Raíz: `D:\DATASETS\archive\AffectNetPrepared`.

Se auditaron las 39853 imágenes de las dos particiones y ocho clases:
todas legibles, JPEG de 224×224 y modo L. La evaluación excluye contempt;
la inspección sí considera esa etiqueta para detectar conflictos.

De 3369 candidatas de las siete clases en `val_class_Final`, se pusieron
en cuarentena 32 por contenido idéntico con etiquetas contradictorias y se
excluyeron otras 74 por coincidencia con contenido de train. Los motivos
se aplicaron en ese orden y no se contaron dos veces. Quedaron **3263**
imágenes únicas, coincidentes con la selección propuesta antes de inferencia.
Los hashes de las **39853 imágenes originales** se comprobaron de nuevo al
final y no cambiaron.

## Resultados reales

| Métrica | OUR RESULT |
|---|---:|
| N | 3263 |
| Aciertos | 1042 |
| Errores | 2221 |
| Accuracy | 31.9338% |
| Macro-F1 | 27.8783% |
| Balanced Accuracy | 31.7338% |
| UAR | 31.7338% |

| Clase | N evaluado | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| neutral | 483 | 30.4196% | 18.0124% | 22.6268% |
| fear | 444 | 38.9771% | 49.7748% | 43.7191% |
| anger | 472 | 23.6097% | 47.6695% | 31.5789% |
| happiness | 491 | 51.1840% | 57.2301% | 54.0385% |
| sadness | 462 | 23.1228% | 41.9913% | 29.8232% |
| disgust | 455 | 71.4286% | 1.0989% | 2.1645% |
| surprise | 456 | 46.7742% | 6.3596% | 11.1969% |

Principales confusiones por conteo: **disgust → anger**, 221/455 (48.5714%
de disgust), y **neutral → sadness**, 190/483 (39.3375% de neutral).
La tercera por tasa normalizada fue **sadness → anger**, 152/462 (32.9004%).

La referencia de Schuller et al. es 43.9% sobre otro subconjunto/preprocesamiento.
El delta descriptivo es **-11.9662 puntos porcentuales**. Las métricas secundarias
del paper se mantienen como N/R. No se interpreta el delta como comparación
experimental directa ni como prueba de superioridad o degradación del método.

## Confidence, representación y atención

Confidence media en aciertos: **0.841673**; en errores: **0.761131**.
ECE top-label, 10 bins de igual ancho: **0.467513** (46.7513%). Son diagnósticos
del clasificador sobre esta población. No se ajustó temperature scaling.

Se guardaron 3263 CLS de 768 dimensiones, extraídos justo antes del clasificador.
Los logits reconstruidos desde esos vectores difieren de los originales como
máximo en **3.814697265625e-06**, dentro de las tolerancias predefinidas.
La proyección usa **PCA**, dado que UMAP no estaba instalado: PC1 explica
19.9025% y PC2 18.1646% de la varianza. No se usan las etiquetas para ajustar
la proyección ni se interpreta su estructura como estados psicológicos.

Se recuperaron pesos de atención reales mediante el backend eager y se validó
la concordancia de logits con la inferencia base. El rollout combina media de
cabezas, conexiones residuales y composición de capas. Los cuatro mapas corresponden
a dos aciertos y dos errores seleccionados automáticamente. Los overlays son
interpretativos, no causales, y se normalizan por imagen para visualización.

## Entorno y reproducibilidad

Python 3.11.18 portátil; PyTorch 2.6.0+cu126; torchvision 0.21.0+cu126;
transformers 4.49.0; NumPy 1.26.4. El freeze completo acompaña la corrida.
GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB. Batch 16, FP32, seed 42,
sin fine-tuning ni modificación de pesos. Modelo y processor fijados al commit
`ef0bc6fc34241b6587e7e009e7711357be28c024`.

Se usó Python portátil y entorno físico en D:, enlazado desde `.venv311`,
para conservar espacio en C: y aislar dependencias. El instalador de Windows
fue rechazado por la política automática; la alternativa portátil funcionó sin
registro global ni cambios de PATH. Las descargas de paquetes usaron caché y
temporales en D:. La descarga del modelo usa `local_dir` para no exigir
privilegios de symlink en Windows.

El piloto de 32 imágenes sólo comprobó infraestructura. La inferencia final
procesó las 3263 imágenes en aproximadamente **33.06 segundos**, sin contar
auditoría, descargas, piloto ni generación de figuras. Este tiempo pertenece
a esta corrida y hardware; no constituye un benchmark comparativo.

El código y el plan originales ejecutados están archivados por hash. La
regeneración posterior de gráficos/índices utilizó los mismos arrays y mantuvo
idénticas las métricas; se archivó su versión de código por separado. No se
repitió la evaluación completa para editar figuras.

## Verificaciones y entregables

Pasaron **25 pruebas** de software. La verificación independiente comprobó:
coincidencia de muestra y orden; exclusión de contempt y contenido compartido;
siete probabilidades válidas por fila; reconstrucción CLS→logits; métricas
recalculables desde CSV; matrices/support; ECE; fuentes y modelo por hash;
copias de error sin alterar bytes; originales intactos; PNG a 300 dpi y GIF
decodificables. El resumen mantiene 3840×2160 y el GIF de inferencia 1280×720,
con cinco ejemplos de 3.5 segundos: tres aciertos y dos errores.

Todos los recursos requeridos se generaron: **13 PNG y 2 GIF**, incluidas
atención, calibración, proyección y la animación opcional de la matriz.
Los intervalos bootstrap de la extensión estadística opcional no se calcularon;
no se presentan como resultados ni como condición pendiente del núcleo solicitado.

El [informe completo](outputs/run_20261111T150020Z_dec91d26/report.md) y el
[índice para presentación](outputs/run_20261111T150020Z_dec91d26/RECURSOS_PRESENTACION.md)
enlazan rutas absolutas. La
[verificación JSON](outputs/run_20261111T150020Z_dec91d26/logs/verification.json)
documenta las comprobaciones. Para copiar a diapositivas, usar
[ppt_results.txt](outputs/run_20261111T150020Z_dec91d26/ppt_results.txt).

## Límites y siguiente punto de control

La distribución es AffectNetFiltered, con preprocesamiento previo e imágenes
en escala de grises. No se han establecido los parámetros exactos de CLAHE/blur,
la versión exacta de descarga, equivalencia de la partición con AffectNet original,
independencia por sujeto ni identidad histórica del checkpoint con el utilizado
en el paper. La depuración cambia la población de evaluación. Los resultados
se refieren a concordancia con etiquetas, sin inferencias sobre atributos sensibles
o experiencias internas de las personas.

El punto de control queda en la revisión de estos resultados por el investigador.
No se instaló ni ejecutó OpenFace/SVC.
