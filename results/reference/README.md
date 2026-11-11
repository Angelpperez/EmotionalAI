# Evidencias de referencia

Estas copias representan los resultados citados en la documentacion. No son
salidas de las pruebas pequenas de reorganizacion.

## E01: RGB, COLMAP y 3DGS

| Archivo | Evidencia |
|---|---|
| `exp01_train_step6999.json` | numero de gaussianas y memoria |
| `exp01_holdout_metrics.json` | PSNR, SSIM y LPIPS |
| `exp01_holdout_contact_sheet.jpg` | comparacion visual de vistas retenidas |

## E02: ToF

| Archivo | Evidencia |
|---|---|
| `exp02_cleaning_metrics.json` | filtros y puntos conservados |
| `exp02_segmentation_metrics.json` | plano, componentes y rocas |

## E03: VREED

| Archivo | Evidencia |
|---|---|
| `exp03_f1_confidence_intervals.csv` | medias e IC 95% del F1 macro |

## E04: lazo cerrado

| Archivo | Evidencia |
|---|---|
| `exp04_eye_detection_quality.json` | calidad de deteccion ocular |
| `exp04_eye_detection.jpg` | fotograma MediaPipe |
| `exp04_end_to_end_summary.json` | estabilidad, eventos y latencias |
| `exp04_difficulty_easy.png` | estado facil |
| `exp04_difficulty_normal.png` | estado normal |
| `exp04_difficulty_hard.png` | estado dificil |

`manifest.csv` resume las metricas principales. `SHA256SUMS.txt` permite
verificar que los archivos no fueron modificados.
