# Reproducibilidad

## Plataforma de referencia

| Recurso | Configuracion |
|---|---|
| GPU | NVIDIA RTX 3060, 6 GB VRAM |
| RAM | 32 GB |
| Sistema | Windows 11 |
| Blender | 4.4.3 |
| COLMAP | compilacion Windows con CUDA |
| Camara ToF | Basler blaze-101 |
| Camara ocular | Android mediante camara conectada a Windows |

Las versiones Python y de paquetes se registran en
`requirements/validated_versions.md`.

## Entradas

| Experimento | Entrada | No incluida en Git |
|---|---|---|
| E01 | video RGB circular del nivel inferior | si |
| E02 | secuencia ToF Basler con profundidad, intensidad y confianza | si |
| E03 | dataset VREED extraido | si |
| E04 | camara Android, modelo E03 y escena Blender | si |

Las rutas se pasan por argumentos. Las ubicaciones locales originales aparecen
en `data/README.md`, pero no son necesarias si se proporcionan rutas
equivalentes.

## Comandos

### E01: RGB, COLMAP y 3DGS

```powershell
D:\emotion3dgs_env\Scripts\python.exe `
  experiments\exp01_rgb_colmap_3dgs.py `
  --video captures\1.mp4 `
  --steps 7000
```

La ejecucion completa extrae fotogramas, estima camaras con COLMAP, genera una
reconstruccion densa, prepara el conjunto 3DGS, entrena, evalua vistas retenidas
y crea una escena de inspeccion para Blender.

### E02: limpieza ToF

```powershell
D:\emotion3dgs_env\Scripts\python.exe `
  experiments\exp02_tof_cleaning.py `
  --input-root captures\ToF
```

La secuencia se filtra por confianza, estabilidad temporal, discontinuidades
de profundidad y Statistical Outlier Removal. Despues se separa el plano
dominante mediante RANSAC y se extraen componentes conectados.

### E03: benchmark VREED

```powershell
D:\emotion3dgs_env\Scripts\python.exe `
  experiments\exp03_vreed_benchmark.py `
  --data-root EmotionRecognition\archive `
  --n-splits 5 `
  --n-repeats 5 `
  --random-state 42
```

El experimento audita los identificadores disponibles, ejecuta validacion
cruzada repetida y calcula el intervalo de confianza descriptivo del F1 macro.

### E04: lazo cerrado

```powershell
D:\emotion_eye_env312\Scripts\python.exe `
  experiments\exp04_closed_loop.py `
  --camera-source 1 `
  --duration 60 `
  --port 5056 `
  --model outputs\emotion_benchmark\run_20260616_034800\models\stress3\eye_legacy6_live_compatible__LogReg_balanced.joblib
```

El telefono debe estar conectado y entregando video. Blender recibe estados por
UDP, aplica el nivel de dificultad y responde con un ACK.

## Salidas

Cada corrida crea:

```text
outputs/experiments/<experimento>/run_<fecha_hora>/
`-- run_manifest.json
```

Las salidas especificas incluyen metricas JSON/CSV, imagenes, nubes PLY,
modelos y registros temporales. Los artefactos pesados permanecen fuera de Git.

## Resultados de referencia

| Experimento | Resultado principal |
|---|---|
| E01 | 373024 gaussianas; 0,610796 GB; PSNR 29,777246; SSIM 0,913685 |
| E02 | 288043 puntos en la nube canonica completa despues de SOR |
| E03 | F1 macro multimodal stress3 0,734171; IC 95% [0,712492, 0,755851] |
| E04 | 60,090 s; 105/105 ACK; 0% perdida; P95 cuadro-a-Blender 98,544 ms |

Los valores completos y sus fuentes se encuentran en
`results/reference/manifest.csv`.

## Controles

- Los resultados oficiales no se sobrescriben.
- Cada corrida usa una carpeta con sello temporal.
- E03 fija `random_state=42`.
- E04 puede recibir una ruta de modelo fija mediante `--model`.
- Las evidencias incluidas tienen hashes SHA-256.
- Las capturas originales no se modifican.

## Limites

- E01 no dispone todavia de escala fisica validada para todas las vistas RGB.
- E02 representa unicamente superficies visibles desde la camara ToF fija.
- E03 no dispone todavia de una evaluacion LOSO completa por sujeto.
- E04 demuestra integracion, no validez psicofisiologica del estado estimado.
