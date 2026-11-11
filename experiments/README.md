# Puntos de entrada experimentales

Cada archivo de esta carpeta representa un experimento completo. Las
implementaciones internas permanecen en `tools/`.

| ID | Script | Entrada principal | Evidencia oficial |
|---|---|---|---|
| E01 | `exp01_rgb_colmap_3dgs.py` | video RGB | `results/reference/exp01_*` |
| E02 | `exp02_tof_cleaning.py` | secuencia ToF | `results/reference/exp02_*` |
| E03 | `exp03_vreed_benchmark.py` | dataset VREED | `results/reference/exp03_*` |
| E04 | `exp04_closed_loop.py` | camara, modelo y Blender | `results/reference/exp04_*` |

## E01: RGB, COLMAP y 3DGS

Inspeccion:

```powershell
python experiments\exp01_rgb_colmap_3dgs.py --dry-run
```

Ejecucion completa:

```powershell
D:\emotion3dgs_env\Scripts\python.exe `
  experiments\exp01_rgb_colmap_3dgs.py `
  --video captures\1.mp4 `
  --steps 7000
```

Las etapas pueden seleccionarse mediante `--stages`. El entrenamiento completo
requiere CUDA y `gsplat v1.5.3`.

## E02: limpieza ToF

Prueba de humo:

```powershell
D:\emotion3dgs_env\Scripts\python.exe `
  experiments\exp02_tof_cleaning.py `
  --max-frames 12 `
  --skip-blender
```

Corrida completa:

```powershell
D:\emotion3dgs_env\Scripts\python.exe `
  experiments\exp02_tof_cleaning.py `
  --input-root captures\ToF
```

La prueba de humo valida el flujo, pero no sustituye las metricas oficiales de
la secuencia completa.

## E03: benchmark VREED

```powershell
D:\emotion3dgs_env\Scripts\python.exe `
  experiments\exp03_vreed_benchmark.py `
  --data-root EmotionRecognition\archive `
  --n-splits 5 `
  --n-repeats 5 `
  --random-state 42
```

Ejecuta auditoria estructural, validacion cruzada repetida, entrenamiento de
artefactos e intervalos descriptivos del F1 macro.

## E04: lazo cerrado

```powershell
D:\emotion_eye_env312\Scripts\python.exe `
  experiments\exp04_closed_loop.py `
  --camera-source 1 `
  --duration 60 `
  --port 5056 `
  --model outputs\emotion_benchmark\run_20260616_034800\models\stress3\eye_legacy6_live_compatible__LogReg_balanced.joblib
```

El telefono debe estar conectado y entregando video. Se recomienda fijar
siempre `--model` para que la corrida pueda auditarse.

## Salidas

Cada script escribe en:

```text
outputs/experiments/<experimento>/run_<fecha_hora>/
```

`run_manifest.json` registra argumentos, version de Python y comandos
ejecutados. Los resultados oficiales incluidos en Git estan en
`results/reference/`.
