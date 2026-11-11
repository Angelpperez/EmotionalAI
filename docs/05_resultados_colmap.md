# Resultados COLMAP

Fecha de ejecucion: 2026-06-15.

Binario usado:

```text
D:\colmap-x64-windows-cuda\COLMAP.bat
COLMAP 4.1.0.dev0, commit 5b76f53, CUDA
```

Configuracion comun:

- Imagenes por nivel: 160.
- Resolucion de imagenes: 1280x720.
- Camara unica por nivel: `--ImageReader.single_camera 1`.
- Extraccion SIFT GPU: `--FeatureExtraction.use_gpu 1`.
- Matching secuencial: `--SequentialMatching.overlap 20`.

## Resumen

| Nivel | Run | Modelo | Imagenes registradas | Puntos | Observaciones | Track medio | Obs./imagen | Error reproy. medio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| level_01 | `data\rocks\level_01\colmap\run_20260615_210259` | 1 | 160/160 | 14413 | 81705 | 5.668841 | 510.656250 | 0.981951 px |
| level_02 | `data\rocks\level_02\colmap\run_20260615_210617` | 0 | 160/160 | 15945 | 104580 | 6.558796 | 653.625000 | 0.977803 px |
| level_03 | `data\rocks\level_03\colmap\run_20260615_210714` | 0 | 160/160 | 12541 | 155784 | 12.421976 | 973.650000 | 0.874171 px |

Lectura tecnica:

- Los tres niveles superan el criterio inicial de registro: 160/160 imagenes.
- Los tres errores medios de reproyeccion estan por debajo de 1 px.
- `level_03` tiene menor error y mayor track medio; `level_01` tiene menor densidad relativa, pero es aceptable para avanzar.

## Artefactos principales

Resumen CSV:

```text
outputs\reports\colmap_summary.csv
```

PLY exportados:

```text
data\rocks\level_01\colmap\run_20260615_210259\level_01_sparse_colmap.ply
data\rocks\level_02\colmap\run_20260615_210617\level_02_sparse_colmap.ply
data\rocks\level_03\colmap\run_20260615_210714\level_03_sparse_colmap.ply
```

Escenas Blender generadas:

```text
data\rocks\level_01\blender\level_01_sparse_colmap.blend
data\rocks\level_02\blender\level_02_sparse_colmap.blend
data\rocks\level_03\blender\level_03_sparse_colmap.blend
```

Prueba ToF en Blender:

```text
data\rocks\level_01\blender\tof_smoke_test_points_only.ply
data\rocks\level_01\blender\tof_smoke_test.blend
```

Nota:

- Blender creo `level_01_sparse_colmap.blend1` como backup automatico. No se elimino.
- Hubo un run parcial de `level_02` en `run_20260615_210551` por manejo de stderr en PowerShell; no se elimino.

## Proximo paso recomendado

Preparar el dataset para 3D Gaussian Splatting usando los modelos COLMAP ya validados:

1. Mantener los `run_...` actuales como referencia.
2. Usar el modelo principal de cada nivel como input SfM.
3. Ejecutar primero entrenamiento 3DGS solo con `level_01`.
4. Exportar resultado a formato compatible con Blender, o mantener Blender para inspeccion PLY y usar un viewer 3DGS externo para alta fidelidad.

