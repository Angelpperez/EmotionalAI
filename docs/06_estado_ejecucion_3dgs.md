# Estado de ejecucion 3DGS

Fecha local de ejecucion: 2026-06-15.

## Resultado ya validado

La reconstruccion sparse COLMAP quedo completa para los tres niveles:

| Nivel | Imagenes registradas | Puntos 3D | Error reproyeccion medio |
|---|---:|---:|---:|
| level_01 lower | 160/160 | 14,413 | 0.981951 px |
| level_02 middle | 160/160 | 15,945 | 0.977803 px |
| level_03 upper | 160/160 | 12,541 | 0.874171 px |

Dataset preparado para 3DGS:

```text
data/gaussian_splatting/level_01/
  images/
  sparse/0/
  sparse_txt/
  dataset_manifest.txt
```

## Toolchain CUDA resuelta

El entrenamiento 3D Gaussian Splatting ya no esta bloqueado por toolchain CUDA.

Estado verificado:

1. CUDA Toolkit 12.6 quedo disponible en `D:\CUDA\v12.6`.
2. `nvcc.exe` responde correctamente:

```text
D:\CUDA\v12.6\bin\nvcc.exe
```

3. Visual Studio C++ x64 esta disponible en:

```text
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
```

4. `gsplat==1.5.3` fue compilado desde `external\gsplat_v1_5_3` con `TORCH_CUDA_ARCH_LIST=8.6`.
5. El backend CUDA carga correctamente:

```text
D:\emotion3dgs_env\Lib\site-packages\gsplat\csrc.pyd
```

## Resultados 3DGS level_01

Entrenamiento smoke, 100 pasos:

```text
D:\emotion3dgs_results\level_01_smoke_cuda\ply\point_cloud_99.ply
D:\emotion3dgs_results\level_01_smoke_cuda\ckpts\ckpt_99_rank0.pt
D:\emotion3dgs_results\level_01_smoke_cuda\level_01_3dgs_smoke_100_steps.blend
```

Entrenamiento intermedio, 3000 pasos:

```text
D:\emotion3dgs_results\level_01_train_3000\ply\point_cloud_2999.ply
D:\emotion3dgs_results\level_01_train_3000\ckpts\ckpt_2999_rank0.pt
D:\emotion3dgs_results\level_01_train_3000\level_01_3dgs_3000_steps.blend
```

Metricas internas 3000 pasos:

```json
{"mem": 0.40694189071655273, "ellipse_time": 114.44639348983765, "num_GS": 168164}
```

Entrenamiento hito, 7000 pasos:

```text
D:\emotion3dgs_results\level_01_train_7000\ply\point_cloud_6999.ply
D:\emotion3dgs_results\level_01_train_7000\ckpts\ckpt_6999_rank0.pt
D:\emotion3dgs_results\level_01_train_7000\level_01_3dgs_7000_steps.blend
```

Metricas internas 7000 pasos:

```json
{"mem": 0.6107964515686035, "ellipse_time": 284.57313680648804, "num_GS": 373024}
```

## Auditoria de conservacion del suelo

La corrida de 373.024 gaussianas no utilizo mascaras de objeto, recorte
espacial, RANSAC ni eliminacion semantica del suelo. Se entreno con las 160
imagenes RGB completas y el modelo disperso de COLMAP:

```text
data\gaussian_splatting\level_01\dataset_manifest.txt
source_run=data\rocks\level_01\colmap\run_20260615_210259
source_model=data\rocks\level_01\colmap\run_20260615_210259\sparse\1
images_source=data\rocks\level_01\images_1280
```

Las vistas 3DGS retenidas muestran que el plano del suelo se conserva en el
renderizado nativo:

```text
D:\emotion3dgs_results\level_01_eval_holdout_7000\renders\
D:\emotion3dgs_results\level_01_eval_holdout_7000\report\
  holdout_real_vs_3dgs_contact_sheet.jpg
```

La configuracion solo contiene el pruning normal del algoritmo:

```text
prune_opa: 0.005
prune_scale2d: 0.15
prune_scale3d: 0.1
```

Estos criterios eliminan gaussianas inestables o de opacidad muy baja; no
identifican ni recortan el suelo como clase.

La conversion de inspeccion para Blender:

```text
point_cloud_6999_rgb_points.ply
```

descarto 19.110 gaussianas con `alpha < 0,01`, equivalentes al 5,123% del total.
Sin embargo, los limites espaciales se mantuvieron exactamente:

```text
X: [-8,254566; 3,336676]
Y: [-3,694469; 5,217452]
Z: [-1,085866; 1,233232]
```

Por tanto, esa conversion no produjo un recorte geometrico del suelo. La
apariencia fragmentada se debe a que Blender representa cada gaussiana como un
punto de radio fijo y no reproduce su escala anisotropica, rotacion, opacidad,
solapamiento ni harmonicos esfericos. Superficies continuas en el rasterizador
3DGS se observan como puntos separados al importarlas de esta manera.

La variante:

```text
point_cloud_6999_rgb_alpha075_points.ply
```

conserva solo 147.174 gaussianas, el 39,454% del total. Esta variante de alta
opacidad si reduce fuertemente la densidad visual y puede hacer que el suelo
parezca mas recortado. Es una prueba diagnostica y no debe presentarse como la
representacion 3DGS completa.

Finalmente, la nube ToF segmentada es un experimento separado. En:

```text
D:\emotion3dgs_results\tof_blaze101_clean\
```

se elimino intencionalmente el plano dominante mediante RANSAC para aislar las
rocas. Esa eliminacion de suelo no fue aplicada al entrenamiento RGB 3DGS de
373.024 gaussianas. Ademas, la corrida 3DGS se inicializo desde COLMAP/RGB, no
desde la Basler blaze-101.

## Inspeccion RGB en Blender

El PLY nativo de 3DGS contiene centros gaussianos y atributos gaussianos
especializados (`f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`). Blender
puede importarlo, pero no lo rasteriza como 3D Gaussian Splatting; por eso la
primera inspeccion aparece como una nube de puntos negros.

Para inspeccion visual dentro de Blender se genero una version RGB estandar,
derivada del componente DC de harmonicos esfericos:

```text
rgb = clamp(0.28209479177387814 * f_dc + 0.5, 0, 1)
```

Artefactos generados:

```text
tools\convert_3dgs_ply_to_rgb_points.py
tools\create_blender_rgb_point_scene.py
D:\emotion3dgs_results\level_01_train_7000\ply\point_cloud_6999_rgb_points.ply
D:\emotion3dgs_results\level_01_train_7000\level_01_3dgs_7000_rgb_points.blend
```

Metricas de conversion:

```text
input_points: 373024
output_points: 353914
min_alpha: 0.01
alpha_mean: 0.46619948744773865
rgb_mean_r: 0.36951175332069397
rgb_mean_g: 0.37285977602005005
rgb_mean_b: 0.36384689807891846
```

Nota: esta escena RGB es una herramienta de inspeccion doctoral en Blender, no
un reemplazo del rasterizador 3DGS. Para visualizacion fotorealista de splats
con radios, opacidades, rotaciones y harmonicos esfericos completos, conviene
usar un visor 3DGS compatible.

## Reconstruccion densa para Blender

La inspeccion en Blender evidencio que los centros 3DGS no son una malla de
roca. Para una salida navegable en Blender se ejecuto el flujo fotogrametrico
denso de COLMAP sobre `level_01`:

```text
image_undistorter -> patch_match_stereo -> stereo_fusion -> poisson_mesher
```

Configuracion conservadora para RTX 3060 6 GB:

```text
PatchMatchStereo.max_image_size = 960
PatchMatchStereo.geom_consistency = 1
PatchMatchStereo.cache_size = 8
PoissonMeshing.depth = 11
PoissonMeshing.trim = 10
```

Artefactos densos:

```text
D:\emotion3dgs_results\level_01_dense_colmap\fused_geometric.ply
D:\emotion3dgs_results\level_01_dense_colmap\meshed_poisson_depth11.ply
D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11.blend
D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_FULL_SCENE_DEMO.blend
```

Metricas iniciales:

```text
depth_maps: 320
normal_maps: 320
fused_geometric points: 1,168,467
meshed_poisson_depth11 vertices: 467,737
meshed_poisson_depth11 faces: 915,009
```

La primera malla reconstruyo una lamina grande de piso/fondo. Esto no es fallo
de Blender: las imagenes RGB incluyen piso, pared, tripodes/sombras y dos
rocas, por lo que COLMAP aprende la escena completa cuando no se usan mascaras
o segmentacion de objeto.

Decision de uso actual:

```text
Demo Blender principal:
D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_FULL_SCENE_DEMO.blend

Motivo:
mantiene continuidad visual y contexto de la escena; aunque incluye suelo/fondo,
es la salida mas legible para demostrar navegacion 3D hoy.
```

Variante para inspeccion de color:

```text
D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_VERTEX_COLOR_EMISSION.blend
outputs\reports\level_01_colmap_dense_mesh_depth11_VERTEX_COLOR_EMISSION_preview.png
```

Esta variante usa el atributo de color `Col` de COLMAP conectado a un shader
`Emission`. Preserva los colores por vertice, pero los muestra sin depender de
luces ni niebla. Debe usarse para inspeccion visual/colorimetrica, no para
evaluar el trigger de luminosidad, porque la emision no responde igual que un
material iluminado fisicamente.

Se intento una mascara automatica GrabCut, pero la cobertura media fue
demasiado alta y visualmente incluia gran parte del piso:

```text
mask coverage mean: 0.7365
mask coverage min: 0.4274
mask coverage max: 0.8124
control sheet: outputs\reports\level_01_masks_grabcut_contact_sheet.jpg
```

Por rigor, esa mascara no se uso como evidencia final. En su lugar se aplico
limpieza geometrica 3D sobre la nube densa:

```text
tools\filter_point_cloud_ransac_planes.py
tools\filter_point_cloud_voxel_components.py
```

Limpieza aplicada:

```text
RANSAC planes removed: 673,205 points
remaining after planes: 495,262 points
main voxel component kept: 436,292 points
voxel_size: 0.25
```

Artefactos filtrados:

```text
D:\emotion3dgs_results\level_01_dense_colmap\fused_geometric_no_2planes_main_component_v025.ply
D:\emotion3dgs_results\level_01_dense_colmap\meshed_poisson_depth11_no_2planes_main_component_v025.ply
D:\emotion3dgs_results\level_01_dense_colmap\level_01_colmap_dense_mesh_depth11_clean_main_component.blend
outputs\reports\level_01_colmap_dense_mesh_depth11_clean_main_component_preview.png
```

Metricas de malla filtrada:

```text
vertices: 789,713
faces: 1,507,969
```

Interpretacion: esta malla filtrada no debe usarse como demo visual principal,
porque al quitar planos/componentes pierde continuidad y queda fragmentada. Su
valor es diagnostico: demuestra que el aislamiento automatico de roca todavia
no es robusto con RGB sin mascaras verificadas. Para mejorarla doctoralmente,
la siguiente etapa debe integrar mascaras verificadas o registrar las nubes ToF
(`captures\ToF`) para usar profundidad real como restriccion geometrica.

## Observacion de espacio

`C:` estaba casi lleno antes de instalar CUDA. Despues de mover artefactos principales a `D:`, el espacio libre observado fue aproximadamente:

```text
C: ~6.746 GB libres
D: ~850.035 GB libres
```

Mediciones previas de candidatos de limpieza:

```text
C:\Users\angel\AppData\Local\Temp                  ~5.378 GB
C:\Users\angel\.cache                              ~1.482 GB
C:\Users\angel\source\repos\EmotionRecognition\env ~5.919 GB
C:\Users\angel\AppData\Local\Docker                ~56.718 GB
```

No se eliminaron estas carpetas. Antes de limpiar Docker, entornos o cache grande, conviene confirmar explicitamente.
