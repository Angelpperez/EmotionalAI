# Decisiones tecnicas y fuentes

## Resumen de decision

La ruta principal debe ser reproducible y medible:

1. RGB videos -> frames controlados.
2. COLMAP para poses y nube sparse.
3. 3D Gaussian Splatting con implementacion eficiente.
4. Export a PLY/SPZ.
5. Blender como entorno principal de visualizacion, inspeccion y demo.
6. Detector ocular + clasificador validado o demo heuristica.

Las herramientas comerciales o cloud pueden acelerar una demo, pero no reemplazan el pipeline reproducible si el objetivo es una defensa doctoral.

Decision actualizada por preferencia del proyecto:

- `1.mp4`, `2.mp4`, `3.mp4` corresponden a nivel inferior, medio y superior de la roca.
- Blender sera la salida principal.
- Unity y HTML quedan como alternativas, no como objetivo primario.

## Evidencia local usada

Repositorio:

- Proyecto Python en Visual Studio: `EmotionRecognition.sln`.
- Scripts oculares existentes: `EyeTracking.py`, `EyeTracking1.py`, `EyeTracking2.py`, `VREED.py`, `pipelinevreed.py`, `vreedpipeline.py`.
- Baseline VREED guardado: `vreed_best_MLP_Deep.pkl`, `vreed_scaler.pkl`, `vreed_feature_names.pkl`.
- Metricas existentes: `vreed_metricas_modelos.csv`.
- Prototipo 3D local: `EmotionRecognition/3D_gaussians/PyshGaussians/3dgs.py`, basado en conversion simple de PCD a gaussianas, no equivalente a entrenamiento 3DGS completo.

## Fuentes tecnicas

### 3D Gaussian Splatting oficial

Fuente:

- Web/paper oficial: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Repo oficial: https://github.com/graphdeco-inria/gaussian-splatting

Puntos relevantes:

- El metodo parte de puntos sparse producidos por calibracion de camaras/SfM.
- Representa la escena con gaussianas 3D y optimiza densidad, covarianza, opacidad y color.
- La publicacion reporta render en tiempo real.
- El repo oficial indica 24 GB VRAM para entrenar a calidad de evaluacion paper.

Decision:

- Usar el metodo como referencia academica.
- No usar configuracion paper completa en la RTX 3060 de 6 GB.
- Si se usa el repo oficial, aplicar downsampling fuerte y entrenamiento corto.

### COLMAP

Fuente:

- CLI oficial: https://colmap.github.io/cli.html

Puntos relevantes:

- COLMAP provee `feature_extractor`, `matcher`, `mapper`, `image_undistorter`, `stereo_fusion` y otros comandos.
- El flujo oficial permite automatizar reconstruccion y obtener poses/camaras.
- Para video circular, el orden temporal permite intentar matching secuencial antes de matching exhaustivo.

Decision:

- Usar COLMAP como etapa de poses para 3DGS.
- Registrar porcentaje de imagenes registradas y error de reproyeccion.

### gsplat / Nerfstudio

Fuentes:

- Documentacion gsplat: https://docs.gsplat.studio/main/
- Paper gsplat: https://arxiv.org/abs/2409.06765
- Nerfstudio Splatfacto: https://docs.nerf.studio/nerfology/methods/splat.html

Puntos relevantes:

- `gsplat` es una libreria CUDA/PyTorch para rasterizacion diferenciable de gaussianas.
- La documentacion reporta hasta 4x menos memoria de entrenamiento frente a la implementacion oficial en capturas Mip-NeRF 360.
- Esta diferencia es importante para una RTX 3060 de 6 GB.

Decision:

- Preferir Nerfstudio/Splatfacto con backend eficiente para entrenamiento local.
- Mantener el repo oficial como referencia teorica y comparativa.

### Postshot

Fuente:

- Jawset/Postshot: https://www.jawset.com/

Puntos relevantes:

- Corre localmente.
- Requiere Windows 10+ y GPU Nvidia CC 7.5+; una RTX 3060 cumple.
- Acepta imagenes y videos, incluidos MP4.
- Permite CLI, region-of-interest, importacion de poses/puntos y export a viewers.

Decision:

- Usarlo como ruta rapida de comparacion/demo si el tiempo del dia es critico.
- No usarlo como unica evidencia si no se exportan metricas y parametros.

### Blender

Fuente:

- Manual Blender 4.5, Stanford PLY: https://docs.blender.org/manual/en/4.5/files/import_export/ply.html

Puntos relevantes:

- Blender importa y exporta archivos Stanford `.ply` desde `File -> Import/Export -> Stanford (.ply)`.
- El importador acepta PLY ASCII o binario.
- El flujo PLY soporta color de vertices, escala y conversion de ejes.
- Esto lo hace adecuado como visor estable para nubes ToF, nubes COLMAP y mallas.
- Ver Gaussian Splats nativos dentro de Blender depende de un add-on o conversion compatible; no debe asumirse como soporte nativo sin prueba local.

Decision:

- Blender sera el entorno principal para ver, alinear, inspeccionar y renderizar la roca.
- La ruta estable es importar PLY/mesh y guardar `.blend`.
- La ruta de alta fidelidad es importar el 3DGS entrenado mediante add-on compatible, si la prueba local funciona.
- El lazo cerrado se implementara con Python en Blender: UDP local, timer no bloqueante y actualizacion de luz/fog/volumetria.

### Luma AI Interactive Scenes

Fuente:

- Luma AI Interactive Scenes: https://lumalabs.ai/interactive-scenes

Puntos relevantes:

- Orientado a escenas compartibles y embebibles en web.
- Reporta 30 FPS en navegadores y tamanos reducidos para objetos/escenas.
- Es comodo para demo web, pero la reconstruccion cloud reduce trazabilidad tecnica.

Decision:

- Util para visualizacion o respaldo si falla el pipeline local.
- No es la ruta principal para resultados doctorales reproducibles.

### Unity Gaussian Splatting

Fuente:

- UnityGaussianSplatting: https://github.com/aras-p/UnityGaussianSplatting

Puntos relevantes:

- Carga PLY de la implementacion original 3DGS y SPZ.
- Funciona en Unity 2022.3, Windows con DX12 o Vulkan.
- Permite integracion C#; por eso es adecuado para listener UDP/TCP.
- El repo advierte que es un viewer/playground, no trainer.

Decision:

- Usarlo solo como respaldo si Blender no puede visualizar el splat con fidelidad suficiente o si se necesita ejecutable interactivo.
- Entrenar splats fuera de Unity y cargar el modelo exportado.

### PlayCanvas Gaussian Splatting

Fuente:

- Docs PlayCanvas: https://developer.playcanvas.com/user-manual/gaussian-splatting/

Puntos relevantes:

- PlayCanvas documenta workflow completo para crear, ver, editar, publicar e integrar splats.
- Es buena salida HTML/Web.
- Un navegador no escucha UDP local directamente, por lo que el lazo cerrado requiere puente WebSocket/HTTP si se elige HTML.

Decision:

- Considerarlo para demo web si Unity no es requisito.
- Para publicacion web, requiere puente WebSocket/HTTP; no es la ruta principal porque el usuario prefiere Blender.

### CLARE

Fuente:

- Paper CLARE: https://arxiv.org/abs/2404.17098

Puntos relevantes:

- Dataset de cognitive load con ECG, EDA, EEG y gaze.
- 24 participantes.
- Etiquetas subjetivas cada 10 segundos.
- Evalua modelos clasicos y deep learning con 10-fold y leave-one-subject-out.

Decision:

- Candidato fuerte para sustentar el clasificador de carga cognitiva.
- La ruta doctoral debe usar etiquetas de carga reales, no solo emociones.

### EyeTheia

Fuentes:

- Paper EyeTheia: https://arxiv.org/abs/2601.06279
- Repo EyeTheia: https://github.com/patherstevenson/EyeTheia

Puntos relevantes:

- Toolbox open-source para mirada 2D con webcam.
- Usa deep learning tipo iTracker.
- Tiene modelos preentrenados en `src/models/`.
- Soporta Windows, Python, PyTorch, inferencia servidor y ONNX browser.
- Requiere solo webcam estandar para inferencia/experimentacion.

Decision:

- Buen candidato para reemplazar o complementar el detector ocular clasico.
- No clasifica carga cognitiva por si mismo; entrega mirada `(x,y)` que alimenta el modelo temporal de 3 estados.

## Decision matrix

| Componente | Opcion primaria | Opcion alternativa | Motivo |
|---|---|---|---|
| Extraccion frames | OpenCV/FFmpeg reproducible | Postshot directo desde video | Control experimental |
| Poses | COLMAP | Postshot/Luma interno | COLMAP entrega metricas y trazabilidad |
| 3DGS training | Nerfstudio/Splatfacto + gsplat | Repo oficial 3DGS | Menor presion de VRAM |
| Demo rapida | Postshot | Luma | Reduce riesgo de quedarse sin escena |
| Visualizacion principal | Blender PLY/mesh/splat add-on | UnityGaussianSplatting | Preferencia del proyecto y control de escena |
| Renderer lazo cerrado | Blender Python UDP listener | UnityGaussianSplatting | Blender permite luz/fog y scripting local |
| Gaze | EyeTheia + calibracion | detector OpenCV actual | Mejor base para mirada `(x,y)` |
| Carga cognitiva | modelo propio validado | heuristica para demo | Necesita etiquetas reales |

## Riesgos tecnicos

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| 6 GB VRAM insuficientes | entrenamiento 3DGS falla | downsample, menos frames, gsplat, Postshot como respaldo |
| COLMAP no registra la roca | no hay poses | mejorar textura/mascaras, sequential/exhaustive, usar ToF para escala |
| ToF sin calibracion con RGB | validacion geometrica debil | usar como auxiliar hasta calibrar extrinsecos |
| Blender sin add-on 3DGS estable | no se ve el splat real en viewport | importar PLY/mesh como respaldo y render externo para metricas |
| VREED no etiqueta carga | conclusion invalida | usar CLARE/NASA-TLX o protocolo propio |
| UDP oscila por ruido ocular | demo inestable | ventana temporal, histeresis, smoothing |
| Blender/COLMAP no estan en PATH | ejecucion local bloqueada | ubicar ejecutables o instalar herramientas |

## Proxima accion recomendada

Antes de instalar o ejecutar entrenamiento:

1. Confirmar mapeo de videos a niveles.
2. Confirmar relacion ToF-RGB.
3. Ubicar o instalar Blender y COLMAP.
4. Probar importacion PLY en Blender con un archivo ToF.
5. Autorizar instalacion de herramientas si no estan presentes.
