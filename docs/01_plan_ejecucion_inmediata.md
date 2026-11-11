# Plan de ejecucion inmediata

## Objetivo

Construir una prueba de concepto doctoral de lazo cerrado: una roca reconstruida en 3D, navegable en tiempo real, cuya luminosidad o niebla reacciona a un estado psicofisiologico estimado desde senales oculares.

El criterio rector es no afirmar mas de lo que los datos permiten. Para hoy se separan dos niveles:

- Demo funcional: muestra el flujo completo RGB/ToF -> reconstruccion 3D -> inferencia ocular -> comando UDP -> cambio visual.
- Protocolo validable: define metricas, baselines y etiquetas necesarias para defender resultados en tesis.

## Inventario local

Datos disponibles en el repositorio:

- Solucion Visual Studio Python: `EmotionRecognition.sln`.
- Videos RGB en `captures/`:
  - `1.mp4`: nivel inferior de la roca, 1920x1080, 868 frames, ~29.99 fps, ~28.95 s.
  - `2.mp4`: nivel medio de la roca, 1920x1080, 1045 frames, ~30.00 fps, ~34.83 s.
  - `3.mp4`: nivel superior de la roca, 1920x1080, 1137 frames, ~30.00 fps, ~37.91 s.
- Datos ToF en `captures/ToF/`:
  - 270 `points_color.ply`.
  - 270 `points_intensity.ply`.
  - 270 `range_gray.tiff`.
  - 270 `range_color.png`.
  - 270 `confidence.tiff`.
  - 270 `intensity.tiff`.
  - 6 archivos `device_settings.pfs`.
- Dataset ocular local VREED en `EmotionRecognition/archive/`.
  - 312 muestras.
  - 4 clases balanceadas en `Quad_Cat`: 78 muestras por clase.
  - 49 columnas: 1 etiqueta + 48 features de fijaciones, sacadas, parpadeos y microsacadas.
- Baseline local actual:
  - `MLP_Deep`: `Accuracy=0.365`, `F1_macro=0.367`.
  - Esto sirve como linea base, no como resultado doctoral final.

Dependencias operativas detectadas:

- `ffmpeg` y `ffprobe` no estan disponibles en PATH.
- `colmap` no esta disponible en PATH.
- `blender` no esta disponible en PATH.
- OpenCV si puede leer los videos desde Python.

## Restricciones de hardware

Equipo declarado:

- GPU: RTX 3060, 6 GB VRAM.
- RAM: 32 GB.
- Disco: 1 TB.

Implicacion: el pipeline oficial de 3D Gaussian Splatting no debe ejecutarse en configuracion de calidad paper completa. La implementacion oficial declara 24 GB VRAM para entrenar a calidad de evaluacion paper; por eso el plan local debe usar muestreo de frames, reduccion de resolucion, iteraciones controladas y/o implementaciones mas eficientes como `gsplat`/Nerfstudio.

## Decision de arquitectura para hoy

Arquitectura propuesta:

```text
captures/*.mp4
    -> recorte temporal de inicio/fin no utiles
    -> extraccion reproducible de fotogramas por nivel
    -> COLMAP / SfM
    -> 3D Gaussian Splatting local
    -> export PLY / mesh / splat-compatible asset
    -> escena Blender para inspeccion, render y demo

webcam
    -> detector ocular clasico o EyeTheia
    -> ventana temporal de features
    -> clasificador de 3 estados
    -> UDP localhost
    -> listener Python en Blender
    -> luz/fog segun estado
```

Decision primaria:

- Reconstruccion: `COLMAP + Nerfstudio/Splatfacto con backend gsplat` como ruta reproducible.
- Ruta rapida comparativa: Postshot puede usarse para generar una escena rapidamente, pero no debe ser la unica evidencia doctoral si no se registran parametros y metricas.
- Entorno de visualizacion principal: Blender.
- Blender estable: importar PLY/mesh/ToF y preparar una escena `.blend` reproducible con luces, niebla y camaras.
- Blender con Gaussian Splatting real: viable si instalamos un add-on compatible con PLY/SPZ/3DGS; se debe validar antes de depender de el para resultados.
- Unity/HTML quedan como respaldos si se necesita un ejecutable o publicacion web.
- Inferencia ocular: mantener el detector clasico como adquisicion minima, pero separar la estimacion de mirada de la clasificacion de carga. EyeTheia es candidato para mirada `(x,y)` con modelo preentrenado; el clasificador de carga debe entrenarse/validarse con etiquetas adecuadas.

## Estructura de carpetas propuesta

No mover datos todavia sin confirmacion. Para ejecucion reproducible:

```text
data/
  rocks/
    level_01/
      source_video/
      images_raw/
      images_1280/
      masks/
      colmap/
      gaussian_splatting/
    level_02/
    level_03/
  tof/
    blaze_sequence/
outputs/
  level_01/
    reports/
    renders/
    splats/
    blender/
      level_01_scene.blend
      udp_listener.py
logs/
  eye_inference/
  udp_bridge/
  renderer_events/
```

## Cronograma para hoy

### Bloque 0: Cierre de supuestos y preparacion reproducible (30 min)

1. Registrar que `1.mp4`, `2.mp4`, `3.mp4` son nivel inferior, medio y superior respectivamente.
2. Confirmar si la secuencia ToF corresponde a una sola roca, a los tres niveles juntos o a una captura independiente.
3. Definir Blender como salida principal del dia.
4. Ubicar o instalar Blender y COLMAP, porque no estan disponibles en PATH.
5. Instalar o habilitar herramientas minimas solo si se aprueba: FFmpeg, COLMAP, entorno Nerfstudio/gsplat, add-on Blender 3DGS o Postshot.

Salida esperada:

- Carpeta de trabajo organizada.
- Tabla de inventario con videos, ToF y versiones de herramientas.

### Bloque 1: Procesamiento de roca y reconstruccion 3D (3 h)

Paso 1.1: Fotogramas

- Recortar los primeros y ultimos segundos si muestran movimiento de inicio/cierre, desenfoque, manos, vibracion o partes fuera de la trayectoria circular util.
- Regla inicial: descartar 2 s al inicio y 2 s al final, y ajustar con deteccion de blur/cobertura.
- Extraer frames por nivel con muestreo uniforme despues del recorte.
- Para RTX 3060 6 GB, comenzar con 120 a 180 imagenes por nivel a ancho maximo 1280 px.
- Mantener un conjunto de validacion: 10% a 15% de frames no usados en entrenamiento.
- Registrar: numero de frames extraidos, resolucion, metodo de muestreo, hash/lista de imagenes.

Paso 1.2: SfM / COLMAP

- Ejecutar SfM por nivel.
- Priorizar `sequential_matcher` si los frames salen de video circular ordenado; usar `exhaustive_matcher` solo si el numero de imagenes es pequeno o falla la secuencia.
- Verificar:
  - porcentaje de imagenes registradas.
  - error medio de reproyeccion.
  - numero de puntos 3D.
  - continuidad circular de camaras.

Paso 1.3: 3D Gaussian Splatting

- Entrenar primero `level_01`.
- Usar configuracion ligera:
  - imagenes a 1280 px de ancho.
  - iteraciones iniciales 3k a 7k para demo.
  - subir a 15k/30k solo si VRAM y tiempo lo permiten.
- Exportar `point_cloud.ply`, malla o asset compatible con Blender.
- Evaluar PSNR/SSIM/LPIPS sobre holdout, no solo inspeccion visual.

Salida esperada:

- Escena Blender de `level_01`, con camaras, luces, niebla y artefactos importados.
- Si el add-on 3DGS funciona: splat navegable dentro de Blender.
- Si el add-on 3DGS no funciona: PLY/mesh ToF-Colmap como respaldo estable en Blender, mas renders externos del splat para metricas.
- Reporte de metricas y capturas de validacion.
- Registro de VRAM, tiempo de entrenamiento y FPS/fluidez de viewport o render.

### Bloque 2: Inferencia ocular y clasificacion de 3 estados (2.5 h)

Paso 2.1: Adquisicion

- Mantener el detector clasico OpenCV como adquisicion minima.
- Mejorar el flujo para emitir ventanas temporales, no predicciones frame a frame.
- Variables minimas por ventana:
  - coordenadas de pupila o mirada `(x,y)`.
  - velocidad/angularidad de mirada.
  - fijaciones estimadas.
  - sacadas estimadas.
  - parpadeos por minuto.
  - perdida de deteccion.

Paso 2.2: Modelo

- No equiparar directamente `Quad_Cat` de VREED con estres/carga cognitiva.
- Usar VREED como baseline de features o pre-entrenamiento exploratorio.
- Para defender 3 estados `Bajo Estres / Carga Media / Alta Sobrecarga`, se requiere una de estas rutas:
  - Dataset con etiquetas de carga cognitiva como CLARE/NASA-TLX.
  - Protocolo propio con etiquetas subjetivas, por ejemplo NASA-TLX o escala corta por bloque.
  - Demo heuristica explicitamente etiquetada como proxy, no como clasificador validado.

Paso 2.3: Salida de estados

Para demo funcional:

```json
{"state": "LOW", "score": 0.18, "timestamp": "..."}
{"state": "MEDIUM", "score": 0.55, "timestamp": "..."}
{"state": "HIGH", "score": 0.84, "timestamp": "..."}
```

Para tesis:

- Reportar `F1_macro`, `balanced_accuracy`, matriz de confusion, calibracion de probabilidades y validacion por sujeto si hay datos multi-sujeto.

### Bloque 3: Puente de lazo cerrado (2.5 h)

Paso 3.1: Comunicacion

- Usar UDP localhost para demo por simplicidad.
- Puerto sugerido: `127.0.0.1:5055`.
- Formato: JSON por datagrama.
- Guardar log local de cada mensaje enviado y recibido.

Paso 3.2: Listener en Blender

Reglas iniciales:

- `LOW`: luminosidad 100%, niebla 0%.
- `MEDIUM`: luminosidad 70%, niebla 20%.
- `HIGH`: luminosidad 40%, niebla 60%.

Controles necesarios:

- Histeresis minima de 2 a 3 ventanas para evitar parpadeo.
- Filtro temporal exponencial sobre `score`.
- Boton o tecla de override manual para demo.
- Script Python en Blender con timer no bloqueante para escuchar UDP y actualizar `World`, luces y densidad volumetrica.

Salida esperada:

- Video corto de evidencia del lazo cerrado.
- Log con timestamps: camara -> inferencia -> UDP -> Blender.
- Metricas de latencia y estabilidad.

## Dudas que deben cerrarse antes de ejecutar

1. Confirmar si `captures/ToF` tiene calibracion extrinseca/intrinseca respecto de los videos RGB.
2. Confirmar si aceptas instalar herramientas externas: FFmpeg, COLMAP, Nerfstudio/gsplat, Blender y add-on Blender 3DGS si hace falta.
3. Confirmar si la tesis necesita inferir estres/carga cognitiva con participantes humanos reales o solo demostrar el mecanismo adaptativo.
