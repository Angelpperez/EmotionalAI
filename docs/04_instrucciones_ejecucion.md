# Instrucciones de ejecucion paso a paso

Estas instrucciones asumen PowerShell en:

```powershell
C:\Users\angel\source\repos\EmotionRecognition
```

No ejecutes todo de una vez. Avanza bloque por bloque y valida cada salida antes de seguir.

## Bloque 0: Preparacion

### 0.1 Entrar al repo

```powershell
cd C:\Users\angel\source\repos\EmotionRecognition
```

### 0.2 Verificar herramientas disponibles

```powershell
python --version
Get-Command blender
Get-Command colmap
Get-Command ffmpeg
```

Estado esperado actual:

- `python` existe.
- `blender`, `colmap` y `ffmpeg` pueden no estar en PATH.

Estado detectado en esta maquina:

- Python: `3.13.5`.
- OpenCV: `4.12.0`.
- Blender: instalado en `C:\Program Files\Blender Foundation\Blender 4.4\blender.exe`.
- Blender CLI probado: `4.4.3`.
- COLMAP: no detectado en PATH ni por busqueda rapida.
- FFmpeg: no detectado en PATH; `winget search ffmpeg` encuentra `Gyan.FFmpeg`.

Si `blender`, `colmap` y `ffmpeg` fallan pero `python` funciona, no te detengas todavia:

- Puedes ejecutar Bloque 1 y Bloque 2 porque OpenCV extrae los fotogramas desde Python.
- Blender sera necesario en Bloque 3.
- COLMAP sera necesario en Bloque 4.
- FFmpeg es conveniente, pero no obligatorio para la primera extraccion porque el script usa OpenCV.

Verificar OpenCV:

```powershell
@'
import cv2
print(cv2.__version__)
'@ | python -
```

Si imprime una version, puedes continuar con Bloque 1 y Bloque 2.

Si Blender esta instalado pero no aparece, buscarlo:

```powershell
Get-ChildItem "C:\Program Files\Blender Foundation" -Recurse -Filter blender.exe -ErrorAction SilentlyContinue | Select-Object -First 10 FullName
```

Si COLMAP esta instalado pero no aparece:

```powershell
Get-ChildItem "C:\Program Files" -Recurse -Filter colmap.exe -ErrorAction SilentlyContinue | Select-Object -First 10 FullName
```

## Bloque 1: Crear estructura de trabajo

```powershell
New-Item -ItemType Directory -Force data\rocks\level_01\source_video
New-Item -ItemType Directory -Force data\rocks\level_01\images_1280
New-Item -ItemType Directory -Force data\rocks\level_01\colmap
New-Item -ItemType Directory -Force data\rocks\level_01\blender

New-Item -ItemType Directory -Force data\rocks\level_02\source_video
New-Item -ItemType Directory -Force data\rocks\level_02\images_1280
New-Item -ItemType Directory -Force data\rocks\level_02\colmap
New-Item -ItemType Directory -Force data\rocks\level_02\blender

New-Item -ItemType Directory -Force data\rocks\level_03\source_video
New-Item -ItemType Directory -Force data\rocks\level_03\images_1280
New-Item -ItemType Directory -Force data\rocks\level_03\colmap
New-Item -ItemType Directory -Force data\rocks\level_03\blender

New-Item -ItemType Directory -Force outputs\reports
New-Item -ItemType Directory -Force logs
```

Copiar los videos sin tocar originales:

```powershell
Copy-Item captures\1.mp4 data\rocks\level_01\source_video\level_01_lower.mp4
Copy-Item captures\2.mp4 data\rocks\level_02\source_video\level_02_middle.mp4
Copy-Item captures\3.mp4 data\rocks\level_03\source_video\level_03_upper.mp4
```

## Bloque 2: Extraer fotogramas utiles

Este bloque descarta 2 segundos al inicio y 2 segundos al final. Extrae 160 imagenes por nivel, redimensionadas a 1280 px de ancho.

```powershell
@'
import csv
import cv2
from pathlib import Path

repo = Path(".").resolve()
jobs = [
    ("level_01", repo / "data/rocks/level_01/source_video/level_01_lower.mp4"),
    ("level_02", repo / "data/rocks/level_02/source_video/level_02_middle.mp4"),
    ("level_03", repo / "data/rocks/level_03/source_video/level_03_upper.mp4"),
]

trim_start_s = 2.0
trim_end_s = 2.0
target_frames = 160
target_width = 1280

manifest_rows = []

for level, video_path in jobs:
    out_dir = repo / "data/rocks" / level / "images_1280"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(trim_start_s * fps)
    end_frame = frame_count - int(trim_end_s * fps) - 1
    if end_frame <= start_frame:
        raise RuntimeError(f"Recorte invalido para {video_path}")

    if target_frames > (end_frame - start_frame + 1):
        indices = list(range(start_frame, end_frame + 1))
    else:
        step = (end_frame - start_frame) / float(target_frames - 1)
        indices = [round(start_frame + i * step) for i in range(target_frames)]

    for out_i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        scale = target_width / frame.shape[1]
        new_h = round(frame.shape[0] * scale)
        frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

        out_path = out_dir / f"{level}_{out_i:04d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest_rows.append({
            "level": level,
            "source_video": str(video_path),
            "source_frame": frame_idx,
            "time_s": frame_idx / fps,
            "output": str(out_path),
            "source_width": width,
            "source_height": height,
            "fps": fps,
        })

    cap.release()
    print(f"{level}: extraidas {len(indices)} imagenes desde {video_path.name}")

manifest_path = repo / "outputs/reports/frame_manifest.csv"
with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Manifest: {manifest_path}")
'@ | python -
```

Validar conteos:

```powershell
Get-ChildItem data\rocks\level_01\images_1280 -Filter *.jpg | Measure-Object
Get-ChildItem data\rocks\level_02\images_1280 -Filter *.jpg | Measure-Object
Get-ChildItem data\rocks\level_03\images_1280 -Filter *.jpg | Measure-Object
Get-Content outputs\reports\frame_manifest.csv -TotalCount 5
```

Punto de control:

- Deben existir 160 imagenes por nivel.
- Abrir algunas imagenes al azar y verificar que no haya manos, inicio/fin inutil o blur fuerte.
- Si el inicio/fin sigue malo, repetir con `trim_start_s = 4.0` o `trim_end_s = 4.0`.

## Bloque 3: Prueba rapida en Blender con ToF

Primero valida que Blender pueda visualizar la nube ToF.

En esta maquina, los PLY ToF originales de `captures/ToF` vienen de PCL e incluyen `range_grid`; Blender 4.4 rechaza ese header. La ruta estable es crear una copia compatible solo con vertices `x,y,z,r,g,b`, sin tocar el PLY original.

Crear copia compatible:

```powershell
@'
from pathlib import Path

src = Path('captures/ToF/blaze-101-24425138-0000-points_color.ply')
dst = Path('data/rocks/level_01/blender/tof_smoke_test_points_only.ply')
dst.parent.mkdir(parents=True, exist_ok=True)

data = src.read_bytes()
marker_pos = data.index(b'end_header')
newline_pos = data.find(b'\n', marker_pos)
header_end = newline_pos + 1
header_text = data[:header_end].decode('ascii', errors='replace').replace('\r\n', '\n').replace('\r', '\n')

vertex_count = None
for line in header_text.splitlines():
    if line.startswith('element vertex '):
        vertex_count = int(line.split()[-1])
        break
if vertex_count is None:
    raise RuntimeError('No se encontro element vertex')

vertex_stride = 4 + 4 + 4 + 1 + 1 + 1
vertex_bytes = data[header_end:header_end + vertex_count * vertex_stride]

new_header = '\n'.join([
    'ply',
    'format binary_little_endian 1.0',
    'comment converted_for_blender_points_only_from_pcl_ply',
    f'element vertex {vertex_count}',
    'property float x',
    'property float y',
    'property float z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'end_header',
    '',
]).encode('ascii')

dst.write_bytes(new_header + vertex_bytes)
print(dst.resolve())
print('vertices', vertex_count)
'@ | python -
```

1. Abrir Blender.
2. `File -> Import -> Stanford (.ply)`.
3. Seleccionar la copia compatible:

```text
C:\Users\angel\source\repos\EmotionRecognition\data\rocks\level_01\blender\tof_smoke_test_points_only.ply
```

4. Guardar:

```text
C:\Users\angel\source\repos\EmotionRecognition\data\rocks\level_01\blender\tof_smoke_test.blend
```

Punto de control:

- Si el PLY se ve en Blender, ya tenemos ruta estable para inspeccion 3D.
- Si se ve negro o sin color, revisar importacion de vertex colors/material.
- Si se ve muy grande/pequeno, no escalar todavia sin registrar el factor.

Prueba automatica ya ejecutada:

- Entrada convertida: `data/rocks/level_01/blender/tof_smoke_test_points_only.ply`.
- Salida Blender: `data/rocks/level_01/blender/tof_smoke_test.blend`.

## Bloque 4: COLMAP por nivel

Solo continuar cuando `colmap.exe` exista o tengas su ruta completa.

Si `colmap` esta en PATH, para nivel 1:

```powershell
colmap feature_extractor --database_path data\rocks\level_01\colmap\database.db --image_path data\rocks\level_01\images_1280 --ImageReader.single_camera 1
colmap sequential_matcher --database_path data\rocks\level_01\colmap\database.db
New-Item -ItemType Directory -Force data\rocks\level_01\colmap\sparse
colmap mapper --database_path data\rocks\level_01\colmap\database.db --image_path data\rocks\level_01\images_1280 --output_path data\rocks\level_01\colmap\sparse
colmap model_analyzer --path data\rocks\level_01\colmap\sparse\0
```

Si no esta en PATH, reemplazar `colmap` por la ruta completa, por ejemplo:

```powershell
& "C:\ruta\a\COLMAP\colmap.exe" feature_extractor --database_path data\rocks\level_01\colmap\database.db --image_path data\rocks\level_01\images_1280 --ImageReader.single_camera 1
```

Punto de control:

- Debe existir `data\rocks\level_01\colmap\sparse\0`.
- Revisar cuantas imagenes fueron registradas.
- Si registra menos de 80%, no seguir a 3DGS todavia.

Repetir para `level_02` y `level_03` cambiando las rutas.

## Bloque 5: Importar reconstruccion a Blender

COLMAP genera una nube sparse. Para visualizar rapido:

```powershell
colmap model_converter --input_path data\rocks\level_01\colmap\sparse\0 --output_path data\rocks\level_01\colmap\sparse\level_01_sparse.ply --output_type PLY
```

En Blender:

1. `File -> Import -> Stanford (.ply)`.
2. Importar:

```text
data\rocks\level_01\colmap\sparse\level_01_sparse.ply
```

3. Guardar:

```text
data\rocks\level_01\blender\level_01_sparse_colmap.blend
```

Punto de control:

- Si la nube sparse tiene forma coherente circular de roca, se puede pasar a 3DGS.
- Si la nube esta rota o dispersa, hay que mejorar extraccion/matching antes.

## Bloque 6: 3D Gaussian Splatting

Este bloque depende de instalar un entorno 3DGS. Decision recomendada:

- Primario: Nerfstudio/Splatfacto + gsplat.
- Alternativa rapida: Postshot.
- Salida hacia Blender: PLY/asset compatible o add-on 3DGS compatible.

No iniciar entrenamiento completo hasta que COLMAP nivel 1 sea aceptable.

Punto de control minimo para nivel 1:

- Imagenes registradas: ideal >= 80%.
- Error reproyeccion: ideal <= 2 px.
- Roca visible desde trayectoria circular.
- Blender puede abrir el PLY de prueba.

## Bloque 7: Lazo cerrado en Blender

Cuando ya exista escena Blender, crear:

```text
data\rocks\level_01\blender\udp_listener.py
```

Logica esperada:

- Escuchar UDP en `127.0.0.1:5055`.
- Recibir JSON:

```json
{"state": "LOW", "score": 0.15}
```

- Cambiar luz/fog:
  - `LOW`: luz 100%, niebla 0%.
  - `MEDIUM`: luz 70%, niebla 20%.
  - `HIGH`: luz 40%, niebla 60%.

No implementar este bloque hasta tener la escena base en Blender.

## Orden recomendado real

1. Ejecutar Bloque 0.
2. Ejecutar Bloque 1.
3. Ejecutar Bloque 2.
4. Abrir Blender y probar Bloque 3 con un PLY ToF.
5. Resolver Blender/COLMAP en PATH.
6. Ejecutar COLMAP solo para `level_01`.
7. Importar `level_01_sparse.ply` en Blender.
8. Si se ve bien, recien ahi pasar a 3DGS.
