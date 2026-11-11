# EmotionRecognition

Repositorio experimental de una investigacion doctoral sobre reconstruccion 3D
de entornos rocosos, inferencia afectiva y adaptacion de una escena Blender en
lazo cerrado.

## Ruta de revision

Para revisar el trabajo en orden:

1. Leer [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
2. Consultar los cuatro puntos de entrada en
   [experiments/README.md](experiments/README.md).
3. Revisar las metricas y evidencias en
   [results/reference/README.md](results/reference/README.md).
4. Revisar la ejecucion integrada en
   [docs/12_resultado_lazo_cerrado_android_mediapipe.md](docs/12_resultado_lazo_cerrado_android_mediapipe.md).
5. Usar [docs/README.md](docs/README.md) para acceder al resto de la memoria
   tecnica.

## Experimentos

| ID | Experimento | Punto de entrada | Estado |
|---|---|---|---|
| E01 | RGB, COLMAP y 3D Gaussian Splatting | `experiments/exp01_rgb_colmap_3dgs.py` | ejecutado |
| E02 | Limpieza y segmentacion ToF Basler | `experiments/exp02_tof_cleaning.py` | ejecutado |
| E03 | Benchmark afectivo VREED | `experiments/exp03_vreed_benchmark.py` | ejecutado |
| E04 | Lazo cerrado Android, MediaPipe, UDP y Blender | `experiments/exp04_closed_loop.py` | ejecutado |

Cada punto de entrada crea una carpeta nueva por corrida y registra sus
argumentos y comandos en `run_manifest.json`.

## Estructura

```text
EmotionRecognition/
|-- experiments/       un punto de entrada por experimento
|-- tools/             implementaciones utilizadas por los experimentos
|-- results/reference/ metricas y evidencias oficiales pequenas
|-- docs/              planificacion, metodos, resultados y limitaciones
|-- requirements/      dependencias por entorno
|-- data/              contrato de datos; los datos privados no se versionan
|-- external/          versiones de dependencias externas
|-- scripts/           verificacion no destructiva del repositorio
|-- State_of_the_art/  antecedentes bibliograficos
`-- EmotionRecognition/ prototipos historicos de Visual Studio
```

La carpeta historica `EmotionRecognition/` no es el flujo principal. Se
conserva para trazabilidad y se explica en
[EmotionRecognition/README.md](EmotionRecognition/README.md).

## Verificacion rapida

La comprobacion no abre Blender, no usa la camara y no entrena modelos:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_repository.ps1
```

Tambien puede inspeccionarse cualquier experimento sin ejecutarlo:

```powershell
python experiments\exp01_rgb_colmap_3dgs.py --dry-run
python experiments\exp02_tof_cleaning.py --dry-run
python experiments\exp03_vreed_benchmark.py --dry-run
python experiments\exp04_closed_loop.py --dry-run
```

Los `dry-run` requieren que las rutas configuradas existan. Todos los scripts
aceptan rutas alternativas por argumentos.

## Entornos

Se usan dos entornos porque MediaPipe 0.10.21 fue validado con Python 3.12,
mientras que el procesamiento ToF, VREED y 3DGS usa el entorno cientifico y
CUDA.

```powershell
py -3.12 -m venv .venv-eye
.\.venv-eye\Scripts\python.exe -m pip install -r requirements\eye.txt

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
```

Las versiones medidas se encuentran en
[requirements/validated_versions.md](requirements/validated_versions.md).
El entrenamiento 3DGS requiere ademas PyTorch con CUDA y `gsplat v1.5.3`.

## Datos y resultados

Los videos RGB, las capturas ToF, VREED, los PLY, los checkpoints y los
archivos Blender no se incluyen en Git. Sus contratos y argumentos estan
documentados en [data/README.md](data/README.md).

Las cifras utilizadas en el informe estan resumidas en
`results/reference/manifest.csv`. La integridad de las evidencias se comprueba
con `results/reference/SHA256SUMS.txt`.

## Alcance cientifico

El estado `stress3` es una etiqueta operacional derivada de cuadrantes
afectivos de VREED. La corrida de lazo cerrado valida integracion, transporte,
latencia y respuesta de Blender. No demuestra todavia validez clinica,
equivalencia MediaPipe-VREED ni generalizacion a sujetos no observados.
