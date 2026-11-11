# Entrega del repositorio experimental

## Organizacion vigente

La entrega usa cuatro scripts publicos:

```text
experiments\exp01_rgb_colmap_3dgs.py
experiments\exp02_tof_cleaning.py
experiments\exp03_vreed_benchmark.py
experiments\exp04_closed_loop.py
```

Cada script:

- valida entradas y ejecutables;
- acepta rutas por argumentos;
- implementa `--dry-run`;
- crea una carpeta nueva por corrida;
- guarda un manifiesto de comandos y parametros;
- no sobrescribe resultados anteriores.

La navegacion para revisores comienza en:

```text
README.md
REPRODUCIBILITY.md
docs/README.md
experiments/README.md
results/reference/README.md
tools/README.md
```

## Preservacion

Los algoritmos validados permanecen en `tools/`. No se movieron ni eliminaron
archivos durante esta migracion. Los nuevos puntos de entrada los invocan como
componentes internos.

Los scripts iniciales de CLIP, eye tracking y VREED permanecen en
`EmotionRecognition/` como antecedentes. No forman parte del flujo principal.
La carpeta local `EmotionRecognition/3D_gaussians/` contiene 26,27 GB de
entornos y salidas exploratorias, por lo que se conserva en disco pero se
excluye de la entrega Git.

## Comentarios

Los nuevos scripts contienen solo:

- un docstring breve que identifica el experimento;
- mensajes de error que describen la causa;
- nombres de funciones y argumentos suficientemente explicitos.

No se modificaron comentarios de codigo externo ni de implementaciones
historicas, porque hacerlo no mejora la reproducibilidad y dificulta comparar
versiones.

## Datos y resultados grandes

No deben versionarse:

- videos y capturas ToF;
- dataset VREED completo;
- checkpoints 3DGS;
- PLY, `.blend` y entornos virtuales;
- clones completos de dependencias externas;
- carpetas completas de salida.

`results/reference/manifest.csv` contiene las metricas oficiales y las rutas de
origen. Las copias pequenas necesarias para auditar cifras se incluyen en
`results/reference/`.

`scripts/verify_repository.ps1` valida la estructura, compila los puntos de
entrada y verifica los hashes sin abrir Blender ni utilizar la camara.

## Verificacion previa a la entrega

```powershell
python experiments\exp01_rgb_colmap_3dgs.py --dry-run
python experiments\exp02_tof_cleaning.py --dry-run
python experiments\exp03_vreed_benchmark.py --dry-run
python experiments\exp04_closed_loop.py --dry-run
```

La revision final incluye:

```powershell
git status --short
git diff --check
```

## Verificacion realizada

Fecha: 22 de junio de 2026.

| Comprobacion | Resultado |
|---|---|
| Compilacion de los cuatro puntos de entrada | correcta |
| `--dry-run` de los cuatro experimentos | correcto |
| Extraccion real RGB | 8 fotogramas, manifiesto generado |
| Limpieza real ToF | 12 cuadros, 287547 puntos despues de SOR |
| Segmentacion real ToF | 4 componentes aceptados, 4606 puntos |
| Compatibilidad `fused_ssim` | SSIM 1,0 para entradas identicas |
| `git diff --check` | sin errores |

No se repitieron el entrenamiento completo de 7000 pasos, el benchmark VREED
de 25 folds ni la corrida con camara y Blender. Esas ejecuciones conservan sus
resultados oficiales en `results/reference/`; repetirlas no era necesario para
validar la reorganizacion y habria creado nuevas corridas no comparables.
