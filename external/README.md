# Dependencias externas

El entrenamiento 3DGS usa el repositorio oficial `gsplat`:

```powershell
git clone --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git external\gsplat_v1_5_3
```

Version usada en los resultados:

```text
tag: v1.5.3
commit: 937e29912570c372bed6747a5c9bf85fed877bae
```

Los clones locales no se versionan. El archivo
`tools/gsplat_compat/fused_ssim.py` implementa el reemplazo usado cuando la
extension CUDA opcional `fused_ssim` no esta disponible.
