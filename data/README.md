# Datos experimentales

Los datos originales no se incluyen en Git por su tamano y por las condiciones
de distribucion de cada fuente. Los scripts reciben rutas alternativas mediante
argumentos.

| Experimento | Entrada esperada | Argumento |
|---|---|---|
| RGB, COLMAP y 3DGS | video circular del nivel inferior | `--video` |
| Limpieza ToF | secuencia Basler blaze-101 | `--input-root` |
| Benchmark afectivo | dataset VREED extraido | `--data-root` |
| Lazo cerrado | camara Android y escena Blender | `--camera-source`, `--scene` |

Ubicaciones locales usadas en los resultados originales:

```text
captures\1.mp4
captures\ToF\
EmotionRecognition\archive\
D:\emotion3dgs_results\closed_loop_e2e\
```

`results/reference/` contiene evidencias pequenas y metricas de referencia,
pero no reemplaza los datos necesarios para repetir los experimentos.
