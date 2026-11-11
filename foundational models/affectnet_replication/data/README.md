Coloca aquí AffectNet obtenido por los canales autorizados o usa `--data-root`.
No se descarga ni redistribuye el dataset automáticamente.

El inspector admite carpetas de clases semánticas en particiones train/validation,
un CSV con filepath,label,split, o dos CSV con `--train-csv` y `--validation-csv`.
Los nombres de columnas se detectan sólo si hay una coincidencia inequívoca;
también pueden fijarse mediante argumentos. Las rutas del CSV son relativas a
`--data-root`. Si las etiquetas son números, se exige `--label-map` con un JSON
verificado frente a la documentación de esa copia del dataset. No se incluye
un mapping numérico de ejemplo para evitar confundir IDs entre distribuciones.

Un valor JSON `null` permite excluir explícitamente clases ajenas a las siete
del protocolo; esas exclusiones quedan contadas. Nunca se recodifica contempt
como neutral. Para un formato no reconocido, crea un manifiesto CSV verificado:

```csv
filepath,label,split
images/example_train.jpg,neutral,train
images/example_validation.jpg,happiness,validation
```

Las dos filas son sólo una descripción del formato, no datos experimentales.
