Cada ejecución crea un directorio nuevo `run_<fecha UTC>_<id>/` para impedir
mezclar predicciones antiguas con corridas incompletas. Dentro se guardan:
`metrics/`, `figures/`, `predictions/`, `inspection/`, `experiment_metadata.json`,
`comparison.csv`, `comparison.md`, `report.md` y, si se ejecuta OpenFace,
`openface_features.csv`. `latest_run.json` señala la corrida más reciente.

Las fases sin datos o dependencias se registran como `blocked` o `failed`, con
motivo y métricas nulas. Nulo no significa cero. No se generan gráficos de
resultados para fases que no se han ejecutado. El diagrama del pipeline sí
puede generarse sin dataset. Los datos, modelos y resultados locales están
excluidos de Git.
