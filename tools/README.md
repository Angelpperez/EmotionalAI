# Implementaciones

Los archivos de esta carpeta son componentes internos. Los revisores deben
ejecutar los puntos de entrada de `experiments/`, que fijan el orden de las
etapas y registran un manifiesto.

## Reconstruccion RGB y 3DGS

| Archivo | Funcion |
|---|---|
| `convert_3dgs_ply_to_rgb_points.py` | convierte un PLY 3DGS para inspeccion |
| `create_blender_mesh_scene.py` | crea una escena Blender desde una malla |
| `create_blender_rgb_point_scene.py` | crea una escena de puntos RGB |
| `filter_point_cloud_ransac_planes.py` | elimina planos dominantes |
| `filter_point_cloud_voxel_components.py` | filtra componentes espaciales |
| `generate_rock_masks_grabcut.py` | genera mascaras provisionales |
| `summarize_3dgs_holdout_metrics.py` | resume PSNR, SSIM y LPIPS |
| `gsplat_compat/fused_ssim.py` | compatibilidad SSIM sin extension CUDA |

## Procesamiento ToF

| Archivo | Funcion |
|---|---|
| `clean_tof_blaze_sequence.py` | limpieza temporal y espacial Basler |
| `segment_tof_rocks_ransac_components.py` | plano RANSAC y componentes |
| `create_blender_tof_clean_scene.py` | escena de inspeccion ToF |

## Inferencia afectiva

| Archivo | Funcion |
|---|---|
| `audit_vreed_subject_ids.py` | audita estructura e identificadores VREED |
| `benchmark_vreed_affective_states.py` | benchmark quad4 y stress3 |
| `calculate_f1_confidence_intervals.py` | intervalos del F1 macro |
| `eye_tracking_backends.py` | backends Haar y MediaPipe |
| `evaluate_eye_detection_source.py` | calidad de deteccion ocular |
| `live_vreed_eye_state_udp.py` | inferencia ocular y emision UDP |

## Lazo cerrado

| Archivo | Funcion |
|---|---|
| `live_closed_loop_end_to_end.py` | ejecucion camara-modelo-Blender-ACK |
| `blender_udp_difficulty_controller_experiment.py` | controlador formal |
| `prepare_closed_loop_difficulty_scene.py` | prepara la escena adaptativa |
| `prepare_closed_loop_difficulty_scene_v2.py` | variante alineada a camara |
| `render_closed_loop_difficulty_states.py` | renderiza los tres estados |
| `start_closed_loop_experiment.ps1` | lanzador historico de la integracion |

## Diagnostico y pruebas

Los demas controladores, emisores UDP, sondas de latencia y utilidades de
camara se conservan para diagnostico. No representan experimentos
independientes ni deben citarse como resultados principales.
