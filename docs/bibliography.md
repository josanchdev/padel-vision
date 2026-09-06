# Bibliografía y fuentes

Referencias usadas en el proyecto, para citar en la memoria. Cada entrada: título,
autores/venue, enlace, y para qué se usó aquí. Verificar el formato de cita final
(APA/IEEE) con la normativa de la memoria; los datos de aquí son la materia prima.

## Dataset base

- **PadelTracker100** — NTT Data + Universidad de Oviedo. Data in Brief (2026),
  CC BY 4.0. Zenodo record 17020011. PDF en `data/Padeltracker100.pdf`.
  *Uso:* dataset de entrenamiento/benchmark (pose COCO-17, pelota bbox, golpes por
  frame, homografía). Cita obligatoria (licencia CC BY).

## Paper base del TFG

- **Padel Two-Dimensional Tracking Extraction from Monocular Video Recordings** —
  Novillo et al. (2024). *Uso:* paper base que extiende el TFG (tracking 2D por
  homografía); referencia en `~/reference/DS_Padel`.

## Clasificación de golpes / acción basada en esqueleto

- **BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition
  in Racket Sports** — Chang (Jing-Yuan). CVPRW 2026 (12th CVsports).
  arXiv:2502.21085. Repo: https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer
  *Uso:* estado del arte de clasificación de tipo de golpe; pose+pelota con
  cross-attention. Confirmó que pose-sola se satura y la trayectoria de pelota
  desambigua el tipo. Tiene rama TenniSet (tenis 1v1).

- **TemPose: a new skeleton-based transformer model for fine-grained motion
  recognition in badminton** — Ibh, Grasshof, Witzner, Madeleine (2023).
  IEEE, https://ieeexplore.ieee.org/document/10208321
  *Uso:* fusión temprana pose+pelota+posición vía TCN + transformer factorizado.

- **ST-GCN (Spatial-Temporal Graph Convolutional Networks)** — Yan et al. (2018).
  *Uso:* baseline clásico de clasificación por esqueleto (nuestro Nivel 2, 0,53).

## Detección de golpes en PÁDEL (clave para el nuevo rumbo)

- **Multi-Modal Hit Detection and Positional Analysis in Padel Competitions** —
  Decorte et al., Ghent University. CVPRW 2024 (CVsports). IEEE Xplore 10677985.
  https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.html
  biblio: https://biblio.ugent.be/publication/01J0X4X6M285CWXTT89316SBCH
  *Uso:* **F1 92 % detectando golpes de pádel con AUDIO + pose.** Motiva añadir
  audio (el golpe suena). Específico de pádel (nuestro deporte exacto).

## Precise Event Spotting (el marco correcto para detectar el golpe)

- **Action Spotting and Precise Event Detection in Sports: Datasets, Methods, and
  Challenges** (survey) — arXiv:2505.03991.
  https://arxiv.org/html/2505.03991
  *Uso:* define Action Spotting vs Precise Event Spotting (PES) vs TAL; nuestro
  problema es PES. Documenta que la cross-entropy plana falla y el desbalance es el
  reto central; técnicas: context-aware loss, etiqueta temporal, NMS.

- **E2E-Spot: Spotting Temporally Precise, Fine-Grained Events in Video** —
  Hong et al. (2022). arXiv:2207.10213. Repo: https://github.com/jhong93/spot
  *Uso:* arquitectura SotA de PES (RegNet-Y + Gate Shift Modules + Bi-GRU,
  per-frame). Base candidata para rehacer la detección sobre RGB. Incluye dataset
  de tenis con "ball contact".

- **Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and
  Class Imbalance** — (2025). arXiv:2503.00147.
  https://arxiv.org/html/2503.00147
  *Uso:* ataca EXACTAMENTE nuestros dos problemas (dependencia larga + desbalance).
  ASTRM + Bi-GRU; Soft Instance Contrastive loss (bate a focal loss); +15,7 % vs
  E2E-Spot en tenis δ=0. Técnicas de loss reutilizables.

- **T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event
  Spotting** — Xarles et al. CVPR 2024.
  *Uso:* otra arquitectura PES SotA (SGP layers + Gate Shift Fusion).

## Modelos de vídeo (enfoque end-to-end, referencia)

- **VideoMAE / VideoMAE v2** — masked autoencoder para vídeo (ViT).
  *Uso:* referencia de backbone de vídeo moderno; usado por SpotFormer y otros
  como extractor de features espacio-temporales.

## Pelota (Nivel 3, ya implementado)

- **TrackNet (V2/V3)** — detección de pelota pequeña por heatmap (3 frames→heatmap).
  *Uso:* nuestro detector de pelota (F1 ~0,94). Base de la trayectoria.

## Detección/pose base

- **YOLO26-pose** (Ultralytics) — detección de personas + pose COCO-17.
  *Uso:* pose de jugadores en todo el pipeline (ADR-0003).
- **ByteTrack** — multi-object tracking. *Uso:* identidad de jugadores.

---

**Nota de método (para la memoria):** toda técnica o arquitectura de terceros
usada o consultada está aquí con su fuente. Las decisiones de por qué se adoptó o
descartó cada una están en `docs/experiments.md` (bitácora) y `docs/decisions/`
(ADRs).
