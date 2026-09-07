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
  Hong, Zhang, Gharbi, Fisher, Fatahalian (Stanford + Adobe). ECCV 2022.
  arXiv:2207.10213. Repo: https://github.com/jhong93/spot
  PDF local: `~/reference/papers/2207.10213.pdf` (33 pp., LEÍDO ENTERO sep 2026).
  *Uso:* arquitectura SotA de PES (RegNet-Y 200MF + Gate Shift Modules + Bi-GRU
  1 capa, per-frame, cross-entropy con peso 5x al foreground). Detalles extraídos:
  clips de 100 frames, 224x224, AdamW, mixup, solapamiento 50% en test + NMS.
  Su dataset de tenis EXCLUYE deliberadamente la clasificación fina (derecha/revés/
  slice/volea) — confirma que el TIPO de golpe es hueco abierto, no resuelto.

- **Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and
  Class Imbalance** — Santra, Chudasama, Wasnik (Sony Research India),
  Balasubramanian (IIT Hyderabad). 2025. arXiv:2503.00147.
  PDF local: `~/reference/papers/2503.00147.pdf` (10 pp., LEÍDO sep 2026).
  *Uso:* ataca EXACTAMENTE nuestros dos problemas (dependencia larga + desbalance).
  ASTRM (spatial + local temporal + global temporal) sustituye a GSM; Bi-GRU;
  Soft Instance Contrastive loss + ASAM. +15,67 % sobre E2E-Spot en tenis δ=0.
  **Su ablation mide que la FOCAL LOSS EMPEORA** (72,65 -> 70,36) y SoftIC mejora
  (-> 73,74); y que Bi-GRU 1 capa bate a transformers y a Bi-LSTM. Clip 128.

- **T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event
  Spotting** — Xarles et al. CVPRW 2024. arXiv:2404.05392.
  PDF local: `~/reference/papers/2404.05392.pdf`.
  *Uso:* otra arquitectura PES SotA (SGP layers + Gate Shift Fusion). Mejor que
  E2E-Spot en Figure Skating/Diving, pero arXiv:2503.00147 mide que colapsa en
  SoccerNet (39,43 tight) -> no es universalmente mejor.

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

## Dataset de AUDIO de golpes (nuevo rumbo, feb 2026)

- **Padel Hit Detection Dataset (CVSPORTS_Padel)** — Decorte, Paré, Vanhaeverbeke,
  Taelman, Slembrouck, Verstockt (Ghent/IDLab). Publicado con el paper CVPRW 2024.
  - Dataset: https://cloud.ilabt.imec.be/index.php/s/TFimLDWno6W9ED3
  - Repo/descripción: https://github.com/robbedec/datasets/tree/master/CVsports/Padel
  - Cita: Proceedings of the IEEE/CVF CVPR Workshops 2024, pp. 3306-3314.
  *Contenido:* 99 rallies (5h28min, 11 torneos), vídeo 25fps H.264 CON AUDIO (aac
  48kHz), **2377 golpes anotados** (`hits.csv`: filename,start,end en segundos),
  asignación de jugador (`hit_assignments.xlsx`), scoreboard oculto.
  *Uso:* GT de audio+golpes alineado para entrenar el detector por audio (CRNN SED
  log-Mel). Resuelve el obstáculo de no tener audio+GT. Descargado en
  `data/raw/padel_audio_dataset/`. Cita obligatoria.
