// The home grid: every exported point, newest tournament last, nothing above it
// but the brand bar — the points are the page.
import { motion } from "motion/react";
import { useEffect, useState } from "react";

import { loadIndex, pointFile, type PointSummary } from "../data.ts";
import { dateEs, duration } from "../play.ts";
import styles from "./Home.module.css";

export function Home() {
  const [points, setPoints] = useState<PointSummary[] | null>(null);

  useEffect(() => {
    loadIndex()
      .then(setPoints)
      .catch(() => setPoints([]));
  }, []);

  return (
    <div className={styles.page}>
      <header className={styles.bar}>
        <div className={styles.brand}>
          <i aria-hidden="true" />
          Padel Vision
        </div>
      </header>
      {points !== null && points.length === 0 && <Empty />}
      {points !== null && points.length > 0 && (
        <main className={styles.grid}>
          {points.map((point, i) => (
            <PointCard key={point.id} point={point} index={i} />
          ))}
        </main>
      )}
    </div>
  );
}

function PointCard({ point, index }: { point: PointSummary; index: number }) {
  // The preview clip loads on first hover only: twelve autoplaying videos on
  // load would cost bandwidth for cards nobody points at.
  const [hovered, setHovered] = useState(false);
  const [previewReady, setPreviewReady] = useState(false);

  return (
    <motion.a
      href={`#/p/${point.id}`}
      className={styles.card}
      aria-label={`${point.title}, ${point.shots} golpes, ${duration(point.duration_s)}`}
      initial={{ opacity: 0.6, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.035, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className={styles.thumb}>
        <img src={pointFile(point.id, "thumb.jpg")} alt="" loading="lazy" />
        {hovered && (
          <video
            src={pointFile(point.id, "preview.mp4")}
            className={previewReady ? styles.ready : undefined}
            muted
            loop
            playsInline
            autoPlay
            onCanPlay={() => setPreviewReady(true)}
          />
        )}
        <span className={styles.duration}>{duration(point.duration_s)}</span>
      </div>
      <div className={styles.meta}>
        <div className={styles.titleRow}>
          <b>{point.title}</b>
          <span>{dateEs(point.date)}</span>
        </div>
        <div className={styles.line}>
          <strong>{point.shots}</strong> golpes
        </div>
      </div>
    </motion.a>
  );
}

function Empty() {
  return (
    <div className={styles.empty}>
      <b>Aún no hay puntos.</b>
      <span>Procesa un vídeo desde la terminal y aparecerá aquí:</span>
      <code>uv run python scripts/export_points.py VIDEO.mp4</code>
    </div>
  );
}
