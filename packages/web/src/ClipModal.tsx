import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";

import { api, shotOffsetInClip, type Shot } from "./api.ts";
import { fmtTime, playerColor } from "./players.ts";
import styles from "./ClipModal.module.css";

interface Props {
  matchId: string;
  fps: number;
  shot: Shot | null;
  onClose: () => void;
}

export function ClipModal({ matchId, fps, shot, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [progress, setProgress] = useState(0);

  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const offset = shot ? shotOffsetInClip(shot.frame_index, fps) : 0;

  const onLoaded = () => {
    const v = videoRef.current;
    if (!v) return;
    // Start ~1s before the shot so it's seen at once, not after 3s of lead-in.
    v.currentTime = Math.max(offset - 1, 0);
  };

  const onTimeUpdate = () => {
    const v = videoRef.current;
    if (v?.duration) setProgress(v.currentTime / v.duration);
  };

  const scrub = (e: React.MouseEvent<HTMLDivElement>) => {
    const v = videoRef.current;
    if (!v?.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    v.currentTime = ((e.clientX - rect.left) / rect.width) * v.duration;
  };

  const markFraction =
    videoRef.current?.duration ? offset / videoRef.current.duration : offset / 6;

  return (
    <AnimatePresence>
      {shot && (
        <motion.div
          className={styles.backdrop}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className={styles.modal}
            initial={{ scale: 0.94, y: 12, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.96, y: 8, opacity: 0 }}
            transition={{ type: "spring", stiffness: 320, damping: 30 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className={styles.head}>
              <div className={styles.title}>
                <span className={styles.dot} style={{ background: playerColor(shot.player_id) }} />
                <span className={styles.player}>J{shot.player_id}</span>
                <span className={styles.label}>{shot.label}</span>
                <span className={styles.time}>{fmtTime(shot.timestamp_s)}</span>
              </div>
              <button className={styles.close} onClick={onClose} aria-label="Cerrar">
                &times;
              </button>
            </div>

            <video
              ref={videoRef}
              className={styles.video}
              src={api.clipUrl(matchId, shot.frame_index)}
              controls
              autoPlay
              onLoadedMetadata={onLoaded}
              onTimeUpdate={onTimeUpdate}
            />

            <div className={styles.timeline} onClick={scrub}>
              <div className={styles.fill} style={{ width: `${progress * 100}%` }} />
              <div className={styles.mark} style={{ left: `${markFraction * 100}%` }}>
                <span className={styles.markLabel}>golpe</span>
              </div>
            </div>

            <div className={styles.meta}>
              frame {shot.frame_index} · confianza {shot.confidence.toFixed(2)} · el marcador
              señala el golpe dentro del clip
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
