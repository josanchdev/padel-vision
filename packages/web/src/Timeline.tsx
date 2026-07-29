import { motion } from "motion/react";
import { useMemo, useState } from "react";

import type { Bounce, Shot } from "./api.ts";
import { fmtTime, playerColor } from "./players.ts";
import styles from "./Timeline.module.css";

interface Props {
  shots: Shot[];
  bounces: Bounce[];
  fps: number;
  onSeek: (seconds: number) => void;
  onSelectShot: (shot: Shot) => void;
}

interface Marker {
  t: number; // seconds
  kind: "shot" | "bounce";
  color: string;
  label: string;
  shot?: Shot;
}

export function Timeline({ shots, bounces, fps, onSeek, onSelectShot }: Props) {
  const [hover, setHover] = useState<Marker | null>(null);

  const { markers, start, end } = useMemo(() => {
    const shotMarks: Marker[] = shots.map((s) => ({
      t: s.timestamp_s,
      kind: "shot",
      color: playerColor(s.player_id),
      label: `J${s.player_id} · ${s.label} · ${fmtTime(s.timestamp_s)}`,
      shot: s,
    }));
    const bounceMarks: Marker[] = bounces.map((b) => ({
      t: b.frame_index / fps,
      kind: "bounce",
      color: "var(--fg-faint)",
      label: `Bote · ${fmtTime(b.frame_index / fps)}`,
    }));
    const all = [...shotMarks, ...bounceMarks];
    const times = all.map((m) => m.t);
    // Position markers across the ACTUAL event range, not [0, max] — a clip may
    // start well into the match (e.g. 17-37s), so anchoring to 0 bunches them.
    const minT = times.length ? Math.min(...times) : 0;
    const maxT = times.length ? Math.max(...times) : 1;
    const pad = Math.max((maxT - minT) * 0.04, 0.5);
    return { markers: all, start: minT - pad, end: maxT + pad };
  }, [shots, bounces, fps]);

  const span = end - start || 1;
  const frac = (t: number) => (t - start) / span;

  // Shots open their clip; bounces (no clip) just seek the video.
  const activate = (m: Marker) => (m.shot ? onSelectShot(m.shot) : onSeek(m.t));

  return (
    <div className={styles.wrap}>
      <div className={styles.track}>
        {markers.map((m, i) => (
          <motion.button
            key={i}
            className={`${styles.mark} ${m.kind === "bounce" ? styles.bounce : ""}`}
            style={{ left: `${frac(m.t) * 100}%`, ["--m" as string]: m.color }}
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: Math.min(i * 0.008, 0.25) }}
            whileHover={{ scaleY: 1.4 }}
            onClick={() => activate(m)}
            onMouseEnter={() => setHover(m)}
            onMouseLeave={() => setHover(null)}
            aria-label={m.label}
          />
        ))}
        <div className={styles.axis}>
          <span>{fmtTime(Math.max(start, 0))}</span>
          <span>{fmtTime(end)}</span>
        </div>
      </div>
      <div className={styles.tip}>{hover ? hover.label : "Pasa el ratón por un evento · clic para verlo"}</div>
    </div>
  );
}
