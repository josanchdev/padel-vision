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
}

interface Marker {
  t: number; // seconds
  kind: "shot" | "bounce";
  color: string;
  label: string;
}

export function Timeline({ shots, bounces, fps, onSeek }: Props) {
  const [hover, setHover] = useState<Marker | null>(null);

  const { markers, span } = useMemo(() => {
    const shotMarks: Marker[] = shots.map((s) => ({
      t: s.timestamp_s,
      kind: "shot",
      color: playerColor(s.player_id),
      label: `J${s.player_id} · ${s.label} · ${fmtTime(s.timestamp_s)}`,
    }));
    const bounceMarks: Marker[] = bounces.map((b) => ({
      t: b.frame_index / fps,
      kind: "bounce",
      color: "var(--fg-faint)",
      label: `Bote · ${fmtTime(b.frame_index / fps)}`,
    }));
    const all = [...shotMarks, ...bounceMarks];
    const maxT = all.length ? Math.max(...all.map((m) => m.t)) : 1;
    return { markers: all, span: maxT * 1.02 || 1 };
  }, [shots, bounces, fps]);

  return (
    <div className={styles.wrap}>
      <div className={styles.track}>
        {markers.map((m, i) => (
          <motion.button
            key={i}
            className={`${styles.mark} ${m.kind === "bounce" ? styles.bounce : ""}`}
            style={{ left: `${(m.t / span) * 100}%`, ["--m" as string]: m.color }}
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: Math.min(i * 0.008, 0.25) }}
            whileHover={{ scaleY: 1.4 }}
            onClick={() => onSeek(m.t)}
            onMouseEnter={() => setHover(m)}
            onMouseLeave={() => setHover(null)}
            aria-label={m.label}
          />
        ))}
        <div className={styles.axis}>
          <span>0:00</span>
          <span>{fmtTime(span)}</span>
        </div>
      </div>
      {hover && <div className={styles.tip}>{hover.label}</div>}
    </div>
  );
}
