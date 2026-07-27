import { motion } from "motion/react";
import { useMemo, useState } from "react";

import type { Shot } from "./api.ts";
import { fmtTime, playerColor } from "./players.ts";
import styles from "./ShotTable.module.css";

interface Props {
  shots: Shot[];
  onSelect: (shot: Shot) => void;
}

export function ShotTable({ shots, onSelect }: Props) {
  const players = useMemo(
    () => [...new Set(shots.map((s) => s.player_id))].sort((a, b) => a - b),
    [shots],
  );
  const types = useMemo(
    () => [...new Set(shots.map((s) => s.label))].sort(),
    [shots],
  );

  const [activePlayers, setActivePlayers] = useState<Set<number>>(new Set(players));
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set(types));

  const toggle = <T,>(set: Set<T>, value: T): Set<T> => {
    const next = new Set(set);
    next.has(value) ? next.delete(value) : next.add(value);
    return next;
  };

  const visible = shots.filter(
    (s) => activePlayers.has(s.player_id) && activeTypes.has(s.label),
  );

  return (
    <section className={styles.wrap}>
      <div className={styles.filters}>
        <span className={styles.filterLabel}>Jugador</span>
        {players.map((p) => (
          <button
            key={p}
            className={`${styles.chip} ${activePlayers.has(p) ? "" : styles.off}`}
            onClick={() => setActivePlayers((s) => toggle(s, p))}
            style={{ ["--chip" as string]: playerColor(p) }}
          >
            <span className={styles.chipDot} style={{ background: playerColor(p) }} />J{p}
          </button>
        ))}
        <span className={styles.filterLabel}>Tipo</span>
        {types.map((t) => (
          <button
            key={t}
            className={`${styles.chip} ${activeTypes.has(t) ? "" : styles.off}`}
            onClick={() => setActiveTypes((s) => toggle(s, t))}
          >
            {t}
          </button>
        ))}
      </div>

      <div className={styles.count}>
        <strong>{visible.length}</strong> golpes
      </div>

      <table className={styles.table}>
        <thead>
          <tr>
            <th>Minuto</th>
            <th>Jugador</th>
            <th>Tipo</th>
            <th className={styles.right}>Confianza</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((s, i) => (
            <motion.tr
              key={`${s.frame_index}-${s.player_id}`}
              className={styles.row}
              onClick={() => onSelect(s)}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.015, 0.3) }}
              whileHover={{ backgroundColor: "var(--surface-2)" }}
            >
              <td className={s.confidence < 0.5 ? styles.low : ""}>{fmtTime(s.timestamp_s)}</td>
              <td>
                <span className={styles.jdot} style={{ background: playerColor(s.player_id) }} />
                J{s.player_id}
              </td>
              <td className={s.confidence < 0.5 ? styles.low : ""}>{s.label}</td>
              <td className={styles.right}>
                <span className={styles.confBar}>
                  <span className={styles.confFill} style={{ width: `${s.confidence * 100}%` }} />
                </span>
                <span className={s.confidence < 0.5 ? styles.low : ""}>
                  {s.confidence.toFixed(2)}
                </span>
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
