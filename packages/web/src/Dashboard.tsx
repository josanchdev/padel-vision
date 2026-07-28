import { motion } from "motion/react";
import { useRef } from "react";

import { api, type Match } from "./api.ts";
import { fmtDate, fmtDuration } from "./players.ts";
import styles from "./Dashboard.module.css";

interface Props {
  matches: Match[];
  onOpen: (id: string) => void;
  onUpload: (file: File) => void;
}

export function Dashboard({ matches, onOpen, onUpload }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const done = matches.filter((m) => m.status === "done");
  const pending = matches.filter((m) => m.status !== "done");

  return (
    <div className={styles.wrap}>
      <header className={styles.bar}>
        <div className={styles.brand}>
          <span className={styles.brandDot} />
          Padel Vision
        </div>
        <button className={styles.upload} onClick={() => fileRef.current?.click()}>
          + Subir partido
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
        />
      </header>

      <main className={styles.main}>
        <div className={styles.sectionHead}>
          <h1 className={styles.title}>Partidos</h1>
          <span className={styles.count}>{done.length} analizados</span>
        </div>

        {pending.length > 0 && (
          <div className={styles.processing}>
            {pending.map((m) => (
              <div key={m.id} className={styles.procRow}>
                <span>{m.filename}</span>
                <span className={styles.procStatus}>
                  {m.status === "processing" ? `${Math.round(m.progress * 100)}%` : m.status}
                </span>
              </div>
            ))}
          </div>
        )}

        {done.length === 0 && pending.length === 0 ? (
          <EmptyState onPick={() => fileRef.current?.click()} />
        ) : (
          <div className={styles.grid}>
            {done.map((m, i) => (
              <MatchCard key={m.id} match={m} index={i} onOpen={onOpen} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

function MatchCard({
  match,
  index,
  onOpen,
}: {
  match: Match;
  index: number;
  onOpen: (id: string) => void;
}) {
  return (
    <motion.button
      className={styles.card}
      onClick={() => onOpen(match.id)}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.04, 0.4) }}
      whileHover={{ y: -4 }}
    >
      <div className={styles.thumb}>
        <img src={api.thumbnailUrl(match.id)} alt="" loading="lazy" />
        <span className={styles.duration}>{fmtDuration(match.duration_s)}</span>
      </div>
      <div className={styles.cardBody}>
        <span className={styles.cardName}>{match.filename}</span>
        <span className={styles.cardDate}>{fmtDate(match.created_at)}</span>
      </div>
    </motion.button>
  );
}

function EmptyState({ onPick }: { onPick: () => void }) {
  return (
    <div className={styles.empty}>
      <p>Aún no has analizado ningún partido.</p>
      <button className={styles.emptyBtn} onClick={onPick}>
        Sube tu primer partido
      </button>
    </div>
  );
}
