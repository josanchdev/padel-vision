import { motion } from "motion/react";
import { useEffect, useRef, useState } from "react";

import { api, type Match, type MatchData, type Shot } from "./api.ts";
import { ClipModal } from "./ClipModal.tsx";
import { Dashboard } from "./Dashboard.tsx";
import { Heatmap } from "./Heatmap.tsx";
import { Timeline } from "./Timeline.tsx";
import styles from "./App.module.css";

export function App() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [data, setData] = useState<MatchData | null>(null);
  const [shot, setShot] = useState<Shot | null>(null);

  const refresh = () => api.listMatches().then(setMatches).catch(() => {});

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, []);

  const open = async (id: string) => {
    setOpenId(id);
    setData(null);
    try {
      setData(await api.getData(id));
    } catch {
      setData(null); // older matches have no structured data
    }
  };

  const upload = async (file: File) => {
    await api.upload(file);
    refresh();
  };

  if (!openId) {
    return <Dashboard matches={matches} onOpen={open} onUpload={upload} />;
  }

  const openMatch = matches.find((m) => m.id === openId) ?? null;
  return (
    <>
      <MatchDetail
        match={openMatch}
        data={data}
        onBack={() => setOpenId(null)}
        onSelectShot={setShot}
      />
      <ClipModal
        matchId={openId}
        fps={data?.fps ?? 30}
        shot={shot}
        onClose={() => setShot(null)}
      />
    </>
  );
}

function MatchDetail({
  match,
  data,
  onBack,
  onSelectShot,
}: {
  match: Match | null;
  data: MatchData | null;
  onBack: () => void;
  onSelectShot: (s: Shot) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);

  const seek = (seconds: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = seconds;
    void v.play();
  };

  if (!match) return null;
  const hasEvents = !!data && (data.shots.length > 0 || data.bounces.length > 0);
  return (
    <motion.div
      className={styles.detail}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <header className={styles.detailBar}>
        <button className={styles.back} onClick={onBack} aria-label="Volver a partidos">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M15 18l-6-6 6-6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Partidos
        </button>
        <h1 className={styles.detailTitle}>{match.filename}</h1>
      </header>

      {/* Bento grid: everything on one screen, no stacked scroll. Cells reveal
          in a short choreographed sequence on load. */}
      <div className={styles.bento}>
        <motion.section className={`${styles.cell} ${styles.videoCell}`} {...cellReveal(0)}>
          <div className={styles.pane}>
            <video
              ref={videoRef}
              className={styles.video}
              src={api.resultUrl(match.id)}
              controls
            />
          </div>
        </motion.section>

        <motion.section className={`${styles.cell} ${styles.heatCell}`} {...cellReveal(1)}>
          <div className={styles.pane}>
            <div className={styles.cellHead}>
              <span className={styles.cellTitle}>Mapa de calor</span>
            </div>
            {data && data.players.length > 0 ? (
              <Heatmap players={data.players} />
            ) : (
              <div className={styles.heatPlaceholder}>
                <span>Sin datos de posición</span>
              </div>
            )}
          </div>
        </motion.section>

        {hasEvents && (
          <motion.section
            className={`${styles.cell} ${styles.timelineCell}`}
            {...cellReveal(2)}
          >
            <div className={styles.pane}>
              <div className={styles.cellHead}>
                <span className={styles.cellTitle}>Cronología del partido</span>
              </div>
              <Timeline
                shots={data.shots}
                bounces={data.bounces}
                fps={data.fps}
                onSeek={seek}
                onSelectShot={onSelectShot}
              />
            </div>
          </motion.section>
        )}
      </div>
    </motion.div>
  );
}

// Staggered reveal for the bento cells — a short choreographed load sequence.
function cellReveal(index: number) {
  return {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    transition: { delay: 0.06 + index * 0.08, ease: [0.22, 1, 0.36, 1] as const },
  };
}
