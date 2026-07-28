import { motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api, type Match, type MatchData, type Shot } from "./api.ts";
import { ClipModal } from "./ClipModal.tsx";
import { Dashboard } from "./Dashboard.tsx";
import { Heatmap } from "./Heatmap.tsx";
import { ShotTable } from "./ShotTable.tsx";
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

  const kpis = useMemo(() => {
    if (!data) return null;
    const types = new Set(data.shots.map((s) => s.label));
    return { shots: data.shots.length, bounces: data.bounces.length, types: types.size };
  }, [data]);

  const seek = (seconds: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = seconds;
    void v.play();
    v.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  if (!match) return null;
  return (
    <motion.div
      className={styles.detail}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className={styles.detailBar}>
        <button className={styles.back} onClick={onBack}>
          ← Partidos
        </button>
        <h1 className={styles.detailTitle}>{match.filename}</h1>
        {kpis && (
          <div className={styles.kpiStrip}>
            <KpiInline label="Golpes" value={kpis.shots} />
            <KpiInline label="Botes" value={kpis.bounces} />
            <KpiInline label="Tipos" value={kpis.types} />
          </div>
        )}
      </div>

      {/* Bento grid: everything on one screen, no stacked scroll. */}
      <div className={styles.bento}>
        <section className={`${styles.cell} ${styles.videoCell}`}>
          <div className={styles.cellHead}>
            <span className={styles.cellTitle}>Vídeo analizado</span>
          </div>
          <video
            ref={videoRef}
            className={styles.video}
            src={api.resultUrl(match.id)}
            controls
          />
        </section>

        <section className={`${styles.cell} ${styles.heatCell}`}>
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
        </section>

        {data && (data.shots.length > 0 || data.bounces.length > 0) && (
          <section className={`${styles.cell} ${styles.timelineCell}`}>
            <div className={styles.cellHead}>
              <span className={styles.cellTitle}>Cronología</span>
            </div>
            <Timeline
              shots={data.shots}
              bounces={data.bounces}
              fps={data.fps}
              onSeek={seek}
            />
          </section>
        )}

        <section className={`${styles.cell} ${styles.tableCell}`}>
          <div className={styles.cellHead}>
            <span className={styles.cellTitle}>Golpes</span>
          </div>
          {data && data.shots.length > 0 ? (
            <ShotTable shots={data.shots} onSelect={onSelectShot} />
          ) : (
            <p className={styles.empty}>Sin datos estructurados para este partido.</p>
          )}
        </section>
      </div>
    </motion.div>
  );
}

function KpiInline({ label, value }: { label: string; value: number }) {
  return (
    <div className={styles.kpiInline}>
      <span className={styles.kpiInlineValue}>{value}</span>
      <span className={styles.kpiInlineLabel}>{label}</span>
    </div>
  );
}
