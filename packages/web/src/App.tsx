import { motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";

import { api, type Match, type MatchData, type Shot } from "./api.ts";
import { ClipModal } from "./ClipModal.tsx";
import { ShotTable } from "./ShotTable.tsx";
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

  const openMatch = matches.find((m) => m.id === openId) ?? null;

  const kpis = useMemo(() => {
    if (!data) return null;
    const byType = new Map<string, number>();
    for (const s of data.shots) byType.set(s.label, (byType.get(s.label) ?? 0) + 1);
    return { shots: data.shots.length, bounces: data.bounces.length, byType };
  }, [data]);

  return (
    <div className={styles.app}>
      <header className={styles.hero}>
        <motion.p
          className={styles.kicker}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
        >
          PADEL VISION
        </motion.p>
        <motion.h1
          className={styles.title}
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
        >
          Analiza cada golpe.
        </motion.h1>
        <motion.p
          className={styles.sub}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.15 }}
        >
          Sube un partido y explora sus datos: golpes, botes y el clip de cada jugada.
        </motion.p>
      </header>

      <main className={styles.main}>
        <UploadCard onUploaded={refresh} />

        {!openId && <MatchList matches={matches} onOpen={open} />}

        {openMatch && (
          <motion.section
            className={styles.detail}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <button className={styles.back} onClick={() => setOpenId(null)}>
              ← Partidos
            </button>
            <h2 className={styles.detailTitle}>{openMatch.filename}</h2>

            {kpis && (
              <div className={styles.kpis}>
                <Kpi label="Golpes" value={kpis.shots} />
                <Kpi label="Botes" value={kpis.bounces} />
                <Kpi label="Tipos" value={kpis.byType.size} />
              </div>
            )}

            <video className={styles.result} src={api.resultUrl(openMatch.id)} controls />

            {data && data.shots.length > 0 ? (
              <ShotTable shots={data.shots} onSelect={setShot} />
            ) : (
              <p className={styles.empty}>
                Este partido no tiene datos estructurados (procesado antes de la capa de datos).
              </p>
            )}
          </motion.section>
        )}
      </main>

      <ClipModal
        matchId={openId ?? ""}
        fps={data?.fps ?? 30}
        shot={shot}
        onClose={() => setShot(null)}
      />
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: number }) {
  return (
    <motion.div
      className={styles.kpi}
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <span className={styles.kpiValue}>{value}</span>
      <span className={styles.kpiLabel}>{label}</span>
    </motion.div>
  );
}

function UploadCard({ onUploaded }: { onUploaded: () => void }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);

  const send = async (file: File) => {
    setBusy(true);
    try {
      await api.upload(file);
      onUploaded();
    } finally {
      setBusy(false);
    }
  };

  return (
    <label
      className={`${styles.drop} ${over ? styles.over : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]);
      }}
    >
      <input
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => e.target.files?.[0] && send(e.target.files[0])}
      />
      {busy ? "Subiendo…" : "Arrastra un vídeo aquí o haz clic para elegir"}
    </label>
  );
}

function MatchList({ matches, onOpen }: { matches: Match[]; onOpen: (id: string) => void }) {
  if (matches.length === 0) return null;
  return (
    <section className={styles.list}>
      <h2 className={styles.listTitle}>Partidos</h2>
      {matches.map((m) => (
        <motion.button
          key={m.id}
          className={styles.matchRow}
          onClick={() => m.status === "done" && onOpen(m.id)}
          whileHover={{ x: 4 }}
          disabled={m.status !== "done"}
        >
          <span className={styles.matchName}>{m.filename}</span>
          <span className={`${styles.status} ${styles[m.status]}`}>
            {m.status === "processing" ? `${Math.round(m.progress * 100)}%` : m.status}
          </span>
          <span className={styles.matchShots}>
            {m.status === "done" ? `${m.shots_detected} golpes` : "—"}
          </span>
        </motion.button>
      ))}
    </section>
  );
}
