// One point, as a cockpit: the video leads, and the court, the timeline and the
// player cards follow its clock. The top of the page (video, court, timeline)
// is sized to the screen HEIGHT, so the timeline stays usable while it drives
// the video.
import { useEffect, useMemo, useState } from "react";

import { CourtPanel } from "../components/CourtPanel.tsx";
import { PlayerCard } from "../components/PlayerCard.tsx";
import { ShotTimeline } from "../components/ShotTimeline.tsx";
import { NET_Y_M, viewPositions } from "../court.ts";
import { loadPoint, pointFile, type Point } from "../data.ts";
import { PLAYERS, clock, dateEs } from "../play.ts";
import { useVideoTime } from "../useVideoTime.ts";
import styles from "./PointView.module.css";

export function PointView({ id }: { id: string }) {
  const [point, setPoint] = useState<Point | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPoint(null);
    setError(null);
    loadPoint(id)
      .then(setPoint)
      .catch((e: Error) => setError(e.message));
  }, [id]);

  if (error) {
    return (
      <div className={styles.message}>
        <p>{error}.</p>
        <a href="#/">Volver a los puntos</a>
      </div>
    );
  }
  if (!point) return null;
  return <Cockpit point={point} />;
}

function Cockpit({ point }: { point: Point }) {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [focus, setFocus] = useState<number | null>(null);
  const time = useVideoTime(video);
  const frame = Math.floor(time * point.fps);

  const positions = useMemo(() => viewPositions(point), [point]);
  const shots = point.shots;
  const currentIndex = useMemo(() => {
    let index = -1;
    shots.forEach((s, i) => {
      if (s.t <= time + 0.05) index = i;
    });
    return index;
  }, [shots, time]);

  // Pairs by where they actually played, not by number: the side of the net
  // each player spent the point on.
  const side = useMemo(() => {
    const out = new Map<number, "camera" | "far">();
    for (const p of PLAYERS) {
      const samples = positions.get(p) ?? [];
      const meanY = samples.length ? samples.reduce((a, s) => a + s[2], 0) / samples.length : 0;
      out.set(p, meanY > NET_Y_M ? "camera" : "far");
    }
    return out;
  }, [positions]);

  const stats = useMemo(() => {
    const gaps = shots.slice(1).map((s, i) => s.t - shots[i].t);
    const meanGap = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
    return [
      { value: String(shots.length), label: "golpes" },
      { value: clock(point.duration_s), label: "duración" },
      { value: `${meanGap.toFixed(2).replace(".", ",")} s`, label: "entre golpes" },
      { value: String(shots.filter((s) => s.type === "Smash").length), label: "remates" },
    ];
  }, [shots, point.duration_s]);

  const seek = (t: number) => {
    if (video) video.currentTime = Math.max(0, t);
  };

  return (
    <div className={styles.page}>
      <header className={styles.bar}>
        <a className={styles.back} href="#/">
          <svg width="16" height="16" viewBox="0 0 20 20" aria-hidden="true">
            <path d="M12.5 4.5 7 10l5.5 5.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Puntos
        </a>
        <div className={styles.title}>
          <h1>{point.title}</h1>
          <span>{dateEs(point.date)}</span>
        </div>
        <div className={styles.stats}>
          {stats.map((s) => (
            <div key={s.label} className={styles.stat}>
              <b className="mono">{s.value}</b>
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      </header>

      <section className={styles.main}>
        <div className={styles.videoCard}>
          <div className={styles.videoWrap}>
            <video ref={setVideo} src={pointFile(point.id, "video.mp4")} controls playsInline preload="auto" />
          </div>
        </div>
        <CourtPanel
          positions={positions}
          frame={frame}
          focus={focus}
          current={currentIndex >= 0 ? { shot: shots[currentIndex], index: currentIndex, total: shots.length } : null}
        />
      </section>

      <ShotTimeline shots={shots} durationS={point.duration_s} time={time} currentIndex={currentIndex} focus={focus} onSeek={seek} />

      <section className={styles.players}>
        {PLAYERS.map((p) => {
          const mine = shots.filter((s) => s.player === p);
          const pairShots = shots.filter((s) => s.player !== null && side.get(s.player) === side.get(p)).length;
          return (
            <PlayerCard
              key={p}
              player={p}
              shots={mine}
              teamShots={pairShots}
              samples={positions.get(p) ?? []}
              active={focus === p}
              onFocus={setFocus}
            />
          );
        })}
      </section>
    </div>
  );
}
