// The court seen from above, following the video: players moving with a short
// trail, or where each one spent the point (heatmap). Below it, the shot being
// played right now.
import { useMemo, useState } from "react";

import {
  COURT_LENGTH_M,
  COURT_WIDTH_M,
  NET_Y_M,
  SERVICE_LINES_Y,
  positionAt,
  type Sample,
} from "../court.ts";
import type { Shot } from "../data.ts";
import { densityCanvas } from "../heatmap.ts";
import { PLAYERS, PLAYER_HEX, TYPE_ES, clock } from "../play.ts";
import styles from "./CourtPanel.module.css";
import { ShotIcon } from "./ShotShape.tsx";

type Props = {
  positions: Map<number, Sample[]>;
  frame: number;
  focus: number | null;
  current: { shot: Shot; index: number; total: number } | null;
};

const TRAIL_FRAMES = 30;

export function CourtPanel({ positions, frame, focus, current }: Props) {
  const [mode, setMode] = useState<"live" | "heat">("live");
  const [heatFor, setHeatFor] = useState<number | null>(null); // null = everyone

  const byFrame = useMemo(() => {
    const out = new Map<number, Map<number, Sample>>();
    for (const [player, samples] of positions) out.set(player, new Map(samples.map((s) => [s[0], s])));
    return out;
  }, [positions]);

  const heatUrl = useMemo(() => {
    if (mode !== "heat") return "";
    const players = heatFor === null ? [...PLAYERS] : [heatFor];
    return densityCanvas(
      players.map((p) => ({ samples: positions.get(p) ?? [], hex: PLAYER_HEX[p] })),
    ).toDataURL();
  }, [mode, heatFor, positions]);

  return (
    <aside className={styles.panel} aria-label="Pista">
      <div className={styles.head}>
        <h2>{mode === "live" ? "Pista en vivo" : "Mapa de calor"}</h2>
        <div className={styles.seg} role="group" aria-label="Vista de la pista">
          <button aria-pressed={mode === "live"} onClick={() => setMode("live")}>
            En vivo
          </button>
          <button aria-pressed={mode === "heat"} onClick={() => setMode("heat")}>
            Mapa de calor
          </button>
        </div>
      </div>

      <div className={styles.courtBox}>
        <svg viewBox="-1.2 -1.4 12.4 22.8" role="img" aria-label="Pista vista desde arriba">
          <CourtLines />
          <text x={COURT_WIDTH_M / 2} y={-0.55} className={styles.edge}>
            Fondo
          </text>
          <text x={COURT_WIDTH_M / 2} y={COURT_LENGTH_M + 1.05} className={styles.edge}>
            Cámara
          </text>
          {mode === "heat" ? (
            <g>
              <image href={heatUrl} x={0} y={0} width={COURT_WIDTH_M} height={COURT_LENGTH_M} preserveAspectRatio="none" />
              <CourtLines overlay />
            </g>
          ) : (
            <g>
              {PLAYERS.map((p) => {
                const map = byFrame.get(p);
                if (!map) return null;
                const trail: string[] = [];
                for (let f = Math.max(0, frame - TRAIL_FRAMES); f <= frame; f += 2) {
                  const s = positionAt(map, f);
                  if (s) trail.push(`${s[1]},${s[2]}`);
                }
                const here = positionAt(map, frame);
                const dim = focus !== null && focus !== p;
                return (
                  <g key={p} className={styles.player} style={{ opacity: dim ? 0.2 : 1 }}>
                    <polyline points={trail.join(" ")} fill="none" stroke={PLAYER_HEX[p]} strokeWidth={0.09} strokeOpacity={0.45} strokeLinecap="round" strokeLinejoin="round" />
                    {here && (
                      <g transform={`translate(${here[1]} ${here[2]})`}>
                        <circle r={0.46} fill={PLAYER_HEX[p]} stroke="#fff" strokeWidth={0.1} />
                        <text dy={0.18} className={styles.dotLabel}>
                          {p}
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}
            </g>
          )}
        </svg>
      </div>

      {mode === "heat" ? (
        <div className={styles.chips}>
          <button className={styles.chip} aria-pressed={heatFor === null} onClick={() => setHeatFor(null)}>
            Todos
          </button>
          {PLAYERS.map((p) => (
            <button key={p} className={styles.chip} aria-pressed={heatFor === p} onClick={() => setHeatFor(p)}>
              <i style={{ background: PLAYER_HEX[p] }} />J{p}
            </button>
          ))}
        </div>
      ) : (
        <NowCard current={current} />
      )}
    </aside>
  );
}

function CourtLines({ overlay = false }: { overlay?: boolean }) {
  const stroke = overlay ? "#ffffff" : "#c9d0dd";
  return (
    <g>
      {!overlay && <rect x={0} y={0} width={COURT_WIDTH_M} height={COURT_LENGTH_M} rx={0.15} fill="#e9eef7" stroke={stroke} strokeWidth={0.08} />}
      {SERVICE_LINES_Y.map((y) => (
        <line key={y} x1={0} x2={COURT_WIDTH_M} y1={y} y2={y} stroke={stroke} strokeWidth={0.07} />
      ))}
      <line x1={COURT_WIDTH_M / 2} x2={COURT_WIDTH_M / 2} y1={SERVICE_LINES_Y[0]} y2={SERVICE_LINES_Y[1]} stroke={stroke} strokeWidth={0.07} />
      <line x1={-0.4} x2={COURT_WIDTH_M + 0.4} y1={NET_Y_M} y2={NET_Y_M} stroke="#7d879b" strokeWidth={0.14} strokeLinecap="round" />
    </g>
  );
}

function NowCard({ current }: { current: Props["current"] }) {
  if (!current) {
    return (
      <div className={styles.now}>
        <span className={styles.nowMeta}>El punto aún no ha empezado. Dale al play o haz clic en un golpe.</span>
      </div>
    );
  }
  const { shot, index, total } = current;
  const colour = shot.player ? PLAYER_HEX[shot.player] : "#8a93a4";
  return (
    <div className={styles.now}>
      <ShotIcon kind={shot.type} color={colour} size={30} />
      <div>
        <div className={styles.nowLabel}>
          {shot.player ? `J${shot.player}` : "Sin jugador"} · {shot.type ? TYPE_ES[shot.type] : "Sin clasificar"}
        </div>
        <div className={styles.nowMeta}>
          Golpe {index + 1} de {total} · <span className="mono">{clock(shot.t)}</span>
        </div>
      </div>
      <div className={styles.conf}>
        <b className="mono">{Math.round(shot.confidence * 100)} %</b>
        <span>confianza</span>
      </div>
    </div>
  );
}
