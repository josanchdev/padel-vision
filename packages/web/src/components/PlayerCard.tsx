// One player's point: how many shots, of which kinds, and where he played.
// Hovering the card lights him up everywhere else (lanes, court) and dims the rest.
import { useEffect, useMemo, useRef } from "react";

import { NET_Y_M, type Sample } from "../court.ts";
import type { Shot } from "../data.ts";
import { paintMiniCourt } from "../heatmap.ts";
import { PLAYER_HEX, TYPES, TYPE_ES } from "../play.ts";
import styles from "./PlayerCard.module.css";
import { ShotIcon } from "./ShotShape.tsx";

type Props = {
  player: number;
  shots: Shot[];
  teamShots: number;
  samples: Sample[];
  active: boolean;
  onFocus: (player: number | null) => void;
};

export function PlayerCard({ player, shots, teamShots, samples, active, onFocus }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const hex = PLAYER_HEX[player];

  const counts = useMemo(
    () => Object.fromEntries(TYPES.map((t) => [t, shots.filter((s) => s.type === t).length])) as Record<string, number>,
    [shots],
  );
  const maxCount = Math.max(1, ...Object.values(counts));
  const top = TYPES.reduce((a, b) => (counts[b] > counts[a] ? b : a));
  const meanY = samples.length ? samples.reduce((sum, s) => sum + s[2], 0) / samples.length : NET_Y_M;
  // View coordinates put the camera at the bottom, so y below the net is its side.
  const side = meanY > NET_Y_M ? "Lado de la cámara" : "Fondo";
  const depth = Math.abs(meanY - NET_Y_M).toFixed(1).replace(".", ",");
  const share = teamShots ? Math.round((shots.length / teamShots) * 100) : 0;

  useEffect(() => {
    if (canvas.current) paintMiniCourt(canvas.current, samples, hex);
  }, [samples, hex]);

  return (
    <article
      className={`${styles.card} ${active ? styles.on : ""}`}
      style={{ "--pc": hex } as React.CSSProperties}
      tabIndex={0}
      onMouseEnter={() => onFocus(player)}
      onMouseLeave={() => onFocus(null)}
      onFocus={() => onFocus(player)}
      onBlur={() => onFocus(null)}
    >
      {/* Two single lines, the same on every card, so the bars below line up. */}
      <div className={styles.who}>
        <div className={styles.badge}>J{player}</div>
        <div className={styles.name}>Jugador {player}</div>
        <div className={styles.total}>
          <b className="mono">{shots.length}</b>
          <span>golpes</span>
        </div>
      </div>
      <div className={styles.side}>
        {side} · {share} % de los golpes de su pareja
      </div>
      <div className={styles.bars}>
        {TYPES.map((type) => (
          <div key={type} className={styles.row}>
            <ShotIcon kind={type} color={hex} size={14} />
            <span>{TYPE_ES[type]}</span>
            <div className={styles.track}>
              <div className={styles.fill} style={{ transform: `scaleX(${counts[type] / maxCount})` }} />
            </div>
            <span className={`${styles.n} mono`}>{counts[type]}</span>
          </div>
        ))}
      </div>
      <div className={styles.mini}>
        <canvas ref={canvas} width={148} height={296} aria-label={`Mapa de calor de J${player}`} />
        <p>
          {shots.length > 0 && (
            <>
              Su golpe más frecuente es <b>{TYPE_ES[top].toLowerCase()}</b>.{" "}
            </>
          )}
          Juega de media a <b>{depth} m</b> de la red.
        </p>
      </div>
    </article>
  );
}
