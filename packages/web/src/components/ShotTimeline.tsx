// One lane per player; every shot sits on its hitter's lane at its instant, as
// his colour and its stroke's shape. A dashed line joins consecutive shots, so
// the ball's back-and-forth reads at a glance. Clicking seeks the video.
import { useEffect, useMemo, useRef, useState } from "react";

import type { Shot } from "../data.ts";
import { PLAYERS, PLAYER_HEX, TYPES, TYPE_ES, clock } from "../play.ts";
import styles from "./ShotTimeline.module.css";
import { ShotIcon, ShotShape } from "./ShotShape.tsx";

type Props = {
  shots: Shot[];
  durationS: number;
  time: number;
  currentIndex: number;
  focus: number | null;
  onSeek: (t: number) => void;
};

const LANE = 32;
const TOP = 8;
const LEFT = 64;
const RIGHT = 16;
const AXIS = 24;
const MIN_WIDTH = 640;
/** How long after a shot its ring stays lit. */
const ACTIVE_S = 1.5;

export function ShotTimeline({ shots, durationS, time, currentIndex, focus, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1000);
  const [tip, setTip] = useState<{ shot: Shot; x: number; y: number } | null>(null);

  useEffect(() => {
    const element = wrap.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(entry.contentRect.width, MIN_WIDTH)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const height = TOP + LANE * PLAYERS.length + AXIS;
  const x = (t: number) => LEFT + (t / durationS) * (width - LEFT - RIGHT);
  const laneY = (player: number) => TOP + (player - 1) * LANE + LANE / 2;
  const assigned = useMemo(() => shots.filter((s) => s.player !== null), [shots]);
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let s = 0; s <= durationS; s += 5) out.push(s);
    return out;
  }, [durationS]);

  const seekAt = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    onSeek(Math.min(durationS, Math.max(0, ((px - LEFT) / (width - LEFT - RIGHT)) * durationS)));
  };

  return (
    <section className={styles.card}>
      <div className={styles.head}>
        <h2>Línea de tiempo · un carril por jugador</h2>
        <div className={styles.legend}>
          {TYPES.map((type) => (
            <span key={type}>
              <ShotIcon kind={type} color="#565f70" size={13} />
              {TYPE_ES[type]}
            </span>
          ))}
        </div>
      </div>
      <div className={styles.wrap} ref={wrap}>
        <svg
          className={styles.svg}
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          onClick={seekAt}
          aria-label="Golpes del punto a lo largo del tiempo"
        >
          {PLAYERS.map((p, i) => (
            <g key={p} style={{ opacity: focus !== null && focus !== p ? 0.25 : 1, transition: "opacity .25s" }}>
              <rect x={LEFT} y={TOP + i * LANE + 4} width={width - LEFT - RIGHT} height={LANE - 8} rx={8} fill={i % 2 ? "#f7f8fb" : "#f1f3f7"} />
              <circle cx={16} cy={laneY(p)} r={5} fill={PLAYER_HEX[p]} />
              <text x={28} y={laneY(p) + 4.5} className={styles.laneLabel}>
                J{p}
              </text>
            </g>
          ))}
          {ticks.map((s) => (
            <g key={s}>
              <line x1={x(s)} x2={x(s)} y1={TOP} y2={TOP + LANE * PLAYERS.length} stroke="#e3e6ee" />
              <text x={x(s)} y={TOP + LANE * PLAYERS.length + 17} className={styles.tick}>
                {s}s
              </text>
            </g>
          ))}
          <polyline
            points={assigned.map((s) => `${x(s.t)},${laneY(s.player!)}`).join(" ")}
            fill="none"
            stroke="#b8bfcc"
            strokeDasharray="3 3"
          />
          {shots.map((shot, i) => {
            if (shot.player === null) return null;
            const cx = x(shot.t);
            const cy = laneY(shot.player);
            const dim = focus !== null && focus !== shot.player;
            const active = i === currentIndex && time - shot.t < ACTIVE_S;
            return (
              <g
                key={i}
                className={styles.shot}
                role="button"
                tabIndex={0}
                aria-label={`Golpe ${i + 1}: J${shot.player}, ${shot.type ? TYPE_ES[shot.type] : "sin clasificar"}, ${clock(shot.t)}`}
                style={{ opacity: dim ? 0.25 : 1 }}
                onClick={(event) => {
                  event.stopPropagation();
                  onSeek(shot.t - 0.6);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSeek(shot.t - 0.6);
                  }
                }}
                onMouseEnter={() => setTip({ shot, x: cx, y: cy })}
                onMouseLeave={() => setTip(null)}
              >
                <circle cx={cx} cy={cy} r={13} fill="transparent" />
                <circle cx={cx} cy={cy} r={11} fill="none" stroke="var(--accent)" strokeWidth={2} opacity={active ? 1 : 0} />
                <ShotShape
                  kind={shot.type}
                  cx={cx}
                  cy={cy}
                  r={6.5}
                  fill={PLAYER_HEX[shot.player]}
                  stroke="#fff"
                  strokeWidth={1.5}
                  style={{ opacity: shot.t <= time + 0.05 ? 1 : 0.35, transition: "opacity .2s" }}
                />
              </g>
            );
          })}
          <line x1={x(time)} x2={x(time)} y1={TOP - 4} y2={TOP + LANE * PLAYERS.length + 2} stroke="var(--accent)" strokeWidth={2} strokeLinecap="round" />
          <rect x={x(time) - 5} y={TOP + LANE * PLAYERS.length + 3} width={10} height={10} rx={2} fill="var(--accent)" />
        </svg>
        {tip && (
          <div className={styles.tip} style={{ left: tip.x - (wrap.current?.scrollLeft ?? 0), top: tip.y - 6 }}>
            <b>
              J{tip.shot.player} · {tip.shot.type ? TYPE_ES[tip.shot.type] : "Sin clasificar"}
            </b>{" "}
            · <span className="mono">{clock(tip.shot.t)}</span> · {Math.round(tip.shot.confidence * 100)} %
          </div>
        )}
      </div>
    </section>
  );
}
