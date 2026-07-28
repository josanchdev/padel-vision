import { useEffect, useMemo, useRef, useState } from "react";

import type { PlayerFrame } from "./api.ts";
import { COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, SERVICE_FROM_NET_M } from "./court.ts";
import { playerColor } from "./players.ts";
import styles from "./Heatmap.module.css";

interface Props {
  players: PlayerFrame[];
}

// Density gradient (low -> high). Blue -> yellow -> red as INTENSITY, not alarm.
const GRADIENT: [number, string][] = [
  [0.0, "rgba(37, 99, 235, 0)"],
  [0.25, "rgba(37, 99, 235, 0.5)"],
  [0.5, "rgba(16, 185, 129, 0.7)"],
  [0.75, "rgba(217, 119, 6, 0.8)"],
  [1.0, "rgba(209, 16, 31, 0.9)"],
];

export function Heatmap({ players }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ids = useMemo(
    () => [...new Set(players.map((p) => p.player_id))].sort((a, b) => a - b),
    [players],
  );
  const [selected, setSelected] = useState(ids[0] ?? 1);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Court is 10x20 m (portrait). Fit to the canvas with a margin.
    const W = canvas.width;
    const H = canvas.height;
    const margin = 16;
    const scale = Math.min((W - 2 * margin) / COURT_WIDTH_M, (H - 2 * margin) / COURT_LENGTH_M);
    const offX = (W - COURT_WIDTH_M * scale) / 2;
    const offY = (H - COURT_LENGTH_M * scale) / 2;
    const toPx = (xm: number, ym: number): [number, number] => [
      offX + xm * scale,
      offY + ym * scale,
    ];

    ctx.clearRect(0, 0, W, H);

    // Court floor + lines.
    ctx.fillStyle = "#eef1f4";
    const [cx0, cy0] = toPx(0, 0);
    ctx.fillRect(cx0, cy0, COURT_WIDTH_M * scale, COURT_LENGTH_M * scale);

    // Heatmap: accumulate the selected player's positions into a density blob
    // layer, then colour-map it. Drawn before the lines so lines stay crisp.
    const pts = players.filter(
      (p) => p.player_id === selected && p.court_x_m != null && p.court_y_m != null,
    );
    if (pts.length > 0) {
      const density = document.createElement("canvas");
      density.width = W;
      density.height = H;
      const dctx = density.getContext("2d")!;
      const radius = scale * 1.6; // ~1.6 m soft blob per sample
      dctx.globalCompositeOperation = "lighter";
      for (const p of pts) {
        const [px, py] = toPx(p.court_x_m!, p.court_y_m!);
        const g = dctx.createRadialGradient(px, py, 0, px, py, radius);
        g.addColorStop(0, "rgba(0,0,0,0.10)");
        g.addColorStop(1, "rgba(0,0,0,0)");
        dctx.fillStyle = g;
        dctx.beginPath();
        dctx.arc(px, py, radius, 0, Math.PI * 2);
        dctx.fill();
      }
      colourize(dctx, W, H);
      ctx.drawImage(density, 0, 0);
    }

    // Court lines on top.
    ctx.strokeStyle = "#c3cad4";
    ctx.lineWidth = 2;
    ctx.strokeRect(cx0, cy0, COURT_WIDTH_M * scale, COURT_LENGTH_M * scale);
    line(ctx, toPx(0, NET_Y_M), toPx(COURT_WIDTH_M, NET_Y_M), "#9aa4b0", 3); // net
    for (const y of [NET_Y_M - SERVICE_FROM_NET_M, NET_Y_M + SERVICE_FROM_NET_M]) {
      line(ctx, toPx(0, y), toPx(COURT_WIDTH_M, y), "#c3cad4", 1);
    }
    line(ctx, toPx(COURT_WIDTH_M / 2, NET_Y_M - SERVICE_FROM_NET_M),
         toPx(COURT_WIDTH_M / 2, NET_Y_M + SERVICE_FROM_NET_M), "#c3cad4", 1); // centre
  }, [players, selected]);

  return (
    <div className={styles.wrap}>
      <div className={styles.chips}>
        {ids.map((id) => (
          <button
            key={id}
            className={`${styles.chip} ${selected === id ? styles.active : ""}`}
            onClick={() => setSelected(id)}
            style={{ ["--c" as string]: playerColor(id) }}
          >
            <span className={styles.dot} style={{ background: playerColor(id) }} />J{id}
          </button>
        ))}
      </div>
      <canvas ref={canvasRef} width={320} height={520} className={styles.canvas} />
    </div>
  );
}

function line(
  ctx: CanvasRenderingContext2D,
  [x0, y0]: [number, number],
  [x1, y1]: [number, number],
  color: string,
  width: number,
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// Map the grayscale density (alpha) to the colour gradient.
function colourize(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const ramp = buildRamp();
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const a = Math.min(d[i + 3] / 255, 1);
    if (a === 0) continue;
    const c = ramp[Math.floor(a * 255)];
    d[i] = c[0];
    d[i + 1] = c[1];
    d[i + 2] = c[2];
    d[i + 3] = c[3];
  }
  ctx.putImageData(img, 0, 0);
}

function buildRamp(): [number, number, number, number][] {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 1;
  const ctx = c.getContext("2d")!;
  const g = ctx.createLinearGradient(0, 0, 256, 0);
  for (const [stop, color] of GRADIENT) g.addColorStop(stop, color);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 1);
  const data = ctx.getImageData(0, 0, 256, 1).data;
  const ramp: [number, number, number, number][] = [];
  for (let i = 0; i < 256; i++) {
    ramp.push([data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]]);
  }
  return ramp;
}
