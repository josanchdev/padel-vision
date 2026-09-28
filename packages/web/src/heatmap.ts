// Heatmaps of where each player stood. Each player is scaled to his own peak,
// so one who moves little is not washed out by the others, and drawn in his own
// colour: players rarely share a zone, so four layers read as four blobs.
import { COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, SERVICE_LINES_Y, type Sample } from "./court.ts";

const CELL = 0.25;
const SIGMA = 0.55;
const W = Math.round(COURT_WIDTH_M / CELL);
const H = Math.round(COURT_LENGTH_M / CELL);

function density(samples: Sample[]): { grid: Float32Array; max: number } {
  const grid = new Float32Array(W * H);
  const radius = Math.ceil((2.2 * SIGMA) / CELL);
  for (let i = 0; i < samples.length; i += 3) {
    const [, x, y] = samples[i];
    const gx = Math.floor(x / CELL);
    const gy = Math.floor(y / CELL);
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const ix = gx + dx;
        const iy = gy + dy;
        if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
        const d2 = (dx * CELL) ** 2 + (dy * CELL) ** 2;
        grid[iy * W + ix] += Math.exp(-d2 / (2 * SIGMA * SIGMA));
      }
    }
  }
  let max = 0;
  for (const v of grid) max = Math.max(max, v);
  return { grid, max };
}

/** A small transparent canvas (one pixel per cell); scale it up with smoothing. */
export function densityCanvas(layers: { samples: Sample[]; hex: string }[]): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  for (const { samples, hex } of layers) {
    const { grid, max } = density(samples);
    const image = ctx.createImageData(W, H);
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
    for (let i = 0; i < W * H; i++) {
      const v = grid[i] / (max || 1);
      image.data[i * 4] = r;
      image.data[i * 4 + 1] = g;
      image.data[i * 4 + 2] = b;
      image.data[i * 4 + 3] = v < 0.03 ? 0 : Math.round(235 * Math.pow(v, 0.75));
    }
    const layer = document.createElement("canvas");
    layer.width = W;
    layer.height = H;
    layer.getContext("2d")!.putImageData(image, 0, 0);
    ctx.drawImage(layer, 0, 0);
  }
  return canvas;
}

/** A self-contained mini court with one player's heatmap, for the player cards. */
export function paintMiniCourt(canvas: HTMLCanvasElement, samples: Sample[], hex: string): void {
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#e9eef7";
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(densityCanvas([{ samples, hex }]), 0, 0, w, h);
  const sx = w / COURT_WIDTH_M;
  const sy = h / COURT_LENGTH_M;
  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.lineWidth = 1.5;
  for (const y of SERVICE_LINES_Y) {
    ctx.beginPath();
    ctx.moveTo(0, y * sy);
    ctx.lineTo(w, y * sy);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo((COURT_WIDTH_M / 2) * sx, SERVICE_LINES_Y[0] * sy);
  ctx.lineTo((COURT_WIDTH_M / 2) * sx, SERVICE_LINES_Y[1] * sy);
  ctx.stroke();
  ctx.strokeStyle = "rgba(15,18,24,.55)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, NET_Y_M * sy);
  ctx.lineTo(w, NET_Y_M * sy);
  ctx.stroke();
}
