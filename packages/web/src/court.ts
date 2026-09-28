// Court geometry in metres (mirrors padel_cv.court) and the top-down view.
// Padel court: 10 x 20 m, net across the middle, service lines 6.95 m from it.
import type { Point } from "./data.ts";

export const COURT_WIDTH_M = 10;
export const COURT_LENGTH_M = 20;
export const NET_Y_M = 10;
export const SERVICE_LINES_Y = [NET_Y_M - 6.95, NET_Y_M + 6.95];

/** [frame, x, y] in VIEW coordinates: y flipped so the camera side is at the bottom. */
export type Sample = [number, number, number];

/**
 * Per player, positions ready to draw. The data's y grows away from the camera
 * (players 3-4 play near it in CVSPORTS), so it is flipped once here and every
 * view draws the court as the video shows it: camera side at the bottom.
 */
export function viewPositions(point: Point): Map<number, Sample[]> {
  const out = new Map<number, Sample[]>();
  for (const [player, samples] of Object.entries(point.players)) {
    out.set(
      Number(player),
      samples.map(([frame, x, y]) => [frame, x, COURT_LENGTH_M - y]),
    );
  }
  return out;
}

/** The latest position at or shortly before `frame` (the detector blinks). */
export function positionAt(byFrame: Map<number, Sample>, frame: number): Sample | undefined {
  for (let k = 0; k < 12; k++) {
    const sample = byFrame.get(frame - k);
    if (sample) return sample;
  }
  return undefined;
}
