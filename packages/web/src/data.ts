// The files the viewer reads, written by scripts/export_points.py (ADR-0017).
// The contract lives in packages/ml/src/padel_ml/web_export.py; these types
// mirror it. There is no API: every point is a folder of static files.

export type ShotType = "Forehand" | "Backhand" | "Smash" | "Serve";

export interface Shot {
  frame: number;
  t: number;
  /** 1-4, or null when no player could be tied to the hit. */
  player: number | null;
  /** null when the hit was not classified. */
  type: ShotType | null;
  confidence: number;
}

export interface Point {
  schema_version: number;
  id: string;
  title: string;
  city: string | null;
  date: string | null;
  fps: number;
  frames: number;
  duration_s: number;
  shots: Shot[];
  /** Per player: [frame, x_m, y_m], y from the camera's baseline (0) to the far one (20). */
  players: Record<string, [number, number, number][]>;
}

export interface PointSummary {
  id: string;
  title: string;
  city: string | null;
  date: string | null;
  duration_s: number;
  shots: number;
}

const BASE = import.meta.env.BASE_URL;

export const pointFile = (id: string, file: string) => `${BASE}points/${id}/${file}`;

export async function loadIndex(): Promise<PointSummary[]> {
  const response = await fetch(`${BASE}points/index.json`);
  if (!response.ok) return []; // nothing exported yet
  const index = (await response.json()) as { points: PointSummary[] };
  return index.points;
}

export async function loadPoint(id: string): Promise<Point> {
  const response = await fetch(pointFile(id, "point.json"));
  if (!response.ok) throw new Error(`No se encuentra el punto ${id}`);
  return (await response.json()) as Point;
}
