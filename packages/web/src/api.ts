// Typed API client — the connection layer, ported from the mockup's api{}.
// This is the part that survives redesigns; components consume these types.
// Shapes mirror the JSON schema written by the pipeline (ADR-0010).

export interface Shot {
  frame_index: number;
  timestamp_s: number;
  player_id: number;
  label: string;
  confidence: number;
}

export interface Bounce {
  frame_index: number;
  image_x: number;
  image_y: number;
}

export interface PlayerFrame {
  frame_index: number;
  player_id: number;
  court_x_m: number | null;
  court_y_m: number | null;
  on_court: boolean | null;
}

export interface MatchData {
  schema_version: number;
  source_video: string;
  fps: number;
  shots: Shot[];
  bounces: Bounce[];
  players: PlayerFrame[];
  ball: unknown[];
}

export type MatchStatus = "pending" | "processing" | "done" | "failed";

export interface Match {
  id: string;
  filename: string;
  status: MatchStatus;
  progress: number;
  shots_detected: number;
  duration_s?: number | null;
  created_at: string;
  error?: string | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  listMatches: () => fetch("/matches").then(json<Match[]>),
  getMatch: (id: string) => fetch(`/matches/${id}`).then(json<Match>),
  getData: (id: string) => fetch(`/matches/${id}/data`).then(json<MatchData>),
  resultUrl: (id: string) => `/matches/${id}/result`,
  thumbnailUrl: (id: string) => `/matches/${id}/thumbnail`,
  clipUrl: (id: string, frame: number) => `/matches/${id}/clip?frame=${frame}`,
  async upload(file: File): Promise<Match> {
    const body = new FormData();
    body.append("video", file);
    return fetch("/matches", { method: "POST", body }).then(json<Match>);
  },
};

// Clip geometry shared with the backend (cut_clip margin). The shot sits at
// min(margin, frame/fps) seconds into the clip — used to place the marker.
export const CLIP_MARGIN_S = 3;

export function shotOffsetInClip(frame: number, fps: number): number {
  return Math.min(CLIP_MARGIN_S, frame / fps);
}
