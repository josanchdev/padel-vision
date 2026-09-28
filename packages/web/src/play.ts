// Vocabulary shared by every view: players, stroke types and how to write times.
// Colour means the player (the same hex as padel_cv.overlay.PLAYER_HEX, so the
// video and the charts agree); a stroke type is a shape, never a colour.
import type { ShotType } from "./data.ts";

export const PLAYERS = [1, 2, 3, 4] as const;

export const PLAYER_HEX: Record<number, string> = {
  1: "#12a150",
  2: "#f0631f",
  3: "#1b7fe0",
  4: "#6d28d9",
};

export const TYPES: ShotType[] = ["Forehand", "Backhand", "Smash", "Serve"];

export const TYPE_ES: Record<ShotType, string> = {
  Forehand: "Derecha",
  Backhand: "Revés",
  Smash: "Remate",
  Serve: "Saque",
};

/** m:ss.s — for positions inside a rally. */
export function clock(seconds: number): string {
  const s = Math.max(0, seconds);
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

/** m:ss — for a rally's length. */
export function duration(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`;
}

export function dateEs(iso: string | null): string {
  if (!iso) return "";
  return new Date(`${iso}T12:00:00`).toLocaleDateString("es-ES", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
