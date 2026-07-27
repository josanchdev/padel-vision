// Per-player identity colour (matches theme tokens --p1..--p4).
const PLAYER_COLORS: Record<number, string> = {
  1: "var(--p1)",
  2: "var(--p2)",
  3: "var(--p3)",
  4: "var(--p4)",
};

export function playerColor(id: number): string {
  return PLAYER_COLORS[id] ?? "var(--fg-muted)";
}

export function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${sec}`;
}
