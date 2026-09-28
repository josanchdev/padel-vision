// A stroke type as a shape: circle forehand, diamond backhand, triangle smash,
// square serve. Colour is left to the caller, because colour means the player.
import type { SVGProps } from "react";

import type { ShotType } from "../data.ts";

type ShapeProps = { kind: ShotType | null; cx: number; cy: number; r: number } & Omit<SVGProps<SVGElement>, "type">;

export function ShotShape({ kind, cx, cy, r, ...rest }: ShapeProps) {
  const props = rest as SVGProps<SVGPathElement> & SVGProps<SVGCircleElement>;
  switch (kind) {
    case "Forehand":
      return <circle cx={cx} cy={cy} r={r * 0.95} {...props} />;
    case "Backhand":
      return (
        <path
          d={`M${cx} ${cy - r * 1.2} L${cx + r * 1.2} ${cy} L${cx} ${cy + r * 1.2} L${cx - r * 1.2} ${cy} Z`}
          {...props}
        />
      );
    case "Smash":
      return (
        <path
          d={`M${cx} ${cy - r * 1.25} L${cx + r * 1.15} ${cy + r * 0.85} L${cx - r * 1.15} ${cy + r * 0.85} Z`}
          {...props}
        />
      );
    case "Serve":
      return (
        <rect x={cx - r * 0.9} y={cy - r * 0.9} width={r * 1.8} height={r * 1.8} rx={r * 0.2} {...(rest as SVGProps<SVGRectElement>)} />
      );
    default:
      // unclassified: a short dash, present but visibly not a verdict
      return <path d={`M${cx - r * 0.7} ${cy} h${r * 1.4}`} strokeWidth={r * 0.45} strokeLinecap="round" stroke={props.fill} {...props} />;
  }
}

export function ShotIcon({ kind, color, size = 14 }: { kind: ShotType | null; color: string; size?: number }) {
  return (
    <svg viewBox="0 0 20 20" width={size} height={size} aria-hidden="true" style={{ flex: "none" }}>
      <ShotShape kind={kind} cx={10} cy={10} r={6.5} fill={color} />
    </svg>
  );
}
