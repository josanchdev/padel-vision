"""Does the assignment vote look at the wrong frames? (tutor's hypothesis)

When this was run the vote was SYMMETRIC around the instant: [hit-6, hit+6]. The
suggestion under test was that this is wrong in a specific way — a few frames
after contact the ball is already flying to the far court, so those frames vote
for whoever is now near it rather than for whoever hit it. Any lag between the
audio instant and the true contact frame makes it worse, since it shifts the
whole window into the "ball has left" region.

Two knobs, swept against the paper's 319-hit ground truth:

    offset          shift the window centre (negative = earlier), tests the lag
    before/after    window bounds, tests the asymmetry

The classifier's own window is NOT a candidate: it runs from the opponent's
previous hit to the next one (up to 1.5 s), which is longer and would pull in
even more frames where the ball is across the net. The useful part of the
tutor's idea is the asymmetry, not that particular window.

Outcome: ±4 frames (`hit_assignment.WINDOW_HALF`). Runs on `build_states` and
the hand-marked court — the same path as the published evidence — so the rows
are comparable to `hit_assignment_replica.json`. A cached variant was tried
first and was not comparable: it skipped the court mask and identity tracking.

    uv run python scripts/experiment_assignment_window.py
"""

from __future__ import annotations

import argparse
import json

from padel_ml.ball_infer import BallDetector
from padel_ml.hit_assignment import (
    MAX_POSE_GAP,
    WINDOW_HALF,
    FrameState,
    assign_hit,
    team_alternation_sweep,
)
from padel_ml.hit_assignment_eval import annotated_rallies
from padel_ml.rally_analysis import POSE_CONFIDENCE

from padel_cv.paths import COURTS, CVSPORTS, METRICS, RUNS
from padel_cv.stages.pose import PlayerPoseStage


def assign_asymmetric(
    hit_frame: int, states: dict[int, FrameState], before: int, after: int, offset: int
) -> int | None:
    """`assign_hit` over exactly [hit+offset-before, hit+offset+after].

    The voting rule is left untouched — only which frames reach it changes. The
    window is applied by blanking the ball outside it rather than by re-centring,
    because a centre/half pair cannot express an odd-width window without
    silently shifting it by a frame, and because `_bridge_gaps` must still see
    the poses just outside to recover a blinking player.
    """
    low, high = hit_frame + offset - before, hit_frame + offset + after
    restricted = {
        f: (st if low <= f <= high else FrameState(poses=st.poses, ball=None))
        for f, st in states.items()
        if low - MAX_POSE_GAP <= f <= high + MAX_POSE_GAP
    }
    centre = (low + high) // 2
    return assign_hit(centre, restricted, window_half=max(high - low, 1))


def main() -> None:
    pose_stage = PlayerPoseStage(confidence=POSE_CONFIDENCE)  # as analyze_rally
    ball_detector = BallDetector(RUNS / "ball_full" / "tracknetv3.pt")

    rallies: list[tuple[dict[int, FrameState], list[tuple[int, int, int]]]] = []
    for annotated in annotated_rallies(CVSPORTS, COURTS, pose_stage, ball_detector):
        fps = annotated.fps
        hits = [(round(h.time_s * fps), h.slot, h.team) for h in annotated.hits if h.slot]
        rallies.append((annotated.states, hits))
        print(f"  {annotated.name}: {len(hits)} golpes", flush=True)
    print(f"\n{len(rallies)} rallies, {sum(len(h) for _, h in rallies)} golpes con verdad\n")

    def score(before: int, after: int, offset: int) -> tuple[float, float]:
        """Player and team accuracy, scored exactly as the published evidence."""
        player_ok = team_ok = n = 0
        for states, hits in rallies:
            predictions = {
                frame: assign_asymmetric(frame, states, before, after, offset)
                for frame, _, _ in hits
            }
            predictions = team_alternation_sweep(predictions)
            for frame, slot, team in hits:
                prediction = predictions[frame]
                n += 1
                if prediction is None:
                    continue
                if prediction < 0:  # team-only recovery from the sweep
                    team_ok += int(-prediction == team)
                    continue
                player_ok += int(prediction == slot)
                team_ok += int((1 if prediction <= 2 else 2) == team)
        return 100 * player_ok / max(n, 1), 100 * team_ok / max(n, 1)

    results: list[dict[str, float | int | str]] = []

    def row(kind: str, before: int, after: int, offset: int) -> None:
        player, team = score(before, after, offset)
        baseline = (before, after, offset) == (WINDOW_HALF, WINDOW_HALF, 0)
        mark = "   <- actual" if baseline else ""
        if kind == "offset":
            print(f"   {offset:+3d}      {player:5.2f}%   {team:5.2f}%{mark}", flush=True)
        else:
            print(f"   {before:4d}  {after:6d}   {player:5.2f}%   {team:5.2f}%{mark}", flush=True)
        results.append(
            {
                "kind": kind,
                "before": before,
                "after": after,
                "offset": offset,
                "player": round(player, 2),
                "team": round(team, 2),
            }
        )

    print("A) DESFASE: misma ventana +-6, movida en el tiempo")
    print("   offset   jugador   equipo")
    for offset in range(-6, 5):
        row("offset", 6, 6, offset)

    print("\nB) ASIMETRIA: cuantos frames antes / despues del golpe")
    print("   antes  despues   jugador   equipo")
    for before, after in [
        (6, 6),
        (6, 4),
        (6, 3),
        (6, 2),
        (6, 0),
        (8, 2),
        (8, 4),
        (4, 4),
        (5, 5),
        (3, 6),
        (0, 6),
        (10, 4),
    ]:
        row("window", before, after, 0)

    out = METRICS / "assignment_window_sweep.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()
