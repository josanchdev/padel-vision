"""Ground truth for the hit-assignment task (CVSPORTS_Padel, Decorte 2024).

The dataset ships 319 hits (16 VIGO rallies) annotated with WHICH player hit the
ball, as `hit_assignments.xlsx`. That is the exact subset the paper reports its
83.70% player / 86.83% team accuracy on (Table 3), so it lets us measure our
replica against their published number.

Labels are strings "t{team}p{player}" (e.g. "t2p1"). The paper numbers players
1-4 by starting position (top team 1-2 left-to-right, bottom team 3-4), so we
map: t1p1->1, t1p2->2, t2p1->3, t2p2->4.

Read without openpyxl: xlsx is a zip of XML.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

#: "t{team}p{n}" -> player slot 1-4 (paper's starting-position numbering).
LABEL_TO_SLOT = {"t1p1": 1, "t1p2": 2, "t2p1": 3, "t2p2": 4}


@dataclass(frozen=True)
class HitAssignment:
    """One annotated hit: which rally, when, and who hit it."""

    video: str
    time_s: float
    slot: int
    """Player 1-4."""

    @property
    def team(self) -> int:
        """1 for players 1-2, 2 for players 3-4."""
        return 1 if self.slot <= 2 else 2


def _parse_timestamp(text: str) -> float:
    """'MM:SS.mmm' -> seconds."""
    minutes, seconds = text.split(":")
    return int(minutes) * 60 + float(seconds)


def load_hit_assignments(xlsx_path: Path) -> dict[str, list[HitAssignment]]:
    """rally name (no extension) -> its annotated hits, ordered by time."""
    with zipfile.ZipFile(xlsx_path) as z:
        strings = [
            "".join(t.text or "" for t in si.iter(f"{_NS}t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml"))
        ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    out: dict[str, list[HitAssignment]] = {}
    for row in sheet.iter(f"{_NS}row"):
        values: list[str] = []
        for cell in row.iter(f"{_NS}c"):
            node = cell.find(f"{_NS}v")
            text = "" if node is None else (node.text or "")
            if cell.get("t") == "s" and text:
                text = strings[int(text)]
            values.append(text)
        if len(values) < 4 or values[1] == "video":  # header row
            continue
        video, timestamp, label = values[1], values[2], values[3]
        slot = LABEL_TO_SLOT.get(label)
        if slot is None:
            continue
        out.setdefault(video, []).append(
            HitAssignment(video=video, time_s=_parse_timestamp(timestamp), slot=slot)
        )
    for hits in out.values():
        hits.sort(key=lambda h: h.time_s)
    return out
