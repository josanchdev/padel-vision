"""End-to-end smoke test of the API: submit a video, poll, download result."""

import sys
import time
from pathlib import Path

import httpx

BASE = "http://localhost:8000"
VIDEO = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/2022_BCN_FinalF_1_sample.mp4")

with httpx.Client(base_url=BASE, timeout=120) as client:
    assert client.get("/health").json() == {"status": "ok"}, "API no responde"
    print("health OK")

    with VIDEO.open("rb") as fh:
        response = client.post("/matches", files={"video": (VIDEO.name, fh, "video/mp4")})
    response.raise_for_status()
    match = response.json()
    match_id = match["id"]
    print(f"subido -> id={match_id} status={match['status']}")

    last = None
    deadline = time.time() + 900
    while time.time() < deadline:
        state = client.get(f"/matches/{match_id}").json()
        line = f"{state['status']} {state['progress']:.0%}"
        if line != last:
            print(f"  {line}")
            last = line
        if state["status"] in ("done", "failed"):
            break
        time.sleep(3)

    if state["status"] != "done":
        print(f"FALLO: {state.get('error')}")
        raise SystemExit(1)

    print(f"golpes detectados: {state['shots_detected']}")
    result = client.get(f"/matches/{match_id}/result")
    result.raise_for_status()
    out = Path("output/e2e_result.mp4")
    out.write_bytes(result.content)
    print(f"resultado descargado: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("END-TO-END OK")
