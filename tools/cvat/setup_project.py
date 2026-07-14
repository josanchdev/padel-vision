"""Create the padel-court CVAT project (13-point skeleton) and upload batch 1."""

import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, "packages/cv/src")
from padel_cv.court import COURT_KEYPOINT_NAMES

BASE = "http://localhost:8080/api"
FRAMES_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/annotation/court_batch1_padelvic")
TASK_NAME = sys.argv[2] if len(sys.argv) > 2 else FRAMES_DIR.name

auth = None


def api() -> httpx.Client:
    return httpx.Client(base_url=BASE, auth=auth, timeout=120)


def build_skeleton_label() -> dict:
    # SVG template: court-like layout on a 100x100 canvas (near half at bottom).
    layout = {
        "corner_near_left": (25, 92),
        "corner_near_right": (75, 92),
        "service_near_left": (22, 72),
        "service_near_center": (50, 72),
        "service_near_right": (78, 72),
        "net_left": (18, 50),
        "net_center": (50, 50),
        "net_right": (82, 50),
        "service_far_left": (22, 28),
        "service_far_center": (50, 28),
        "service_far_right": (78, 28),
        "corner_far_left": (25, 8),
        "corner_far_right": (75, 8),
    }
    sublabels = [{"name": n, "type": "points", "attributes": []} for n in COURT_KEYPOINT_NAMES]
    circles = []
    for i, name in enumerate(COURT_KEYPOINT_NAMES, start=1):
        x, y = layout[name]
        circles.append(
            f'<circle r="1.5" cx="{x}" cy="{y}" data-type="element node" '
            f'data-element-id="{i}" data-node-id="{i}" data-label-name="{name}"/>'
        )
    return {
        "name": "court",
        "type": "skeleton",
        "attributes": [],
        "svg": "".join(circles),
        "sublabels": sublabels,
    }


def main() -> None:
    global auth
    user, password = os.environ["CVAT_USER"], os.environ["CVAT_PASS"]
    auth = httpx.BasicAuth(user, password)

    with api() as client:
        # Reuse project if a previous run already created it.
        existing = client.get("/projects", params={"search": "padel-court"}).json()["results"]
        project = next((p for p in existing if p["name"] == "padel-court"), None)
        if project is None:
            r = client.post(
                "/projects", json={"name": "padel-court", "labels": [build_skeleton_label()]}
            )
            r.raise_for_status()
            project = r.json()
            print(f"proyecto creado: id={project['id']}")
        else:
            print(f"proyecto ya existe: id={project['id']}")

        labels = client.get("/labels", params={"project_id": project["id"]}).json()["results"]
        court = next(lbl for lbl in labels if lbl["name"] == "court")
        order = [s["name"] for s in court["sublabels"]]
        print("orden de sublabels:", order)
        assert order == COURT_KEYPOINT_NAMES, "ORDEN INCORRECTO"

        r = client.post(
            "/tasks",
            json={
                "name": TASK_NAME,
                "project_id": project["id"],
                "subset": "",
            },
        )
        r.raise_for_status()
        task = r.json()
        print(f"task creada: id={task['id']}")

        files = sorted(FRAMES_DIR.glob("*.png"))
        print(f"subiendo {len(files)} imagenes...")
        upload = [
            (f"client_files[{i}]", (p.name, p.read_bytes(), "image/png"))
            for i, p in enumerate(files)
        ]
        r = client.post(
            f"/tasks/{task['id']}/data",
            data={"image_quality": 100},
            files=upload,
            headers={"Upload-Multiple": "true"},
        )
        r.raise_for_status()

        for _ in range(60):
            time.sleep(5)
            status = client.get(f"/tasks/{task['id']}").json()
            size = status.get("size") or 0
            if size > 0:
                print(f"task lista: {size} frames -> http://localhost:8080/tasks/{task['id']}")
                return
        print("timeout esperando procesado; revisar en la UI")


if __name__ == "__main__":
    main()
