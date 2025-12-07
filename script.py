#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import shlex
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import requests
from deep_sort_realtime.deepsort_tracker import DeepSort
from dotenv import load_dotenv
from ultralytics import YOLO

# -------------------------- configuration --------------------------

load_dotenv()

def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or not str(val).strip()):
        raise SystemExit(f"Missing env: {name}")
    return str(val)

CVAT_URL: str           = _env("CVAT_URL", required=True).rstrip("/")
API_BASE: str           = f"{CVAT_URL}/api"
CVAT_USER: str          = _env("CVAT_USER", required=True)
CVAT_PASS: str          = _env("CVAT_PASS", required=True)
PROJECT_ID: int         = int(_env("CVAT_PROJECT_ID", required=True))
PERSON_LABEL_NAME: str  = _env("PERSON_LABEL_NAME", "person")
SHARE_HOST: Path        = Path(_env("SHARE_HOST", "/srv/cvat_share"))

TARGET_FPS = 15

YOLO_WEIGHTS_DEFAULT    = "yolo11l-pose.pt"
YOLO_IMGSZ_DEFAULT      = 2560
YOLO_CONF_DEFAULT       = 0.5
YOLO_DEVICE_DEFAULT     = "cuda:0"

HTTP_TIMEOUT = (10, 600)

SESSION = requests.Session()
SESSION.auth = (CVAT_USER, CVAT_PASS)

log = logging.getLogger("cvat_pose")


# -------------------------- cvat api --------------------------

def cvat_create_task(project_id: int, name: str) -> int:
    r = SESSION.post(
        f"{API_BASE}/tasks",
        json={"name": name, "project_id": project_id},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    task_id = int(r.json()["id"])
    log.info("Created task %s (%s)", task_id, name)
    return task_id


def cvat_upload_from_share(
    task_id: int,
    rel_files: Iterable[str],
    *,
    image_quality: int = 70,
    frame_step: int = 1,
    segment_size: int = 1_000_000,
    chunk_size: int = 500,
    copy_data: bool = False,
    use_cache: bool = True,
) -> None:
    data = {
        "storage": "share",
        "data_type": "images",
        "image_quality": str(image_quality),
        "frame_step": str(frame_step),
        "segment_size": str(segment_size),
        "chunk_size": str(chunk_size),
        "copy_data": str(copy_data).lower(),
        "use_cache": str(use_cache).lower(),
    }
    files = [(f"server_files[{i}]", (None, rel)) for i, rel in enumerate(rel_files)]
    r = SESSION.post(
        f"{API_BASE}/tasks/{task_id}/data", data=data, files=files, timeout=HTTP_TIMEOUT
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Upload failed: HTTP {r.status_code}\n{r.text[:800]}")
    log.info("Share upload accepted by CVAT")


def cvat_list_jobs(task_id: int) -> List[Dict[str, Any]]:
    # Try new endpoint first
    r = SESSION.get(f"{API_BASE}/tasks/{task_id}/jobs", params={"page_size": 500}, timeout=HTTP_TIMEOUT)
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, dict) and "results" in data:
            return list(data["results"])
        if isinstance(data, list):
            return data
    # Fallback to legacy
    r = SESSION.get(f"{API_BASE}/jobs", params={"task_id": task_id, "page_size": 500}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "results" in data:
        return list(data["results"])
    if isinstance(data, list):
        return data
    return []


def cvat_wait_for_job(task_id: int, *, timeout_s: int = 900, poll_s: float = 2.0) -> int:
    log.info("Waiting for job creation...")
    t0 = time.time()
    last_state = None

    while True:
        # status endpoint
        try:
            s = SESSION.get(f"{API_BASE}/tasks/{task_id}/status", timeout=HTTP_TIMEOUT)
            if s.ok:
                state = (s.json().get("state") or s.json().get("status") or "").lower()
                if state and state != last_state:
                    log.debug("Task %s state: %s", task_id, state)
                    last_state = state
        except Exception:
            pass

        jobs = []
        try:
            jobs = cvat_list_jobs(task_id)
        except Exception:
            pass

        if jobs:
            jid = int(jobs[0]["id"])
            log.info("Job %s ready", jid)
            return jid

        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timeout waiting for jobs for task {task_id}")
        time.sleep(poll_s)


def cvat_get_skeleton_spec(task_id: int, label_name: str) -> Tuple[int, List[Dict[str, Any]]]:
    r = SESSION.get(f"{API_BASE}/labels", params={"task_id": task_id, "page_size": 500}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    labels = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [])
    for lab in labels:
        if lab.get("name") == label_name and lab.get("type") == "skeleton":
            subs = [{"id": s["id"], "name": s["name"]} for s in (lab.get("sublabels") or [])]
            if len(subs) != 17:
                raise SystemExit(f'Skeleton "{label_name}" has {len(subs)} sublabels, expected 17.')
            return int(lab["id"]), subs
    raise SystemExit(f'Skeleton label "{label_name}" not found in task {task_id}.')


# --- NUEVO: obtener especificación del tag scene / group_activity ---
def cvat_get_tag_spec(task_id: int, label_name: str, attr_name: str) -> Tuple[int, Dict[str, Any]]:
    r = SESSION.get(f"{API_BASE}/labels", params={"task_id": task_id, "page_size": 500}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    labels = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [])
    for lab in labels:
        if lab.get("name") == label_name and lab.get("type") == "tag":
            for attr in lab.get("attributes") or []:
                if attr.get("name") == attr_name:
                    return int(lab["id"]), attr
    raise SystemExit(f'Tag "{label_name}" with attribute "{attr_name}" not found in task {task_id}.')


# --- MODIFICADO: ahora también puede subir tags ---
def cvat_upload_tracks(
    job_id: int,
    tracks_payload: List[Dict[str, Any]],
    tags_payload: List[Dict[str, Any]] | None = None,
) -> None:
    if not tracks_payload and not tags_payload:
        return

    body: Dict[str, Any] = {}
    if tracks_payload:
        body["tracks"] = tracks_payload
    if tags_payload:
        body["tags"] = tags_payload

    r = SESSION.put(
        f"{API_BASE}/jobs/{job_id}/annotations",
        params={"action": "create"},
        json=body,
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Annotations upload failed: HTTP {r.status_code}\n{r.text[:800]}")
    log.info(
        "Uploaded %d tracks and %d tags",
        len(tracks_payload),
        len(tags_payload or []),
    )


# -------------------------- media/io --------------------------

def extract_frames_ffmpeg(video_path: str, rel_dir: str, fps: float) -> List[str]:
    out_dir = SHARE_HOST / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-start_number", "0",
        out_pattern,
    ]
    log.info("Extracting frames → %s @ %.1f FPS", rel_dir, fps)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:500] or result.stdout.strip()[:500]}")

    rels = sorted(f"{rel_dir}/{p.name}" for p in out_dir.glob("*.jpg"))
    if not rels:
        raise RuntimeError(f"No frames found in {out_dir}")
    return rels


# -------------------------- detection/tracking --------------------------

def _iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))

def extract_person_tracks_from_jpgs(
    frame_rels: List[str],
    *,
    model_path: str = YOLO_WEIGHTS_DEFAULT,
    imgsz: int = YOLO_IMGSZ_DEFAULT,
    conf: float = YOLO_CONF_DEFAULT,
    device: str = YOLO_DEVICE_DEFAULT,
    out_preview: str | None = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:

    # lee primer frame para dimensiones
    first = cv2.imread(str(SHARE_HOST / frame_rels[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frame_rels[0]}")
    height, width = first.shape[:2]

    model = YOLO(model_path)
    tracker = DeepSort(
        max_age=50, n_init=5,
        nms_max_overlap=0.7,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        embedder="mobilenet",
        half=True, bgr=True,
    )

    preview_writer = None
    if out_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(out_preview, fourcc, TARGET_FPS, (width, height))

    tracks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    log.info("Running YOLO-Pose + DeepSORT over extracted JPGs (%d frames) @ %.1f FPS",
             len(frame_rels), TARGET_FPS)

    for fidx, rel in enumerate(frame_rels):
        frame = cv2.imread(str(SHARE_HOST / rel))
        if frame is None:
            continue

        results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=0.65, device=device, verbose=False)

        dets_xyxy: List[List[float]] = []
        det_kpts: List[List[List[float]]] = []
        tracker_inputs = []

        if results:
            r = results[0]
            boxes = r.boxes
            kps = getattr(r, "keypoints", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                kp_xy = kps.xy.cpu().numpy() if (kps is not None and getattr(kps, "xy", None) is not None) else None

                for i in range(len(xyxy)):
                    if clss[i] != 0 or kp_xy is None:
                        continue
                    det_box = xyxy[i].tolist()
                    dets_xyxy.append(det_box)
                    det_kpts.append(kp_xy[i].tolist())
                    tracker_inputs.append(([det_box[0], det_box[1],
                                            det_box[2]-det_box[0], det_box[3]-det_box[1]],
                                           float(confs[i]), "person"))

        track_objs = tracker.update_tracks(tracker_inputs, frame=frame)

        for t in track_objs:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            tb = list(map(float, t.to_ltrb()))
            # empareja bbox→kpts por IoU
            best_j, best_iou = -1, 0.0
            for j, bb in enumerate(dets_xyxy):
                iou = _iou_xyxy(tb, bb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j == -1:
                continue
            kps = det_kpts[best_j]
            if len(kps) != 17:
                continue
            tracks[int(t.track_id)].append({"frame": fidx, "kpts_xy": kps})

            if preview_writer:
                x1, y1, x2, y2 = map(int, tb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(t.track_id)}", (x1, max(15, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if preview_writer:
            preview_writer.write(frame)

    if preview_writer:
        preview_writer.release()

    meta = {
        "in_fps": TARGET_FPS,
        "target_fps": TARGET_FPS,
        "in_frames": len(frame_rels),
        "out_frames": len(frame_rels),
        "width": width,
        "height": height,
    }
    return tracks, meta



# -------------------------- cvat payload --------------------------
def to_cvat_track_with_elements(parent_label_id, sublabels, frame_samples, *, img_w=None, img_h=None):
    if not frame_samples:
        return None

    elem_tracks = [{
        "type": "points",
        "label_id": int(sub["id"]),
        "frame": int(frame_samples[0]["frame"]),
        "group": 0,
        "source": "manual",
        "attributes": [],
        "shapes": []
    } for sub in sublabels]

    skel_shapes = []

    for sample in sorted(frame_samples, key=lambda s: s["frame"]):
        f = int(sample["frame"])
        kpts = sample["kpts_xy"]
        if len(kpts) != 17:
            continue

        skel_shapes.append({
            "type": "skeleton",
            "label_id": int(parent_label_id),
            "frame": f,
            "outside": False,
            "occluded": False,
            "keyframe": True,
            "z_order": 0,
            "attributes": [],
        })

        for idx, (x, y) in enumerate(kpts):
            if img_w is not None and img_h is not None:
                x = max(0.0, min(float(x), img_w - 1))
                y = max(0.0, min(float(y), img_h - 1))
            elem_tracks[idx]["shapes"].append({
                "type": "points",
                "frame": f,
                "points": [float(x), float(y)],
                "rotation": 0,
                "outside": False,
                "occluded": False,
                "keyframe": True,
                "z_order": 0,
                "attributes": [],
            })

    # --- cierre explícito del track ---
    last_f = int(max(s["frame"] for s in frame_samples))
    end_f  = last_f + 1  # frame de cierre

    skel_shapes.append({
        "type": "skeleton",
        "label_id": int(parent_label_id),
        "frame": end_f,
        "outside": True,
        "occluded": False,
        "keyframe": True,
        "z_order": 0,
        "attributes": [],
    })
    for et in elem_tracks:
        et["shapes"].append({
            "type": "points",
            "frame": end_f,
            "points": [0.0, 0.0],  # coordenadas ignoradas cuando outside=True
            "rotation": 0,
            "outside": True,
            "occluded": False,
            "keyframe": True,
            "z_order": 0,
            "attributes": [],
        })
    # ----------------------------------

    return {
        "label_id": int(parent_label_id),
        "frame": int(frame_samples[0]["frame"]),
        "group": 0,
        "source": "manual",
        "attributes": [],
        "shapes": skel_shapes,
        "elements": elem_tracks,
    }

# -------------------------- cli --------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create CVAT task from frames and upload YOLO-Pose+DeepSORT skeleton tracks."
    )
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--rel-dir", required=True, help="Relative dir under CVAT share (e.g. dataset/clip_001)")
    ap.add_argument("--model", default=YOLO_WEIGHTS_DEFAULT, help="YOLO weights")
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ_DEFAULT, help="Model image size")
    ap.add_argument("--conf", type=float, default=YOLO_CONF_DEFAULT, help="Confidence threshold")
    ap.add_argument("--device", default=YOLO_DEVICE_DEFAULT, help="Device: cuda:0 | cpu")
    ap.add_argument("--preview", action="store_true", help="Save an annotated preview video")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.verbose == 0 else (logging.INFO if args.verbose == 1 else logging.DEBUG),
        format="%(levelname)s: %(message)s",
    )

    video_path = Path(args.video)
    clip_id = video_path.stem
    rel_dir = args.rel_dir.strip("/")

    frames = extract_frames_ffmpeg(str(video_path), rel_dir, TARGET_FPS)

    task_id = cvat_create_task(PROJECT_ID, f"pose_{clip_id}")
    cvat_upload_from_share(task_id, frames)
    job_id = cvat_wait_for_job(task_id)

    preview_path = None
    if args.preview:
        Path("preview").mkdir(exist_ok=True)
        preview_path = str(Path("preview") / f"{clip_id}_preview.mp4")

    tracks, meta = extract_person_tracks_from_jpgs(
        frames,
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        out_preview=preview_path,
    )

    Path("tracks").mkdir(parents=True, exist_ok=True)
    with open(Path("tracks") / f"{clip_id}.json", "w") as f:
        json.dump({"clip_id": clip_id, "meta": meta, "tracks": tracks}, f, indent=2)
    log.info("Saved tracks/%s.json", clip_id)

    skel_label_id, sublabels = cvat_get_skeleton_spec(task_id, PERSON_LABEL_NAME)
    payload: List[Dict[str, Any]] = []
    for samples in tracks.values():
        track = to_cvat_track_with_elements(skel_label_id, sublabels, samples)
        if track:
            payload.append(track)

    # --- NUEVO: construir tag scene en frame 0 ---
    scene_label_id, scene_attr = cvat_get_tag_spec(task_id, "scene", "group_activity")
    default_val = scene_attr.get("default_value") or (scene_attr.get("values") or ["transit"])[0]
    tags_payload: List[Dict[str, Any]] = [
        {
            "frame": 0,
            "label_id": int(scene_label_id),
            "group": 0,
            "source": "manual",
            "attributes": [
                {
                    "spec_id": int(scene_attr["id"]),
                    "name": str(scene_attr["name"]),
                    "value": str(default_val),
                }
            ],
        }
    ]

    if payload or tags_payload:
        cvat_upload_tracks(job_id, payload, tags_payload)
        log.info(
            "Uploaded %d tracks and %d tags to job %s",
            len(payload),
            len(tags_payload),
            job_id,
        )
    else:
        log.warning("No valid tracks or tags to upload")

    log.info("Done. Task %s (job %s).", task_id, job_id)


if __name__ == "__main__":
    main()

