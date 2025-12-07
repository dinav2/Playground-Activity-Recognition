#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import yaml


CENTROIDS_PATH = Path("data/centroids.yaml")


def is_missing_label(val: str | None) -> bool:
    """
    True if the value is unknown and it can be replaced by a previous value
    """
    if val is None:
        return True
    if not isinstance(val, str):
        return False
    return val.strip().lower() in {"unkown", "none", ""}


@lru_cache(maxsize=1)
def load_centroids() -> dict:
    """
    Load data/centroids.yaml and return cameras dict:
    """
    with CENTROIDS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cams = data.get("cameras", {})
    return cams


def get_camera_info(video_id: str):
    """
    Find the video's prefix centroids
    """
    cameras = load_centroids()
    for cam_key, cam_info in cameras.items():
        if video_id.startswith(cam_key + "-"):
            return cam_key, cam_info
    return None, None


def cvat_to_intermediate(input_path: Path, video_id: str | None, fps: float | None):
    ann = json.loads(input_path.read_text(encoding="utf-8"))

    tracks = ann.get("tracks", [])
    tags = ann.get("tags", [])

    max_frame = -1

    for tag in tags:
        max_frame = max(max_frame, tag.get("frame", -1))

    for tr in tracks:
        max_frame = max(max_frame, tr.get("frame", -1))
        for s in tr.get("shapes", []):
            max_frame = max(max_frame, s.get("frame", -1))
        for el in tr.get("elements", []):
            max_frame = max(max_frame, el.get("frame", -1))
            for s in el.get("shapes", []):
                max_frame = max(max_frame, s.get("frame", -1))

    num_frames = max_frame + 1 if max_frame >= 0 else 0

    SCENE_ATTR_SPEC_ID = int(os.getenv("SCENE_ATTR_SPEC_ID", "2"))
    ROLE_ATTR_SPEC_ID = int(os.getenv("ROLE_ATTR_SPEC_ID", "5"))
    ACTION_ATTR_SPEC_ID = int(os.getenv("ACTION_ATTR_SPEC_ID", "4"))
    SAFETY_ATTR_SPEC_ID = int(os.getenv("SAFETY_ATTR_SPEC_ID", "6"))

    scene_class = None
    for tag in tags:
        for a in tag.get("attributes", []):
            if a.get("spec_id") == SCENE_ATTR_SPEC_ID:
                scene_class = a.get("value")
                break
        if scene_class is not None:
            break

    frames_dict: dict[int, dict] = defaultdict(lambda: {"persons": []})

    for tr in tracks:
        track_id = tr.get("id")

        # role (adult/child/teen)
        role = None
        for a in tr.get("attributes", []):
            if a.get("spec_id") == ROLE_ATTR_SPEC_ID:
                role = a.get("value")
                break

        skeleton_attrs_by_frame: dict[int, dict] = {}
        for s in tr.get("shapes", []):
            if s.get("type") != "skeleton":
                continue
            frame = s.get("frame")
            if s.get("outside"):
                continue

            action = None
            safety = None
            for a in s.get("attributes", []):
                if a.get("spec_id") == ACTION_ATTR_SPEC_ID:
                    action = a.get("value")
                elif a.get("spec_id") == SAFETY_ATTR_SPEC_ID:
                    safety = a.get("value")

            skeleton_attrs_by_frame[frame] = {
                "action": action,
                "safety_flag": safety,
            }

        if not skeleton_attrs_by_frame:
            continue

        joint_labels = sorted({el.get("label_id") for el in tr.get("elements", [])})
        label_to_joint_idx = {lid: idx for idx, lid in enumerate(joint_labels)}
        J = len(joint_labels)

        keypoints_by_frame: dict[int, list[list[float]]] = defaultdict(
            lambda: [[math.nan, math.nan] for _ in range(J)]
        )

        for el in tr.get("elements", []):
            lid = el.get("label_id")
            ji = label_to_joint_idx.get(lid)
            if ji is None:
                continue

            for s in el.get("shapes", []):
                if s.get("type") != "points":
                    continue
                frame = s.get("frame")
                if s.get("outside"):
                    continue

                pts = s.get("points", [])
                if len(pts) >= 2:
                    keypoints_by_frame[frame][ji] = [float(pts[0]), float(pts[1])]

        for frame, attr in skeleton_attrs_by_frame.items():
            persons_list = frames_dict[frame]["persons"]
            kps = keypoints_by_frame.get(frame)
            if kps is None:
                kps = [[math.nan, math.nan] for _ in range(J)]

            persons_list.append(
                {
                    "track_id": track_id,
                    "role": role,
                    "action": attr.get("action"),
                    "safety_flag": attr.get("safety_flag"),
                    "keypoints": kps,
                }
            )

    track_to_items: dict[int, list[tuple[int, dict]]] = defaultdict(list)

    for frame in sorted(frames_dict.keys()):
        for person in frames_dict[frame]["persons"]:
            tid = person.get("track_id")
            if tid is None:
                continue
            track_to_items[tid].append((frame, person))

    for tid, items in track_to_items.items():
        items.sort(key=lambda x: x[0])

        last_action = None
        last_safety = None

        for frame, person in items:
            act = person.get("action")
            saf = person.get("safety_flag")

            # action
            if is_missing_label(act):
                if last_action is not None:
                    person["action"] = last_action
            else:
                last_action = act

            # safety_flag
            if is_missing_label(saf):
                if last_safety is not None:
                    person["safety_flag"] = last_safety
            else:
                last_safety = saf

    video_id_str = video_id or input_path.stem

    cam_id, cam_info = get_camera_info(video_id_str)
    scene_dict: dict = {
        "class": scene_class,
        "video_id": video_id_str,
        "camera_id": cam_id,
    }

    if cam_info is not None:
        scene_dict["width"] = cam_info.get("width")
        scene_dict["height"] = cam_info.get("height")
        scene_dict["static_objects"] = cam_info.get("objects", [])
    else:
        scene_dict["width"] = None
        scene_dict["height"] = None
        scene_dict["static_objects"] = []

    out = {
        "video_id": video_id_str,
        "fps": fps,
        "num_frames": num_frames,
        "scene": scene_dict,
        "frames": {
            str(int(f)): frames_dict[f]
            for f in sorted(frames_dict.keys())
        },
    }

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Converts annotations from CVAT's jobs to an intermediate JSON per frame/person"
        )
    )
    parser.add_argument("input", type=Path, help="JSON extracted from /api/jobs/{id}/annotations")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/intermediate/"),
        help="Output dir (data/intermediate/)",
    )
    parser.add_argument("--video-id", help="Video Id")
    parser.add_argument("--fps", type=float, help="FPS")

    args = parser.parse_args()

    out = cvat_to_intermediate(args.input, args.video_id, args.fps)

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{args.input.stem}.intermediate.json"
    output_path = output_dir / out_name

    output_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved in: {output_path}")
    print(
        f"scene.class={out['scene']['class']!r}, "
        f"num_frames={out['num_frames']}, "
        f"frames_con_personas={len(out['frames'])}, "
        f"camera_id={out['scene'].get('camera_id')!r}"
    )


if __name__ == "__main__":
    main()


