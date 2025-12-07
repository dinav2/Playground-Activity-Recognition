#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.playground_data.labels import (
    ACTION_CLASSES,
    ROLE_CLASSES,
    SAFETY_FLAG_CLASSES,
)

# Value used in DataLoader
IGNORE_INDEX = -100

DEFAULT_OUTPUT_DIR = Path("data/npz")
MAX_STATIC_OBJECTS = 4


def intermediate_to_npz(input_path: Path, output_dir: Path | None = None) -> Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))

    frames = data.get("frames", {})

    if "num_frames" in data and isinstance(data["num_frames"], int):
        T = data["num_frames"]
    else:
        T = max(int(k) for k in frames.keys()) + 1

    N_max = max(len(frames[f].get("persons", [])) for f in frames)

    J_skel = 0
    for f in frames.values():
        for p in f.get("persons", []):
            kps = p.get("keypoints", [])
            if isinstance(kps, list):
                J_skel = max(J_skel, len(kps))

    J_total = J_skel + MAX_STATIC_OBJECTS

    action2id = {a: i for i, a in enumerate(ACTION_CLASSES)}
    role2id = {r: i for i, r in enumerate(ROLE_CLASSES)}
    safety2id = {s: i for i, s in enumerate(SAFETY_FLAG_CLASSES)}

    scene_class = data.get("scene", {}).get("class")
    scene_vocab = []
    scene_id = -1
    if scene_class is not None:
        scene_vocab = [scene_class]
        scene_id = 0

    keypoints = np.full((T, N_max, J_total, 2), np.nan, dtype=np.float32)
    roles = np.full((T, N_max), IGNORE_INDEX, dtype=np.int16)
    actions = np.full((T, N_max), IGNORE_INDEX, dtype=np.int16)
    safety_flags = np.full((T, N_max), IGNORE_INDEX, dtype=np.int16)
    track_ids = np.full((T, N_max), -1, dtype=np.int32)
    mask = np.zeros((T, N_max), dtype=np.uint8)

    for f_str, frame in frames.items():
        t = int(f_str)
        persons = frame.get("persons", [])
        objects = frame.get("objects", [])

        obj_xy = [(np.nan, np.nan)] * MAX_STATIC_OBJECTS
        if isinstance(objects, list):
            for k, obj in enumerate(objects):
                if k >= MAX_STATIC_OBJECTS:
                    break
                cx = obj.get("cx", np.nan)
                cy = obj.get("cy", np.nan)
                obj_xy[k] = (float(cx), float(cy))

        persons_sorted = sorted(
            persons,
            key=lambda p: (p.get("track_id") is None, p.get("track_id")),
        )

        for idx, p in enumerate(persons_sorted):
            if idx >= N_max:
                break

            mask[t, idx] = 1

            tid = p.get("track_id", -1)
            track_ids[t, idx] = -1 if tid is None else int(tid)

            r = p.get("role")
            a = p.get("action")
            s = p.get("safety_flag")

            if isinstance(r, str) and r in role2id:
                roles[t, idx] = role2id[r]
            if isinstance(a, str) and a in action2id:
                actions[t, idx] = action2id[a]
            if isinstance(s, str) and s in safety2id:
                safety_flags[t, idx] = safety2id[s]

            kps = p.get("keypoints", [])
            for j_idx in range(min(J_skel, len(kps))):
                pt = kps[j_idx]
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    keypoints[t, idx, j_idx, 0] = float(pt[0])
                    keypoints[t, idx, j_idx, 1] = float(pt[1])

            for k in range(MAX_STATIC_OBJECTS):
                cx, cy = obj_xy[k]
                keypoints[t, idx, J_skel + k, 0] = cx
                keypoints[t, idx, J_skel + k, 1] = cy

    video_id = data.get("video_id", input_path.stem)
    fps = data.get("fps", None)

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{input_path.stem}.npz"

    # 7) Guardar .npz
    np.savez(
        output_path,
        keypoints=keypoints,
        roles=roles,
        actions=actions,
        safety_flags=safety_flags,
        track_ids=track_ids,
        mask=mask,
        scene_label=np.array(scene_id, dtype=np.int16),
        roles_vocab=np.array(ROLE_CLASSES, dtype=object),
        actions_vocab=np.array(ACTION_CLASSES, dtype=object),
        safety_vocab=np.array(SAFETY_FLAG_CLASSES, dtype=object),
        scene_vocab=np.array(scene_vocab, dtype=object),
        video_id=np.array(video_id, dtype=object),
        fps=np.array(fps if fps is not None else -1.0, dtype=np.float32),
        J_skel=np.array(J_skel, dtype=np.int16),
        J_total=np.array(J_total, dtype=np.int16),
        max_static_objects=np.array(MAX_STATIC_OBJECTS, dtype=np.int16),
    )

    print(f"Saved .npz in: {output_path}")
    print(f"T={T}, N_max={N_max}, J_skel={J_skel}, J_total={J_total}")
    print(f"scene_class={scene_class!r}")
    n_actions = np.sum(actions != IGNORE_INDEX)
    n_roles = np.sum(roles != IGNORE_INDEX)
    n_safety = np.sum(safety_flags != IGNORE_INDEX)
    print(f"#valid actions: {n_actions}, #valid roles: {n_roles}, #valid risks: {n_safety}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Converts intermediate JSON to tensors and saves as .npz."
    )
    parser.add_argument("input", type=Path, help="Intermediate JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output dir (data/npz/)",
    )
    args = parser.parse_args()

    intermediate_to_npz(args.input, args.output)


if __name__ == "__main__":
    main()

