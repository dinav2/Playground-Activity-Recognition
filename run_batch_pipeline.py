#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    p = subprocess.run(cmd)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(
        description="Batch runner pipeline for pose-extraction"
    )
    ap.add_argument("--input-dir", required=True, help="Directory")
    ap.add_argument("--pattern", default="**/*.mp4", help="Search pattern")
    ap.add_argument(
        "--rel-prefix", default="dataset", help="Prefix for shared folder in CVAT"
    )
    ap.add_argument("--model", default="yolo11l-pose.pt")
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if tracks/<clip>.json already exists",
    )
    ap.add_argument("--script", default="script.py", help="Main pose-extraction script")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    videos = sorted(input_dir.glob(args.pattern))
    if not videos:
        print(
            f"No videos found with pattern {args.pattern}in {input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    tracks_dir = Path("tracks")
    tracks_dir.mkdir(exist_ok=True)

    total, ok, fail, skipped = len(videos), 0, 0, 0

    for vid in videos:
        stem = vid.stem
        rel_dir = f"{args.rel_prefix}/{stem}"

        if args.skip_existing and (tracks_dir / f"{stem}.json").exists():
            print(f"[ SKIP ] {vid} (tracks/{stem}.json already exists)")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            args.script,
            "--video",
            str(vid),
            "--rel-dir",
            rel_dir,
            "--model",
            args.model,
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--device",
            args.device,
        ]

        print(f"[ RUN ] {vid}  ->  rel-dir={rel_dir}")
        rc = run(cmd)
        if rc == 0:
            print(f"[ OK ] {vid}")
            ok += 1
        else:
            print(f"[ FAIL ] {vid} (rc={rc})")
            fail += 1

    print(f"\ntotal={total} ok={ok} fail={fail} skipped={skipped}")


if __name__ == "__main__":
    main()
