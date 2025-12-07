#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

def run(cmd: list[str]) -> int:
    p = subprocess.run(cmd)
    return p.returncode

def main():
    ap = argparse.ArgumentParser(
        description="Batch runner para el pipeline sobre un directorio de videos"
    )
    ap.add_argument("--input-dir", required=True, help="Directorio raíz con videos")
    ap.add_argument("--pattern", default="**/*.mp4", help="Patrón de búsqueda (glob)")
    ap.add_argument("--rel-prefix", default="dataset", help="Prefijo para rel-dir en CVAT share")
    ap.add_argument("--model", default="yolo11l-pose.pt")
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Saltar si tracks/<clip>.json ya existe")
    ap.add_argument("--script", default="script.py",
                    help="Ruta al script principal (el tuyo)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    videos = sorted(input_dir.glob(args.pattern))
    if not videos:
        print(f"No se encontraron videos con patrón {args.pattern} en {input_dir}", file=sys.stderr)
        sys.exit(1)

    tracks_dir = Path("tracks")
    tracks_dir.mkdir(exist_ok=True)

    total, ok, fail, skipped = len(videos), 0, 0, 0

    for vid in videos:
        stem = vid.stem
        rel_dir = f"{args.rel_prefix}/{stem}"

        if args.skip_existing and (tracks_dir / f"{stem}.json").exists():
            print(f"[SKIP] {vid} (tracks/{stem}.json ya existe)")
            skipped += 1
            continue

        cmd = [
            sys.executable, args.script,
            "--video", str(vid),
            "--rel-dir", rel_dir,
            "--model", args.model,
            "--imgsz", str(args.imgsz),
            "--conf", str(args.conf),
            "--device", args.device,
        ]
        if args.preview:
            cmd.append("--preview")

        print(f"[RUN ] {vid}  ->  rel-dir={rel_dir}")
        rc = run(cmd)
        if rc == 0:
            print(f"[ OK ] {vid}")
            ok += 1
        else:
            print(f"[FAIL] {vid} (rc={rc})")
            fail += 1

    print(f"\nResumen: total={total} ok={ok} fail={fail} skipped={skipped}")

if __name__ == "__main__":
    main()
