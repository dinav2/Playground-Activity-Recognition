#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or not str(val).strip()):
        raise SystemExit(f"Missing env: {name}")
    return str(val)

CVAT_URL  = _env("CVAT_URL", required=True).rstrip("/")
CVAT_USER = _env("CVAT_USER")
CVAT_PASS = _env("CVAT_PASS")

def make_session():

    session = requests.Session()

    resp = session.post(
        f"{CVAT_URL}/api/auth/login",
        json={"username": CVAT_USER, "password": CVAT_PASS},
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"Login failed ({resp.status_code}): {resp.text[:300]}"
        )

    # cvat format
    session.headers["Accept"] = "application/vnd.cvat+json"
    return session


def main():
    parser = argparse.ArgumentParser(
        description="Extract annotations from a CVAT job."
    )
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default="./data/cvat_exports/",
        help="<output>/job_<job-id>_annotations.json)",
    )

    args = parser.parse_args()

    session = make_session()
    url = f"{CVAT_URL}/api/jobs/{args.job_id}/annotations"

    print(f"GET {url}")
    resp = session.get(url, timeout=60)
    print("Status:", resp.status_code)

    if not resp.ok:
        print(resp.text[:1000])
        raise SystemExit(1)

    data = resp.json()
    out_path = args.output / f"job_{args.job_id}_annotations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved in: {out_path}")
    print(
        f"tags={len(data.get('tags', []))}, "
        f"shapes={len(data.get('shapes', []))}, "
        f"tracks={len(data.get('tracks', []))}"
    )


if __name__ == "__main__":
    main()
