#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or str(val).strip() == ""):
        raise SystemExit(f"Missing env var: {name}")
    return str(val)


CVAT_URL: str   = _env("CVAT_URL", required=True).rstrip("/")
API_BASE: str   = f"{CVAT_URL}/api"
CVAT_USER: str  = _env("CVAT_USER", required=True)
CVAT_PASS: str  = _env("CVAT_PASS", required=True)
PROJECT_ID: int = int(_env("CVAT_PROJECT_ID", required=True))

ROOT = Path(__file__).resolve().parent
DATA_ROOT        = ROOT / "data"
CVAT_EXPORTS_DIR = DATA_ROOT / "cvat_exports"
INTERMEDIATE_DIR = DATA_ROOT / "intermediate"
NPZ_DIR          = DATA_ROOT / "npz"

SESSION = requests.Session()
SESSION.auth = (CVAT_USER, CVAT_PASS)


def run_cmd(cmd: List[str]) -> None:
    """Wrapper to execute scripts"""
    print("$", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def iter_paginated(url: str, params: Dict[str, Any] | None = None) -> Iterator[Dict[str, Any]]:
    """Iterate through CVAT's endpoint."""
    while url:
        resp = SESSION.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "results" in data:
            for item in data["results"]:
                yield item
            url = data.get("next")
            params = None
        elif isinstance(data, list):
            for item in data:
                yield item
            break
        else:
            break


def list_tasks_for_project(project_id: int) -> List[Dict[str, Any]]:
    """Return all tasks in the CVAT project."""
    url = f"{API_BASE}/tasks"
    params = {"page_size": 100}
    tasks: List[Dict[str, Any]] = []
    for t in iter_paginated(url, params):
        if int(t.get("project_id", -1)) == project_id:
            tasks.append(t)
    return tasks


def list_jobs_for_task(task_id: int) -> List[Dict[str, Any]]:
    """
    List jobs from each task.
    """
    resp = SESSION.get(f"{API_BASE}/tasks/{task_id}/jobs", params={"page_size": 500})
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict) and "results" in data:
            return list(data["results"])
        if isinstance(data, list):
            return list(data)

    # Fallback
    resp = SESSION.get(f"{API_BASE}/jobs", params={"task_id": task_id, "page_size": 500})
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "results" in data:
        return list(data["results"])
    if isinstance(data, list):
        return list(data)
    return []


def find_intermediate_json(job_id: int) -> Optional[Path]:
    """
    Searches for a job's intermediate JSON.
    """
    p1 = INTERMEDIATE_DIR / f"job_{job_id}_annotations.intermediate.json"
    if p1.is_file():
        return p1

    return None


# Pipeline

def process_job(job_id: int, *, force: bool = False) -> None:
    """
    Runs:
      1) extract_job_annotations.py
      2) cvat_to_intermediate.py
      3) intermediate_to_npz.py
    """
    CVAT_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    cvat_json = CVAT_EXPORTS_DIR / f"job_{job_id}_annotations.json"

    npz_path = NPZ_DIR / f"job_{job_id}_annotations.intermediate.npz"

    if npz_path.exists() and not force:
        print(f"[ SKIP ] Job {job_id}: {npz_path} already exists.")
        return

    try:
        if not cvat_json.exists() or force:
            print(f"[ 1 ] Extracting annotations: {job_id} -> {cvat_json}")
            run_cmd([
                "python",
                "extract_job_annotations.py",
                "--job-id",
                str(job_id),
                "--output",
                str(CVAT_EXPORTS_DIR),
            ])
        else:
            print(f"[ 1 ] {cvat_json} already exists")

        if not cvat_json.exists():
            raise FileNotFoundError(f"[ ERROR ] at saving {cvat_json}")

        interm_json = find_intermediate_json(job_id)

        if not interm_json or force:
            print(f"[ 2 ] Converting to intermediate -> {INTERMEDIATE_DIR}")
            run_cmd([
                "python",
                "cvat_to_intermediate.py",
                str(cvat_json),
                "--output",
                str(INTERMEDIATE_DIR),
            ])
            interm_json = find_intermediate_json(job_id)

        if not interm_json or not interm_json.exists():
            raise FileNotFoundError(
                f"[ ERROR ] at saving intermediate JSON from job {job_id}"
            )

        print(f"[ 2 ] Intermediate: {interm_json}")

        print(f"[ 3 ] Converting to .npz -> {npz_path}")
        run_cmd([
            "python",
            "intermediate_to_npz.py",
            str(interm_json),
            "--output",
            str(NPZ_DIR),
        ])

        print(f"[ OK ] Job {job_id} processed as .npz: {npz_path}")

    except subprocess.CalledProcessError as e:
        print(f"[ ERROR ] Pipeline failed for {job_id}: {e}")
    except Exception as e:
        print(f"[ ERROR ] Exception in job {job_id}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs the (extract -> intermediate -> npz) pipeline for all completed jobs in the CVAT project"
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=PROJECT_ID,
        help="CVAT project's ID (default: CVAT_PROJECT_ID from .env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if .npz already exists",
    )
    args = parser.parse_args()

    print(f"CVAT_URL        = {CVAT_URL}")
    print(f"Project         = {args.project_id}")
    print(f"data root       = {DATA_ROOT}")
    print(f"cvat_exports    = {CVAT_EXPORTS_DIR}")
    print(f"intermediate    = {INTERMEDIATE_DIR}")
    print(f"npz output dir  = {NPZ_DIR}")

    tasks = list_tasks_for_project(args.project_id)
    if not tasks:
        print("No tasks found in project.")
        return

    print(f"Found {len(tasks)} tasks in project {args.project_id}.")

    jobs_to_process: List[int] = []
    for t in tasks:
        task_id = t["id"]
        task_name = t.get("name", f"task_{task_id}")
        jobs = list_jobs_for_task(task_id)

        for j in jobs:
            j_id = j["id"]
            state = (j.get("state") or j.get("status") or "").lower()
            stage = (j.get("stage") or "").lower()
            if state == "completed":
                jobs_to_process.append(j_id)

    if not jobs_to_process:
        print("No jobs with state='completed'.")
        return

    print(f"\nProcessing {len(jobs_to_process)} jobs\n")

    for jid in jobs_to_process:
        process_job(jid, force=args.force)


if __name__ == "__main__":
    main()

