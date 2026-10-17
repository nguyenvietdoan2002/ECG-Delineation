"""RDB states (decision 90): frozen TimesFM 3 states for the 2,399 RDB records, as scripts/extract_states_ptbxl.py.

    modal deploy scripts/extract_states_rdb.py               # once, after any change to this file
    python scripts/extract_states_rdb.py launch              # spawn 5 chunks on the deployed app and exit
    python scripts/extract_states_rdb.py status              # which chunks have finished
    modal run scripts/extract_states_rdb.py::collect         # merge the chunk manifests into results/

Each record's signal from /vol/rdb/records/<id>.npz becomes /vol/states/<revision>/rdb/<id>.npy, fp16
(20, 12, 156, 2560), the same revision, preprocessing and layout as step 1. The files are packed to 4
leads per record by scripts/pretrain_ptbxl.py and then deleted (decision 82's rule).
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "splits" / "rdb_pretrain_seed0.json"
STEP1_MANIFEST = REPO / "results" / "states_manifest.json"
CALLS = REPO / "results" / "rdb_states_calls.json"
N_CHUNKS = 5
FUNCTION = "extract_rdb"


@app.function(gpu="T4", volumes={VOL: volume}, timeout=6 * 3600, memory=16384, retries=1)
def extract_rdb(chunk: int, ids: list[str], revision: str, meta: dict[str, Any]) -> dict[str, Any]:
    import time

    import numpy as np

    from ecgdelin.preprocess import prepare
    from ecgdelin.states import TimesFMBackbone, extract_states

    backbone = TimesFMBackbone(revision)
    if backbone.revision != revision:
        raise RuntimeError(f"asked for revision {revision}, got {backbone.revision}")
    out_dir = Path(VOL) / "states" / revision / "rdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped, max_abs, seconds = 0, 0, 0.0, 0.0
    for k, rec_id in enumerate(ids, 1):
        path = out_dir / f"{rec_id}.npy"
        if path.exists():
            skipped += 1
            continue
        with np.load(Path(VOL) / "rdb" / "records" / f"{rec_id}.npz") as z:
            signal, fs = z["signal"].astype(np.float64), int(z["fs"])
        start = time.perf_counter()
        states = extract_states(prepare(signal, fs), backbone)
        seconds += time.perf_counter() - start
        if not np.all(np.isfinite(states)):
            raise RuntimeError(f"non-finite states for RDB {rec_id}")
        max_abs = max(max_abs, float(np.max(np.abs(states.astype(np.float32)))))
        np.save(path, states)
        written += 1
        if written % 25 == 0:
            volume.commit()
            print(f"chunk {chunk}: {k}/{len(ids)}, {seconds / written:.2f} s of model time each", flush=True)
    manifest = {
        **meta,
        "chunk": chunk,
        "revision": backbone.revision,
        "records": len(ids),
        "written": written,
        "skipped_existing": skipped,
        "model_seconds_per_record": seconds / written if written else None,
        "max_abs_state": max_abs if written else None,
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (out_dir / f"manifest_chunk{chunk}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    volume.commit()
    return manifest


@app.function(volumes={VOL: volume}, timeout=600)
def read_manifests(revision: str) -> dict[str, Any]:
    out_dir = Path(VOL) / "states" / revision / "rdb"
    chunks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("manifest_chunk*.json"))]
    return {"chunks": chunks, "n_files": sum(1 for _ in out_dir.glob("*.npy"))}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _plan() -> tuple[str, list[list[str]]]:
    revision = json.loads(STEP1_MANIFEST.read_text(encoding="utf-8"))["revision"]
    sample = json.loads(SPLIT.read_text(encoding="utf-8"))["sample"]
    ids = sample["train"] + sample["stop"]
    return revision, [ids[k::N_CHUNKS] for k in range(N_CHUNKS)]


@app.local_entrypoint()
def collect() -> None:
    revision, chunks = _plan()
    result = read_manifests.remote(revision)
    summary = {
        "name": "rdb_states",
        "revision": revision,
        "n_ecgs": sum(len(c) for c in chunks),
        "n_files": result["n_files"],
        "chunks": result["chunks"],
        "path": f"{VOL}/states/{revision}/rdb",
        "shape_per_record": [20, 12, 156, 2560],
        "dtype": "float16",
    }
    (REPO / "results" / "rdb_states_manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(f"{summary['n_files']} of {summary['n_ecgs']} files, {len(result['chunks'])} of {N_CHUNKS} chunks finished")


def launch() -> None:
    import modal

    revision, chunks = _plan()
    fn = modal.Function.from_name(app.name, FUNCTION)
    meta = {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_head()}
    calls = {}
    for k, ids in enumerate(chunks):
        call = fn.spawn(k, ids, revision, meta)
        calls[f"chunk{k}"] = call.object_id
        print("spawned chunk", k, len(ids), "records", call.object_id, flush=True)
    CALLS.write_text(json.dumps({"revision": revision, "meta": meta, "calls": calls}, indent=1), encoding="utf-8", newline="\n")


def status() -> None:
    import modal

    calls = json.loads(CALLS.read_text(encoding="utf-8"))["calls"]
    for name, call_id in calls.items():
        call = modal.FunctionCall.from_id(call_id)
        try:
            m = call.get(timeout=0)
            print(name, "done:", m["written"], "written,", m["skipped_existing"], "skipped")
        except TimeoutError:
            print(name, "running")


if __name__ == "__main__":
    {"launch": launch, "status": status}[sys.argv[1]]()
