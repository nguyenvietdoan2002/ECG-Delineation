"""M2: extract frozen TimesFM 3 states for every LUDB recording into the Modal Volume.

    modal run scripts/extract_states.py                    # latest checkpoint revision
    modal run scripts/extract_states.py --revision <sha>   # a specific revision
    modal run scripts/extract_states.py --overwrite        # recompute existing files

One T4 container loads the model once and writes /vol/states/<revision>/<record>.npy, an fp16 array
of shape (20 layers, 12 leads, 156 blocks, 2560) per recording (about 183 MiB), plus manifest.json.
The manifest is also saved to results/states_manifest.json.
"""

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]


@app.function(gpu="T4", volumes={VOL: volume}, timeout=4 * 3600, memory=16384)
def extract_all(revision: str | None, overwrite: bool, meta: dict[str, Any]) -> dict[str, Any]:
    import time

    import numpy as np

    from ecgdelin.dataset import load_record, record_ids
    from ecgdelin.paths import ludb_root
    from ecgdelin.preprocess import CUTOFF_HZ, ORDER, N_SAMPLES, prepare
    from ecgdelin.states import N_BLOCKS, TimesFMBackbone, extract_states

    backbone = TimesFMBackbone(revision)
    out_dir = Path(VOL) / "states" / backbone.revision
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ludb_root()
    ids = record_ids(root)
    written, skipped, max_abs, seconds = 0, 0, 0.0, 0.0
    for k, rec_id in enumerate(ids, 1):
        path = out_dir / f"{rec_id}.npy"
        if path.exists() and not overwrite:
            skipped += 1
            continue
        rec = load_record(root, rec_id)
        start = time.perf_counter()
        states = extract_states(prepare(rec.signal, rec.fs), backbone)
        seconds += time.perf_counter() - start
        if not np.all(np.isfinite(states)):
            raise RuntimeError(f"non-finite states for record {rec_id}")
        max_abs = max(max_abs, float(np.max(np.abs(states.astype(np.float32)))))
        np.save(path, states)
        written += 1
        if written % 25 == 0:
            volume.commit()
            print(f"{k}/{len(ids)} records, {seconds / written:.2f} s of model time each, max |state| {max_abs:.0f}")
    manifest = {
        **meta,
        "hf_repo": "google/timesfm-3.0-pytorch",
        "revision": backbone.revision,
        "device": backbone.device,
        "n_layers": backbone.n_layers,
        "d_model": backbone.d_model,
        "shape_per_record": [backbone.n_layers, 12, N_BLOCKS, 2 * backbone.d_model],
        "dtype": "float16",
        "layout": "[layer, lead (LEADS order), block, forward states | time-reversed states]",
        "preprocessing": {"highpass_hz": CUTOFF_HZ, "butterworth_order": ORDER, "zero_phase": True, "samples": N_SAMPLES},
        "patch_is_target": True,
        "records": len(ids),
        "written": written,
        "skipped_existing": skipped,
        "model_seconds_per_record": seconds / written if written else None,
        "max_abs_state": max_abs if written else None,
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "path": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    volume.commit()
    return manifest


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(revision: str = "", overwrite: bool = False) -> None:
    meta = {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_head()}
    manifest = extract_all.remote(revision or None, overwrite, meta)
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "states_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=1))
