"""Step 2 data (decisions 73, 75 and 76): fetch the chosen PTB-XL ECGs and their ECGDeli marks into the
Modal Volume and build their training labels.

    modal run scripts/ptbxl_fetch.py                # every ECG in splits/ptbxl_pretrain_seed0.json
    modal run scripts/ptbxl_fetch.py --limit 20     # a smoke test on the first 20
    modal run scripts/ptbxl_fetch.py --overwrite    # rebuild files that exist

Modal CPU containers download each ECG's 500 Hz signal and its 12 ECGDeli lead files from PhysioNet's S3
mirror, turn the marks into step 1's labels (``ptbxl.ecgdeli_waves`` then ``labels.labels_for``) and
write /vol/ptbxl/records/<ecg_id>.npz: ``signal`` (12, 5000) float32 in mV in LEADS order, ``classes``
(12, 5000) int8, ``mask`` (12, 5000) bool, and per lead ``n_beats``, ``n_masked`` and ``aligned``. One
file per ECG keeps the Volume's file count low. A label summary is written to
results/ptbxl_pretrain_labels.json. At most five containers write at once (the v1 Volume's guidance).
"""

import ast
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "splits" / "ptbxl_pretrain_seed0.json"
DATABASE = REPO / "work" / "ptbxl" / "ptbxl_database.csv"
MIRROR = "https://physionet-open.s3.amazonaws.com"
OUT_DIR = f"{VOL}/ptbxl/records"
BATCH = 100


def _download(url: str, dest: str, missing_ok: bool = False) -> bool:
    """Download with retries: the S3 mirror occasionally resets a connection. Returns False when the file
    does not exist and ``missing_ok``."""
    import time
    import urllib.error
    import urllib.request

    for attempt in range(6):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except urllib.error.HTTPError as err:
            if missing_ok and err.code in (403, 404):
                return False
            if err.code < 500 or attempt == 5:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            if attempt == 5:
                raise
        time.sleep(2**attempt)
    return False


@app.function(volumes={VOL: volume}, timeout=3600, cpu=1.0, memory=2048, max_containers=5, retries=2)
def fetch_batch(items: list[tuple[int, bool]], overwrite: bool) -> list[dict[str, Any]]:
    import tempfile

    import numpy as np
    import wfdb

    from ecgdelin.dataset import LEADS
    from ecgdelin.labels import labels_for
    from ecgdelin.ptbxl import PTB_LEADS, ecgdeli_waves, read_ecgdeli_lead

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ecg_id, af in items:
        path = out_dir / f"{ecg_id}.npz"
        if path.exists() and not overwrite:
            with np.load(path) as z:
                summaries.append(_summary(ecg_id, af, z["n_beats"], z["n_masked"], z["aligned"], z["mask"], z["classes"], skipped=True))
            continue
        folder = f"{ecg_id // 1000 * 1000:05d}"
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / f"{ecg_id:05d}_hr"
            for ext in ("hea", "dat"):
                _download(f"{MIRROR}/ptb-xl/1.0.3/records500/{folder}/{ecg_id:05d}_hr.{ext}", f"{stem}.{ext}")
            record = wfdb.rdrecord(str(stem))
            names = [n.lower() for n in record.sig_name]
            signal = record.p_signal.T[[names.index(lead) for lead in LEADS]].astype(np.float32)
            n_samples = signal.shape[1]
            classes = np.zeros((12, n_samples), dtype=np.int8)
            mask = np.zeros((12, n_samples), dtype=bool)
            n_beats = np.zeros(12, dtype=np.int16)
            n_masked = np.zeros(12, dtype=np.int16)
            aligned = np.ones(12, dtype=bool)
            for j, lead in enumerate(PTB_LEADS):
                lead_stem = Path(tmp) / f"{ecg_id:05d}_points_lead_{lead}"
                url = f"{MIRROR}/ptb-xl-plus/1.0.1/fiducial_points/ecgdeli/{folder}/{ecg_id:05d}_points_lead_{lead}.atr"
                _download(url, f"{lead_stem}.atr", missing_ok=True)
                waves = ecgdeli_waves(read_ecgdeli_lead(lead_stem), n_samples, af)
                labels = labels_for(waves.waves, n_samples)
                classes[j], mask[j] = labels.classes, labels.mask
                n_beats[j], n_masked[j], aligned[j] = waves.n_beats, waves.n_masked_beats, waves.aligned
        np.savez_compressed(path, signal=signal, classes=classes, mask=mask, n_beats=n_beats, n_masked=n_masked,
                            aligned=aligned, fs=np.int16(record.fs), af=np.bool_(af))
        summaries.append(_summary(ecg_id, af, n_beats, n_masked, aligned, mask, classes, skipped=False))
    volume.commit()
    return summaries


def _summary(ecg_id: int, af: bool, n_beats: Any, n_masked: Any, aligned: Any, mask: Any, classes: Any, skipped: bool) -> dict[str, Any]:
    return {
        "id": int(ecg_id),
        "af": bool(af),
        "n_beats": [int(v) for v in n_beats],
        "n_masked": [int(v) for v in n_masked],
        "aligned": [bool(v) for v in aligned],
        "labelled_fraction": [round(float(m.mean()), 4) for m in mask],
        "p_samples": int(((classes == 1) & mask).sum()),
        "skipped": skipped,
    }


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(limit: int = 0, overwrite: bool = False) -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = [("train", i) for i in split["sample"]["train"]] + [("stop", i) for i in split["sample"]["stop"]]
    if limit:
        ids = ids[:limit]
    with DATABASE.open(encoding="utf-8-sig", newline="") as f:
        codes = {int(float(r["ecg_id"])): ast.literal_eval(r["scp_codes"]) for r in csv.DictReader(f)}
    af = {i: ("AFIB" in codes[i] or "AFLT" in codes[i]) for _, i in ids}
    items = [(i, af[i]) for _, i in ids]
    batches = [items[k : k + BATCH] for k in range(0, len(items), BATCH)]
    summaries = [s for batch in fetch_batch.map(batches, kwargs={"overwrite": overwrite}) for s in batch]

    role = dict((i, r) for r, i in ids)
    beats = sum(sum(s["n_beats"]) for s in summaries)
    masked = sum(sum(s["n_masked"]) for s in summaries)
    unaligned = sum(not a for s in summaries for a in s["aligned"])
    no_marks = sum(n == 0 for s in summaries for n in s["n_beats"])
    p_in_af = sum(s["p_samples"] for s in summaries if s["af"])
    result = {
        "name": "ptbxl_pretrain_labels",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": _git_head(),
        "decisions": [75, 76],
        "n_records": len(summaries),
        "n_train": sum(role[s["id"]] == "train" for s in summaries),
        "n_stop": sum(role[s["id"]] == "stop" for s in summaries),
        "lead_beats": beats,
        "masked_lead_beats": masked,
        "masked_share": round(masked / beats, 4) if beats else None,
        "masked_share_af": round(sum(sum(s["n_masked"]) for s in summaries if s["af"]) / max(1, sum(sum(s["n_beats"]) for s in summaries if s["af"])), 4),
        "unaligned_leads": unaligned,
        "leads_without_marks": no_marks,
        "mean_labelled_fraction": round(sum(sum(s["labelled_fraction"]) for s in summaries) / (12 * len(summaries)), 4),
        "p_samples_in_af": p_in_af,
        "records": summaries,
    }
    out = REPO / "results" / ("ptbxl_pretrain_labels.json" if not limit else "ptbxl_pretrain_labels_smoke.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(summaries)} records ({result['n_train']} train, {result['n_stop']} stop), {beats} lead-beats, "
          f"{result['masked_share']:.1%} masked ({result['masked_share_af']:.1%} in AF/flutter), {unaligned} unaligned leads, "
          f"{no_marks} leads without marks, labelled fraction {result['mean_labelled_fraction']:.1%}, P samples in AF {p_in_af}")
    print(f"wrote {out}")
