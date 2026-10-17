"""RDB into the Modal Volume (decisions 84 and 90): the stopping split and step 1's labels from the shared marks.

    MSYS_NO_PATHCONV=1 modal volume put ecg-delineation <dataset.zip> /rdb/dataset.zip   # once (1.1 GB)
    modal run scripts/rdb_prepare.py::split          # splits/rdb_pretrain_seed0.json: 10% of every rhythm class held out
    modal run scripts/rdb_prepare.py                 # /vol/rdb/records/<id>.npz for every record; results/rdb_labels.json
    modal run scripts/rdb_prepare.py --limit 20      # a smoke test

Five CPU containers read the zip on the Volume and write one file per record, PTB-XL's layout
(``scripts/ptbxl_fetch.py``): ``signal`` (12, 5000) float32 mV in LEADS order, ``classes`` (12, 5000)
int8 and ``mask`` (12, 5000) bool, the same for every lead because RDB's marks are shared, plus
``n_beats``, ``n_masked``, ``aligned``, ``fs``, ``af`` and ``rhythm``.
"""

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "splits" / "rdb_pretrain_seed0.json"
ZIP = f"{VOL}/rdb/dataset.zip"
OUT_DIR = f"{VOL}/rdb/records"
BATCH = 100
STOP_FRACTION = 0.1


@app.function(volumes={VOL: volume}, timeout=600)
def make_split(seed: int, fraction: float) -> dict[str, Any]:
    """The record ids from the zip and the stratified stopping split (runs remotely: the laptop's Modal client has no numpy)."""
    import zipfile

    from ecgdelin.rdb import rhythm_class, stop_split

    with zipfile.ZipFile(ZIP) as z:
        ids = sorted(Path(n).stem for n in z.namelist() if n.startswith("dat_csv/") and n.endswith(".csv"))
    train, stop = stop_split(ids, fraction, seed)
    classes = sorted({rhythm_class(i) for i in ids})
    return {"n_records": len(ids), "per_class": {c: {"train": sum(rhythm_class(i) == c for i in train), "stop": sum(rhythm_class(i) == c for i in stop)} for c in classes},
            "sample": {"train": train, "stop": stop}}


@app.function(volumes={VOL: volume}, timeout=3600, cpu=1.0, memory=2048, max_containers=5, retries=2)
def prepare_batch(ids: list[str], overwrite: bool) -> list[dict[str, Any]]:
    import zipfile

    import numpy as np

    from ecgdelin.labels import labels_for
    from ecgdelin.rdb import AF_CLASSES, FS, parse_marks, rdb_waves, read_signal, rhythm_class

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with zipfile.ZipFile(ZIP) as z:
        for rec_id in ids:
            path = out_dir / f"{rec_id}.npz"
            rhythm = rhythm_class(rec_id)
            af = rhythm in AF_CLASSES
            if path.exists() and not overwrite:
                with np.load(path) as f:
                    summaries.append(_summary(rec_id, rhythm, af, int(f["n_beats"][0]), int(f["n_masked"][0]), f["mask"][0], f["classes"][0], skipped=True))
                continue
            signal = read_signal(z.read(f"dat_csv/{rec_id}.csv").decode("utf-8-sig"))
            n_samples = signal.shape[1]
            waves = rdb_waves(parse_marks(z.read(f"ann_txt/{rec_id}.all.txt").decode("utf-8-sig")), n_samples)
            labels = labels_for(waves.waves, n_samples)
            classes = np.repeat(labels.classes[None], 12, axis=0)
            mask = np.repeat(labels.mask[None], 12, axis=0)
            np.savez_compressed(path, signal=signal, classes=classes, mask=mask, n_beats=np.full(12, waves.n_beats, dtype=np.int16),
                                n_masked=np.full(12, waves.n_masked_beats, dtype=np.int16), aligned=np.ones(12, dtype=bool),
                                fs=np.int16(FS), af=np.bool_(af), rhythm=np.str_(rhythm))
            summaries.append(_summary(rec_id, rhythm, af, waves.n_beats, waves.n_masked_beats, labels.mask, labels.classes, skipped=False))
    volume.commit()
    return summaries


def _summary(rec_id: str, rhythm: str, af: bool, n_beats: int, n_masked: int, mask: Any, classes: Any, skipped: bool) -> dict[str, Any]:
    return {"id": rec_id, "rhythm": rhythm, "af": af, "n_beats": int(n_beats), "n_masked": int(n_masked),
            "labelled_fraction": round(float(mask.mean()), 4), "p_samples": int(((classes == 1) & mask).sum()), "skipped": skipped}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def split(seed: int = 0) -> None:
    plan = make_split.remote(seed, STOP_FRACTION)
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps({"seed": seed, "stop_fraction": STOP_FRACTION, "decisions": [84, 90], **plan}, indent=1), encoding="utf-8", newline="\n")
    print(f"{plan['n_records']} records: {len(plan['sample']['train'])} train, {len(plan['sample']['stop'])} stop; wrote {SPLIT}")


@app.local_entrypoint()
def main(limit: int = 0, overwrite: bool = False) -> None:
    sample = json.loads(SPLIT.read_text(encoding="utf-8"))["sample"]
    role = {i: "train" for i in sample["train"]} | {i: "stop" for i in sample["stop"]}
    ids = sample["train"] + sample["stop"]
    if limit:
        ids = ids[:limit]
    batches = [ids[k : k + BATCH] for k in range(0, len(ids), BATCH)]
    summaries = [s for batch in prepare_batch.map(batches, kwargs={"overwrite": overwrite}) for s in batch]
    beats = sum(s["n_beats"] for s in summaries)
    masked = sum(s["n_masked"] for s in summaries)
    result = {
        "name": "rdb_labels",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": _git_head(),
        "decisions": [75, 84, 90],
        "n_records": len(summaries),
        "n_train": sum(role[s["id"]] == "train" for s in summaries),
        "n_stop": sum(role[s["id"]] == "stop" for s in summaries),
        "beats": beats,
        "masked_waves": masked,
        "masked_share": round(masked / max(1, sum(3 * s["n_beats"] for s in summaries)), 4),
        "records_without_marks": sum(s["n_beats"] == 0 for s in summaries),
        "mean_labelled_fraction": round(sum(s["labelled_fraction"] for s in summaries) / max(1, len(summaries)), 4),
        "p_samples_in_af": sum(s["p_samples"] for s in summaries if s["af"]),
        "per_rhythm": {r: sum(s["rhythm"] == r for s in summaries) for r in sorted({s["rhythm"] for s in summaries})},
        "records": summaries,
    }
    out = REPO / "results" / ("rdb_labels.json" if not limit else "rdb_labels_smoke.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(summaries)} records ({result['n_train']} train, {result['n_stop']} stop), {beats} beats, {masked} masked waves "
          f"({result['masked_share']:.1%} of waves), labelled fraction {result['mean_labelled_fraction']:.1%}, P samples in AF {result['p_samples_in_af']}")
    print(f"wrote {out}")
