"""ECGdeli 1.1 on LUDB, scored with step 1's code (decision 78). Runs on the laptop: MATLAB is not on Modal.

    python scripts/ecgdeli_ludb.py export      # LUDB -> work/ecgdeli/ludb_input.mat
    matlab -batch "ecgdeli_root='<ECGdeli checkout>'; work='<this repo>/work/ecgdeli'; run('<this repo>/scripts/ecgdeli_ludb.m')"
    python scripts/ecgdeli_ludb.py score       # work/ecgdeli/ludb_fpt.mat -> results/ecgdeli_ludb.json

The MATLAB step filters and delineates exactly as ECGdeli's own example (Example/Annotate_ExampleECG.m
at tag v1.1): 1-40 Hz band-pass, 50 Hz notch, isoline correction, ``Annotate_ECG_Multi``. Writes
results/ecgdeli_ludb.json (metrics for all leads, lead I and each lead) and
results/predictions/ecgdeli_ludb.json (every point for every lead and record), the same layout as
scripts/run_neurokit.py.
"""

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from ecgdelin.dataset import LEADS, load_record
from ecgdelin.delineate import LeadPoints
from ecgdelin.ecgdeli import failed_lead, lead_points_from_fpt
from ecgdelin.paths import ludb_root
from ecgdelin.score import score

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work" / "ecgdeli"
SPLITS = REPO / "splits" / "ludb_5fold_seed0.json"
NAME = "ecgdeli_ludb"


def _ids() -> list[str]:
    folds = json.loads(SPLITS.read_text(encoding="utf-8"))
    return sorted({p for fold in folds["folds"] for p in fold["test"]}, key=int)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def export() -> None:
    """Every LUDB record as (records, samples, 12) mV in LEADS order, the layout ECGdeli reads."""
    root = ludb_root()
    ids = _ids()
    recs = [load_record(root, rid) for rid in ids]
    fs = {r.fs for r in recs}
    n = {r.signal.shape[1] for r in recs}
    if len(fs) != 1 or len(n) != 1:
        raise ValueError(f"LUDB records differ in rate or length: {fs}, {n}")
    signals = np.stack([r.signal.T for r in recs])  # (records, samples, 12)
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "ludb_input.mat"
    scipy.io.savemat(out, {"signals": signals, "ids": np.array(ids, dtype=object), "fs": float(fs.pop()), "leads": np.array(LEADS, dtype=object)})
    print(f"wrote {out}: {signals.shape}, {signals.dtype}")


def _predictions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mat = scipy.io.loadmat(WORK / "ludb_fpt.mat", squeeze_me=False)
    fpt, failed, msg = mat["fpt"], mat["failed"].ravel().astype(bool), mat["msg"].ravel()
    ids = _ids()
    if fpt.shape != (len(ids), len(LEADS)):
        raise ValueError(f"expected an FPT cell array of {(len(ids), len(LEADS))}, got {fpt.shape}")
    n_samples = int(mat["n_samples"].ravel()[0])
    predictions = []
    for k, rid in enumerate(ids):
        if failed[k]:
            leads = {lead: failed_lead() for lead in LEADS}
        else:
            leads = {lead: lead_points_from_fpt(fpt[k, j], n_samples) for j, lead in enumerate(LEADS)}
        predictions.append({"id": rid, "leads": {lead: lp.to_dict() for lead, lp in leads.items()}})
    errors = {ids[k]: str(np.asarray(msg[k]).ravel()[0]) for k in np.flatnonzero(failed)}
    return predictions, {"failed_records": errors, "matlab": str(mat.get("matlab_version", np.array([""])).ravel()[0])}


def score_all() -> None:
    predictions, extra = _predictions()
    root = ludb_root()
    recordings = [load_record(root, p["id"]) for p in predictions]
    preds = {p["id"]: {lead: LeadPoints.from_dict(d) for lead, d in p["leads"].items()} for p in predictions}

    def one_lead(lead: str) -> tuple[list[Any], dict[str, Any]]:
        recs = [dataclasses.replace(r, marks={k: (v if k == lead else []) for k, v in r.marks.items()}) for r in recordings]
        return recs, {rid: {lead: lp[lead]} for rid, lp in preds.items()}

    report = score(recordings, preds)
    per_lead = {lead: score(*one_lead(lead), n_draws=200).to_dict() for lead in LEADS}
    for d in per_lead.values():
        d.pop("by_rhythm", None)
    meta = {
        "ecgdeli": "1.1 (KIT-IBT/ECGdeli tag v1.1, 3c13b1b)",
        "pipeline": "ECG_High_Low_Filter 1-40 Hz, Notch_Filter 50 Hz, Isoline_Correction, Annotate_ECG_Multi",
        "scored": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": _git_head(),
        "n_people": len(recordings),
        **extra,
    }
    result = {"name": NAME, "meta": meta, "all_leads": report.to_dict(), "lead_i": score(*one_lead("i")).to_dict(), "per_lead": per_lead}
    (REPO / "results" / "predictions").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / f"{NAME}.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    (REPO / "results" / "predictions" / f"{NAME}.json").write_text(json.dumps(predictions), encoding="utf-8", newline="\n")

    a = result["all_leads"]
    print(f"\n{NAME}: {a['n_people']} people, failed leads {a['failed_leads']}, failed records {len(extra['failed_records'])}, orphans {a['orphans']}")
    print(f"primary SD (5 CSE points, all leads): {a['primary_sd_ms']:.2f} ms, 95% CI {a['primary_ci95']}")
    print(f"{'point':9s} {'F1%':>7s} {'Se%':>7s} {'PPV%':>7s} {'mean':>7s} {'SD':>7s} {'SDwithin':>9s}")
    for point, s in a["points"].items():
        at = s["at"]["150"]
        print(f"{point:9s} {100 * at['f1']:7.2f} {100 * at['se']:7.2f} {100 * at['ppv']:7.2f} {s['mean_ms']:7.1f} {s['sd_ms']:7.1f} {s['sd_within_ms']:9.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["export", "score"])
    args = parser.parse_args()
    {"export": export, "score": score_all}[args.step]()


if __name__ == "__main__":
    sys.exit(main())
