"""Run the NeuroKit2 `prominence` baseline over all of LUDB on Modal CPU, score it, save the results.

    modal run scripts/run_neurokit.py                        # the baseline: default R-peak detector
    modal run scripts/run_neurokit.py --rpeaks emrich2023    # paper-faithful variant (FastNVG R-peaks)

Writes results/<name>.json (metrics for all leads, lead I alone, each lead, and the Emrich 2024
comparison) and results/predictions/<name>.json (every point for every lead and record). The same
files are kept in the Modal Volume under /vol/results and /vol/predictions.
"""

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "splits" / "ludb_5fold_seed0.json"


@app.function(volumes={VOL: volume}, timeout=900, cpu=1.0)
def delineate_record(record_id: str, rpeak_method: str) -> dict[str, Any]:
    from ecgdelin.dataset import load_record
    from ecgdelin.neurokit import NeuroKitProminence
    from ecgdelin.paths import ludb_root

    rec = load_record(ludb_root(), record_id)
    out = NeuroKitProminence(rpeak_method).delineate(rec.signal, rec.fs)
    return {"id": record_id, "leads": {lead: points.to_dict() for lead, points in out.items()}}


@app.function(volumes={VOL: volume}, timeout=1800, memory=4096)
def score_predictions(name: str, predictions: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    import dataclasses

    import neurokit2

    from ecgdelin.dataset import LEADS, load_record
    from ecgdelin.delineate import LeadPoints
    from ecgdelin.emrich2024 import compare_to_published
    from ecgdelin.paths import ludb_root
    from ecgdelin.score import score

    root = ludb_root()
    recordings = [load_record(root, p["id"]) for p in predictions]
    preds = {p["id"]: {lead: LeadPoints.from_dict(d) for lead, d in p["leads"].items()} for p in predictions}

    def one_lead(lead: str) -> tuple[list[Any], dict[str, Any]]:
        recs = [dataclasses.replace(r, marks={k: (v if k == lead else []) for k, v in r.marks.items()}) for r in recordings]
        return recs, {rid: {lead: lp[lead]} for rid, lp in preds.items()}

    report = score(recordings, preds)
    lead_i = score(*one_lead("i"))
    per_lead = {lead: score(*one_lead(lead), n_draws=200).to_dict() for lead in LEADS}
    for d in per_lead.values():
        d.pop("by_rhythm", None)
    result = {
        "name": name,
        "meta": {**meta, "neurokit2": neurokit2.__version__, "n_people": len(recordings)},
        "all_leads": report.to_dict(),
        "lead_i": lead_i.to_dict(),
        "per_lead": per_lead,
        "emrich2024_single_lead_i": compare_to_published(lead_i),
    }
    results_dir = Path(VOL) / "results"
    preds_dir = Path(VOL) / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (preds_dir / f"{name}.json").write_text(json.dumps(predictions), encoding="utf-8")
    volume.commit()
    return result


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(rpeaks: str = "neurokit") -> None:
    name = "neurokit_prominence" if rpeaks == "neurokit" else f"neurokit_prominence_{rpeaks}"
    folds = json.loads(SPLITS.read_text(encoding="utf-8"))
    ids = sorted({p for fold in folds["folds"] for p in fold["test"]}, key=int)
    meta = {"rpeak_method": rpeaks, "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_head()}

    predictions = list(delineate_record.map(ids, kwargs={"rpeak_method": rpeaks}))
    result = score_predictions.remote(name, predictions, meta)

    (REPO / "results" / "predictions").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    (REPO / "results" / "predictions" / f"{name}.json").write_text(json.dumps(predictions), encoding="utf-8", newline="\n")

    all_leads = result["all_leads"]
    print(f"\n{name}: {all_leads['n_people']} people, failed leads {all_leads['failed_leads']}, orphans {all_leads['orphans']}")
    print(f"primary SD (5 CSE points, all leads): {all_leads['primary_sd_ms']:.2f} ms, 95% CI {all_leads['primary_ci95']}")
    print("\nAll 12 leads, +/-150 ms:")
    print(f"{'point':9s} {'F1%':>7s} {'Se%':>7s} {'PPV%':>7s} {'mean':>7s} {'SD':>7s} {'SDwithin':>9s} {'CSE2sd':>7s}")
    for point, s in all_leads["points"].items():
        at = s["at"]["150"]
        cse = s["cse_2sd_ms"] if s["cse_2sd_ms"] is not None else ""
        print(f"{point:9s} {100 * at['f1']:7.2f} {100 * at['se']:7.2f} {100 * at['ppv']:7.2f} {s['mean_ms']:7.1f} {s['sd_ms']:7.1f} {s['sd_within_ms']:9.1f} {cse!s:>7s}")
    print("\nLead I against Emrich et al. 2024 (single lead, 150 ms):")
    print(f"{'point':9s} {'F1 ours':>8s} {'F1 paper':>9s} {'mean ours':>10s} {'mean paper':>11s} {'SDw ours':>9s} {'SDw paper':>10s} {'band':>5s}")
    for r in result["emrich2024_single_lead_i"]:
        print(f"{r['point']:9s} {r['f1_ours']:8.2f} {r['f1_paper']:9.2f} {r['mean_ours']:10.1f} {r['mean_paper']:11.1f} {r['sd_within_ours']:9.1f} {r['sd_within_paper']:10.1f} {'ok' if r['within_band'] else 'OUT':>5s}")
    print("\nPer lead, QRS onset F1% at 150 ms / failed leads:")
    print("  ".join(f"{lead}:{100 * d['points']['QRS_on']['at']['150']['f1']:.1f}/{d['failed_leads']}" for lead, d in result["per_lead"].items()))
