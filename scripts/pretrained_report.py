"""Steps 5 and 6 results: the PTB-XL+-pretrained head on LUDB against ECGDeli, the step-1 head and
NeuroKit2, and the adoption rule (decisions 72 and 78).

    python scripts/pretrained_report.py              # docs/ptbxl-pretrain-report.md, results/ludb_pretrained/summary.json
    python scripts/pretrained_report.py --draws 500  # a quick look
    python scripts/pretrained_report.py --zero results/predictions/ecgdeli_ludb.json --finetuned "m4_fold{fold}_head.json" --out <dir>/r.md --summary <dir>/s.json
                                                     # a dry run on existing predictions: the paired comparisons must come out even

Reads arm A's predictions (results/predictions/ludb_zeroshot4_s0_head.json), arm B's per-fold
predictions when all five folds are in (ludb_finetune4_fold<k>_head.json), step 1's head predictions,
ECGDeli's and NeuroKit2's, all for the same 200 people and scored by step 1's scorer. The primary
metric is the pooled error SD, which ignores a constant offset, so each point's mean error (the
offset) is reported beside it.
"""

import argparse
import datetime as dt
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ecgdelin.crossval import assemble, per_fold_primary
from ecgdelin.dataset import load_record
from ecgdelin.delineate import CSE_2SD_MS, CSE_POINTS, POINT_TYPES, LeadPoints
from ecgdelin.finetune import adoption
from ecgdelin.paths import ludb_root
from ecgdelin.reporting import fmt, md_table
from ecgdelin.score import Predictions, Report, compare_seeds, score, seed_averaged_primary
from ecgdelin.splits import Folds

REPO = Path(__file__).resolve().parents[1]
PREDS = REPO / "results" / "predictions"
OUT = REPO / "results" / "ludb_pretrained"
ZERO = "ludb_zeroshot4_s0"
FINETUNE = "ludb_finetune4_fold{fold}"
LABEL = {"zero": "Pretrained head, zero-shot (arm A)", "finetuned": "Pretrained head, fine-tuned (arm B)", "head": "Step 1 head",
         "ecgdeli": "ECGDeli 1.1 (the teacher)", "neurokit": "NeuroKit2"}
TOL = 150


def load_predictions(path: Path) -> dict[int, dict[str, dict[str, LeadPoints]]]:
    """``{seed: {record id: {lead: points}}}``; files without a seed (ECGDeli, NeuroKit2) get seed 0."""
    out: dict[int, dict[str, dict[str, LeadPoints]]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        out.setdefault(int(entry.get("seed", 0)), {})[entry["id"]] = {lead: LeadPoints.from_dict(d) for lead, d in entry["leads"].items()}
    return out


def by_fold(pattern: str, n_folds: int) -> dict[int, dict[int, Predictions]]:
    """``{fold: {seed: predictions}}`` from one predictions file per fold."""
    return {k: load_predictions(PREDS / pattern.format(fold=k)) for k in range(n_folds)}


def pooled(folds: Folds, per_fold: dict[int, dict[int, Predictions]]) -> list[Predictions]:
    """Per seed, the folds' test predictions pooled over all people (decision 64)."""
    seeds = sorted(per_fold[0])
    assert all(sorted(p) == seeds for p in per_fold.values()), "every fold needs the same seeds"
    return [assemble(folds, {k: per_fold[k][s] for k in per_fold}) for s in seeds]


def training_rows(results_pattern: str, n_folds: int) -> list[list[str]]:
    """Arm B's training per fold and seed, from the fold result files (information only)."""
    rows: list[list[str]] = []
    for k in range(n_folds):
        path = Path(results_pattern.format(fold=k))
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        for name, m in r["models"].items():
            if name.startswith("head"):
                rows.append([f"fold {k}", name, f"{m['epochs_run']} ({m['best_epoch']})", fmt(m["best_val_loss"], 4), f"{m['train_seconds'] / 60:.1f}", fmt(m["test_report"]["primary_sd_ms"], 2)])
        meta = r["meta"]
        rows.append([f"fold {k}", "container", "", "", f"load {meta['load_seconds']:.0f} s, train {meta['train_seconds'] / 60:.0f} min, test {meta['test_seconds']:.0f} s", meta.get("device", "")])
    return rows


def mean_over(values: Sequence[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(clean)) if clean else math.nan


def point_row(reports: Sequence[Report], p: str) -> str:
    """Mean error (offset) / pooled SD / F1 at 150 ms, averaged over seeds."""
    mean = mean_over([r.points[p].mean_ms for r in reports])
    sd = mean_over([r.points[p].sd_ms for r in reports])
    f1 = mean_over([r.points[p].at[TOL].f1 for r in reports])
    return f"{fmt(mean, 1)} / {fmt(sd, 1)} / {100 * f1:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--out", default=str(REPO / "docs" / "ptbxl-pretrain-report.md"))
    parser.add_argument("--summary", default=str(OUT / "summary.json"))
    parser.add_argument("--zero", default=str(PREDS / f"{ZERO}_head.json"), help="arm A's predictions")
    parser.add_argument("--finetuned", default=FINETUNE + "_head.json", help="arm B's per-fold predictions under results/predictions/")
    parser.add_argument("--finetuned-results", default=str(OUT / (FINETUNE + ".json")), help="arm B's per-fold result files (training table)")
    args = parser.parse_args()

    folds = Folds.from_json((REPO / "splits" / "ludb_5fold_seed0.json").read_text(encoding="utf-8"))
    people = [rec_id for f in folds.folds for rec_id in f.test]
    root = ludb_root()
    recordings = [load_record(root, rec_id) for rec_id in people]

    n_folds = len(folds.folds)
    head_folds = by_fold("m4_fold{fold}_head.json", n_folds)
    methods: dict[str, list[Predictions]] = {
        "zero": [load_predictions(Path(args.zero))[0]],
        "head": pooled(folds, head_folds),
        "ecgdeli": [load_predictions(PREDS / "ecgdeli_ludb.json")[0]],
        "neurokit": [load_predictions(PREDS / "neurokit_prominence.json")[0]],
    }
    arm_b = all((PREDS / args.finetuned.format(fold=k)).exists() for k in range(n_folds))
    per_fold: dict[str, list[float]] = {}
    if arm_b:
        ft_folds = by_fold(args.finetuned, n_folds)
        methods = {"finetuned": pooled(folds, ft_folds), **methods}
    for name, sets in methods.items():
        for preds in sets:
            missing = [i for i in people if i not in preds]
            assert not missing, f"{name}: no predictions for {missing[:5]}"

    reports = {name: [score(recordings, preds, n_draws=0) for preds in sets] for name, sets in methods.items()}
    primary = {name: seed_averaged_primary(recordings, sets, n_draws=args.draws, seed=0) for name, sets in methods.items()}
    pairs = [("zero", "ecgdeli"), ("zero", "head"), ("zero", "neurokit")] + ([("finetuned", "head"), ("finetuned", "zero")] if arm_b else [])
    verdicts = {f"{a}_vs_{b}": compare_seeds(recordings, methods[a], methods[b], n_draws=args.draws, seed=0) for a, b in pairs}
    decision = None
    if arm_b:
        diff = primary["finetuned"][0] - primary["head"][0]
        decision = {"point_diff_ms": diff, "outcome": adoption(verdicts["finetuned_vs_head"], diff)}
        by_id = {rec.id: rec for rec in recordings}
        per_fold = {name: per_fold_primary(by_id, folds, {k: [d[s] for s in sorted(d)] for k, d in src.items()}) for name, src in (("finetuned", ft_folds), ("head", head_folds))}

    lines: list[str] = []
    w = lines.append
    w("# PTB-XL+ pretraining on LUDB: results" + ("" if arm_b else " (arm A only; arm B pending)") + "\n")
    w(f"Generated by `scripts/pretrained_report.py` on {dt.date.today().isoformat()} for the {len(people)} LUDB people of `splits/ludb_5fold_seed0.json`. "
      "The head was pretrained on ECGDeli's cleaned marks for 4 leads of each of 4,500 PTB-XL ECGs (`results/ptbxl_pretrain4_s0.json`, decisions 72 to 80). "
      "Primary metric: mean over the five CSE points of the pooled error SD (step 1's scorer); it ignores a constant offset, so the mean error per point is shown beside it. "
      f"Intervals: 95% bootstrap over people, {args.draws:,} draws; comparisons are paired.\n")
    if decision is not None:
        w("## Verdict on pretraining (decision 78)\n")
        v = verdicts["finetuned_vs_head"]
        text = {"adopt": "**Adopt: pretraining enters the step-2 detector.**",
                "borderline": "**Borderline: the point estimate favours pretraining but the interval reaches 0; two more pretraining seeds run first (step 7).**",
                "no": "**No: pretraining does not enter the step-2 detector.**"}[decision["outcome"]]
        w(text + "\n")
        w(md_table(["Test", "Value", "Rule"], [
            ["Arm B minus step 1 head, primary SD (point)", f"{decision['point_diff_ms']:+.2f} ms", "below 0"],
            ["95% paired interval of that difference", f"[{v.diff_ci95[0]:.2f}, {v.diff_ci95[1]:.2f}] ms", "entirely below 0"],
            ["95% paired interval, F1 at 150 ms difference", f"[{v.f1_diff_ci95[0]:+.4f}, {v.f1_diff_ci95[1]:+.4f}]", "not entirely below 0"],
        ]))
        w("")

    w("## Headline\n")
    rows = []
    for name in methods:
        point, ci = primary[name]
        f1 = mean_over([mean_over([r.points[p].at[TOL].f1 for p in CSE_POINTS]) for r in reports[name]])
        per_seed = ", ".join(fmt(r.primary_sd_ms, 2) for r in reports[name]) if len(reports[name]) > 1 else "–"
        rows.append([LABEL[name], f"{fmt(point, 2)} [{fmt(ci[0], 2)}, {fmt(ci[1], 2)}]", per_seed, f"{100 * f1:.2f}",
                     fmt(mean_over([r.miou["mean"] for r in reports[name]]), 3)])
    w(md_table(["Method", "Primary SD ms [95% CI]", "Per seed", "Mean F1@150, CSE points %", "mIoU"], rows))
    w("")

    w("## Paired comparisons\n")
    rows = [[f"{LABEL[a]} vs {LABEL[b]}", f"[{v.diff_ci95[0]:.2f}, {v.diff_ci95[1]:.2f}]", f"[{v.f1_diff_ci95[0]:+.4f}, {v.f1_diff_ci95[1]:+.4f}]",
             "yes" if v.a_better else "no", "yes" if v.a_wins else "no"] for (a, b), v in zip(pairs, verdicts.values())]
    w(md_table(["Pair (a vs b)", "a minus b, primary SD 95% CI ms", "a minus b, F1@150 95% CI", "a lower SD", "a wins (with F1 safeguard)"], rows))
    w("")

    w("## Per point: mean error (offset) / pooled SD / F1@150 %\n")
    rows = [[p.replace("_", " "), fmt(CSE_2SD_MS.get(p)) if p in CSE_2SD_MS else "–", *(point_row(reports[n], p) for n in methods)] for p in POINT_TYPES]
    w(md_table(["Point", "CSE 2σ", *(LABEL[n] for n in methods)], rows))
    w("")
    w("A positive mean error means the method marks the point later than the cardiologists. ECGDeli's conventions differ from LUDB's (for example its QRS is wider), "
      "and a student trained on its marks inherits those offsets; the SD measures how consistently it marks, the offset where.\n")

    w("## By rhythm (primary SD, mean over seeds)\n")
    groups = ("sinus", "af_flutter")
    rows = [[LABEL[n], *(fmt(mean_over([r.by_rhythm[g].primary_sd_ms for r in reports[n] if g in r.by_rhythm]), 2) for g in groups)] for n in methods]
    w(md_table(["Method", "Sinus", "AF/flutter"], rows))
    w("")

    if arm_b:
        w("## Arm B per fold (information only, decision 64)\n")
        rows = [[f"fold {k}", len(folds.folds[k].test), fmt(per_fold["finetuned"][k], 2), fmt(per_fold["head"][k], 2), f"{per_fold['finetuned'][k] - per_fold['head'][k]:+.2f}"] for k in range(n_folds)]
        w(md_table(["Fold", "Test people", LABEL["finetuned"], LABEL["head"], "Arm B minus step 1 head"], rows))
        w("")
        rows = training_rows(args.finetuned_results, n_folds)
        if rows:
            w("## Arm B training\n")
            w(md_table(["Fold", "Model", "Epochs (best)", "Best val loss", "Minutes", "Fold test primary SD ms"], rows))
            w("")
            w("Step 1's recipe from the pretrained weights: AdamW, learning rate 1e-3, weight decay 1e-4, batch 8, up to 200 epochs, patience 20 on the fold's validation people, best weights restored; "
              "each seed is its own batch order, seed 0's being step 1's (decision 81).\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "arm_b": arm_b,
        "n_people": len(people),
        "draws": args.draws,
        "primary": {n: {"seed_averaged": primary[n][0], "ci95": list(primary[n][1]), "per_seed": [r.primary_sd_ms for r in reports[n]]} for n in methods},
        "verdicts": {k: v.to_dict() for k, v in verdicts.items()},
        "decision": decision,
        "per_fold_primary": per_fold,
        "reports": {n: [r.to_dict() for r in reports[n]] for n in methods},
    }
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {out} and {summary_path}")
    for n in methods:
        print(f"{LABEL[n]:40s} primary {primary[n][0]:.2f} [{primary[n][1][0]:.2f}, {primary[n][1][1]:.2f}]")
    for k, v in verdicts.items():
        print(f"{k:22s} diff CI [{v.diff_ci95[0]:.2f}, {v.diff_ci95[1]:.2f}] F1 CI [{v.f1_diff_ci95[0]:+.4f}, {v.f1_diff_ci95[1]:+.4f}] wins {v.a_wins}")
    if decision is not None:
        print("decision 78:", decision)


if __name__ == "__main__":
    main()
