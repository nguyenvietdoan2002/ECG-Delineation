"""Step 1 results: pooled cross-validated scores, the stop-rule verdict and docs/report.md (decisions 64 to 66).

    uv run python scripts/final_report.py                                   # all five folds -> docs/report.md, results/m4/
    uv run python scripts/final_report.py --folds 0 --suffix _smoke --out <dir>   # a partial check, e.g. the rehearsal

Reads results/m4/m4_fold<k><suffix>.json and the matching results/predictions/*.json for the chosen
folds, NeuroKit2's predictions for all 200 people, the saved splits and the local LUDB copy. Every
person's predictions come from the models of the fold that held them out; the primary metric is
computed per seed and averaged inside each bootstrap draw; the per-fold numbers are for information.
Writes <out>/report.md, <figdir>/fig_*.png and <figdir>/summary.json.
"""

import argparse
import datetime as dt
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ecgdelin.crossval import assemble, per_fold_primary
from ecgdelin.dataset import Recording, load_record
from ecgdelin.delineate import CSE_2SD_MS, CSE_POINTS, POINT_TYPES, LeadPoints
from ecgdelin.paths import ludb_root
from ecgdelin.reporting import INK, INK2, LABEL, SERIES, SURFACE, apply_style, fmt, md_table
from ecgdelin.score import Predictions, Report, compare_seeds, score, seed_averaged_primary
from ecgdelin.splits import Folds

REPO = Path(__file__).resolve().parents[1]
TRAINED = ("head", "unet", "probe")
METHODS = ("head", "unet", "probe", "neurokit_prominence")
NK = "neurokit_prominence"
PAIRS = (("head", "unet"), ("head", NK), ("unet", NK), ("probe", NK), ("head", "probe"))
TOL = 150


def load_predictions(path: Path) -> dict[int, dict[str, dict[str, LeadPoints]]]:
    """``{seed: {record id: {lead: points}}}`` from a predictions file; NeuroKit2's file has no seed and gets seed 0."""
    out: dict[int, dict[str, dict[str, LeadPoints]]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        out.setdefault(int(entry.get("seed", 0)), {})[entry["id"]] = {lead: LeadPoints.from_dict(d) for lead, d in entry["leads"].items()}
    return out


def mean_over(values: Sequence[float]) -> float:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(clean)) if clean else math.nan


def fig_points(figdir: Path, point_stats: dict[str, dict[str, dict[str, list[float]]]]) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(POINT_TYPES))
    offsets = {"head": -0.27, "unet": -0.09, "probe": 0.09, NK: 0.27}
    for name in METHODS:
        ys = [mean_over(point_stats[name][p]["sd_ms"]) for p in POINT_TYPES]
        lo = [min(point_stats[name][p]["sd_ms"]) for p in POINT_TYPES]
        hi = [max(point_stats[name][p]["sd_ms"]) for p in POINT_TYPES]
        ax.vlines(x + offsets[name], lo, hi, color=SERIES[name], lw=1.4, alpha=0.8)
        ax.scatter(x + offsets[name], ys, s=30, color=SERIES[name], edgecolor=SURFACE, linewidth=0.6, label=LABEL[name], zorder=3)
    cse = [CSE_2SD_MS.get(p) for p in POINT_TYPES]
    ax.scatter([k for k, v in enumerate(cse) if v], [v for v in cse if v], marker="_", s=260, color=INK, linewidth=1.6, label="CSE 2σ limit", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in POINT_TYPES])
    ax.set_ylabel("pooled error SD (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Pooled error SD per point over all test people (dot: mean over seeds; bar: seed range)", loc="left", color=INK)
    ax.legend(ncol=5, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.14))
    ax.grid(axis="x", visible=False)
    path = figdir / "fig_points.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_folds(figdir: Path, fold_list: Sequence[int], per_fold: dict[str, list[float]]) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(fold_list))
    offsets = {"head": -0.24, "unet": -0.08, "probe": 0.08, NK: 0.24}
    for name in METHODS:
        ax.scatter(x + offsets[name], per_fold[name], s=34, color=SERIES[name], edgecolor=SURFACE, linewidth=0.6, label=LABEL[name], zorder=3)
    for k, (h, u) in enumerate(zip(per_fold["head"], per_fold["unet"])):
        ax.annotate(f"{h - u:+.1f}", (x[k], max(h, u)), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {k}" for k in fold_list])
    ax.set_xlim(-0.6, len(fold_list) - 0.4)
    ax.set_ylabel("primary SD on the fold's test people (ms)")
    ax.set_ylim(0, 1.18 * max(v for name in METHODS for v in per_fold[name] if not math.isnan(v)))
    ax.set_title("Per fold, mean over seeds (label: head minus U-Net, ms)", loc="left", color=INK)
    ax.legend(ncol=4, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.14))
    ax.grid(axis="x", visible=False)
    path = figdir / "fig_folds.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", default="0,1,2,3,4", help="folds to pool (default: all five)")
    parser.add_argument("--suffix", default="", help="run-name suffix, e.g. _smoke for the rehearsal")
    parser.add_argument("--out", default=str(REPO / "docs"), help="directory for report.md")
    parser.add_argument("--figdir", default=str(REPO / "results" / "m4"), help="directory for figures and summary.json")
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    fold_list = [int(f) for f in args.folds.split(",") if f.strip()]
    out_dir, figdir = Path(args.out), Path(args.figdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    apply_style()

    folds = Folds.from_json((REPO / "splits" / "ludb_5fold_seed0.json").read_text(encoding="utf-8"))
    partial = sorted(fold_list) != list(range(len(folds.folds)))
    runs = {k: f"m4_fold{k}{args.suffix}" for k in fold_list}
    results = {k: json.loads((REPO / "results" / "m4" / f"{run}.json").read_text(encoding="utf-8")) for k, run in runs.items()}
    for k, r in results.items():
        assert r["meta"]["fold"] == k, (k, r["meta"]["fold"])
        assert set(r["meta"]["test_ids"]) == set(folds.folds[k].test), f"fold {k}: the run's test people differ from the saved split"

    people = [rec_id for k in fold_list for rec_id in folds.folds[k].test]
    root = ludb_root()
    recordings = {rec_id: load_record(root, rec_id) for rec_id in people}
    ordered = [recordings[i] for i in people]

    # Predictions: trained models per fold and seed, pooled across folds per seed; NeuroKit2 once.
    by_fold: dict[str, dict[int, dict[int, Predictions]]] = {name: {} for name in TRAINED}
    for k, run in runs.items():
        for name in TRAINED:
            by_fold[name][k] = load_predictions(REPO / "results" / "predictions" / f"{run}_{name}.json")
    seeds = sorted({s for name in TRAINED for k in fold_list for s in by_fold[name][k]})
    for name in TRAINED:
        for k in fold_list:
            missing = [s for s in seeds if s not in by_fold[name][k]]
            assert not missing, f"{name}, fold {k}: no predictions for seeds {missing}"
    pooled: dict[str, list[Predictions]] = {name: [assemble(folds, {k: by_fold[name][k][s] for k in fold_list}, only=fold_list) for s in seeds] for name in TRAINED}
    nk_all = load_predictions(REPO / "results" / "predictions" / "neurokit_prominence.json")[0]
    pooled[NK] = [{rec_id: nk_all[rec_id] for rec_id in people}]

    # Scores per seed, seed-averaged primary with its interval, verdicts, per-fold numbers.
    reports: dict[str, list[Report]] = {name: [score(ordered, preds, n_draws=0) for preds in pooled[name]] for name in METHODS}
    primary = {name: seed_averaged_primary(ordered, pooled[name], n_draws=args.draws, seed=0) for name in METHODS}
    verdicts = {f"{a}_vs_{b}": compare_seeds(ordered, pooled[a], pooled[b], n_draws=args.draws, seed=0) for a, b in PAIRS}
    per_fold = {name: per_fold_primary(recordings, folds, {k: ([by_fold[name][k][s] for s in seeds] if name in TRAINED else [pooled[NK][0]]) for k in fold_list}, only=fold_list) for name in METHODS}
    point_stats = {
        name: {p: {stat: [getattr(r.points[p], stat) for r in reports[name]] for stat in ("mean_ms", "sd_ms", "sd_within_ms")} | {"f1_150": [r.points[p].at[150].f1 for r in reports[name]], "f1_40": [r.points[p].at[40].f1 for r in reports[name]]} for p in POINT_TYPES}
        for name in METHODS
    }
    rhythm_primary = {name: {g: mean_over([r.by_rhythm[g].primary_sd_ms for r in reports[name] if g in r.by_rhythm]) for g in ("sinus", "af_flutter")} for name in METHODS}
    n_params = {name: results[fold_list[0]]["models"][f"{name}_s{seeds[0]}"]["n_params"] for name in TRAINED}

    figs = {"points": fig_points(figdir, point_stats), "folds": fig_folds(figdir, fold_list, per_fold)}
    stop = verdicts["head_vs_unet"]

    # The report.
    lines: list[str] = []
    w = lines.append
    title = "Step 1 results: TimesFM 3 states against a U-Net and NeuroKit2 on LUDB" + (" (PARTIAL)" if partial else "")
    w(f"# {title}\n")
    w(
        f"Generated by `scripts/final_report.py` on {dt.date.today().isoformat()} from runs {', '.join(runs.values())} "
        f"(git {', '.join(sorted({r['meta'].get('git', '?') for r in results.values()}))}; states revision {results[fold_list[0]]['meta']['states_revision'][:8]}; splits `splits/ludb_5fold_seed0.json`). "
        f"{len(people)} test people from {len(fold_list)} fold{'s' if len(fold_list) != 1 else ''}, each predicted by models that never saw them; seeds {seeds}. "
        "Primary metric: mean over the five CSE points of the pooled error SD, computed per seed and averaged over seeds inside every bootstrap draw (decision 64).\n"
    )
    if partial:
        w("**This is a partial pooling over a subset of folds (a rehearsal or a check), not the pre-registered result.**\n")

    w("## 1. Stop-rule verdict\n")
    outcome = "**TimesFM helps: the main head beats the U-Net.**" if stop.a_wins else ("**The head's primary SD is lower, but the F1 safeguard fails, so no claim is made.**" if stop.a_better else "**No claim: the head does not beat the U-Net by the pre-registered test.**")
    w(outcome + "\n")
    w(md_table(
        ["Test", "Value", "Rule"],
        [
            ["Head minus U-Net, primary SD, 95% paired bootstrap interval", f"[{stop.diff_ci95[0]:.2f}, {stop.diff_ci95[1]:.2f}] ms", "entirely below zero"],
            ["Head minus U-Net, mean F1 at 150 ms over the CSE points, 95% interval", f"[{stop.f1_diff_ci95[0]:+.4f}, {stop.f1_diff_ci95[1]:+.4f}]", "not entirely below zero"],
            ["Per-fold head minus U-Net (information only)", ", ".join(f"{h - u:+.2f}" for h, u in zip(per_fold["head"], per_fold["unet"])) + " ms", "–"],
        ],
    ))
    w("")

    w("## 2. Headline\n")
    rows = []
    for name in METHODS:
        point, ci = primary[name]
        f1 = mean_over([mean_over([r.points[p].at[TOL].f1 for p in CSE_POINTS]) for r in reports[name]])
        rows.append([
            LABEL[name],
            f"{n_params[name] / 1e6:.2f}M" if name in n_params else "–",
            f"{fmt(point, 2)} [{fmt(ci[0], 2)}, {fmt(ci[1], 2)}]",
            ", ".join(fmt(r.primary_sd_ms, 2) for r in reports[name]) if name in TRAINED else "–",
            f"{100 * f1:.2f}",
            fmt(mean_over([r.miou['mean'] for r in reports[name]]), 3),
            fmt(rhythm_primary[name]["sinus"], 2),
            fmt(rhythm_primary[name]["af_flutter"], 2),
        ])
    w(md_table(["Method", "Params", "Primary SD ms [95% CI]", "Per seed", "Mean F1@150, CSE points %", "mIoU", "Sinus", "AF/flutter"], rows))
    w("")
    w("Sinus and AF/flutter columns: the primary SD within each rhythm group, averaged over seeds; AF people have no P marks, so their P points carry no error SD and the group's primary is over QRS onset, QRS offset and T offset only.\n")
    w(f"![points]({Path('..') / figdir.relative_to(REPO) / figs['points']})\n" if figdir.is_relative_to(REPO) else f"![points]({figs['points']})\n")

    w("## 3. Per point (mean over seeds)\n")
    rows = []
    for p in POINT_TYPES:
        row = [p.replace("_", " "), fmt(CSE_2SD_MS.get(p)) if p in CSE_2SD_MS else "–"]
        for name in METHODS:
            s = point_stats[name][p]
            row.append(f"{fmt(mean_over(s['mean_ms']))} / {fmt(mean_over(s['sd_ms']))} / {fmt(mean_over(s['sd_within_ms']))}")
        rows.append(row)
    w(md_table(["Point", "CSE 2σ", *(f"{LABEL[n]} mean / pooled SD / within SD" for n in METHODS)], rows))
    w("")
    rows = []
    for p in POINT_TYPES:
        row = [p.replace("_", " ")]
        for name in METHODS:
            s = point_stats[name][p]
            row.append(f"{100 * mean_over(s['f1_150']):.2f} / {100 * mean_over(s['f1_40']):.2f}")
        rows.append(row)
    w(md_table(["Point", *(f"{LABEL[n]} F1@150 / F1@40 %" for n in METHODS)], rows))
    w("")

    w("## 4. Other comparisons\n")
    rows = [[f"{LABEL[a]} vs {LABEL[b]}", f"[{v.diff_ci95[0]:.2f}, {v.diff_ci95[1]:.2f}]", f"[{v.f1_diff_ci95[0]:+.4f}, {v.f1_diff_ci95[1]:+.4f}]", "yes" if v.a_better else "no", "yes" if v.f1_safeguard_ok else "no", "yes" if v.a_wins else "no"] for (a, b), v in zip(PAIRS, verdicts.values())]
    w(md_table(["Pair (a vs b)", "a minus b, primary SD 95% CI ms", "a minus b, F1@150 95% CI", "a better", "F1 safeguard", "a wins"], rows))
    w("")

    w("## 5. Per fold (information only)\n")
    w(f"![folds]({Path('..') / figdir.relative_to(REPO) / figs['folds']})\n" if figdir.is_relative_to(REPO) else f"![folds]({figs['folds']})\n")
    rows = [[f"fold {k}", len(folds.folds[k].test)] + [fmt(per_fold[n][i], 2) for n in METHODS] + [f"{per_fold['head'][i] - per_fold['unet'][i]:+.2f}"] for i, k in enumerate(fold_list)]
    w(md_table(["Fold", "Test people", *(f"{LABEL[n]}" for n in METHODS), "Head minus U-Net"], rows))
    w("")

    w("## 6. Training\n")
    rows = []
    for k in fold_list:
        meta = results[k]["meta"]
        for name, m in results[k]["models"].items():
            rows.append([f"fold {k}", name, f"{m['epochs_run']} ({m['best_epoch']})", fmt(m["best_val_loss"], 4), f"{m['train_seconds'] / 60:.1f}", fmt(m["test_report"]["primary_sd_ms"], 2)])
        rows.append([f"fold {k}", "container", "", "", f"load {meta['load_seconds']:.0f} s, train {meta['train_seconds'] / 60:.0f} min, test {meta['test_seconds']:.0f} s", f"GPU {meta['max_gpu_memory_gib']:.1f} GiB, {meta['device']}"])
    w(md_table(["Fold", "Model", "Epochs (best)", "Best val loss", "Model minutes", "Fold test primary SD ms"], rows))
    w("")
    w("All models: AdamW, learning rate 1e-3, weight decay 1e-4, batch 8, up to 200 epochs, patience 20 on the validation loss, best weights restored; one data stream per fold shared by all seeds and models (decisions 31 and 58).\n")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    summary = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "partial": partial,
        "folds": fold_list,
        "runs": runs,
        "seeds": seeds,
        "n_people": len(people),
        "primary": {name: {"seed_averaged": primary[name][0], "ci95": list(primary[name][1]), "per_seed": [r.primary_sd_ms for r in reports[name]]} for name in METHODS},
        "verdicts": {pair: v.to_dict() for pair, v in verdicts.items()},
        "per_fold_primary": per_fold,
        "reports": {name: [r.to_dict() for r in reports[name]] for name in METHODS},
    }
    (figdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {report_path}, {figdir / 'summary.json'} and {len(figs)} figures")
    print("verdict head vs U-Net:", stop.to_dict())
    for name in METHODS:
        print(f"{LABEL[name]:14s} primary {primary[name][0]:.2f} CI [{primary[name][1][0]:.2f}, {primary[name][1][1]:.2f}] per seed {[round(r.primary_sd_ms, 2) for r in reports[name]]}")


if __name__ == "__main__":
    main()
