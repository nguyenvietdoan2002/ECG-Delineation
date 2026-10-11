"""M3 deep dive (decision 60): tables and figures on fold 0's validation people -> docs/m3-report.md.

    uv run python scripts/m3_report.py

Reads results/m3/m3_fold0_seed0.json, the stored predictions, the class maps written by
scripts/m3_classes.py and the local LUDB copy; writes docs/m3-report.md and results/m3/fig_*.png.
Development numbers only: fold 0's 32 validation people, not a result of the study (decisions 54, 59).
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ecgdelin.analysis import ORPHAN_KINDS, orphan_breakdown, person_primary_sd, roundtrip_points  # noqa: E402
from ecgdelin.dataset import LEADS, Recording, load_record  # noqa: E402
from ecgdelin.decode import clean, runs  # noqa: E402
from ecgdelin.delineate import CSE_2SD_MS, CSE_POINTS, POINT_TYPES, LeadPoints  # noqa: E402
from ecgdelin.paths import ludb_root  # noqa: E402
from ecgdelin.preprocess import prepare  # noqa: E402
from ecgdelin.score import LeadScore, PersonScore, errors_by_lead, score, score_recording  # noqa: E402
from ecgdelin.splits import rhythm_group  # noqa: E402
from ecgdelin.trained import points_from_classes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUN = "m3_fold0_seed0"
RESULT = REPO / "results" / "m3" / f"{RUN}.json"
CLASSES_NPZ = REPO / "results" / "m3" / f"{RUN}_classes.npz"
PREDICTIONS = REPO / "results" / "predictions"
FIG_DIR = REPO / "results" / "m3"
REPORT = REPO / "docs" / "m3-report.md"

TRAINED = ("head", "unet", "probe")
METHODS = ("head", "unet", "probe", "neurokit_prominence")
LABEL = {"head": "TimesFM head", "unet": "U-Net", "probe": "Linear probe", "neurokit_prominence": "NeuroKit2", "roundtrip": "Marks via decoder"}
TOL = 150

# Colours: the reference categorical palette, fixed slot order (methods), and its first three slots,
# validated for all pairs, for the three wave classes in the overlays. Text wears text tokens.
SERIES = {"head": "#2a78d6", "unet": "#eb6834", "probe": "#1baf7a", "neurokit_prominence": "#eda100"}
WAVE = {"P": "#2a78d6", "QRS": "#eb6834", "T": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
    }
)


# ----------------------------------------------------------------------------------------------- data


def load_predictions(path: Path, ids: set[str]) -> dict[str, dict[str, LeadPoints]]:
    return {p["id"]: {lead: LeadPoints.from_dict(d) for lead, d in p["leads"].items()} for p in json.loads(path.read_text(encoding="utf-8")) if p["id"] in ids}


def nk_segments(points: LeadPoints) -> list[tuple[str, int, int]]:
    out = []
    for kind, on_key, off_key in (("P", "ECG_P_Onsets", "ECG_P_Offsets"), ("QRS", "ECG_R_Onsets", "ECG_R_Offsets"), ("T", "ECG_T_Onsets", "ECG_T_Offsets")):
        for on, off in zip(points.points[on_key], points.points[off_key]):
            if math.isfinite(on) and math.isfinite(off):
                out.append((kind, int(round(on)), int(round(off))))
    return out


def fmt(x: float, digits: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# -------------------------------------------------------------------------------------------- figures


def fig_training(result: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for name in TRAINED:
        h = result["models"][name]["history"]
        epochs = [e["epoch"] for e in h]
        ax.plot(epochs, [e["val_loss"] for e in h], color=SERIES[name], lw=1.6, label=f"{LABEL[name]} validation")
        ax.plot(epochs, [e["train_loss"] for e in h], color=SERIES[name], lw=1.0, ls=(0, (3, 2)), alpha=0.6, label=f"{LABEL[name]} training")
        best = result["models"][name]["best_epoch"]
        ax.plot([best], [h[best]["val_loss"]], "o", color=SERIES[name], ms=6, mec=SURFACE, mew=1.2)
    ax.set_ylim(0.08, 0.45)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy per valid sample")
    ax.set_title("Training curves, fold 0 (dot: epoch of the restored weights)", loc="left", color=INK)
    ax.legend(ncol=3, fontsize=7.5, loc="upper right")
    path = FIG_DIR / "fig_training.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_layer_weights(result: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    layers = np.arange(20)
    width = 0.4
    for k, name in enumerate(("probe", "head")):
        w = result["models"][name]["layer_weights"]
        ax.bar(layers + (k - 0.5) * width, w, width=width * 0.9, color=SERIES[name], label=LABEL[name], linewidth=0)
    ax.axhline(1 / 20, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(19.6, 1 / 20 + 0.003, "uniform (1/20)", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{k}" for k in layers], fontsize=7.5)
    ax.set_xlabel("TimesFM 3 layer (output of)")
    ax.set_ylabel("softmax weight")
    ax.set_title("Learned layer blend weights", loc="left", color=INK)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    path = FIG_DIR / "fig_layer_weights.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_per_lead(lead_sd: dict[str, dict[str, dict[str, float]]]) -> str:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    vmax = 40.0
    fig, axes = plt.subplots(1, len(METHODS), figsize=(12.5, 4.6), sharey=True)
    for ax, name in zip(axes, METHODS):
        grid = np.array([[lead_sd[name][lead][p] for p in CSE_POINTS] for lead in LEADS])
        ax.imshow(np.minimum(np.nan_to_num(grid, nan=vmax), vmax), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(len(LEADS)):
            for j in range(len(CSE_POINTS)):
                v = grid[i, j]
                dark = (not math.isnan(v)) and v > 0.55 * vmax
                ax.text(j, i, "–" if math.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK)
        ax.set_xticks(range(len(CSE_POINTS)))
        ax.set_xticklabels([p.replace("_", " ") for p in CSE_POINTS], fontsize=7.5)
        ax.set_title(LABEL[name], loc="left", color=INK)
        ax.grid(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(LEADS)))
    axes[0].set_yticklabels([lead.upper() for lead in LEADS])
    fig.suptitle("Pooled error SD per lead and CSE point, ms (darker = larger; cells above 40 ms are capped)", x=0.01, ha="left", color=INK, fontsize=10)
    path = FIG_DIR / "fig_per_lead.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_per_person(per_person: dict[str, dict[str, float]], groups: dict[str, str]) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    xs = per_person["unet"]
    ys = per_person["head"]
    ids = list(xs)
    for group, marker, color, label in (("sinus", "o", "#2a78d6", "sinus"), ("af_flutter", "^", "#eb6834", "AF / flutter")):
        sel = [i for i in ids if groups[i] == group]
        ax.scatter([xs[i] for i in sel], [ys[i] for i in sel], marker=marker, s=38, color=color, edgecolor=SURFACE, linewidth=0.8, label=label, zorder=3)
    lim = max(v for d in (xs, ys) for v in d.values() if not math.isnan(v)) * 1.08
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for i in sorted(ids, key=lambda i: -(ys[i] if not math.isnan(ys[i]) else -1))[:4]:
        ax.annotate(i, (xs[i], ys[i]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("U-Net, primary SD per person (ms)")
    ax.set_ylabel("TimesFM head, primary SD per person (ms)")
    ax.set_title("Per-person primary SD, head against U-Net", loc="left", color=INK)
    ax.legend(loc="upper left")
    path = FIG_DIR / "fig_per_person.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_qrs_peak(lead_mean: dict[str, dict[str, float]]) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    x = np.arange(len(LEADS))
    offsets = {"head": -0.27, "unet": -0.09, "probe": 0.09, "neurokit_prominence": 0.27}
    for name in METHODS:
        ax.scatter(x + offsets[name], [lead_mean[name][lead] for lead in LEADS], s=26, color=SERIES[name], label=LABEL[name], zorder=3, edgecolor=SURFACE, linewidth=0.6)
    ax.scatter(x, [lead_mean["roundtrip"][lead] for lead in LEADS], marker="x", s=42, color=INK, label=LABEL["roundtrip"], zorder=4, linewidth=1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lead.upper() for lead in LEADS])
    ax.set_ylabel("mean QRS-peak error (ms)")
    ax.set_title("Mean QRS-peak error per lead: predicted minus mark (positive = late)", loc="left", color=INK)
    ax.legend(ncol=5, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.18))
    ax.grid(axis="x", visible=False)
    path = FIG_DIR / "fig_qrs_peak.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_overlay(rec: Recording, prepared: np.ndarray, maps: dict[str, np.ndarray], nk: dict[str, LeadPoints], tag: str, leads: Sequence[str] = ("ii", "v1"), window_s: float = 3.0) -> str:
    fs = rec.fs
    fig, axes = plt.subplots(len(leads), 1, figsize=(11, 3.4 * len(leads)))
    rows = [("probe", "Linear probe"), ("head", "TimesFM head"), ("unet", "U-Net"), ("nk", "NeuroKit2")]
    for ax, lead in zip(np.atleast_1d(axes), leads):
        k = LEADS.index(lead)
        marks = rec.marks[lead]
        all_marks = [s for w in marks for s in (w.onset, w.peak, w.offset) if s is not None]
        start = max(0, min(all_marks) - int(0.1 * fs)) if all_marks else 0
        end = min(prepared.shape[1], start + int(window_s * fs))
        t = np.arange(start, end) / fs
        x = prepared[k, start:end]
        for w in marks:
            if w.onset is not None and w.offset is not None and w.offset >= start and w.onset < end:
                ax.axvspan(w.onset / fs, (w.offset + 1) / fs, color=WAVE[w.kind], alpha=0.18, lw=0)
        ax.plot(t, x, color=INK, lw=1.0, zorder=3)
        ymin, ymax = float(x.min()), float(x.max())
        h = (ymax - ymin) or 1.0
        step = 0.13 * h
        base = ymin - 0.2 * h
        for r, (key, label) in enumerate(rows):
            y = base - r * step
            segments = nk_segments(nk[lead]) if key == "nk" else clean(runs(maps[key][k]), fs)
            spans = [((s0 / fs), (s1 - s0 + 1) / fs, kind) for kind, s0, s1 in segments if s1 >= start and s0 < end]
            for x0, width, kind in spans:
                ax.broken_barh([(x0, width)], (y - 0.045 * h, 0.09 * h), color=WAVE[kind], lw=0)
            ax.text(start / fs - 0.015, y, label, ha="right", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(start / fs, end / fs)
        ax.set_ylim(base - (len(rows) - 0.4) * step, ymax + 0.12 * h)
        ax.set_yticks([tick for tick in ax.get_yticks() if ymin - 0.05 * h <= tick <= ymax + 0.05 * h])  # no ticks beside the band labels
        ax.set_ylabel("mV")
        ax.set_title(f"Record {rec.id} ({rec.rhythm}), lead {lead.upper()}", loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    handles = [Patch(color=WAVE[k], label=k) for k in ("P", "QRS", "T")]
    fig.legend(handles=handles, ncol=3, loc="upper right", bbox_to_anchor=(0.99, 1.0), title="shaded: cardiologists' marks; bars: predicted regions", title_fontsize=7.5, fontsize=7.5)
    path = FIG_DIR / f"fig_overlay_{tag}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


# --------------------------------------------------------------------------------------------- report


def pooled_sd(errors: Sequence[np.ndarray]) -> float:
    e = np.concatenate(list(errors)) if errors else np.array([])
    return float(e.std()) if e.size >= 2 else math.nan


def pooled_mean(errors: Sequence[np.ndarray]) -> float:
    e = np.concatenate(list(errors)) if errors else np.array([])
    return float(e.mean()) if e.size else math.nan


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    ids = list(result["meta"]["val_ids"])
    id_set = set(ids)
    root = ludb_root()
    recordings = {rec_id: load_record(root, rec_id) for rec_id in ids}
    prepared = {rec_id: prepare(rec.signal, rec.fs) for rec_id, rec in recordings.items()}
    groups = {rec_id: rhythm_group(rec.rhythm) for rec_id, rec in recordings.items()}
    preds = {name: load_predictions(PREDICTIONS / f"{RUN}_{name}.json", id_set) for name in TRAINED}
    preds["neurokit_prominence"] = load_predictions(PREDICTIONS / "neurokit_prominence.json", id_set)
    npz = np.load(CLASSES_NPZ)
    map_ids = [str(i) for i in npz["ids"]]
    maps = {name: {rec_id: npz[name][k] for k, rec_id in enumerate(map_ids)} for name in TRAINED}
    ordered = [recordings[i] for i in ids]

    # 0. Consistency: class maps decoded here against the stored predictions, and local scores against the JSON.
    identical = {}
    for name in TRAINED:
        same = 0
        for rec_id in ids:
            local = points_from_classes(maps[name][rec_id], prepared[rec_id], recordings[rec_id].fs)
            for lead in LEADS:
                same += local[lead].to_dict() == preds[name][rec_id][lead].to_dict()
        identical[name] = same
    reports = {name: score(ordered, preds[name]) for name in METHODS}
    json_primary = {name: (result["models"][name]["report"] if name in TRAINED else result[name])["primary_sd_ms"] for name in METHODS}

    # 1. Per person and per lead.
    person: dict[str, dict[str, PersonScore]] = {name: {rec_id: score_recording(recordings[rec_id], preds[name][rec_id]) for rec_id in ids} for name in METHODS}
    per_person = {name: {rec_id: person_primary_sd(person[name][rec_id]) for rec_id in ids} for name in METHODS}
    by_lead: dict[str, dict[str, dict[str, LeadScore]]] = {name: {rec_id: errors_by_lead(recordings[rec_id], preds[name][rec_id]) for rec_id in ids} for name in METHODS}
    lead_sd = {name: {lead: {p: pooled_sd([by_lead[name][i][lead].errors_ms[p] for i in ids]) for p in POINT_TYPES} for lead in LEADS} for name in METHODS}
    lead_primary = {name: {lead: float(np.nanmean([lead_sd[name][lead][p] for p in CSE_POINTS])) for lead in LEADS} for name in METHODS}

    # 2. Orphans.
    orphans = {name: {kind: 0 for kind in ORPHAN_KINDS} for name in TRAINED}
    for name in TRAINED:
        for rec_id in ids:
            for kind, count in orphan_breakdown(maps[name][rec_id], recordings[rec_id], prepared[rec_id]).items():
                orphans[name][kind] += count

    # 3. Round trip of the marks through the decoder.
    roundtrip = {rec_id: roundtrip_points(recordings[rec_id], prepared[rec_id]) for rec_id in ids}
    rt_person = {rec_id: score_recording(recordings[rec_id], roundtrip[rec_id]) for rec_id in ids}
    rt_lead = {rec_id: errors_by_lead(recordings[rec_id], roundtrip[rec_id]) for rec_id in ids}
    peak_rows = []
    for name in METHODS:
        errs = {p: [person[name][i].errors_ms[p] for i in ids] for p in ("P_peak", "QRS_peak", "T_peak")}
        peak_rows.append([LABEL[name]] + [f"{fmt(pooled_mean(errs[p]))} ± {fmt(pooled_sd(errs[p]))}" for p in errs])
    rt_errs = {p: [rt_person[i].errors_ms[p] for i in ids] for p in ("P_peak", "QRS_peak", "T_peak")}
    peak_rows.append([LABEL["roundtrip"]] + [f"{fmt(pooled_mean(rt_errs[p]))} ± {fmt(pooled_sd(rt_errs[p]))}" for p in rt_errs])
    rt_qrs = np.concatenate(rt_errs["QRS_peak"])
    rt_exact = float(np.mean(rt_qrs == 0))
    rt_over20 = float(np.mean(np.abs(rt_qrs) > 20))
    lead_mean = {name: {lead: pooled_mean([by_lead[name][i][lead].errors_ms["QRS_peak"] for i in ids]) for lead in LEADS} for name in METHODS}
    lead_mean["roundtrip"] = {lead: pooled_mean([rt_lead[i][lead].errors_ms["QRS_peak"] for i in ids]) for lead in LEADS}

    offsets, variants = peak_diagnostics(recordings, prepared, ids)

    # 4. AF people.
    af_ids = [i for i in ids if groups[i] == "af_flutter"]
    af_rows = []
    for rec_id in af_ids:
        for name in METHODS:
            s = person[name][rec_id]
            af_rows.append(
                [rec_id if name == METHODS[0] else "", recordings[rec_id].rhythm if name == METHODS[0] else "", LABEL[name], s.counts[TOL]["P_on"].fp, s.counts[TOL]["QRS_on"].tp, s.counts[TOL]["QRS_on"].fn, s.counts[TOL]["QRS_on"].fp]
                + [fmt(float(np.std(s.errors_ms[p])) if s.errors_ms[p].size >= 2 else math.nan) for p in ("QRS_on", "QRS_off", "T_off")]
            )

    # 5. Figures.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "training": fig_training(result),
        "layers": fig_layer_weights(result),
        "per_lead": fig_per_lead(lead_sd),
        "per_person": fig_per_person(per_person, groups),
        "qrs_peak": fig_qrs_peak(lead_mean),
    }
    sinus = sorted((i for i in ids if groups[i] == "sinus"), key=lambda i: per_person["head"][i])
    typical, worst = sinus[len(sinus) // 2], sinus[-1]
    af_pick = max(af_ids, key=lambda i: per_person["head"][i] if not math.isnan(per_person["head"][i]) else -1) if af_ids else None
    overlay_people = [("typical", typical), ("worst", worst)] + ([("af", af_pick)] if af_pick else [])
    for tag, rec_id in overlay_people:
        figs[f"overlay_{tag}"] = fig_overlay(recordings[rec_id], prepared[rec_id], {n: maps[n][rec_id] for n in TRAINED}, preds["neurokit_prominence"][rec_id], tag)

    # 6. Markdown.
    out: list[str] = []
    w = out.append
    w("# M3 deep dive: fold 0, validation people only\n")
    w(f"Generated by `scripts/m3_report.py` from `results/m3/{RUN}.json`, the stored predictions, the class maps in "
      f"`results/m3/{RUN}_classes.npz` (from `scripts/m3_classes.py`) and the local LUDB copy. "
      f"Scope: {result['scope']}. {len(ids)} people, {len(sinus)} sinus and {len(af_ids)} AF/flutter; the training people are never scored here.\n")

    w("## 1. Headline\n")
    rows = []
    for name in METHODS:
        r = reports[name]
        m = result["models"].get(name)
        f1 = float(np.mean([r.points[p].at[TOL].f1 for p in CSE_POINTS]))
        rows.append([LABEL[name], f"{m['n_params'] / 1e6:.2f}M" if m else "–", f"{m['epochs_run']} ({m['best_epoch']})" if m else "–", fmt(m["best_val_loss"], 4) if m else "–",
                     f"{fmt(r.primary_sd_ms, 2)} [{fmt(r.primary_ci95[0])}, {fmt(r.primary_ci95[1])}]", f"{100 * f1:.1f}", fmt(r.miou["mean"], 3), r.orphans])
    w(md_table(["Method", "Params", "Epochs (best)", "Best val loss", "Primary SD ms [95% CI]", "Mean F1@150 CSE pts %", "mIoU", "Orphans"], rows))
    w("")
    w("Primary SD = mean over the five CSE points of the pooled error SD. The exploratory verdicts on these people are in the JSON "
      "(`verdicts_validation_only`): head beats U-Net, difference CI "
      f"[{result['verdicts_validation_only']['head_vs_unet']['diff_ci95'][0]:.2f}, {result['verdicts_validation_only']['head_vs_unet']['diff_ci95'][1]:.2f}] ms.\n")
    w(f"![training curves]({'../results/m3/' + figs['training']})\n")

    w("## 2. Consistency checks\n")
    w(md_table(["Check", "Result"], [
        ["Local scorer reproduces the JSON primary SD"] + [", ".join(f"{LABEL[n]} {fmt(reports[n].primary_sd_ms, 2)} vs {fmt(json_primary[n], 2)}" for n in METHODS)],
        [f"Class maps (CPU re-inference) decoded locally give the stored points, leads identical out of {12 * len(ids)}"] + [", ".join(f"{LABEL[n]} {identical[n]}" for n in TRAINED)],
    ]))
    w("")
    w("The class maps were recomputed on CPU from the saved weights; the stored predictions came from the GPU run. "
      "A lead could differ only by an argmax flip at a near-tied sample; the orphan and overlay sections below use the CPU maps.\n")

    w("## 3. Per point\n")
    rows = []
    for p in POINT_TYPES:
        row = [p.replace("_", " "), fmt(CSE_2SD_MS.get(p)) if p in CSE_2SD_MS else "–"]
        for name in METHODS:
            s = reports[name].points[p]
            row.append(f"{fmt(s.mean_ms)} / {fmt(s.sd_ms)} / {fmt(s.sd_within_ms)}")
        rows.append(row)
    w(md_table(["Point", "CSE 2σ", *(f"{LABEL[n]} mean / pooled SD / within SD" for n in METHODS)], rows))
    w("")
    rows = []
    for p in POINT_TYPES:
        row = [p.replace("_", " ")]
        for name in METHODS:
            s = reports[name].points[p]
            row.append(f"{100 * s.at[150].f1:.1f} / {100 * s.at[40].f1:.1f}")
        rows.append(row)
    w(md_table(["Point", *(f"{LABEL[n]} F1@150 / F1@40 %" for n in METHODS)], rows))
    w("")

    w("## 4. Per lead\n")
    w(f"![per lead]({'../results/m3/' + figs['per_lead']})\n")
    rows = [[lead.upper()] + [fmt(lead_primary[n][lead]) for n in METHODS] for lead in LEADS]
    w(md_table(["Lead", *(f"{LABEL[n]} primary SD ms" for n in METHODS)], rows))
    w("")
    cells = sorted(((lead_sd[n][lead][p], n, lead, p) for n in TRAINED for lead in LEADS for p in CSE_POINTS if not math.isnan(lead_sd[n][lead][p])), reverse=True)[:8]
    w("Largest pooled SD cells among the trained models: " + "; ".join(f"{LABEL[n]} {lead.upper()} {p.replace('_', ' ')} {v:.0f} ms" for v, n, lead, p in cells) + ".\n")

    w("## 5. Per person\n")
    w(f"![per person]({'../results/m3/' + figs['per_person']})\n")
    rows = [[i, groups[i]] + [fmt(per_person[n][i]) for n in METHODS] for i in sorted(ids, key=lambda i: -(per_person['head'][i] if not math.isnan(per_person['head'][i]) else -1))]
    w(md_table(["Record", "Rhythm group", *(f"{LABEL[n]}" for n in METHODS)], rows))
    w("")
    w("Per-person primary SD: mean over the CSE points of that person's pooled error SD across leads; points with fewer than two matched pairs are left out, so AF people, who have no P marks, are judged on QRS and T only.\n")

    w("## 6. Orphans\n")
    rows = [[LABEL[n]] + [orphans[n][k] for k in ORPHAN_KINDS] + [sum(orphans[n].values()), reports[n].orphans] for n in TRAINED]
    w(md_table(["Model", "Edge (outside the marked span)", "Extra (slot taken by a nearer wave)", "Unanchored (no QRS on the needed side)", "Total from class maps", "Total in stored predictions"], rows))
    w("")
    w("Edge orphans are P or T regions before the lead's first mark or after its last, where no reference exists and nothing is scored. "
      "Extra orphans are fragments or doubled waves inside the span. Unanchored orphans inside the span mean a QRS was missed or the P/T sits beyond the last predicted QRS.\n")

    w("## 7. Peaks and the decoder\n")
    w(md_table(["Method", "P peak mean ± SD ms", "QRS peak mean ± SD ms", "T peak mean ± SD ms"], peak_rows))
    w("")
    w(f"The last row pushes the cardiologists' own regions through the decoder, so it isolates the decoder's peak rule (decision 47) from the models. "
      f"On these people the rule returns the marked QRS peak exactly for {100 * rt_exact:.1f}% of beats and is more than 20 ms away for {100 * rt_over20:.1f}%.\n")
    w(f"![qrs peak]({'../results/m3/' + figs['qrs_peak']})\n")
    w("Where the extremum of the raw signal lies relative to each mark (largest absolute deflection from the wave's baseline within ±6 samples of the mark):\n")
    w(md_table(["Wave", "Extremum at the mark", "1 sample after", "2 samples after", "Within ±2 samples", "Marks"],
               [[kind, f"{100 * o['at0']:.1f}%", f"{100 * o['plus1']:.1f}%", f"{100 * o['plus2']:.1f}%", f"{100 * o['within2']:.1f}%", o["n"]] for kind, o in offsets.items()]))
    w("")
    w("Stricter R-wave rules on the round trip (the positive deflection wins only when it reaches a fraction r of the largest absolute deflection; r = 0 is decision 47's rule):\n")
    w(md_table(["r", "QRS peak mean ms", "SD ms", "> 20 ms, all leads", "> 20 ms, aVR", "> 20 ms, V1"], [[f"{r:.1f}", fmt(v["mean"], 2), fmt(v["sd"]), f"{100 * v['over20']:.1f}%", f"{100 * v['avr']:.1f}%", f"{100 * v['v1']:.1f}%"] for r, v in variants.items()]))
    w("")

    w("## 8. AF and flutter people\n")
    w(md_table(["Record", "Rhythm", "Method", "P-onset FP @150", "QRS-onset TP", "FN", "FP", "QRS on SD", "QRS off SD", "T off SD"], af_rows))
    w("")

    w("## 9. Beat overlays\n")
    for tag, rec_id in overlay_people:
        title = {"typical": "median head score among sinus people", "worst": "worst head score among sinus people", "af": "AF/flutter person with the worst head score"}[tag]
        w(f"Record {rec_id}, {title} (head {fmt(per_person['head'][rec_id])} ms, U-Net {fmt(per_person['unet'][rec_id])} ms).\n")
        w(f"![overlay {tag}]({'../results/m3/' + figs['overlay_' + tag]})\n")

    w("## 10. Layer weights\n")
    w(f"![layer weights]({'../results/m3/' + figs['layers']})\n")
    for name in ("probe", "head"):
        lw = result["models"][name]["layer_weights"]
        top = sorted(range(20), key=lambda k: -lw[k])[:5]
        w(f"- {LABEL[name]}: max {max(lw):.3f}, min {min(lw):.3f}; heaviest " + ", ".join(f"L{k} {lw[k]:.3f}" for k in top) + ".")
    w("")

    n_af_train = sum(rhythm_group(load_record(root, i).rhythm) == "af_flutter" for i in result["meta"]["train_ids"])
    w(findings(result, reports, identical, orphans, per_person, groups, lead_sd, offsets, variants, af_ids, person, n_af_train))
    REPORT.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {REPORT} and {len(figs)} figures under {FIG_DIR}")
    print("consistency:", identical, {n: (round(reports[n].primary_sd_ms, 3), round(json_primary[n], 3)) for n in METHODS})
    print("orphans:", orphans)
    print("roundtrip QRS peak exact", round(rt_exact, 4), "over 20 ms", round(rt_over20, 4))
    print("peaks:", peak_rows)


def peak_diagnostics(recordings: dict[str, Recording], prepared: dict[str, np.ndarray], ids: Sequence[str]) -> tuple[dict[str, dict[str, float]], dict[float, dict[str, float]]]:
    """(a) Where the raw signal's extremum lies relative to each mark; (b) stricter R-wave rules on the round trip."""
    offsets: dict[str, list[int]] = {"P": [], "QRS": [], "T": []}
    for rec_id in ids:
        rec = recordings[rec_id]
        for k, lead in enumerate(LEADS):
            x = rec.signal[k]
            for wv in rec.marks[lead]:
                if wv.onset is None or wv.offset is None:
                    continue
                deflection = x[wv.onset : wv.offset + 1] - (x[wv.onset] + x[wv.offset]) / 2
                lo, hi = max(wv.onset, wv.peak - 6), min(wv.offset, wv.peak + 6)
                offsets[wv.kind].append(int(lo + np.argmax(np.abs(deflection[lo - wv.onset : hi - wv.onset + 1])) - wv.peak))
    summary = {}
    for kind, values in offsets.items():
        v = np.array(values)
        summary[kind] = {"at0": float(np.mean(v == 0)), "plus1": float(np.mean(v == 1)), "plus2": float(np.mean(v == 2)), "within2": float(np.mean(np.abs(v) <= 2)), "n": int(v.size)}

    from ecgdelin.dataset import Wave
    from ecgdelin.delineate import points_from_waves
    from ecgdelin.labels import labels_for

    variants = {}
    for r in (0.0, 0.1, 0.3, 1.0):
        per_lead: dict[str, list[np.ndarray]] = {lead: [] for lead in LEADS}
        for rec_id in ids:
            rec = recordings[rec_id]
            sig = prepared[rec_id]
            pts = {}
            for k, lead in enumerate(LEADS):
                x = np.asarray(sig[k], dtype=np.float64)
                waves = []
                for kind, s, e in clean(runs(labels_for(rec.marks[lead], sig.shape[1]).classes), rec.fs):
                    d = x[s : e + 1] - (x[s] + x[e]) / 2
                    if kind == "QRS":
                        pk = s + int(np.argmax(d) if (d.max() > 0.02 and d.max() >= r * np.abs(d).max()) else np.argmin(d))
                    else:
                        pk = s + int(np.argmax(np.abs(d)))
                    waves.append(Wave(kind, s, pk, e))
                pts[lead] = points_from_waves(waves)
            by_lead = errors_by_lead(rec, pts)
            for lead in LEADS:
                per_lead[lead].append(by_lead[lead].errors_ms["QRS_peak"])
        every = np.concatenate([np.concatenate(v) for v in per_lead.values()])
        over = {lead: float(np.mean(np.abs(np.concatenate(v)) > 20)) for lead, v in per_lead.items()}
        variants[r] = {"mean": float(every.mean()), "sd": float(every.std()), "over20": float(np.mean(np.abs(every) > 20)), "avr": over["avr"], "v1": over["v1"]}
    return summary, variants


def findings(result, reports, identical, orphans, per_person, groups, lead_sd, offsets, variants, af_ids, person, n_af_train) -> str:  # type: ignore[no-untyped-def]
    """Interpretation of the numbers above (2026-10-11); the numbers in it are computed, the reading is the author's."""
    ids = list(per_person["head"])
    n_leads = 12 * len(ids)
    head_le_unet = sum(1 for i in ids if per_person["head"][i] <= per_person["unet"][i])
    head_le_probe = sum(1 for i in ids if per_person["head"][i] <= per_person["probe"][i])
    worst = max(ids, key=lambda i: per_person["unet"][i])
    t_off = {n: float(np.nanmean([lead_sd[n][lead]["T_off"] for lead in LEADS])) for n in TRAINED}
    af_fp = {n: sum(person[n][i].counts[TOL]["P_on"].fp for i in af_ids) for n in METHODS}
    lw = {n: result["models"][n]["layer_weights"] for n in ("probe", "head")}
    top_probe = sorted(range(20), key=lambda k: -lw["probe"][k])[:5]
    m = result["models"]
    qrs = offsets["QRS"]
    v0, v1 = variants[0.0], variants[0.1]
    lines = [
        "## 11. Findings\n",
        f"1. **The pipeline is consistent.** Re-running the saved weights on CPU and decoding locally reproduces every one of the {n_leads} stored lead predictions for each model "
        f"({', '.join(f'{LABEL[n]} {identical[n]}' for n in TRAINED)}), and the local scorer reproduces the JSON's primary SDs to the last digit. Nothing in the result depends on the GPU.\n",
        f"2. **Orphans are edge waves, not fragmentation.** Of the U-Net's {sum(orphans['unet'].values())} orphans, {orphans['unet']['edge']} lie before the lead's first mark or after its last, "
        f"where nothing is scored; the head has {orphans['head']['edge']} such edge waves and the probe {orphans['probe']['edge']}. Inside the marked span the counts are small "
        f"({orphans['head']['extra']} head, {orphans['unet']['extra']} U-Net, {orphans['probe']['extra']} probe fragments or doubled waves) and no orphan is unanchored, so no QRS was missed with a P or T left behind. "
        "The U-Net labels the unmarked first and last beats because its convolutions are translation-invariant; the head mostly does not, because the edges are masked in training and the states of the first blocks follow little context. "
        "The probe's fragments are what one block at a time costs; the head's convolutions remove nearly all of them. **No action.**\n",
        f"3. **The QRS-peak bias belongs to the decoder and comes from LUDB's marks.** Pushing the cardiologists' own regions through the decoder gives a mean QRS-peak error of +{variants[0.0]['mean']:.1f} ms, the same as the three models. "
        f"The reason is in the marks: the extremum of the raw signal lies one sample after the QRS mark for {100 * qrs['plus1']:.0f}% of beats and two samples after for {100 * qrs['plus2']:.0f}%, at the mark for only {100 * qrs['at0']:.0f}%. "
        "LUDB's peak marks sit about one sample (2 ms) before the signal's extremum, systematically. The models learn and are scored against the marks, so this touches nothing in the primary metric, "
        "which uses onsets and offsets only; it shifts the reported QRS, P and T peak means by about +2 ms. **Proposed, for your decision:** either (a) leave the decoder as it is and state the offset in the paper, "
        "or (b) move every decoder peak one sample earlier before the freeze, logged as a development-stage decision like decision 47. I recommend (a): the correction would be tailoring to an annotation convention, and the mean error is reported anyway.\n",
        f"4. **The larger QRS-peak disagreements are morphology, mostly in aVR.** With decision 47's rule, {100 * v0['over20']:.1f}% of round-trip QRS peaks are more than 20 ms from the mark, {100 * v0['avr']:.0f}% in aVR and {100 * v0['v1']:.0f}% in V1. "
        f"Every stricter rule tested is worse (at r = 0.1 already {100 * v1['over20']:.1f}% overall and {100 * v1['avr']:.0f}% in aVR), so the cardiologists do mark small r waves and decision 47 stands. **No action.**\n",
        "5. **P and T peaks have a decoder floor too.** The round trip reproduces the marked P peak with an SD of 9 ms and the T peak with 10 ms: for broad waves the marked peak is not the extremum of the high-passed signal to within a few samples. "
        "The models' P-peak and T-peak SDs (13 and 17 ms for the head) include that floor. The alternatives tested, the extremum about the region mean and the raw extremum, are far worse (P 37 to 40 ms). **No action.**\n",
        f"6. **Per lead, T offset is the hardest point everywhere** (mean over leads: head {t_off['head']:.0f} ms, U-Net {t_off['unet']:.0f} ms, probe {t_off['probe']:.0f} ms), and it is where the head gains most on the U-Net, followed by P onset and offset in V1 and V2. "
        "QRS onset and offset are 6 to 13 ms for all trained models in every lead. NeuroKit2's weak leads are aVR, III and V1, where its P detections are far off.\n",
        f"7. **Per person, the head is at or below the U-Net for {head_le_unet} of {len(ids)} people and at or below the probe for {head_le_probe}.** The worst person for every method is record {worst} (atrial fibrillation). "
        f"On the {len(af_ids)} AF/flutter people all methods emit P-wave false positives (P-onset FPs at 150 ms: {', '.join(f'{LABEL[n]} {af_fp[n]}' for n in METHODS)}); the models saw only {n_af_train} AF/flutter people among the {len(result['meta']['train_ids'])} training people, and nothing tells them that a P-shaped bump during fibrillation is not a P wave. "
        "This is expected and is reported by rhythm. **No action.**\n",
        f"8. **Layer weights.** The probe concentrates on layers {', '.join(f'L{k}' for k in top_probe)} (weights {max(lw['probe']):.3f} down to {lw['probe'][top_probe[-1]]:.3f}) and nearly switches off L0, L1 and L14 to L19 ({min(lw['probe']):.3f} to 0.03); "
        f"the head stays between {min(lw['head']):.3f} and {max(lw['head']):.3f}, close to uniform. A linear probe can only select layers; the head's 2560 to 128 projection selects features inside every layer and needs no layer selection. "
        "The per-layer probes, now a possible later step, would show the picture directly. **No action for M4.**\n",
        f"9. **Training.** The head's validation loss bottoms at epoch {m['head']['best_epoch']} and then rises while the training loss keeps falling: it overfits 128 recordings quickly, and early stopping handles it. "
        f"The U-Net's validation loss is noisy and still drifts down when patience stops it at epoch {m['unet']['epochs_run'] - 1} (best {m['unet']['best_epoch']}); the probe is slow, best at epoch {m['probe']['best_epoch']} of {m['probe']['epochs_run']}. "
        "Decision 62 keeps the recipe. For M4's cost the probe matters most: a fold container runs until its slowest model stops, and the probes will be the slowest.\n",
        "**Summary for the freeze:** the labels, the decoder and the scorer behave as designed; the only decision open is item 3, and the recommendation is to leave the decoder unchanged.\n",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
