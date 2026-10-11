"""Illustrations for the step-1 deck that no report produces (docs/slides-outline.md).

    uv run python scripts/deck_figures.py

Writes results/deck/fig_preprocess_steps.png: steps 2 to 5 of `preprocess.prepare` on a real lead with
visible baseline wander (record 91, lead II), and the same beat filtered one way against both ways. The
padded two-way filtering below reproduces `preprocess.highpass` exactly (checked at run time).
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from ecgdelin.dataset import LEADS, load_record
from ecgdelin.paths import ludb_root
from ecgdelin.preprocess import CUTOFF_HZ, ORDER, PAD_SECONDS, highpass
from ecgdelin.reporting import GRID, INK, INK2, SURFACE, apply_style

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "deck"
RAW, ONE_WAY = "#a9a8a2", "#eb6834"
PAD_FILL, PAD_EDGE = "#f3f2ee", "#c9c7c0"
QRS_FILL, T_FILL = "#ecebe7", "#dcf1e8"


def preprocess_steps(record: str = "91", lead: str = "ii") -> Path:
    rec = load_record(ludb_root(), record)
    k = LEADS.index(lead)
    x, fs = rec.signal[k], rec.fs
    sos = butter(ORDER, CUTOFF_HZ, btype="highpass", fs=fs, output="sos")
    pad = int(PAD_SECONDS * fs)
    x_ext = np.concatenate([x[1 : pad + 1][::-1], x, x[-pad - 1 : -1][::-1]])  # scipy's "even" extension
    both_ext = sosfiltfilt(sos, x_ext, padtype=None)
    both = both_ext[pad : pad + x.size]
    one = sosfilt(sos, x_ext)[pad : pad + x.size]
    assert np.allclose(both, highpass(rec.signal, fs)[k], atol=1e-9), "the illustration must match preprocess.highpass"

    fig = plt.figure(figsize=(16.96, 2.6), dpi=200)
    gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.07, left=0.035, right=0.995, top=0.80, bottom=0.17, figure=fig)
    plt.rcParams.update({"font.size": 12})

    ax = fig.add_subplot(gs[0])
    t = (np.arange(x_ext.size) - pad) / fs
    for a, b in ((-PAD_SECONDS, 0.0), (x.size / fs, x.size / fs + PAD_SECONDS)):
        ax.axvspan(a, b, facecolor=PAD_FILL, edgecolor=PAD_EDGE, hatch="///", linewidth=0, zorder=0)
    ax.plot(t, x_ext, color=RAW, lw=0.9, zorder=2)
    ax.plot(t, both_ext, color=INK, lw=1.0, zorder=3)
    keep_end = 4992 / fs
    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axvline(keep_end, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    top = float(max(x_ext.max(), both_ext.max()))
    bottom = float(min(x_ext.min(), both_ext.min()))
    span = top - bottom
    ax.set_ylim(bottom - 0.08 * span, top + 0.30 * span)
    ytxt = top + 0.16 * span
    ax.text(-PAD_SECONDS / 2, ytxt, "step 2: mirrored 6 s", ha="center", va="center", fontsize=12, color=INK2)
    ax.text(x.size / fs + PAD_SECONDS / 2, ytxt, "step 2: mirrored 6 s", ha="center", va="center", fontsize=12, color=INK2)
    ax.text(keep_end / 2, ytxt, "step 5 keeps 0 to 9.98 s: the first 4992 samples", ha="center", va="center", fontsize=12, color=INK)
    ax.set_xlim(-PAD_SECONDS, x.size / fs + PAD_SECONDS)
    ax.set_xticks([-6, 0, 2, 4, 6, 8, 10, 16])
    ax.set_xticklabels(["−6 s", "0", "2", "4", "6", "8", "10", "16 s"], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("mV", fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Steps 2 to 5 on record {record}, lead {lead.upper()}: padded to 22 s, filtered, padding dropped", loc="left", fontsize=13, color=INK)
    handles = [
        plt.Line2D([], [], color=RAW, lw=2, label="raw, with baseline wander"),
        plt.Line2D([], [], color=INK, lw=2, label="after steps 3 and 4"),
        Patch(facecolor=PAD_FILL, edgecolor=PAD_EDGE, hatch="///", label="padding, dropped in step 5"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=3, fontsize=11, bbox_to_anchor=(0.0, -0.36), frameon=False)

    qrs = [w for w in rec.marks[lead] if w.kind == "QRS" and w.onset is not None and w.offset is not None]
    tw = [w for w in rec.marks[lead] if w.kind == "T" and w.onset is not None and w.offset is not None]
    beat = qrs[2]
    t_beat = next(w for w in tw if w.onset > beat.offset)
    lo, hi = beat.onset - int(0.25 * fs), t_beat.offset + int(0.2 * fs)
    bx = fig.add_subplot(gs[1])
    tb = np.arange(lo, hi) / fs
    bx.axvspan(beat.onset / fs, beat.offset / fs, color=QRS_FILL, lw=0, zorder=0)
    bx.axvspan(t_beat.onset / fs, t_beat.offset / fs, color=T_FILL, lw=0, zorder=0)
    bx.axhline(0, color=GRID, lw=0.8, zorder=1)
    bx.plot(tb, one[lo:hi], color=ONE_WAY, lw=1.6, zorder=3)
    bx.plot(tb, both[lo:hi], color=INK, lw=1.6, zorder=4)
    seg = np.concatenate([one[lo:hi], both[lo:hi]])
    span_b = float(seg.max() - seg.min())
    bx.set_ylim(float(seg.min()) - 0.08 * span_b, float(seg.max()) + 0.12 * span_b)
    bx.text((beat.onset + beat.offset) / 2 / fs, float(seg.max()) + 0.04 * span_b, "QRS", ha="center", fontsize=11, color=INK2)
    bx.text((t_beat.onset + t_beat.offset) / 2 / fs, float(seg.max()) + 0.04 * span_b, "T", ha="center", fontsize=11, color=INK2)
    bx.tick_params(labelsize=11)
    bx.set_xticks(bx.get_xticks()[1:-1])
    bx.set_xticklabels([f"{v:.1f} s" for v in bx.get_xticks()], fontsize=11)
    bx.grid(axis="x", visible=False)
    bx.set_title("Why step 4: one way pulls ST and T down", loc="left", fontsize=13, color=INK)
    handles = [
        plt.Line2D([], [], color=INK, lw=2, label="both ways"),
        plt.Line2D([], [], color=ONE_WAY, lw=2, label="one way only"),
        Patch(facecolor=T_FILL, edgecolor=PAD_EDGE, linewidth=0.6, label="marks"),
    ]
    bx.legend(handles=handles, loc="lower left", ncol=3, fontsize=11, bbox_to_anchor=(0.0, -0.36), frameon=False, handlelength=1.4, columnspacing=1.2)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_preprocess_steps.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("wrote", preprocess_steps())
