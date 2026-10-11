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


CLASS_COLORS = {"P": "#2a78d6", "QRS": "#eb6834", "T": "#1baf7a", "none": "#b9b8b2"}
CLASS_TINTS = {"P": "#d5e5f8", "QRS": "#fbdccd", "T": "#d3efe3"}
STATES_CACHE = OUT / "states_pca_115_layer5.npz"


def states_projection(states_path: Path, record: str = "115", layer: int = 5) -> Path:
    """Principal components of one record's stored states at one layer, all 12 leads, with each block's
    majority label from the cardiologists' marks; cached so the figure needs no 183 MB download."""
    from ecgdelin.labels import CLASSES, labels_for
    from ecgdelin.preprocess import N_SAMPLES

    states = np.load(states_path, mmap_mode="r")[layer].astype(np.float32)  # (12, 156, 2560)
    x = states.reshape(-1, states.shape[-1])
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)  # per-vector normalisation, as the heads
    xc = x - x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    pcs = (xc @ vt[:5].T).reshape(12, 156, 5)
    rec = load_record(ludb_root(), record)
    majority = np.full((12, 156), -1, dtype=np.int64)
    for k, lead in enumerate(LEADS):
        lab = labels_for(rec.marks[lead], N_SAMPLES)
        cls, mask = lab.classes.reshape(156, 32), lab.mask.reshape(156, 32)
        for b in range(156):
            if mask[b].any():
                majority[k, b] = int(np.bincount(cls[b][mask[b]], minlength=len(CLASSES)).argmax())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STATES_CACHE, pcs=pcs.astype(np.float32), majority=majority, explained=(s**2 / (s**2).sum())[:5], record=record, layer=layer)
    return STATES_CACHE


def states_figure(record: str = "115", lead: str = "ii", with_scatter: bool = False) -> Path:
    from ecgdelin.labels import CLASSES
    from ecgdelin.preprocess import N_SAMPLES, prepare

    data = np.load(STATES_CACHE)
    pcs, majority, layer = data["pcs"], data["majority"], int(data["layer"])
    ok = majority >= 0

    def class_r2(values: np.ndarray) -> float:
        y, g = values[ok], majority[ok]
        total = float(((y - y.mean()) ** 2).sum())
        between = sum(float(((y[g == c].mean() - y.mean()) ** 2) * (g == c).sum()) for c in np.unique(g))
        return between / total

    r2 = [class_r2(pcs[..., j]) for j in range(pcs.shape[-1])]
    a, b = sorted(np.argsort(r2)[-2:])  # the two components that follow the waves best
    k = LEADS.index(lead)
    pc = pcs[k, :, a].copy()
    if np.median(pc[majority[k] == CLASSES.index("QRS")]) < 0:  # sign is arbitrary: QRS blocks up
        pc = -pc
        pcs[..., a] = -pcs[..., a]
    rec = load_record(ludb_root(), record)
    sig = prepare(rec.signal, rec.fs)[k]
    t = np.arange(N_SAMPLES) / rec.fs
    tb = np.arange(157) * 32 / rec.fs

    if with_scatter:
        fig = plt.figure(figsize=(16.96, 4.4), dpi=200)
        gs = GridSpec(2, 2, width_ratios=[2.3, 1], height_ratios=[1, 1], wspace=0.10, hspace=0.42, left=0.04, right=0.99, top=0.90, bottom=0.10, figure=fig)
    else:  # slide 14: the time panels only; the chain that makes the line is drawn on the slide beside them
        fig = plt.figure(figsize=(11.1, 4.4), dpi=200)
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.42, left=0.065, right=0.99, top=0.90, bottom=0.10, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    px = fig.add_subplot(gs[1, 0], sharex=ax)
    marked = [s for w in rec.marks[lead] for s in (w.onset, w.peak, w.offset) if s is not None]
    for w in rec.marks[lead]:
        if w.onset is not None and w.offset is not None:
            ax.axvspan(w.onset / rec.fs, (w.offset + 1) / rec.fs, color=CLASS_TINTS[w.kind], lw=0, zorder=0)
        if w.kind == "QRS":  # in the state panel, only thin guides at the cardiologists' QRS marks
            px.axvline(w.peak / rec.fs, color="#8a8378", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.plot(t, sig, color=INK, lw=0.9, zorder=2)
    ax.set_ylabel("mV", fontsize=11)
    ax.tick_params(labelbottom=False, labelsize=11)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Record {record}, lead {lead.upper()}: the ECG, with the cardiologists' marks shaded", loc="left", fontsize=13, color=INK)
    px.step(tb, np.append(pc, pc[-1]), where="post", color=INK, lw=1.3, zorder=2)
    px.set_title(f"Frozen TimesFM's state per 64 ms block (layer {layer}) as one number, no labels · dashed: QRS marks", loc="left", fontsize=12, color=INK)
    for lo, hi in ((0, min(marked)), (max(marked) + 1, N_SAMPLES)):  # beats the cardiologists left unmarked
        if hi - lo > rec.fs // 4:
            r = lo + int(np.argmax(sig[lo:hi]))
            px.annotate("unmarked beat", (r / rec.fs, pc[r // 32]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10.5, color=INK2)
    px.set_ylabel(f"layer {layer}, PC {a + 1}", fontsize=11)
    px.tick_params(labelsize=11)
    px.set_xlim(0, N_SAMPLES / rec.fs)
    px.set_xticks([0, 2, 4, 6, 8])
    px.set_xticklabels(["0", "2", "4", "6", "8 s"])
    px.grid(axis="x", visible=False)

    if with_scatter:
        sx = fig.add_subplot(gs[:, 1])
        for name in ("none", "T", "P", "QRS"):
            sel = majority == CLASSES.index(name)
            sx.scatter(pcs[..., a][sel], pcs[..., b][sel], s=7, color=CLASS_COLORS[name], alpha=0.75 if name != "none" else 0.45, lw=0, label=name, zorder=2 if name != "none" else 1)
        sx.set_xlabel(f"PC {a + 1}", fontsize=11)
        sx.set_ylabel(f"PC {b + 1}", fontsize=11)
        sx.tick_params(labelsize=11)
        sx.set_title(f"The {int(ok.sum())} labelled blocks of the 12 leads, layer {layer}", loc="left", fontsize=13, color=INK)
        handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLORS[n], markersize=6, label=n) for n in ("P", "QRS", "T", "none")]
        sx.legend(handles=handles, loc="upper right", fontsize=11, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    path = OUT / ("fig_states_scatter.png" if with_scatter else "fig_states.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"class R2 of PC1-5 over all blocks: {np.round(r2, 2)}; plotted PC {a + 1} (time) and PC {a + 1} vs PC {b + 1}; explained {np.round(data['explained'], 3)}")
    return path


if __name__ == "__main__":
    import sys

    print("wrote", preprocess_steps())
    if "--states" in sys.argv:  # path to a downloaded /vol/states/<revision>/115.npy
        print("wrote", states_projection(Path(sys.argv[sys.argv.index("--states") + 1])))
    if STATES_CACHE.exists():
        print("wrote", states_figure())
