"""Illustrations for the step-1 deck that no report produces (docs/slides-outline.md).

    uv run python scripts/deck_figures.py

Writes results/deck/fig_preprocess_steps.png: steps 2 to 5 of `preprocess.prepare` on a real lead with
visible baseline wander (record 91, lead II), and the same beat filtered one way against both ways. The
padded two-way filtering below reproduces `preprocess.highpass` exactly (checked at run time).

Writes results/deck/fig_decoder_steps.png: one real beat through the decoder for slide 24 (the linear
probe's class map on record 22, lead III, from the M3 run), its clean-up and the decoded points.
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


def decoder_steps(record: str = "22", lead: str = "iii", model: str = "probe", start: int = 1440, stop: int = 1760) -> Path:
    """Slide 24: one real beat through the decoder. The class map is the linear probe's on a fold-0
    validation person from the M3 run (results/m3/m3_fold0_seed0_classes.npz). Row 1: the raw runs of
    each class; row 2: the runs after `decode.clean`; row 3: the decoded onsets, peaks and offsets on
    the filtered lead, with the cardiologists' marks below."""
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from ecgdelin.decode import JOIN_GAP_MS, MIN_MS, clean, runs, waves_from_classes
    from ecgdelin.preprocess import prepare
    from ecgdelin.reporting import WAVE

    maps = np.load(REPO / "results" / "m3" / "m3_fold0_seed0_classes.npz")
    k = LEADS.index(lead)
    rec = load_record(ludb_root(), record)
    fs = rec.fs
    signal = prepare(rec.signal, fs)[k].astype(np.float64)
    classes = maps[model][list(maps["ids"]).index(record), k]

    def inside(s: int, e: int) -> bool:
        return e >= start and s < stop

    # The clean-up step by step, mirroring decode.clean, to know what was joined and what was dropped.
    gap = int(round(JOIN_GAP_MS * fs / 1000))
    shortest = {kind: int(round(ms * fs / 1000)) for kind, ms in MIN_MS.items()}
    raw = runs(classes)
    joined: list[tuple[str, int, int]] = []
    gaps: list[list[tuple[int, int]]] = []
    for kind, s, e in raw:
        if joined and joined[-1][0] == kind and s - joined[-1][2] - 1 < gap:
            gaps[-1].append((joined[-1][2] + 1, s - 1))
            joined[-1] = (kind, joined[-1][1], e)
        else:
            joined.append((kind, s, e))
            gaps.append([])
    kept = [(kd, s, e, g) for (kd, s, e), g in zip(joined, gaps) if e - s + 1 >= shortest[kd] and inside(s, e)]
    dropped = [(kd, s, e) for kd, s, e in joined if e - s + 1 < shortest[kd] and inside(s, e)]
    assert [(kd, s, e) for kd, s, e, _ in kept] == [r for r in clean(raw, fs) if inside(r[1], r[2])]
    waves = [w for w in waves_from_classes(classes, signal, fs) if inside(w.onset, w.offset)]
    marks = [m for m in rec.marks[lead] if start <= m.peak < stop]
    errors_ms = []
    for w in waves:
        m = min((m for m in marks if m.kind == w.kind), key=lambda m: abs(m.peak - w.peak))
        errors_ms += [abs(a - b) * 1000 / fs for a, b in ((w.onset, m.onset), (w.peak, m.peak), (w.offset, m.offset)) if b is not None]

    fig = plt.figure(figsize=(11.0, 4.24), dpi=150)
    left, width = 20 / 1100, 1060 / 1100

    def frac(px: float) -> float:  # pixels from the top of the 1100 x 424 slide box -> figure fraction
        return 1 - px / 424

    def band(top_px: float, regions: list[tuple[str, int, int]]) -> Axes:
        ax = fig.add_axes((left, frac(top_px + 24), width, 24 / 424))
        ax.set_xlim(start, stop)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((start, 0), stop - start, 1, facecolor="white", edgecolor=GRID, linewidth=0.8))
        for kind, s, e in regions:
            if inside(s, e):
                ax.add_patch(Rectangle((max(s, start), 0), min(e + 1, stop) - max(s, start), 1, facecolor=WAVE[kind], edgecolor="none"))
        return ax

    ax1 = band(28, raw)
    ax2 = band(118, [(kd, s, e) for kd, s, e, _ in kept])
    fig.text(left, frac(21), "1  Model output: one class per sample", fontsize=11, fontweight="bold", color=INK)
    fig.text(left, frac(111), "2  Clean-up: pieces under 10 ms apart joined; P and QRS under 20 ms, T under 40 ms dropped", fontsize=11, fontweight="bold", color=INK)
    fig.text(left, frac(175), "3  Points on the filtered signal", fontsize=11, fontweight="bold", color=INK)

    # What the clean-up did, marked on row 1.
    one_gap = sorted({b - a + 1 for _, _, _, g in kept if len(g) == 1 for a, b in g})
    for kind, s, e, g in kept:
        if len(g) >= 2:
            ax1.plot([s, s, e + 1, e + 1], [-0.12, -0.32, -0.32, -0.12], color=INK2, linewidth=0.9, clip_on=False)
            ax1.text((s + e + 1) / 2, -1.05, f"{len(g) + 1} {kind} pieces, joined", ha="center", va="center", fontsize=10, color=INK)
        elif len(g) == 1:
            a, b = g[0]
            ax1.text((a + b + 1) / 2, 1.42, "▼", ha="center", va="center", fontsize=8, color=INK)
    if one_gap:
        size = one_gap[0] if len(one_gap) == 1 else f"{one_gap[0]}–{one_gap[-1]}"
        fig.text(left + width, frac(21), f"▼ {size}-sample gaps, joined", ha="right", fontsize=10, color=INK)
    if dropped:
        kinds = sorted({kd for kd, _, _ in dropped})
        s0, e0 = min(s for _, s, _ in dropped), max(e for _, _, e in dropped)
        ax1.plot([s0, s0, e0 + 1, e0 + 1], [-0.12, -0.32, -0.32, -0.12], color=INK2, linewidth=0.9, clip_on=False)
        ax1.text((s0 + e0 + 1) / 2, -1.05, f"{len(dropped)} {'/'.join(kinds)} blips, dropped", ha="center", va="center", fontsize=10, color=INK)

    ax3 = fig.add_axes((left, frac(392), width, (392 - 186) / 424))
    t = np.arange(start, stop)
    lo, hi = signal[start:stop].min(), signal[start:stop].max()
    pad = 0.12 * (hi - lo)
    ax3.set_xlim(start, stop)
    ax3.set_ylim(lo - 1.6 * pad, hi + 1.5 * pad)
    for w in waves:
        ax3.axvspan(w.onset, w.offset + 1, color=WAVE[w.kind], alpha=0.12, linewidth=0)
        for edge in (w.onset + 0.5, w.offset + 0.5):
            ax3.axvline(edge, color=WAVE[w.kind], linewidth=1.6)
        ax3.text((w.onset + w.offset + 1) / 2, hi + 1.0 * pad, w.kind, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    ax3.plot(t + 0.5, signal[start:stop], color=INK, linewidth=1.4, zorder=3)
    for w in waves:
        ax3.scatter([w.peak + 0.5], [signal[w.peak]], s=48, color=WAVE[w.kind], edgecolor=SURFACE, linewidth=1.5, zorder=5)
    for m in marks:
        for point in (m.onset, m.peak, m.offset):
            if point is not None:
                ax3.plot([point + 0.5], [lo - 1.25 * pad], marker="^", markersize=6, color=INK, linestyle="none", zorder=4)
    ticks = np.arange(np.ceil(start / fs * 10) / 10, stop / fs, 0.1)
    ax3.set_xticks(ticks * fs)
    ax3.set_xticklabels([f"{x:.1f}" for x in ticks[:-1]] + [f"{ticks[-1]:.1f} s"], fontsize=9.5)
    ax3.set_yticks([])
    ax3.grid(False)
    for side in ("left", "top", "right"):
        ax3.spines[side].set_visible(False)
    ax3.tick_params(axis="x", length=0, pad=4)
    handles = [
        Line2D([], [], marker="^", color=INK, linestyle="none", markersize=6, label="cardiologists' marks"),
        Line2D([], [], marker="o", color=INK2, markeredgecolor=SURFACE, linestyle="none", markersize=7, label="decoded peak"),
        Line2D([], [], color=INK2, linewidth=1.6, label="decoded onset and offset"),
    ]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(left + width, frac(186)), ncol=3, fontsize=9.5, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_decoder_steps.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"decoder example: largest gap between a decoded point and its mark {max(errors_ms):.0f} ms")
    return path


if __name__ == "__main__":
    import sys

    print("wrote", preprocess_steps())
    print("wrote", decoder_steps())
    if "--states" in sys.argv:  # path to a downloaded /vol/states/<revision>/115.npy
        print("wrote", states_projection(Path(sys.argv[sys.argv.index("--states") + 1])))
    if STATES_CACHE.exists():
        print("wrote", states_figure())
