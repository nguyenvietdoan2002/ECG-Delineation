# Slide outline: "Frozen TimesFM 3 states for ECG wave delineation: what we built and what we found"

Brief for the deck, agreed on 2026-10-11: a 60-minute live talk in English, now 35 slides (36 agreed;
on 2026-10-12 slides 17 to 19 were added, absorbing the old `models.py` slide, and the tests, rigor
and M3 deep-dive slides were removed), for
Meddies teammates who know machine learning but not TimesFM 3, LUDB or ECG delineation. Thematic order:
question, grounding, pipeline, data, processing, models, baseline, training, decoding, evaluation,
results, limits, process, close. Every source file and script is walked in the main flow. Terms are
defined on the slide where they first matter. Figures are the repository's PNGs. Code snippets are copied
from commit `e107f87` (tag `step1-frozen` plus the post-freeze launch tooling and results).

Numbers: every result number comes from `docs/report.md` and `results/m4/summary.json` (five folds, three
seeds, 200 people). M3 development figures and numbers appear only where labelled development-only (slides 6, 22, 24 and 29 to
31). Speaker notes are
what the presenter says; bullets are what the slide shows.

Designed on 2026-10-11 as a claude.ai Design canvas, one 1920 x 1080 artboard per slide in the agreed
look (Option A, Clinical ledger: the figures' light surface, IBM Plex Sans and Mono, one blue accent):
https://claude.ai/artifact/EBBfxAa8RPGVTWedYDWg4t (private until shared from the page).

Visual assets to reuse (all PNG, 150 dpi, light surface):

| File | Used on |
|---|---|
| `results/m3/fig_overlay_typical.png` | 6, 31 |
| `results/m3/fig_overlay_worst.png`, `results/m3/fig_overlay_af.png` | 31 |
| `results/m3/fig_training.png` | 22 |
| `results/m3/fig_per_lead.png` | 29 |
| `results/m3/fig_layer_weights.png` | 30 |
| `results/m4/fig_points.png` | 29 |
| `results/m4/fig_folds.png` | 28 |

New diagrams to draw (described where they appear): D1 pipeline (slide 9), D2 state extraction (slide 8),
D3 three architectures (slide 16), D4 scoring chain (slides 24 to 26), D5 annotated beat (slide 4),
D6 to D8 one architecture each, with its code (slides 17 to 19).

---

## 1. Title

**Visual:** title, subtitle "Step 1 of the TimesFM delineation study, LUDB", date, presenter.

**Bullets:**
- Frozen TimesFM 3 states for ECG wave delineation: what we built and what we found
- A pre-registered three-way comparison on LUDB: TimesFM head, U-Net, NeuroKit2

**Notes:** This talk walks through everything built for step 1, from the raw LUDB files to the stop-rule
verdict, file by file. You do not need to know ECGs or TimesFM; the first slides give the grounding. The
whole thing is reproducible from the repository, and I will end with how to run it.

## 2. The question and the roadmap

**Visual:** one sentence in large type; a roadmap strip of the sections.

**Bullets:**
- Question: can the frozen internal states of a general time-series forecaster, TimesFM 3, delineate P, QRS
  and T waves better than a U-Net trained on the same data, and better than the rule-based NeuroKit2?
- Why it matters: foundation models are cheap to reuse; a positive answer means their features carry
  sample-level cardiac morphology without any label-specific training of the backbone
- Roadmap: ECG and LUDB grounding, TimesFM 3, the pipeline, data, processing, models, training, decoding,
  evaluation, results, limits, process

**Notes:** The question was fixed before any model was trained, together with the metric and the rule that
decides the answer. Keep this in mind: nothing you will see was chosen after looking at a test result. The
roadmap follows the data through the code, so each section names the files it lives in.

## 3. ECG waves and intervals

**Visual:** a schematic beat with P, QRS, T labelled and the PR, QRS and QT intervals as brackets.

**Bullets:**
- One heartbeat on an ECG: the P wave (atria contract), the QRS complex (ventricles contract), the T wave
  (ventricles recover)
- Each wave has an onset, a peak and an offset; intervals between them (PR, QRS duration, QT) are what
  clinicians measure and what drug-safety and arrhythmia work depends on
- 12 leads look at the same heart from 12 angles; a wave can look different, or be invisible, in some leads

**Notes:** Delineation means placing those boundaries, sample by sample, in every lead. A few milliseconds
matter: QT prolongation thresholds are tens of milliseconds, and a 500 Hz signal has a sample every 2 ms.
The 12 leads are the reason multi-lead models have an advantage: a boundary that is unclear in one lead is
often clear in another.

## 4. Delineation as a task

**Visual:** D5, a real LUDB beat (record 115, lead II) with the nine points marked and the five CSE points
highlighted, with the CSE 2σ tolerances in a small table.

**Bullets:**
- Nine points per beat: onset, peak, offset of P, QRS, T; LUDB has no separate Q or S marks
- Five of them are the "CSE points" that define the intervals: P onset, P offset, QRS onset, QRS offset,
  T offset; the CSE Working Party (1985) set 2σ tolerances of 10.2, 12.7, 6.5, 11.6 and 30.6 ms
- Two families of methods: rule-based (NeuroKit2's `prominence`, no training) and learned (U-Nets trained
  on LUDB); error SDs in the literature are 6 to 17 ms for the rule-based method, 10 to 30 ms pooled for
  U-Nets
- Our output format is NeuroKit2's: eleven keys per lead (`ECG_P_Onsets` … `ECG_T_Offsets`), so any
  method is a drop-in replacement

**Notes:** CSE stands for Common Standards for quantitative Electrocardiography, a 1980s European project
in which a panel of cardiologists marked a reference set; the tolerances are twice the panel's own
spread. Every delineation paper since reports mean ± SD per point against them, and so do we. Note the
asymmetry: the QRS onset tolerance is 6.5 ms, three samples; the T offset tolerance is 30 ms, because even
cardiologists disagree about where a T wave ends.

## 5. LUDB

**Visual:** a fact card; small bar of the rhythm split (182 sinus, 18 AF/flutter).

**Bullets:**
- Lobachevsky University Database v1.0.1 (Kalyakulina et al., IEEE Access 2020; PhysioNet, ODC
  Attribution): 200 people, one 10 s 12-lead ECG each at 500 Hz, 5000 samples per lead
- Two cardiologists marked onset, peak and offset of every P, QRS and T, separately in each lead: 58,423
  waves (21,965 QRS, 19,661 T, 16,797 P)
- Rhythm from the header: 182 sinus, 18 atrial fibrillation or flutter (no P waves at all)
- Quirks we had to handle: only the span from a lead's first to its last mark is annotated (4.4 to 8.8 s
  of 10); 134 waves lack an onset or offset; 24 records have no P marked in lead II; peak marks sit one
  sample before the signal's extremum

**Notes:** Two hundred people is small, which is why the whole study uses person-level cross-validation
and confidence intervals from a bootstrap over people, never over beats. The unmarked first and last beats
are a trap: a model that finds a real P wave there must not be punished, so those samples are masked out
of both training and scoring. The one-sample peak offset was discovered in the M3 deep dive and is
explained on slide 29.

## 6. A real LUDB record

**Visual:** `results/m3/fig_overlay_typical.png` cropped to the signal and shaded marks (top panel; the
prediction bars can stay as a preview).

**Bullets:**
- Record 115, lead II: shaded spans are the cardiologists' P (blue), QRS (orange), T (green) regions
- The models predict exactly this: one of four labels per sample (none, P, QRS, T)
- The bars underneath preview what the four methods predicted on this record; we come back to them at the end

**Notes:** This is the whole learning problem in one picture: paint every sample with one of four colours,
then derive the points from the painted regions. Lead V1 in the lower panel shows why multi-lead matters:
the P wave is tiny there, yet the marks are the same points in time.

## 7. TimesFM 3: from a 64 ms block to a 1280-number state

Figure-only slide, revised on 2026-10-11. The architecture follows the TimesFM 3 code
(`timesfm3/torch/model.py` `_preprocess`, `dense.ResidualBlock`, `transformer.MixingTransformer`) and the
checkpoint's `config.json`, drawn in the left-to-right arrangement of Das et al. (ICML 2024) and the Google
Research post of February 2024; our own drawing, credited in the footer.

**Visual:** five numbered stages, left to right, each with its tensor shape (leads, blocks, numbers per
block; batch omitted):

1. Cut into blocks, `(12, 156, 32)`: a 12-lead strip of about 1 s cut into 32-sample tiles (64 ms), one
   block of lead II highlighted.
2. Block input, `(12, 156, 192)`: a bar of four segments, drawn to scale: 32 values normalised by the
   lead's running mean and SD over blocks 1 to k; a 64-sample future slot, all 0 and masked, because every
   lead is a target; 32 mask flags at 0; 64 mask flags at 1.
3. Embed, `(12, 156, 1280)`: the residual block, Linear 192 → 1280, ReLU, Linear 1280 → 1280, plus a
   linear skip 192 → 1280.
4. One layer, repeated 20 times, `20 × (12, 156, 1280)`: three sub-blocks in sequence, each RMSNorm, the
   operation, RMSNorm, then its input added back. Time attention (blue): each lead over its 156 blocks,
   causal, 16 heads × 80, rotary positions. Lead attention (aqua): the 12 leads at the same block, not
   causal, no positions. Feed-forward (orange): 1280 → 1280 → 1280, ReLU.
5. Forecast head, `(12, 156, 576)`: 1280 → 64 samples × 9 quantiles, greyed and dashed, "never used".

Below stage 4, an accent arrow, "forward hooks read the output of every one of the 20 layers", into the
states box: the 1280-vector after each layer for each block of each lead, 20 × 12 × 156 × 1280 per pass,
forward and reversed joined to 2560 on the next slide.

**Highlighted line** (a tinted band under the figure, arrows in the accent colour; it replaced the earlier
caption, whose definition of a state lives in the states box): 4992 samples per lead → 156 blocks of 32 →
192 numbers per block (96 values, 96 flags) → residual block → 1280 → 20 × [time attention, lead attention,
feed-forward].

The 96 flags follow TimesFM 3's own convention: 0 = a real value, 1 = a blank the model must not use. Its
forecaster sets 0 for observed context and for a covariate's known future, and 1 for padding and for the
future of the series being forecast; our extractor passes all-zero masks and marks every lead as a target,
so TimesFM's `_preprocess` itself writes the 1s in the 64-sample slot.

**Spec strip:** 1 block = 32 samples = 64 ms at 500 Hz · width 1280, 16 heads × 80 · RMSNorm before and
after every sub-block, residual adds · rotary positions on time, none across leads · feed-forward and
residual block: ReLU, no bias · forecast: 64 samples × quantiles 0.1 to 0.9.

**Footer:** After Das et al., ICML 2024, and Google Research, February 2024; drawn from the TimesFM 3 code
and checkpoint config · TimesFM licence: non-commercial, research only.

**Notes:** Walk the figure left to right with one block of lead II. Its 32 samples are normalised by
everything lead II has shown so far, then paired with a 64-sample slot. TimesFM forecasts 64 samples per
block, and the slot is how it accepts series whose future is known in advance, such as a calendar signal.
For a series it must forecast, the slot is blanked and flagged as masked; we mark every lead as a target,
so no future ever enters. Values and flags together are 192 numbers, and the residual block turns them into
a 1280-number token. Each of the 20 layers then lets the token look back along its own lead (causal: only
blocks 1 to k), across the other 11 leads at the same moment, and through a feed-forward block, each step
added on top of what was there. We read the token after every layer; that is a state. The forecast head on
the right is what TimesFM was trained for, and we never use it.

## 8. Two passes: every block gets both sides

Redrawn on 2026-10-11 as a picture with four short lines; the storage numbers live on slide 14.

**Visual:** a real beat, record 6, lead V3, about 1.2 s cut into 64 ms blocks, with LUDB's T wave shaded.
Block k (block 84) is highlighted; LUDB's T offset falls at sample 24 of its 32, marked by a dashed line,
with a bracket over the flat stretch that follows. A blue arrow runs from the left edge to the end of block
k: "Forward pass: block k is built from blocks 1 to k", annotated "sees only 14 ms past the T offset". An
orange arrow runs from the right edge to the start of block k: "Reversed pass, flipped back: block k is
built from blocks k to 156". Under block k, a bar of two halves, forward 1280 and reversed 1280, "= 2560":
the joined state of block k, per layer.

**Text (two columns):**

*Why the forward pass alone is not enough*
- TimesFM only looks backwards: the state of block k is built from blocks 1 to k.
- A T wave's offset is where the signal turns flat after it. The forward state never sees that far, only up
  to 62 ms ahead inside its own block.

*What the reversed pass adds*
- The same 10 s fed backwards: everything after block k becomes its past.
- Flipped back into time order and joined to the forward states, each block gets both sides:
  1280 + 1280 = 2560 numbers.

**Footer:** Verified on a T4: later samples never change earlier forward states · its benefit was not
measured in step 1 (decision 61) · Record 6, lead V3, LUDB marks.

**Notes:** Every boundary is found by comparing what comes before it with what comes after it, and three of
the five CSE points are offsets. In this beat the T offset sits late in block k, so the forward state of
that block has seen only 14 ms of the flat signal that defines the offset. Running the same 10 s backwards
turns everything after block k into its past; flipping those states back and joining them gives every block
both sides. Two caveats if asked: the conv head already reads three blocks ahead, so even forward-only states
give it some lookahead; the reversed pass gives every block the whole future, processed by all 20 layers
and across the 12 leads, which matters most for the one-block probe. How much it helps was not measured in
step 1: the forward-only run was deferred.

## 9. The pipeline and the repository

**Visual:** D1: LUDB files → `dataset` → `labels` and `splits` → `preprocess` → `states` (TimesFM on Modal
GPU) → `models` + `train` → class maps → `decode` → NeuroKit2-style points → `score` and `crossval` →
report and verdict; NeuroKit2 enters at the points stage. Underneath, the repository tree.

**Bullets:**
- `src/ecgdelin/`: 18 modules, each one stage of the pipeline behind a small interface
- `scripts/`: 10 Modal or local entry points, one per milestone step
- `tests/`: 14 test files on two seams (delineator and scorer), plus synthetic ECGs
- `splits/`, `results/` (JSONs, predictions, figures), `docs/` (plan, spec, decision log, reports, paper)
- Everything that touches a GPU or all 200 records runs on Modal; the laptop runs unit tests

**Notes:** Two interfaces hold the code together. A delineator takes a 12 × 5000 ECG in millivolts and
returns NeuroKit2-style points per lead; every method, trained or not, sits behind it. A scorer takes
points and LUDB marks and returns every metric with intervals. Both are tested with hand-built cases, so
the rest can be refactored freely. The milestones were M0 foundations, M1 scorer and NeuroKit2, M2 states,
M3 fold-0 models, M4 final runs.

## 10. `dataset.py`: reading LUDB

**Main function (highlighted band under the title):** `load_record(root, id) → Recording`: reads one LUDB
record, its 12 leads in millivolts, the rhythm from the header, and every cardiologist mark, lead by lead.
The bullet that repeated it was removed.

**Visual:** the `Recording` dataclass fields and a `Wave` example.

**Bullets:**
- `load_record(root, id)` → `Recording(id, fs, signal (12, 5000) in mV, rhythm, marks)`; leads selected by
  name in a fixed order (I, II, III, aVR, aVL, aVF, V1 to V6), never by position
- `waves_from_annotations` groups the WFDB symbols `(`, `p`/`N`/`t`, `)` into `Wave(kind, onset, peak,
  offset)`; a missing `(` or `)` becomes `None`
- `rhythm` is read from the header's `Rhythm:` line
- Reads LUDB in place from the existing download; `paths.py` resolves the location (env overrides for Modal)

**Notes:** The one design rule here is "select by name": the WFDB files list leads in their own order, and a
silent lead swap would corrupt every multi-lead model without any error. Incomplete waves are kept as data
with `None` sides, and it is the labels module that decides what to do with them.

## 11. `labels.py`: from marks to training targets

**Main function (highlighted band under the title):** `labels_for(waves, n) → Labels(classes, mask)`: turns
the cardiologists' marks of one lead into one label per sample. Every sample from a wave's onset to its
offset gets that wave's class, the rest is none, and a mask marks the samples that count. The bullet that
repeated it was removed.

**Real example (box under the strip):** record 115, lead II: the first QRS is marked from sample 913 to 970,
so those 58 samples, 116 ms, are labelled QRS. Only samples 841 to 4384 count, from the first mark to the
last; the other 1456 of 5000 are masked.

**Visual:** a strip of samples coloured by class with the mask shown as a grey band, including an
incomplete wave's masked gap.

**Bullets:**
- `labels_for(waves, n) → Labels(classes, mask)`: one class per sample (none, P, QRS, T), a wave covering
  onset..offset inclusive
- Mask = usable samples: from the lead's first mark to its last (the unmarked edges never count)
- Incomplete wave (peak without onset or offset): the whole gap between its complete neighbours is masked,
  because its true extent is unknown (decision 35)
- Training: masked samples get target −1 and are ignored by the loss; scoring: predictions outside the
  mask are never false positives

**Notes:** Both models learn from exactly these targets, and the scorer uses the same mask, so the two
trained models and NeuroKit2 are judged on the same samples. The round trip, marks to labels to decoder to
points, reproduces the marked onsets and offsets exactly; that is one of the tests.

## 12. `splits.py` and `make_splits.py`: who is trained on, who is tested

**Visual:** a 200-person grid coloured by fold, with the validation slice and the test fold called out.

**Bullets:**
- Five folds by person, stratified by rhythm group (sinus / AF-flutter / other) so every test fold holds
  3 or 4 AF people; 40 test people per fold
- Within each training fold, 20% of its people (32) are the validation set, used only for early stopping;
  128 train
- Nested 25% and 50% subsets were saved for a learning curve (not run in step 1)
- Generated once with seed 0 and committed as `splits/ludb_5fold_seed0.json`; experiments read the file,
  never regenerate it

**Notes:** Person-level splitting is non-negotiable with 10 s records: beats of one person are near copies
of each other, so a beat-level split leaks. The test fold of a container is never read before its training
ends; the trainer loads only train and validation people into memory and streams the test people
afterwards. Development, every choice made while building, used fold 0's validation people only.

## 13. `preprocess.py`: the preprocessing pipeline

Redrawn on 2026-10-11 as a highlighted main function plus a pipeline with a reference per step. The order
follows the code: `prepare` filters the full 5000 samples first and crops to 4992 afterwards (the earlier
slide said the opposite).

**Main function (highlighted band):** `prepare(signal, fs) → float32 (12, 4992)`: removes baseline drift from
all 12 leads with a 0.5 Hz zero-phase high-pass over the full 10 s, then keeps the first 4992 samples. Both
trained models get exactly this input.

**Pipeline, shared steps** (steps 2 to 5 on a band labelled `preprocess.py · prepare(signal, fs)`; short
author-year references inside each box):

1. Read in millivolts (`dataset.py`): 12 × 5000 samples at 500 Hz. Goldberger 2000 · Kalyakulina 2020.
2. Mirror padding: 6 s mirrored onto each end, 10 s → 22 s (`padtype="even"`, `padlen=3000`, so 3000 +
   5000 + 3000 samples). SciPy, Virtanen 2020 · decision 51.
3. Butterworth high-pass: 0.5 Hz, order 5, second-order sections; removes baseline wander. Butterworth 1930 ·
   same cutoff and order as NeuroKit2's default cleaning.
4. Same filter, backwards (`sosfiltfilt`): the second pass runs over the time-reversed output, which cancels
   the first pass's delay exactly: zero phase. Kormylo & Jain 1974 · Kligfield 2007 (up to 0.67 Hz is
   acceptable for zero-phase digital filters).
5. Drop padding, crop: 22 s → 10 s → the first 4992 samples (156 × 32; the last 8 samples, 16 ms, are
   dropped). Das 2024 · decision 24.

**Figure under the pipeline** (`results/deck/fig_preprocess_steps.png`, made by `scripts/deck_figures.py`;
the script checks that its padded two-way filtering reproduces `preprocess.highpass` exactly): left, record
91, lead II, padded to 22 s with the mirrored 6 s hatched at each end, the raw trace drifting by about
0.7 mV and the trace after steps 3 and 4 flat, with "step 5 keeps 0 to 9.98 s: the first 4992 samples";
right, "Why step 4: one way pulls ST and T down": one beat filtered one way only (orange) against both ways
(black), with the cardiologists' QRS and T marks shaded.

**Branches:** the shared output feeds both trained models ("the same input for both trained models").

- NeuroKit2 only (from step 1, raw 5000 samples): its own `ecg_clean`, a 0.5 Hz Butterworth high-pass of
  order 5, then a 50 Hz powerline filter (a moving average over one 50 Hz period). Makowski et al. 2021.
- U-Net only: each lead scaled to mean 0, SD 1 over its 4992 samples, inside the network; the operation of
  instance normalisation, Ulyanov et al. 2016; decision 55.
- TimesFM: cut into 12 × 156 blocks of 32 (patching: Nie et al. 2023; Das et al. 2024), then each block
  scaled by its lead's running mean and SD over blocks 1 to k, plus mask flags, 192 numbers (RevIN, Kim et
  al. 2022, applied causally), → TimesFM 3 states (slides 7 and 8).

**Footer:** tests/test_preprocess.py: drift is removed, a 10 Hz wave passes unchanged and undelayed,
filtering comes before the crop · no resampling, no augmentation · figure: scripts/deck_figures.py. The
strip of full references at the bottom of the slide was removed on request; they stay listed here.

**References** (for the speaker, not on the slide):
- Butterworth S. On the theory of filter amplifiers. Experimental Wireless and the Wireless Engineer
  7:536–541, 1930.
- Kormylo JJ, Jain VK. Two-pass recursive digital filter with zero phase shift. IEEE Transactions on
  Acoustics, Speech, and Signal Processing 22(5):384–387, 1974. DOI 10.1109/TASSP.1974.1162602.
- Kligfield P et al. Recommendations for the standardization and interpretation of the electrocardiogram,
  part I. Circulation 115:1306–1324, 2007.
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215–e220, 2000.
- Kalyakulina AI et al. LUDB: a new open-access validation tool for electrocardiogram delineation
  algorithms. IEEE Access 8:186181–186190, 2020.
- Virtanen P et al. SciPy 1.0. Nature Methods 17:261–272, 2020.
- Makowski D et al. NeuroKit2: a Python toolbox for neurophysiological signal processing. Behavior Research
  Methods 53:1689–1696, 2021.
- Ulyanov D, Vedaldi A, Lempitsky V. Instance normalization: the missing ingredient for fast stylization.
  arXiv:1607.08022, 2016.
- Kim T et al. Reversible instance normalization for accurate time-series forecasting against distribution
  shift. ICLR 2022.
- Nie Y et al. A time series is worth 64 words: long-term forecasting with transformers. ICLR 2023.
- Das A et al. A decoder-only foundation model for time-series forecasting. ICML 2024.

**Notes:** Walk the top row: the record is read in millivolts, mirrored at both ends, high-passed at 0.5 Hz,
filtered a second time backwards so nothing shifts in time, then cropped to 156 blocks. The 0.5 Hz cutoff is
what NeuroKit2 uses, and the AHA recommendations allow up to 0.67 Hz for zero-phase digital filters, so
the ST segment and wave edges are not distorted. Then the branches: NeuroKit2 never sees our pipeline, it
cleans the raw signal itself; the U-Net only adds a per-lead scaling; TimesFM cuts and normalises inside its
own code. The two trained models therefore see exactly the same signal.

## 14. `states.py` and `extract_states.py`: the extractor

Redrawn on 2026-10-11: a highlighted main function, three short cards instead of five bullets, and a figure
from real stored states.

**Main function (highlighted band):** `extract_states(signal, backbone) → float16 (20, 12, 156, 2560)`: the
step computes the hidden states of every ECG through frozen TimesFM 3 and stores them once, so the probe and
the head can train on them without running TimesFM again.

**Cards:**
- `pass_states`: one pass over the 156 blocks; hooks copy all 20 layers' outputs.
- `extract_states`: forward and reversed passes, joined: 1280 + 1280 = 2560 per block, float16.
- `scripts/extract_states.py`: 200 records on one T4, 1.9 s each (about 13 minutes in all); 183 MB per
  record, 36.6 GB in all; it resumes after a stop.

**Figure** (`results/deck/fig_states.png`, made by `scripts/deck_figures.py` from
`results/deck/states_pca_115_layer5.npz`, a 5-component projection computed once from the stored states of
record 115 with `--states <downloaded 115.npy>`), 1110 px wide: top, record 115, lead II, the ECG with the
cardiologists' P, QRS and T marks shaded; bottom, frozen TimesFM's layer-5 state of each 64 ms block
squeezed to one number (principal component 1), drawn as 156 flat steps because TimesFM gives one state per
32-sample block, with no shading, only dashed lines at the cardiologists' QRS marks and "unmarked beat" at
the first and last beats. Principal component 1 alone accounts for 73% of the variance between the four
block types over all 12 leads (82% in lead II); layer 5 is also the layer the linear probe weighted most.
The earlier scatter of all labelled blocks is still available as `fig_states_scatter.png`
(`states_figure(with_scatter=True)`) but is no longer on the slide.

**Chain diagram beside the figure** (560 px, editable boxes), headed "How the line on the left is made":
12-lead ECG, 4992 samples per lead, filtered (slide 13) → Frozen TimesFM 3, forward and time-reversed
passes, nothing trained → Layer-5 states, per lead 156 blocks × 2560 numbers, stored once → Principal
component analysis, for this figure only: 2560 numbers → 1 per block, no labels, the line on the left
(dashed box).

**Caption:** What a stored state holds: one number per 64 ms block from frozen TimesFM's layer-5 states,
computed with no labels, jumps at every QRS, even at the first and last beats, which the cardiologists left
unmarked. The shading and dashed lines are the cardiologists' marks, added only for comparison.

**Footer:** tests: stub backbone on the laptop; the real model on a T4 checks shapes, alignment and
causality · checkpoint revision 43046b85 · decisions 50 to 53, 63 · figure: scripts/deck_figures.py.

**Notes:** The extractor is two functions and a batch job. Show the figure: nothing in it was trained. A
plain principal-component projection of the layer-5 states already rises at every QRS and separates the
four kinds of block, which is why a single linear layer, the probe, can already match the U-Net. One record,
one layer: an illustration, not a result.

## 15. `modal_app.py`, `paths.py`, `smoke.py`, `gpu_tests.py`: the compute

**Visual:** a box diagram: laptop (tests) → Modal image (pinned versions) → CPU functions / T4 functions →
one Volume (`ludb/`, `hf/`, `states/`, `checkpoints/`, `models/`, `results/`, `predictions/`).

**Bullets:**
- One Modal app and one image: Python 3.12, PyTorch 2.14.1 with CUDA 13, NeuroKit2 0.2.13; `ecgdelin` and
  `timesfm3` copied in at run time, so Modal always runs the working tree
- One Volume holds LUDB, the TimesFM weights, states, checkpoints, trained weights, predictions and result
  JSONs; predictions and JSONs are copied into the repository, weights stay (licence)
- `smoke.py` checks the image and Volume on CPU and T4; `gpu_tests.py` runs the `gpu`-marked tests inside a
  T4 container with the real model
- `paths.py`: sibling-folder defaults, env overrides inside containers
- Cost of the whole step: on the order of ten dollars of credit

**Notes:** The laptop never loads the backbone; it runs unit tests on CPU PyTorch. The same test files run
unchanged inside the container. Keeping everything in one Volume is what makes the fold containers simple:
a fold's training states are 28.6 GB, loaded into RAM once in about 35 s with eight threads.

## 16. Three models, one output

**Visual:** D3: side by side, the linear probe, the conv head and the U-Net, each from its input to
"(4 classes, 12 leads, 4992 samples)"; parameter counts; receptive fields.

**Bullets:**
- All three map a recording to four class scores per sample and lead; one loss, one decoder, one scorer
- Linear probe (0.33M): layer blend → one linear layer 2560 → 32 samples × 4 classes; sees one block, ±0 ms
- Main head (0.49M): layer blend → linear 2560 → 128 + ReLU → three width-3 convolutions along the block
  axis → linear 128 → 32 × 4; sees ±3 blocks, about ±190 ms; shared across leads
- U-Net control (1.86M): 12 high-passed leads in, 5 levels of two Conv-BN-ReLU blocks, kernel 9, channels
  16 to 256, max-pool and transposed-conv, skip connections, 1 × 1 output; receptive field about ±0.75 s
- The head is deliberately smaller than the U-Net, so a win cannot be put down to size

**Notes:** The probe answers "how much do the states hold on their own"; the head gives TimesFM a fair
chance at exact boundaries with a little context and non-linearity; the U-Net answers "is it TimesFM, or
just training on LUDB". The "32 × 4" at the end of the heads is the trick that turns a per-block feature
into per-sample labels: the last layer emits four scores for each of the 32 samples of the block.

## 17. Linear probe: architecture and code

Added on 2026-10-12 so the audience can check each architecture against its code; the old slide on
`models.py` (normalisation, the layer blend, `_to_scores`) was folded into slides 17 to 19 the same day.
The three slides share one layout: a header band with the main class and its learned-parameter count;
the architecture as a pipeline of boxes with real shapes, per-box counts and the code lines each box
comes from, learned boxes outlined in the model's colour; and on the right a numbered snapshot of
`src/ecgdelin/models.py` at tag `step1-frozen` (commit `2f3afc5`; the file is unchanged since). A "⋮"
row marks lines shown on a sibling slide.

**Visual:** D6: six boxes top to bottom, learned ones in green. Stored TimesFM states (20 layers,
12 leads, 156 blocks, 2560), float16 → `normalize_layers`, nothing learned, lines 24–29: each layer's
2560-vector to mean 0 and variance 1 in fp16, run once per batch by the trainer (`normalized=True`)
→ `LayerBlend`, 20 learned weights, lines 50–64: softmax over 20 logits, then a weighted sum of the
layers → (12, 156, 2560), ELMo's scalar mix, the sum computed in `_Blend` (slide 18) → linear
2560 → 128, 327,808 learned, line 79 → `_to_scores`, a fixed reshape, lines 67–70: 128 = 32 samples
× 4 classes, blocks laid end to end → (4, 12, 4992) → argmax over the 4 classes, fixed, outside the
model → (12, 4992). Code: lines 20–29 (`count_parameters`, `normalize_layers`) and 50–82
(`LayerBlend`, `_to_scores`, `LinearProbe`); lines 30–49 (`_Blend`) are on slide 18.

**Header band:** `LinearProbe(n_layers=20, d_in=2560)`: 327,828 learned numbers, as `count_parameters`
reports. It reads one 64 ms block's stored state at a time.

**Snippet S1, `src/ecgdelin/models.py` (the slide shows the exact lines):**

```python
def normalize_layers(states: Tensor) -> Tensor:
    """(b, L, v, n, d) -> fp16 copy with every layer's d-vector at zero mean and unit variance (decision 56)."""
    out = torch.empty(states.shape, dtype=torch.float16, device=states.device)
    for layer in range(states.shape[1]):
        out[:, layer] = F.layer_norm(states[:, layer].float(), (states.shape[-1],)).half()
    return out


class LayerBlend(nn.Module):
    """Softmax-weighted sum over the layers of per-layer-normalised states."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, states: Tensor, normalized: bool = False) -> Tensor:
        out: Tensor = _Blend.apply(torch.softmax(self.logits, 0), states if normalized else normalize_layers(states))
        return out
```

**Notes:** The counts are there to be checked: 20 blend logits, then 2560 × 128 weights and 128 biases,
327,828 in all; every count on slides 17 to 19 is what `count_parameters` returns. The normalisation
comes first because residual-stream magnitudes grow with depth (the largest absolute state is 1080):
without it the deep layers would dominate the mix whatever the weights said, and the weights would no
longer read as importance (decision 56). It has no learned affine, so it adds nothing to the counts.
The trainer normalises each batch once and passes `normalized=True` to both state models; a test checks
that the result is identical. The probe never sees a neighbouring block; whatever it knows about context
comes from TimesFM's own attention inside the stored states.

## 18. Main head: architecture and code

**Visual:** D7: eight boxes, learned ones in blue. Stored states → `normalize_layers` (slide 17) →
`LayerBlend`, 20 learned (slide 17), whose sum runs in `_Blend`, lines 32–47, a custom autograd
function → linear 2560 → 128 and ReLU (327,808, line 92) → (12, 156, 128) → three Conv1d of width 3
with ReLU, 128 → 128 along each lead's blocks (147,840, line 93; ±3 blocks, about ±190 ms) → linear
128 → 128 (16,512, line 94) → `_to_scores` (slide 17) → (4, 12, 4992) → argmax, outside the model
→ (12, 4992). Code: lines 32–47 (`_Blend`) and 85–103 (`ConvHead`); lines 48–84 are on slide 17.

**Header band:** `ConvHead(n_layers=20, d_in=2560, width=128)`: 492,180 learned numbers. The probe's
path plus three convolutions, so each block sees 3 neighbours on each side.

**Notes:** `_Blend` exists for memory. A batch of eight recordings' states is 1.5 GB in fp16; a plain
weighted sum in PyTorch would build a 3.1 GB fp32 copy and keep it for the backward pass. `_Blend` adds
the layers into an fp32 output one at a time and saves only the fp16 input, from which the gradient of
the 20 weights is computed; the states themselves need no gradient. The convolutions run along the
block axis of one lead at a time, with the same weights for all 12 leads: lines 98 and 99 fold the leads
into the batch before the convolutions and unfold them after. 20 + 327,808 + 3 × 49,280 + 16,512 =
492,180.

## 19. U-Net: architecture and code

**Visual:** D8: the U drawn as boxes, learned ones in orange. Input 12 × 4992, each lead standardised
→ encoder ConvBlocks 12 → 16, 16 → 32, 32 → 64, 64 → 128 at lengths 4992, 2496, 1248, 624, a
max-pool (÷2) between them → bottleneck ConvBlock 128 → 256 at length 312 = 4992 ÷ 16 → decoder
ConvBlocks 256 → 128, 128 → 64, 64 → 32, 32 → 16, each after a transposed convolution (×2) → 1 × 1
conv 16 → 48 = 12 leads × 4 classes. Dashed skip arrows join each encoder level to the decoder level of
the same length. Every box and every ×2 carries its learned count; the legend notes that `forward`
checks the length and standardises each lead (lines 137 to 139). Code: lines 106 to 148
(`_ConvBlock`, `UNet1D`).

**Header band:** `UNet1D(n_leads=12, channels=(16, 32, 64, 128, 256), kernel=9)`: 1,859,104 learned
numbers. It reads the filtered signal itself; TimesFM is not involved.

**Notes:** A ConvBlock is Conv1d of width 9, BatchNorm and ReLU, twice. Each decoder block reads the
upsampled path and the encoder's skip side by side, so the lowest one takes 128 + 128 = 256 channels.
Four halvings need a length divisible by 16, and 4992 = 16 × 312; `forward` checks it and standardises
each lead first. The 48 output channels are reshaped to (4 classes, 12 leads, 4992), so the U-Net shares
the heads' loss, decoder and scorer. The counts add up: encoder 295,776, bottleneck 886,272, decoder
blocks 588,960, transposed convolutions 87,280, output 816.

## 20. The rule-based baseline: `neurokit.py`, `emrich2024.py`, `run_neurokit.py`

**Visual:** reduced on 2026-10-12 to three highlight cards and one table. The table puts our lead-I run
next to Emrich et al. 2024, Table II, for all nine points: mean, within-recording SD and F1, paper and
ours side by side, our SDs in blue and our P-wave F1 in orange. The footer says how `neurokit.py` calls
the `neurokit2` package (0.2.13 from PyPI).

**Highlight cards:**
- Reproduced: error SDs within 0.3 ms of the paper at all nine points; means within 0.5 ms for P and QRS
  and within 2.6 ms for T. Our scorer measures what the paper measured
- One deliberate difference: P-wave F1 92%, not 99.8%. NeuroKit2 reports a P wave on every beat, even
  in atrial fibrillation; we count those as false positives. 191 of the 247 extra lead-I P onsets come
  from 18 AF or flutter people (decision 48)
- The bar to beat: 31.8 ms primary SD over 12 leads, more than twice the TimesFM head (14.1 ms) and the
  U-Net (16.0 ms)

**Notes:** The baseline was reproduced and checked before any model was trained, which is how we know the
scorer is right: a scorer bug would have shown up as a disagreement with a published table. The FastNVG
R-peak variant the paper used was also run but missed many beats in NeuroKit2's port, so the baseline uses
the library defaults, which is what a user gets.

## 21. `train.py`: one loop trains all nine models of a fold

Redrawn on 2026-10-12 as a picture of one training step and one epoch, with three highlight cards.

**Visual:** top, one batch: 8 of the fold's 128 training people, reshuffled every epoch in one order
shared by all nine models. It splits into three inputs: TimesFM states (8, 20, 12, 156, 2560) in fp16,
1.5 GB, copied to the GPU once and normalised once by `normalize_layers` for all six state models; the
filtered signal (8, 12, 4992), copied once for the three U-Nets; and the per-sample labels
(8, 12, 4992), −1 where masked. Nine chips in the models' colours: linear probe, main head and U-Net,
each with seeds 0, 1 and 2. One box below: each of the nine, on every batch, computes scores
(8, 4, 12, 4992), the cross-entropy against the labels with masked samples ignored, a backward pass and
its own AdamW step (lines 227–240). Bottom, the epoch loop as four steps (lines 242–262): validation loss
on the 32 validation people; a new best is copied; 20 epochs without one stops that model only; when all
have stopped or epoch 200 is reached, each gets its best weights back. Under it a chart of the final
runs, median of 15 per model: the probe's best after 123 epochs and stop after 143 (the 15 runs stopped
between 106 and 187), the head 28 and 48 (47 to 58), the U-Net 76 and 96 (65 to 178). No run reached
the 200-epoch cap.

**Highlight cards:**
- Load once, train nine: the fold's 160 training and validation people are loaded once, 28.6 GiB of
  states in RAM; each batch reaches the GPU once per input kind and six models share one normalised copy
- A fair race: seeds 0, 1 and 2 change only the starting weights; all nine models see the same batches
  in the same order under the same recipe (slide 22)
- Safe to interrupt: weights, optimisers and the shuffling generator are saved after every epoch; a
  resumed run matches an uninterrupted one bit for bit (`test_train.py`)

**Snippet S2, `src/ecgdelin/train.py`, the inner loop of `train_jointly` (not on the slide; for
questions):**

```python
        for b in _batches([train_idx[i] for i in order], config.batch_size):
            y = targets[b].to(device)
            inputs: dict[str, Tensor] = {}
            for t in active:
                tick = time.perf_counter()
                if t.source not in inputs:
                    inputs[t.source] = sources[t.source].prepare(sources[t.source].batch(b, device))
                loss = F.cross_entropy(t.model(inputs[t.source], **t.kwargs), y, ignore_index=IGNORE)
                t.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                t.optimizer.step()
                sums[t.name] += loss.item()
                t.seconds += time.perf_counter() - tick
            n_batches += 1
```

**Notes:** The joint loop exists for cost: RAM-hours cost as much as the GPU on Modal, and a fold's states
take 28.6 GiB, so loading them once and training every model and seed over the same stream keeps a fold
at 76 to 109 minutes of training. The three seeds differ in initialisation only; they share the batch
order, which is what decision 31 pre-registered. The chart explains slide 22's curves: the head peaks
early and stops after about 48 epochs, the probe keeps improving slowly for over 120, and the U-Net
varies most between runs.

## 22. The training recipe

**Visual:** `results/m3/fig_training.png`; the recipe as a table.

**Bullets:**
- Loss: per-sample cross-entropy, masked; AdamW, lr 1e-3, weight decay 1e-4; batch 8 recordings; up to
  200 epochs; early stopping after 20 epochs without validation improvement, best weights restored
- Three seeds per fold; no augmentation; no hyperparameter search; identical for head, probe and U-Net
- Fixed before the first run and never changed (decisions 26 and 62)
- What the curves show (fold 0): the head overfits after about epoch 27 and early stopping handles it; the
  U-Net's validation loss is noisy and stops around epochs 60 to 150; the probe converges slowly, best
  around epoch 120 to 165

**Notes:** Equal budgets are the whole point of the control. A reviewer's first objection to "TimesFM
beats the U-Net" is "your U-Net is weak"; the answer is that it uses the same recipe, and that its
within-recording errors and F1 sit in the range of published LUDB U-Nets. We did not help either model
beyond this table.

## 23. Running the folds: `train_fold.py` and `launch_m4.py`

Redrawn on 2026-10-12 around the purpose of each file, with three highlight cards.

**Visual:** top, who runs where. On the laptop, `launch_m4.py` plans each fold (its people, the git
tag), spawns `train_fold` on the deployed app five times, saves the call ids and exits. On Modal, five T4
containers run in parallel, one per fold, each training on 160 people and predicting the other 40. They
write to the Modal Volume: `models/<run>/*.pt` (45 trained models that never leave), `checkpoints/<run>.pt`
(the whole training state after every epoch), `results/<run>.json` (histories, timings, git tag,
scores) and `predictions/<run>_<model>.json` (every point of every test person and seed). An arrow back
to the laptop shows `train_fold.py::fetch` copying each finished run into `results/`. Bottom, inside one
container (`train_fold()`, T4, 40 to 64 GiB of RAM): 1 load the 128 training and 32 validation people's
signals, labels and states, 28.6 GiB, in 33 to 36 s; 2 train the nine models jointly, slide 21, 76 to
109 min; 3 save the weights, free the training states, then read the 40 test people one at a time,
states → 9 models → decoder → points, 17 to 18 s; 4 score the fold for information only, write the
results and predictions to the Volume and commit it. Two callouts: the test people stay unseen until
training ends, and the verdict pools all 200 people (slide 27); every results JSON records the git tag
`step1-frozen`, the people's ids, the TimesFM revision, the GPU, timings and peak GPU memory (4.7 GiB),
and the five folds started within two seconds and finished 77 to 110 minutes later.

**Highlight cards:**
- File 1, `train_fold.py`, one fold start to finish: the GPU function, plus two laptop commands, `main`
  (starts folds and waits for their results) and `fetch` (copies a finished run back later)
- File 2, `launch_m4.py`, start five folds and walk away: spawns the five runs on the deployed app and
  exits at once; the call ids go to `results/m4/calls.json`; `--poll` reports each fold as it ends
- Why the second file exists, no laptop left waiting: `modal run --detach` keeps only the last spawned
  function alive once the client disconnects; the first launch was stopped within minutes and relaunched
  this way (decision 69)

**Notes:** `launch_m4.py` imports `train_fold.py` and reuses its planner, so a launched fold gets exactly
the arguments `main` would give it. The M3 version of the fold script trained fold 0 with one seed and
scored its validation people only. The commands, in order: `modal deploy scripts/train_fold.py`, `python
scripts/launch_m4.py results/m4/calls.json`, optionally `--poll`, then
`modal run scripts/train_fold.py::fetch --run m4_fold<k>` for each fold.

## 24. `decode.py`, `delineate.py`, `trained.py`: class map to points

Redrawn on 2026-10-12 around one real beat, a schematic of the beat grouping and three highlight cards.

**Visual:** `results/deck/fig_decoder_steps.png` (from `scripts/deck_figures.py`, `decoder_steps`): the
linear probe's class map for record 22, lead III, a fold-0 development person from the M3 run, samples
1440 to 1760. Row 1, the model output with one class per sample: the P wave in four pieces, two short P
blips, one-sample gaps in the QRS and the T. Row 2, after the clean-up: the pieces joined, the blips
dropped. Row 3, the decoded onsets, offsets and peaks on the filtered lead with the cardiologists' marks
below; every decoded point lies within 12 ms of its mark. Below the figure, a schematic of the beat
grouping (peaks in samples): P 900, QRS 1000, T 1150, T 1250, P 1300, QRS 1400 give beat 1 with P, QRS
and T, an orphan T at 1250, and beat 2 without a T; a three-row table shows ECG_P_Peaks, ECG_R_Peaks and
ECG_T_Peaks with one value per beat and NaN for the missing T.

**Highlight cards:**
- `delineate.py`, one format for every method: NeuroKit2's eleven keys per lead, one value per beat, NaN
  where a wave is missing; NeuroKit2, both heads and the U-Net all answer this way, so one scorer grades
  them all
- `decode.py`, the same decoder for all: thresholds fixed before any model was trained; peaks from the
  signal, the largest swing for P and T, the R wave for QRS when it rises 0.02 mV, as LUDB marks it
- `trained.py`, models answer like NeuroKit2: the highest-scoring class per sample, then the decoder on
  the filtered signal the model saw; the wrappers serve new recordings, the fold script decodes the stored
  states directly

**Snippet S3, `src/ecgdelin/delineate.py`:**

```python
def group_beats(waves: Sequence[Wave]) -> tuple[list[Beat], int]:
    """Beats in time order plus the number of orphan P/T waves (decision 40)."""
    qrs = sorted((w for w in waves if w.kind == "QRS"), key=lambda w: w.peak)
    others = [w for w in waves if w.kind != "QRS"]
    beats = []
    used = 0
    for k, q in enumerate(qrs):
        lo = qrs[k - 1].peak if k > 0 else -math.inf
        hi = qrs[k + 1].peak if k + 1 < len(qrs) else math.inf
        ps = [w for w in others if w.kind == "P" and lo < w.peak < q.peak]
        ts = [w for w in others if w.kind == "T" and q.peak < w.peak < hi]
        p = max(ps, key=lambda w: w.peak) if ps else None
        t = min(ts, key=lambda w: w.peak) if ts else None
        used += (p is not None) + (t is not None)
        beats.append(Beat(p, q, t))
    return beats, len(others) - used
```

**Notes:** The decoder is shared by every trained model, with thresholds fixed in advance, so no model is
decoded more cleverly than another. The example uses the linear probe because it leaves the most
fragments; the head and the U-Net go through exactly the same steps. Joining happens before dropping and
only between neighbouring regions of the same kind, so a blip of another class between two pieces keeps
them apart and the extra piece becomes an orphan. Beats are grouped the way NeuroKit2 reports them
because NeuroKit2 can only report waves attached to a detected R peak; scoring all methods on the same
kind of output is what makes the comparison fair. The orphan count stays visible so nothing is silently
lost.

## 25. `score.py`: from matched points to numbers

Redrawn on 2026-10-12, then relabelled the same day after a review found the outcome labels easy to
misread: each item now carries one outcome and its reason, and a card defines TP, FN, FP and TN. The
bootstrap and the stop rule stay on slide 26.

**Visual:** 1, how one lead is scored, a schematic of QRS onsets (times in ms) with a legend: marks at
300, 700, 1150 and 1550 inside the lead's marked span (150 to 1750), ±150 ms shaded around each mark,
predictions at 316, 680, 790, 1330 and 1880, and a green box around each TP pair. Labels: "TP, error
+16 ms" (316 with 300); "TP, error −20 ms" (680 with 700); "FP, duplicate: mark 700 is already paired
with 680" (790); "FP, no mark within 150 ms" (1330); "not counted, outside the marked span" (1880); "FN,
missed: nearest prediction 180 ms away" (mark 1150); "FN, missed: nothing within 150 ms" (mark 1550). The
scorer's own `_match_lead` gives the same TP 2, FN 2, FP 2 and errors. 2, the functions in the order they
run: `score_recording` (a `PersonScore`), `summarize` (a `Report`), `score` (overall and per rhythm group)
and `compare_seeds` (a `Verdict`, slide 26). 3, bars for P onset over 200 people in the final runs:
NeuroKit2 39.2 ms pooled and 15.9 ms within recordings, the TimesFM head 16.2 and 9.4 ms, with one line
saying that each TP gives one error, prediction minus mark, and how the two SDs are taken.

**Definitions card** (Emrich et al. 2024, section III.B, after ANSI/AAMI EC57):
- TP: a mark and a prediction paired within ±150 ms; one prediction per mark, nearest first
- FN: a mark left without a prediction, a missed boundary
- FP: a prediction left without a mark inside the marked span, a wrong or duplicate claim
- TN: not used; empty samples are not events, and none of the scores needs it
- Se = TP / (TP + FN), PPV = TP / (TP + FP), F1 = 2TP / (2TP + FN + FP); in the example all are 0.50
- One prediction per mark and the span rule are decisions 41 and 44; a second pass at ±40 ms gives the
  strict figures

**Snippet S4, `src/ecgdelin/score.py`:**

```python
def match(ref: NDArray[np.int64], pred: NDArray[np.int64], tol: int) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, nearest pair first, |pred - ref| <= tol (decision 41)."""
    if ref.size == 0 or pred.size == 0:
        return []
    d = pred[None, :] - ref[:, None]
    i, j = np.nonzero(np.abs(d) <= tol)
    order = np.lexsort((j, i, np.abs(d[i, j])))
    used_ref: set[int] = set()
    used_pred: set[int] = set()
    pairs = []
    for k in order:
        a, b = int(i[k]), int(j[k])
        if a in used_ref or b in used_pred:
            continue
        used_ref.add(a)
        used_pred.add(b)
        pairs.append((a, b))
    return pairs
```

**Notes:** Matching follows the AAMI EC57 / `bxb` convention. Errors are prediction minus mark, and the
SDs are population SDs, which makes the pooled SD decomposable into per-person sums for the bootstrap
(slide 26). Region IoU rebuilds each method's P, QRS and T regions from its onset and offset points, so
NeuroKit2 is measured the same way as the trained models. Rhythm groups are sinus, AF or flutter, and
other; a group without P waves is summarised over the points it has.

## 26. The primary metric and the stop rule

Redrawn on 2026-10-12 as a four-step pipeline with little text, ending in the real bootstrap
distributions behind the verdict.

**Visual:** three steps in a row, then the result. 1, one number per method: the head's pooled error SDs
at the five CSE points (P on 16.2, P off 14.2, QRS on 8.7, QRS off 10.5, T off 20.9 ms) as bars with their
mean, 14.1 ms; the U-Net's mean is 16.0 ms; lower is better. 2, resample the people 10,000 times: eight
numbered people and one draw with repeats and gaps; a person's beats stay together. 3, same draw, both
methods: head (mean of 3 seeds) minus U-Net (mean of 3 seeds) gives one paired difference per draw. 4,
`results/deck/fig_verdict_draws.png` (from `scripts/deck_figures.py`, `verdict_draws` and
`verdict_figure`): histograms of the 10,000 differences, rebuilt from `results/predictions` with the
scorer's own functions and checked against `results/m4/summary.json`; primary SD −2.28 to −1.53 ms with
every draw below zero, F1 at ±150 ms −0.09 to +0.86 percentage points with about 6% of draws below zero.
A strip states the stop rule, fixed in advance: SD interval entirely below 0, so the head is better; F1
interval not entirely below 0, so F1 is no worse; rule met.

**Snippet S5, `src/ecgdelin/score.py`:**

```python
def _draw_counts(n_people: int, n_draws: int, seed: int) -> NDArray[np.float64]:
    """How often each person appears in each bootstrap draw of ``n_people`` with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_people, size=(n_draws, n_people))
    counts = np.zeros((n_draws, n_people))
    np.add.at(counts, (np.arange(n_draws)[:, None], idx), 1.0)
    return counts


def _primary_per_draw(scores: Sequence[PersonScore], counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """The primary metric of every draw, from per-person sums (population SD decomposes exactly)."""
    sds = []
    for point in CSE_POINTS:
        n = counts @ np.array([s.errors_ms[point].size for s in scores], dtype=np.float64)
        s1 = counts @ np.array([s.errors_ms[point].sum() for s in scores])
        s2 = counts @ np.array([np.square(s.errors_ms[point]).sum() for s in scores])
        with np.errstate(invalid="ignore", divide="ignore"):
            var = s2 / n - (s1 / n) ** 2
        sds.append(np.sqrt(np.maximum(var, 0.0)))
    return np.asarray(np.mean(sds, axis=0))
```

**Notes:** Why population SD: it decomposes exactly into per-person sums, so a draw of 200 people is a
matrix product rather than a re-scoring, and 10,000 draws take seconds. Why pooled over folds rather than
averaged per fold: five fold numbers are few and correlated (Nadeau and Bengio 2003; Varoquaux 2018), and
F1 averaged over folds is biased (Forman and Scholz 2010). Why averaged over seeds inside a draw: the
verdict is about the recipe's typical model, not one lucky seed.

## 27. `crossval.py` and `final_report.py`: five folds, one result

Redrawn on 2026-10-12 as two pictures and three cards (the slide builder is a scratch script; the numbers
come from `results/m4/summary.json`).

**Visual:** 1, `crossval.py` pools the folds: five fold tiles, each with its 40 held-out people as dots,
arrows into `assemble` ("every person exactly once, from the one fold that held them out", run once per
model and per seed) with its three refusals (a person missing, a person from another fold, a person
predicted twice), then one box with the same 200 dots: one prediction set of 200 people, each predicted by
models that never saw them, one set per seed. 2, `final_report.py`, the only place the five runs meet, as
five steps: check that each run's fold and test people match the saved split; load the 200 recordings and
every model's predictions per fold and seed plus NeuroKit2's; pool per seed with `assemble`; score (the
primary SD with its 95% interval, verdicts for five pairs, tables per point, rhythm and fold); write
`docs/report.md`, two figures and `results/m4/summary.json`.

**Highlight cards:**
- `assemble`, no silent gaps: a missing person, a stranger or a duplicate stops the run; a quietly missing
  hard case would make the pooled score look better than it is
- `per_fold_primary`, per fold is information only: five 40-person numbers are noisy, so the verdict pools
  all 200 people; still, the head wins in every fold, head minus U-Net −1.93, −1.79, −1.48, −1.91 and
  −2.27 ms for folds 0 to 4
- `final_report.py`, where the verdict is computed: the verdict, the headline table and the paper's pooled
  numbers come from its outputs; its one change after the freeze fixed image links (decision 70)

**Notes:** The five per-fold numbers are correlated as well as noisy, because the folds share most of
their training people (Nadeau and Bengio 2003; Varoquaux 2018), which is why decision 64 pools all 200
people under one paired bootstrap. `reporting.py` holds the shared table and figure style. A two-epoch
rehearsal of the whole chain ran on fold 0 before the freeze; the report script was frozen with the rest.

## 28. Results: the verdict

**Visual:** `results/m4/fig_folds.png`; the headline table.

**Bullets:**
- 200 people, each predicted by models that never saw them; three seeds; 10,000 bootstrap draws
- Primary SD, ms [95% CI]: TimesFM head 14.11 [13.38, 14.86]; U-Net 16.01 [15.17, 16.81]; linear probe
  15.80 [14.93, 16.67]; NeuroKit2 31.84 [30.82, 32.85]
- Head minus U-Net: [−2.28, −1.53] ms, entirely below zero; F1 difference [−0.0009, +0.0086], safeguard
  holds: the pre-registered claim is met
- Negative in every fold (−1.93, −1.79, −1.48, −1.91, −2.27 ms); seeds agree within 0.04 ms for the head
- Mean F1 at ±150 ms over the CSE points: 98.8%, 98.5%, 98.4%, 93.7%

**Notes:** Read the interval, not the point: the head is better by between 1.5 and 2.3 ms of primary SD,
about 12% of the U-Net's error. Small, but it was predicted, tested once, and it holds in every fold and
seed. The U-Net itself beats NeuroKit2 by 16 ms, so most of the distance to the rule-based baseline is
"training on LUDB"; the TimesFM states add the last two milliseconds on top of that with a quarter of the
parameters.

## 29. Results: per point, against the CSE limits

**Visual:** `results/m4/fig_points.png`; `results/m3/fig_per_lead.png` as a secondary image (development
people, per lead).

**Bullets:**
- Head, pooled / within-recording SD, ms (CSE 2σ): P onset 16.2 / 9.4 (10.2); P offset 14.2 / 8.4 (12.7);
  QRS onset 8.7 / 5.5 (6.5); QRS offset 10.5 / 6.5 (11.6); T offset 20.9 / 11.1 (30.6)
- Within recordings, the head, the U-Net and the probe meet every CSE limit; pooled, no method meets the
  P-onset, P-offset or QRS-onset limits
- Mean errors of the head and U-Net within ±1.1 ms at every boundary; QRS peaks read about +4 ms, because
  LUDB marks the R peak one or two samples before the signal's maximum, which the decoder picks (the M3
  deep dive in `docs/m3-report.md` traced this; its slide was removed on 2026-10-12)
- T wave is where the head gains most: T onset 22.7 vs 28.9 ms, T offset 20.9 vs 25.2 ms (pooled)
- Per lead (development people): T offset is the hardest point everywhere; P on/off hardest in V1, V2, aVL

**Notes:** The pooled SD is dominated by hard beats, matches to a neighbouring wave within 150 ms, and
between-person bias; the within-recording SD is what the Emrich paper reports and what clinicians compare
to the CSE limits. Both are in the report; we chose the stricter one as the primary metric before the
study, so the pass/fail against CSE is a secondary reading.

## 30. Results: where the gain comes from

**Visual:** `results/m3/fig_layer_weights.png`; a small comparison table with published numbers.

**Bullets:**
- Probe vs U-Net: 15.80 vs 16.01 ms, statistically indistinguishable: a linear readout of one 64 ms block
  already matches a supervised U-Net
- Head vs probe: [−2.00, −1.38] ms: six neighbouring blocks and one non-linearity add about 1.7 ms
- Rhythm: AF people are harder for everyone (head 19.0 vs 13.6 ms sinus; U-Net 23.5 vs 15.2); the head's
  margin is larger on AF people (4.5 ms) than on sinus people (1.6 ms)
- Published comparisons: NeuroKit2's within-recording SDs reproduce Emrich et al.'s single-lead table;
  our U-Net's pooled SDs are below Moskalenko et al.'s LUDB U-Net at every CSE point
- Layer weights: the probe leans on layers 4 to 8 and 11; the head stays near uniform because its
  projection can pick features inside every layer

**Notes:** The pattern says what the states are good at: context. Where the signal is sharp, at the QRS,
all trained models are within 2 ms of each other; where it is ambiguous, at the T wave and in AF, the head's
view across 12 leads and 10 seconds wins. The per-layer probe ablation that would pin down which layers
carry the information was deferred.

## 31. What predictions look like

**Visual:** `results/m3/fig_overlay_typical.png`, `fig_overlay_worst.png`, `fig_overlay_af.png`
(development people).

**Bullets:**
- Record 115 (sinus bradycardia): all four methods agree with the marks to within a few samples
- Record 22 (sinus arrhythmia, worst sinus person for the head): the first beat's P wave is unmarked yet
  predicted by all models; in V1 the trained models split a P wave in two; NeuroKit2's P runs into the QRS
- Record 83 (atrial fibrillation): no P marks; the trained models split a wide QRS; NeuroKit2 reports P
  waves on every beat

**Notes:** Three pictures to calibrate the numbers: a typical record where everything works, the worst
sinus record, and an AF record. The failure modes are honest ones: an unmarked but real P wave, a fragmented
P in a lead where it is tiny, a wide fibrillation QRS.

## 32. Limits and what was left out

**Visual:** a plain list.

**Bullets:**
- One dataset, 200 people, 500 Hz; one foundation model, frozen, no fine-tuning or LoRA
- No augmentation and no hyperparameter search by design; a tuned U-Net might close part of the 1.9 ms gap,
  and that is a legitimate next experiment, not this one
- The pooled primary SD mixes jitter with gross mismatches; within-recording SDs tell the clinical story
- The decoder's peak convention follows LUDB's one-sample offset; edge waves are unscored; AF P-wave
  false positives
- Removed from step 1 after M3: the forward-only ablation, per-layer probes, the learning curve
- Research only under the TimesFM licence: no distribution of trained heads, no commercial use

**Notes:** The study answers one narrow question and was built so that the answer is credible either way.
Everything in this list is a separate decision; nothing here was skipped silently, each has a decision-log
entry.

## 33. How it was built

**Visual:** the milestone timeline with commit count and the tooling.

**Bullets:**
- Planned on 2026-10-09: plan, spec with 47 user stories, decision log; built on 10-10 and 10-11
- Milestones: M0 foundations, M1 scorer and NeuroKit2 (checked against the published table), M2 states,
  M3 fold-0 models and deep dive, M4 code, rehearsal, freeze, final runs; one commit per step, a report
  after each
- Test-first at the two seams; every decision logged with its reason as it arose (70 entries)
- Built with Claude Code as the pair programmer; every design choice was put to the user as a question
  with a recommendation before it was made
- Compute: Modal, T4 GPUs, five parallel fold containers; the whole step on the order of ten dollars

**Notes:** The process is as reusable as the code. The decision log turned into the methods section; the
tests turned refactors into non-events; the milestone reports caught the one thing that could have gone
wrong, the launch mode, within minutes.

## 34. Run it yourself

**Visual:** the command list; where the outputs land.

**Bullets:**
- `uv sync`, `uv run pytest` (unit tests, no GPU)
- `uv run modal run scripts/run_neurokit.py` (baseline), `scripts/extract_states.py` (states, once),
  `scripts/train_fold.py::main --fold 0 --max-epochs 2` (rehearsal)
- `uv run modal deploy scripts/train_fold.py` then `uv run python scripts/launch_m4.py results/m4/calls.json`
  (the five folds), `scripts/train_fold.py::fetch --run m4_fold<k>`, `uv run python scripts/final_report.py`
- Outputs: `docs/report.md`, `results/m4/summary.json`, `results/m4/m4_fold<k>.json`,
  `results/predictions/*.json`; the paper draft in `docs/paper.md`
- Data read in place from the LUDB download; TimesFM weights downloaded once into the Volume

**Notes:** Everything a teammate needs is in the README. The states stay in the Volume, so a new head can
be trained without re-extracting; the trained weights stay there too, under the licence.

## 35. Close

**Visual:** three lines.

**Bullets:**
- Frozen TimesFM 3 states, read by a 0.49M-parameter head, delineate LUDB better than a 1.86M-parameter
  U-Net on the same data: 14.11 vs 16.01 ms, interval [−2.28, −1.53] ms, F1 safeguard held
- A linear probe on the states already matches the U-Net; the gain is at the T wave and on AF people
- Pre-registered, person-level cross-validated, three seeds, frozen before the runs, reproducible from the
  repository

**Notes:** Questions welcome; the deep-dive report, the full result tables and the paper draft are in the
repository for anyone who wants the numbers behind any slide.
