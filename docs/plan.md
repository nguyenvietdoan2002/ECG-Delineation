# Plan: TimesFM 3 for P/QRS/T wave delineation on LUDB (step 1)

Agreed 2026-10-09; implementation details agreed 2026-10-10. Decisions and their reasons are in
[decision-log.md](decision-log.md); the spec is in [spec.md](spec.md).

## Question

Can frozen TimesFM 3 states, with a small trained head, delineate P, QRS and T waves better than
NeuroKit2 and better than a plain U-Net trained on the same data?

Step 1 uses **LUDB only**. QTDB and MIMIC-IV-ECG are out of scope until step 1's result is in.

## Data

- **LUDB v1.0.1**, read in place from
  `Meddies/20260926_Elsevier 2025/data/ludb/lobachevsky-university-electrocardiography-database-1.0.1/`
  (no copy, no download). 200 people, one 10 s ECG each, 12 leads, 500 Hz, 5000 samples per lead.
  Two cardiologists marked onset `(`, peak (`p` / `N` / `t`) and offset `)` of every P, QRS and T,
  separately in each lead. 58,423 marked waves (21,965 QRS, 19,661 T, 16,797 P). No Q/S or U marks.
  Licence: ODC Attribution 1.0 (cite Kalyakulina et al., IEEE Access 2020, and the PhysioNet record).
- **Signal length:** trained models use the first **4992 samples** of each lead (156 blocks of 32),
  with no padding. The last 8 samples (16 ms) are dropped; the latest mark in any lead is at sample
  4832, so nothing scored is lost. NeuroKit2 receives the full 5000 samples, as a user would give it.
- **Splits:** 5-fold cross-validation by person (one record = one person), stratified by rhythm
  (sinus / AF or flutter / other) from the header `Rhythm:` line. Folds are saved to `splits/` and
  never regenerated. Within each training fold, 20% of its people form the validation set used for
  early stopping; the test fold is never used for any choice.
- **Learning curve:** within each training fold, also train on 25% and 50% of its people
  (nested subsets, fixed by seed).
- **Marked span:** per lead, only the span from the first to the last mark is used for training and
  scoring; the first and last beat or two are unmarked (marked span is 4.4–8.8 s, median 7.3 s).
- **Incomplete marks:** 134 peaks lack their `(` or `)`. Those waves are excluded from labels and
  scoring, and their samples are masked out of the loss.
- **Labels:** one class per sample: P, QRS, T, none, built from each `(`…`)` pair.
- **Input preprocessing:** the same for every trained model: physical units (mV), 0.5 Hz
  high-pass (zero-phase Butterworth, order 5), nothing else. NeuroKit2 uses its own cleaning.
- **No augmentation** for any model: both trained models see exactly the same data.
- **Lead order:** LUDB order (I, II, III, aVR, aVL, aVF, V1–V6), selected by name.

## Models

### 1. TimesFM 3 (frozen)

- Weights `google/timesfm-3.0-pytorch` (not gated), loaded with `TimesFM3Torch.from_pretrained`,
  eval mode, inference mode, fp32, no gradients into the backbone.
- Input: all 12 leads as 12 variates of one sample, 500 Hz, 4992 samples = 156 blocks of 32
  samples (64 ms). Masks all false; all variates are targets.
- Two passes: forward and time-reversed. The reversed states are flipped back and concatenated with
  the forward states per block → 2 × 1280 = 2560 values per block per lead. (TimesFM attention and
  normalisation are causal, so one pass sees at most 62 ms past the end of each block.)
- States are read with forward hooks on every layer of `transformer_stack.layers` (output `[0]`,
  shape `(b, 12, 156, 1280)`), for all 20 layers, computed once and stored in fp16 in the Modal
  Volume: 183 MB per recording, 36.6 GB for LUDB. The probe on record 1 found no value above 865
  (fp16 limit 65,504) and a round-trip error of ≈ 2 × 10⁻⁴, so fp16 is safe.
- **Main head (b)**, shared across leads, ≈ 0.49M parameters, fixed before the first run:
  1. learned softmax weighting over the 20 layers → one 2560-vector per block;
  2. linear 2560 → 128 per block;
  3. three 1-D convolutions along the block axis, 128 → 128, width 3 (receptive field ±3 blocks,
     ≈ ±200 ms);
  4. linear 128 → 128, read as 32 samples × 4 classes;
  5. blocks laid end to end → 4992 samples × 4 scores → argmax → per-sample label.
- **Also reported (a):** linear probe: the same layer weighting, then one linear layer
  2560 → 32 × 4.
- **Ablations:** forward pass only (no reversed pass); per-layer linear probe (one probe per layer,
  fold 0 only) to show where wave information lives.

### 2. U-Net control

- Standard 1-D U-Net, 12-lead input, 12 × 4-class output (one map per lead), 5 levels,
  16 → 256 channels, kernel width 9, ≈ 2M parameters.
- Same folds, labels, masks, loss, optimiser, budget and seeds as the TimesFM head.

### 3. NeuroKit2 baseline

- `neurokit2==0.2.13`, `ecg_delineate(method="prominence")`, run one lead at a time as a user
  would: `ecg_clean` default, R-peaks from `ecg_peaks` default, R-peaks reported as
  `ECG_R_Peaks`. Not trained. The paper states that `prominence` was chosen as NeuroKit2's
  best-scoring method (the library default is `dwt`).

### Training (both trained models)

| Setting | Value |
|---|---|
| Loss | per-sample cross-entropy, masked outside the marked span and at incomplete waves |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Batch | 8 recordings |
| Length | up to 200 epochs; stop after 20 epochs without improvement in validation loss |
| Seeds | 3 per fold |
| Tuning | none; these settings are used as written |
| Decoder clean-up | drop P/QRS regions < 20 ms and T regions < 40 ms; join same-wave gaps < 10 ms |

All development happens on fold 0's validation people. Settings, decoder thresholds and analysis code
are then frozen in a tagged commit before the final 5-fold × 3-seed runs.

## Compute (Modal)

- Every experiment step runs on Modal (Starter plan, $30/month credits): CPU functions for loading,
  splits, NeuroKit2 and scoring; **T4** GPU functions for state extraction, training and inference.
- Volume `ecg-delineation` holds LUDB (uploaded once), the TimesFM weights (downloaded once), the
  stored states, checkpoints, trained weights, predictions and result JSONs. Predictions and JSONs
  are downloaded into the repo's `results/` and committed; weights stay in the Volume.
- Images pin the probed versions (Python 3.12, PyTorch 2.14.1 with CUDA 13, NeuroKit2 0.2.13) and
  copy `timesfm3` from `Meddies/timesfm/src`, so Modal runs the exact code in the repo.
- Head training: one container per fold loads that fold's states into RAM once, and all seeds and
  variants for the fold train jointly over the same data stream (each batch moved to the GPU once).
  Memory costs as much as the GPU, so this keeps the heads at a few dollars in total.
- GPU containers can be preempted: extraction saves per recording and resumes; training
  checkpoints to the Volume; long runs use `modal run --detach`.
- Expected cost of one full step-1 run: extraction < $0.50, heads $2–4, U-Nets $4–7, CPU cents,
  storage $0 (first 1 TiB/month free): about $7–12. Stored states are deleted after the final runs.
- The laptop runs the unit tests only (CPU PyTorch); the same suite can also run on Modal CPU.

## Output format

Per lead, a dict with NeuroKit2's keys: `ECG_P_Onsets`, `ECG_P_Peaks`, `ECG_P_Offsets`,
`ECG_R_Onsets`, `ECG_Q_Peaks`, `ECG_R_Peaks`, `ECG_S_Peaks`, `ECG_R_Offsets`, `ECG_T_Onsets`,
`ECG_T_Peaks`, `ECG_T_Offsets` (sample indices, NaN where absent). Derived from predicted regions:

- onset/offset = first/last sample of each region, after the decoder clean-up above;
- P and T peak = largest absolute deflection from the region's baseline;
- R peak = largest absolute deflection inside QRS; Q / S = lowest point before / after it inside QRS.

## Scoring

- Per lead and point type, predictions are matched one-to-one to LUDB marks within ±150 ms, inside
  the marked span only (AAMI EC57 / WFDB `bxb` convention).
- **Primary metric:** mean, over the five CSE points (P onset, P offset, QRS onset, QRS offset,
  T offset), of the error SD in ms, pooled over all matched beats. 95% CIs by bootstrap over people.
  CSE 2σ limits for reference: 10.2, 12.7, 6.5, 11.6, 30.6 ms.
- **Safeguard:** F1 at ±150 ms must not be lower than the comparator's beyond its CI.
- **Also reported:** Se, PPV, F1 at ±150 ms and ±40 ms; mean error per point; SD averaged within
  recordings (comparable with Emrich et al. 2024); region mIoU; all of these split by rhythm;
  learning curve (25/50/100%).

## Stop rule (fixed before any experiment)

- **Test:** paired bootstrap over people. Draw 200 people with replacement 10,000 times; each time
  compute the primary metric for the TimesFM head and for the U-Net on the same drawn people
  (averaged over the 3 seeds) and take the difference. TimesFM beats the U-Net if the 95% interval
  of the difference lies entirely on the "TimesFM better" side of zero. The F1 safeguard uses the
  same procedure: the 95% interval of the F1 difference must not lie entirely on the "TimesFM
  worse" side.
- TimesFM 3 is claimed to help **only if** its main head (b) beats the U-Net by that test.
- If it beats NeuroKit2 but not the U-Net: report that and stop. Step 2 (hybrid U-Net + TimesFM
  states, then QTDB and MIMIC-IV-ECG) is a new decision, not an automatic continuation.

## Milestones

Built milestone by milestone, test-first, one commit per step, a report to the user after each.

- **M0 Foundations:** project scaffold (uv, git), local test environment, Modal app skeleton and
  Volume, LUDB loader, label conversion, saved folds.
- **M1 Scorer + NeuroKit2:** the scorer; NeuroKit2 `prominence` over all of LUDB; sanity check
  that lead-I results land near Emrich et al. 2024's published table (F1 98.3–99.8%, error SD
  6–16 ms).
- **M2 TimesFM states:** extraction to the Volume; causality and forward/reversed alignment checks.
- **M3 Models on fold 0:** probe, main head, U-Net trained on fold 0 as a sanity check.
- **M4 Final runs:** frozen tagged commit, then 5 folds × 3 seeds, ablations, learning curve,
  report and stop-rule verdict.

## Project layout

`Meddies/ecg-delineation/`, uv project, Python 3.12, local git repo (one commit per step):

- `src/ecgdelin/` — data loading, labels, splits, TimesFM state extraction, heads, U-Net, decoding,
  metrics, Modal app
- `tests/` — written first
- `scripts/` — extract states, train, evaluate (each a `modal run` entry point)
- `splits/`, `results/` (one JSON per run), `docs/plan.md`, `docs/spec.md`, `docs/decision-log.md`

Local dependencies: CPU PyTorch, `neurokit2==0.2.13`, `wfdb`, `scipy`, `numpy`, `modal`;
`timesfm3` made importable from `Meddies/timesfm/src` through a `.pth` file (an editable install
would write an `egg-info` folder into that repo). The Modal image installs the CUDA build.

## Tests to write first

- LUDB marks → per-sample labels, including incomplete triplets and the marked-span mask.
- Regions → NeuroKit2-style points on small synthetic cases.
- One-to-one matching and metrics on hand-built cases with known answers.
- TimesFM state extraction: output shapes `(12, 156, 2560)`, and forward/reversed block alignment
  (runs on Modal GPU, marked `gpu`).

## Constraints

- **TimesFM 3 weights:** TimesFM Non-Commercial License v1.0. Research only; trained heads are not
  distributed; results are not used for commercial decisions.
- **LUDB:** attribution required.
- **MIMIC-IV-ECG:** not used in step 1.
