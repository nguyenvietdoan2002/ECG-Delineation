# Spec: TimesFM 3 ECG wave delineation on LUDB (step 1)

Status: ready-for-agent. Companion documents: [plan.md](plan.md) (experiment plan) and
[decision-log.md](decision-log.md) (decisions and reasons).

## Problem Statement

I want to know whether Google's TimesFM 3, a general time-series foundation model, can find P waves,
QRS complexes and T waves in 12-lead ECGs the way NeuroKit2 does, and whether it does so better.
TimesFM 3 only forecasts. It has no wave-detection ability out of the box, no training code, and its
internal states only look backwards in time. NeuroKit2 is a fixed rule-based tool, so beating it with
any trained model proves little about TimesFM in particular.

I need a fair, reproducible, publishable comparison. It must use cardiologist-marked ECGs, fixed
splits, a metric chosen in advance, and a stop rule, so the answer is credible whichever way it goes.

## Solution

Build a small research codebase that:

- turns LUDB's cardiologist marks into training labels;
- extracts frozen TimesFM 3 states for all 12 leads at once, in forward and time-reversed passes;
- trains a small head on those states to label every sample as P, QRS, T or none;
- trains a 12-lead U-Net on the same data as a control;
- runs NeuroKit2's `prominence` method as the rule-based baseline.

All three methods sit behind one delineator interface. Each returns NeuroKit2-style points
(`ECG_P_Onsets` … `ECG_T_Offsets`) for every lead. One scorer compares any method's points with
LUDB's marks using standard matching and the pre-registered metric. The result is a table with
confidence intervals and a yes/no answer from the stop rule:

> **Stop rule:** TimesFM helps only if its main head beats the U-Net on the primary metric by more
> than the confidence intervals, while F1 holds.

## User Stories

### Data and labels

1. As a researcher, I want LUDB read in place from my existing `ecgrepro` download, so that I don't
   keep duplicate copies of the dataset.
2. As a researcher, I want each recording loaded as a 12 × 5000 array in millivolts with leads
   selected by name, so that lead order can never be silently mixed up.
3. As a researcher, I want each lead's cardiologist marks converted into per-sample labels
   (P, QRS, T, none), so that models can be trained on them.
4. As a researcher, I want the unmarked start and end of each lead excluded from training and
   scoring, so that models aren't penalised for finding real waves there.
5. As a researcher, I want waves with a missing onset or offset mark excluded and masked, so that
   incomplete labels don't teach the model wrong boundaries.
6. As a researcher, I want each recording's rhythm read from its header, so that splits can be
   balanced and results broken down by rhythm.
7. As a researcher, I want 5-fold person-level splits balanced by rhythm, saved to disk once, so that
   every experiment uses identical folds and no person appears in both training and test.
8. As a researcher, I want fixed nested 25% and 50% subsets of each training fold, so that I can draw
   a learning curve showing whether more data would help.
9. As a researcher, I want a validation subset carved by person from each training fold, so that
   early stopping never touches the test fold.
10. As a researcher, I want identical preprocessing (0.5 Hz high-pass) for every trained model, so
    that differences come from the models, not the inputs.

### TimesFM 3 states

11. As a researcher, I want the TimesFM 3 checkpoint loaded from the local Hugging Face cache with
    weights frozen, so that the backbone never changes and runs offline.
12. As a researcher, I want all 12 leads fed to TimesFM 3 together as variates, so that each lead's
    state can use the other leads.
13. As a researcher, I want each recording cropped to a whole number of 32-sample blocks (the first
    4992 samples) with no padding, so that TimesFM sees only real signal and the forward and
    reversed passes line up block by block.
14. As a researcher, I want a forward pass and a time-reversed pass, flipped back and joined per
    patch, so that every patch has context from both sides despite TimesFM being causal.
15. As a researcher, I want states from all 20 layers extracted once and cached to disk, so that
    head training is fast and repeatable without rerunning TimesFM.
16. As a researcher, I want state extraction to resume where it stopped, so that a preempted or
    interrupted Modal container doesn't start over.

### Models

17. As a researcher, I want a small convolutional head over neighbouring patch states, with a learned
    weighting across layers, so that TimesFM gets a fair chance to place wave edges precisely.
18. As a researcher, I want that head always smaller than the U-Net, so that a win can't be put down
    to a bigger model.
19. As a researcher, I want a one-layer linear probe on the same states, so that I can report how
    much wave information the states contain on their own.
20. As a researcher, I want a forward-only variant, so that I can measure what the reversed pass adds.
21. As a researcher, I want a per-layer probe, so that I can show which TimesFM layers hold wave
    information.
22. As a researcher, I want a 12-lead U-Net trained with the same folds, labels, loss, budget and
    seeds, so that it is a fair control for "is it TimesFM, or just training on LUDB?".
23. As a researcher, I want NeuroKit2 0.2.13 `prominence` run one lead at a time with its own cleaning
    and R-peak detection, so that the baseline is what a real user would get.
24. As a researcher, I want 3 seeds per fold for every trained model, so that results reflect
    training variability.
25. As a researcher, I want every experiment step to run on Modal T4 GPUs and CPU containers within
    the Starter plan's $30 monthly credit, so that results come from one consistent environment and
    cost nothing extra.
25a. As a researcher, I want all heads for one fold trained jointly in one container that loads the
    fold's states once, so that RAM-hours (which cost as much as the GPU) stay low.
25b. As a researcher, I want training to checkpoint to the Volume and long runs to survive my
    laptop disconnecting, so that a preempted container or a closed lid costs minutes, not a run.
25c. As a researcher, I want no data augmentation and no hyperparameter search for either trained
    model, so that both see exactly the same data and neither is tuned harder than the other.

### Delineator output

26. As a researcher, I want every method behind one delineator interface (12-lead ECG in, per-lead
    points out), so that scoring and reporting treat all methods identically.
27. As a NeuroKit2 user, I want the output to use NeuroKit2's keys with sample indices and NaN for
    absent waves, so that existing NeuroKit2-based code works unchanged.
28. As a researcher, I want predicted regions cleaned of implausibly short fragments and tiny gaps
    with thresholds fixed in advance, so that decoding is not tuned on test results.
29. As a researcher, I want onsets and offsets taken from region boundaries and peaks from the
    largest deflection in each region, so that points are derived the same way for every model.
30. As a researcher, I want Q and S found as the lowest points before and after the R peak within the
    QRS region, so that all 11 NeuroKit2 keys are filled.

### Scoring and reporting

31. As a researcher, I want predicted points matched one-to-one to LUDB marks within ±150 ms, inside
    the marked span only, so that scoring follows the AAMI/`bxb` convention.
32. As a researcher, I want sensitivity, PPV and F1 at ±150 ms and ±40 ms per point type, so that
    results compare with published papers at both loose and strict tolerances.
33. As a researcher, I want mean error and error SD per point in ms, both pooled and averaged within
    recordings, so that my numbers compare with both reporting conventions.
34. As a researcher, I want the primary metric (mean error SD over the five CSE points) computed with
    95% bootstrap confidence intervals over people, so that the stop rule has a sound statistical basis.
35. As a researcher, I want CSE 2σ limits shown next to each point's SD, so that I can see where each
    method meets clinical tolerances.
36. As a researcher, I want region overlap (mIoU), so that results compare with segmentation papers.
37. As a researcher, I want every metric broken down by rhythm (sinus, AF/flutter, other), so that I
    can see where each method fails.
38. As a researcher, I want the stop-rule verdict computed automatically by a paired bootstrap over
    people (10,000 draws, 95% interval of the TimesFM-minus-U-Net difference), so that nobody
    reinterprets it after seeing the numbers.
39. As a researcher, I want one JSON result file per run, recording method, fold, seed, data subset
    and code version, so that every number in the paper traces back to a run.
40. As a researcher, I want a summary table and learning-curve figure produced from the result files,
    so that writing up needs no manual copying.

### Reproducibility and project

41. As a researcher, I want a uv project with pinned versions (including `neurokit2==0.2.13` and the
    local editable TimesFM), so that anyone can recreate the environment.
42. As a researcher, I want a fixed seed for every random choice (splits, subsets, initialisation),
    so that reruns produce the same results.
43. As a researcher, I want decisions recorded in the decision log as they arise during
    implementation, so that the paper's methods section is easy to write.
44. As a researcher, I want scripts for extracting states, training and evaluating, so that each
    stage can be rerun on its own.
45. As a Meddies team member, I want the TimesFM 3 licence restrictions noted in the repo, so that
    nobody distributes trained heads or uses results commercially.
46. As a researcher, I want the project under local git with one commit per step and a tagged
    "frozen" commit before the final runs, so that every number traces to code that cannot have been
    tuned after seeing test results.
47. As a researcher, I want the work built milestone by milestone (M0 foundations, M1 scorer +
    NeuroKit2, M2 states, M3 models on fold 0, M4 final runs) with a report after each, so that I can
    check results before the next stage starts.

## Implementation Decisions

- **Modules.** Each module hides its details behind a small interface:
  - **LUDB dataset**: loads recordings, leads by name, rhythm, marks.
  - **Labels**: marks → per-sample class array plus a loss/scoring mask.
  - **Splits**: person-level, rhythm-stratified folds and nested subsets, persisted.
  - **TimesFM state extractor**: 12-lead recording → cached states with shape
    layers × leads × patches × 2560.
  - **Heads**: the conv head and the linear probe, with layer weighting.
  - **U-Net**.
  - **Decoder**: per-sample class probabilities → NeuroKit2-style points.
  - **NeuroKit2 baseline**.
  - **Scorer**.
  - **Reporting.**
- **Delineator interface (main test seam).** Every method takes a 12 × 5000 ECG at 500 Hz in
  millivolts with named leads. It returns, per lead, a mapping from the eleven NeuroKit2 keys to
  sample indices (NaN where absent). Trained methods are built from a fold's trained weights;
  NeuroKit2 needs no training.
- **Scorer interface (second test seam).** It takes predicted points and LUDB reference marks with
  marked-span masks for a set of recordings. It returns all metrics in the plan plus confidence
  intervals, and a stop-rule verdict when given two methods.
- **TimesFM access.** The extractor uses the low-level model's forward pass with auxiliary outputs or
  layer hooks, not the forecaster API, which can't expose hidden states. Inputs are already patched
  (32 samples per patch) with explicit masks. Reversed-pass states are flipped back along the patch
  axis before joining. The extractor runs with inference mode and fp32.
- **Signal length.** Trained models use the first 4992 samples of each lead (156 blocks of 32), no
  padding, in both passes; the U-Net gets the same 4992 samples. NeuroKit2 gets the full 5000. The
  latest mark in any lead is at sample 4832, so the dropped 16 ms never affects a score.
- **Head shape.** Fixed before the first training run, shared across leads, ≈ 0.49M parameters:
  learned softmax weighting over the 20 layers; linear 2560 → 128 per block; three width-3
  convolutions along the block axis (128 → 128, receptive field ±3 blocks ≈ ±200 ms); linear
  128 → 128 read as 32 samples × 4 classes; blocks laid end to end give 4992 × 4 scores. The probe is
  the layer weighting plus one linear layer 2560 → 32 × 4.
- **Training protocol.** AdamW (lr 1e-3, weight decay 1e-4), batch 8 recordings, up to 200 epochs
  with early stopping after 20 epochs without validation improvement, 3 seeds per fold, no
  augmentation, no hyperparameter search, identical for head and U-Net. Validation people are 20% of
  each training fold. Decoder clean-up: drop P/QRS regions < 20 ms and T regions < 40 ms, join
  same-wave gaps < 10 ms. Everything is frozen in a tagged commit after development on fold 0's
  validation people and before the final runs.
- **Loss.** Per-sample cross-entropy, masked outside the marked span and at incomplete waves.
- **Leads.** LUDB order is I, II, III, aVR, aVL, aVF, V1–V6. Code always selects by name.
- **State cache.** Stored per recording in the Modal Volume in fp16, all 20 layers, both passes
  (183 MB per recording, 36.6 GB total), keyed by checkpoint revision so a changed checkpoint
  invalidates it. A probe on record 1 found a maximum magnitude of 865 (fp16 limit 65,504) and a
  round-trip error of ≈ 2 × 10⁻⁴. Deleted after step 1's final runs.
- **Compute.** All experiment steps run on Modal: CPU functions for loading, splits, NeuroKit2 and
  scoring; T4 GPU functions for extraction, training and inference. One Volume (`ecg-delineation`)
  holds LUDB, the TimesFM weights, states, checkpoints, weights, predictions and result JSONs;
  predictions and JSONs are downloaded into the repo. Images pin the probed versions (Python 3.12,
  PyTorch 2.14.1 with CUDA 13, NeuroKit2 0.2.13) and copy `timesfm3` from the TimesFM checkout. Head
  training uses one container per fold with the fold's states in RAM and all seeds/variants trained
  jointly over one data stream. Training checkpoints to the Volume; long runs are detached.
- **Stop-rule test.** Paired bootstrap over people: 10,000 draws of 200 people with replacement; per
  draw, the primary metric of each model (averaged over seeds) on the same people, and their
  difference. TimesFM wins if the 95% interval of the difference lies entirely on its side of zero.
  The F1 safeguard uses the same procedure in the other direction.
- **Results.** One JSON file per run, carrying everything needed to rebuild tables.
- **Primary metric and stop rule** are as in the plan. They are fixed and must not change after any
  test-fold result has been seen.

## Testing Decisions

- **What makes a good test.** Tests check behaviour at the two seams, not internals. A test must not
  break when we refactor how a module works, only when its observable output changes. No tests on
  private helpers, tensor-layout tricks or call sequences.
- **Delineator seam.** Use synthetic 12-lead ECGs with known wave positions (built in a shared
  synthetic-signal helper). Every method must:
  - return exactly the eleven keys per lead, with indices inside the signal;
  - put P before QRS before T within each beat;
  - give NaN where a wave is absent (for example, no P wave in an AF-like signal).

  Trained methods use small fixtures trained on synthetic data, so tests run in seconds. One extra
  check on the TimesFM extractor covers what the seam can't: state shapes for a 12 × 4992 input
  (`(12, 156, 2560)` after joining), and that the forward states for early blocks don't change when
  late samples change (causality), while the joined states do (the reversed pass works). This check
  needs the real model, so it is marked `gpu` and runs on Modal.
- **Scorer seam.** Hand-built cases with known answers:
  - perfect prediction (Se = PPV = F1 = 1, zero error);
  - a constant 10 ms shift (mean error 10, SD 0);
  - a missed wave (FN) and an extra wave (FP);
  - a prediction just outside ±150 ms;
  - points inside the unmarked edges (ignored);
  - two methods with clearly separated errors (the stop rule says yes) and with overlapping errors
    (it says no).
- **Labels through the scorer.** Converting real LUDB marks to labels and back through the decoder
  must reproduce the marked points almost exactly. This also tests label conversion and decoding
  without testing them separately.
- **End-to-end LUDB checkpoint.** A slower test runs the full pipeline (load, extract, predict,
  score) on a handful of real LUDB records for NeuroKit2 and one trained fixture. It checks that the
  outputs are well-formed and that the scores are in a plausible range.
- **Prior art.** The sibling `ecgrepro` project uses this structure: a `synthetic.py` helper for
  synthetic signals, per-module tests (labels, datasets, delineate), and an LUDB checkpoint test
  (`test_m1_ludb_checkpoint.py`) that runs real data end to end and records results to JSON.
- Unit tests run locally with pytest (CPU PyTorch); the same suite can run on Modal CPU for the
  record.
- Tests are written before the code they cover (test-first).

## Out of Scope

- MIMIC-IV-ECG in any role (training, testing, interval checks against `machine_measurements.csv`).
- QTDB and any other dataset than LUDB.
- The hybrid (U-Net with TimesFM states) and any head that reads the raw signal.
- Fine-tuning or LoRA on TimesFM; TimesFM 2.5; ECG-FM or other ECG foundation models.
- NeuroKit2 methods other than `prominence`.
- Upsampling or downsampling (500 Hz only).
- Data augmentation and hyperparameter search.
- Running experiments on the local GPU.
- Real-time or streaming use, clinical deployment, or any commercial use.
- Publishing trained heads (forbidden by the TimesFM 3 licence).
- The Hugging Face dataset clean-up (removing the credentialed ICD zip), which the user handles
  separately.

## Further Notes

- **Expected outcome.** The literature suggests frozen foundation-model states with a light head
  struggle with exact boundaries, and NeuroKit2's `prominence` reports error SDs close to U-Nets on
  LUDB (Emrich et al. 2024). A negative result is a likely, legitimate outcome, and the stop rule
  exists to report it cleanly.
- **Known LUDB quirks.**
  - Marked span per lead is 4.4–8.8 s of 10 s.
  - 134 peaks lack an onset or offset mark.
  - 24 records have no P marked in lead II (mostly AF and flutter).
  - There are no separate Q or S marks, so Q and S points are reported but not scored.
- **Environment facts from the 2026-10-10 probe.**
  - A fresh uv environment (Python 3.12) installs everything cleanly; `timesfm3` is made importable
    from `Meddies/timesfm/src` with a `.pth` file (an editable install would write `egg-info` into
    that repo).
  - Windows Smart App Control sometimes blocks freshly downloaded wheel DLLs on first import
    (`An Application Control policy has blocked this file`) and allows them minutes later; retry.
  - The laptop GPU is heavily power-throttled (17 s per TimesFM pass); it is not used for
    experiments.
  - Modal is installed and authenticated on this machine; existing Volumes `mimic4-ecg` and
    `mimic4-ecg-lora-adapters` are left untouched. GPU use on the Starter plan needs a payment
    method on file even when only free credits are spent.
  - Expected cost of one full step-1 run: about $7–12 of the $30 monthly credit.
- **Licences.**
  - TimesFM 3 weights: non-commercial, and derivatives must not be distributed.
  - LUDB: ODC Attribution.
  - NeuroKit2: MIT.
