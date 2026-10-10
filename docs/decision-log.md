# Decision log

Decisions agreed on 2026-10-09 while planning step 1. The plan itself is in [plan.md](plan.md).

| # | Decision | Why |
|---|---|---|
| 1 | TimesFM 3 is a frozen backbone with a trained head. | TimesFM only forecasts; it cannot delineate without a trained head. |
| 2 | The goal is research and a publishable result. | We want evidence for or against TimesFM, not a product yet. |
| 3 | Signals are 10 s, 12-lead, 500 Hz resting ECGs. | This matches LUDB (and later MIMIC-IV-ECG). |
| 4 | Output uses NeuroKit2's keys, per beat and lead. | It is a drop-in replacement for `nk.ecg_delineate`. |
| 5 | The model predicts per-sample regions (P/QRS/T/none), and points are derived from them. | This is the standard target in LUDB papers and is easy to train. |
| 6 | Code lives in a new project, `Meddies/ecg-delineation/`. | `Meddies/timesfm` is a clone of upstream google-research/timesfm. |
| 7 | Rigor is publication-level: fixed person-level splits, seeds, CIs, and this log. | Rigor costs little up front, and LUDB is small enough that single splits vary a lot. |
| 9 | TimesFM 3 is the only foundation model; we use neither 2.5 nor ECG-FM. | This was the user's choice. The non-commercial licence is accepted. |
| 10 | TimesFM runs two passes, forward and time-reversed. | TimesFM is causal, so wave ends need right-hand context. |
| 11 | Step 1 trains and tests on LUDB only, plus a learning curve. | MIMIC has no expert labels. The learning curve shows whether more data would help. |
| 12 | Input stays at 500 Hz. | LUDB is recorded at 500 Hz, so no resampling is needed. |
| 13 | We compare TimesFM 3, NeuroKit2 and a U-Net control. | Without a trained control, a win over NeuroKit2 cannot be credited to TimesFM. |
| 14 | The NeuroKit2 baseline is `prominence` only. | It is NeuroKit2's best-scoring method on LUDB (Emrich et al. 2024). |
| 15 | The primary metric is error SD at the five CSE points, with F1 at ±150 ms as a safeguard. | F1 at 150 ms is saturated, and the SD separates methods. It was fixed before any run. |
| 16 | TimesFM takes all 12 leads together. | Multi-lead input uses TimesFM 3's variate attention. |
| 17 | The U-Net also takes all 12 leads. | Both get equal information, so the only difference is TimesFM. |
| 18 | ECG-FM is not used. | This was the user's choice. |
| 19 | The main head is a small conv head over neighbouring patches; a linear probe is also reported. | The conv head uses context but stays smaller than the U-Net. The probe shows what the states hold on their own. |
| 20 | Step 1 includes the U-Net. | The stop rule needs it. |

Implementation-stage decisions, agreed on 2026-10-10:

| # | Decision | Why |
|---|---|---|
| 21 | Local git repo, one commit per step, no remote yet; a tagged "frozen" commit before the final runs. | Every number traces to code that could not have been tuned after seeing test results. |
| 22 | Built milestone by milestone, test-first, with a report after each (M0–M4). | Same workflow as `ecgrepro`; checkpoints without slowing every file down. |
| 23 | Order: M0 foundations → M1 scorer + NeuroKit2 → M2 states → M3 fold-0 models → M4 final runs. | The scorer is checked against Emrich 2024's published NeuroKit2 numbers before any model is trained. |
| 24 | Crop each lead to 4992 samples (156 blocks); no padding. NeuroKit2 gets the full 5000. | 5000 is not a multiple of 32; padding would misalign the forward and reversed passes. No mark lies after sample 4832. |
| 25 | No data augmentation for either trained model. | The stop rule asks whether TimesFM adds something given the same data. |
| 26 | No hyperparameter search; fixed settings table, developed on fold 0's validation people only. | Neither model is tuned harder than the other; nothing is chosen after a test result. |
| 27 | Main head: layer blend, 2560 → 128, three width-3 convolutions, 128 → 32 × 4 (≈ 0.49M). Probe: layer blend + one linear layer. | Context of ±3 blocks without becoming a second U-Net; always smaller than the U-Net (≈ 2M). |
| 28 | Stop rule tested by paired bootstrap over people (10,000 draws, 95% interval of the difference). | Both models are tested on the same people; non-overlapping intervals would be needlessly harsh. |
| 29 | Every experiment step runs on Modal (CPU and T4 GPU functions), not the local GPU. | One consistent environment; the laptop GPU is power-throttled to a fraction of its speed; fits the $30/month free credit. |
| 30 | Store all 20 layers in fp16 in the Modal Volume (36.6 GB); delete after the final runs. | Containers can hold a fold's states in RAM, so the spec's 20-layer blend stands. Probe showed fp16 is safe. |
| 31 | All heads for one fold train jointly in one container over a single data stream. | RAM-hours cost as much as the GPU; loading 29 GB once per fold instead of per run keeps heads at a few dollars. |
| 32 | T4 GPUs for everything; no per-launch cost check; credits reported at milestone ends. | The workload is small; T4 is the cheapest adequate GPU. User's choice on the check. |
| 33 | `timesfm3` is imported from `Meddies/timesfm/src` via a `.pth` file locally and copied into the Modal image. | An editable install would write an `egg-info` folder into the upstream clone. |

Implementation details fixed during M0 (2026-10-10):

| # | Decision | Why |
|---|---|---|
| 34 | A wave's label covers onset..offset inclusive; the mask covers first mark..last mark inclusive. | LUDB's `)` marks the last sample of the wave. |
| 35 | For an incomplete wave (peak without onset or offset) the whole gap between the neighbouring complete waves is masked out. | Its true extent is unknown; masking only the known side would teach a wrong boundary. 134 such waves in LUDB. |
| 36 | Rhythm groups for stratification: `sinus` (any "sinus" rhythm), `af_flutter`, `other`. LUDB has 182 / 18 / 0. | Only two groups occur; AF/flutter is the hard case and must appear in every test fold (3–4 people each). |
| 37 | Folds are dealt round-robin per rhythm group with one running counter; validation (20%) and learning-curve subsets (25%, 50%) are stratified the same way; subsets are nested prefixes of one shuffle, taken from the training people with the validation people held fixed. | Balanced fold sizes (40 each), every subset keeps AF people, and early stopping is comparable across subset sizes. |
| 38 | `splits/ludb_5fold_seed0.json` was generated locally (deterministic, seed 0) and committed; experiments read the file, never regenerate it. | A 20-second deterministic job; the file is the single source of truth on both laptop and Modal. |
| 39 | Local environment uses CPU PyTorch; the Modal image installs PyTorch 2.14.1 with CUDA 13 and copies `timesfm3` and `ecgdelin` from the local checkouts at run time. | The laptop only runs unit tests; Modal always runs the exact code in the working tree. |

Decisions for M1 (scorer and NeuroKit2 baseline), agreed on 2026-10-10:

| # | Decision | Why |
|---|---|---|
| 40 | Predicted regions are grouped into beats the way NeuroKit2 reports them: one entry per QRS, the nearest P region before it and the nearest T region after it. P or T regions with no QRS neighbour in their interval are orphans: dropped from the output and counted. | NeuroKit2 can only report waves attached to a detected R peak, so all methods are scored on the same kind of output. The orphan count keeps the loss visible. |
| 41 | One-to-one matching is greedy, nearest pair first, within the tolerance window (inclusive). The strict ±40 ms figures come from a separate matching run at that tolerance. | Deterministic and simple; beats are far enough apart that it equals optimal assignment in practice. |
| 42 | A NeuroKit2 exception on a lead counts as no predictions for that lead; its marks are misses. The number of failed leads is reported. | A real user gets nothing from an exception. Excluding failed leads would flatter the baseline. |
| 43 | Sanity band for the Emrich 2024 lead-I check: F1 within 1 percentage point and each error SD within 3 ms. Beyond that, M1 stops for investigation. | Matching details and the incomplete-wave rule can move numbers by that much; anything larger points to a scorer bug. |
| 44 | Nine point types are scored (P, QRS, T onsets, peaks, offsets; LUDB's QRS peak against `ECG_R_Peaks`); Q and S are produced but not scored. Unmatched predictions outside the marked span or inside a masked gap are ignored, but a prediction there may still match a mark at the span's edge. Error = prediction − reference in ms. Within-recording SD is taken per recording and lead, then averaged. NeuroKit2 runs over all 200 records on Modal CPU; its predictions and results JSON are committed under `results/`. | LUDB has no Q or S marks. Without the edge rule, a prediction a few samples late for the last T offset of a lead would be lost and the mark counted as a miss, biasing T offset and P onset sensitivity by up to one beat in ten. The rest keeps the scorer identical for every method and the baseline reproducible from the repo. |
| 45 | Error SDs are population SDs (divide by N) over matched pairs. | With tens of thousands of pairs the choice is immaterial, and the population form makes the pooled SD exactly decomposable into per-person sums, which the paired bootstrap resamples. |
| 47 | Decoder QRS peak = the highest positive deflection from the region's baseline when it exceeds 0.02 mV, else the nadir. P and T peaks stay the largest absolute deflection. Chosen on fold 0's validation people only. | LUDB marks the R wave even when the S wave is far deeper (rS in V1–V3, aVR, III). The absolute-deflection rule disagreed with the marks by over 20 ms for 34% of QRS peaks on those people; this rule for 4%. P and T peaks already agreed for 97%. |
| 46 | The Emrich 2024 sanity check compares our lead-I within-recording SD and pooled mean against their single-lead rows, and also runs a paper-faithful variant with FastNVG R-peaks (`ecg_peaks(method="emrich2023")`) for that check only. The baseline itself keeps NeuroKit2's default R-peak detector (decision 14 and story 23). | The paper's σ is the average of per-record SDs and its R-peaks came from FastNVG, so a like-for-like check needs both; the baseline stays what a NeuroKit2 user gets by default. See `docs/references.md`. |

| 48 | Outcome of the Emrich 2024 check (2026-10-10): lead-I mean and within-recording SD agree with the paper within 0.3 ms for all nine points, and F1 within the band for the six QRS and T points. P-wave F1 is 91.8% against 99.8%: of our 247 lead-I P-onset false positives, 191 are in the 18 AF/flutter people (no P waves exist; `prominence` reports one per beat) and 51 are at sinus beats where the cardiologists marked no P in lead I; only 3 P onsets are missed. The scorer's false-positive rule stands; the band is treated as met for the scorer. | The differences are confined to false-positive counting on beats without a P annotation, which the paper evidently did not count; matching and error computation are reproduced to a fraction of a sample. Reporting a P wave during atrial fibrillation is a real delineation error, and the trained models are judged by the same rule. |

Decision 8 is not in the table because it was not a plan decision. It was to remove the credentialed ICD zip and its history from the Hugging Face dataset repo. The repo can stay public, since the MIMIC-IV-ECG waveforms are ODbL. The user handles this, and it does not affect step 1.
