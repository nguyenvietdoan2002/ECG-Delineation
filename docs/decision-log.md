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

Decision 8 is not in the table because it was not a plan decision. It was to remove the credentialed ICD zip and its history from the Hugging Face dataset repo. The repo can stay public, since the MIMIC-IV-ECG waveforms are ODbL. The user handles this, and it does not affect step 1.
