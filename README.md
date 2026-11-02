# ECG wave delineation from frozen TimesFM 3 states

Can a time-series foundation model that was only ever trained to forecast find the P wave, the QRS
complex and the T wave in a 12-lead ECG? This project reads the frozen internal states of
[TimesFM 3](https://github.com/google-research/timesfm), trains a small head (0.49M parameters) on
them, and compares it with a 1-D U-Net trained on the same data, the rule-based
[NeuroKit2](https://github.com/neuropsychology/NeuroKit) delineator and
[ECGdeli](https://github.com/KIT-IBT/ECGdeli).

- **Step 1**: a pre-registered comparison on [LUDB](https://physionet.org/content/ludb/1.0.1/)
  (200 people, cardiologist marks in every lead), person-level 5-fold cross-validation, 3 seeds, with
  the settings and stop rule frozen in the tag `step1-frozen` before the final runs.
- **Step 2**: pretraining the head on ECGdeli's marks for 4,500 [PTB-XL+](https://physionet.org/content/ptb-xl-plus/)
  ECGs, then on the 2,399 ECGs of the [Resting ECG Segmentation Dataset](https://doi.org/10.6084/m9.figshare.28892186)
  (RDB), then fine-tuning on LUDB. The result is the production detector.

Developed at Meddies.

## Results

Primary metric: the mean, over the five CSE points (P onset, P offset, QRS onset, QRS offset, T offset),
of the boundary error SD in ms pooled over all matched beats of the 200 LUDB people. Lower is better;
95% bootstrap intervals over people.

| Method | Error SD, ms [95% CI] |
|---|---|
| Head, pretrained on PTB-XL+ then RDB, fine-tuned on LUDB (**production detector**) | **13.04** [12.36, 13.74] |
| Head, pretrained on PTB-XL+, fine-tuned on LUDB | 13.38 [12.68, 14.07] |
| Head, LUDB only (step 1) | 14.11 [13.38, 14.86] |
| 1-D U-Net, LUDB only (step 1) | 16.01 [15.17, 16.81] |
| ECGdeli 1.1 | 26.66 [25.71, 27.61] |
| NeuroKit2 0.2.13 (`prominence`) | 31.84 [30.82, 32.85] |

The step-1 head beats the U-Net, PTB-XL+ pretraining beats the step-1 head, and the RDB stage beats
PTB-XL+ alone, each with a paired 95% interval entirely below zero. Full reports:
[docs/report.md](docs/report.md) (step 1) and [docs/ptbxl-pretrain-report.md](docs/ptbxl-pretrain-report.md) (step 2).

## Repository map

| Path | What it holds |
|---|---|
| `src/ecgdelin/` | The package. Data: `dataset.py` (LUDB), `ptbxl.py`, `rdb.py`, `labels.py`, `preprocess.py`, `splits.py`. Model: `states.py` (TimesFM 3 state extraction), `models.py` (head, probe, U-Net), `train.py`, `pretrain.py`, `finetune.py`, `crossval.py`. Output: `decode.py` (classes to wave points), `delineate.py`, `score.py` (CSE-point scoring), `analysis.py`, `reporting.py`. Baselines: `neurokit.py`, `ecgdeli.py`, `emrich2024.py`. Infrastructure: `modal_app.py` (image, Volume), `paths.py` |
| `scripts/` | One entry point per experiment step (see [Running](#running)); most run on Modal |
| `tests/` | Unit tests on synthetic ECGs; `-m data` tests read LUDB, `-m gpu` tests run on Modal |
| `splits/` | The fixed splits: LUDB 5 folds, PTB-XL and RDB pretraining and stopping sets |
| `results/` | Every result JSON and every predicted point (`predictions/`), figures (`m3/`, `m4/`, `deck/`). Trained weights are not included (see [Licences](#licences)) |
| `docs/plan.md`, `docs/spec.md` | The experiment plan with its stop rule, and the implementation spec |
| `docs/decision-log.md` | Every decision (1 to 94) and the reason for it |
| `docs/report.md`, `docs/m3-report.md`, `docs/ptbxl-pretrain-report.md` | Generated reports |
| `docs/references.md` | Papers and code used in step 1 |
| `docs/slides-outline.md` | Outline of the presentation deck |
| `docs/commit-map.tsv` | Old to new commit hashes (see [Notes](#notes)) |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12), git, a
[Modal](https://modal.com) account for anything that trains or extracts states, and optionally MATLAB
for the ECGdeli baseline. Nothing trains on a laptop; locally you run tests and reports.

**1. Code.** TimesFM sits next to this repo, at the commit used here:

```bash
git clone https://github.com/nguyenvietdoan2002/ECG-Delineation.git
git clone https://github.com/google-research/timesfm.git
git -C timesfm checkout e51928e
cd ECG-Delineation
uv sync                 # CPU PyTorch and the dev tools
uv run pytest           # unit tests, seconds, no data needed
```

`TIMESFM_SRC` overrides the default `../timesfm/src`. The Modal image copies `timesfm3` from there;
the weights (`google/timesfm-3.0-pytorch`) are downloaded from Hugging Face inside Modal on first use.

**2. LUDB.** Unzip [LUDB 1.0.1](https://physionet.org/static/published-projects/ludb/lobachevsky-university-electrocardiography-database-1.0.1.zip)
so that `data/ludb/RECORDS` exists (or set `LUDB_DIR`), then run `uv run pytest -m data`.

**3. Modal.** Log in, run the smoke test (it creates the Volume `ecg-delineation`), and upload LUDB once:

```bash
uv run modal setup
uv run modal run scripts/smoke.py                       # image and Volume check on CPU and a T4
uv run modal volume put ecg-delineation data/ludb /ludb
uv run modal run scripts/gpu_tests.py                   # the gpu tests, real TimesFM 3, on a T4
```

In Git Bash on Windows, prefix `modal volume put` with `MSYS_NO_PATHCONV=1`.

**4. Step 2 data.** The scripts fetch PTB-XL and PTB-XL+ from PhysioNet's S3 mirror. For RDB,
download `dataset.zip` from [figshare](https://doi.org/10.6084/m9.figshare.28892186) and upload it:
`uv run modal volume put ecg-delineation dataset.zip /rdb/dataset.zip`. Step 2 keeps about 300 GiB
of states on the Volume while pretraining.

## Running

Commands run from the repo root. Long jobs use `modal deploy` and then `python … launch`, which spawns
them on the deployed app so they survive the client; `status` shows progress and `fetch`/`collect`
copy results into `results/`. Each script's docstring lists all its options.

**Step 1: LUDB**

| Stage | Command |
|---|---|
| Folds (already in `splits/`) | `uv run python scripts/make_splits.py --seed 0` |
| NeuroKit2 baseline | `uv run modal run scripts/run_neurokit.py` |
| ECGdeli baseline (local, MATLAB) | `uv run python scripts/ecgdeli_ludb.py export`, then the `matlab -batch` line in its docstring, then `... score` |
| TimesFM 3 states for all 200 records | `uv run modal run scripts/extract_states.py` |
| Development deep dive (fold 0) | `uv run modal run scripts/m3_classes.py`, then `uv run python scripts/m3_report.py` |
| Final runs: head, probe, U-Net × 5 folds × 3 seeds | `uv run modal deploy scripts/train_fold.py`, `uv run python scripts/launch_m4.py results/m4/calls.json`, then `uv run modal run scripts/train_fold.py::fetch --run m4_fold<k>` |
| Report and verdict | `uv run python scripts/final_report.py` |

**Step 2: pretraining and the production detector**

| Stage | Command |
|---|---|
| Choose the PTB-XL ECGs (local) | `uv run python scripts/ptbxl_pretrain_data.py select` |
| Fetch them with ECGdeli labels | `uv run modal run scripts/ptbxl_fetch.py` |
| PTB-XL states | `uv run modal deploy scripts/extract_states_ptbxl.py`, `uv run python scripts/extract_states_ptbxl.py launch`, then `uv run modal run scripts/extract_states_ptbxl.py::collect` |
| Pretrain on PTB-XL+ | `uv run modal deploy scripts/pretrain_ptbxl.py`, `uv run python scripts/pretrain_ptbxl.py launch`, then `uv run modal run scripts/pretrain_ptbxl.py::fetch` |
| Zero-shot and fine-tuned on LUDB | `uv run modal run scripts/ludb_pretrained.py::zero`; `uv run modal deploy scripts/ludb_pretrained.py`, `uv run python scripts/ludb_pretrained.py launch` |
| RDB labels and states | `uv run modal run scripts/rdb_prepare.py::split`, `uv run modal run scripts/rdb_prepare.py`; then `extract_states_rdb.py` as for PTB-XL |
| Pretrain on RDB, from the PTB-XL+ head | `uv run python scripts/pretrain_ptbxl.py launch dataset=rdb` |
| Fine-tune on LUDB from the RDB head | `uv run python scripts/ludb_pretrained.py launch pretrained=rdb_pretrain4_s{seed} prefix=ludb_finetune_rdb_fold{fold}` |
| Report and verdict | `uv run python scripts/pretrained_report.py` |
| Production detector (all 200 people, 3 seeds) | `uv run python scripts/ludb_pretrained.py launch_production` |
| Deck figures | `uv run python scripts/deck_figures.py` |

Checks: `uv run pytest` (add `-m data` once LUDB is in place).

## Licences

- This code: [MIT](LICENSE).
- TimesFM 3: code Apache 2.0; weights under the TimesFM Non-Commercial License v1.0, research only.
  For that reason trained heads are not distributed; they stay in the Modal Volume under `/models/`.
- LUDB: ODC Attribution 1.0. Cite Kalyakulina et al., IEEE Access 8:186181 (2020), and PhysioNet.
- PTB-XL and PTB-XL+: CC BY 4.0. Cite Wagner et al., Sci Data 7:154 (2020), and Strodthoff et al.,
  Sci Data 10:279 (2023).
- RDB: CC BY 4.0. Cite the figshare record, doi:10.6084/m9.figshare.28892186.
- NeuroKit2: MIT. ECGdeli: GPL-3.0 (called from MATLAB, not included).

## Notes

- The commit history was rewritten on 2026-11-02 to leave out the paper draft and third-party PDFs,
  which changed almost every commit hash. Hashes recorded inside `results/*.json` refer to the old
  history; [docs/commit-map.tsv](docs/commit-map.tsv) maps them to the new ones. The M3 run
  (`results/m3/`) came from commit `aa38779`, before the freeze.
- The paper draft named in `docs/plan.md` and decision 66 is kept private until submission.
