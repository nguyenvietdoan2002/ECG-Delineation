# ecgdelin

Can frozen [TimesFM 3](https://github.com/google-research/timesfm) states, with a small trained
head, delineate P, QRS and T waves better than NeuroKit2 and better than a plain 1-D U-Net trained on
the same data? Step 1 answers this on [LUDB](https://physionet.org/content/ludb/1.0.1/).

- [docs/plan.md](docs/plan.md) — the experiment plan and stop rule
- [docs/spec.md](docs/spec.md) — the implementation spec
- [docs/decision-log.md](docs/decision-log.md) — every decision and why

## Setup

```powershell
uv sync                       # local environment: CPU PyTorch, tests only
```

TimesFM 3 is imported from the sibling checkout `../timesfm/src` (set `TIMESFM_SRC` to override).
Locally that path is added through `.venv/Lib/site-packages/timesfm_src.pth`; the Modal image copies
the `timesfm3` package from the same place. LUDB is read in place from
`../20260926_Elsevier 2025/data/ludb/...` (set `LUDB_DIR` to override).

Experiments run on Modal (`modal run scripts/<step>.py`); nothing trains on the laptop. The Modal
Volume `ecg-delineation` holds LUDB under `/ludb` (uploaded once with `modal volume put`), and every
run writes its predictions and results there as well as into `results/` here.

```powershell
uv run pytest                 # unit tests (seconds, no GPU)
uv run pytest -m data         # tests that read the local LUDB copy
uv run modal run scripts/smoke.py          # image + Volume check on CPU and T4
uv run modal run scripts/run_neurokit.py   # M1: NeuroKit2 prominence baseline over all of LUDB
uv run modal run scripts/run_neurokit.py --rpeaks emrich2023   # paper-faithful variant (FastNVG R-peaks)
uv run modal run scripts/gpu_tests.py      # the gpu-marked tests (real TimesFM 3) inside a T4 container
uv run modal run scripts/extract_states.py # M2: TimesFM 3 states for all 200 records into the Volume
uv run modal run scripts/train_fold.py     # M3: probe, main head and U-Net on fold 0, scored on its validation people
uv run modal run scripts/train_fold.py --max-epochs 2 --models head   # a smoke run
uv run modal run scripts/m3_classes.py     # M3 deep dive, step 1: class maps of the fold-0 models on the validation people (CPU)
uv run python scripts/m3_report.py         # M3 deep dive, step 2: docs/m3-report.md and results/m3/fig_*.png (local)
```

The `gpu` tests are deselected locally; the laptop never loads the backbone.

Results: `results/<method>.json` (all metrics, with the Emrich 2024 comparison for lead I),
`results/m3/<run>.json` (M3 training histories and validation-people scores, with the deep dive in
[docs/m3-report.md](docs/m3-report.md)) and `results/predictions/<name>.json` (every predicted point). Trained weights stay in the Volume under
`/models/<run>/`. References are in [docs/references.md](docs/references.md).

## Licences

- TimesFM 3 weights: TimesFM Non-Commercial License v1.0 — research only; trained heads are not
  distributed.
- LUDB: ODC Attribution 1.0 — cite Kalyakulina et al., IEEE Access 8:186181 (2020) and PhysioNet.
- NeuroKit2: MIT.
