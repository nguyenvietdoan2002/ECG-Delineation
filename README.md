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

Experiments run on Modal (`modal run scripts/<step>.py`); nothing trains on the laptop.

```powershell
uv run pytest                 # unit tests (seconds, no GPU)
uv run pytest -m data         # tests that read the local LUDB copy
```

## Licences

- TimesFM 3 weights: TimesFM Non-Commercial License v1.0 — research only; trained heads are not
  distributed.
- LUDB: ODC Attribution 1.0 — cite Kalyakulina et al., IEEE Access 8:186181 (2020) and PhysioNet.
- NeuroKit2: MIT.
