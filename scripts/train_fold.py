"""M4: train the linear probe, the main head and the U-Net on one fold, three seeds each, and predict
the fold's test people (decisions 31, 58, 64 and 65).

    modal run scripts/train_fold.py::main --fold 0                      # fold 0, seeds 0,1,2, all three models
    modal run --detach scripts/train_fold.py::main --folds 0,1,2,3,4    # every fold, one container each, in parallel
    modal run scripts/train_fold.py::main --fold 0 --max-epochs 2       # the rehearsal (run name <run>_smoke)
    modal run scripts/train_fold.py::fetch --run m4_fold0               # copy a finished run's files from the Volume

One T4 container per fold loads the fold's training and validation states into RAM once (about
29 GiB), normalises each batch's layers once for every head, trains the nine models jointly over the
same batches (each with its own AdamW and early stopping), checkpoints to the Volume after each epoch
so a preempted container resumes, saves the final weights under /vol/models/<run>/ (never downloaded:
the TimesFM licence is non-commercial), then streams the fold's test people one at a time and writes
their predictions for every model and seed. The fold's test people are never read before training
ends. Scores over all 200 people and the verdict come from scripts/final_report.py; the per-fold test
report in the JSON is for information only (decision 64).
"""

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "splits" / "ludb_5fold_seed0.json"
MANIFEST = REPO / "results" / "states_manifest.json"
RESULTS_DIR = REPO / "results" / "m4"
PREDICTIONS_DIR = REPO / "results" / "predictions"
MODELS = ("probe", "head", "unet")


@app.function(gpu="T4", volumes={VOL: volume}, timeout=12 * 3600, memory=(40960, 65536))
def train_fold(
    run: str,
    fold: int,
    seeds: list[int],
    model_names: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    revision: str,
    max_epochs: int,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    import gc
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    from ecgdelin.dataset import load_record
    from ecgdelin.models import ConvHead, LinearProbe, UNet1D, count_parameters, normalize_layers
    from ecgdelin.paths import ludb_root
    from ecgdelin.preprocess import N_SAMPLES, prepare
    from ecgdelin.score import score
    from ecgdelin.states import N_BLOCKS
    from ecgdelin.train import TensorSource, TrainConfig, make_trainee, targets_for, train_jointly
    from ecgdelin.trained import classes_of, points_from_classes

    device = torch.device("cuda")
    config = TrainConfig(max_epochs=max_epochs)
    ids = list(train_ids) + list(val_ids)
    train_idx = list(range(len(train_ids)))
    val_idx = list(range(len(train_ids), len(ids)))
    t_start = time.perf_counter()

    # Signals, labels and states for the fold's training and validation people (decision 31).
    root = ludb_root()
    recordings = [load_record(root, rec_id) for rec_id in ids]
    signals = torch.from_numpy(np.stack([prepare(rec.signal, rec.fs) for rec in recordings]))  # (N, 12, 4992) float32
    targets = torch.stack([targets_for(rec.marks, N_SAMPLES) for rec in recordings])  # (N, 12, 4992) int64, -1 = masked
    states_dir = Path(VOL) / "states" / revision
    states = np.empty((len(ids), 20, 12, N_BLOCKS, 2560), dtype=np.float16)

    def load_states(k: int) -> None:
        states[k] = np.load(states_dir / f"{ids[k]}.npy")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(load_states, range(len(ids))))
    load_seconds = time.perf_counter() - t_start
    print(f"fold {fold}: loaded {len(ids)} recordings ({len(train_ids)} train, {len(val_ids)} val), states {states.nbytes / 2**30:.1f} GiB, in {load_seconds:.0f} s")

    # Every head reads the same per-layer-normalised batch (decision 56), normalised once per batch.
    sources = {"states": TensorSource(torch.from_numpy(states), transform=normalize_layers), "signal": TensorSource(signals)}
    trainees = []
    for seed in seeds:
        for name in model_names:
            torch.manual_seed(seed)  # initialisation depends on the seed only, not on which other models run
            if name == "probe":
                trainees.append(make_trainee(f"probe_s{seed}", LinearProbe(n_layers=20, d_in=2560), "states", config, normalized=True))
            elif name == "head":
                trainees.append(make_trainee(f"head_s{seed}", ConvHead(n_layers=20, d_in=2560), "states", config, normalized=True))
            elif name == "unet":
                trainees.append(make_trainee(f"unet_s{seed}", UNet1D(), "signal", config))
            else:
                raise ValueError(f"unknown model {name!r}; choose from {MODELS}")

    checkpoint = Path(VOL) / "checkpoints" / f"{run}.pt"
    shuffle_seed = int(seeds[0])  # one data stream for all seeds and models (decision 31)
    t_train = time.perf_counter()
    train_jointly(trainees, sources, targets, train_idx, val_idx, config, shuffle_seed, device, checkpoint=checkpoint, after_epoch=lambda _epoch: volume.commit())
    train_seconds = time.perf_counter() - t_train

    models_dir = Path(VOL) / "models" / run
    models_dir.mkdir(parents=True, exist_ok=True)
    for t in trainees:
        torch.save(t.model.state_dict(), models_dir / f"{t.name}.pt")

    # The fold's test people, read only now, one at a time (their states never enter RAM together).
    del sources, states
    gc.collect()
    test_recordings = [load_record(root, rec_id) for rec_id in test_ids]
    predictions: dict[str, dict[str, Any]] = {t.name: {} for t in trainees}
    t_test = time.perf_counter()
    for rec in test_recordings:
        x = prepared = prepare(rec.signal, rec.fs)
        signal = torch.from_numpy(prepared)[None]
        normalised = normalize_layers(torch.from_numpy(np.load(states_dir / f"{rec.id}.npy"))[None].to(device))
        for t in trainees:
            classes = classes_of(t.model, normalised, normalized=True) if t.source == "states" else classes_of(t.model, signal)
            predictions[t.name][rec.id] = points_from_classes(classes, x, rec.fs)
        del normalised
    test_seconds = time.perf_counter() - t_test
    reports = {t.name: score(test_recordings, predictions[t.name]) for t in trainees}

    result: dict[str, Any] = {
        "name": run,
        "scope": (
            f"fold {fold}: {len(test_ids)} test people predicted by models trained on the other folds. One fold of the pre-registered "
            "5-fold run; per-fold numbers are for information only, the verdict is computed over all 200 people by scripts/final_report.py (decision 64)."
        ),
        "meta": {
            **meta,
            "fold": fold,
            "seeds": list(seeds),
            "shuffle_seed": shuffle_seed,
            "states_revision": revision,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "test_ids": list(test_ids),
            "load_seconds": load_seconds,
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
            "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        },
        "config": config.to_dict(),
        "models": {
            t.name: {
                "model": t.name.rsplit("_s", 1)[0],
                "seed": int(t.name.rsplit("_s", 1)[1]),
                "n_params": count_parameters(t.model),
                **t.summary(),
                "layer_weights": t.model.blend.weights() if hasattr(t.model, "blend") else None,
                "test_report": reports[t.name].to_dict(),
            }
            for t in trainees
        },
    }
    serial: dict[str, list[dict[str, Any]]] = {name: [] for name in model_names}
    for t in trainees:
        base, seed = t.name.rsplit("_s", 1)
        serial[base] += [{"id": rec.id, "seed": int(seed), "leads": {lead: lp.to_dict() for lead, lp in predictions[t.name][rec.id].items()}} for rec in test_recordings]

    results_dir = Path(VOL) / "results"
    preds_dir = Path(VOL) / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, preds in serial.items():
        (preds_dir / f"{run}_{name}.json").write_text(json.dumps(preds), encoding="utf-8")
    volume.commit()
    return result, serial


def _git_version() -> str:
    """The commit, with the tag when one points at it (``step1-frozen`` for the final runs) and ``-dirty`` for uncommitted changes."""
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _write(run_name: str, result: dict[str, Any], serial: dict[str, list[dict[str, Any]]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    for name, preds in serial.items():
        (PREDICTIONS_DIR / f"{run_name}_{name}.json").write_text(json.dumps(preds), encoding="utf-8", newline="\n")


def _print_summary(result: dict[str, Any]) -> None:
    meta = result["meta"]
    print(f"\n{result['name']}: fold {meta['fold']}, seeds {meta['seeds']}, git {meta.get('git')}")
    print(
        f"{meta['n_train']} training, {meta['n_val']} validation, {meta['n_test']} test people; load {meta['load_seconds']:.0f} s, "
        f"train {meta['train_seconds'] / 60:.1f} min, test {meta['test_seconds']:.0f} s, GPU memory {meta['max_gpu_memory_gib']:.1f} GiB"
    )
    print(f"{'model':10s} {'params':>9s} {'epochs':>6s} {'best':>5s} {'val loss':>8s} {'test primary SD':>15s}  within-SD P_on/P_off/QRS_on/QRS_off/T_off (this fold only)")
    for name, m in result["models"].items():
        pts = m["test_report"]["points"]
        within = "/".join(f"{pts[p]['sd_within_ms']:.1f}" for p in ("P_on", "P_off", "QRS_on", "QRS_off", "T_off"))
        print(f"{name:10s} {m['n_params']:9d} {m['epochs_run']:6d} {m['best_epoch']:5d} {m['best_val_loss']:8.4f} {m['test_report']['primary_sd_ms']:15.2f}  {within}")


def _plan(fold: int, seeds: str, models: str, max_epochs: int, run: str, revision: str) -> tuple[str, dict[str, Any]]:
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    run_name = run or f"m4_fold{fold}" + ("" if max_epochs >= 200 else "_smoke")
    revision = revision or json.loads(MANIFEST.read_text(encoding="utf-8"))["revision"]
    folds = json.loads(SPLITS.read_text(encoding="utf-8"))["folds"][fold]
    meta = {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_version(), "models": model_names}
    args = dict(run=run_name, fold=fold, seeds=seed_list, model_names=model_names, train_ids=list(folds["train"]), val_ids=list(folds["val"]), test_ids=list(folds["test"]), revision=revision, max_epochs=max_epochs, meta=meta)
    return run_name, args


@app.local_entrypoint()
def main(fold: int = 0, seeds: str = "0,1,2", models: str = "probe,head,unet", max_epochs: int = 200, run: str = "", revision: str = "", folds: str = "") -> None:
    fold_list = [int(f) for f in folds.split(",") if f.strip()] or [fold]
    calls = []
    for f in fold_list:
        run_name, args = _plan(f, seeds, models, max_epochs, run if len(fold_list) == 1 else "", revision)
        calls.append((run_name, train_fold.spawn(**args)))
        print(f"spawned {run_name}")
    for run_name, call in calls:
        result, serial = call.get()
        _write(run_name, result, serial)
        _print_summary(result)


@app.local_entrypoint()
def fetch(run: str) -> None:
    """Copy a finished run's JSON and predictions from the Volume into results/ (for runs that finished after the client left)."""
    result = json.loads(b"".join(volume.read_file(f"results/{run}.json")).decode("utf-8"))
    serial = {}
    for name in result["meta"]["models"]:
        serial[name] = json.loads(b"".join(volume.read_file(f"predictions/{run}_{name}.json")).decode("utf-8"))
    _write(run, result, serial)
    _print_summary(result)
