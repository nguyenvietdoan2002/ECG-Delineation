"""Steps 5 and 6 of the plan (decisions 72 and 78): the PTB-XL+-pretrained head on LUDB.

    modal run scripts/ludb_pretrained.py::zero --smoke      # arm A on 3 people (a few cents)
    modal run scripts/ludb_pretrained.py::zero              # arm A: the pretrained head on all 200 people, no LUDB training
    modal deploy scripts/ludb_pretrained.py                 # then
    python scripts/ludb_pretrained.py launch                # arm B: spawn the 5 fold containers on the deployed app and exit
    python scripts/ludb_pretrained.py status                # which folds are done
    modal run scripts/ludb_pretrained.py::fetch --run ludb_finetune4_fold0   # copy a finished run into results/
    python scripts/pretrained_report.py                     # scores, comparisons and the verdict

Arm A streams the 200 people's step-1 states one at a time through the pretrained head (all 12 leads;
the head treats each lead on its own) and decodes with step 1's decoder. Arm B is step 1's
``train_fold`` for heads only, each started from pretrained weights: one T4 container per fold loads
the fold's training and validation states (about 29 GiB), fine-tunes one head per seed with step 1's
recipe (AdamW, lr 1e-3, weight decay 1e-4, batch 8, up to 200 epochs, patience 20 on the same
validation people), then predicts the fold's test people. Weights stay in /vol/models/<run>/ and are
never downloaded (TimesFM 3 licence, non-commercial).
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "splits" / "ludb_5fold_seed0.json"
MANIFEST = REPO / "results" / "states_manifest.json"
OUT_DIR = REPO / "results" / "ludb_pretrained"
PREDICTIONS_DIR = REPO / "results" / "predictions"
CALLS = OUT_DIR / "calls.json"
PRETRAINED = "ptbxl_pretrain4_s{seed}"  # step 4's run; weights at /vol/models/<run>/head_s<seed>.pt
ZERO = "ludb_zeroshot4_s{seed}"
FINETUNE = "ludb_finetune4_fold{fold}"


def _weights(run: str, seed: int) -> str:
    return f"{VOL}/models/{run}/head_s{seed}.pt"


def _states(path: Path, device: Any) -> Any:
    """One person's stored states on the device, with step 1's per-layer normalisation applied."""
    import torch

    from ecgdelin.models import normalize_layers
    from ecgdelin.pretrain import read_npy

    return normalize_layers(torch.from_numpy(read_npy(path))[None].to(device))


def _points(model: Any, normalised: Any, rec: Any) -> dict[str, Any]:
    """Points per lead from step 1's argmax and decoder, peaks read from the signal the model saw."""
    from ecgdelin.preprocess import prepare
    from ecgdelin.trained import classes_of, points_from_classes

    x = prepare(rec.signal, rec.fs)
    return points_from_classes(classes_of(model, normalised, normalized=True), x, rec.fs)


@app.function(gpu="T4", volumes={VOL: volume}, timeout=2 * 3600, memory=16384)
def zero_shot(run: str, weights: str, seed: int, ids: list[str], revision: str, meta: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import time

    import torch

    from ecgdelin.dataset import load_record
    from ecgdelin.models import ConvHead, count_parameters
    from ecgdelin.paths import ludb_root

    device = torch.device("cuda")
    model = ConvHead(n_layers=20, d_in=2560)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(device)
    root = ludb_root()
    states_dir = Path(VOL) / "states" / revision
    t0 = time.perf_counter()
    serial = []
    for k, rec_id in enumerate(ids):
        points = _points(model, _states(states_dir / f"{rec_id}.npy", device), load_record(root, rec_id))
        serial.append({"id": rec_id, "seed": seed, "leads": {lead: lp.to_dict() for lead, lp in points.items()}})
        if k % 25 == 0:
            print(f"{run}: {k + 1} of {len(ids)} people, {time.perf_counter() - t0:.0f} s", flush=True)
    result = {
        "name": run,
        "scope": "Arm A (decision 72): the PTB-XL+-pretrained head on LUDB people it never trained on, no LUDB training; scored by scripts/pretrained_report.py.",
        "meta": {**meta, "seed": seed, "weights": weights, "parameters": count_parameters(model), "states_revision": revision, "n_people": len(ids),
                 "device": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "seconds": time.perf_counter() - t0,
                 "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")},
    }
    for sub, name, data in (("results", f"{run}.json", result), ("predictions", f"{run}_head.json", serial)):
        (Path(VOL) / sub).mkdir(parents=True, exist_ok=True)
        (Path(VOL) / sub / name).write_text(json.dumps(data, indent=1 if sub == "results" else None), encoding="utf-8")
    volume.commit()
    return result, serial


@app.function(gpu="T4", volumes={VOL: volume}, timeout=12 * 3600, memory=(40960, 65536), retries=1)
def finetune_fold(
    run: str,
    fold: int,
    init: dict[str, str],
    streams: str,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    revision: str,
    max_epochs: int,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    from ecgdelin.dataset import load_record
    from ecgdelin.finetune import finetune_heads
    from ecgdelin.models import ConvHead, count_parameters, normalize_layers
    from ecgdelin.paths import ludb_root
    from ecgdelin.preprocess import N_SAMPLES
    from ecgdelin.pretrain import read_npy
    from ecgdelin.score import score
    from ecgdelin.states import N_BLOCKS
    from ecgdelin.train import TensorSource, TrainConfig, targets_for

    device = torch.device("cuda")
    volume.reload()
    config = TrainConfig(max_epochs=max_epochs)
    ids = list(train_ids) + list(val_ids)
    t_start = time.perf_counter()

    # The fold's training and validation people, as in step 1's train_fold (decision 31).
    root = ludb_root()
    recordings = [load_record(root, rec_id) for rec_id in ids]
    targets = torch.stack([targets_for(rec.marks, N_SAMPLES) for rec in recordings])
    states_dir = Path(VOL) / "states" / revision
    states = np.empty((len(ids), 20, 12, N_BLOCKS, 2560), dtype=np.float16)

    def load_states(k: int) -> None:
        states[k] = read_npy(states_dir / f"{ids[k]}.npy")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(load_states, range(len(ids))))
    load_seconds = time.perf_counter() - t_start
    print(f"fold {fold}: loaded {len(ids)} people ({len(train_ids)} train, {len(val_ids)} val), {states.nbytes / 2**30:.1f} GiB in {load_seconds:.0f} s", flush=True)

    starts = {int(seed): torch.load(path, map_location="cpu", weights_only=True) for seed, path in init.items()}
    sources = {"states": TensorSource(torch.from_numpy(states), transform=normalize_layers)}
    t_train = time.perf_counter()
    trainees = finetune_heads(lambda: ConvHead(n_layers=20, d_in=2560), starts, sources, targets, list(range(len(train_ids))),
                              list(range(len(train_ids), len(ids))), config, device, streams=streams,
                              checkpoint_dir=Path(VOL) / "checkpoints", run=run, log=lambda s: print(s, flush=True),
                              after_epoch=lambda _epoch: volume.commit())
    train_seconds = time.perf_counter() - t_train
    models_dir = Path(VOL) / "models" / run
    models_dir.mkdir(parents=True, exist_ok=True)
    for t in trainees:
        torch.save(t.model.state_dict(), models_dir / f"{t.name}.pt")

    # The fold's test people, read only now, one at a time.
    del sources, states
    test_recordings = [load_record(root, rec_id) for rec_id in test_ids]
    predictions: dict[str, dict[str, Any]] = {t.name: {} for t in trainees}
    t_test = time.perf_counter()
    for rec in test_recordings:
        normalised = _states(states_dir / f"{rec.id}.npy", device)
        for t in trainees:
            predictions[t.name][rec.id] = _points(t.model, normalised, rec)
        del normalised
    test_seconds = time.perf_counter() - t_test
    reports = {t.name: score(test_recordings, predictions[t.name]) for t in trainees} if test_ids else {}

    result: dict[str, Any] = {
        "name": run,
        "scope": ("Production detector (decision 91): one head per seed trained on all LUDB people but the stopping set, from the current best pretrained head; no test people, "
                  "its quality is the cross-validated figure of the same recipe." if fold < 0 else
                  f"Arm B (decision 72), fold {fold}: {len(test_ids)} test people predicted by heads started from pretrained weights and trained "
                  "on the other folds with step 1's recipe; per-fold numbers are for information only, the verdict comes from scripts/pretrained_report.py."),
        "meta": {**meta, "fold": fold, "init": init, "streams": streams, "states_revision": revision, "device": torch.cuda.get_device_name(0),
                 "torch": str(torch.__version__),  # str: TorchVersion would need torch to unpickle locally
                 "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids), "train_ids": list(train_ids),
                 "val_ids": list(val_ids), "test_ids": list(test_ids), "load_seconds": load_seconds, "train_seconds": train_seconds,
                 "test_seconds": test_seconds, "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")},
        "config": config.to_dict(),
        "models": {t.name: {"model": "head", "seed": int(t.name.rsplit("_s", 1)[1]), "n_params": count_parameters(t.model), **t.summary(),
                            "layer_weights": t.model.blend.weights(), "test_report": reports[t.name].to_dict() if test_ids else None} for t in trainees},
    }
    serial = [{"id": rec.id, "seed": int(t.name.rsplit("_s", 1)[1]), "leads": {lead: lp.to_dict() for lead, lp in predictions[t.name][rec.id].items()}}
              for t in trainees for rec in test_recordings]
    for sub, name, data in (("results", f"{run}.json", result), ("predictions", f"{run}_head.json", serial)):
        (Path(VOL) / sub).mkdir(parents=True, exist_ok=True)
        (Path(VOL) / sub / name).write_text(json.dumps(data, indent=1 if sub == "results" else None), encoding="utf-8")
    volume.commit()
    return result, serial


@app.function(volumes={VOL: volume}, timeout=600)
def read_run(run: str) -> tuple[str, str]:
    volume.reload()
    return ((Path(VOL) / "results" / f"{run}.json").read_text(encoding="utf-8"),
            (Path(VOL) / "predictions" / f"{run}_head.json").read_text(encoding="utf-8"))


def _git_head() -> str:
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _meta() -> dict[str, Any]:
    return {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_head()}


def _revision() -> str:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["revision"]


def _folds() -> list[dict[str, Any]]:
    return json.loads(SPLITS.read_text(encoding="utf-8"))["folds"]


def _write(result: dict[str, Any], serial: list[dict[str, Any]] | str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{result['name']}.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    text = serial if isinstance(serial, str) else json.dumps(serial)
    (PREDICTIONS_DIR / f"{result['name']}_head.json").write_text(text, encoding="utf-8", newline="\n")


def _print(result: dict[str, Any]) -> None:
    m = result["meta"]
    print(f"{result['name']}: git {m.get('git')}, {m.get('n_people') or m.get('n_test')} people")
    for name, r in result.get("models", {}).items():
        print(f"  {name}: {r['epochs_run']} epochs, best {r['best_epoch']}, val loss {r['best_val_loss']:.4f}, fold test primary SD {r['test_report']['primary_sd_ms']:.2f} ms")


def _finetune_plan(fold: int, seeds: list[int], init_seeds: list[int], streams: str, max_epochs: int, pretrained: str = PRETRAINED, prefix: str = FINETUNE) -> dict[str, Any]:
    """A smoke run's name carries the time, so it never resumes an older smoke run's checkpoint."""
    if len(init_seeds) != len(seeds):
        raise ValueError(f"one pretrained seed per seed: {seeds} vs {init_seeds}")
    f = _folds()[fold]
    smoke = "" if max_epochs >= 200 else "_smoke_" + dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    run = prefix.format(fold=fold) + smoke
    init = {str(s): _weights(pretrained.format(seed=i), i) for s, i in zip(seeds, init_seeds)}
    return dict(run=run, fold=fold, init=init, streams=streams, train_ids=list(f["train"]), val_ids=list(f["val"]), test_ids=list(f["test"]),
                revision=_revision(), max_epochs=max_epochs, meta=_meta())


def _production_plan(seeds: list[int], init_seeds: list[int], streams: str, max_epochs: int, pretrained: str, run: str) -> dict[str, Any]:
    """Decision 91: every LUDB person trains except fold 0's 32 validation people, who stop the training; no test people."""
    if len(init_seeds) != len(seeds):
        raise ValueError(f"one pretrained seed per seed: {seeds} vs {init_seeds}")
    folds = _folds()
    everyone = sorted({p for f in folds for p in f["test"]}, key=int)
    val = list(folds[0]["val"])
    train = [p for p in everyone if p not in set(val)]
    smoke = "" if max_epochs >= 200 else "_smoke_" + dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    init = {str(s): _weights(pretrained.format(seed=i), i) for s, i in zip(seeds, init_seeds)}
    return dict(run=run + smoke, fold=-1, init=init, streams=streams, train_ids=train, val_ids=val, test_ids=[], revision=_revision(), max_epochs=max_epochs, meta=_meta())


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


@app.local_entrypoint()
def zero(seed: int = 0, smoke: bool = False, pretrained: str = PRETRAINED, run: str = "") -> None:
    """Arm A; ``pretrained`` names another pretraining run (``rdb_pretrain4_s{seed}``), ``run`` its result name."""
    ids = sorted({p for f in _folds() for p in f["test"]}, key=int)
    run = (run or ZERO.format(seed=seed)) + ("_smoke" if smoke else "")
    result, serial = zero_shot.remote(run, _weights(pretrained.format(seed=seed), seed), seed, ids[:3] if smoke else ids, _revision(), _meta())
    _write(result, serial)
    print(f"{run}: {len(serial)} people in {result['meta']['seconds']:.0f} s; written to {OUT_DIR}")


@app.local_entrypoint()
def finetune_smoke(fold: int = 0) -> None:
    """Two epochs, two seeds, on one fold: the whole arm-B path at a few cents."""
    plan = _finetune_plan(fold, [0, 1], [0, 0], "per_seed", 2)
    result, serial = finetune_fold.remote(**plan)
    _write(result, serial)
    _print(result)


@app.local_entrypoint()
def fetch(run: str) -> None:
    text, preds = read_run.remote(run)
    result = json.loads(text)
    _write(result, preds)
    _print(result)


def _calls_file(prefix: str) -> Path:
    return CALLS if prefix == FINETUNE else OUT_DIR / f"calls_{prefix.format(fold='').rstrip('_')}.json"


def launch(folds: str = "0,1,2,3,4", seeds: str = "0,1,2", init_seeds: str = "0,0,0", streams: str = "per_seed", max_epochs: int = 200,
           pretrained: str = PRETRAINED, prefix: str = FINETUNE) -> None:
    """``pretrained`` names the pretraining run to start from, ``prefix`` the fold runs (both with {seed} / {fold}):
    python scripts/ludb_pretrained.py launch pretrained=rdb_pretrain4_s{seed} prefix=ludb_finetune_rdb_fold{fold}"""
    import modal

    fn = modal.Function.from_name(app.name, "finetune_fold")
    calls = {}
    for fold in _ints(folds):
        plan = _finetune_plan(fold, _ints(seeds), _ints(init_seeds), streams, max_epochs, pretrained, prefix)
        calls[plan["run"]] = fn.spawn(**plan).object_id
        print("spawned", plan["run"], calls[plan["run"]])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _calls_file(prefix).write_text(json.dumps({"launched": _meta(), "pretrained": pretrained, "prefix": prefix, "streams": streams, "seeds": seeds, "init_seeds": init_seeds, "calls": calls}, indent=1),
                                   encoding="utf-8", newline="\n")


def launch_production(seeds: str = "0,1,2", init_seeds: str = "0,0,0", streams: str = "per_seed", max_epochs: int = 200,
                      pretrained: str = "rdb_pretrain4_s{seed}", run: str = "ludb_production_rdb") -> None:
    """python scripts/ludb_pretrained.py launch_production [max_epochs=2]; then status prefix=<run>"""
    import modal

    plan = _production_plan(_ints(seeds), _ints(init_seeds), streams, max_epochs, pretrained, run)
    call = modal.Function.from_name(app.name, "finetune_fold").spawn(**plan)
    print("spawned", plan["run"], call.object_id, f"({len(plan['train_ids'])} train, {len(plan['val_ids'])} stopping)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _calls_file(plan["run"]).write_text(json.dumps({"launched": _meta(), "pretrained": pretrained, "prefix": plan["run"], "streams": streams, "seeds": seeds, "init_seeds": init_seeds,
                                                   "calls": {plan["run"]: call.object_id}}, indent=1), encoding="utf-8", newline="\n")


def status(prefix: str = FINETUNE) -> None:
    import modal

    for run, call_id in json.loads(_calls_file(prefix).read_text(encoding="utf-8"))["calls"].items():
        try:
            result, serial = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            print(run, "running")
            continue
        _write(result, serial)
        print(run, "done, written to results/")


if __name__ == "__main__":
    # python scripts/ludb_pretrained.py launch [streams=per_seed] [init_seeds=0,0,0] [seeds=0,1,2] [folds=0,1,2,3,4]
    options = dict(arg.split("=", 1) for arg in sys.argv[2:])
    if "max_epochs" in options:
        options["max_epochs"] = int(options["max_epochs"])
    {"launch": launch, "launch_production": launch_production, "status": status}[sys.argv[1]](**options)
