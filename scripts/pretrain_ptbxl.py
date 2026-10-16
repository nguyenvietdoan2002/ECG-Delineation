"""Step 4 of the plan (decision 77): pretrain step 1's ConvHead on the 4,500 PTB-XL ECGs' ECGDeli labels,
stopping on the 500 fold-9 ECGs.

    modal run scripts/pretrain_ptbxl.py::smoke                 # 64 + 16 ECGs, one epoch: checks the chain
    modal deploy scripts/pretrain_ptbxl.py                     # then
    python scripts/pretrain_ptbxl.py launch                    # spawn seed 0 on the deployed app and exit
    python scripts/pretrain_ptbxl.py status                    # running or done
    modal run scripts/pretrain_ptbxl.py::fetch                 # copy the result JSON into results/

States stream from /vol/states/<revision>/ptbxl/; targets come from /vol/ptbxl/records/<id>.npz. The
weights go to /vol/models/<run>/head_s<seed>.pt and are never downloaded (the TimesFM licence is
non-commercial). The call is spawned on the deployed function, so it runs without this client.
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "splits" / "ptbxl_pretrain_seed0.json"
STEP1_MANIFEST = REPO / "results" / "states_manifest.json"
CALLS = REPO / "results" / "ptbxl_pretrain_calls.json"
RUN = "ptbxl_pretrain_s{seed}"


@app.function(gpu="T4", cpu=4.0, volumes={VOL: volume}, timeout=12 * 3600, memory=(16384, 32768), retries=2)  # 4 cores: file reads starved the heartbeat on the default share
def pretrain_head(run: str, seed: int, train_ids: list[int], stop_ids: list[int], revision: str, max_epochs: int, meta: dict[str, Any]) -> dict[str, Any]:
    import time

    import numpy as np
    import torch

    from ecgdelin.models import ConvHead, count_parameters, normalize_layers
    from ecgdelin.pretrain import FileSource, PretrainConfig, Targets, pretrain, targets_from_labels

    device = torch.device("cuda")
    ids = list(train_ids) + list(stop_ids)
    t0 = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor

    def target(i: int) -> np.ndarray:
        with np.load(Path(VOL) / "ptbxl" / "records" / f"{i}.npz") as z:
            return targets_from_labels(z["classes"], z["mask"])

    with ThreadPoolExecutor(16) as pool:
        targets_t = Targets(np.stack(list(pool.map(target, ids))))
    states_dir = Path(VOL) / "states" / revision / "ptbxl"
    source = FileSource([states_dir / f"{i}.npy" for i in ids], transform=normalize_layers, threads=8)
    print(f"{run}: {len(train_ids)} train, {len(stop_ids)} stop, targets loaded in {time.perf_counter() - t0:.0f} s", flush=True)

    torch.manual_seed(seed)
    model = ConvHead(n_layers=20, d_in=2560)
    config = PretrainConfig(max_epochs=max_epochs)
    out = pretrain(model, source, targets_t, list(range(len(train_ids))), list(range(len(train_ids), len(ids))), config, seed, device,
                   checkpoint=Path(VOL) / "checkpoints" / f"{run}.pt", log=lambda s: print(s, flush=True),
                   after_check=lambda _c: volume.commit(), model_kwargs={"normalized": True})
    models_dir = Path(VOL) / "models" / run
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / f"head_s{seed}.pt")
    result = {
        "name": run,
        "meta": {**meta, "seed": seed, "revision": revision, "n_train": len(train_ids), "n_stop": len(stop_ids),
                 "config": config.to_dict(), "parameters": count_parameters(model), "decisions": [76, 77]},
        "best_val_loss": out["best_val"],
        "best_check": out["best_check"],
        "best_epoch": out["best_epoch"],
        "stopped_early": out["stopped"],
        "train_seconds": out["seconds"],
        "history": out["history"],
        "weights": str(models_dir / f"head_s{seed}.pt"),
    }
    results_dir = Path(VOL) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    volume.commit()
    return result


@app.function(volumes={VOL: volume}, timeout=600)
def read_result(run: str) -> str:
    return (Path(VOL) / "results" / f"{run}.json").read_text(encoding="utf-8")


@app.function(volumes={VOL: volume}, timeout=600)
def read_progress(run: str) -> dict[str, Any]:
    """The running job's checks so far, from its checkpoint (written after every check)."""
    import torch

    volume.reload()
    path = Path(VOL) / "checkpoints" / f"{run}.pt"
    if not path.exists():
        return {"checks": 0, "history": []}
    state = torch.load(path, map_location="cpu", weights_only=False)["state"]
    return {"checks": state["check"], "history": state["history"], "best_check": state["best_check"],
            "best_val": state["best_val"], "stopped": state["stopped"], "seconds": state["seconds"]}


@app.local_entrypoint()
def progress(seed: int = 0) -> None:
    run = RUN.format(seed=seed)
    p = read_progress.remote(run)
    out = REPO / "results" / f"{run}_progress.json"
    out.write_text(json.dumps({"name": run, "read": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **p}, indent=1),
                   encoding="utf-8", newline="\n")
    print(f"{run}: {p['checks']} checks")
    for h in p["history"]:
        print(f"  epoch {h['epoch']:4.1f}: train {h['train_loss']:.4f}, stop-set {h['val_loss']:.4f}")


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _plan(limit: int = 0) -> tuple[str, list[int], list[int]]:
    revision = json.loads(STEP1_MANIFEST.read_text(encoding="utf-8"))["revision"]
    sample = json.loads(SPLIT.read_text(encoding="utf-8"))["sample"]
    train, stop = sample["train"], sample["stop"]
    if limit:
        train, stop = train[:limit], stop[: max(1, limit // 4)]
    return revision, train, stop


def _meta() -> dict[str, Any]:
    return {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_head()}


@app.local_entrypoint()
def smoke() -> None:
    revision, train, stop = _plan(limit=64)
    run = "ptbxl_pretrain_smoke_" + dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")  # fresh: no old checkpoint
    result = pretrain_head.remote(run, 0, train, stop, revision, 1, _meta())
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))
    for h in result["history"]:
        print(h)


@app.local_entrypoint()
def fetch(seed: int = 0) -> None:
    run = RUN.format(seed=seed)
    text = read_result.remote(run)
    (REPO / "results" / f"{run}.json").write_text(text, encoding="utf-8", newline="\n")
    r = json.loads(text)
    print(f"{run}: best stop-set loss {r['best_val_loss']:.4f} at epoch {r['best_epoch']}, stopped early {r['stopped_early']}, {r['train_seconds'] / 3600:.1f} h")


def launch(seed: int = 0) -> None:
    import modal

    revision, train, stop = _plan()
    run = RUN.format(seed=seed)
    call = modal.Function.from_name(app.name, "pretrain_head").spawn(run, seed, train, stop, revision, 20, _meta())
    CALLS.write_text(json.dumps({run: call.object_id}, indent=1), encoding="utf-8", newline="\n")
    print("spawned", run, call.object_id)


def status() -> None:
    import modal

    for run, call_id in json.loads(CALLS.read_text(encoding="utf-8")).items():
        try:
            r = modal.FunctionCall.from_id(call_id).get(timeout=0)
            print(run, "done: best epoch", r["best_epoch"], "loss", round(r["best_val_loss"], 4))
        except TimeoutError:
            print(run, "running")


if __name__ == "__main__":
    {"launch": launch, "status": status}[sys.argv[1]]()
