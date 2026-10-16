"""Step 4 of the plan (decisions 77 and 80): pretrain step 1's ConvHead on 4 leads of each of the 4,500
PTB-XL ECGs, stopping on 4 leads of each of the 500 fold-9 ECGs, with every state held in memory.

    modal run scripts/pretrain_ptbxl.py::smoke        # 16 + 8 ECGs, one epoch, in the real 320 GiB container
    modal deploy scripts/pretrain_ptbxl.py            # then
    python scripts/pretrain_ptbxl.py launch           # spawn packing + training on the deployed app and exit
    python scripts/pretrain_ptbxl.py status           # packing, training or done
    modal run scripts/pretrain_ptbxl.py::progress     # the checks so far, from the checkpoint
    modal run scripts/pretrain_ptbxl.py::fetch        # copy the result JSON into results/

Packing (CPU): five containers read every state file once with plain reads, keep its 4 assigned leads
and write 20 shards of 250 ECGs to /vol/ptbxl/packed4/ (about 300 GiB in all). Training (T4, 320 GiB):
loads the shards into one array and runs ``pretrain`` from memory. Streaming the full states failed:
numpy's reads held the interpreter lock, starving the heartbeat, and one container reads about
220 MB/s at best (decision 80). Weights stay in /vol/models/<run>/ and are never downloaded.
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
RUN = "ptbxl_pretrain4_s{seed}"
PACK = f"{VOL}/ptbxl/packed4"
LEADS_PER_ECG = 4
SHARD = 250
MEMORY = (327680, 344064)  # 320 GiB request, Modal's 336 GiB maximum as the limit


@app.function(volumes={VOL: volume}, cpu=2.0, memory=8192, timeout=3 * 3600, max_containers=5, retries=1)
def pack_shard(pack_dir: str, shard: int, ids: list[int], leads: list[list[int]], revision: str) -> dict[str, Any]:
    import os
    import time

    import numpy as np

    from ecgdelin.pretrain import read_npy, targets_from_labels

    out = Path(pack_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"shard{shard:02d}.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    states_dir = Path(VOL) / "states" / revision / "ptbxl"
    t0 = time.perf_counter()
    shape = (len(ids), 20, LEADS_PER_ECG, 156, 2560)
    tmp = out / f"shard{shard:02d}_states.npy.tmp"
    targets = []
    with open(tmp, "wb") as f:
        np.lib.format.write_array_header_2_0(f, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)), "fortran_order": False, "shape": shape})
        for ecg_id, keep in zip(ids, leads):
            states = read_npy(states_dir / f"{ecg_id}.npy")  # (20, 12, 156, 2560)
            f.write(memoryview(np.ascontiguousarray(states[:, keep])).cast("B"))
            with np.load(Path(VOL) / "ptbxl" / "records" / f"{ecg_id}.npz") as z:
                targets.append(targets_from_labels(z["classes"], z["mask"])[keep])
    os.replace(tmp, out / f"shard{shard:02d}_states.npy")
    np.save(out / f"shard{shard:02d}_targets.npy", np.stack(targets))
    info = {"shard": shard, "ids": ids, "leads": leads, "seconds": round(time.perf_counter() - t0, 1)}
    done.write_text(json.dumps(info), encoding="utf-8")
    volume.commit()
    return info


@app.function(gpu="T4", cpu=4.0, volumes={VOL: volume}, memory=MEMORY, timeout=8 * 3600, retries=1)
def pretrain_head_mem(run: str, pack_dir: str, n_train_shards: int, n_shards: int, seed: int, max_epochs: int, meta: dict[str, Any]) -> dict[str, Any]:
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    from ecgdelin.models import ConvHead, count_parameters, normalize_layers
    from ecgdelin.pretrain import MemorySource, PretrainConfig, Targets, pretrain

    device = torch.device("cuda")
    volume.reload()
    pack = Path(pack_dir)
    infos = [json.loads((pack / f"shard{s:02d}.json").read_text(encoding="utf-8")) for s in range(n_shards)]
    sizes = [len(i["ids"]) for i in infos]
    starts = np.cumsum([0] + sizes)
    t0 = time.perf_counter()
    data = np.empty((int(starts[-1]), 20, LEADS_PER_ECG, 156, 2560), dtype=np.float16)

    def load(s: int) -> None:
        with open(pack / f"shard{s:02d}_states.npy", "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, _, _ = header(f)
            if tuple(shape) != (sizes[s], 20, LEADS_PER_ECG, 156, 2560):
                raise ValueError(f"shard {s}: shape {shape}")
            view = memoryview(data[starts[s] : starts[s + 1]].reshape(-1).view(np.uint8))
            got = 0
            while got < len(view):
                n = f.readinto(view[got:])
                if not n:
                    raise EOFError(f"shard {s}: {got} of {len(view)} bytes")
                got += n

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(load, range(n_shards)))
    targets = Targets(np.concatenate([np.load(pack / f"shard{s:02d}_targets.npy") for s in range(n_shards)]))
    load_seconds = time.perf_counter() - t0
    n_train = int(starts[n_train_shards])
    print(f"{run}: {n_train} train + {int(starts[-1]) - n_train} stop ECGs x {LEADS_PER_ECG} leads, {data.nbytes / 2**30:.0f} GiB loaded in {load_seconds:.0f} s", flush=True)

    source = MemorySource(torch.from_numpy(data), transform=normalize_layers)
    torch.manual_seed(seed)
    model = ConvHead(n_layers=20, d_in=2560)
    config = PretrainConfig(max_epochs=max_epochs)
    out = pretrain(model, source, targets, list(range(n_train)), list(range(n_train, int(starts[-1]))), config, seed, device,
                   checkpoint=Path(VOL) / "checkpoints" / f"{run}.pt", log=lambda s: print(s, flush=True),
                   after_check=lambda _c: volume.commit(), model_kwargs={"normalized": True})
    models_dir = Path(VOL) / "models" / run
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / f"head_s{seed}.pt")
    result = {
        "name": run,
        "meta": {**meta, "seed": seed, "n_train": n_train, "n_stop": int(starts[-1]) - n_train, "leads_per_ecg": LEADS_PER_ECG,
                 "pack": pack_dir, "load_seconds": load_seconds, "config": config.to_dict(), "parameters": count_parameters(model),
                 "decisions": [76, 77, 80]},
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


@app.function(volumes={VOL: volume}, timeout=4 * 3600)
def pack_then_train(run: str, pack_dir: str, train_ids: list[int], stop_ids: list[int], revision: str, seed: int, max_epochs: int,
                    meta: dict[str, Any]) -> dict[str, Any]:
    """Runs on Modal: packs every shard (five CPU containers at a time), then spawns the training call."""
    from ecgdelin.pretrain import assign_leads

    train_leads = assign_leads(len(train_ids), LEADS_PER_ECG, seed=seed)
    stop_leads = assign_leads(len(stop_ids), LEADS_PER_ECG, seed=seed + 1)
    shards = []
    for ids, leads in ((train_ids, train_leads), (stop_ids, stop_leads)):
        for k in range(0, len(ids), SHARD):
            shards.append((ids[k : k + SHARD], leads[k : k + SHARD]))
    n_train_shards = -(-len(train_ids) // SHARD)
    infos = list(pack_shard.starmap([(pack_dir, s, ids, leads, revision) for s, (ids, leads) in enumerate(shards)]))
    call = pretrain_head_mem.spawn(run, pack_dir, n_train_shards, len(shards), seed, max_epochs, meta)
    return {"train_call": call.object_id, "n_shards": len(shards), "pack_seconds": [i["seconds"] for i in infos]}


@app.function(volumes={VOL: volume}, timeout=600)
def read_result(run: str) -> str:
    volume.reload()
    return (Path(VOL) / "results" / f"{run}.json").read_text(encoding="utf-8")


@app.function(volumes={VOL: volume}, timeout=600)
def read_progress(run: str) -> dict[str, Any]:
    """The running job's checks so far, from its checkpoint (written after every check)."""
    import torch

    volume.reload()
    path = Path(VOL) / "checkpoints" / f"{run}.pt"
    packed = sorted(p.name for p in Path(PACK).glob("shard*.json")) if Path(PACK).exists() else []
    if not path.exists():
        return {"checks": 0, "history": [], "packed_shards": len(packed)}
    state = torch.load(path, map_location="cpu", weights_only=False)["state"]
    return {"checks": state["check"], "history": state["history"], "best_check": state["best_check"], "packed_shards": len(packed),
            "best_val": state["best_val"], "stopped": state["stopped"], "seconds": state["seconds"]}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _plan() -> tuple[str, list[int], list[int]]:
    revision = json.loads(STEP1_MANIFEST.read_text(encoding="utf-8"))["revision"]
    sample = json.loads(SPLIT.read_text(encoding="utf-8"))["sample"]
    return revision, sample["train"], sample["stop"]


def _meta() -> dict[str, Any]:
    return {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_head()}


@app.local_entrypoint()
def smoke() -> None:
    revision, train, stop = _plan()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    run, pack_dir = f"ptbxl_pretrain4_smoke_{stamp}", f"{VOL}/ptbxl/packed4_smoke_{stamp}"
    import modal

    plan = pack_then_train.remote(run, pack_dir, train[:16], stop[:8], revision, 0, 1, _meta())
    print("packed", plan["n_shards"], "shards in", plan["pack_seconds"], "s")
    result = modal.FunctionCall.from_id(plan["train_call"]).get()
    print(json.dumps({k: v for k, v in result.items() if k not in ("history",)}, indent=1))
    for h in result["history"]:
        print(h)


@app.local_entrypoint()
def progress(seed: int = 0) -> None:
    run = RUN.format(seed=seed)
    p = read_progress.remote(run)
    out = REPO / "results" / f"{run}_progress.json"
    out.write_text(json.dumps({"name": run, "read": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **p}, indent=1),
                   encoding="utf-8", newline="\n")
    print(f"{run}: {p['packed_shards']} shards packed, {p['checks']} checks")
    for h in p["history"]:
        print(f"  epoch {h['epoch']:4.1f}: train {h['train_loss']:.4f}, stop-set {h['val_loss']:.4f}")


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
    call = modal.Function.from_name(app.name, "pack_then_train").spawn(run, PACK, train, stop, revision, seed, 20, _meta())
    CALLS.write_text(json.dumps({"run": run, "pack_then_train": call.object_id}, indent=1), encoding="utf-8", newline="\n")
    print("spawned", run, call.object_id)


def status() -> None:
    import modal

    calls = json.loads(CALLS.read_text(encoding="utf-8"))
    try:
        r = modal.FunctionCall.from_id(calls["pack_then_train"]).get(timeout=0)
    except TimeoutError:
        print("packing")
        return
    try:
        t = modal.FunctionCall.from_id(r["train_call"]).get(timeout=0)
        print("done: best epoch", t["best_epoch"], "loss", round(t["best_val_loss"], 4))
    except TimeoutError:
        print("training", r["train_call"])


if __name__ == "__main__":
    {"launch": launch, "status": status}[sys.argv[1]]()
