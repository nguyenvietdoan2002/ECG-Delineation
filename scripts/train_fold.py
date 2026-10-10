"""M3: train the linear probe, the main head and the U-Net on one fold; score the fold's validation people.

    modal run scripts/train_fold.py                          # fold 0, seed 0, all three models
    modal run scripts/train_fold.py --models head,unet       # a subset
    modal run scripts/train_fold.py --max-epochs 2           # a smoke run (named <run>_smoke)

One T4 container loads the fold's stored TimesFM states (training + validation people, about 29 GiB)
into RAM once, trains every requested model jointly over the same batches (decision 58), checkpoints
to the Volume after each epoch so a preempted container resumes, saves the final weights under
/vol/models/<run>/ (never downloaded: the TimesFM licence is non-commercial), predicts the validation
people, scores them with the M1 scorer next to NeuroKit2's M1 predictions on the same people
(decision 59), and writes /vol/results/<run>.json plus results/m3/<run>.json here. The fold's test
people are never read (decision 54).
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
NEUROKIT_PREDICTIONS = REPO / "results" / "predictions" / "neurokit_prominence.json"
MODELS = ("probe", "head", "unet")


@app.function(gpu="T4", volumes={VOL: volume}, timeout=8 * 3600, memory=(40960, 65536))
def train_fold(
    run: str,
    fold: int,
    seed: int,
    model_names: list[str],
    train_ids: list[str],
    val_ids: list[str],
    revision: str,
    max_epochs: int,
    neurokit_predictions: list[dict[str, Any]],
    meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    from ecgdelin.dataset import load_record
    from ecgdelin.delineate import LeadPoints
    from ecgdelin.models import ConvHead, LinearProbe, UNet1D, count_parameters
    from ecgdelin.paths import ludb_root
    from ecgdelin.preprocess import N_SAMPLES, prepare
    from ecgdelin.score import compare, score
    from ecgdelin.states import N_BLOCKS
    from ecgdelin.train import TensorSource, TrainConfig, make_trainee, predict_classes, targets_for, train_jointly
    from ecgdelin.trained import points_from_classes

    device = torch.device("cuda")
    config = TrainConfig(max_epochs=max_epochs)
    ids = list(train_ids) + list(val_ids)
    train_idx = list(range(len(train_ids)))
    val_idx = list(range(len(train_ids), len(ids)))
    t_start = time.perf_counter()

    # Signals, labels and states for the fold's training and validation people.
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
    print(f"loaded {len(ids)} recordings ({len(train_ids)} train, {len(val_ids)} val), states {states.nbytes / 2**30:.1f} GiB, in {load_seconds:.0f} s")

    sources = {"states": TensorSource(torch.from_numpy(states)), "signal": TensorSource(signals)}
    trainees = []
    for name in model_names:
        torch.manual_seed(seed)  # every model's initialisation depends on the seed only, not on which others run
        if name == "probe":
            trainees.append(make_trainee(name, LinearProbe(n_layers=20, d_in=2560), "states", config))
        elif name == "head":
            trainees.append(make_trainee(name, ConvHead(n_layers=20, d_in=2560), "states", config))
        elif name == "unet":
            trainees.append(make_trainee(name, UNet1D(), "signal", config))
        else:
            raise ValueError(f"unknown model {name!r}; choose from {MODELS}")

    checkpoint = Path(VOL) / "checkpoints" / f"{run}.pt"
    t_train = time.perf_counter()
    train_jointly(trainees, sources, targets, train_idx, val_idx, config, seed, device, checkpoint=checkpoint, after_epoch=lambda _epoch: volume.commit())
    train_seconds = time.perf_counter() - t_train

    models_dir = Path(VOL) / "models" / run
    models_dir.mkdir(parents=True, exist_ok=True)
    for t in trainees:
        torch.save(t.model.state_dict(), models_dir / f"{t.name}.pt")

    # Validation people: predictions, scores, and the comparison with NeuroKit2 on the same people.
    val_recordings = [recordings[k] for k in val_idx]
    predictions: dict[str, dict[str, dict[str, LeadPoints]]] = {}
    for t in trainees:
        classes = predict_classes(t.model, sources[t.source], val_idx, config.batch_size, device)
        predictions[t.name] = {rec.id: points_from_classes(classes[i], signals[k].numpy(), rec.fs) for i, (k, rec) in enumerate(zip(val_idx, val_recordings))}
    neurokit = {p["id"]: {lead: LeadPoints.from_dict(d) for lead, d in p["leads"].items()} for p in neurokit_predictions}
    predictions["neurokit_prominence"] = {rec.id: neurokit[rec.id] for rec in val_recordings}

    reports = {name: score(val_recordings, preds) for name, preds in predictions.items()}
    pairs = [("head", "unet"), ("head", "neurokit_prominence"), ("probe", "neurokit_prominence"), ("unet", "neurokit_prominence"), ("head", "probe")]
    verdicts = {f"{a}_vs_{b}": compare(val_recordings, predictions[a], predictions[b]).to_dict() for a, b in pairs if a in predictions and b in predictions}

    result: dict[str, Any] = {
        "name": run,
        "scope": "fold 0 validation people only; development numbers, not results of the study (decisions 54, 59)",
        "meta": {
            **meta,
            "fold": fold,
            "seed": seed,
            "states_revision": revision,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "load_seconds": load_seconds,
            "train_seconds": train_seconds,
            "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        },
        "config": config.to_dict(),
        "models": {
            t.name: {
                "n_params": count_parameters(t.model),
                **t.summary(),
                "layer_weights": t.model.blend.weights() if hasattr(t.model, "blend") else None,
                "report": reports[t.name].to_dict(),
            }
            for t in trainees
        },
        "neurokit_prominence": reports["neurokit_prominence"].to_dict(),
        "verdicts_validation_only": verdicts,
    }
    serial = {name: [{"id": rec_id, "leads": {lead: lp.to_dict() for lead, lp in leads.items()}} for rec_id, leads in preds.items()] for name, preds in predictions.items() if name != "neurokit_prominence"}

    results_dir = Path(VOL) / "results"
    preds_dir = Path(VOL) / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, preds in serial.items():
        (preds_dir / f"{run}_{name}.json").write_text(json.dumps(preds), encoding="utf-8")
    volume.commit()
    return result, serial


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _print_summary(result: dict[str, Any]) -> None:
    print(f"\n{result['name']}: {result['scope']}")
    meta = result["meta"]
    print(f"{meta['n_train']} training people, {meta['n_val']} validation people; load {meta['load_seconds']:.0f} s, train {meta['train_seconds'] / 60:.1f} min, GPU memory {meta['max_gpu_memory_gib']:.1f} GiB")
    print(f"\n{'model':20s} {'params':>9s} {'epochs':>6s} {'best':>5s} {'val loss':>8s} {'primary SD':>10s} {'95% CI':>16s} {'F1 P':>6s} {'F1 QRS':>7s} {'F1 T':>6s}  within-SD P_on/P_off/QRS_on/QRS_off/T_off")
    rows = [(name, m["n_params"], m["epochs_run"], m["best_epoch"], m["best_val_loss"], m["report"]) for name, m in result["models"].items()]
    rows.append(("neurokit_prominence", None, None, None, None, result["neurokit_prominence"]))
    for name, n_params, epochs, best, val_loss, r in rows:
        pts = r["points"]
        f1 = [100 * pts[p]["at"]["150"]["f1"] for p in ("P_on", "QRS_on", "T_off")]
        within = "/".join(f"{pts[p]['sd_within_ms']:.1f}" for p in ("P_on", "P_off", "QRS_on", "QRS_off", "T_off"))
        ci = f"[{r['primary_ci95'][0]:.1f}, {r['primary_ci95'][1]:.1f}]"
        print(f"{name:20s} {n_params if n_params else '':>9} {epochs if epochs is not None else '':>6} {best if best is not None else '':>5} {f'{val_loss:.4f}' if val_loss is not None else '':>8s} {r['primary_sd_ms']:10.2f} {ci:>16s} {f1[0]:6.1f} {f1[1]:7.1f} {f1[2]:6.1f}  {within}")
    for name, m in result["models"].items():
        if m["layer_weights"]:
            top = sorted(range(len(m["layer_weights"])), key=lambda k: -m["layer_weights"][k])[:5]
            print(f"{name}: heaviest layers " + ", ".join(f"L{k}={m['layer_weights'][k]:.3f}" for k in top))
    print("\nValidation-only verdicts (lower primary SD wins; F1 safeguard at 150 ms):")
    for pair, v in result["verdicts_validation_only"].items():
        print(f"  {pair:32s} diff CI {v['diff_ci95'][0]:7.2f} .. {v['diff_ci95'][1]:7.2f} ms  a_better={v['a_better']!s:5s} f1_ok={v['f1_safeguard_ok']!s:5s} a_wins={v['a_wins']}")


@app.local_entrypoint()
def main(fold: int = 0, seed: int = 0, models: str = "probe,head,unet", max_epochs: int = 200, run: str = "", revision: str = "") -> None:
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    run_name = run or f"m3_fold{fold}_seed{seed}" + ("" if max_epochs >= 200 else "_smoke")
    revision = revision or json.loads(MANIFEST.read_text(encoding="utf-8"))["revision"]
    folds = json.loads(SPLITS.read_text(encoding="utf-8"))["folds"][fold]
    val_ids = list(folds["val"])
    neurokit = [p for p in json.loads(NEUROKIT_PREDICTIONS.read_text(encoding="utf-8")) if p["id"] in set(val_ids)]
    meta = {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": _git_head(), "models": model_names}

    result, predictions = train_fold.remote(run_name, fold, seed, model_names, list(folds["train"]), val_ids, revision, max_epochs, neurokit, meta)

    out_dir = REPO / "results" / "m3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    (REPO / "results" / "predictions").mkdir(parents=True, exist_ok=True)
    for name, preds in predictions.items():
        (REPO / "results" / "predictions" / f"{run_name}_{name}.json").write_text(json.dumps(preds), encoding="utf-8", newline="\n")
    _print_summary(result)
