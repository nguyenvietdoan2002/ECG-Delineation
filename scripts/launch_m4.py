"""Launch the five M4 fold runs on the deployed app and exit; the calls outlive this client (decision 69).

    uv run modal deploy scripts/train_fold.py
    uv run python scripts/launch_m4.py results/m4/calls.json
    uv run python scripts/launch_m4.py results/m4/calls.json --poll     # wait, printing each call's end
    uv run modal run scripts/train_fold.py::fetch --run m4_fold<k>      # then copy each run's files here

``modal run --detach`` keeps only the last spawned function alive once the client disconnects (Modal's
own warning), so a client that waits on five parallel folds is a single point of failure. Spawning on
the deployed function instead makes each call independent of any client.
"""

import json
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_fold as tf  # noqa: E402

APP, FUNCTION = "ecg-delineation", "train_fold"


def launch(path: Path, folds: range = range(5)) -> dict[str, str]:
    train = modal.Function.from_name(APP, FUNCTION)
    calls = {}
    for fold in folds:
        run_name, args = tf._plan(fold, "0,1,2", "probe,head,unet", 200, "", "")
        call = train.spawn(**args)
        calls[run_name] = call.object_id
        print("spawned", run_name, call.object_id, "git", args["meta"]["git"], "started", args["meta"]["started"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calls, indent=1), encoding="utf-8", newline="\n")
    return calls


def poll(calls: dict[str, str], every: int = 120) -> None:
    pending = dict(calls)
    while pending:
        for run, call_id in list(pending.items()):
            try:
                result, _ = modal.FunctionCall.from_id(call_id).get(timeout=0)
                meta = result["meta"]
                print(f"{run} done: train {meta['train_seconds'] / 60:.0f} min, test {meta['test_seconds']:.0f} s, GPU {meta['max_gpu_memory_gib']:.1f} GiB", flush=True)
                del pending[run]
            except TimeoutError:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"{run} FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                del pending[run]
        if pending:
            time.sleep(every)
    print("all calls ended", flush=True)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    if "--poll" in sys.argv:
        poll(json.loads(target.read_text(encoding="utf-8")))
    else:
        launch(target)
