"""M3 deep dive, step 1: the per-sample class maps of the three fold-0 models on the validation people.

    modal run scripts/m3_classes.py

The stored predictions hold only the decoded points; the orphan breakdown and the beat overlays of
the M3 report (decision 60) need the raw argmax class maps. One CPU container loads the saved weights
from /vol/models/m3_fold0_seed0/, runs each recording through the probe, the head (both on its stored
states) and the U-Net (on the prepared signal), and returns int8 maps of shape (people, 12, 4992),
saved to results/m3/m3_fold0_seed0_classes.npz. Weights never leave the Volume.
"""

import json
from pathlib import Path
from typing import Any

from ecgdelin.modal_app import VOL, app, volume

REPO = Path(__file__).resolve().parents[1]
RUN = "m3_fold0_seed0"


@app.function(volumes={VOL: volume}, timeout=1800, memory=6144, cpu=4)
def class_maps(run: str, ids: list[str], revision: str) -> dict[str, Any]:
    import numpy as np
    import torch

    from ecgdelin.dataset import load_record
    from ecgdelin.models import ConvHead, LinearProbe, UNet1D
    from ecgdelin.paths import ludb_root
    from ecgdelin.preprocess import prepare
    from ecgdelin.trained import classes_of

    torch.set_num_threads(4)
    models_dir = Path(VOL) / "models" / run
    models = {"probe": LinearProbe(n_layers=20, d_in=2560), "head": ConvHead(n_layers=20, d_in=2560), "unet": UNet1D()}
    for name, model in models.items():
        model.load_state_dict(torch.load(models_dir / f"{name}.pt", map_location="cpu"))
        model.eval()
    states_dir = Path(VOL) / "states" / revision
    root = ludb_root()
    out = {name: [] for name in models}
    for rec_id in ids:
        rec = load_record(root, rec_id)
        signal = torch.from_numpy(prepare(rec.signal, rec.fs))[None]
        states = torch.from_numpy(np.load(states_dir / f"{rec_id}.npy"))[None]
        out["probe"].append(classes_of(models["probe"], states))
        out["head"].append(classes_of(models["head"], states))
        out["unet"].append(classes_of(models["unet"], signal))
        print(f"record {rec_id}: done")
    return {"ids": list(ids), **{name: np.stack(maps).astype(np.int8) for name, maps in out.items()}}


@app.local_entrypoint()
def main(run: str = RUN) -> None:
    import numpy as np

    result = json.loads((REPO / "results" / "m3" / f"{run}.json").read_text(encoding="utf-8"))
    ids = list(result["meta"]["val_ids"])
    revision = result["meta"]["states_revision"]
    maps = class_maps.remote(run, ids, revision)
    path = REPO / "results" / "m3" / f"{run}_classes.npz"
    np.savez_compressed(path, ids=np.array(maps["ids"]), probe=maps["probe"], head=maps["head"], unet=maps["unet"])
    print(f"saved {path} ({path.stat().st_size / 2**20:.1f} MiB): " + ", ".join(f"{k} {maps[k].shape}" for k in ("probe", "head", "unet")))
